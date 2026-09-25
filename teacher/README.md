# VLA teacher (π0.5 `lehome_sim`) on Modal

Sprint: **VLA_TEACHER**. Goal: distill a cloth-folding VLA teacher into a small student policy.
Teacher = [`IliaLarchenko/lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim) (π0.5, JAX/Orbax, ~10 GB,
1st of 62 in the LeHome 2026 sim round). Code: [`IliaLarchenko/lehome_solution`](https://github.com/IliaLarchenko/lehome_solution).

**Current plan (team lead, 2026-09-24): Project 1, where teacher and student both live in his Isaac Sim world.**
Everything here also works for WorldFold/MuJoCo experiments, but see *Findings*: the teacher does not transfer to WorldFold zero-shot.

## Quick start (free Modal: Starter plan = $30/month credit)

```bash
pip install modal && modal setup                     # once; also set a spending limit in the Modal dashboard
# Windows: prefix every modal command with PYTHONUTF8=1 (the CLI crashes on unicode otherwise)

modal run teacher/modal_teacher.py::stage_checkpoint # CPU only, builds image (~3 min, one-time) + downloads 10 GB to a Volume
modal run teacher/modal_teacher.py::smoke            # L40S: starts his policy server, benchmarks latency (~6 min, ~$0.30)
```

Everything persists in the Modal Volume `lehome-teacher` (checkpoint, JAX compile cache, results, logs).
Use `modal run --detach ...` for anything long, so a Wi-Fi drop doesn't kill the run. **Finish every session with
`modal container list` showing nothing running.**

| Command | What | GPU | Typical cost |
|---|---|---|---|
| `::stage_checkpoint` | image build + checkpoint → Volume | none | ~$0.05 |
| `::smoke` | serve + latency sweep (batch 1/4 × candidates 1/2/3) | L40S | ~$0.30 |
| `::pilot --n 6` | 6 teacher episodes in WorldFold (sim=main, 0.45 m cloth), in parallel | 6 × L40S | ~$0.90 |
| `::cache --run-name X --n-seeds 60` | resumable parallel caching (per-seed files + sha256 markers) | N × L40S | ~$0.11/episode |
| `::server_profile2` | teacher-server bottleneck profile (yappi in all threads + K servers/GPU) | L40S | ~$0.80 |
| `::profile_all --main-episode <dir>` | physics profile of a recorded episode | CPU only | ~$0.02 |

## The teacher's I/O contract (his `scripts/serve.py`)

- **Transport:** plain `websockets`, JSON text frames (NOT openpi-client/msgpack), `max_size` 100 MiB, port 8000.
- **Request** `{"type":"infer_chunk", ...}` with:
  - `observation.images.{top_rgb,left_rgb,right_rgb}`: 640×480×3 **uint8 RGB**, sent as `{"base64","shape","dtype"}`. Send raw: **don't resize** (the model resizes to 224). **Top camera un-rotated**: the raw top view has the **arms at the TOP** of the image, and the model rotates it 180° itself.
  - `observation.state`: 12 floats, **absolute joint angles in radians**, `[L pan, lift, elbow, wrist_flex, wrist_roll, gripper, R same]`.
  - `garment_type_id`: 0 top_long, 1 top_short, 2 pant_long, 3 pant_short. There's **no text prompt**. His eval does one warm-up call and reuses `garment_type_pred`.
  - `inference_config`: send the per-garment dict from `assets/inference_config.json`, otherwise CFG etc. fall back to weak defaults.
  - `initial_actions`: pass back the previous response's `next_initial_actions` (inpainting).
- **Response:** `actions` = `execute_in_n_steps` × 12 **absolute joint targets (rad) at 30 Hz**, plus `next_initial_actions` and diagnostics.
- **Start pose in his data** (identical in all 250 episodes checked): `L = [-1.24, -1.69, 1.49, 1.05, -0.08, -0.01]`, `R` = same with pan `+1.24`. Gripper ≈ 0 = closed, opens to ~0.5 rad.
- Pinned: openpi submodule **`c23745b5`**, `jax[cuda12]==0.5.3`, cuDNN 9.5.1 (pinned in the image: fixes the known jax 0.5.3 cuDNN init bug). **Never run his `setup.sh`** blindly: it moves openpi off the pinned commit and installs Isaac/Vulkan packages.

## Findings so far (2026-09-24)

**Phase 0: the teacher serves fine.** L40S: load 30–56 s, VRAM 5.9 GB, **0.29–0.34 s per call** (batch 1, 1–3 candidates),
best-of-N nearly free. Raw numbers: `results/phase0-*.json`.

**Zero-shot in WorldFold: 0/12 successes.** After fixing the camera orientation, start pose, gripper mapping and
wrist cameras, it does the right *kind* of motion (grabs the far edge and pulls it back), but crumpled. On the 0.22 m cloth
(arms-on-flanks sim) it doesn't touch the cloth at all. Remaining gaps: cloth size/shape/look, 11×11 flex cloth physics,
weld "glue" grasp, rendering. **It's a specialist and doesn't transfer**, hence Project 1 (use its own world).

**Teacher server bottleneck (this matters for Project 1, same server):**
- Throughput is capped at **~3 calls/s regardless of concurrency**. The GPU is busy only 19–33% (measured on H200, which was no faster than L40S).
- yappi inside the server (`::server_profile2`): the single `inference-batcher` thread spends its time on **(a) recompiling / compile-cache lookups for new batch shapes** (the 5 ms window forms random batch sizes 1–6), **(b) hundreds of slow-path JAX dispatches + ~740 tiny eager GPU ops per call**, and **(c) running the model twice per request** (retry pass).
- Unpacking requests costs only ~30 ms/call (websocket deflate ~12 ms + base64/JSON ~17 ms). Easy 10%: client `compression=None`, raw bytes.
- **Several server processes per GPU does NOT fix it** (measured on one L40S): 1 / 2 / 4 servers = **3.15 / 3.75 / 4.38 calls/s**,
  GPU busy 47% / 69% / 87%. The GPU fills up with inefficient work (tiny ops, a second model run per request, best-of-3, CFG).
- Fix = cut the work per call, easiest first: (1) check whether the retry pass is needed (up to 2×), (2) pad batches to a fixed size
  (no recompiles), (3) jit the whole pre/post-processing + sampling pipeline as one function (biggest win).
- **Speed recipe, measured** (`fast_server_patch.py`, env-gated, `::server_fastbench`, L40S, real frames from his dataset, clients replay `next_initial_actions` like real episodes):

  | L40S, top_short config (best-of-3) | baseline | **fast** (same quality) | **fast_noretry** |
  |---|---|---|---|
  | 1 client: time per call | 0.52 s | 0.35 s | **0.19 s** |
  | 1 client: calls/s | 1.92 | 2.81 | **5.26** |
  | 4 clients: calls/s | 2.37 | 3.60 | **6.91** |

  `fast` = `TEACHER_FAST_OUT=1 TEACHER_FLAT_STATE=1 TEACHER_BUCKETS=1` (one device->host copy instead of hundreds of slices,
  pre-flattened model state in `module_jit`, fixed batch buckets). `fast_noretry` adds `TEACHER_RETRY=0`.
  **The retry pass fired on 100% of calls** even with real frames, so it doubles the model work; switching it off is a
  quality trade-off to validate (success rate in his eval) before using it for caching.
- **Round 2** (`::server_fastbench2`, L40S with 8 CPU / 32 GB, all phases with the `fast_noretry` env, so retry is OFF):

  | Phase | calls/s | vs original (2.37) | GPU busy |
  |---|---|---|---|
  | full-res images, 4 clients | 7.55 | 3.2× | 60% |
  | full-res images, 16 clients | 12.57 | 5.3× | 60% |
  | **client-side 224 resize**, 4 clients | 11.08 | 4.7× | 76% |
  | **client-side 224 resize, 16 clients** | **15.82** | **6.7×** | 80% |
  | + candidates 3→1 (quality knob) | 18.25 | 7.7× | 72% |
  | + 2 server processes | 15.08 | no gain | 88% |

  **Client-side resize = identical model input.** Resize with his own `openpi_client.image_tools.resize_with_pad(img, 224, 224)`
  before sending; the server's `ResizeImages` then returns the image unchanged (it short-circuits on 224×224, `image_tools.py:28`).
  Requests shrink 3.7 MB → 0.6 MB.
- **Stopped here: the GPU is the limit (~80–88% busy, ~16 calls/s on an L40S).** Two servers don't help, and candidates 3→1 gives only +15%
  (the VLM prefix dominates) so it isn't worth the quality loss. Going further needs model-level changes (fewer denoise steps, no CFG, smaller images).
- **Recipe to use:** env `TEACHER_FAST_OUT=1 TEACHER_FLAT_STATE=1 TEACHER_BUCKETS=1` + client-side 224 resize + batch many sim clients
  (≥16) per server. `TEACHER_RETRY=0` roughly doubles throughput again but is a **quality trade-off: validate the success rate in his
  eval before caching with it.** The same-quality config (retry on) measured 1.5× at 4 clients; with resize + 16 clients it should stack, but that isn't measured yet.

**WorldFold MuJoCo physics (only relevant if you keep using MuJoCo):** 80% of each physics step is the constraint solver
(121-vertex flex cloth ≈ 300 edge constraints + ~50 contacts). A 1 ms timestep instead of 0.5 ms is **1.8× faster**,
stable, and gives an identical result on the flank sim. 2 ms is ~4× faster but changes the cloth behaviour (5–8 cm).

## Project 1 (Isaac Sim) must-knows

- Isaac Sim needs an **RTX GPU (L40S, RTX PRO 6000; his rollouts used RTX PRO 6000)**. **A100/H100/H200/B200 are NOT supported** (no RT cores).
- Isaac needs **Vulkan graphics drivers even headless**. Modal containers may only ship compute drivers, so test that first (~30 min, ~$1). If Modal can't, RunPod RTX pods usually can.
- His repo already has the loop: `scripts/run_eval.py` / `src/lehome_solution/eval/*` workers talk to `serve.py` (the same JSON protocol as above). Caching = saving what the eval workers already produce. Benchmark gate = his eval's per-garment success rate.
- His 1,000-episode dataset `lehome/dataset_challenge_merged` (LeRobot v3; `four_types_merged/` etc.) is in the same world, so it's usable for the student.
- The real-robot version of the teacher: [`IliaLarchenko/lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real).

## Data safety (lessons from past RunPod/Modal runs)

One file per episode, written `tmp → os.replace`. A sha256 **completion marker written last**. `vol.commit()` after
every episode. Resume = skip seeds with a verified marker. `modal run --detach`. Never tar or aggregate outputs inside a
timed function. Always verify `modal container list` is empty at the end.

## Files

| File | What |
|---|---|
| `modal_teacher.py` | Modal app: image (his repo pinned + openpi + MuJoCo), checkpoint staging, server smoke/benchmark, WorldFold rollouts/pilot/cache, physics + server profilers |
| `wf_adapter.py` | WorldFold ⇄ teacher adapter: 3 rendered cameras, 12-dim state, absolute→joint-delta actions, continuous gripper, grasp-anywhere, start pose |
| `yappi_serve.py` | runs his `serve.py` under yappi (all threads); SIGUSR1 dumps per-thread stats |
| `render_cams.py` | local camera-view check (WorldFold top + wrist views) |
| `results/` | Phase 0 benchmark JSONs, physics profile JSONs |

Optional: `ROY_SIM_DIR=<path to stats-worldfold/mujuco>` enables `sim="roy"` (arms-on-flanks sim).
MuJoCo functions need the SO101 assets vendored. `modal_teacher.py` copies them from a local `so101_nexus` install
automatically (`pip install -r cloth_fold_rl/requirements.txt`).
