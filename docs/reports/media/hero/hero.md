# Portfolio hero renders: Isaac Sim quarter fold (2026-10-21)

Real simulated episode, no overlay text. Scripted expert on the **R1 preset** (the quarter fold's default:
`qf_preset()` gentle release + fold-1 friction + lay-down, nothing set by hand), backend `isaac_friction`,
`WORLDFOLD_QF_SECOND=parallel`, a real fold 1 (`WORLDFOLD_QF_START` unset), `--replan-every 8`, 1 env.

| file | what |
|---|---|
| `hero.png` | 1600×1000 PNG, step **100** of the episode: both arms hold the folded strip up mid-carry (fold 1) |
| `hero_loop.mp4` | H.264 (High, yuv420p, CRF 18, faststart), 1600×1000, 20 fps (sim real time), **7.0 s** = steps **50–189**: grasp, lift, carry and lay-down of fold 1, then the arms reach for fold 2 |
| `hero_episode.json` | the run's sidecar: cameras and outcome |

**Episode:** seed **660000**, `success: true`, `termination_reason: success`, 349 steps, fold score 0.937
(`hero_episode.json`). The same seed in `quarter_fold_parallel_expert_r1.mp4` ran 341 steps / 0.989: the Isaac
cloth is not bit-for-bit repeatable across runs (this session's four runs of the seed: 331, 340, 349 steps, all successes).

## Camera (both deliverables, `hero0`)

Look-at pinhole camera in the world frame (`isaac/lab_scene.py::_hero_camera`, image-up = world +Z):

| parameter | value |
|---|---|
| position | (−0.30, 0.62, 1.12) m |
| aim point | (0.00, −0.20, 0.55) m |
| vertical fov | 31° |
| resolution | 1600 × 1000 |

A 3/4 oblique view from the far side of the table, facing both SO-101 arms (bases at x = ±0.22, y = −0.24; cloth centre
(0, −0.135), table top z = 0.42). Lighting is the scene's own (dome + two distant lights + the headlight direction),
unchanged. The run also rendered a second candidate, `hero1` = (−0.30, 0.66, 1.20) → (0, −0.20, 0.56), fov 30, not used.

## Commands

```powershell
# the episode, frames of every step (outputs/imitation/hero/final_frames/hero{0,1}/660000_<step>.png), ~6 min, 1 Isaac process
powershell -ExecutionPolicy Bypass -File scripts\hero_render.ps1 -Seed 660000 -Out outputs\imitation\hero\final.mp4 `
    -FramesDir outputs\imitation\hero\final_frames -Size 1600x1000 `
    -Cams "-0.30 0.62 1.12 0.0 -0.20 0.55 31; -0.30 0.66 1.20 0.0 -0.20 0.56 30"
```
```bash
# hero.png = step 100 of camera hero0
cp outputs/imitation/hero/final_frames/hero0/660000_0100.png docs/reports/media/hero/hero.png
# hero_loop.mp4 = steps 50..189 of camera hero0 (ffmpeg from imageio_ffmpeg in .venv)
ffmpeg -framerate 20 -start_number 50 -i outputs/imitation/hero/final_frames/hero0/660000_%04d.png -frames:v 140 \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart docs/reports/media/hero/hero_loop.mp4
```

`scripts/hero_render.ps1` wraps `python -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt expert
--seeds <seed> --hero-cam PX PY PZ TX TY TZ FOVY [...] --hero-size WxH --no-overlay --frames-dir <dir>`. The flags are
opt-in; without them the demo renders as before (top-down `demo` camera, overlay on).

Camera search on the same seed (frames in `outputs/imitation/hero/`, not kept in git): `probe` (4 views, 800×500,
80 steps) → `full` (3 wider views) → `tight` (fov 31; the front view showed a grey horizon wedge at the top) → `final`
(the front view pitched down so the frame ends on the floor).
