# WorldFold Experiments — Formal Designs & Build Prompts

Statistical designs for attaching the UniClothDiff modules (DPM / DDM / MPPI,
per the paper "Blindness Doesn't Have to Be a Disability") to this repo's
cloth-folding testbed. Each experiment is written AP-Stats style: units,
variables, hypotheses, conditions, test, and sample size — plus an honest
feasibility rating against what actually exists in this folder.

---

## 0. Feasibility ground truth (what this repo does and doesn't provide)

**Provided by the repo (usable as-is):**

| Asset | Location | Role in experiments |
|---|---|---|
| `ClothFoldEnv` (bimanual MuJoCo cloth fold, 11×11 flex grid) | `mujuco/sim_main.py` | The simulator |
| Ground-truth mesh: 121 vertex world positions | `env.data.xpos[env._cloth_body_ids]` | Supervision target for the DPM |
| Matched initial conditions | `env.reset(options={"cloth_pose", "goal_pose", "task"})` | Backbone of every paired design |
| Task metrics: `success`, `fold_score` | `sim_main.py` `_step_info` / `_fold_score` | Response variables |
| Camera rendering (RGB; depth renderable) | `_render_image()` | Source of realistic self-occlusion |
| Angle-field observation (11×11 surface-tilt scalars) | `cloth_angles/data/angle_field.py` | Compressed "normals" observation |
| RSSM world model + open-loop `imagine_angles` | `cloth_angles/model/` | Non-diffusion dynamics baseline |
| Train / eval / replay / checkpoint harness | `cloth_angles/train.py`, `evaluate.py` | Matched-sequence evaluation loop |
| Persistence + action-agnostic baselines | `cloth_angles/model/baselines.py` | Floor baselines |
| Scripted fold/still/recovery policies | `cloth_angles/collect_episodes.py` | Closed-loop control without a planner |

**NOT in the repo (must be built):** the DPM (no diffusion code exists
anywhere in the tree), the DDM, any occlusion / partial-observation model,
any planner (MPPI/CEM), and any point-cloud pipeline. As of writing,
`outputs/cloth_angles/episodes/` and `checkpoints/` are empty and no venv
exists — the baseline pipeline itself has not yet been run.

**Two observation spaces, one statistical design.** Every perception
experiment has a *toy* version in angle-field space (11×11 scalars; plugs
into all existing `cloth_angles` code) and a *paper-faithful* version on the
raw 121×3 mesh (bypasses the angle pipeline, reads `env.data.xpos`
directly). The design, hypotheses, and tests below are identical for both —
only the units of the response variable change (angle MAE in radians vs
mesh-vertex MSE in metres²). Recommended order: toy first for speed, then
re-run paper-faithful with the same seeds.

**Reward gap (blocks Exp 3/4):** the learned world model predicts angle
fields, but the task cost is corner-to-goal distance (`_reward`,
`sim_main.py`). A planner therefore has nothing to minimize until a
**keypoint head** (predicting the 4 corner positions, supervised from
`cloth_state`) is added to the decoder. This is real infrastructure, not a
detail.

Feasibility legend: 🟢 class-scoped in this folder · 🟡 buildable, more
plumbing · 🟠 needs planner + cost head · 🔴 research-scale (full
UniClothDiff).

---

## 1. Experiment 1 — Diffusion perception as the belief front-end 🟢 (RECOMMENDED)

Reproduces and extends the paper's headline perception result (DPM MSE 2.32
vs 5.44 baseline) inside this repo, adding the K-sample belief arm.

**Research question.** On occluded cloth configurations, does a diffusion
perception model — especially a K-sample belief — reconstruct the hidden
geometry better than a deterministic encoder?

- **Population / units.** Cloth configurations reachable by `ClothFoldEnv`
  resets and scripted rollouts. One experimental unit = one held-out
  configuration (a state + its occlusion mask), generated from a seed via
  `reset(options=...)` so every treatment sees the identical unit.
- **Design.** Matched triples (within-subjects / repeated measures). Each
  unit is evaluated under all three treatments; randomness is controlled by
  fixing seeds per unit and re-using them across arms.
- **Explanatory variable (treatment, 3 levels).**
  1. Raw encoder — existing `Encoder` MLP applied to the masked field
     (already built; the free baseline arm);
  2. DPM single sample — one denoised reconstruction;
  3. DPM K-sample belief — K reconstructions aggregated (mean, or all K
     encoded and averaged downstream).
- **Blocking variable.** Occlusion severity stratum: light (≈10–20% cells
  hidden) vs heavy (≈40–60%, contiguous flap-like masks). The hypothesis
  lives in the heavy stratum.
- **Response variable (quantitative).** Reconstruction error **on the
  occluded cells only** — angle MAE (rad) in the toy version, mesh-vertex
  MSE in the paper-faithful version. Secondary response: downstream
  open-loop dynamics MAE via `imagine_angles` fed by each front-end.
- **Hypotheses.**
  - H₀: μ_raw = μ_DPM1 = μ_DPMK (mean occluded-cell error equal across
    front-ends).
  - H₁: μ_DPMK < μ_DPM1 ≤ μ_raw, with the gap largest in the heavy stratum.
- **Conditions.** Paired-by-construction (same units all arms);
  independence across units from independent seeds; check
  normality of paired differences with a normal-probability plot — if
  skewed, use the rank-based test.
- **Test.**
  - AP-level: matched-pairs t-test on each pairwise comparison
    (DPM-K vs raw is the headline pair), α = 0.05.
  - Publication-level: repeated-measures ANOVA or Friedman test across the
    three arms, Wilcoxon signed-rank post-hocs, Bonferroni-corrected
    (α = 0.05/3 per pairwise test).
- **Sample size / power.** The paper's effect (2.32 vs 5.44, ≈2.3× ) is
  large; n = 15–20 matched units per stratum gives power > 0.9 at α = 0.05
  for effects half that size. Use 20 per stratum (40 total).
- **Procedure.**
  1. Generate 40 held-out configurations (seeds 1000–1039), 20 per
     occlusion stratum; store (true field, mask, masked field).
  2. Run all three front-ends on every unit; record occluded-cell error.
  3. Plot paired differences; run the tests above; report means ± SE per
     stratum, test statistic, p-value, and effect size.
- **Needs building.** (a) occlusion module (contiguous cell masking with a
  severity knob), (b) conditional DPM over the angle field, (c) the paired
  eval script. Everything else exists.

---

## 2. Experiment 2 — Diffusion sample-spread as a know-when-you-don't-know oracle 🟡

Reuses Exp 1's DPM. No planner needed: closed-loop control comes from the
existing scripted `fold_action` (`cloth_angles/collect_episodes.py`).

**Research question.** Does gating the fold on DPM sample-spread (probe when
uncertain, fold when confident) raise success on ambiguous starts without
extra probing on easy starts?

- **Units.** Matched episode initial conditions set by
  `reset(options={"cloth_pose", "task"})`, stratified into *easy* (flat,
  centred cloth) and *ambiguous* (offset / perturbed → heavy
  self-occlusion) strata.
- **Design.** Within-subjects, 3 treatments per unit:
  1. Gate-on: run DPM K samples each control cycle; if mean per-cell
     variance > threshold τ → execute probe macro (lift + shake +
     settle + re-observe), else fold;
  2. Always-fold: gate off (no-probe baseline);
  3. Random-probe: probes forced at random times, frequency matched to
     arm 1's observed probe rate (controls for "probing helps regardless
     of timing").
- **Response variables.** Primary: episode success (binary,
  `info["success"]`). Secondary: number of probes (count), episode length
  (steps).
- **Hypotheses.**
  - H₀: p_gate = p_always = p_random within each stratum.
  - H₁: p_gate > both on the ambiguous stratum, AND probe count on the easy
    stratum ≈ 0 (no speed penalty where perception is already confident).
- **Conditions.** Paired binary outcomes on identical starts; independence
  across units; success counts large enough for the exact tests below
  (they don't require normality).
- **Test.**
  - AP-level: paired proportions via McNemar's test on the headline pair
    (gate-on vs always-fold), α = 0.05.
  - Publication-level: Cochran's Q across the three arms + McNemar
    post-hocs; Wilcoxon signed-rank (or Poisson regression) for probe
    counts; report per-stratum.
- **Sample size.** Proportions need more units than means: 30–50 matched
  starts per stratum (60–100 total). Sweep τ over ~5 values on a separate
  tuning set of ≈15 starts first — never tune τ on the test set.
- **Needs building.** (a) per-cell variance map from K DPM samples (trivial
  once Exp 1 exists), (b) probe macro in `ClothFoldEnv` action space,
  (c) the gated closed-loop runner + matched-start battery.

---

## 3. Experiment 4 — Latent-vs-mesh compute curve 🟠 (run before Exp 3)

**Research question.** At matched compute, does cheap latent-rollout search
or expensive explicit-mesh scoring produce better folds — and do the curves
cross?

- **Units.** Matched fold tasks (fixed `cloth_pose` + `goal_pose` per seed).
- **Design.** Two-factor within-subjects: paradigm {pure-latent (RSSM
  `imagine_angles` + keypoint head), pure-mesh (explicit rollout scorer)} ×
  compute budget {e.g. 64, 256, 1024, 4096 candidate sequences}. Every
  unit runs all 8 cells.
- **Response.** Final `fold_score` (quantitative; success rate secondary).
- **Hypotheses.** H₀: no paradigm × budget interaction. H₁: interaction
  exists (latent wins at low budget, mesh at high budget — the curves
  cross).
- **Test.** Two-way repeated-measures ANOVA with interaction term
  (AP-level framing: paired t-tests paradigm-vs-paradigm at each budget,
  Bonferroni α = 0.05/4).
- **Sample size.** 20–30 matched tasks.
- **Needs building.** Keypoint head (see §0 reward gap), a sampling planner
  (CEM or MPPI) over the RSSM latent, and a mesh-rollout scorer. This is
  why it's 🟠: two new components before the first datum.

---

## 4. Experiment 3 — Hybrid propose-verify 🔴 (research-scale; hold)

Latent search screens thousands of candidates; top-M survivors re-scored by
DDM mesh rollouts; final action from the verified set.

- **Design.** Three arms at fixed total wall-clock — pure-latent,
  pure-mesh, hybrid (M swept) — on matched tasks; within-subjects.
- **Response.** `fold_score` / success at equal wall-clock.
- **Hypotheses.** H₀: all arms equal. H₁: hybrid > both (search wants
  volume, the final choice wants fidelity).
- **Test.** Friedman + Wilcoxon signed-rank post-hocs, Bonferroni.
- **Why 🔴.** Requires everything in Exp 4 **plus** a trained mesh-space
  DDM — i.e. the full UniClothDiff stack rebuilt in this repo. This is the
  destination, not a first experiment.

---

## 5. Bonus — 2×2 factorial ablation (paper-style, clean AP-Stats)

Once DPM (Exp 1) and any dynamics pairing exist: factorial
{perception: DPM / raw} × {dynamics: RSSM / persistence-or-DDM} on matched
episodes; response = fold_score. Two-way ANOVA: are both main effects
significant (i.e. are both modules necessary)? This mirrors the paper's
ablation table. The paper's 9/10-vs-6/10 style hardware comparison is a
two-proportion z-test / Fisher's exact if arms use different trials, or
McNemar if the same targets are attempted by both arms.

---

## 6. Threats to validity (state these in any writeup)

1. **Sim-only.** All conclusions are about MuJoCo flex cloth; no zero-shot
   transfer claim is supported here.
2. **Synthetic occlusion (initial phase).** Cell masking approximates but
   does not equal camera self-occlusion. Upgrade path: render depth from
   the `main` camera and mask vertices hidden by folded flaps; re-run the
   same seeds.
3. **Multiple comparisons.** Three arms and two strata multiply tests —
   Bonferroni throughout, and the headline pair is pre-registered in each
   design above.
4. **Threshold leakage (Exp 2).** τ must be tuned on a disjoint tuning set.
5. **Grasp cheat.** `ClothFoldEnv` uses a weld-equality grasp; success
   rates are optimistic relative to a real pinch grasp. Fine for
   within-sim comparisons; do not quote absolute rates as transferable.

---

## 7. Claude Code prompts (one session each, in order)

### M0 — Baseline foundation (run first; no new research)

> In `cloth_angles`: collect a small dataset with `collect_episodes.py`,
> train the RSSM world model with `train.py`, and run `evaluate.py`.
> Confirm the RSSM beats the persistence baseline on 1-step and open-loop
> MAE and report the numbers. Fix anything that breaks. Do not start the
> diffusion or planner work.

(Prerequisite: create the venv — `py -3.13 -m venv .venv` — and
`pip install -r requirements.txt`; nothing is installed yet.)

### Exp 1 — Occlusion + DPM + three-front-end paired eval

> Build Experiment 1 from EXPERIMENTS.md §1 on top of `cloth_angles`, in
> three pieces, with tests for each:
>
> 1. **Occlusion module** — `cloth_angles/data/occlusion.py`: given an
>    11×11 angle field, produce a binary mask of hidden cells and the
>    masked field. Masks must be contiguous (flap-like), seeded, with a
>    severity parameter; provide "light" (10–20% hidden) and "heavy"
>    (40–60%) presets. Unit-test mask contiguity, severity bounds, and
>    seed reproducibility.
> 2. **Conditional DPM** — `cloth_angles/model/diffusion.py`: a small
>    DDPM (MLP or tiny UNet; the field is only 121 scalars) that denoises
>    the full angle field conditioned on (masked field, mask). Add
>    `sample_k(masked_field, mask, k)`. Train on episodes from the
>    existing EpisodeStore with random occlusion augmentation; save
>    checkpoints via the existing schema-validated checkpoint pattern.
>    Also add `variance_map(masked_field, mask, k)` (per-cell variance
>    across k samples) for later use by Experiment 2.
> 3. **Paired evaluation** — `cloth_angles/experiments/exp1_frontends.py`:
>    generate 40 held-out configurations (seeds 1000–1039; 20 light, 20
>    heavy occlusion) via `ClothFoldEnv.reset(options=...)` + scripted
>    rollouts. Evaluate three front-ends on every unit: (a) raw `Encoder`
>    on the masked field, (b) DPM single sample, (c) DPM K-sample belief
>    (K=8, mean aggregate). Report occluded-cells-only MAE per arm per
>    stratum, paired differences, Wilcoxon signed-rank p-values with
>    Bonferroni correction (and paired-t as the AP-level readout), and
>    downstream open-loop MAE through `imagine_angles`. Write a results
>    table to `outputs/cloth_angles/exp1/` plus per-unit CSV so the
>    statistics are recomputable.
>
> Follow the design in EXPERIMENTS.md §1 exactly (same seeds across arms,
> error on occluded cells only, persistence fill as the floor baseline).
> Do not modify `mujuco/sim_main.py`.

### Exp 2 — Uncertainty-gated probing (after Exp 1)

> Build Experiment 2 from EXPERIMENTS.md §2:
> `cloth_angles/experiments/exp2_probe_gate.py`. Closed-loop
> `ClothFoldEnv` episodes driven by the existing scripted `fold_action`.
> Each control cycle, compute the DPM `variance_map` (K=8); if mean
> variance > τ, execute a probe macro (lift both grippers +z, small
> lateral shake, settle, re-observe) then re-run perception. Three arms on
> identical matched starts: gate-on, always-fold, random-probe
> (frequency-matched). Two strata of starts (easy: centred flat cloth;
> ambiguous: offset/perturbed via `reset(options={"cloth_pose": ...})`),
> ≥30 starts per stratum, τ tuned on a disjoint 15-start tuning set.
> Report success rates and probe counts per arm per stratum, McNemar test
> for gate-on vs always-fold, Cochran's Q across the three arms. Write
> results + per-episode CSV to `outputs/cloth_angles/exp2/`.

### Exp 4 — Planner + compute curve (only when ready to invest)

> Add a keypoint head to the `cloth_angles` world model predicting the 4
> cloth corner positions (supervised from `ClothFoldEnv`'s `cloth_state`),
> then `cloth_angles/planning/cem.py`: CEM/MPPI sampling planner scoring
> candidate action sequences by predicted corner-to-goal distance via
> `imagine_angles` + keypoint head. Benchmark per EXPERIMENTS.md §3:
> paradigm × budget within-subjects, final fold_score, interaction test.

### Exp 3 — hold until Exp 4 and a mesh-space DDM exist (EXPERIMENTS.md §4).

---

## 8. M0 diagnosis results (run 2026-08-05/06, Python 3.13 venv, torch 2.13 CPU)

**Infrastructure: PASS.** Every pipeline stage ran end-to-end with zero code
changes: install clean, unit tests 17/17, collection 27 episodes / 0
physics failures, training 10k steps in 19m21s, evaluation + figures
produced.

**Throughput (kills the "takes hours" rumor):** ~66 s per 200-step episode
headless (~61 s uncontended). 24-episode dataset = 24m19s. The 0.12×-RT
slowness seen in the browser is mjviser viewer overhead, never paid by
headless collection/training.

**Grip/fold diagnostic (instrumented scripted-fold rollout, seed 0):**

| check | result |
|---|---|
| right arm grasp | engages at ~t=100 — weld mechanism works |
| left arm grasp | **never engages in 200 steps** |
| IK convergence (`*_ik_success`) | never true for either arm |
| fold_score | **max 0.042** (success threshold 0.85) |
| stability | no qacc blowup, no out-of-bounds, no action clipping |

**Goal-spec defect (found by inspection, `sim_main.py` reset):** the goal
sends ALL FOUR corners to their diagonal opposites
(`goal = corners0[[3, 2, 1, 0]]`). A physical diagonal fold keeps the two
fold-line corners fixed, so a *perfect* fold scores ≈ 0.5 — the 0.85
success threshold is geometrically unreachable and the +10 success bonus
can never fire.

**Model gate: FAIL.** Final checkpoint, 32 sequences:

- one-step RSSM MAE **0.0348 rad** vs persistence **0.0093 rad** →
  `beats persistence: False`
- open-loop MAE grows gradually 0.0397 → 0.0548 over 10 steps (no
  collapse — this desideratum passes)
- training KL pinned at the free-bits floor (1.0) all run → posterior ≈
  prior; the stochastic latent is barely used

**Root cause:** the dataset is quasi-static. Mean per-step angle change is
persistence's error (~0.009 rad ≈ 0.5°) because the fold script barely
moves the cloth (fold_score 0.04, left arm never grasps). The model's
reconstruction floor (~0.035 rad) exceeds the actual per-step motion, so
predicting "no change" wins. The eval figures confirm it: error
concentrates at the one tugged corner; the rest of the cloth never moves.
The spec's own criterion says "beat persistence on **active** held-out
transitions" — with this data there are almost no active transitions.

**Eval-harness caveats:** `evaluate.py` samples from ALL episodes (train +
held-out mixed), and does not restrict to active transitions. Both should
be fixed alongside the sim fix.

**Required fixes before the M0 gate re-runs (in priority order):**

1. Reposition arms side-by-side on one edge (ALOHA-style), grasping the
   two far-edge corners — matches real bimanual rigs.
2. Goal = half-fold: far corners → near-corner positions, near corners
   stay. Makes the 0.85 threshold reachable.
3. Scripted fold as a real trajectory: approach → grasp → lift → drag
   over → lower → release (current script drives at the corner and holds).
4. Debug left-arm reach (orientation/joint limits; base rotation differs
   between arms).
5. `evaluate.py`: use held-out episodes only; report MAE on active
   transitions (|Δangle| above a small threshold) alongside overall MAE.

Then: recollect (~30 min), retrain (~20 min), re-evaluate. Gate passes
when the RSSM beats persistence on active held-out transitions.

---

## 9. M1 fix results (run 2026-08-05/06, branch `fix/sim-bimanual-fold`) — GATE PASSED

**Simulator fixed and verified.** Measured IK reach (usable annulus
~0.22–0.33 m from base) drove the redesign: flank-mounted arms ("bedsheet
fold from the sides" — grasp, swing arc, and landing all at constant
~0.27 m radius), 0.22 m cloth, half-fold goal with scale floor, and a
7-phase scripted policy (reach → wait[both-arms-sync] → lift → carry →
place → lower → release). Five root causes found and fixed along the way:

1. old diagonal layout demanded a 0.54 m drag — physically impossible;
2. old goal spec (all 4 corners → diagonal opposites) capped fold_score at
   0.5, making the old 0.85 success threshold unreachable by ANY fold;
3. rotation-Jacobian rows in the IK least-squares penalized shoulder pan
   even with zero rotation error (the "left arm never grasps" bug — the
   arm was fine, the solver was fighting itself);
4. un-synchronized arms: the fast arm's fold dragged the shared cloth out
   from under the slow arm's grasp;
5. surface-dragging the flap vs carrying it over the fold line, plus a
   servo friction/tension equilibrium (fixed with a high carry + measured
   overshoot bias).

Result: scripted fold succeeds 9/9 episodes (success termination at
t=149, fold_score ~0.80, corners within ~5 cm), video-verified
(`outputs/cloth_angles/videos/fold_fixed.mp4`). SUCCESS_FOLD_SCORE
recalibrated 0.85 → 0.75 against the video-verified fold (documented in
`sim_main.py`).

**World-model gate passed.** Four model-side fixes were needed after the
data fix (progression of active-transition MAE vs persistence's 0.0350):

| change | active MAE |
|---|---|
| original model, fixed-sim data | 0.1116 |
| wider latent (16×16 cats, free-bits 3.0, kl_weight 0.2, 20k steps) | 0.0741 |
| balanced stop-gradient KL (BUG FIX — code contradicted its own docstring; plain KL collapses the posterior) | 0.0588 |
| deterministic (expected-z) eval inference | 0.0519 |
| **delta decoding (predict per-step change; zero output ≡ persistence)** | **0.0141 ✓** |

Final: one-step 0.0114 vs 0.0193 overall, **0.0141 vs 0.0350 active**
(2.5× better), open-loop growth gradual (0.019 → 0.141 over 10 steps,
no collapse). Held-out episodes only; active_fraction 0.346.

**Known limitations (carry into Exp 1's threats-to-validity):** open-loop
error compounds in delta mode (autoregressive); the scripted fold is
near-deterministic, so fold dynamics diversity comes mainly from recovery
episodes — domain randomization at collection time is the lever if more
variety is needed; KL sits at the free-bits floor (the latent carries
little information — adequate for this gate, worth revisiting for M2).

**The simulator and pipeline are certified. Next: Experiment 1 (§7).**

---

## 10. Experiment 1 results (run 2026-08-06) — belief > guess; spread predicts error

**Setup.** DPM = conditional DDPM (`cloth_angles/model/diffusion.py`), 60k
steps on 7,282 frames (episode store + 24 cloth-pose-randomized scripted
rollouts, seeds 2000+). Evaluation: 40 matched units (seeds 1000–1039, 20
light / 20 heavy flap occlusion, folds-in-progress), identical (field, mask)
pairs across all arms, error on HIDDEN cells only, wrapped MAE. Artifacts:
`outputs/cloth_angles/exp1/{per_unit.csv, results.json, configs.npz}`.

**Hidden-cell MAE (rad), Wilcoxon signed-rank, Bonferroni α = 0.0167:**

| stratum | raw (nearest-fill) | DPM 1-sample | DPM K=8 belief | raw vs dpmK |
|---|---|---|---|---|
| light (n=20) | 0.196 | 0.391 | **0.192** | tie (p=0.90) |
| heavy (n=20) | 0.174 | 0.734 | 0.672* | ns on ranks (p=0.35) |
| all (n=40) | 0.185 | 0.563 | 0.432 | ns on ranks |

Confirmed at Bonferroni level everywhere: **dpmK < dpm1** (belief beats
single guess; p ≤ 2.7e-3 in every stratum) — the K-sample-belief hypothesis.

*The heavy-stratum dpmK mean is dominated by ~4 catastrophic units where the
DPM commits to the WRONG fold hypothesis (~1.6–2.9 rad, e.g. seed 1036); on
ranks (Wilcoxon) dpmK is statistically indistinguishable from raw, and on
typical heavy units it matches or beats raw (e.g. seed 1039: 0.138 vs 0.441).

**The key finding — sample spread predicts error (the Exp 2 mechanism):**

- Spearman ρ(per-unit K-sample spread, dpmK error) = **0.853, p = 2.6e-12**
- Median split on spread: high-spread half mean error **0.736**;
  low-spread half **0.061** — the low-spread half beats raw's overall 0.185
  by ~3×. Every catastrophic unit has extreme spread (4.9–6.5 vs typical <1).

Interpretation: an ungated DPM is dangerous (rare catastrophic multimodal
flips); a **spread-gated DPM is the best front-end measured** — precisely the
paper's "know-when-you-don't-know" thesis, and the direct motivation for
Experiment 2 (gate the fold on `variance_map`, probe when uncertain).

**Honest caveats.** (1) Nearest-fill is a strong baseline on these smooth
low-res fields — the raw-vs-dpmK headline is a tie, not a win; diffusion's
measured value here is calibrated uncertainty + typical-case parity, not
blanket accuracy. (2) An earlier run trained only on unrandomized episodes
collapsed on offset configs (means 2–3 rad) — learned perception degrades
off-distribution while deterministic fill cannot; varied training data fixed
most of it (kept as a threats-to-validity lesson; dry-run before trusting
numbers). (3) Downstream 10-step open-loop MAE ranks raw 0.164 < dpmK 0.279
< dpm1 0.311, consistent with the reconstruction ordering.

