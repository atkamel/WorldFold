# SO-101 arm fit on NVIDIA Anchor-Lab (held-out, scored once)

Fitted JSON sha256 `a767743fbb08ca75…` frozen 2026-10-15 14:48:10 before held-out was read. Friction setting `F4_imp995_ref008`; delay 0.0 ms; dt 0.004 s.

## Held-out, whole motions (deg RMSE; equal weight per motion; arm = mean of 5 arm joints)

| model | arm | moving | holding | pan | lift | elbow | w_flex | w_roll | jaw |
|---|---|---|---|---|---|---|---|---|---|
| fitted | 0.284 | 0.286 | 0.270 | 0.17 | 0.41 | 0.43 | 0.22 | 0.19 | 0.22 |
| B_nvidia_same_opts | 0.385 | 0.425 | 0.372 | 0.27 | 0.43 | 0.53 | 0.47 | 0.22 | 0.55 |
| BAM_m6_12V | 0.477 | 0.380 | 0.526 | 0.27 | 0.64 | 0.88 | 0.31 | 0.28 | 0.46 |
| B_nvidia_cam13.44 | 0.482 | 0.461 | 0.500 | 0.35 | 0.50 | 0.80 | 0.51 | 0.25 | 0.55 |
| B_nvidia_cam23.1 | 0.483 | 0.460 | 0.503 | 0.35 | 0.49 | 0.81 | 0.51 | 0.25 | 0.55 |
| B_nvidia_cam0 | 0.484 | 0.465 | 0.497 | 0.35 | 0.50 | 0.79 | 0.52 | 0.25 | 0.55 |
| B_nvidia_native | 0.497 | 0.468 | 0.520 | 0.35 | 0.50 | 0.88 | 0.51 | 0.25 | 0.60 |
| BAM_m1_12V | 0.532 | 0.447 | 0.557 | 0.29 | 0.65 | 1.03 | 0.33 | 0.36 | 0.59 |
| pure_delay | 0.857 | 0.889 | 0.894 | 0.49 | 0.69 | 2.02 | 0.61 | 0.47 | 0.93 |
| A_stock | 0.866 | 0.901 | 0.900 | 0.51 | 0.70 | 2.01 | 0.62 | 0.49 | 1.15 |
| perfect_tracking | 1.280 | 1.431 | 1.195 | 0.91 | 1.08 | 2.35 | 1.09 | 0.97 | 2.28 |
| hold_start | 7.858 | 8.218 | 7.855 | 7.58 | 9.90 | 8.02 | 7.42 | 6.37 | 12.93 |

## Per motion (arm deg): fitted vs NVIDIA fit vs plain delay

| motion | fitted | B_nvidia (same opts) | B_nvidia (cam 23.1) | pure delay | perfect tracking |
|---|---|---|---|---|---|
| heldout-backlash-detection | 0.322 | 0.332 | 0.331 | 0.819 | 1.230 |
| heldout-coupled-joints | 0.347 | 0.424 | 0.424 | 0.582 | 1.184 |
| heldout-custom-motion | 0.281 | 0.351 | 0.351 | 0.664 | 1.215 |
| heldout-diagonal-sweep | 0.173 | 0.316 | 0.381 | 0.836 | 1.207 |
| heldout-frequency-sweep | 0.307 | 0.524 | 0.526 | 0.944 | 1.881 |
| heldout-friction-gravity | 0.255 | 0.252 | 0.262 | 0.805 | 1.023 |
| heldout-hold-under-gravity | 0.292 | 0.489 | 0.842 | 1.076 | 1.175 |
| heldout-square-pattern | 0.295 | 0.389 | 0.747 | 1.127 | 1.324 |

## Sign tests (fitted better on k of 8 held-out motions; one-sided)

- vs B_nvidia_same_opts:arm: 7/8, p = 0.035
- vs B_nvidia_same_opts:arm_move: 7/8, p = 0.035
- vs B_nvidia_cam23.1:arm: 8/8, p = 0.004
- vs B_nvidia_cam23.1:arm_move: 8/8, p = 0.004
- vs pure_delay:arm: 8/8, p = 0.004
- vs pure_delay:arm_move: 8/8, p = 0.004

## Model selection evidence (VAL whole motions; held-out not used)

- F0_default: VAL arm fitted 0.479 vs B 0.546; creep (30 s hold) B 0.95 deg, fitted 1.08 deg
- F1_imp990: VAL arm fitted 0.334 vs B 0.436; creep (30 s hold) B 0.78 deg, fitted 0.76 deg
- F2_imp995: VAL arm fitted 0.287 vs B 0.399; creep (30 s hold) B 0.49 deg, fitted 0.50 deg
- F3_imp990_ref008: VAL arm fitted 0.282 vs B 0.387; creep (30 s hold) B 0.41 deg, fitted 0.55 deg
- F4_imp995_ref008: VAL arm fitted 0.251 vs B 0.363; creep (30 s hold) B 0.23 deg, fitted 0.46 deg
- FN_noslip1: VAL arm fitted 0.225 vs B 0.330; creep (30 s hold) B 0.18 deg, fitted 0.19 deg
- chosen: F4_imp995_ref008 (lowest VAL whole-motion arm RMSE among noslip=0 settings (Newton GPU cannot run noslip); tie -> least creep)
- jaw fit: FIT jaw 0.184 deg from 0.699; VAL jaw 0.125

## Teleop cross-check (6 public arms, unknown voltage; robustness only; per-arm delay fitted on TRAIN)

- B_nvidia: TEST arm-mean 0.990 deg (verified-3 1.069)
- fitted: TEST arm-mean 0.901 deg (verified-3 0.980)

## Fitted parameters (SI per rad)

| joint | kp | kv | damping | frictionloss | armature | forcerange |
|---|---|---|---|---|---|---|
| shoulder_pan | 39.18 | 1.433 | 0.683 | 0.509 | 0.0317 | 2.46 |
| shoulder_lift | 25.61 | 1.194 | 0.262 | 0.377 | 0.0339 | 2.46 |
| elbow_flex | 15.99 | 0.242 | 0.561 | 0.484 | 0.0149 | 2.46 |
| wrist_flex | 19.73 | 0.941 | 0.244 | 0.246 | 0.0246 | 2.46 |
| wrist_roll | 58.55 | 2.286 | 1.261 | 0.557 | 0.0378 | 2.46 |
| gripper | 68.79 | 3.687 | 1.254 | 0.379 | 0.0335 | 4.02 |

## Independent verification (3 agents, 2026-10-15; scripts + outputs in results/verify/)
- Every held-out number was recomputed from raw parquet with code that doesn't import ours (max difference 5e-7 deg). A separate MuJoCo build from so101_fitted.json reproduces the fitted model at 0.2840. The pre-registration hash holds and no held-out leaked into fitting or selection. Held-out raw files were read earlier, but only by data-convention checks and jaw exploration, never for fitting.
- **Fair comparison**: about half of the 0.483 -> 0.284 gain comes from the friction-softness setting F4, which helps NVIDIA's parameters equally (B_nvidia_same_opts 0.385). Re-fitting the parameters alone gives 0.385 -> 0.284 (-26%):
  - wins 7 of 8 motions; one-sided sign test p = 0.035, two-sided 0.070
  - motion-level 95% CI of the difference: [-0.168, -0.034] deg
  - the within-motion block bootstrap gives [-0.111, -0.087], but treats motions as fixed
- Chaos floor: stick/slip under the stiff F4 friction amplifies 1e-9 rad differences to as much as 0.09 deg on 3 motions. Per-motion scores there are good to about 0.02 deg; the overall means move by at most 0.002.
- Newton: only the CPU path (use_mujoco_cpu=True) is validated on whole motions. Its gap to MuJoCo matches the perturbation chaos floor, so the RMS < 0.01 deg criterion is too tight. The MuJoCo-Warp GPU path still has to be checked on CUDA (L4).
- Solver iterations must be >= 10/20. At 5/2 the fitted model scores 0.484 vs 0.504 for NVIDIA's, so the gain is gone. At 20/40 and above it stays 0.283.
- Delay sits at its 0 ms bound and isn't identified: the real step onset is 24 ms vs 12 ms in the sim, with the lag absorbed into softer dynamics. The RL copy randomizes the delay over 0-30 ms.
- Hold creep in the sim: 0.2-0.6 deg per 36 s hold, against about 0 on the real arm (45 of 50 holds show 0 ticks of drift). This is negligible for holds of 3 s or less.
- Teleop arms: gain 9% (0.901 vs 0.990). Our wrist_roll friction (2.5x NVIDIA's) transfers worse, losing on 4 of 6 arms. The RL ranges (so101_fitted_rl.json) therefore span both fits, widened by 20%.
- Jaw: the x2.218 unit premise for NVIDIA's jaw is inferred from anchor-bench code, not stated in any NVIDIA file. The fitted jaw effort of 4.02 N*m is above the servo's stall torque and must not be used as a grasp-force limit.

## Newton GPU check (L4, CUDA, MuJoCo-Warp float32; 2026-10-15, results/newton_gpu_*.{json,md})
- Temporary VM wf-l4-tmp: g2-standard-8, Toronto-b. Toronto-a was out of L4s. Ran 16:23-16:44, then deleted. Newton 1.6.1, warp 1.18.0, mujoco/mujoco-warp 3.12.0.
- Model arrays on the GPU equal the reference: max abs diff 5e-8. Iterations 10/20.
- Per held-out motion, arm RMSE vs real (warp vs MuJoCo) differs by at most 0.021 deg (limit 0.03).
- Mean arm RMSE: warp 0.2825, MuJoCo 0.2844 (limit 0.01). B_nvidia: warp 0.3811, MuJoCo 0.3847.
- Fitted beats B_nvidia on the GPU on 7 of 8 motions, the same as on CPU (friction-gravity +0.003). No NaNs. Gripper 0.223 deg.
- mujoco_warp printed "linesearch iterations limit reached" 7067 times (fitted) and 5207 times (B) over about 205k steps each. Accuracy is unaffected. Try ls_iterations 50 when RL speed is measured.
- One world runs at 5 ms/step (kernel-launch bound, no CUDA graph). This says nothing about batched RL throughput, which is measured in the RL step.
