"""Phase F, F4 on any OS: scripts/f4_retrain.ps1 in Python, for Linux runs (Modal, WATcloud). Same stages, outputs and
skip-if-done markers, so a run started by either driver resumes in the other. Run it with the Isaac venv's python
from anywhere; it works from the repo root.

    python scripts/f4_retrain.py --no-vision --final-n 100 --noise-n 50 --episodes 200 --bc-steps 20000
    python scripts/f4_retrain.py --only dagger
    modal run --detach isaac/modal_isaac.py::pipeline --background --timeout-min 720 --cmd "scripts/f4_retrain.py ..."

The pipeline: diffusion BC -> diffusion DAgger (takeover labels) -> vision student distilled from the best privileged
policy -> detector -> final evals -> demo videos, on the physical (friction) grasp, isaac_friction. Logs to
outputs/imitation/isaac_friction/f4/f4.log with STEP / DONE / FAIL / SKIP markers.

Differences from the PowerShell driver:
- Stages run one after another. The .ps1 overlaps privileged demos with vision training in a second process to save
  wall time on the laptop; on a cloud GPU that is not worth the second process.
- --no-demos skips every demo stage (the .ps1 honours -NoDemos only with -NoVision).
- --media-dir sets where the demo videos go (default docs/reports/media).
- No laptop heat budget: scripts/thermal_guard.py is not needed, and imitation/thermal.py is a no-op without its flag.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
B = "isaac_friction"
TAG = "isaac_v1_friction"
CAMS = "main=128,left_wrist_cam=64,right_wrist_cam=64"
# F3b (2026-10-08): the perturbation suite replaces the legacy recovery set; finals on the clean, shifted and
# re-grasp sets at --final-n, the noise sets at --noise-n
SETS = ["id_easy", "id_hard", "knock_arm"]         # forced drop skipped (user, 2026-10-08)
SETS_NOISE = ["joint_noise", "overshoot"]
KINDS = ["knock_arm"]                               # perturbed demos and DAgger rollouts
SEL = ["id_hard", "knock_arm"]                      # DAgger selection sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["privileged", "vision", "final", "all"], default="all")
    ap.add_argument("--only", default="", help="run this one stage")
    ap.add_argument("--envs-per-proc", type=int, default=8)
    ap.add_argument("--camera-envs-per-proc", type=int, default=6)
    ap.add_argument("--wait-f3", action="store_true", help="start once the F3 expert eval has written its JSON")
    ap.add_argument("--expert", default="", help="IsaacArmExpert attributes F3b kept (WORLDFOLD_EXPERT_PARAMS)")
    ap.add_argument("--no-vision", action="store_true", help="state-only: no vision student, detector or vision finals")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--final-n", type=int, default=200)
    ap.add_argument("--sel-n", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--dagger-episodes", type=int, default=128)
    ap.add_argument("--skip-dagger", action="store_true")
    ap.add_argument("--no-demos", action="store_true")
    ap.add_argument("--bc-steps", type=int, default=30000)
    ap.add_argument("--noise-n", type=int, default=100)
    ap.add_argument("--media-dir", default="docs/reports/media")
    a = ap.parse_args()

    os.chdir(REPO)
    py = sys.executable
    out = Path("outputs/imitation", B, "f4")
    runs = Path("outputs/imitation/runs")
    ds = Path("outputs/imitation/datasets")
    media = Path(a.media_dir)
    out.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    os.environ["WORLDFOLD_EXPERT_PARAMS"] = a.expert

    def note(msg):
        print(msg, flush=True)
        with open(out / "f4.log", "a") as f:
            f.write(msg + "\n")

    def stage(name, done, body, envs=None):
        """Runs body(run) unless --only names another stage or `done` exists; body returns 0 on success."""
        if a.only and a.only != name:
            return
        if done and Path(done).exists():
            note(f"SKIP {name} (have {done})")
            return
        note(f"STEP {name} {datetime.datetime.now():%Y-%m-%dT%H:%M:%S}")
        t0 = time.time()
        os.environ["WORLDFOLD_ISAAC_ENVS_PER_PROC"] = str(envs or a.envs_per_proc)
        with open(out / f"{name}.log", "a") as log:
            def run(*args):
                return subprocess.run([py, "-u", "-m", *map(str, args)], stdout=log, stderr=subprocess.STDOUT).returncode
            code = body(run)
        if code != 0:
            note(f"FAIL {name} exit {code}")
            sys.exit(1)
        note(f"DONE {name} {int(time.time() - t0)}s")

    def best(run_dir):
        """Best checkpoint of a DAgger run: the last kept round in history.json."""
        return [r for r in json.loads(Path(run_dir, "history.json").read_text()) if r["kept"]][-1]["checkpoint"]

    def mark(path, code):
        if code == 0:
            Path(path).write_text(f"{datetime.datetime.now():%Y-%m-%dT%H:%M:%S}")
        return code

    def demos(policy, ckpt, replan):
        """Demo videos: 3 episodes of each eval set, one Isaac env."""
        def body(run):
            for s in SETS:
                mp4 = media / f"half_fold_friction_{policy}_{s}.mp4"
                if mp4.exists():
                    continue
                code = run("imitation.demo", "--backend", B, "--ckpt", ckpt, "--replan-every", replan, "--set", s,
                           "--n", 3, "--out", mp4)
                if code:
                    return code
            return 0
        if not a.no_demos:
            stage(f"demos_{policy}", media / f"half_fold_friction_{policy}_knock_arm.mp4", body, envs=1)

    priv = runs / f"{TAG}_dagger"
    s0 = runs / f"{TAG}_diff_s0" / "final.pt"
    if a.wait_f3:
        f3 = Path("outputs/imitation", B, "f3", "eval_expert_recovery.json")
        note(f"WAIT for {f3}")
        while not f3.exists():
            time.sleep(30)
        time.sleep(30)      # its Isaac processes exit after the JSON is written

    if a.phase in ("privileged", "all") or a.only:
        collect = ["imitation.data.collect", "--backend", B, "--episodes", a.episodes, "--workers", 2,
                   "--recovery-fraction", 0.5, "--perturb-kinds", *KINDS, "--version", TAG, "--root", ds, "--resume"]
        if not a.no_vision:
            collect += ["--render", "--cameras", CAMS]
        stage("collect", ds / TAG / "manifest.json", lambda run: run(*collect),
              envs=None if a.no_vision else a.camera_envs_per_proc)      # rig cameras: 2 x 8 state-only uses 14 GB
        stage("train_diff_s0", s0, lambda run: run(
            "imitation.train", "--policy", "diffusion", "--dataset", TAG, "--root", ds, "--run", s0.parent,
            "--steps", a.bc_steps, "--seed", 0))
        if not a.skip_dagger:
            # DAgger from seed 0 (its round 0 is the seed-0 BC eval on the selection sets), takeover p = 0.6
            stage("dagger", priv / "done.txt", lambda run: mark(priv / "done.txt", run(
                "imitation.dagger", "--backend", B, "--init", s0, "--dataset", TAG, "--root", ds, "--out", priv,
                "--rounds", a.rounds, "--episodes", a.dagger_episodes, "--train-steps", 15000, "--eval-n", a.sel_n,
                "--eval-sets", *SEL, "--score-sets", *SEL, "--min-gain-se", 1, "--workers", 2, "--labels", "takeover",
                "--takeover-p", 0.6, "--resume", "--recovery-fraction", 0.5, "--perturb-kinds", *KINDS)))
        if not a.only:
            note(f"PRIVILEGED COMPLETE best={s0 if a.skip_dagger else best(priv)}")

    # operating points from M5b.6: privileged replan 4, vision replan 2
    bp = s0 if a.skip_dagger else (best(priv) if (priv / "history.json").exists() else "")
    bv = runs / f"{TAG}_vision_t0" / "final.pt"    # the teacher-relabelled vision BC is the student (M5b.3)

    if a.phase in ("vision", "all") or a.only:
        if not a.no_vision:
            demos("privileged", bp, 4)
            stage("train_vision_t0", bv, lambda run: run(
                "imitation.train", "--policy", "vision", "--dataset", TAG, "--root", ds, "--run", bv.parent,
                "--steps", 30000, "--batch", 256, "--seed", 0, "--teacher", bp))

        def final_privileged(run):
            code = run("imitation.evaluate", "--backend", B, "--ckpt", bp, "--sets", *SETS, "--n", a.final_n,
                       "--replan-every", 4, "--workers", 2, "--out", out / "final_privileged_r4.json", "--resume")
            if code:
                return code
            return mark(out / "final_privileged_r4.done", run(
                "imitation.evaluate", "--backend", B, "--ckpt", bp, "--sets", *SETS_NOISE, "--n", a.noise_n,
                "--replan-every", 4, "--workers", 2, "--out", out / "final_privileged_r4_n100.json", "--resume"))
        stage("final_privileged", out / "final_privileged_r4.done", final_privileged)
        if a.no_vision:
            demos("privileged", bp, 4)      # after the finals: never two Isaac jobs at once
            if not a.only:
                note("F4 COMPLETE (no vision)")
            return
        stage("detector_train", runs / f"{TAG}_detector" / "detector.pt", lambda run: run(
            "imitation.vision.success", "train", "--versions", TAG, f"{TAG}_failures", "--root", ds,
            "--out", runs / f"{TAG}_detector", "--steps", 2000))
        stage("detector_agree", runs / f"{TAG}_detector" / "agreement.json", lambda run: run(
            "imitation.vision.success", "agree", "--ckpt", runs / f"{TAG}_detector" / "detector.pt",
            "--versions", TAG, f"{TAG}_failures", "--root", ds))
        if not a.only:
            note("VISION COMPLETE")

    if a.phase in ("final", "all") or a.only:
        # each set is saved as it finishes and --resume skips it, so a job-limit cut costs at most one set
        def final_vision(run):
            code = run("imitation.evaluate", "--backend", B, "--ckpt", bv, "--sets", *SETS, "--n", 200,
                       "--replan-every", 2, "--workers", 2, "--out", out / "final_vision_r2.json", "--resume")
            if code:
                return code
            return mark(out / "final_vision_r2.done", run(
                "imitation.evaluate", "--backend", B, "--ckpt", bv, "--sets", *SETS_NOISE, "--n", 100,
                "--replan-every", 2, "--workers", 2, "--out", out / "final_vision_r2_n100.json", "--resume"))
        stage("final_vision", out / "final_vision_r2.done", final_vision, envs=a.camera_envs_per_proc)
        demos("sensor", bv, 2)
        if not a.only:
            note("F4 COMPLETE")


if __name__ == "__main__":
    main()
