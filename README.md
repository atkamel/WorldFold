# WorldFold

Two SO-101 arms folding a cloth in simulation. The active work trains a closed-loop RL policy in Isaac Sim (LeHome's
stack, PhysX particle cloth, a friction gripper with no weld) to do a quarter fold whose edges and body line up.
Earlier phases built the half fold in MuJoCo with an imitation pipeline (BC, DAgger, diffusion, a vision student) and
a world model over the cloth's surface-angle field.

WatAI, 2026

## Where things stand

[docs/status.md](docs/status.md) says where the project is and what comes next. The plan is in
[docs/roadmap.md](docs/roadmap.md) and every measured number is in [docs/results.md](docs/results.md).

## Layout

```
isaac/                 Isaac Sim env and the active work
  sim/                 the env, its profiles (lehome, friction, quarter2), the IsaacLab scene, IK
  experts/             scripted pinch expert and the closed-loop servo experts
  rl/                  residual-RL window env, push corrector, RLPD learner and their CLIs
  tools/               quarter-fold batch runner, grasp bench, gate re-score, install smoke test
  tests/               sim-free tests
  watcloud.sbatch      WATcloud job wrapper (Isaac Sim in Docker)
mujoco_sim/            MuJoCo ClothFoldEnv and the shared cloth, arm and camera constants (cloth_params.py)
cloth_fold_rl/         fold-task wrappers, scripted experts and the fold alignment score (both simulators)
imitation/             imitation pipeline on MuJoCo and Isaac (spec in docs/imitation.md)
cloth_angles/          RSSM world model over cloth angle fields (parked)
docs/                  status, roadmap, results, design docs, dated reports
docs/reports/media/    showcase videos
```

## Setup

MuJoCo, the imitation pipeline and every sim-free test run in a local venv on the pinned versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r imitation/requirements.txt
pytest -m "not slow"        # fast suite; -m slow adds the MuJoCo runs
```

Isaac Sim runs in Docker on WATcloud through `isaac/watcloud.sbatch`, or natively on Windows (see
[isaac/README.md](isaac/README.md)).

## Running the Isaac work

```bash
# scripted two-arm quarter fold, 8 copies in one scene, top-view video per seed
sbatch --gres=gpu:rtx_3090:1,tmpdisk:200000 isaac/watcloud.sbatch isaac/tools/quarter_batch.py \
    --copies 8 --seeds 0:8 --profile quarter2 --fold2 two --drag none --video --out outputs/isaac/quarter/q2

# residual RL on the half fold, then a demo hunt with video
python isaac/rl/precision_rl.py train --run outputs/isaac/rl/p9 --copies 8 --hours 8
python isaac/rl/precision_rl.py demo --run outputs/isaac/rl/p9 --ckpt best.pt --seeds 990000:990064 --video
```

Run outputs (logs, episode records, videos) stay on WATcloud or the local disk and are gitignored. Code and outputs
removed in the October 2026 cleanup are at tag `pre-cleanup-2026-10-22`.
