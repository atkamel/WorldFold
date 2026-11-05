# Running the teacher yourself

This is setup for the RL, not a task in itself. You need the teacher on your GPU, and this run is also what produces your
reset bank.

Tested on a GCP L4 VM (24 GB, Toronto) in October. Nobody has run it from a clean clone yet, so expect a snag or two, and fix
this page when it's wrong. The image and checkpoint below were confirmed downloadable without login on 2026-11-05.

## You need
- A cloud GPU with an RTX card and NVIDIA **driver 580.x**. Isaac needs RTX: A100, H100, H200 and B200 can't run it.
- On GCP: image family `common-cu129-ubuntu-2404-nvidia-580` (project `deeplearning-platform-release`), a large enough
  disk, and a max run duration with STOP so a forgotten VM can't run up a bill. New projects often have 0 GPU quota, so
  request it first.
- The image `ghcr.io/cimurghe/worldfold-isaac:5.1` (public, about 10 GB compressed): Isaac + LeHome + assets.
- The teacher checkpoint, [`IliaLarchenko/lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim) (about 10 GB).
  `teacher_setup.sh` installs it.

## Steps
Copy this repo to `~/wf/WorldFold` on the VM, then:
```bash
cd ~/wf/WorldFold/teacher/gcp
bash docker_run.sh prep     # checks docker and GPU, pulls the image
bash docker_run.sh setup    # one time: teacher venv + checkpoint, kept on the VM disk
TEACHER_SNAPSHOT_EVERY=5 PHYSICS=orig PER_TYPE=5 WORKERS=2 bash docker_run.sh teacher     # 20 folds + your reset bank
```
Results land in `~/wf/results/teachergrasp/<time>-orig/`, with mid-fold states in `snapshots/`.

**Sanity check:** ours took about 25 min on an L4 (about $0.40), passed LeHome's check on 18/20, and saved 1,049 states
(294 MB). If yours is close, your setup matches ours.

## Reset bank and scores
- Our own 1,049 states exist on Roy's laptop. They aren't in git because of size. Ask on the board if you'd rather have those.
- `grasp_test_patch.py` restores them (`TEACHER_RESTORE_DIR`, `TEACHER_RESTORE_SHIFT_CM`, `TEACHER_RESTORE_ROT_DEG`).
- `select_pregrasp.py` picks snapshots just before a grasp.
- `bash docker_run.sh flatten` writes flat garment shapes, about 1 min per garment.
- `python neat/teacher_end_states.py <run dir> <garments_flat dir> teacher/mujoco_live <out dir>` scores end states.
- `neat/fold_videos_3d.py` renders them with the arms.

Our scores for the 20 folds are in `neat_data/`, so you don't need to rerun those to start calibrating.

## Known failures
- **First Isaac start looks hung.** It compiles shaders on the CPU for about 5 min, then Kit's hang detector kills it.
  `docker_run.sh` already turns the detector off, inside one long-lived container `wfisaac`. Keep that container.
- **Never mount empty folders over Isaac's cache paths.** Isaac then crashes at sim start.
- **Stockouts.** Stopping a VM releases its GPU, and restarts can fail with `ZONE_RESOURCE_POOL_EXHAUSTED`. Try another
  zone, or clone from a disk snapshot.
- **Don't run Ilia's `setup.sh`.** It moves openpi off the pinned commit (`c23745b5`).
- **Windows line endings break `.sh` files silently.** Check with `bash -n` and `file`.
- **Killing jobs over ssh.** A `pkill` pattern can match your own shell, so put kill commands in a script on the VM.
- **More GPU isn't faster.** The teacher server is limited per call. An H200 wasn't faster than an L40S. Profile first.

## Model I/O
See `teacher/README.md`: camera format, 12 joint angles, 30 Hz chunks, and why the top camera must stay unrotated.
