# Running the teacher yourself

Tested on a GCP L4 VM (24 GB, Toronto) in October. Not yet tested from a clean clone by someone else, so team `plumbing`
is checking this page. Fix it when it's wrong.

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
PHYSICS=orig PER_TYPE=5 WORKERS=2 bash docker_run.sh teacher     # 20 folds in the teacher's own physics
```
Results land in `~/wf/results/teachergrasp/<time>-orig/`.

**Pass check:** our run took 25 min for 20 episodes (76 s/episode, 2 workers) and passed LeHome's check on 18/20.
Reproduce about that before changing anything.

## Reset bank and scores
- Add `TEACHER_SNAPSHOT_EVERY=5` to save mid-fold states.
- `grasp_test_patch.py` restores them (`TEACHER_RESTORE_DIR`, `TEACHER_RESTORE_SHIFT_CM`, `TEACHER_RESTORE_ROT_DEG`).
- `select_pregrasp.py` picks snapshots just before a grasp.
- `bash docker_run.sh flatten` writes flat garment shapes, about 1 min per garment.
- `python neat/teacher_end_states.py <run dir> <garments_flat dir> teacher/mujoco_live <out dir>` scores end states.
- `neat/fold_videos_3d.py` renders them with the arms.

Snapshots and flat garments aren't in git (hundreds of MB), so regenerate them, or ask on the board.

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
