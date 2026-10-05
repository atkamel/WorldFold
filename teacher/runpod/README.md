# Isaac (LeHome) on RunPod

Same stack and behaviour as the Modal functions in `teacher/modal_teacher.py` (pinned commits, Isaac Sim 5.1,
LeHome's IsaacLab fork, fixed physics, flat pads, oracle tools, teleop), on a cheaper RTX 4090.
Every session ends by **stopping the pod itself** (`runpodctl stop pod`), so nothing keeps billing.

## 1. Create the pod (RunPod console)
- GPU: **RTX 4090** (24 GB is plenty for one Isaac). In the filters pick **CUDA 13** (Isaac Sim 5.1 needs driver 580+).
- Template: **RunPod PyTorch** (Ubuntu 22.04). Container disk **50 GB**.
- Storage: either a **network volume** of ~80 GB mounted at `/workspace` (install once, ~$5-6/month), or none
  (re-run setup each session, ~20-30 min of pod time).
- Expose **TCP port 7777** (needed for teleop only).

## 2. Put this repo on the pod
Either push the branch and clone on the pod:
```bash
git clone -b ROY-vla-teacher https://github.com/atkamel/WorldFold /workspace/WorldFold
```
or send the local folder (works with uncommitted changes): on the laptop `tar -czf wf.tgz WorldFold` then
`runpodctl send wf.tgz`; on the pod `cd /workspace && runpodctl receive <code> && tar -xzf wf.tgz`.

## 3. Install (once per volume, or every session without a volume)
```bash
bash /workspace/WorldFold/teacher/runpod/setup.sh        # Isaac stack; add WITH_TEACHER=1 for the pi0.5 teacher
```
It checks the driver first and prints `SETUP DONE` at the end. Safe to rerun.

## 4. Run a session (the pod stops itself afterwards; KEEP_POD=1 to keep it)
```bash
bash /workspace/WorldFold/teacher/runpod/run_isaac.sh grip     # adhesion A/B, prints a GRIP/RELEASE table
bash /workspace/WorldFold/teacher/runpod/run_isaac.sh teleop   # then on the laptop:
#   python mujoco_live/teleop_client.py <pod public IP> <public port mapped to 7777>
```
Results: `/workspace/results/isaac/<stamp>-<mode>/` (run.log, json rows, summary.txt, teleop_demo.jsonl).
The teleop client also keeps its own copy of the demo log on the laptop.

## Differences from Modal
- Modal tunnel -> RunPod's direct TCP port. Modal Queue live sessions (`isaac_live`) are not ported; teleop replaces them.
- Modal volume -> `/workspace` (network volume) or copy results off before the pod is deleted.
- Modal timeouts -> `MAX_S` hard cap (default 1800 s), teleop idle stop (300 s), and the pod stopping itself.
- 4090 (24 GB) vs L40S (48 GB): one Isaac + teacher fits; the 4-worker batch setup does not. Re-measure speeds.
