# Notes for Claude sessions working in `teacher/`

Read `README.md` first: it has the teacher's I/O contract, today's measured findings, and the Project 1 (Isaac) must-knows.

Working rules the team learned the hard way:
- **Cloud = Modal.** Before any billed run, state the GPU, the expected minutes and the $. After every run, verify `modal container list` shows nothing and report the actual cost.
- **Profile before scaling.** Measure calls/s, GPU utilisation and episode wall time first. Don't buy a bigger GPU (or add server copies) to fix it: the teacher server is capped at ~3 calls/s by per-call overhead (recompiles, hundreds of tiny eager ops, a second model run per request). An H200 was no faster than an L40S, and 4 servers on one GPU gave only +39%. Cut the work per call instead (see README).
- **Isaac Sim: RTX GPUs only** (L40S / RTX PRO 6000). A100/H100/H200/B200 cannot run it. JAX 0.5.3 in his stack also predates Blackwell (no B200).
- Long jobs: `modal run --detach`. A detached app can stay alive after its function finishes, so check `modal app list` and `modal app stop <app-id> --yes` it. Data: per-episode files + sha256 completion marker + `vol.commit()`; resume from disk truth.
- Windows: prefix modal commands with `PYTHONUTF8=1`.
- Don't run his `setup.sh`. The image in `modal_teacher.py` pins openpi `c23745b5` and cuDNN 9.5.1 on purpose.
- Top camera must be sent **raw with the arms at the top** (the model flips it). Send the per-garment `inference_config`; the start pose is in the README.
- Keep estimates honest and grounded in measured numbers from `results/`. Flag guesses as guesses.
