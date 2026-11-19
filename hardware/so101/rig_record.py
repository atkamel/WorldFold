"""Record a fold played from the command line as the control panel records its own: the phone's pictures and depth
beside the step log, in a new folder under runs/, and at the end whether the cloth folded.

    python rig_record.py replay trajectories/half_fold_30cm.json --left arm2 --right arm1 --torque-limit 50
    python rig_record.py policy --left arm2 --right arm1 --layout behind

The first word picks replay.py or policy_run.py; everything after it goes to that script unchanged, so the run is
checked, played and stopped exactly as without the recording. The folder is what rig_dataset.py reads: frames/ and
frames.jsonl (recorder.py), steps.jsonl with the script's own notes beside it, and label.json. A run that never
moved (a refused trajectory, a dry run) leaves no folder.
"""

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import recorder

RUNS = Path(__file__).parent / "runs"
FRAME_SECONDS = 0.2         # a picture this often, as a panel fold saves them
LABEL = "label.json"        # rig_dataset.LABEL


def record(play, argv, phone, runs=RUNS, ask=input):
    """Run play([*argv, "--log", <run>/steps.jsonl]) with `phone`'s frames saved beside the log. Returns (the
    script's exit code, the run folder or None if nothing was played). A stop (Ctrl-C inside the script, a refusal)
    still closes the recording."""
    run = recorder.Recorder(runs, meta={"task": "half_fold", "command": list(argv)})
    done = threading.Event()

    def pictures():
        while not done.wait(FRAME_SECONDS):
            frame = phone.latest()
            if frame is not None:
                run.frame(time.time(), frame.rgb, frame.depth, frame.K)
    camera = threading.Thread(target=pictures, daemon=True)
    camera.start()
    steps = run.path / "steps.jsonl"
    try:
        code = play([*argv, "--log", str(steps)])
    finally:
        done.set()
        camera.join(timeout=2)
        run.close()
        moved = steps.exists() and steps.stat().st_size > 0
        if not moved:
            shutil.rmtree(run.path)
    if not moved:
        return code, None
    answer = ask("Did the cloth fold? y or n, or Enter to leave this run unlabelled: ").strip().lower()
    if answer in ("y", "n"):
        (run.path / LABEL).write_text(json.dumps({"success": answer == "y", "note": ""}))
    print(f"Recorded in {run.path}" + ("" if answer in ("y", "n") else
                                       f". Label it later: rig_dataset.py label {run.path} good|bad"))
    return code, run.path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("replay", "policy"):
        sys.exit("rig_record.py replay ... or rig_record.py policy ..., then that script's own arguments")
    if argv[0] == "replay":
        import replay as script
    else:
        import policy_run as script
    if "--dry-run" in argv or "--log" in argv:
        sys.exit("rig_record.py is for a run on the arms and names the log itself: leave out --dry-run and --log")
    import rgbd
    phone = rgbd.Phone()
    phone.start()
    phone.wait()
    return record(script.main, argv[1:], phone)[0]


if __name__ == "__main__":
    sys.exit(main())
