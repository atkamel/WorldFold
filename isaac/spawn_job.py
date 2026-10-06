"""Starts one imitation pipeline command on the deployed worldfold-isaac app and returns at once: the job runs on Modal
whether or not this machine stays on, and its log and outputs land on the worldfold-isaac volume.

    modal deploy isaac/modal_isaac.py                  # once, and after changing code the job runs
    python isaac/spawn_job.py --tag demo --gpu A10G --cpu 8 --memory-gb 32 --timeout-min 40 -- imitation.demo --ckpt ...

cpu, memory and timeout bound what the job can cost, with the GPU's rate.
"""

import argparse
import json

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", default="A10G")
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--memory-gb", type=int, default=32)
    ap.add_argument("--timeout-min", type=int, required=True)
    ap.add_argument("--env-json", default="{}", help='extra environment, e.g. {"ISAAC_ENV_PARAMS": "..."}')
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <imitation module> <args>")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    job = modal.Function.from_name("worldfold-isaac", "imitation_job").with_options(
        gpu=args.gpu, cpu=args.cpu, memory=args.memory_gb * 1024, timeout=args.timeout_min * 60)
    call = job.spawn(cmd, args.tag, json.loads(args.env_json))
    print(f"spawned {call.object_id}; log: modal volume get worldfold-isaac imitation/logs/{args.tag}.log .")


if __name__ == "__main__":
    main()
