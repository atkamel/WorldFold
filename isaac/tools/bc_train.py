"""Train behaviour-cloning students on the bc_collect demos (overnight plan 2026-11-09), several at once on one GPU.

Merges every frozen version matching --versions into --merged (unless it exists), then runs imitation.train for each
--runs entry in parallel: name:policy:steps[:batch], into <out>/<name>_<merged>.

  isaac/watcloud.sbatch isaac/tools/bc_train.py --versions 'bc_stiff_t*_r*' --merged bc_stiff_m1 \\
      --runs state_diff:diffusion:30000,state_mlp:chunk_mlp:30000,vision:vision:30000:256
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaac.tools.bc_collect import ROOT   # noqa: E402


def frozen(root, pattern):
    return sorted(Path(p).parent.name for p in glob.glob(str(Path(root) / pattern / "manifest.json")))


def parse_runs(text):
    out = []
    for item in text.split(","):
        name, policy, steps, *rest = item.split(":")
        out.append({"name": name, "policy": policy, "steps": int(steps), "batch": int(rest[0]) if rest else None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--versions", required=True, help="glob of frozen versions to merge")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--runs", required=True, help="name:policy:steps[:batch],...")
    ap.add_argument("--out", default="outputs/isaac/bc/runs")
    ap.add_argument("--allow-failures", action="store_true", help="train on the expert's failed episodes too")
    args = ap.parse_args()
    from imitation.data.schema import merge_versions
    if not (Path(args.root) / args.merged / "manifest.json").exists():
        versions = frozen(args.root, args.versions)
        m = merge_versions(args.root, versions, args.merged)
        print("BCTRAIN merged", len(versions), "versions:", m["n_episodes"], "episodes", m["n_steps"], "steps",
              flush=True)
    procs = []
    for r in parse_runs(args.runs):
        run = Path(args.out) / f"{r['name']}_{args.merged}"
        cmd = [sys.executable, "-u", "-m", "imitation.train", "--policy", r["policy"], "--dataset", args.merged,
               "--root", args.root, "--run", str(run), "--steps", str(r["steps"])]
        if r["batch"]:
            cmd += ["--batch", str(r["batch"])]
        if args.allow_failures:
            cmd += ["--allow-failures"]
        run.mkdir(parents=True, exist_ok=True)
        procs.append((r["name"], subprocess.Popen(cmd, stdout=open(run / "train.log", "w"), stderr=subprocess.STDOUT,
                                                  cwd=REPO)))
        print("BCTRAIN start", r["name"], " ".join(cmd), flush=True)
    for name, p in procs:
        print("BCTRAIN done", name, p.wait(), flush=True)


if __name__ == "__main__":
    main()
