"""Render camera images for an already-frozen dataset by replaying its actions (M4.1).

    python -m imitation.vision.replay --version v1 --out v1_img --workers 14

Collection is deterministic per seed and the camera rig only touches render-side model
fields, so replaying an episode's executed actions revisits exactly its states; every
replayed observation is checked against the stored one. The result is a new frozen
version holding the same episodes plus images (the source stays write-once and
untouched). Resumable with --resume.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.schema import DatasetWriter, Episode, load_manifest
from imitation.spec import ACTION_DIM, OBS_DIM

_ENV = None


def _replay(path):
    global _ENV
    from imitation.tasks import HalfFoldEnv
    from imitation.vision.render import CameraRig
    if _ENV is None:
        env = HalfFoldEnv()
        _ENV = (env, CameraRig(env))
    env, rig = _ENV
    ep = Episode.load(path)
    seed = ep.meta["seed"]
    obs, _ = env.reset(seed=seed)
    rig.reset(seed)
    frames = []
    for t, action in enumerate(ep.actions):
        if not np.array_equal(obs, ep.obs[t]):
            raise RuntimeError(f"replay of seed {seed} diverged at step {t}")
        frames.append(rig.render())
        obs, *_ = env.step(action)
    ep.images = {cam: np.stack([f[cam] for f in frames]) for cam in frames[0]}
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", default=None, help="default <version>_img")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = args.out or f"{args.version}_img"

    src = load_manifest(args.root, args.version)
    writer = DatasetWriter(args.root, out, resume=args.resume,
                           config={"rendered_from": args.version, "source_hash": src["content_hash"]})
    todo = [Path(args.root) / e["file"] for e in src["episodes"] if e["seed"] not in writer.done_seeds]
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, ep in enumerate(pool.imap_unordered(_replay, todo), 1):
            writer.add(ep, OBS_DIM, ACTION_DIM)
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    m = writer.freeze()
    print(f"froze {out}: {m['n_episodes']} episodes with images, hash {m['content_hash'][:12]}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
