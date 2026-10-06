"""Merge frozen dataset versions (e.g. collection shards from separate machines) into one new version.

    python -m imitation.data.merge --version v1 --from v1_s00 v1_s01 ...

A shard whose collection was cut off before it froze is frozen first, from its journal. The merged manifest lists
the shards' episodes where they are, as an aggregated DAgger version lists its parent's,
so nothing is copied; the shards must stay. Seeds must not repeat across shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imitation.data.collect import DEFAULT_ROOT, summary
from imitation.data.schema import DatasetWriter, load_manifest


def freeze_interrupted(root, version) -> None:
    """Freeze a version whose collection was cut off (no manifest yet), keeping every episode its journal lists."""
    if not (Path(root) / version / "manifest.json").exists():
        DatasetWriter(root, version, resume=True).freeze()


def merge(root, version, shards, config=None) -> dict:
    for s in shards:
        freeze_interrupted(root, s)
    entries = [e for s in shards for e in load_manifest(root, s)["episodes"]]
    seen = set()
    for e in entries:
        key = (e["source"], e["seed"])
        if key in seen:
            raise ValueError(f"{key} is in more than one of {shards}")
        seen.add(key)
    writer = DatasetWriter(root, version, config={"merged_from": list(shards), **(config or {})})
    writer.entries = entries
    return writer.freeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--from", dest="shards", nargs="+", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    m = merge(args.root, args.version, args.shards)
    print(f"froze {args.root}/{args.version}: {summary(m['episodes'])}, hash {m['content_hash'][:12]}")


if __name__ == "__main__":
    main()
