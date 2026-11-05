"""Download meta/* and the data parquet files covering the first ~N episodes of
public real-hardware SO-101 LeRobot datasets (no videos).
Usage: python download_real_datasets.py [max_episodes]
"""
import sys, os, json
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATASETS = [
    "dobri420/pick-cube-so101",
    "HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1",
    "DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task",
    "Chaenn/so101_cube_pnp_real_0908",
    "dharak22/so101-red-cube_20260908_113622",
    "vasco281204/so101_green_block",
]
MAXEP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
api = HfApi()
log = {}
for rid in DATASETS:
    local = os.path.normpath(os.path.join(ROOT, rid.replace("/", "__")))
    sha = api.dataset_info(rid).sha
    # 1) metadata only (pinned revision)
    snapshot_download(rid, repo_type="dataset", revision=sha, local_dir=local, allow_patterns=["meta/*"])
    info = json.load(open(os.path.join(local, "meta", "info.json")))
    cv = info.get("codebase_version")
    if cv == "v3.0":
        epdir = os.path.join(local, "meta", "episodes")
        eps = pd.concat([pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(epdir) for f in fs if f.endswith(".parquet")])
        eps = eps.sort_values("episode_index")
        want = eps[eps.episode_index < MAXEP]
        pairs = sorted(set(zip(want["data/chunk_index"], want["data/file_index"])))
        tmpl = info["data_path"]
        files = [tmpl.format(chunk_index=c, file_index=f) for c, f in pairs]
    else:
        tmpl = info["data_path"]; cs = info.get("chunks_size", 1000)
        files = [tmpl.format(episode_chunk=e // cs, episode_index=e) for e in range(min(MAXEP, info["total_episodes"]))]
    snapshot_download(rid, repo_type="dataset", revision=sha, local_dir=local, allow_patterns=files)
    log[rid] = dict(local_dir=local, revision=sha, files=files)
    print(rid, sha[:10], len(files), "data files", flush=True)
json.dump(log, open(os.path.join(ROOT, "download_log.json"), "w"), indent=1)
