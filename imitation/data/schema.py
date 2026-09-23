"""Episode schema and versioned, write-once dataset store (design doc section 5).

On disk a dataset version is a directory:

    <root>/<version>/
        manifest.json      version, parent, created, config, per-episode entries
                           {file, sha256, seed, source, success, steps, ...}
        episodes/*.npz     one file per episode (only the ones NEW in this version)

A version lists its parent's episodes plus its own, so DAgger aggregation never
copies or edits earlier data. Once `freeze()` writes the manifest the version is
read-only: the store refuses to add to it, and `load()` verifies every hash.

Per-episode arrays (T = executed steps):
    obs          [T, D]     observation the action was chosen from
    actions      [T, A]     action actually executed (expert or student)
    rewards      [T]
    stage        [T]  int8
    fold_score   [T]
    grasped      [T, 2] bool (left, right)
    actor        [T]  int8  who chose each executed action: 0 teacher, 1 student, 2 perturbation
    final_obs    [D]        observation after the last step
    label_steps  [L]  int   steps that carry a teacher chunk label (DAgger)
    labels       [L, K, A]  the teacher's chunk from that step's state
Expert episodes have no labels: the teacher executed its own actions, so the
chunk at t is actions[t:t+K].
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SOURCES = ("expert", "student", "dagger")
ARRAY_KEYS = ("obs", "actions", "rewards", "stage", "fold_score", "grasped", "actor", "final_obs",
              "label_steps", "labels")
ACTOR_TEACHER, ACTOR_STUDENT, ACTOR_PERTURB = 0, 1, 2


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    stage: np.ndarray
    fold_score: np.ndarray
    grasped: np.ndarray
    actor: np.ndarray
    final_obs: np.ndarray
    meta: dict
    label_steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32))

    @property
    def steps(self):
        return len(self.actions)

    def save(self, path: Path) -> str:
        arrays = {k: getattr(self, k) for k in ARRAY_KEYS}
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **arrays)
        return _sha256(path)

    @classmethod
    def load(cls, path: Path) -> "Episode":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ARRAY_KEYS}
            meta = json.loads(str(z["meta"]))
        return cls(meta=meta, **arrays)


def validate_episode(ep: Episode, obs_dim=None, action_dim=None) -> list[str]:
    """Integrity problems with one episode; an empty list means it is valid."""
    errs = []
    T = ep.steps
    if T == 0:
        return ["empty episode"]
    for k in ("obs", "rewards", "stage", "fold_score", "grasped", "actor"):
        if len(getattr(ep, k)) != T:
            errs.append(f"{k} has length {len(getattr(ep, k))}, expected {T}")
    for k in ("obs", "actions", "rewards", "fold_score", "final_obs", "labels"):
        if not np.all(np.isfinite(getattr(ep, k))):
            errs.append(f"{k} has non-finite values")
    if np.any(np.abs(ep.actions) > 1.0 + 1e-6):
        errs.append("actions outside [-1, 1]")
    if len(ep.labels):
        if len(ep.labels) != len(ep.label_steps):
            errs.append("labels / label_steps length mismatch")
        if np.any(np.abs(ep.labels) > 1.0 + 1e-6):
            errs.append("labels outside [-1, 1]")
        if np.any((ep.label_steps < 0) | (ep.label_steps >= T)):
            errs.append("label_steps out of range")
    if obs_dim is not None and ep.obs.shape[1:] != (obs_dim,):
        errs.append(f"obs dim {ep.obs.shape[1:]} != {obs_dim}")
    if action_dim is not None and ep.actions.shape[1:] != (action_dim,):
        errs.append(f"action dim {ep.actions.shape[1:]} != {action_dim}")
    for k in ("seed", "source", "success", "termination_reason"):
        if k not in ep.meta:
            errs.append(f"meta missing {k}")
    if ep.meta.get("source") not in SOURCES:
        errs.append(f"unknown source {ep.meta.get('source')}")
    return errs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DatasetWriter:
    """Builds one new dataset version, optionally on top of a frozen parent."""

    def __init__(self, root, version, parent=None, config=None):
        self.root = Path(root)
        self.dir = self.root / version
        if (self.dir / "manifest.json").exists():
            raise FileExistsError(f"dataset version {version} is frozen; write a new version instead")
        (self.dir / "episodes").mkdir(parents=True, exist_ok=True)
        self.version, self.parent, self.config = version, parent, config or {}
        self.entries = list(load_manifest(self.root, parent)["episodes"]) if parent else []
        self._n_new = 0

    def add(self, ep: Episode, obs_dim=None, action_dim=None):
        errs = validate_episode(ep, obs_dim, action_dim)
        if errs:
            raise ValueError(f"invalid episode (seed {ep.meta.get('seed')}): {errs}")
        name = f"ep_{self._n_new:05d}_{ep.meta['source']}_s{ep.meta['seed']}.npz"
        sha = ep.save(self.dir / "episodes" / name)
        self.entries.append({"file": f"{self.version}/episodes/{name}", "sha256": sha, "steps": ep.steps,
                             "n_labels": int(len(ep.label_steps)),
                             **{k: ep.meta[k] for k in ("seed", "source", "success", "termination_reason")},
                             **({"round": ep.meta["round"]} if "round" in ep.meta else {})})
        self._n_new += 1

    def freeze(self) -> dict:
        digest = hashlib.sha256("".join(e["sha256"] for e in self.entries).encode()).hexdigest()
        manifest = {"version": self.version, "parent": self.parent,
                    "created": _dt.datetime.now().isoformat(timespec="seconds"),
                    "content_hash": digest, "config": self.config,
                    "n_episodes": len(self.entries), "n_new": self._n_new,
                    "n_success": sum(e["success"] for e in self.entries),
                    "n_steps": sum(e["steps"] for e in self.entries), "episodes": self.entries}
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest


def load_manifest(root, version) -> dict:
    with open(Path(root) / version / "manifest.json") as f:
        return json.load(f)


def load_dataset(root, version, verify=True, sources=None) -> tuple[dict, list[Episode]]:
    """All episodes of a frozen version (its own and its ancestors')."""
    root = Path(root)
    manifest = load_manifest(root, version)
    episodes = []
    for e in manifest["episodes"]:
        if sources is not None and e["source"] not in sources:
            continue
        path = root / e["file"]
        if verify and _sha256(path) != e["sha256"]:
            raise ValueError(f"{path} does not match its manifest hash: frozen data was modified")
        episodes.append(Episode.load(path))
    return manifest, episodes
