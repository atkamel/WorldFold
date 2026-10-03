"""Phase I milestone verifier: ``python -m imitation.verify <ID>... | --all | --list``.

Sim-free (stdlib + numpy + json + git), so it runs in either venv. Checks read committed
artifacts only and write ``outputs/imitation/isaac/verify/<ID>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    milestone: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    evidence: list[str] = field(default_factory=list)


CHECKS: dict[str, tuple[str, Callable[[], Result]]] = {}


def check(mid: str, description: str):
    def deco(fn):
        CHECKS[mid] = (description, fn)
        return fn
    return deco


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent."""
    if n <= 0:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def eval_counts(path) -> dict[str, tuple[int, int]]:
    res = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    return {s: (round(v["success_rate"] * v["n"]), int(v["n"])) for s, v in res.items()}


def results_md_has(token: str) -> bool:
    p = ROOT / "docs" / "results.md"
    return p.exists() and token in p.read_text(encoding="utf-8")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def git_tracked(path) -> bool:
    return bool(_git("ls-files", "--error-unmatch", "--", str(path)).returncode == 0)


def loss_dropped(history_path, ratio: float = 0.5) -> bool:
    """True if the last logged loss <= ratio * the first. Accepts a list of
    {step, train_loss, val_loss} or a run.json-style dict with a "history" key."""
    p = Path(history_path)
    if not p.exists():
        return False
    h = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(h, dict):
        h = h.get("history", [])
    key = "val_loss" if h and "val_loss" in h[0] else "train_loss"
    vals = [e[key] for e in h if key in e]
    return len(vals) >= 2 and vals[-1] <= ratio * vals[0]


def isaac_available() -> bool:
    return importlib_util.find_spec("isaacsim") is not None


def _is_ancestor(commit: str) -> bool:
    return _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def _head() -> str:
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else "unknown"


# ---------------------------------------------------------------- checks
@check("I0.1", "merge of origin/main is an ancestor; expert benchmark unchanged vs M1.5; logged in results.md")
def _i01() -> Result:
    ev, ok = [], True
    if _is_ancestor("ce02b44"):
        ev.append("ce02b44 is an ancestor of HEAD")
    else:
        ok = False
        ev.append("ce02b44 is NOT an ancestor of HEAD")
    a = ROOT / "outputs/imitation/expert_benchmark_i0_1.json"
    b = ROOT / "outputs/imitation/expert_benchmark_m1_5.json"
    missing = [p for p in (a, b) if not p.exists()]
    for p in missing:
        ok = False
        ev.append(f"missing artifact: {p}")
    if not missing:
        same = json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
        ok &= same
        ev.append(f"{a.name} {'==' if same else '!='} {b.name}")
    has = results_md_has("I0.1")
    ok &= has
    ev.append("docs/results.md mentions I0.1" if has else "docs/results.md has no I0.1 row")
    return Result("I0.1", "PASS" if ok else "FAIL", ev)


_LOCKS = ["requirements-lehome-windows.lock", "requirements-isaaclab-windows.lock",
          "requirements-torch-cu128-windows.lock"]


def _unhashed_requirements(text: str) -> list[str]:
    """Requirement names lacking a --hash=sha256: (continuation lines are joined)."""
    bad, cur, hashed = [], None, False
    for raw in text.splitlines() + [""]:
        s = raw.strip()
        starts = bool(s) and not s.startswith(("#", "--")) and not raw[:1].isspace()
        if starts:
            if cur is not None and not hashed:
                bad.append(cur)
            cur, hashed = s.rstrip("\\").strip(), False
        if "--hash=sha256:" in s:
            hashed = True
    if cur is not None and not hashed:
        bad.append(cur)
    return bad


@check("I0.2", "install review tracked and SAFE; Windows lock files tracked and fully hashed")
def _i02() -> Result:
    ev, ok = [], True
    rev = ROOT / "isaac/INSTALL_REVIEW.md"
    if not rev.exists():
        ok = False
        ev.append(f"missing artifact: {rev}")
    else:
        t = git_tracked("isaac/INSTALL_REVIEW.md")
        s = "SAFE" in rev.read_text(encoding="utf-8")
        ok &= t and s
        ev.append(f"INSTALL_REVIEW.md tracked={t} contains SAFE={s}")
    for name in _LOCKS:
        p = ROOT / "isaac" / name
        if not p.exists():
            ok = False
            ev.append(f"missing artifact: {p}")
            continue
        t = git_tracked(f"isaac/{name}")
        bad = _unhashed_requirements(p.read_text(encoding="utf-8"))
        ok &= t and not bad
        ev.append(f"{name} tracked={t} unhashed={len(bad)}" + (f" e.g. {bad[:3]}" if bad else ""))
    return Result("I0.2", "PASS" if ok else "FAIL", ev)


@check("I0.3", "Isaac env installs and imports (setup/env scripts)")
def _i03() -> Result:
    return Result("I0.3", "SKIP", ["not implemented yet"])


@check("I0.4", "verifier: tests/imitation/test_verify.py exists and passes; --list shows every ID")
def _i04() -> Result:
    t = ROOT / "tests/imitation/test_verify.py"
    if not t.exists():
        return Result("I0.4", "FAIL", [f"missing artifact: {t}"])
    r = subprocess.run([sys.executable, "-m", "pytest", str(t), "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return Result("I0.4", "PASS" if r.returncode == 0 else "FAIL", [f"pytest exit {r.returncode}: {tail}"])


for _mid, _desc in [
    ("I1.1", "backend switch: MuJoCo unchanged, Isaac obs 139-D finite, imports without mujoco"),
    ("I1.2", "Isaac seed sets disjoint; id_hard ring reachable"),
    ("I1.3", "camera rig: 3 cameras, right shapes, wrist pose within tolerance"),
    ("I2.1", "Isaac expert (grasp as built): runs clean, rates recorded"),
    ("I2.2", "DAgger labels: >= 80% of chunks within 0.05"),
    ("I3.1", "e2e micro chain: collect -> train -> eval -> dagger -> vision -> detector"),
    ("I3.2", "pilot run: every artifact valid"),
    ("I3.3", "close-out: verify --all PASS in both venvs"),
]:
    CHECKS[_mid] = (_desc, (lambda m=_mid: Result(m, "SKIP", ["not implemented yet"])))


# ---------------------------------------------------------------- CLI
def run_one(mid: str) -> Result:
    desc, fn = CHECKS[mid]
    try:
        res = fn()
    except Exception as e:  # a crashing check is a failing check
        res = Result(mid, "FAIL", [f"check raised {type(e).__name__}: {e}"])
    out = ROOT / "outputs/imitation/isaac/verify"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"milestone": mid, "status": res.status, "evidence": res.evidence, "commit": _head(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / f"{mid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="imitation.verify")
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for mid, (desc, _) in CHECKS.items():
            print(f"{mid}  {desc}")
        return 0
    ids = list(CHECKS) if a.all else a.ids
    if not ids:
        ap.error("give milestone IDs, --all, or --list")
    failed = False
    for mid in ids:
        if mid not in CHECKS:
            print(f"FAIL {mid}\n  unknown milestone (see --list)")
            failed = True
            continue
        res = run_one(mid)
        print(f"{res.status} {mid}")
        for line in res.evidence:
            print(f"  {line}")
        failed |= res.status == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
