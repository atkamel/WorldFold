"""Phase I milestone verifier: ``python -m imitation.verify <ID>... | --all | --list``.

Sim-free (stdlib + numpy + json + git), so it runs in either venv. Checks read committed
artifacts only and write ``outputs/imitation/isaac/verify/<ID>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


@check("I0.3", "Isaac smoke test passes in state and hybrid mode; throughput measured; install scripts and footprint tracked")
def _i03() -> Result:
    ev, ok = [], True
    smoke = sorted((ROOT / "outputs" / "isaac" / "smoke").glob("*/"))
    runs = [d for d in smoke if (d / "state.log").exists() and (d / "hybrid.log").exists()]
    if not runs:
        return Result("I0.3", "FAIL", [f"missing artifact: {ROOT / 'outputs/isaac/smoke/<stamp>/{state,hybrid}.log'}"])
    run = runs[-1]
    for mode in ("state", "hybrid"):
        log = (run / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
        passed = "SMOKE OK" in log and "throughput" in log
        tracked = git_tracked(run / f"{mode}.log")
        ev.append(f"{run.name}/{mode}.log SMOKE OK={passed} tracked={tracked}")
        ok &= passed and tracked
    runtime = ROOT / "outputs" / "isaac" / "runtime.json"
    if runtime.exists():
        rows = json.loads(runtime.read_text())["rows"]
        ev.append("throughput " + ", ".join(f"{r['procs']} proc: {r['aggregate']} steps/s ({r['peak_vram_mib']} MiB)"
                                            for r in rows))
        ok &= all(r["completed"] == r["procs"] for r in rows) and git_tracked(runtime)
    else:
        ev.append(f"missing artifact: {runtime}")
        ok = False
    for f in ("isaac/setup_windows.ps1", "isaac/env_windows.ps1", "isaac/INSTALL_REVIEW.md"):
        ok &= git_tracked(ROOT / f)
    review = (ROOT / "isaac" / "INSTALL_REVIEW.md").read_text(encoding="utf-8")
    has_scope = "Measured footprint outside" in review
    ev.append(f"install scripts tracked; INSTALL_REVIEW documents the footprint outside .venv-isaac={has_scope}")
    ok &= has_scope and results_md_has("I0.3")
    return Result("I0.3", "PASS" if ok else "FAIL", ev)


@check("I0.4", "verifier: tests/imitation/test_verify.py exists and passes; --list shows every ID")
def _i04() -> Result:
    t = ROOT / "tests/imitation/test_verify.py"
    if not t.exists():
        return Result("I0.4", "FAIL", [f"missing artifact: {t}"])
    r = subprocess.run([sys.executable, "-m", "pytest", str(t), "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return Result("I0.4", "PASS" if r.returncode == 0 else "FAIL", [f"pytest exit {r.returncode}: {tail}"])


def _pytest(*args: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or ["(no output)"])[-1]


@check("I1.1", "backend switch: MuJoCo unchanged (collection hash), pipeline imports without mujoco, Isaac obs 139-D")
def _i11() -> Result:
    ev, ok = [], True
    code, tail = _pytest("tests/imitation/test_backend.py")
    ev.append(f"test_backend.py exit {code}: {tail}")
    ok &= code == 0
    ok &= results_md_has("I1.1")
    ev.append(f"results.md mentions I1.1={results_md_has('I1.1')} (MuJoCo collection-hash regression row)")
    if isaac_available():
        code, tail = _pytest("-m", "isaac", "tests/imitation/test_isaac_backend.py")
        ev.append(f"test_isaac_backend.py exit {code}: {tail}")
        ok &= code == 0
        return Result("I1.1", "PASS" if ok else "FAIL", ev)
    ev.append("Isaac half: SKIP here (run from .venv-isaac)")
    return Result("I1.1", "PASS" if ok else "FAIL", ev)


@check("I1.3", "camera rig: main 128 + two 64 wrist cameras, right shapes, wrist mount within 1 mm / 0.5 deg")
def _i13() -> Result:
    rig = ROOT / "outputs" / "isaac" / "rig" / "rig.json"
    if not rig.exists():
        return Result("I1.3", "FAIL", [f"missing artifact: {rig}"])
    rep = json.loads(rig.read_text())
    ev, ok = [], git_tracked(rig)
    for name, f in rep["frames"].items():
        good = f["shape"] == f["expected"] and f["dtype"] == "uint8" and f["std"] > 1.0
        ev.append(f"{name}: {f['shape']} {f['dtype']} std {f['std']:.1f} ok={good}")
        ok &= good
    for prefix, p in rep["wrist_pose"].items():
        good = p["pos_err_mm"] < 1.0 and p["rot_err_deg"] < 0.5
        ev.append(f"{prefix}wrist_cam pose error {p['pos_err_mm']:.3f} mm / {p['rot_err_deg']:.2f} deg ok={good}")
        ok &= good
    ev.append(f"dict-mode {rep['dict_steps_per_s']} steps/s; rig.json tracked={git_tracked(rig)}")
    ok &= results_md_has("I1.3")
    return Result("I1.3", "PASS" if ok else "FAIL", ev)


@check("I1.2", "Isaac seed sets: ranges disjoint, MuJoCo sets unchanged, id_hard offsets reachable 200/200 at R_max")
def _i12() -> Result:
    ev, ok = [], True
    code, tail = _pytest("tests/imitation/test_seeds.py")
    ev.append(f"test_seeds.py exit {code}: {tail}")
    ok &= code == 0
    reach = ROOT / "outputs" / "isaac" / "reach.json"
    if not reach.exists():
        return Result("I1.2", "FAIL", ev + [f"missing artifact: {reach}"])
    d = json.loads(reach.read_text())
    row = next((r for r in d["rows"] if abs(r["r_cm"] - d["r_max_cm"]) < 1e-9), None)
    good = row is not None and row["reachable"] == 1.0 and row["n"] >= 200
    ev.append(f"R_max {d['r_max_cm']} cm: reachable {row and row['reachable']} of {row and row['n']} ok={good}")
    ok &= good and git_tracked(reach) and results_md_has("I1.2")
    return Result("I1.2", "PASS" if ok else "FAIL", ev)


@check("I2.1", "Isaac expert (grasp as built) runs as the teacher through EnvPool; pilot rates recorded (no threshold)")
def _i21() -> Result:
    ev, ok = [], True
    for name in ("eval_expert.json", "eval_expert_recovery.json"):
        path = ROOT / "outputs" / "imitation" / "isaac" / "pilot" / name
        if not path.exists():
            ev.append(f"missing artifact: {path}")
            ok = False
            continue
        for s, (k, n) in eval_counts(path).items():
            lo, hi = wilson(k, n)
            ev.append(f"{name} {s}: {k}/{n} [{lo:.1f}, {hi:.1f}] (pilot, no threshold)")
        ok &= git_tracked(path)
    ok &= results_md_has("I2.1")
    if isaac_available():
        code, tail = _pytest("-m", "isaac", "tests/imitation/test_isaac_expert.py")
        ev.append(f"test_isaac_expert.py exit {code}: {tail}")
        ok &= code == 0
    return Result("I2.1", "PASS" if ok else "FAIL", ev)


@check("I3.1", "e2e micro chain on Isaac (collect -> train -> eval -> dagger -> vision -> detector) green")
def _i31() -> Result:
    log = ROOT / "outputs" / "isaac" / "chain" / "i3_1.log"
    if not log.exists():
        return Result("I3.1", "FAIL", [f"missing artifact: {log}"])
    text = log.read_text(encoding="utf-8-sig")
    ok = "6 passed" in text and git_tracked(log) and results_md_has("I3.1")
    ev = [line for line in text.splitlines() if "passed" in line]
    if isaac_available() and os.environ.get("VERIFY_RERUN_CHAIN"):     # 30+ min of sim, so opt-in
        code, tail = _pytest("-m", "isaac and slow", "tests/imitation/test_isaac_chain.py")
        ev.append(f"re-run: exit {code}: {tail}")
        ok &= code == 0
    return Result("I3.1", "PASS" if ok else "FAIL", ev)


@check("I2.2", "DAgger takeover labels: >= 98% of takeovers record a full K-step expert chunk (pilot round)")
def _i22() -> Result:
    f = ROOT / "outputs" / "imitation" / "isaac" / "pilot" / "takeover_labels.json"
    if not f.exists():
        return Result("I2.2", "FAIL", [f"missing artifact: {f}"])
    d = json.loads(f.read_text())
    good = d["takeovers"] > 0 and d["full_rate"] >= 0.98 and d["label_shape"][1] == 12
    ev = [f"{d['version']}: {d['full_chunks']}/{d['takeovers']} full chunks ({d['full_rate']:.1%}), "
          f"{d['cut_by_episode_end']} cut by episode end, label shape {d['label_shape']}"]
    ok = good and git_tracked(f) and results_md_has("I2.2")
    return Result("I2.2", "PASS" if ok else "FAIL", ev)


def _pilot_artifacts(backend: str, tag: str) -> tuple[bool, list[str]]:
    """scripts/isaac_pilot.ps1 -Backend <backend>: every stage's artifact present, tracked and valid."""
    pilot = ROOT / "outputs" / "imitation" / backend / "pilot"
    runs = ROOT / "outputs" / "imitation" / "runs"
    ev, ok = [], True
    want = {"eval_mlp_id_easy": 20, "eval_mlp_recovery": 10, "eval_diff_id_easy": 20, "eval_diff_id_hard": 10,
            "eval_diff_recovery": 10, "eval_vision_id_easy": 10}
    for name, n in want.items():
        path = pilot / f"{name}.json"
        if not path.exists():
            ev.append(f"missing artifact: {path}")
            ok = False
            continue
        counts = eval_counts(path)
        good = all(nn == n for _, nn in counts.values()) and git_tracked(path)
        ev.append(f"{name}: " + ", ".join(f"{s} {k}/{nn}" for s, (k, nn) in counts.items()) + f" ok={good}")
        ok &= good
    for run in (f"{tag}_mlp", f"{tag}_diff", f"{tag}_vision"):
        rj = runs / run / "run.json"
        if not rj.exists():
            ev.append(f"missing artifact: {rj}")
            ok = False
            continue
        hist = json.loads(rj.read_text())["history"]
        fit = hist[-1]["train_loss"] <= 0.5 * hist[0]["train_loss"]
        ev.append(f"{run}: train loss {hist[0]['train_loss']:.4f} -> {hist[-1]['train_loss']:.4f} fits={fit}")
        ok &= fit and git_tracked(rj)
    dag = runs / f"{tag}_dagger" / "history.json"
    if dag.exists():
        rounds = json.loads(dag.read_text())
        rounds = rounds if isinstance(rounds, list) else rounds.get("rounds", [])
        ev.append(f"dagger history: {len(rounds)} entries; tracked={git_tracked(dag)}")
        ok &= len(rounds) >= 2 and git_tracked(dag)
    else:
        ev.append(f"missing artifact: {dag}")
        ok = False
    agree = runs / f"{tag}_detector" / "agreement.json"
    if agree.exists():
        a = json.loads(agree.read_text())
        ev.append(f"detector agreement {a['agree']}/{a['n']} = {a['rate']:.1%}")
        ok &= a["n"] > 0 and git_tracked(agree)
    else:
        ev.append(f"missing artifact: {agree}")
        ok = False
    return ok, ev


@check("I3.2", "pilot run through the real CLIs: every stage's artifact present and valid (viability, not performance)")
def _i32() -> Result:
    ok, ev = _pilot_artifacts("isaac", "isaac_pilot")
    for name in ("expert", "diffusion", "vision"):
        mp4 = ROOT / "docs" / "reports" / "media" / f"isaac_pilot_{name}.mp4"
        ok &= mp4.exists() and git_tracked(mp4)
        ev.append(f"demo {mp4.name}: exists={mp4.exists()} tracked={git_tracked(mp4)}")
    ok &= results_md_has("I3.2")
    return Result("I3.2", "PASS" if ok else "FAIL", ev)


@check("W4", "weld-baseline pilot (isaac_weld): every stage's artifact valid; diffusion id_easy Wilson lower bound > 0")
def _w4() -> Result:
    ok, ev = _pilot_artifacts("isaac_weld", "isaac_weld_pilot")
    path = ROOT / "outputs" / "imitation" / "isaac_weld" / "pilot" / "eval_diff_id_easy.json"
    if path.exists():
        k, n = eval_counts(path)["id_easy"]
        lo, hi = wilson(k, n)
        ev.append(f"diffusion id_easy {k}/{n} [{lo:.1f}, {hi:.1f}] (bar: lower bound > 0)")
        ok &= lo > 0
    ok &= results_md_has("W4")
    return Result("W4", "PASS" if ok else "FAIL", ev)


@check("W5", "weld retrain (scripts/isaac_retrain.ps1): final n = 200 x 3 evals for the privileged and vision bests, "
             "DAgger history and vision run, detector >= 90%, a demo per policy and eval set tracked")
def _w5() -> Result:
    w5 = ROOT / "outputs" / "imitation" / "isaac_weld" / "w5"
    runs = ROOT / "outputs" / "imitation" / "runs"
    media = ROOT / "docs" / "reports" / "media"
    ev, ok = [], True
    for name in ("final_privileged_r4", "final_vision_r2"):
        path = w5 / f"{name}.json"
        if not path.exists():
            ev.append(f"missing artifact: {path}")
            ok = False
            continue
        counts = eval_counts(path)
        good = set(counts) == {"id_easy", "id_hard", "recovery"} and all(n == 200 for _, n in counts.values())
        ev.append(f"{name}: " + ", ".join(f"{s} {k}/{n} [{wilson(k, n)[0]:.1f}, {wilson(k, n)[1]:.1f}]"
                                          for s, (k, n) in counts.items()) + f" ok={good}")
        ok &= good and git_tracked(path)
    for f in (runs / "isaac_v1_weld_dagger" / "history.json", runs / "isaac_v1_weld_vision_t0" / "run.json"):
        ok &= f.exists() and git_tracked(f)
        ev.append(f"{f.parent.name}/{f.name}: exists={f.exists()} tracked={git_tracked(f)}")
    agree = runs / "isaac_v1_weld_detector" / "agreement.json"
    if agree.exists():
        a = json.loads(agree.read_text())
        ev.append(f"detector agreement {a['agree']}/{a['n']} = {a['rate']:.1%} (bar >= 90%)")
        ok &= a["rate"] >= 0.9 and git_tracked(agree)
    else:
        ev.append(f"missing artifact: {agree}")
        ok = False
    for name in [f"{p}_{s}" for p in ("privileged", "sensor") for s in ("id_easy", "id_hard", "recovery")]:
        mp4 = media / f"half_fold_isaac_{name}.mp4"
        ok &= mp4.exists() and git_tracked(mp4)
        ev.append(f"demo {mp4.name}: exists={mp4.exists()} tracked={git_tracked(mp4)}")
    ok &= results_md_has("W5")
    return Result("W5", "PASS" if ok else "FAIL", ev)


@check("W1", "weld grasp on Isaac (GPU pipeline): engages, tracks like MuJoCo's weld lifted/carried, holds on neutral, releases, honours mask")
def _w1() -> Result:
    f = ROOT / "outputs" / "isaac" / "weld_cuda" / "weld.json"
    if not f.exists():
        return Result("W1", "FAIL", [f"missing artifact: {f}"])
    d = json.loads(f.read_text())
    ph = d["phases"]
    ev = [f"device {d['device']}: " + ", ".join(f"{k}={v}" for k, v in d["checks"].items()),
          "max tracking mm: " + ", ".join(f"{k} {ph[k]['max_track_mm']}" for k in ("lift", "carry", "hold_neutral")),
          f"s/step while welded: {ph['carry']['s_per_step']}"]
    ok = d["passed"] and git_tracked(f) and results_md_has("W1")
    if isaac_available():
        code, tail = _pytest("-m", "isaac", "tests/imitation/test_isaac_weld.py")
        ev.append(f"test_isaac_weld.py exit {code}: {tail}")
        ok &= code == 0
    return Result("W1", "PASS" if ok else "FAIL", ev)


W2_REACH_PARITY_PP = 5.0     # Isaac may reach at most this many points fewer than MuJoCo's own arm, per set


@check("W2", "weld profile on Isaac (MuJoCo setup carried over): expert waypoints reachable on MuJoCo poses at parity with MuJoCo's arm, arm >= 90% of a step in one control step, DR deterministic and applied")
def _w2() -> Result:
    reach = ROOT / "outputs" / "isaac" / "reach_mujoco.json"
    ref = ROOT / "outputs" / "isaac" / "reach_mujoco_ref.json"
    prof = ROOT / "outputs" / "isaac" / "profile" / "profile.json"
    missing = [str(p) for p in (reach, ref, prof) if not p.exists()]
    if missing:
        return Result("W2", "FAIL", [f"missing artifact: {m}" for m in missing])
    r, m, p = json.loads(reach.read_text()), json.loads(ref.read_text()), json.loads(prof.read_text())
    ev, ok = [], True
    for name, s in r["sets"].items():
        mj = m["sets"][name]
        gap = 100.0 * (mj["reachable"] / mj["n"] - s["reachable"] / s["n"])
        ev.append(f"reach {name}: isaac {s['reachable']}/{s['n']}, mujoco {mj['reachable']}/{mj['n']} "
                  f"(gap {gap:.1f} pp), isaac worst mm {s['worst_mm']}")
        ok &= s["n"] >= 200 and mj["n"] == s["n"] and gap <= W2_REACH_PARITY_PP
    ok &= r["sets"]["id_easy"]["reachable"] == r["sets"]["id_easy"]["n"]
    ev.append("profile: " + ", ".join(f"{k}={v}" for k, v in p["checks"].items()))
    ev.append(f"arm step fraction per joint: isaac {p['arm_step_fraction']}, mujoco {p['arm_step_fraction_mujoco']}")
    ok &= (p["passed"] and git_tracked(reach) and git_tracked(ref) and git_tracked(prof) and results_md_has("W2"))
    if isaac_available():
        code, tail = _pytest("-m", "isaac", "tests/imitation/test_isaac_profile.py")
        ev.append(f"test_isaac_profile.py exit {code}: {tail}")
        ok &= code == 0
    return Result("W2", "PASS" if ok else "FAIL", ev)


# MuJoCo's M1.7 ceilings were 100 / 97 / 96. Recovery was planned at 90; the Isaac expert reached 81 and the user
# accepted it as the Isaac recovery ceiling on 2026-10-03 (docs/results.md W3), so the bar recorded here is 80.
W3_GATE = {"id_easy": 95.0, "id_hard": 90.0, "recovery": 80.0}


@check("W3", "MuJoCo's expert on Isaac (isaac_weld): gate n >= 100 id_easy >= 95, id_hard >= 90, recovery >= 80 (accepted, planned 90); MuJoCo expert byte-identical")
def _w3() -> Result:
    ev, ok = [], True
    for s, bar in W3_GATE.items():
        path = ROOT / "outputs" / "imitation" / "isaac_weld" / "w3" / f"eval_expert_{s}.json"
        if not path.exists():
            ev.append(f"missing artifact: {path}")
            ok = False
            continue
        k, n = eval_counts(path)[s]
        lo, hi = wilson(k, n)
        rate = 100.0 * k / n
        ev.append(f"{s}: {k}/{n} = {rate:.1f}% [{lo:.1f}, {hi:.1f}] (bar {bar:.0f})")
        ok &= n >= 100 and rate >= bar and git_tracked(path)
    a = ROOT / "outputs" / "imitation" / "expert_benchmark_w3.json"
    b = ROOT / "outputs" / "imitation" / "expert_benchmark_m1_5.json"
    same = a.exists() and json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
    ev.append(f"MuJoCo expert benchmark after the FoldExpert split {'==' if same else '!='} M1.5")
    ok &= same and git_tracked(a) and results_md_has("W3")
    return Result("W3", "PASS" if ok else "FAIL", ev)


# ---------------------------------------------------------------- track G: friction grasp (lehome profile)
GRASP = ROOT / "outputs" / "isaac" / "grasp"
TUNE_BLOCK = (600_000, 700_000)            # imitation.seeds TUNE_SEED_BASE block
GRASP_METRICS = ("acquired", "held", "placed", "released")
IG2_BAR = {"acquired": 98.0, "held": 98.0, "released": 98.0, "placed": 95.0}
IG2_BLOCKS = ("ig2_block_a", "ig2_block_b", "ig2_fresh")


def _grasp_summary(name):
    d = GRASP / name
    s, rows = d / "summary.json", d / "rows.jsonl"
    if not (s.exists() and rows.exists()):
        return None, [f"missing artifact: {s} / {rows.name}"]
    return json.loads(s.read_text(encoding="utf-8")), []


def _grasp_lines(name, s):
    out = []
    for p, st in s["arms"].items():
        out.append(f"{name} {p}: " + ", ".join(
            f"{m} {st[m]['k']}/{st[m]['n']} [{st[m]['wilson'][0]}, {st[m]['wilson'][1]}]" for m in GRASP_METRICS)
            + f", anchor drift max {st['anchor_drift_m']['max']} m")
    return out


@check("IG.1", "friction-grasp bench: per-arm acquired / held / placed / released + anchor drift, baseline (grasp as built) on tune seeds 600000+")
def _ig1() -> Result:
    s, ev = _grasp_summary("ig1_baseline")
    if s is None:
        return Result("IG.1", "FAIL", ev)
    seeds = s["seeds"]
    in_block = all(TUNE_BLOCK[0] <= x < TUNE_BLOCK[1] for x in seeds)
    as_built = all(v is None for v in (s.get("config") or {}).get("knobs", {"x": 1}).values())
    complete = sorted(s["arms"]) == ["left_", "right_"] and all(
        m in st for st in s["arms"].values() for m in GRASP_METRICS + ("anchor_drift_m",))
    ev += _grasp_lines("ig1_baseline", s)
    ev.append(f"n {s['n']}, seeds {seeds[0]}..{seeds[-1]} in tune block={in_block}; knobs as built={as_built}")
    tracked = git_tracked(GRASP / "ig1_baseline" / "summary.json") and git_tracked(GRASP / "ig1_baseline" / "rows.jsonl")
    code, tail = _pytest("tests/imitation/test_grasp_metrics.py")
    ev.append(f"test_grasp_metrics.py exit {code}: {tail}")
    ok = s["n"] >= 20 and in_block and as_built and complete and tracked and code == 0 and results_md_has("IG.1")
    return Result("IG.1", "PASS" if ok else "FAIL", ev)


@check("IG.2", "friction-grasp reliability: per arm, n = 100 on two tune blocks plus a fresh block, acquired / held / released >= 98, placed >= 95, one config")
def _ig2() -> Result:
    ev, ok, configs, seeds = [], True, [], []
    for name in IG2_BLOCKS:
        s, miss = _grasp_summary(name)
        if s is None:
            ev += miss
            ok = False
            continue
        good = s["n"] >= 100 and all(TUNE_BLOCK[0] <= x < TUNE_BLOCK[1] for x in s["seeds"])
        for st in s["arms"].values():
            good &= all(st[m]["rate"] >= IG2_BAR[m] for m in GRASP_METRICS)
        good &= git_tracked(GRASP / name / "summary.json")
        ev += _grasp_lines(name, s)
        ev.append(f"{name}: n {s['n']} meets bar={good}")
        ok &= good
        configs.append(json.dumps(s.get("config", {}).get("knobs"), sort_keys=True)
                       + json.dumps(s.get("config", {}).get("expert"), sort_keys=True))
        seeds.append(set(s["seeds"]))
    if len(seeds) == len(IG2_BLOCKS):
        disjoint = all(not (seeds[i] & seeds[j]) for i in range(3) for j in range(i + 1, 3))
        same = len(set(configs)) == 1
        # the fresh block's seeds appear in no other grasp run (it was never tuned on)
        others = [json.loads(line)["seed"] for d in GRASP.glob("*/rows.jsonl") if d.parent.name != "ig2_fresh"
                  for line in d.read_text(encoding="utf-8").splitlines() if line.strip()]
        fresh = not (seeds[2] & set(others))
        ev.append(f"blocks disjoint={disjoint}, one config={same}, fresh block untouched by tuning={fresh}")
        ok &= disjoint and same and fresh
    ok &= results_md_has("IG.2")
    return Result("IG.2", "PASS" if ok else "FAIL", ev)


IG3_GATE = {"id_easy": 95.0, "id_hard": 90.0, "recovery": 90.0}
IG3_RESYNC = 90.0


@check("IG.3", "friction expert gate on Isaac (lehome): n >= 100 id_easy >= 95, id_hard >= 90, recovery >= 90, check_resync >= 90")
def _ig3() -> Result:
    d = ROOT / "outputs" / "imitation" / "isaac" / "ig3"
    ev, ok = [], True
    for s, bar in IG3_GATE.items():
        path = d / f"eval_expert_{s}.json"
        if not path.exists():
            ev.append(f"missing artifact: {path}")
            ok = False
            continue
        k, n = eval_counts(path)[s]
        lo, hi = wilson(k, n)
        rate = 100.0 * k / n
        ev.append(f"{s}: {k}/{n} = {rate:.1f}% [{lo:.1f}, {hi:.1f}] (bar {bar:.0f})")
        ok &= n >= 100 and rate >= bar and git_tracked(path)
    rs = d / "check_resync.json"
    if rs.exists():
        r = json.loads(rs.read_text(encoding="utf-8"))
        for m, v in r["modes"].items():
            rate = 100.0 * v["success"] / v["n"]
            lo, hi = wilson(v["success"], v["n"])
            ev.append(f"check_resync {m}: {v['success']}/{v['n']} = {rate:.1f}% [{lo:.1f}, {hi:.1f}] (bar {IG3_RESYNC:.0f})")
            ok &= v["n"] >= 100 and rate >= IG3_RESYNC
        ok &= git_tracked(rs)
    else:
        ev.append(f"missing artifact: {rs}")
        ok = False
    ok &= results_md_has("IG.3")
    return Result("IG.3", "PASS" if ok else "FAIL", ev)


@check("I3.3", "close-out: every other Phase I check passes here, roadmap Phase I rows closed, docs updated")
def _i33() -> Result:
    ev, ok = [], True
    for mid, (_, fn) in CHECKS.items():
        if mid == "I3.3" or not mid.startswith("I") or mid.startswith("IG"):   # Phase I's viability checks only
            continue
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001 -- a crashing check is a failing check
            r = Result(mid, "FAIL", [f"check raised {exc!r}"])
        ev.append(f"{mid}: {r.status}")
        ok &= r.status == "PASS"
    roadmap = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    open_rows = [ln.split("|")[1].strip() for ln in roadmap.splitlines()
                 if ln.startswith("| I") and ln.rstrip().endswith("☐ |") and not ln.startswith("| I3.3 ")]   # this row closes on PASS
    ev.append(f"open Phase I roadmap rows: {open_rows or 'none'}")
    ok &= not open_rows
    spec = "Simulator backends" in (ROOT / "docs" / "imitation.md").read_text(encoding="utf-8")
    # the friction-grasp follow-up (IG.1) stays scheduled in the Next action section, beside later phases' work
    nxt = "IG.1" in (ROOT / "docs" / "status.md").read_text(encoding="utf-8").split("## Next action")[1].split("\n## ")[0]
    ev.append(f"imitation.md backend section={spec}; status next action is IG.1={nxt}")
    ok &= spec and nxt
    return Result("I3.3", "PASS" if ok else "FAIL", ev)


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
