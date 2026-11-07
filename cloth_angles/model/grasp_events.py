"""Grasp-flag events and their metrics, numpy only (the Isaac collector imports it before Kit starts).

A gained grasp is an onset. A lost grasp is a release when the arm's gripper command at that step opens the jaw (above
OPEN_ABOVE, the simulators' hysteresis) and a slip when it does not.
"""

from __future__ import annotations

import numpy as np

OPEN_ABOVE = 0.3
EVENTS = ("onset", "release", "slip")


def grasp_events(now, nxt, command):
    """Boolean masks of the same shape as now ([T] or [N, arms]): onset (0 -> 1), release (1 -> 0 with the jaw
    commanded open), slip (1 -> 0 with it commanded closed)."""
    now, nxt, opened = np.asarray(now) > 0.5, np.asarray(nxt) > 0.5, np.asarray(command) > OPEN_ABOVE
    lost = now & ~nxt
    return {"onset": ~now & nxt, "release": lost & opened, "slip": lost & ~opened}


def grasp_counts(now, true, pred, command):
    """Counts for accuracy and per-event precision / recall: pred / true are next flags, now the current ones."""
    true_ev, pred_ev = grasp_events(now, true, command), grasp_events(now, pred, command)
    out = {"correct": int(((np.asarray(pred) > 0.5) == (np.asarray(true) > 0.5)).sum()), "total": int(np.size(true))}
    for e in EVENTS:
        out[f"{e}_tp"] = int((true_ev[e] & pred_ev[e]).sum())
        out[f"{e}_pred"] = int(pred_ev[e].sum())
        out[f"{e}_true"] = int(true_ev[e].sum())
    return out


def grasp_rates(counts):
    """Accuracy, and per event precision and recall (None where nothing was predicted or nothing happened)."""
    out = {"accuracy": counts["correct"] / max(counts["total"], 1)}
    for e in EVENTS:
        tp, pred, true = counts[f"{e}_tp"], counts[f"{e}_pred"], counts[f"{e}_true"]
        out[e] = {"precision": tp / pred if pred else None, "recall": tp / true if true else None, "n": true}
    return out
