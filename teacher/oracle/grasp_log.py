"""Per-grasp outcome logger for a policy running in the LeHome Isaac env: pure numpy/scipy, testable offline.

Every time an arm's jaw closes it records which cloth vertices ended up between the flat pads, then follows them
until the jaw opens again:
  empty          fewer than MIN_PINCH vertices between the pads when the jaw closed
  slipped        the held share fell below SLIP_FRAC while the jaw was still closed
  held_lifted    held to the end and the pinched cloth rose at least LIFT_M since the jaw closed
  held_dragged   held to the end, not lifted, but carried at least DRAG_M along the table
  held_static    held to the end without lifting or moving
plus the number of cloth layers taken: mesh-connected pieces among the pinched vertices (two layers that touch in
space are far apart on the mesh). Used by grasp_log_patch.py (the teacher's episode loop) and test_grasp_log.py.
"""
import numpy as np

CLOSE_RAD = 0.19        # midpoint of GRIP_OPEN 0.55 / GRIP_CLOSED -0.17: the jaw is "closing" below this
OPEN_RAD = 0.25         # ... and "open again" above this (hysteresis)
SETTLE_STEPS = 6        # policy steps after the close crossing before the pinched set is taken (jaw finishes closing)
MIN_PINCH = 5           # fewer pinched vertices = an empty grasp
HELD_R = 0.02           # a pinched vertex still counts as held within 2 cm of the jaw tip (as in assay_trial)
SLIP_FRAC = 0.5
LIFT_M = 0.02
DRAG_M = 0.03
MIN_LAYER = 3           # mesh pieces smaller than this are not counted as a layer

# gripper-link frame from the jaw-tip frame (oracle_fold._t_link_tip) and the flat-pad pinch box in that frame
# (oracle_fold.between_pads), copied here because oracle_fold imports Isaac at module level
_RY = np.array([[-1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]])
_T = np.eye(4); _T[:3, 3] = [-0.0079, -0.000218, -0.0981]
T_LINK_TIP = _T @ _RY
BOX = dict(y=0.012, z_lo=-0.103, z_hi=-0.075, x_lo=-0.010, x_hi=0.004)


def pad_mask(T_tip, verts):
    """Boolean mask of the vertices inside the flat-pad pinch zone. T_tip: 4x4 jaw-tip pose in the world."""
    T = T_tip @ np.linalg.inv(T_LINK_TIP)
    v = (np.linalg.inv(T) @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    return ((np.abs(v[:, 1]) < BOX["y"]) & (v[:, 2] > BOX["z_lo"]) & (v[:, 2] < BOX["z_hi"])
            & (v[:, 0] > BOX["x_lo"]) & (v[:, 0] < BOX["x_hi"]))


def layers(idx, faces):
    """Mesh-connected pieces (>= MIN_LAYER vertices) among the vertex indices idx."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0 or faces is None:
        return 0, []
    pos = -np.ones(int(max(idx.max(), faces.max())) + 1, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a, b = pos[e[:, 0]], pos[e[:, 1]]
    keep = (a >= 0) & (b >= 0)
    g = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(len(idx), len(idx)))
    n, lab = connected_components(g, directed=False)
    sizes = sorted((int(s) for s in np.bincount(lab, minlength=n)), reverse=True)
    big = [s for s in sizes if s >= MIN_LAYER]
    return len(big), sizes


class GraspLogger:
    """Feed it every policy step; it returns finished grasp records.

    tip_pose(q5, side) -> 4x4 jaw-tip pose in the world (fold_plans.K.fk in the env).
    faces: (F,3) cloth triangles; table_z: table height (m)."""

    def __init__(self, tip_pose, faces, table_z, episode=None):
        self.tip_pose, self.faces, self.table_z, self.episode = tip_pose, faces, float(table_z), episode
        # armed: the jaw has been seen open since the last grasp (an episode that starts with closed jaws is no grasp)
        self.state = {s: {"closed": False, "armed": False} for s in ("left", "right")}
        self.records = []

    def _start(self, side, t, q, verts):
        st = self.state[side]
        T = self.tip_pose(q, side)
        idx = np.nonzero(pad_mask(T, verts))[0]
        nl, sizes = layers(idx, self.faces)
        z0 = float(np.median(verts[idx, 2])) if len(idx) else 0.0
        st.update(closed=True, t_close=t, idx=idx, n_pinch=int(len(idx)), layers=nl, piece_sizes=sizes[:4],
                  tip0=T[:3, 3].copy(), z0=z0, h0=z0 - self.table_z, max_lift=0.0, max_carry=0.0, min_held=1.0,
                  slip_t=None, slip_lift=None)

    def _track(self, side, t, q, verts):
        st = self.state[side]
        if st["n_pinch"] < MIN_PINCH:
            return
        tip = self.tip_pose(q, side)[:3, 3]
        v = verts[st["idx"]]
        held = float((np.linalg.norm(v - tip, axis=1) < HELD_R).mean())
        lift = float(np.median(v[:, 2]) - st["z0"])         # rise of the pinched cloth since the jaw closed
        st["max_lift"] = max(st["max_lift"], lift)
        if lift >= 0.04 and st.get("t_lift4") is None:
            st["t_lift4"] = t                                # first step the pinched cloth was 4 cm up
        st["last_lift"] = lift                               # the rise at the last step before the release
        st["max_carry"] = max(st["max_carry"], float(np.linalg.norm((tip - st["tip0"])[:2])))
        st["min_held"] = min(st["min_held"], held)
        if held < SLIP_FRAC and st["slip_t"] is None:
            st["slip_t"], st["slip_lift"] = t, round(lift * 100, 1)

    def _finish(self, side, t, end_reason):
        st = self.state[side]
        if st["n_pinch"] < MIN_PINCH:
            outcome = "empty"
        elif st["slip_t"] is not None:
            outcome = "slipped"
        elif st["max_lift"] >= LIFT_M:
            outcome = "held_lifted"
        elif st["max_carry"] >= DRAG_M:
            outcome = "held_dragged"
        else:
            outcome = "held_static"
        r = dict(episode=self.episode, side=side, t_close=st["t_close"], t_open=t, end=end_reason, outcome=outcome,
                 n_pinch=st["n_pinch"], layers=st["layers"], piece_sizes=st["piece_sizes"],
                 h_close_cm=round(st.get("h0", 0.0) * 100, 1),
                 max_lift_cm=round(st["max_lift"] * 100, 1), lift_open_cm=round(st.get("last_lift", 0.0) * 100, 1),
                 carry_cm=round(st["max_carry"] * 100, 1),
                 min_held=round(st["min_held"], 2), slip_t=st["slip_t"], slip_lift_cm=st["slip_lift"],
                 t_lift4=st.get("t_lift4"), pinch_idx=[int(i) for i in st["idx"][:400]])
        self.records.append(r)
        self.state[side] = {"closed": False, "armed": end_reason == "opened"}
        return r

    def update(self, t, state12, verts):
        """t: policy step index; state12: the env's 12-dim joint state (arm 0-4, grip 5, arm 6-10, grip 11);
        verts: (N,3) cloth vertices in the world. Returns the grasp records finished at this step."""
        s = np.asarray(state12, dtype=float).ravel()
        out = []
        for side, q, g in (("left", s[0:5], s[5]), ("right", s[6:11], s[11])):
            st = self.state[side]
            if st["closed"]:
                self._track(side, t, q, verts)
                if g > OPEN_RAD:
                    out.append(self._finish(side, t, "opened"))
            elif "pending" in st:
                if g > OPEN_RAD:                     # reopened before settling: not a grasp
                    st.pop("pending"); st["armed"] = True
                elif t - st["pending"] >= SETTLE_STEPS:
                    st.pop("pending"); self._start(side, t, q, verts)
            elif g > OPEN_RAD:
                st["armed"] = True
            elif g < CLOSE_RAD and st["armed"]:
                st["pending"], st["armed"] = t, False
        return out

    def close_episode(self, t):
        """Episode ended: finish any grasp still held (end='episode_end')."""
        return [self._finish(s, t, "episode_end") for s in ("left", "right") if self.state[s]["closed"]]


def summary(records):
    """Counts per outcome and layer count, over all records."""
    out = {"grasps": len(records)}
    for k in ("empty", "slipped", "held_lifted", "held_dragged", "held_static"):
        out[k] = sum(r["outcome"] == k for r in records)
    ne = [r for r in records if r["outcome"] != "empty"]
    out["held_share_nonempty"] = round(sum(r["outcome"].startswith("held") for r in ne) / len(ne), 3) if ne else None
    out["layers"] = {str(n): sum(r["layers"] == n for r in ne) for n in sorted({r["layers"] for r in ne})}
    return out
