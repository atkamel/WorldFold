"""Live MuJoCo folding server: opens a viewer window and executes one command at a time
sent by client.py over TCP (localhost:7788). Nothing here plans a fold; it only moves
the arm tips where it is told, opens/closes grippers, and reports state + area score.

Grasp is simplified: on 'close', cloth vertices within GRAB_R of the jaw tip are pinned
to the tip and follow it until 'open'. Arms do not collide with the cloth.
"""
import json, socket, threading, queue, time, sys
import numpy as np
import mujoco, mujoco.viewer
import scene

GRAB_R = 0.022
ARMS = ("L", "R")
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OPEN, CLOSED = 1.2, 0.0


def build_model():
    import importlib
    txt = open(scene.__file__, encoding="utf-8-sig").read()
    txt = txt.replace("<option ", '<option solver="CG" ').replace('radius="0.002"', 'radius="0.006"')
    # stiffer, thicker cloth: folded layers sank into each other 30% -> 15% (foldtest.py), looks solid not liquid
    txt = txt.replace('young="3e4"', 'young="1e5"').replace('solref="0.004 1"', 'solref="0.008 1"')
    # 150 g: lighter cloth slid as a whole sheet when a sleeve was pulled (drag test: 4.5 cm -> 1.1 cm)
    txt = txt.replace('mass="0.05"', 'mass="0.15"')
    txt = txt.replace('timestep="0.002"', 'timestep="0.004"')  # 2x faster; cloth stayed stable in tests
    # draw the table 6 mm low so the cloth (which rests ~1-3 mm into the contact) is visible
    txt = txt.replace('<geom name="table" type="plane" size="0.6 0.6 0.01" material="table"',
                      '<geom name="table_vis" type="plane" pos="0 0 -0.006" size="0.6 0.6 0.01" material="table" contype="0" conaffinity="0"/>\n'
                      '    <geom name="table" type="plane" size="0.6 0.6 0.01" rgba="0 0 0 0"')
    ns = {"__file__": scene.__file__, "__name__": "scene_live"}
    exec(compile(txt, "scene_live", "exec"), ns)
    return ns["build"]()


class Sim:
    def __init__(self):
        self.m, self.pts, self.tris = build_model()
        self.d = mujoco.MjData(self.m)
        self.ik = mujoco.MjData(self.m)
        self.keys = scene.key_vertices(self.pts)
        m = self.m
        self.act = {a: [m.actuator(f"{a}_{j}").id for j in JN] for a in ARMS}
        self.qadr = {a: [m.jnt_qposadr[m.joint(f"{a}_{j}").id] for j in JN] for a in ARMS}
        self.site = {a: m.site(f"{a}_gripperframe").id for a in ARMS}
        self.gbody = {a: m.body(f"{a}_gripper").id for a in ARMS}
        self.vbody = m.flex_vertbodyid.copy()
        self.vqadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.vbody])
        self.vdadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.vbody])
        self.v0 = np.array([m.body_pos[b] for b in self.vbody])
        self.held = {a: None for a in ARMS}  # (vertex ids, offsets)
        self.target = {}
        self.saved = {}
        self.reset()

    # ---------- state ----------
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.held = {a: None for a in ARMS}
        self.frozen, self.free_since = None, 0.0
        for a in ARMS:
            q = [0.0, -1.2, 1.0, 1.3, 0.0, OPEN]  # parked, folded up
            for adr, act, v in zip(self.qadr[a], self.act[a], q):
                self.d.qpos[adr] = v
                self.d.ctrl[act] = v
        mujoco.mj_forward(self.m, self.d)
        self.area0 = None
        self.area0 = self.score()["area_cm2"]

    def tip(self, a, d=None):
        return (d or self.d).site_xpos[self.site[a]].copy()

    def verts(self):
        return self.d.flexvert_xpos.copy()

    # ---------- IK: tip position + gripper pointing straight down ----------
    def solve_ik(self, a, goal, q_init, roll=0.0, iters=60):
        m, ik = self.m, self.ik
        ik.qpos[:] = self.d.qpos
        idx = self.qadr[a][:4]
        q = np.array(q_init[:4], float)
        lo = np.array([m.jnt_range[m.joint(f"{a}_{j}").id][0] for j in JN[:4]])
        hi = np.array([m.jnt_range[m.joint(f"{a}_{j}").id][1] for j in JN[:4]])

        def resid(qq):
            ik.qpos[idx] = qq
            ik.qpos[self.qadr[a][4]] = roll
            mujoco.mj_kinematics(m, ik)
            p = ik.site_xpos[self.site[a]]
            g = ik.xpos[self.gbody[a]]
            ax = (p - g) / (np.linalg.norm(p - g) + 1e-9)
            return np.concatenate([p - goal, 0.03 * (ax - np.array([0, 0, -1.0]))])

        for _ in range(iters):
            r = resid(q)
            if np.linalg.norm(r[:3]) < 5e-4 and np.linalg.norm(r[3:]) < 3e-3:
                break
            J = np.zeros((6, 4))
            for k in range(4):
                dq = np.zeros(4); dq[k] = 1e-4
                J[:, k] = (resid(q + dq) - r) / 1e-4
            lam = 1e-4
            step = -np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
            q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
        r = resid(q)
        return q, float(np.linalg.norm(r[:3])), float(np.degrees(np.arccos(np.clip(1 - 0.5 * (np.linalg.norm(r[3:]) / 0.03) ** 2, -1, 1))))

    # ---------- grasp pinning ----------
    def grip(self, a, close, radius=GRAB_R, zbelow=0.03):
        """zbelow: how far under the jaw tip cloth is still caught (small = top layer only)."""
        if close:
            t = self.tip(a)
            v = self.verts()
            dist = np.linalg.norm(v[:, :2] - t[:2], axis=1)
            ids = np.where((dist < radius) & (v[:, 2] < t[2] + 0.02) & (v[:, 2] > t[2] - zbelow))[0]
            other = [i for b in ARMS if b != a and self.held[b] for i in self.held[b][0]]
            ids = np.array([i for i in ids if i not in other], int)
            self.held[a] = (ids, v[ids] - t) if len(ids) else None
            self.d.ctrl[self.act[a][5]] = CLOSED
            return int(len(ids))
        self.held[a] = None
        self.d.ctrl[self.act[a][5]] = OPEN
        return 0

    def stick(self, settle=1.5):
        """MuJoCo's soft contacts make resting cloth creep ~1 cm/min across the table.
        Once nothing has held the cloth for `settle` seconds, freeze it exactly where it lies."""
        if any(self.held[a] is not None for a in ARMS):
            self.frozen = None
            self.free_since = self.d.time
            return
        if self.d.time - getattr(self, "free_since", 0.0) < settle:
            return
        qa = (self.vqadr[:, None] + np.arange(3)).ravel()
        da = (self.vdadr[:, None] + np.arange(3)).ravel()
        if getattr(self, "frozen", None) is None:
            self.frozen = self.d.qpos[qa].copy()
        self.d.qpos[qa] = self.frozen
        self.d.qvel[da] = 0.0

    def apply_pins(self):
        self.stick()
        for a in ARMS:
            h = self.held[a]
            if h is None:
                continue
            ids, off = h
            t = self.tip(a)
            for i, o in zip(ids, off):
                b = self.vbody[i]
                self.d.qpos[self.vqadr[i]:self.vqadr[i] + 3] = t + o - self.v0[i]
                self.d.qvel[self.vdadr[i]:self.vdadr[i] + 3] = 0.0

    # ---------- area / squareness score ----------
    def score(self):
        v = self.verts()
        res = 0.002
        lo = v[:, :2].min(0) - 0.01
        hi = v[:, :2].max(0) + 0.01
        nx, ny = np.ceil((hi - lo) / res).astype(int)
        gx = lo[0] + (np.arange(nx) + 0.5) * res
        gy = lo[1] + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(gx, gy)
        P = np.stack([X.ravel(), Y.ravel()], 1)
        occ = np.zeros(len(P), bool)
        for tri in self.tris:
            A, B, C = v[tri, :2]
            mn, mx = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
            sel = np.where((P[:, 0] >= mn[0]) & (P[:, 0] <= mx[0]) & (P[:, 1] >= mn[1]) & (P[:, 1] <= mx[1]))[0]
            if not len(sel):
                continue
            Q = P[sel]
            def cr(p, q, r):
                return (q[0] - p[0]) * (r[:, 1] - p[1]) - (q[1] - p[1]) * (r[:, 0] - p[0])
            d1, d2, d3 = cr(A, B, Q), cr(B, C, Q), cr(C, A, Q)
            inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
            occ[sel[inside]] = True
        area = occ.sum() * res * res
        pts = P[occ]
        best = None
        for ang in np.radians(np.arange(0, 90, 1.0)):
            c, s = np.cos(ang), np.sin(ang)
            R = pts @ np.array([[c, -s], [s, c]])
            ext = R.max(0) - R.min(0) + res
            ar = ext[0] * ext[1]
            if best is None or ar < best[0]:
                best = (ar, ext, np.degrees(ang))
        rect_area, ext, ang = best
        out = dict(
            area_cm2=round(area * 1e4, 1),
            rect_cm=[round(float(max(ext)) * 100, 1), round(float(min(ext)) * 100, 1)],
            rectangularity=round(float(area / rect_area), 3),
            squareness=round(float(min(ext) / max(ext)), 3),
            height_mm=round(float(v[:, 2].max() - v[:, 2].min()) * 1000, 1),
        )
        if self.area0:
            out["area_vs_flat"] = round(area * 1e4 / self.area0, 3)
        self._occ = (occ.reshape(ny, nx), lo, res)
        return out

    def ascii_map(self):
        occ, lo, res = self._occ
        k = 5  # 1 cm per char
        ny, nx = occ.shape
        rows = []
        for j in range(ny // k - 1, -1, -1):  # far side (collar) on top
            rows.append("".join("#" if occ[j * k:(j + 1) * k, i * k:(i + 1) * k].mean() > 0.4 else "." for i in range(nx // k)))
        return {"x_from_cm": round(lo[0] * 100, 1), "y_from_cm": round(lo[1] * 100, 1), "rows_top_is_far": rows}

    def obs(self, with_map=True):
        v = self.verts()
        o = {
            "landmarks_cm": {k: [round(c * 100, 1) for c in v[i]] for k, i in self.keys.items()},
            "tips_cm": {a: [round(c * 100, 1) for c in self.tip(a)] for a in ARMS},
            "holding": {a: (0 if self.held[a] is None else len(self.held[a][0])) for a in ARMS},
            "score": self.score(),
            "t": round(self.d.time, 2),
        }
        if with_map:
            o["map"] = self.ascii_map()
        return o


def server_thread(cmdq):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 7788))
    srv.listen(1)
    while True:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            try:
                cmd = json.loads(buf.decode())
            except Exception as e:
                conn.sendall((json.dumps({"error": str(e)}) + "\n").encode()); continue
            done = queue.Queue()
            cmdq.put((cmd, done))
            conn.sendall((json.dumps(done.get()) + "\n").encode())


def main():
    sim = Sim()
    m, d = sim.m, sim.d
    cmdq = queue.Queue()
    threading.Thread(target=server_thread, args=(cmdq,), daemon=True).start()
    job = None  # active motion: dict(plan=[(arm, [q waypoints])], i, done, reply)

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0, 0.17, 0.0]
        viewer.cam.distance = 0.85
        viewer.cam.elevation = -55
        viewer.cam.azimuth = 90
        last_sync = 0
        print("READY on 127.0.0.1:7788", flush=True)
        while viewer.is_running():
            if job is None and not cmdq.empty():
                cmd, reply = cmdq.get()
                try:
                    job = handle(sim, cmd, reply)
                except Exception as e:
                    reply.put({"error": repr(e)})
                    job = None
            with viewer.lock():
                for _ in range(5):
                    if job is not None and job["kind"] == "traj":
                        i = job["i"]
                        for a, qs in job["q"].items():
                            qq = qs[min(i, len(qs) - 1)]
                            for k in range(4):
                                d.ctrl[sim.act[a][k]] = qq[k]
                            d.ctrl[sim.act[a][4]] = job["roll"][a]
                        job["i"] += 1
                    sim.apply_pins()
                    mujoco.mj_step(m, d)
                    sim.apply_pins()
                    if job is not None:
                        job["steps_left"] -= 1
                        if job["steps_left"] <= 0:
                            res = {"ok": True}
                            res.update(job.get("info", {}))
                            res["obs"] = sim.obs(with_map=job.get("map", False))
                            job["reply"].put(res)
                            job = None
                            break
            if time.time() - last_sync > 1 / 60:
                viewer.sync()
                last_sync = time.time()


def handle(sim, cmd, reply):
    op = cmd.get("op")
    d = sim.d
    if op == "obs":
        reply.put(sim.obs(with_map=cmd.get("map", True))); return None
    if op == "reset":
        sim.reset(); reply.put({"ok": True, "obs": sim.obs()}); return None
    if op == "save":
        st = np.empty(mujoco.mj_stateSize(sim.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(sim.m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        sim.saved[cmd.get("name", "s")] = (st, dict(sim.held))
        reply.put({"ok": True, "saved": cmd.get("name", "s")}); return None
    if op == "restore":
        st, held = sim.saved[cmd.get("name", "s")]
        mujoco.mj_setState(sim.m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        sim.held = dict(held)
        mujoco.mj_forward(sim.m, d)
        reply.put({"ok": True, "obs": sim.obs()}); return None
    if op == "grip":
        info = {}
        for a in cmd["arms"]:
            info[a] = sim.grip(a, cmd["state"] == "close", cmd.get("radius_cm", GRAB_R * 100) / 100)
        return dict(kind="wait", steps_left=int(0.4 / sim.m.opt.timestep), reply=reply, info={"grabbed_vertices": info})
    if op == "wait":
        return dict(kind="wait", steps_left=int(cmd.get("t", 1.0) / sim.m.opt.timestep), reply=reply, map=cmd.get("map", False))
    if op == "move":
        # cmd["to"] = {"L": [x,y,z] in cm, "R": [...]}; straight-line tip path at speed cm/s
        speed = cmd.get("speed", 5.0) / 100
        q_paths, info, roll = {}, {}, {}
        n_max = 1
        for a, goal_cm in cmd["to"].items():
            goal = np.array(goal_cm, float) / 100
            start = sim.tip(a)
            dist = np.linalg.norm(goal - start)
            n = max(2, int(dist / speed / sim.m.opt.timestep))
            n_max = max(n_max, n)
            q_paths[a] = (start, goal, n)
            roll[a] = float(cmd.get("roll", {}).get(a, 0.0))
        qs = {}
        for a, (start, goal, n) in q_paths.items():
            q = np.array([d.qpos[k] for k in sim.qadr[a][:4]])
            path = []
            n_way = max(2, int(np.linalg.norm(goal - start) / 0.004))  # IK every 4 mm
            way = []
            for s in np.linspace(0, 1, n_way):
                q, perr, aerr = sim.solve_ik(a, start + s * (goal - start), q, roll[a])
                way.append(q.copy())
            info[a] = {"ik_pos_err_mm": round(perr * 1000, 1), "tilt_from_vertical_deg": round(aerr, 1)}
            way = np.array(way)
            ti = np.linspace(0, len(way) - 1, n_max)
            qs[a] = [way[int(np.floor(t))] + (t - np.floor(t)) * (way[min(int(np.floor(t)) + 1, len(way) - 1)] - way[int(np.floor(t))]) for t in ti]
        settle = int(0.3 / sim.m.opt.timestep)
        return dict(kind="traj", q=qs, roll=roll, i=0, steps_left=n_max + settle, reply=reply, info={"ik": info})
    reply.put({"error": f"unknown op {op}"}); return None


if __name__ == "__main__":
    main()
