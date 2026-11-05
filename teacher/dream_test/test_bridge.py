"""Offline tests of bridge.py, with Dream's real client and IK against a fake gateway (no inference is billed; the
client's one model lookup goes to the real platform, which is free).

The fake gateway answers each chunk with the hand poses the client sent (each arm's tool pose from Dream's FK of our
joints) moved 5 cm forward and 3 cm up over the 48 rows, grip closing to 0. Checked against OUR kinematics
(oracle/so101_kin.py, the URDF the LeHome arm was matched to): the joints the bridge returns must move each jaw tip
those 5 cm / 3 cm in that arm's base frame. That pins the joint order, units, sides and frames end to end.

    <python 3.11-3.13 with dreamlabs> test_bridge.py
"""
import http.server
import json
import os
import pathlib
import sys
import threading

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
import bridge  # noqa: E402

SEEN = {}


class FakeGateway(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        obs = body.get("observation", {})
        SEEN["last"] = body
        if self.path.endswith("end_session"):
            out = {"status": {"code": "DA-0000"}}
        else:
            per = {}
            for side in ("left", "right"):
                xyz = np.asarray(obs[f"state.{side}_eef_xyz"], float)
                rot6 = np.asarray(obs[f"state.{side}_eef_rot6"], float)
                s = np.linspace(0, 1, 48)[:, None]
                per[f"action.{side}_eef_xyz"] = (xyz + s * np.array([0.05, 0.0, 0.03])).tolist()
                per[f"action.{side}_eef_rot6"] = np.repeat(rot6[None], 48, 0).tolist()
                per[f"action.{side}_grip"] = np.linspace(1.0, 0.0, 48).tolist()
                per[f"action.{side}_arm"] = None
            new = body.get("session_id") is None
            out = {"status": {"code": "DA-0001" if new else "DA-0000", "reason": "new"} if new else {"code": "DA-0000"},
                   "session_id": "fake-1", "actions": {"per_key": per}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def check(name, ok, detail=""):
    print(("ok  " if ok else "FAIL") + f"  {name}  {detail}")
    if not ok:
        check.failed = True


check.failed = False


def unit_tests():
    for q in (-0.1745, 0.0, 0.55, 1.745):
        check(f"grip round trip {q}", abs(bridge.grip_to_rad(bridge.rad_to_grip(q)) - q) < 1e-6)
    c = bridge.RowClock(1 / 90)
    n = [c.next() for _ in range(20)]
    check("20 rows = 90 physics steps at dt 1/90 (1 s)", sum(n) == 90 and set(n) == {4, 5}, str(n[:6]))
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    f = bridge.to_dream_frame(img)
    check("frame 640x480 -> crop 640x360 -> 320x176", f.shape == (176, 320, 3), str(f.shape))
    ql = np.c_[np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (48, 1)), np.zeros(48)]
    qr = np.c_[np.tile([-0.1, -0.2, -0.3, -0.4, -0.5], (48, 1)), np.ones(48)]
    a = bridge.rows_to_actions(ql, qr)
    check("rows -> [left5, grip, right5, grip]", a.shape == (48, 12) and np.allclose(a[0, :5], ql[0, :5])
          and np.allclose(a[0, 6:11], qr[0, :5]) and abs(a[0, 5] - bridge.GRIP_LO) < 1e-9
          and abs(a[0, 11] - bridge.GRIP_HI) < 1e-9)


def client_test():
    import dream
    import so101_kin
    srv = http.server.HTTPServer(("127.0.0.1", 0), FakeGateway)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["DREAM_GATEWAY_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    kin = so101_kin.SO101()
    # a LeHome-like working pose: both hands in front of their bases, jaws open
    s12 = np.array([0.25, -0.6, 0.9, 0.9, 0.0, 0.55, -0.25, -0.6, 0.9, 0.9, 0.0, 0.55])
    imgs = {k: np.full((480, 640, 3), 120, np.uint8) for k in bridge.CAMERAS}
    with dream.DreamClient() as client:
        client.policy(bridge.make_config())
        br = bridge.DreamBridge(client)
        br.begin("pick up the left sleeve")
        state, t, rows = s12.copy(), 0.0, []
        for i in range(48):
            br.observe(imgs, state, t)
            a = br.next_action()
            rows.append(a)
            state, t = a.copy(), t + 0.05          # the arm follows its command exactly
        sent = SEEN["last"]["observation"]
        check("model gets no joint state (YAM slots zero)", np.allclose(sent["state.left_arm"], 0)
              and len(sent["state.left_arm"]) == 6, str(sent["state.left_arm"]))
        check("robot sent as other_embodiment", sent.get("embodiment") == "other_embodiment", str(sent.get("embodiment")))
        check("three camera windows", all(isinstance(sent.get(f"video.{r}"), list) and sent[f"video.{r}"]
                                          for r in ("overhead", "wrist_l", "wrist_r")),
              str({k: len(v) for k, v in sent.items() if k.startswith("video.")}))
        check("prompt passed", sent.get("prompt") == "pick up the left sleeve")
        rows = np.array(rows)
        for side, j in (("left", 0), ("right", 6)):
            p0 = kin.fk_base(s12[j:j + 5])[:3, 3]
            p1 = kin.fk_base(rows[-1, j:j + 5])[:3, 3]
            d = p1 - p0
            check(f"{side} jaw tip moved +5 cm forward / +3 cm up in its base frame",
                  abs(d[0] - 0.05) < 0.012 and abs(d[1]) < 0.012 and abs(d[2] - 0.03) < 0.012, str(np.round(d, 4)))
        check("first row starts at the measured pose", np.abs(rows[0, :5] - s12[:5]).max() < 0.05,
              str(np.round(rows[0, :5] - s12[:5], 3)))
        check("grip closes over the chunk", rows[0, 5] > rows[-1, 5] and abs(rows[-1, 5] - bridge.GRIP_LO) < 0.05,
              f"{rows[0, 5]:.2f} -> {rows[-1, 5]:.2f}")
        # second chunk: the client must skip no rows (commanded = last row played)
        br.observe(imgs, state, t)
        a = br.next_action()
        check("second chunk continues from the last row", np.abs(a[:5] - rows[-1, :5]).max() < 0.05,
              str(np.round(a[:5] - rows[-1, :5], 3)))
        print("calls:", br.calls)
    srv.shutdown()


if __name__ == "__main__":
    unit_tests()
    client_test()
    print("FAILED" if check.failed else "ALL BRIDGE TESTS PASSED")
    sys.exit(1 if check.failed else 0)
