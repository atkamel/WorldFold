"""The panel's HTTP handler on a fake servo bus: which requests it refuses."""

import importlib
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_ui_lend  # noqa: E402


class NoCamera:
    frame = None

    def latest(self):
        return self.frame

    def count(self):
        if self.frame is None:
            return 0
        return 3

    def status(self):
        return "no phone"


class NoRecording:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"recording": False}

    def start(self, correction=False):
        self.calls.append(("start", correction))

    def label(self, word):
        self.calls.append(("label", word))
        return word != "bad"


@pytest.fixture
def panel(tmp_path, monkeypatch, request):
    ports = {}
    monkeypatch.setitem(sys.modules, "scservo_sdk", test_ui_lend.make_fake_sdk(ports))
    sys.modules.pop("ui", None)
    ui = importlib.import_module("ui")
    monkeypatch.setattr(ui.arms, "folder", lambda name: tmp_path / name)
    monkeypatch.setattr(ui.arms, "port", lambda name: "dev" + name)
    limits = {str(m): {"true": {"min": 685, "max": 3410}} for m in range(1, 7)}
    arms_by_name = {}
    for name in ("arm1", "arm2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "limits.json").write_text(json.dumps(limits))
        arms_by_name[name] = ui.Arm(name)
    fold_stops = []
    class Gamepad:
        status = "Ready"
        driving = True
        dragged = []

        def stop(self):
            self.status = "Stopped"

        def set_tip(self, name, point):
            if not self.driving:
                return False
            self.dragged.append((name, point))
            return True

        def tips_state(self):
            return {"arm1": None, "arm2": None}

        def press_drive(self):
            self.dragged.append("drive")

        def press_home(self):
            self.dragged.append("home")

        def sticks_state(self):
            return {"left": [0.0, 0.0], "right": [0.0, 0.0], "raw": {"left": [0, 0], "right": [0, 0]}}

    class Fold:
        started = []

        def start(self, mode, options):
            self.started.append(mode)
            return None

        def stop(self):
            self.started.append("stop")

        def status(self):
            return {"state": "stopped"}

    gamepad = Gamepad() if getattr(request, "param", False) else None
    fold = Fold()
    camera = NoCamera()
    recording = NoRecording()
    handler = ui.make_handler(arms_by_name, camera, recording, fold=fold,
                              stop_fold=lambda: fold_stops.append(1), gamepad=gamepad)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def post(path, body, headers=None):
        """The HTTP status of a POST to the panel."""
        sent = {"Content-Type": "application/json"}
        if headers is not None:
            sent = headers
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                         headers=sent, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as answer:
                return answer.status
        except urllib.error.HTTPError as err:
            return err.code

    def get(path):
        """(HTTP status, body) of a GET to the panel."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as answer:
                return answer.status, answer.read()
        except urllib.error.HTTPError as err:
            return err.code, b""

    post.get = get
    post.gamepad = gamepad
    post.fold_stops = fold_stops
    post.fold = fold
    post.camera = camera
    post.recording = recording
    yield arms_by_name, post, port
    server.shutdown()
    server.server_close()
    sys.modules.pop("ui", None)


def test_an_episode_label_reaches_the_recording_and_a_refusal_is_reported(panel):
    _, post, _ = panel
    assert post("/record/label", {"label": "good"}) == 200
    assert post.recording.calls == [("label", "good")]
    assert post("/record/label", {"label": "bad"}) == 409      # the stand-in refuses this one
    assert post("/record/label", {"label": "perfect"}) == 400
    assert len(post.recording.calls) == 2


def test_taking_over_stops_the_policy_and_starts_a_correction_recording(panel):
    _, post, _ = panel
    assert post("/policy/takeover", {}) == 200
    assert post.fold.started == ["stop"]
    assert post.recording.calls == [("start", True)]


def test_the_panel_hands_its_camera_frame_to_a_script_that_asks(panel):
    import numpy as np

    import panel_phone
    import rgbd

    _, post, port = panel
    phone = panel_phone.PanelPhone(f"http://127.0.0.1:{port}/frame.npz")
    assert phone.latest() is None
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.full((2, 3), 0.5, dtype=np.float32)
    post.camera.frame = rgbd.Frame(0.0, rgb, depth, (100.0, 101.0, 3.0, 2.0))
    frame = phone.latest()
    assert phone.count == 3
    assert np.array_equal(frame.rgb, rgb)
    assert np.array_equal(frame.depth, depth)
    assert list(frame.K) == [100.0, 101.0, 3.0, 2.0]
    with pytest.raises(SystemExit):     # the count has not moved on: no new picture
        phone.wait(timeout=0.3)


def test_mirror_and_leader_are_refused_while_an_arm_is_lent(panel):
    arms_by_name, post, _ = panel
    assert post("/mirror", {"leader": "arm1"}) == 200
    assert arms_by_name["arm2"].follows is not None
    arms_by_name["arm2"].lend()
    assert arms_by_name["arm2"].follows is None
    assert post("/mirror", {"leader": "arm1"}) == 409
    assert post("/leader", {"on": True}) == 409
    assert arms_by_name["arm2"].follows is None


def test_an_arm_comes_back_following_nothing(panel):
    arms_by_name, _, _ = panel
    arm = arms_by_name["arm2"]
    arm.lend()
    arm.follows = (arms_by_name["arm1"], True)      # set behind the handler's back while lent
    arm.take_back()
    assert arm.follows is None


def test_a_post_that_another_web_page_could_send_is_refused(panel):
    arms_by_name, post, port = panel
    arm = arms_by_name["arm1"]
    before = dict(arm.target)
    as_a_form = {"Content-Type": "text/plain"}
    assert post("/goal", {"arm": "arm1", "pose": {"1": 3000}}, as_a_form) == 415
    from_elsewhere = {"Content-Type": "application/json", "Host": "evil.example:" + str(port)}
    assert post("/goal", {"arm": "arm1", "pose": {"1": 3000}}, from_elsewhere) == 403
    assert arm.target == before
    assert post("/goal", {"arm": "arm1", "pose": {"1": 3000}}) == 200
    assert arm.target[1] == 3000


def test_hold_stops_a_cloth_fold_too(panel):
    """The phone's page has no fold buttons, and its STOP sends /hold: a fold must not carry on behind it."""
    _, post, _ = panel
    assert post("/hold", {}) == 200
    assert post.fold_stops == [1]


@pytest.mark.parametrize("panel", [True], indirect=True)
def test_gamepad_panel_reserves_motion_and_routes_stop(panel):
    arms_by_name, post, port = panel
    arm = arms_by_name["arm1"]
    before = dict(arm.target)
    for path in ("/goal", "/pose", "/idle", "/mirror", "/leader"):
        assert post(path, {"arm": "arm1", "pose": {"1": 3000}}) == 409
    assert arm.target == before
    assert post("/hold", {}) == 200
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5) as response:
        assert json.load(response)["gamepad"] == "Stopped"


@pytest.mark.parametrize("panel", [True], indirect=True)
def test_a_cloth_fold_can_be_started_in_gamepad_mode(panel):
    _, post, _ = panel
    assert post("/fold/start", {"mode": "fold", "options": {}}) == 200
    assert post.fold.started == ["fold"]


def test_the_state_carries_each_arm_s_map_from_encoder_steps_to_model_angles(panel):
    _, post, _ = panel
    status, body = post.get("/state")
    assert status == 200
    model = json.loads(body)["arms"]["arm1"]["model"]
    assert sorted(model["zero"]) == ["1", "2", "3", "4", "5", "6"]
    for motor in model["sign"]:
        assert model["sign"][motor] in (1, -1)


def test_the_panel_serves_the_arm_description_and_only_its_mesh_files(panel):
    pytest.importorskip("so101_nexus")
    _, post, _ = panel
    status, body = post.get("/so101.urdf")
    assert status == 200
    assert b"<robot" in body
    status, body = post.get("/assets/base_so101_v2.stl")
    assert status == 200
    assert len(body) > 1000
    assert post.get("/assets/base_so101_v2.part")[0] == 404
    assert post.get("/assets/..%2Fso101_new_calib.urdf")[0] == 404
    assert post.get("/assets/../so101_new_calib.urdf")[0] == 404


@pytest.mark.parametrize("panel", [True], indirect=True)
def test_a_dragged_point_reaches_the_gamepad_only_as_three_plausible_numbers(panel):
    _, post, _ = panel
    assert post("/gamepad/target", {"arm": "arm1", "tip": [0.25, -0.05, 0.1]}) == 200
    assert post.gamepad.dragged[-1] == ("arm1", [0.25, -0.05, 0.1])
    sent = len(post.gamepad.dragged)
    for bad in ([0.25, 0.0], [0.25, 0.0, "up"], [0.25, 0.0, 5.0], [0.25, 0.0, True], "there", None):
        assert post("/gamepad/target", {"arm": "arm1", "tip": bad}) == 400
    assert post("/gamepad/target", {"arm": "nobody", "tip": [0.25, 0.0, 0.1]}) == 400
    assert len(post.gamepad.dragged) == sent
    post.gamepad.driving = False
    assert post("/gamepad/target", {"arm": "arm1", "tip": [0.25, 0.0, 0.1]}) == 409


def test_a_dragged_point_is_not_taken_outside_gamepad_mode(panel):
    _, post, _ = panel
    assert post("/gamepad/target", {"arm": "arm1", "tip": [0.25, 0.0, 0.1]}) == 404


@pytest.mark.parametrize("panel", [True], indirect=True)
def test_the_drive_and_idle_buttons_reach_the_gamepad(panel):
    _, post, _ = panel
    assert post("/gamepad/drive", {}) == 200
    assert post("/gamepad/home", {}) == 200
    assert post.gamepad.dragged[-2:] == ["drive", "home"]


def test_the_drive_and_idle_buttons_are_not_taken_outside_gamepad_mode(panel):
    _, post, _ = panel
    assert post("/gamepad/drive", {}) == 404
    assert post("/gamepad/home", {}) == 404
