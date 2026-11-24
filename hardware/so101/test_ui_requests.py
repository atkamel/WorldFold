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
    def latest(self):
        return None

    def status(self):
        return "no phone"


class NoRecording:
    def status(self):
        return {"recording": False}


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

        def toggle_claw(self, name):
            if not self.driving:
                return False
            self.dragged.append((name, "claw"))
            return True

        def sticks_state(self):
            return {"left": [0.0, 0.0], "right": [0.0, 0.0], "raw": {"left": [0, 0], "right": [0, 0]}}

    gamepad = Gamepad() if getattr(request, "param", False) else None
    handler = ui.make_handler(arms_by_name, NoCamera(), NoRecording(), stop_fold=lambda: fold_stops.append(1),
                              gamepad=gamepad)
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
    yield arms_by_name, post, port
    server.shutdown()
    server.server_close()
    sys.modules.pop("ui", None)


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
    for path in ("/goal", "/pose", "/idle", "/mirror", "/leader", "/fold/start"):
        assert post(path, {"arm": "arm1", "pose": {"1": 3000}}) == 409
    assert arm.target == before
    assert post("/hold", {}) == 200
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5) as response:
        assert json.load(response)["gamepad"] == "Stopped"


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
def test_a_claw_key_reaches_the_gamepad_for_a_known_arm_while_driving(panel):
    _, post, _ = panel
    assert post("/gamepad/claw", {"arm": "arm1"}) == 200
    assert post.gamepad.dragged[-1] == ("arm1", "claw")
    assert post("/gamepad/claw", {"arm": "nobody"}) == 400
    post.gamepad.driving = False
    assert post("/gamepad/claw", {"arm": "arm1"}) == 409


def test_a_claw_key_is_not_taken_outside_gamepad_mode(panel):
    _, post, _ = panel
    assert post("/gamepad/claw", {"arm": "arm1"}) == 404
