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
def panel(tmp_path, monkeypatch):
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
    handler = ui.make_handler(arms_by_name, NoCamera(), NoRecording())
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
