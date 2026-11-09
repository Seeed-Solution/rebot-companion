import httpx
import pytest

from rebot_companion.camera_client import CameraError, CameraTriggerTimeout, MarkerDetection, ReCameraClient


@pytest.fixture
def patch_httpx(monkeypatch):
    def install(handler):
        transport = httpx.MockTransport(handler)
        real = httpx.Client
        monkeypatch.setattr(
            "rebot_companion.camera_client.httpx.Client", lambda *a, **kw: real(*a, transport=transport, **kw)
        )
        return ReCameraClient("http://cam.test")

    return install


MARKER = {"id": 3, "center": [1, 2], "corners": [], "pose": {"t": [0.1, 0.2, 0.5], "r_deg": [0, 0, 0], "error": 0.01}}


def test_trigger_returns_markers_with_pose_only(patch_httpx):
    cam = patch_httpx(
        lambda req: httpx.Response(200, json={"success": True, "markers": [MARKER, {"id": 4, "center": [0, 0]}]})
    )
    found = cam.trigger(wait_ms=100)
    assert [m.id for m in found] == [3]  # marker 4 has no pose and is skipped, not placed at the camera origin
    assert list(found[0].pose.translation) == pytest.approx([0.1, 0.2, 0.5])


def test_trigger_no_marker_is_empty_list(patch_httpx):
    cam = patch_httpx(lambda req: httpx.Response(200, json={"success": True, "markers": []}))
    assert cam.trigger(wait_ms=100) == []


def test_trigger_unsuccessful_raises_timeout(patch_httpx):
    cam = patch_httpx(lambda req: httpx.Response(200, json={"success": False, "error": "timeout"}))
    with pytest.raises(CameraTriggerTimeout, match="no result within 100 ms"):
        cam.trigger(wait_ms=100)


def test_trigger_http_timeout_raises_timeout(patch_httpx):
    def slow(req):
        raise httpx.ReadTimeout("read timed out", request=req)

    cam = patch_httpx(slow)
    with pytest.raises(CameraTriggerTimeout):
        cam.trigger(wait_ms=100)


def test_marker_without_pose_is_rejected():
    with pytest.raises(ValueError, match="no pose"):
        MarkerDetection.from_dict({"id": 1})


def test_trigger_posts_json_content_type(patch_httpx):
    seen = []

    def handler(req):
        seen.append(req)
        if req.headers.get("content-type") != "application/json":
            return httpx.Response(415)
        return httpx.Response(200, json={"success": True, "markers": []})

    cam = patch_httpx(handler)
    assert cam.trigger(wait_ms=100) == []
    assert seen[0].method == "POST" and seen[0].headers["content-type"] == "application/json"
    assert seen[0].content == b"{}"
    assert seen[0].url.params["wait_ms"] == "100"
    cam.set_inference(True)
    assert seen[1].headers["content-type"] == "application/json"


def test_trigger_415_is_a_clear_error(patch_httpx):
    cam = patch_httpx(lambda req: httpx.Response(415))
    with pytest.raises(CameraError, match="Content-Type: application/json"):
        cam.trigger(wait_ms=100)
