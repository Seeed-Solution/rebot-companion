"""parse_detections / annotate, mirroring the cockpit's parseDetections (rebot-armd index.html)."""

from __future__ import annotations

import io

import pytest
from http_mock import MockTransport, Request, Response

from rebot_companion.snapshot import (
    Detection,
    DetectionClient,
    DetectorUnavailable,
    Snapshot,
    annotate,
    parse_detections,
)


def test_sscma_array_is_centre_normalised():
    (d,) = parse_detections([[0.5, 0.5, 0.2, 0.4, 0.9, "cup", 7]])
    assert d.label == "cup" and d.score == pytest.approx(0.9) and d.track_id == 7
    assert d.box_norm == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert d.center_norm == pytest.approx((0.5, 0.5))


def test_single_array_and_score_clamp():
    (d,) = parse_detections([0.5, 0.5, 0.2, 0.2, 1.7])
    assert d.score == 1.0 and d.label == "obj"


def test_xyxy_pixels_with_payload_resolution():
    raw = {"width": 640, "height": 480, "boxes": [{"format": "xyxy", "x1": 64, "x2": 192, "y1": 48, "y2": 240}]}
    (d,) = parse_detections(raw)
    assert d.box_norm == pytest.approx((0.1, 0.1, 0.3, 0.5))
    assert d.score == 0.0  # no score field


def test_pixels_without_resolution_are_dropped_unless_ref_given():
    raw = {"detections": [{"format": "xywh_center", "cx": 160, "cy": 120, "w": 32, "h": 24, "score": 0.5}]}
    assert parse_detections(raw) == []
    (d,) = parse_detections(raw, (320, 240))
    assert d.box_norm == pytest.approx((0.45, 0.45, 0.55, 0.55))


def test_xywh_center_normalised_and_envelope_result():
    raw = {"result": {"detections": [{"format": "xywh_center", "x": 0.25, "y": 0.5, "w": 0.1, "h": 0.2, "score": 0.8}]}}
    (d,) = parse_detections(raw)
    assert d.box_norm == pytest.approx((0.2, 0.4, 0.3, 0.6))


def test_top_left_bbox_and_zero_width_dropped():
    raw = {
        "data": {
            "detections": [
                {"format": "xywh_top_left", "bbox": [0.1, 0.2, 0.3, 0.4], "score": 0.6, "name": "box"},
                {"format": "xywh_top_left", "x": 0.1, "y": 0.2, "w": 0, "h": 0.4, "score": 0.6},
                {"format": "unknown", "x": 0.1},
            ]
        }
    }
    (d,) = parse_detections(raw)
    assert d.label == "box" and d.box_norm == pytest.approx((0.1, 0.2, 0.4, 0.6))


def test_markers_with_corners():
    raw = {
        "frame_id": 3,
        "detections": [],
        "markers": [{"id": 4, "corners": [[100, 100], [140, 100], [140, 140], [100, 140]]}],
        "resolution": [400, 200],
    }
    (d,) = parse_detections(raw)
    assert d.label == "aruco" and d.track_id == 4 and d.score == 1.0
    assert d.box_norm == pytest.approx((0.25, 0.5, 0.35, 0.7))
    assert d.corners_norm[0] == pytest.approx((0.25, 0.5))


def test_box_clamped_to_frame():
    (d,) = parse_detections([[0.95, 0.5, 0.4, 0.2, 0.5, "edge"]])
    assert d.box_norm == pytest.approx((0.75, 0.4, 1.0, 0.6))  # cx=0.95 stays normalised


def test_annotate_draws_in_pixel_coordinates():
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (200, 100), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    # PNG-backed snapshot keeps the test exact (JPEG would blur the 1-px probes)
    snap = Snapshot(jpeg=buf.getvalue(), width=200, height=100, device_ts_ms=1, host_rx_ms=1)
    det = Detection("cup", 0.9, (0.25, 0.5, 0.75, 0.9))
    out = annotate(snap, [det, Detection("low", 0.1, (0.0, 0.0, 0.1, 0.1))], min_score=0.5)
    res = pil.open(io.BytesIO(out.jpeg)).convert("RGB")
    assert (out.width, out.height) == (200, 100)
    left_edge = res.getpixel((round(0.25 * 199), 70))
    assert left_edge[1] > 120  # green outline at x0
    inside = res.getpixel((100, 75))
    assert max(inside) < 40  # not filled
    corner = res.getpixel((2, 2))
    assert max(corner) < 40  # low-score box skipped


def _proxy(routes):
    calls = []

    def handler(req: Request) -> Response:
        calls.append((req.method, req.url.path, dict(req.url.params)))
        fn = routes.get((req.method, req.url.path))
        return fn(req) if fn else Response(404)

    return calls, MockTransport(handler)


def test_detect_once_uses_trigger():
    calls, t = _proxy(
        {
            ("POST", "/api/v1/detection/trigger"): lambda r: Response(
                200, json={"success": True, "frame_id": 9, "ts_ms": 5, "count": 0, "markers": []}
            )
        }
    )
    res = DetectionClient("http://armd", transport=t).detect_once(wait_ms=1000)
    assert res["frame_id"] == 9 and not res["timeout"]
    # a provider without /api/v1/vision/mode (404) gets the trigger without pipelines
    assert calls == [("GET", "/api/v1/vision/mode", {}), ("POST", "/api/v1/detection/trigger", {"wait_ms": "1000"})]


def test_detect_once_falls_back_and_disables_only_if_enabled_here():
    state = {"enabled": False}

    def set_inf(r):
        import json

        state["enabled"] = json.loads(r.content)["enabled"]
        return Response(200, json=state)

    calls, t = _proxy(
        {
            ("GET", "/api/v1/detection/latest"): lambda r: Response(
                200, json={"enabled": state["enabled"], "frame_id": 11 if r.url.params.get("after") else 10}
            ),
            ("GET", "/api/v1/detection/inference"): lambda r: Response(200, json=state),
            ("POST", "/api/v1/detection/inference"): set_inf,
        }
    )
    client = DetectionClient("http://armd", transport=t)
    res = client.detect_once(wait_ms=500)
    assert res["frame_id"] == 11 and client.trigger_supported is False
    assert state["enabled"] is False
    assert ("GET", "/api/v1/detection/latest", {"wait_ms": "500", "after": "10"}) in calls
    # already enabled by someone else: left enabled
    state["enabled"] = True
    client.detect_once(wait_ms=0)
    assert state["enabled"] is True


def test_detector_502_maps_to_unavailable():
    _, t = _proxy({("GET", "/api/v1/detection/latest"): lambda r: Response(502, json={"error": "x"})})
    with pytest.raises(DetectorUnavailable):
        DetectionClient("http://armd", transport=t).latest()


def _v2_routes(seen_bodies, per_request=True):
    import json

    def trigger(r):
        body = json.loads(r.content) if r.content else None
        seen_bodies.append(body)
        p = (body or {}).get("pipelines", ["yolo"])
        ran = {k: {"ran": k in p} for k in ("yolo", "aruco")}
        return Response(200, json={"api_version": 2, "success": True, "frame_id": 12, "ts_ms": 7, "pipelines": ran})

    mode = {"api_version": 2 if per_request else 1, "mode": "off"}
    if per_request:
        mode["per_request_pipelines"] = True
    return {
        ("GET", "/api/v1/vision/mode"): lambda r: Response(200, json=mode),
        ("POST", "/api/v1/detection/trigger"): trigger,
    }


def test_detect_once_sends_pipelines_to_api2_providers():
    bodies = []
    calls, t = _proxy(_v2_routes(bodies))
    client = DetectionClient("http://armd", transport=t)
    res = client.detect_once(wait_ms=800)
    assert bodies[-1] == {"pipelines": ["yolo", "aruco"]} and res["pipelines"] == ["yolo", "aruco"]
    res = client.detect_once(wait_ms=800, pipelines=["aruco"])
    assert bodies[-1] == {"pipelines": ["aruco"]} and res["pipelines"] == ["aruco"]
    client.detect_once(pipelines=("yolo", "yolo"))
    assert bodies[-1] == {"pipelines": ["yolo"]}
    # the mode is read once and never written
    assert [c for c in calls if c[1] == "/api/v1/vision/mode"] == [("GET", "/api/v1/vision/mode", {})]
    with pytest.raises(ValueError):
        client.detect_once(pipelines=["video"])
    with pytest.raises(ValueError):
        client.detect_once(pipelines=[])


def test_detect_once_keeps_the_old_trigger_for_api1_and_older_providers():
    for routes in (_v2_routes([], per_request=False), {}):
        bodies = []

        def trigger(req, bodies=bodies):
            bodies.append(req.content)
            return Response(200, json={"success": True, "frame_id": 3, "ts_ms": 1})

        _, t = _proxy({**routes, ("POST", "/api/v1/detection/trigger"): trigger})
        client = DetectionClient("http://armd", transport=t)
        res = client.detect_once(wait_ms=100, pipelines=["aruco"])
        assert bodies == [b""] and client.pipelines_supported is False
        assert res["frame_id"] == 3 and res["pipelines"] is None
