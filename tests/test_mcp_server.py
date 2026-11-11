"""MCP server over the in-memory client (mcp 2.x ``mcp.Client(server)``) against fake armd + camera."""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import pytest
from http_mock import MockTransport, Request, Response

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from test_snapshot import FakeCamera, load_messages  # noqa: E402
from websockets.sync.server import serve  # noqa: E402

from rebot_companion.mcp_server import Controller, build_server  # noqa: E402
from rebot_companion.snapshot import DetectionClient, VideoGrabber  # noqa: E402

TOOLS = {
    "get_state",
    "snapshot",
    "snapshot_annotated",
    "detect",
    "move_to",
    "move_relative",
    "gripper",
    "home",
    "park",
    "stop",
    "e_stop",
    "enable",
    "take_control",
    "release_control",
    "list_poses",
    "save_pose",
    "move_to_pose",
    "place",
    "get_keepout",
    "set_keepout",
    "recover",
    "collision_guard_status",
    "set_collision_guard",
    "describe_safety",
}


TRIGGER_BODIES: list = []


def detector_transport() -> MockTransport:
    def handler(req: Request) -> Response:
        if req.url.path == "/api/v1/vision/mode":
            return Response(200, json={"api_version": 2, "mode": "off", "per_request_pipelines": True})
        if req.url.path == "/api/v1/detection/trigger":
            TRIGGER_BODIES.append(json.loads(req.content) if req.content else None)
            return Response(
                200,
                json={
                    "success": True,
                    "frame_id": 5,
                    "ts_ms": 1_762_000_000_050,
                    "count": 1,
                    "detections": [[0.5, 0.5, 0.2, 0.2, 0.9, "cup"], [0.1, 0.1, 0.1, 0.1, 0.1, "noise"]],
                },
            )
        if req.url.path == "/api/v1/detection/latest":
            return Response(200, json={"enabled": False, "frame_id": 4, "ts_ms": 1_762_000_000_000, "detections": []})
        return Response(404)

    return MockTransport(handler)


@pytest.fixture
def camera_url():
    cam = FakeCamera(load_messages(), repeat=True)
    server = serve(cam.handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/"
    cam.stopped.set()
    server.shutdown()


@pytest.fixture
def make_ctrl(armd, camera_url):
    made = []

    def make(start=True, **kw):
        ctrl = Controller(
            armd.client,
            VideoGrabber(camera_url, backend="auto"),
            DetectionClient(armd.url, transport=detector_transport()),
            client_id="rebot-mcp",
            **kw,
        )
        if start:
            ctrl.start()
        made.append(ctrl)
        return ctrl

    yield make
    for c in made:
        c.shutdown()


def body(res) -> dict:
    texts = [c.text for c in res.content if c.type == "text"]
    return json.loads(texts[-1])


@pytest.mark.asyncio
async def test_list_tools(make_ctrl):
    async with Client(build_server(make_ctrl(start=False))) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == TOOLS
    mv = tools["move_to"].input_schema
    assert mv["required"] == ["x", "y", "z"]
    assert mv["properties"]["speed_mps"]["anyOf"][0]["maximum"] == 0.15
    rel = tools["move_relative"].input_schema["properties"]
    assert rel["frame"]["enum"] == ["base", "tool", "camera"] and rel["dx"]["anyOf"][0]["maximum"] == 0.3
    assert rel["dyaw_deg"]["anyOf"][0]["maximum"] == 90.0
    tc = tools["take_control"]
    assert "force" not in tc.input_schema.get("properties", {})  # no preemption unless --allow-preempt
    assert "Cannot take control from the web cockpit" in tc.description
    assert "a person clears it in the web cockpit" in tools["e_stop"].description
    assert tools["gripper"].input_schema["properties"]["command"]["enum"] == ["open", "close", "move"]


@pytest.mark.asyncio
async def test_get_state_shape(make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        res = await c.call_tool("get_state", {})
    assert not res.is_error
    s = body(res)
    assert s["safety_state"] == "Idle" and s["lease"] == {"held": True, "generation": 1, "status": "held"}
    assert set(s["joints"]) == {"positions", "commanded", "velocities", "torques", "temperatures"}
    assert len(s["joints"]["positions"]) == 7
    assert s["tcp"]["xyz_m"] == pytest.approx([0.260, 0.0, 0.192], abs=2e-3)  # home pose, host FK
    assert len(s["tcp"]["quat"]) == 4 and len(s["tcp"]["rpy_deg"]) == 3
    assert s["gripper"] == {"rad": 0.0, "fraction": 0.0}
    assert {"action", "feedback_valid", "epoch_id", "fault_reason", "is_torqued"} <= set(s)


@pytest.mark.asyncio
async def test_move_to_done_and_gripper(make_ctrl, armd):
    async with Client(build_server(make_ctrl())) as c:
        assert body(await c.call_tool("enable", {}))["is_torqued"] is True
        res = await c.call_tool("move_to", {"x": 0.25, "y": 0.0, "z": 0.25, "pitch_deg": 30, "speed_mps": 0.05})
        assert not res.is_error, res.content
        r = body(res)
        assert r["state"] == "done" and r["action_id"] == 1 and r["settled"] is True
        assert r["tcp_after"]["xyz_m"] and r["duration_s"] >= 0.2
        path, sent = armd.action_bodies[-1]
        assert path == "/api/v1/action/move_to" and sent["speed"] == {"linear": 0.05}
        rpy = sent["orientation"]["rpy"]
        assert rpy[1] == pytest.approx(0.5235988, abs=1e-6)  # pitch given; roll/yaw kept from the current pose
        g = body(await c.call_tool("gripper", {"command": "close"}))
        assert g["holding"] is True and g["state"] == "done" and g["gap_rad"] > 0.3
        rel = body(await c.call_tool("move_relative", {"frame": "camera", "dz": 0.05, "dpitch_deg": 10}))
        assert rel["state"] == "done"
        assert armd.action_bodies[-1][1]["dpitch"] == pytest.approx(0.1745329, abs=1e-6)


@pytest.mark.asyncio
async def test_lease_busy_never_forces_until_take_control(armd, make_ctrl):
    cockpit = armd.client.session(client_id="cockpit", safe_exit="off").open()
    try:
        ctrl = make_ctrl()
        async with Client(build_server(ctrl)) as c:
            res = await c.call_tool("move_to", {"x": 0.25, "y": 0.0, "z": 0.25})
            assert res.is_error
            err = body(res)
            assert err["code"] == "LEASE_BUSY" and "release it in the cockpit" in err["hint"]
            assert "force" not in err["hint"]
            assert err["safety_state"] == "Idle"
            assert body(await c.call_tool("get_state", {}))["lease"]["held"] is False
            assert body(await c.call_tool("take_control", {}))["code"] == "LEASE_BUSY"
            res = await c.call_tool("take_control", {"force": True})  # not part of the default schema
            assert res.is_error
            assert all(not force for cid, force in armd.acquires if cid == "rebot-mcp")
            assert cockpit.has_lease
    finally:
        cockpit.close()


@pytest.mark.asyncio
async def test_allow_preempt_flag_enables_forced_take_control(armd, make_ctrl):
    cockpit = armd.client.session(client_id="cockpit", safe_exit="off").open()
    try:
        async with Client(build_server(make_ctrl(), allow_preempt=True)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            tc = tools["take_control"]
            assert tc.input_schema["properties"]["force"]["default"] is False
            assert "only call after the user explicitly confirms" in tc.description
            err = body(await c.call_tool("move_to", {"x": 0.25, "y": 0.0, "z": 0.25}))
            assert err["code"] == "LEASE_BUSY" and "take_control(force=true)" in err["hint"]
            assert body(await c.call_tool("take_control", {}))["code"] == "LEASE_BUSY"
            assert all(not force for cid, force in armd.acquires if cid == "rebot-mcp")
            res = await c.call_tool("take_control", {"force": True})
            assert body(res)["held"] is True
            assert armd.acquires[-1] == ("rebot-mcp", True)
            await asyncio.sleep(0.1)
            assert not cockpit.has_lease
    finally:
        cockpit.close()


@pytest.mark.asyncio
async def test_lease_lost_after_preemption(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        with armd.client.session(client_id="cockpit", force=True, safe_exit="off"):
            await asyncio.sleep(0.1)
            err = body(await c.call_tool("home", {}))
            assert err["code"] == "LEASE_LOST" and err["hint"] == "control was taken by another client"
            assert body(await c.call_tool("get_state", {}))["lease"]["status"] == "lost"
        # no silent re-acquire after a loss, even once the cockpit has gone
        assert body(await c.call_tool("home", {}))["code"] == "LEASE_LOST"
        assert body(await c.call_tool("take_control", {}))["held"] is True


@pytest.mark.asyncio
async def test_e_stop_without_lease(armd, make_ctrl):
    cockpit = armd.client.session(client_id="cockpit", safe_exit="off").open()
    try:
        async with Client(build_server(make_ctrl())) as c:
            res = await c.call_tool("e_stop", {})
            assert not res.is_error and body(res)["safety_state"] == "EStop"
            err = body(await c.call_tool("enable", {}))
            assert err["code"] == "ESTOP_ACTIVE" and "clear the fault in the web cockpit" in err["hint"]
            assert "clear_fault" not in {t.name for t in (await c.list_tools()).tools}
    finally:
        cockpit.close()


@pytest.mark.asyncio
async def test_allow_clear_fault_flag_exposes_the_tool(armd, make_ctrl):
    ctrl = make_ctrl()
    async with Client(build_server(ctrl, allow_clear_fault=True)) as c:
        assert "clear_fault" in {t.name for t in (await c.list_tools()).tools}
        assert not (await c.call_tool("e_stop", {})).is_error
        # e_stop goes out on the WebSocket and over REST, and each path advances armd's epoch.
        # clear_fault carries the epoch of the latest observation, so it must not run before an
        # observation after both paths has arrived (otherwise armd answers EPOCH_MISMATCH, by
        # design: a stale decision never clears a newer stop).
        s = ctrl.actuator()
        await asyncio.to_thread(lambda: [s.wait_for_observation() for _ in range(3)])
        res = await c.call_tool("clear_fault", {})
        assert not res.is_error, body(res)
        assert body(res)["safety_state"] == "Idle"


@pytest.mark.asyncio
async def test_action_errors_map_to_codes(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        armd.action_reject = (422, "WORKSPACE_VIOLATION")
        err = body(await c.call_tool("move_to", {"x": 0.25, "y": 0.0, "z": -0.1}))
        assert err["code"] == "WORKSPACE_VIOLATION" and err["hint"] == "choose a closer or higher target"
        armd.action_reject = None
        armd.action_failure = ("TRACKING_STALL", "J2 blocked")
        err = body(await c.call_tool("move_relative", {"frame": "base", "dz": 0.05}))
        assert err["code"] == "TRACKING_STALL" and "snapshot" in err["hint"] and err["action_id"] == 1


@pytest.mark.asyncio
async def test_second_motion_call_is_busy_and_stop_bypasses(armd, make_ctrl):
    armd.action_duration = 0.6
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        first = asyncio.create_task(c.call_tool("move_to", {"x": 0.25, "y": 0.0, "z": 0.25}))
        await asyncio.sleep(0.15)
        busy = await c.call_tool("home", {})
        assert body(busy)["code"] == "BUSY"
        stopped = await c.call_tool("stop", {})
        assert not stopped.is_error
        err = body(await first)
        assert err["code"] == "STOPPED"


@pytest.mark.asyncio
async def test_release_control_then_no_lease(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        assert body(await c.call_tool("release_control", {})) == {"released": True}
        assert armd.lease is None
        err = body(await c.call_tool("home", {}))
        assert err["code"] == "NO_LEASE"


@pytest.mark.asyncio
async def test_snapshot_returns_image(make_ctrl):
    pytest.importorskip("av")
    async with Client(build_server(make_ctrl(start=False))) as c:
        res = await c.call_tool("snapshot", {"max_width": 160, "quality": 70})
        assert not res.is_error, res.content
        img = [x for x in res.content if x.type == "image"][0]
        assert img.mime_type == "image/jpeg" and base64.b64decode(img.data)[:2] == b"\xff\xd8"
        meta = body(res)
        assert (meta["width"], meta["height"]) == (160, 120) and meta["device_ts_ms"] == 1_762_000_000_100
        res = await c.call_tool("snapshot_annotated", {"min_score": 0.3})
        meta = body(res)
        assert meta["note"].startswith("inference disabled") and meta["det_age_vs_frame_ms"] == 100
        assert [x.type for x in res.content] == ["image", "text"]


@pytest.mark.asyncio
async def test_detect(make_ctrl):
    async with Client(build_server(make_ctrl(start=False))) as c:
        d = body(await c.call_tool("detect", {"wait_ms": 1000, "min_score": 0.3}))
    assert d["frame_id"] == 5 and d["count"] == 1
    (det,) = d["detections"]
    assert det["label"] == "cup" and det["box_norm"] == [0.4, 0.4, 0.6, 0.6] and det["center_norm"] == [0.5, 0.5]
    assert TRIGGER_BODIES[-1] == {"pipelines": ["yolo", "aruco"]}  # default: both


@pytest.mark.asyncio
async def test_detect_with_pipelines(make_ctrl):
    async with Client(build_server(make_ctrl(start=False))) as c:
        d = body(await c.call_tool("detect", {"pipelines": "aruco"}))
        assert TRIGGER_BODIES[-1] == {"pipelines": ["aruco"]} and d["frame_id"] == 5
        body(await c.call_tool("detect", {"pipelines": "yolo"}))
        assert TRIGGER_BODIES[-1] == {"pipelines": ["yolo"]}
        res = await c.call_tool("detect", {"pipelines": "video"})
        assert res.is_error


@pytest.mark.asyncio
async def test_poses_place_keepout_tools(armd, make_ctrl):
    armd.devplat = True
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        assert body(await c.call_tool("list_poses", {}))["count"] == 0
        assert body(await c.call_tool("save_pose", {"name": "bin_a"}))["saved"] == "bin_a"
        assert body(await c.call_tool("save_pose", {"name": "bad name"}))["code"] == "INVALID_ARGUMENT"
        assert body(await c.call_tool("list_poses", {}))["poses"][0]["name"] == "bin_a"
        r = body(await c.call_tool("move_to_pose", {"name": "bin_a", "speed_scale": 0.3}))
        assert r["kind"] == "move_to_pose" and armd.action_bodies[-1][1]["speed_scale"] == 0.3
        assert body(await c.call_tool("move_to_pose", {"name": "nope"}))["code"] == "POSE_NOT_FOUND"
        r = body(await c.call_tool("place", {"pose": "bin_a"}))
        assert r["kind"] == "place" and armd.action_bodies[-1][1]["pose"] == "bin_a"
        await c.call_tool("place", {"x": 0.2, "y": 0.0, "z": 0.05, "rpy_deg": [0, 90, 0], "release": False})
        sent = armd.action_bodies[-1][1]
        assert sent["position"] == [0.2, 0.0, 0.05] and sent["release"] is False
        assert sent["orientation"]["rpy"][1] == pytest.approx(1.5708, abs=1e-3)
        assert body(await c.call_tool("place", {}))["code"] == "INVALID_ARGUMENT"
        assert body(await c.call_tool("place", {"pose": "a", "x": 1, "y": 0, "z": 0}))["code"] == "INVALID_ARGUMENT"
        box = {"name": "wall", "min_m": [0.3, -0.1, 0.0], "max_m": [0.4, 0.1, 0.3]}
        assert body(await c.call_tool("set_keepout", {"boxes": [box]}))["boxes"] == [box]
        assert body(await c.call_tool("get_keepout", {}))["boxes"] == [box]
        assert (
            await c.call_tool("set_keepout", {"boxes": [{"name": "x", "min_m": [0, 0], "max_m": [1, 1, 1]}]})
        ).is_error


@pytest.mark.asyncio
async def test_new_tools_report_old_armd(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:  # capabilities version 5
        for tool, args in (
            ("list_poses", {}),
            ("save_pose", {"name": "a"}),
            ("move_to_pose", {"name": "a"}),
            ("place", {"pose": "a"}),
            ("get_keepout", {}),
        ):
            err = body(await c.call_tool(tool, args))
            assert err["code"] == "UNSUPPORTED" and "Update rebot-armd" in err["message"], tool


@pytest.mark.asyncio
async def test_recover_only_for_recoverable_faults(armd, make_ctrl):
    armd.devplat = True
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        assert body(await c.call_tool("recover", {}))["recovered"] is False
        armd.trigger_fault("FollowingError")
        err = body(await c.call_tool("recover", {}))
        assert err["code"] == "NOT_RECOVERABLE" and "web cockpit" in err["hint"]
        assert armd.state == "Fault"
        armd.state, armd.fault = "Armed", None
        armd.trigger_fault("WatchdogTimeout")
        ok = body(await c.call_tool("recover", {}))
        assert ok["recovered"] is True and ok["cleared"] == "WATCHDOG_TIMEOUT"
        assert armd.state == "Armed"


@pytest.mark.asyncio
async def test_recover_does_not_clear_an_estop_that_latched_after_the_check(armd, make_ctrl):
    from test_arm_facade import estop_after_first_observation

    armd.devplat = True
    ctrl = make_ctrl()
    async with Client(build_server(ctrl)) as c:
        await c.call_tool("enable", {})
        armd.trigger_fault("WatchdogTimeout")
        await asyncio.sleep(0.3)
        estop_after_first_observation(armd, ctrl.actuator())
        err = body(await c.call_tool("recover", {}))
        assert err["code"] == "NOT_RECOVERABLE" and "EStop" in err["message"]
        assert armd.state == "EStop"


@pytest.mark.asyncio
async def test_collision_tools_and_state(armd, make_ctrl):
    armd.devplat = True
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        assert body(await c.call_tool("get_state", {}))["collision"] is None
        assert body(await c.call_tool("collision_guard_status", {}))["enabled"] is False
        cfg = body(await c.call_tool("set_collision_guard", {"enabled": True, "sensitivity": "low"}))
        assert cfg["enabled"] is True and cfg["sensitivity"] == "low"
        assert body(await c.call_tool("set_collision_guard", {}))["code"] == "INVALID_ARGUMENT"
        armd.trip_collision(joint=3, residual_nm=6.5, resume_after_s=30)
        await asyncio.sleep(0.3)
        col = body(await c.call_tool("get_state", {}))["collision"]
        assert col["active"] is True and col["joint"] == 3
        err = body(await c.call_tool("home", {}))
        assert err["code"] == "COLLISION_HOLD" and "collision.active" in err["hint"]


@pytest.mark.asyncio
async def test_recover_refuses_collision_fault(armd, make_ctrl):
    armd.devplat = True
    async with Client(build_server(make_ctrl())) as c:
        await c.call_tool("enable", {})
        await c.call_tool("set_collision_guard", {"enabled": True})
        for _ in range(4):
            armd.trip_collision(resume_after_s=0.05)
        await asyncio.sleep(0.3)
        err = body(await c.call_tool("recover", {}))
        assert err["code"] == "NOT_RECOVERABLE" and "COLLISION" in err["message"]
        assert "check the workspace" in err["hint"] and "cockpit" in err["hint"]
        assert armd.state == "Fault"
        assert "/api/v1/safety/clear_fault" not in armd.commands


@pytest.mark.asyncio
async def test_collision_tools_need_capability(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        assert body(await c.call_tool("collision_guard_status", {}))["code"] == "UNSUPPORTED"
        assert body(await c.call_tool("set_collision_guard", {"enabled": True}))["code"] == "UNSUPPORTED"


@pytest.mark.asyncio
async def test_describe_safety(armd, make_ctrl):
    async with Client(build_server(make_ctrl())) as c:
        assert body(await c.call_tool("describe_safety", {}))["code"] == "UNSUPPORTED"
    armd.devplat = True
    async with Client(build_server(make_ctrl())) as c:
        d = body(await c.call_tool("describe_safety", {}))
    assert d["hardware"]["physical_estop"] is False
    assert "ABSENT" in d["summary"] and "WATCHDOG_TIMEOUT" in d["summary"] and "e_stop" in d["summary"]
