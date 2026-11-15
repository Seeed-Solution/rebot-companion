"""Comprehensive test suite for Stage 2 Playground Bridge deliverables:

1. Built-in tool packs (`suction`, `screwdriver`) and scene pack (`pcb_inspection`).
2. Level 5 closed-loop multimodal ONNX policy (camera + state control).
3. Static bundle exporter (`bundle_export.py` + `armd.wasm` / fallback).
4. Deploy profile validator, safe applier, and Python 3.8 `main.py` generator (`deploy_profile.py`).
5. Same-origin course proxy (`playground_proxy.py`).
"""

from __future__ import annotations

import ast
import asyncio
import json
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets.asyncio.client
from websockets.asyncio.server import serve as ws_serve

from rebot_companion.deploy_profile import (
    DEPLOY_PROFILE_FORMAT,
    DeployProfile,
    apply_deploy_profile,
    render_main_py,
    validate_deploy_profile,
)
from rebot_companion.onnx_policy import (
    scaffold_closed_loop_demo_policy,
    validate_policy_dir,
)
from rebot_companion.scenes.bundle_export import (
    compose_mjcf_via_wasm,
    export_static_bundle,
    resolve_armd_wasm,
)
from rebot_companion.scenes.playground_proxy import PlaygroundProxyServer
from rebot_companion.scenes.spec import (
    list_builtin_packs,
    load_registry,
    resolve_builtin_path,
    validate_scene_dir,
    validate_tool_dir,
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# 1. Built-in Packs & Registry
# ---------------------------------------------------------------------------

def test_builtin_tool_packs_materialized():
    suction_path = resolve_builtin_path("tool", "suction")
    assert suction_path is not None and suction_path.is_dir()
    suction_m = validate_tool_dir(suction_path)
    assert suction_m.id == "suction"
    assert suction_m.kind == "suction"
    assert suction_m.driven_joint == "suction_valve"
    assert suction_m.coupled_joint is None
    assert suction_m.actuator == "m7_suction"
    assert len(suction_m.io) == 2

    driver_path = resolve_builtin_path("tool", "screwdriver")
    assert driver_path is not None and driver_path.is_dir()
    driver_m = validate_tool_dir(driver_path)
    assert driver_m.id == "screwdriver"
    assert driver_m.kind == "screwdriver"
    assert driver_m.driven_joint == "spindle_joint"
    assert driver_m.actuator == "m7_spindle"
    assert driver_m.motor_to_joint_scale == 1.0
    assert len(driver_m.io) == 1


def test_builtin_scene_pcb_inspection():
    scene_path = resolve_builtin_path("scene", "pcb_inspection")
    assert scene_path is not None and scene_path.is_dir()
    m = validate_scene_dir(scene_path)
    assert m.id == "pcb_inspection"
    assert m.default_tool == "suction"
    assert "suction" in m.recommended_tools
    assert "screwdriver" in m.recommended_tools

    # Zones
    assert "infeed_tray" in m.zones
    assert "test_fixture" in m.zones
    assert "outfeed_bin" in m.zones
    assert "ng_bin" in m.zones

    # Keepout boxes: strictly min < max
    assert len(m.keepout_boxes) == 2
    box_names = {b["name"] for b in m.keepout_boxes}
    assert "divider_wall" in box_names
    assert "bin_divider" in box_names
    for b in m.keepout_boxes:
        for i in range(3):
            assert b["min"][i] < b["max"][i]

    # Task stages
    stage_ids = [s["id"] for s in m.task_stages]
    assert stage_ids == ["pcb_picked", "fixture_tested", "pcb_sorted"]

    # Registry
    reg = load_registry()
    scene_entry = next((s for s in reg.get("scenes", []) if s.get("id") == "pcb_inspection"), None)
    assert scene_entry is not None
    assert scene_entry["default_tool"] == "suction"

    packs = list_builtin_packs()
    assert any(s["id"] == "pcb_inspection" for s in packs["scenes"])
    assert any(t["id"] == "suction" for t in packs["tools"])
    assert any(t["id"] == "screwdriver" for t in packs["tools"])


# ---------------------------------------------------------------------------
# 2. Closed-Loop Level 5 ONNX Policy
# ---------------------------------------------------------------------------

def test_closed_loop_onnx_policy_contraction_and_camera_control(tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")
    policy_dir = tmp_path / "closed_loop_policy"
    target_q = [0.08, -0.35, 0.55, 0.0, 1.15, 0.0, 0.0]

    scaffold_closed_loop_demo_policy(
        policy_dir,
        policy_id="test_visual_servo",
        action_dim=7,
        robot="b601dm",
        target_q=target_q,
        state_gain=0.88,
        image_layout="NCHW",
        image_hw=(64, 64),
    )

    manifest = validate_policy_dir(policy_dir)
    assert manifest.inputs["image"]["shape"] == [1, 3, 64, 64]
    assert manifest.inputs["state"]["shape"] == [1, 7]

    model_file = policy_dir / manifest.model_file
    session = ort.InferenceSession(str(model_file))

    # Test camera influence
    q = np.zeros((1, 7), dtype=np.float32)
    img_black = np.zeros((1, 3, 64, 64), dtype=np.float32)
    img_green = np.zeros((1, 3, 64, 64), dtype=np.float32)
    img_green[:, 1, :, :] = 1.0  # Full green channel

    act_b = session.run(["action"], {"observation.state": q, "observation.images.wrist": img_black})[0]
    act_g = session.run(["action"], {"observation.state": q, "observation.images.wrist": img_green})[0]

    diff = np.abs(act_b - act_g).max()
    assert diff > 0.01, f"Expected camera image to measurably change action; diff was {diff}"

    # Test 40-step monotonic contraction starting from q0
    curr_q = np.zeros((1, 7), dtype=np.float32)
    prev_dist = float("inf")
    target_arr = np.array(target_q, dtype=np.float32)

    for step in range(40):
        curr_q = session.run(["action"], {"observation.state": curr_q, "observation.images.wrist": img_black})[0]
        dist = float(np.linalg.norm(curr_q[0] - target_arr))
        assert dist <= prev_dist + 1e-6, f"Step {step}: distance to target_q did not decrease ({dist} > {prev_dist})"
        prev_dist = dist

    assert prev_dist < 0.01, f"Expected final state to be close to target_q; distance was {prev_dist}"


def test_closed_loop_onnx_policy_nhwc(tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")
    policy_dir = tmp_path / "nhwc_policy"
    scaffold_closed_loop_demo_policy(
        policy_dir,
        policy_id="test_nhwc",
        action_dim=7,
        image_layout="NHWC",
        image_hw=(32, 32),
    )
    manifest = validate_policy_dir(policy_dir)
    assert manifest.inputs["image"]["shape"] == [1, 32, 32, 3]

    session = ort.InferenceSession(str(policy_dir / "model.onnx"))
    q = np.zeros((1, 7), dtype=np.float32)
    img = np.zeros((1, 32, 32, 3), dtype=np.float32)
    act = session.run(["action"], {"observation.state": q, "observation.images.wrist": img})[0]
    assert act.shape == (1, 7)


# ---------------------------------------------------------------------------
# 3. Static Bundle Exporter
# ---------------------------------------------------------------------------

def test_compose_mjcf_via_live_wasm():
    wasm_path = resolve_armd_wasm()
    assert wasm_path is not None and wasm_path.is_file(), f"armd.wasm not found at {wasm_path}"

    xml = compose_mjcf_via_wasm(
        wasm_path=wasm_path,
        tool_preset="suction",
    )
    assert "<mujoco" in xml
    assert "suction_valve" in xml or "m7_suction" in xml


def test_export_static_bundle_invariants(tmp_path: Path):
    out_dir = tmp_path / "bundle_export_test"
    policy_dir = tmp_path / "policy_for_bundle"
    scaffold_closed_loop_demo_policy(policy_dir, policy_id="demo_policy")

    exported_dir = export_static_bundle(
        out_dir=out_dir,
        scene_id="pcb_inspection",
        tool_id="suction",
        policy_dir=policy_dir,
        robot="b601dm",
        overwrite=True,
        allow_fallback=True,
    )

    bundle_json_path = exported_dir / "bundle.json"
    model_onnx_path = exported_dir / "model.onnx"
    assert bundle_json_path.is_file()
    assert model_onnx_path.is_file()

    content = bundle_json_path.read_text(encoding="utf-8")
    assert "/assets/wasm/" not in content
    assert "/api/v1/" not in content

    data = json.loads(content)
    assert data["format"] == "rebot-playground-bundle/v1"
    assert data["scene_id"] == "pcb_inspection"
    assert data["tool_id"] == "suction"
    assert data["robot"] == "b601dm"
    assert data["model_onnx_rel"] == "./model.onnx"
    assert "infeed_tray" in data["mjcf_xml"]
    assert "test_fixture" in data["mjcf_xml"]


# ---------------------------------------------------------------------------
# 4. Deploy Profile & Python 3.8 main.py Generator
# ---------------------------------------------------------------------------

def test_deploy_profile_validation_and_rejections():
    valid_raw: dict[str, Any] = {
        "format": DEPLOY_PROFILE_FORMAT,
        "robot": "b601dm",
        "scene_id": "pcb_inspection",
        "tool_id": "suction",
        "poses": {
            "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "pick": [0.1, -0.2, 0.3, 0.0, 1.0, 0.0, 0.0],
        },
        "keepout_boxes": [
            {"name": "wall", "min_m": [0.1, -0.1, 0.0], "max_m": [0.2, 0.1, 0.1]},
        ],
        "motion": {"speed_scale": 0.3, "linear_speed_mps": 0.04},
        "safety": {"collision_guard_enabled": True, "collision_sensitivity": "medium"},
        "program": {"cycles": 5, "dwell_s": 0.2},
    }

    prof = validate_deploy_profile(valid_raw)
    assert prof.scene_id == "pcb_inspection"
    assert len(prof.poses) == 2

    # Inverted keepout box rejection
    bad_box = json.loads(json.dumps(valid_raw))
    bad_box["keepout_boxes"] = [{"name": "inv", "min_m": [0.2, 0.0, 0.0], "max_m": [0.1, 0.1, 0.1]}]
    with pytest.raises(ValueError, match="Inverted keepout box"):
        validate_deploy_profile(bad_box)

    # Excessive speed rejection (> 0.15 m/s)
    bad_speed = json.loads(json.dumps(valid_raw))
    bad_speed["motion"]["linear_speed_mps"] = 0.20
    with pytest.raises(ValueError, match="linear_speed_mps"):
        validate_deploy_profile(bad_speed)

    # Invalid pose name rejection
    bad_pose = json.loads(json.dumps(valid_raw))
    bad_pose["poses"]["invalid name with spaces!"] = [0.0] * 7
    with pytest.raises(ValueError, match="Invalid pose name"):
        validate_deploy_profile(bad_pose)


def test_render_main_py_python38_ast():
    prof = DeployProfile(
        format=DEPLOY_PROFILE_FORMAT,
        robot="b601dm",
        scene_id="pcb_inspection",
        tool_id="suction",
        poses={
            "scan": [0.08, -0.35, 0.55, 0.0, 1.15, 0.0, 0.0],
            "pick": [0.10, -0.25, 0.45, 0.0, 1.05, 0.0, 0.0],
        },
        keepout_boxes=[{"name": "box1", "min_m": [0.1, -0.1, 0.0], "max_m": [0.2, 0.1, 0.1]}],
        motion={"speed_scale": 0.5, "linear_speed_mps": 0.08},
        safety={"collision_guard_enabled": False, "collision_sensitivity": "medium"},
        program={"cycles": 10, "dwell_s": 0.1},
    )

    code = render_main_py(prof)
    # Check Python 3.8 AST parse compatibility
    parsed = ast.parse(code, filename="main.py", mode="exec", feature_version=(3, 8))
    assert parsed is not None

    # Verify first-run safety speed clamp (0.5 clamped to 0.3)
    assert '"speed_scale": 0.3' in code
    assert '"linear_speed_mps": 0.04' in code
    assert "set_tool_io" in code
    assert "arm.wait_until_clear()" in code


def test_apply_deploy_profile_dry_run(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    out_script = tmp_path / "main.py"
    data = {
        "format": DEPLOY_PROFILE_FORMAT,
        "robot": "b601dm",
        "scene_id": "pcb_inspection",
        "tool_id": "suction",
        "poses": {"home": [0.0] * 7},
        "keepout_boxes": [{"name": "box", "min": [0.1, 0.1, 0.0], "max": [0.2, 0.2, 0.1]}],
        "motion": {"speed_scale": 0.2},
    }
    profile_path.write_text(json.dumps(data), encoding="utf-8")

    res = apply_deploy_profile(
        profile_path,
        host="127.0.0.1",
        port=9999,  # Unused host/port in dry-run
        apply_changes=False,
        out_script=out_script,
    )
    assert res["applied"] is False
    assert res["dry_run"] is True
    assert "home" in res["diff"]["poses_to_set"]
    assert "box" in res["diff"]["keepout_boxes_to_set"]
    assert out_script.is_file()


def test_apply_deploy_profile_live_apply(armd, tmp_path: Path):
    armd.devplat = True
    profile_path = tmp_path / "apply_live_profile.json"
    backup_file = tmp_path / "custom_backup.json"
    out_script = tmp_path / "rendered_script.py"

    data = {
        "format": DEPLOY_PROFILE_FORMAT,
        "robot": "b601dm",
        "scene_id": "pcb_inspection",
        "tool_id": "suction",
        "poses": {
            "pose_a": [0.08, -0.35, 0.55, 0.0, 1.15, 0.0, 0.0],
            "pose_b": [0.10, -0.25, 0.45, 0.0, 1.05, 0.0, 0.0],
        },
        "keepout_boxes": [
            {"name": "box1", "min_m": [0.1, -0.1, 0.0], "max_m": [0.2, 0.1, 0.1]},
            {"name": "box2", "min_m": [0.2, 0.1, 0.0], "max_m": [0.3, 0.2, 0.1]},
        ],
        "motion": {"speed_scale": 0.25, "linear_speed_mps": 0.03},
        "safety": {"collision_guard_enabled": True, "collision_sensitivity": "high"},
        "program": {"cycles": 2, "dwell_s": 0.05},
    }
    profile_path.write_text(json.dumps(data), encoding="utf-8")

    res = apply_deploy_profile(
        profile_path,
        client=armd.client,
        apply_changes=True,
        backup_path=backup_file,
        out_script=out_script,
    )

    assert res["applied"] is True
    assert res["dry_run"] is False
    assert backup_file.is_file()
    assert out_script.is_file()

    # Verify armd state
    assert "pose_a" in armd.poses
    assert "pose_b" in armd.poses
    assert len(armd.keepout) == 2


def test_cli_export_bundle_and_apply_profile(tmp_path: Path):
    from rebot_companion.scenes.cli import main as cli_main

    # 1. CLI export-bundle
    bundle_dir = tmp_path / "cli_bundle"
    ret = cli_main(["export-bundle", "--out", str(bundle_dir), "--scene", "pcb_inspection", "--tool", "suction"])
    assert ret == 0
    assert (bundle_dir / "bundle.json").is_file()

    # 2. CLI apply-profile (dry run with script generation)
    profile_path = tmp_path / "cli_profile.json"
    out_script = tmp_path / "cli_main.py"
    data = {
        "format": DEPLOY_PROFILE_FORMAT,
        "robot": "b601dm",
        "scene_id": "pcb_inspection",
        "tool_id": "suction",
        "poses": {"home": [0.0] * 7},
        "keepout_boxes": [{"name": "box", "min": [0.1, 0.1, 0.0], "max": [0.2, 0.2, 0.1]}],
        "motion": {"speed_scale": 0.2},
    }
    profile_path.write_text(json.dumps(data), encoding="utf-8")
    ret2 = cli_main(
        ["apply-profile", str(profile_path), "--host", "127.0.0.1", "--port", "9999", "--out-script", str(out_script)]
    )
    assert ret2 == 0
    assert out_script.is_file()


# ---------------------------------------------------------------------------
# 5. Playground Proxy Server (HTTP / WS / Static)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_playground_proxy_static_and_spa(tmp_path: Path):
    static_dir = tmp_path / "public"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!DOCTYPE html><html><body>Index</body></html>", encoding="utf-8")
    (static_dir / "app.js").write_text("console.log('hi');", encoding="utf-8")
    (static_dir / "model.onnx").write_bytes(b"\x08\x01\x12\x04ONNX")
    (static_dir / "core.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

    proxy = PlaygroundProxyServer(static_dir=static_dir, armd_url="http://127.0.0.1:8765", port=0, enable_coop=True)
    async with proxy:
        port = proxy.port

        def get(path: str) -> tuple[int, bytes, dict[str, str]]:
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    h = {k.lower(): v for k, v in resp.headers.items()}
                    return resp.status, resp.read(), h
            except urllib.error.HTTPError as err:
                h = {k.lower(): v for k, v in err.headers.items()}
                return err.code, err.read(), h

        # 1. Root -> index.html
        status, body, headers = await asyncio.to_thread(get, "/")
        assert status == 200
        assert b"Index" in body
        assert headers.get("cross-origin-opener-policy") == "same-origin"

        # 2. Static file with correct MIME type
        status, body, headers = await asyncio.to_thread(get, "/core.wasm")
        assert status == 200
        assert headers.get("content-type") == "application/wasm"

        # 3. Missing JS file -> 404 (NOT index.html)
        status, body, headers = await asyncio.to_thread(get, "/nonexistent.js")
        assert status == 404

        # 4. Extensionless route -> SPA fallback to index.html
        status, body, headers = await asyncio.to_thread(get, "/course/level5")
        assert status == 200
        assert b"Index" in body


@pytest.mark.asyncio
async def test_playground_proxy_http_origin_rewrite():
    fake_armd_port = get_free_port()
    received_requests: list[dict[str, Any]] = []

    async def armd_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        line = await reader.readline()
        method, path, _ = line.decode("latin1").strip().split(" ", 2)
        headers = {}
        while True:
            h = await reader.readline()
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.decode("latin1").rstrip("\r\n").partition(":")
            headers[k.strip().lower()] = v.strip()
        cl = int(headers.get("content-length", 0))
        body = await reader.readexactly(cl) if cl > 0 else b""
        received_requests.append({"method": method, "path": path, "headers": headers, "body": body})

        reply = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 12\r\n"
            b"Connection: close\r\n\r\n"
            b'{"ok": true}'
        )
        writer.write(reply)
        await writer.drain()
        writer.close()

    armd_server = await asyncio.start_server(armd_handler, "127.0.0.1", fake_armd_port)

    proxy = PlaygroundProxyServer(
        static_dir="/tmp",
        armd_url=f"http://127.0.0.1:{fake_armd_port}",
        port=0,
    )
    async with proxy:
        proxy_port = proxy.port

        def post_api() -> tuple[int, bytes]:
            payload = json.dumps({"test": 123}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/api/v1/action/move_to",
                data=payload,
                headers={"Content-Type": "application/json", "Origin": "http://rogue-origin.com"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status, resp.read()

        status, body = await asyncio.to_thread(post_api)
        assert status == 200
        assert b'"ok": true' in body

        assert len(received_requests) == 1
        rec = received_requests[0]
        # Verified: Host and Origin were rewritten to fake armd target origin
        assert rec["headers"]["origin"] == f"http://127.0.0.1:{fake_armd_port}"
        assert rec["headers"]["host"] == f"127.0.0.1:{fake_armd_port}"

    armd_server.close()
    await armd_server.wait_closed()


@pytest.mark.asyncio
async def test_playground_proxy_websocket_relay():
    fake_armd_port = get_free_port()
    ws_received: list[str] = []

    async def ws_handler(websocket: websockets.asyncio.server.ServerConnection):
        async for msg in websocket:
            ws_received.append(str(msg))
            await websocket.send(f"echo:{msg}")

    armd_ws = await ws_serve(ws_handler, "127.0.0.1", fake_armd_port)

    proxy = PlaygroundProxyServer(
        static_dir="/tmp",
        armd_url=f"http://127.0.0.1:{fake_armd_port}",
        port=0,
    )
    async with proxy:
        proxy_port = proxy.port
        uri = f"ws://127.0.0.1:{proxy_port}/api/v1/ws"

        async with websockets.asyncio.client.connect(uri, origin="http://browser.local") as client_ws:
            await client_ws.send('{"type": "acquire_lease", "seq": 1}')
            reply = await client_ws.recv()
            assert reply == 'echo:{"type": "acquire_lease", "seq": 1}'

    armd_ws.close()
    await armd_ws.wait_closed()
    assert len(ws_received) == 1
