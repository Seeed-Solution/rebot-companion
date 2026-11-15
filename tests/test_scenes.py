"""Tests for rebot_companion.scenes manifests, assets, client, CLI, and coffee_kiosk example."""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from rebot_companion.scenes import (
    AssetChecksumError,
    AssetDownloadError,
    AssetRef,
    SceneAssetManager,
    SceneManifest,
    SimClientError,
    SimSceneClient,
    idempotent_asset_name,
    list_builtin_packs,
    prepare_asset_metadata,
    resolve_builtin_path,
    scaffold_pack,
    validate_scene_dir,
    validate_tool_dir,
)
from rebot_companion.scenes.cli import main as scene_cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_builtin_coffee_kiosk_and_wide_cup_gripper_manifests():
    packs = list_builtin_packs()
    scene_ids = {s["id"] for s in packs["scenes"]}
    tool_ids = {t["id"] for t in packs["tools"]}
    assert "coffee_kiosk" in scene_ids
    assert {"default", "wide_cup_gripper", "suction", "screwdriver"} <= tool_ids

    coffee_dir = resolve_builtin_path("scene", "coffee_kiosk")
    gripper_dir = resolve_builtin_path("tool", "wide_cup_gripper")
    assert coffee_dir is not None and gripper_dir is not None

    sm = validate_scene_dir(coffee_dir)
    tm = validate_tool_dir(gripper_dir)
    assert [st["id"] for st in sm.task_stages] == ["cup_grasped", "cup_on_tray", "brew_pressed", "cup_served"]
    assert tm.kind == "gripper"
    assert tm.driven_joint == "cup_jaw_left"
    assert tm.coupled_joint == "cup_jaw_right"


def test_scaffold_and_validation_security(tmp_path: Path):
    scene_dir = scaffold_pack(tmp_path / "custom_pcb", kind="scene", pack_id="custom_pcb")
    tool_dir = scaffold_pack(tmp_path / "custom_jaw", kind="tool", pack_id="custom_jaw")
    sm = validate_scene_dir(scene_dir)
    tm = validate_tool_dir(tool_dir)
    assert sm.id == "custom_pcb"
    assert tm.id == "custom_jaw"

    # Reject path traversal in world_xml / invalid ids
    (scene_dir / "scene.json").write_text(
        json.dumps({"id": "custom_pcb", "name": "Bad", "world_xml": "../escape.xml"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        validate_scene_dir(scene_dir)

    (scene_dir / "scene.json").write_text(json.dumps({"id": "bad/id", "world_xml": "world.xml"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid scene id"):
        validate_scene_dir(scene_dir)


def test_asset_manager_md5_naming_sha256_pull_and_tamper_rejection(tmp_path: Path):
    srv_dir = tmp_path / "cdn"
    srv_dir.mkdir()
    payload = b"solid cup_mesh\nendsolid cup_mesh\n"
    hashed_name = idempotent_asset_name("cup.stl", data=payload)
    expected_md58 = hashlib.md5(payload).hexdigest()[:8]
    assert hashed_name == f"cup-{expected_md58}.stl"

    src_file = tmp_path / "cup.stl"
    src_file.write_bytes(payload)
    meta = prepare_asset_metadata(src_file, remote_prefix="rebot/scenes/coffee_kiosk")
    assert meta["idempotent_filename"] == hashed_name
    assert (
        meta["oss_url"] == f"https://sensecraft-statics.seeed.cc/solution-app/rebot/scenes/coffee_kiosk/{hashed_name}"
    )

    served_file = srv_dir / hashed_name
    served_file.write_bytes(payload)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(srv_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        cache_dir = tmp_path / "cache"
        mgr = SceneAssetManager(cache_dir=cache_dir)

        asset = AssetRef(
            id="cup_mesh",
            filename="cup.stl",
            url=f"http://127.0.0.1:{port}/{hashed_name}",
            sha256=meta["sha256"],
            tier=1,
            required=True,
        )
        manifest = SceneManifest(id="coffee_kiosk", name="Coffee Kiosk", assets=[asset])
        resolved = mgr.ensure_scene_assets(manifest, allow_download=True)
        cached_path = resolved["cup_mesh"]
        assert cached_path.read_bytes() == payload
        assert mgr.status(manifest)[0]["cached"] is True

        # Subsequent cache hit succeeds even when remote URL is unreachable
        offline_asset = AssetRef(
            id="cup_mesh",
            filename="cup.stl",
            url="http://127.0.0.1:9/unreachable.stl",
            sha256=meta["sha256"],
            tier=1,
            required=True,
        )
        assert mgr.ensure_asset("coffee_kiosk", offline_asset, allow_download=True) == cached_path

        # Corrupt downloaded payload -> AssetChecksumError & clean removal of .part file
        tampered = AssetRef(
            id="tampered",
            filename="bad_cup.stl",
            url=f"http://127.0.0.1:{port}/{hashed_name}",
            sha256="0" * 64,
            tier=1,
            required=True,
        )
        with pytest.raises(AssetChecksumError):
            mgr.ensure_asset("coffee_kiosk", tampered, allow_download=True)
        bad_dest = cache_dir / "coffee_kiosk" / "bad_cup.stl"
        assert not bad_dest.exists()
        assert not (bad_dest.with_name("bad_cup.stl.part")).exists()

        # Optional Tier-1 offline fallback returns None; required missing raises AssetDownloadError
        opt_asset = AssetRef(id="opt", filename="opt.stl", url="http://127.0.0.1:9/missing.stl", required=False)
        assert mgr.ensure_asset("coffee_kiosk", opt_asset, allow_download=True) is None
        req_asset = AssetRef(id="req", filename="req.stl", url="http://127.0.0.1:9/missing.stl", required=True)
        with pytest.raises(AssetDownloadError):
            mgr.ensure_asset("coffee_kiosk", req_asset, allow_download=True)
    finally:
        httpd.shutdown()
        httpd.server_close()


def _make_mock_sim_server():
    state: dict[str, Any] = {
        "scene": "coffee_kiosk",
        "tool": "wide_cup_gripper",
        "io": {"vacuum": False, "brew_led": False},
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: dict):
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            state["requests"].append(("GET", self.path, None))
            if self.path == "/sim/scene":
                self._send(
                    200,
                    {
                        "scene": state["scene"],
                        "tool": state["tool"],
                        "scene_pack": {"id": state["scene"]},
                        "tool_pack": {"id": state["tool"]},
                        "io": state["io"],
                    },
                )
            elif self.path == "/sim/io":
                self._send(200, {"io": state["io"]})
            elif self.path == "/sim/state":
                self._send(
                    200,
                    {
                        "scene": state["scene"],
                        "tool": state["tool"],
                        "objects": {"cup": {"pos": [0.27, -0.1, 0.035], "grasped": False}},
                        "task": {"stages": {"cup_grasped": False, "cup_on_tray": False}, "completed": False},
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            state["requests"].append(("POST", self.path, body))
            if self.path == "/sim/scene":
                if body.get("scene") == "bad_scene":
                    self._send(400, {"error": "invalid scene"})
                    return
                if "scene" in body and body["scene"]:
                    state["scene"] = Path(str(body["scene"])).name
                if "tool" in body and body["tool"]:
                    state["tool"] = Path(str(body["tool"])).name
                self._send(
                    200,
                    {
                        "scene": state["scene"],
                        "tool": state["tool"],
                        "scene_pack": {"id": state["scene"]},
                        "tool_pack": {"id": state["tool"]},
                        "task": {"stages": {"cup_grasped": False, "cup_served": False}},
                    },
                )
            elif self.path == "/sim/io":
                state["io"].update(body)
                self._send(200, {"io": state["io"]})
            elif self.path == "/sim/reset":
                self._send(
                    200,
                    {
                        "seed": body.get("seed"),
                        "objects": {"cup": {"pos": [0.28, -0.09, 0.035]}},
                        "task": {"stages": {"cup_grasped": False, "cup_served": False}},
                    },
                )
            else:
                self._send(404, {"error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def test_sim_scene_client_and_cli_roundtrip(tmp_path: Path, capsys, monkeypatch):
    for env_var in ("REBOT_SIM_HTTP_HOST", "REBOT_SIM_HTTP_PORT", "REBOT_HOST"):
        monkeypatch.delenv(env_var, raising=False)
    assert SimSceneClient().base_url == "http://127.0.0.1:8090"

    httpd, state = _make_mock_sim_server()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        client = SimSceneClient(base_url=base_url)
        info = client.get_scene()
        assert info["scene"] == "coffee_kiosk"

        swapped = client.load_scene("coffee_kiosk", tool="wide_cup_gripper", seed=5)
        assert swapped["tool_pack"]["id"] == "wide_cup_gripper"
        # Ensure built-in names resolved to actual directory paths before sending HTTP payload
        last_post = [r for r in state["requests"] if r[0] == "POST" and r[1] == "/sim/scene"][-1]
        assert Path(last_post[2]["scene"]).name == "coffee_kiosk"
        assert Path(last_post[2]["tool"]).name == "wide_cup_gripper"

        assert client.set_io(vacuum=True)["io"]["vacuum"] is True
        with pytest.raises(SimClientError, match="invalid scene"):
            client.load_scene("bad_scene")

        # Exercise CLI subcommands cleanly
        assert scene_cli_main(["list", "--json"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert any(s["id"] == "coffee_kiosk" for s in out["scenes"])

        assert scene_cli_main(["info", "coffee_kiosk"]) == 0
        assert "coffee_kiosk" in capsys.readouterr().out

        custom_dir = tmp_path / "cli_tool"
        assert scene_cli_main(["init", str(custom_dir), "--kind", "tool"]) == 0
        assert scene_cli_main(["validate", str(custom_dir)]) == 0
        assert scene_cli_main(["load", "--tool", str(custom_dir), "--http-url", base_url]) == 0
        assert scene_cli_main(["io", "--set", "vacuum=false", "--http-url", base_url]) == 0
        assert state["io"]["vacuum"] is False
        capsys.readouterr()

        sample_asset = tmp_path / "part.stl"
        sample_asset.write_bytes(b"mesh")
        assert scene_cli_main(["asset-hash", str(sample_asset), "--json"]) == 0
        meta = json.loads(capsys.readouterr().out)
        assert meta["idempotent_filename"].startswith("part-")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_coffee_kiosk_fsm_is_stdlib_and_rebot_only():
    fsm_path = REPO_ROOT / "examples" / "scenes" / "coffee_kiosk" / "coffee_kiosk_fsm.py"
    tree = ast.parse(fsm_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {"rebot"}
    third_party = imports - allowed
    assert not third_party, f"coffee_kiosk_fsm.py must only use stdlib + rebot, found {third_party}"
    assert "CONFIG" in fsm_path.read_text(encoding="utf-8")


def test_train_and_eval_demo_dry_run():
    httpd, _ = _make_mock_sim_server()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    script = REPO_ROOT / "examples" / "scenes" / "coffee_kiosk" / "train_and_eval_demo.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--dry-run", "--episodes", "2", "--sim-http", base_url],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        assert "seed=0" in proc.stdout and "seed=1" in proc.stdout
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_builtin_pcb_suction_manifest_and_fsm_stdlib():
    packs = list_builtin_packs()
    assert "pcb_suction" in {s["id"] for s in packs["scenes"]}

    pcb_dir = resolve_builtin_path("scene", "pcb_suction")
    assert pcb_dir is not None
    sm = validate_scene_dir(pcb_dir)
    assert sm.default_tool == "suction"
    assert [st["id"] for st in sm.task_stages] == ["pcb_picked", "pcb_in_fixture", "pcb_in_outfeed"]

    fsm_path = REPO_ROOT / "examples" / "scenes" / "pcb_suction" / "pcb_suction_fsm.py"
    tree = ast.parse(fsm_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    third_party = imports - (set(sys.stdlib_module_names) | {"rebot"})
    assert not third_party, f"pcb_suction_fsm.py must only use stdlib + rebot, found {third_party}"
    assert "CONFIG" in fsm_path.read_text(encoding="utf-8")


def test_onnx_policy_scaffold_validate_and_runner(tmp_path: Path):
    import numpy as np

    from rebot_companion.onnx_policy import (
        OnnxPolicyManifest,
        build_linear_onnx_bytes,
        scaffold_onnx_policy,
        validate_policy_dir,
    )
    from rebot_companion.policy_runner import OnnxPolicy, parse_policy

    pol_dir = scaffold_onnx_policy(tmp_path / "linear_pol", policy_id="linear_pol", action_dim=7)
    manifest = validate_policy_dir(pol_dir)
    assert manifest.name == "linear_pol"
    assert manifest.action_dim == 7

    # Also test custom weights + bias + io output
    w = np.eye(7, dtype=np.float32) * 0.5
    b = np.full(7, 0.1, dtype=np.float32)
    (pol_dir / "model.onnx").write_bytes(build_linear_onnx_bytes("observation.state", "action", w, b))
    loaded = OnnxPolicyManifest.from_dict(json.loads((pol_dir / "policy.json").read_text(encoding="utf-8")))
    assert loaded.format == "rebot-onnx-policy/v1"

    pol = parse_policy(f"onnx:{pol_dir}")
    assert isinstance(pol, OnnxPolicy)
    x = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
    out = pol.select({"observation.state": x}, step=0, t=0.0)
    assert out is not None and np.allclose(out, x * 0.5 + 0.1, atol=1e-5)

    # Multimodal policy scaffold (state + wrist image)
    mm_dir = scaffold_onnx_policy(tmp_path / "mm_pol", policy_id="mm_pol", action_dim=7, include_image=True)
    mm_pol = parse_policy(str(mm_dir))
    assert isinstance(mm_pol, OnnxPolicy)
    assert mm_pol.image_key == "observation.images.wrist"
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    mm_out = mm_pol.select({"observation.state": x, "observation.images.wrist": img}, step=1, t=0.03)
    assert mm_out is not None and np.allclose(mm_out, x, atol=1e-5)


def test_armd_sandbox_client_and_cli_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from rebot_companion.scenes import ArmdSandboxClient

    recorded: list[tuple[str, str, Any]] = []
    catalog_data = {
        "mode": "wasm",
        "active_scene": "pcb_suction",
        "active_tool": "suction",
        "active_policy": None,
        "wasm_runtime_ready": True,
        "scenes": [],
        "tools": [],
        "policies": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            recorded.append(("GET", self.path, None))
            if self.path == "/api/v1/sandbox/catalog":
                self._send(200, catalog_data)
            elif self.path == "/api/v1/sandbox/state":
                self._send(200, {"mode": catalog_data["mode"], "task": {"scene": "pcb_suction", "completed": False}})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            recorded.append(("POST", self.path, body))
            if self.path == "/api/v1/sandbox/scenes":
                self._send(200, {"id": body.get("id", "custom_scene"), "sha256": "abc"})
            elif self.path == "/api/v1/sandbox/tools":
                self._send(200, {"id": body.get("id", "custom_tool"), "sha256": "def"})
            elif self.path == "/api/v1/sandbox/policies":
                self._send(200, {"id": body.get("id", "custom_pol"), "sha256": "123"})
            elif self.path == "/api/v1/sandbox/activate":
                catalog_data["mode"] = body.get("mode", "none")
                catalog_data["active_scene"] = body.get("scene")
                catalog_data["active_tool"] = body.get("tool")
                catalog_data["active_policy"] = body.get("policy")
                self._send(200, catalog_data)
            elif self.path == "/api/v1/sandbox/io":
                self._send(200, {"io_seq": 1, "desired_io": body.get("updates", {})})
            elif self.path == "/api/v1/sandbox/reset":
                self._send(200, {"reset_seq": 1, "seed": body.get("seed")})
            else:
                self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    armd_url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        scene_dir = scaffold_pack(tmp_path / "sc1", kind="scene", pack_id="sc1")
        pol_dir = tmp_path / "pol1"
        assert scene_cli_main(["init", str(pol_dir), "--kind", "policy", "--id", "pol1"]) == 0
        assert scene_cli_main(["validate", str(pol_dir)]) == 0
        assert scene_cli_main(["upload", str(scene_dir), "--id", "sc1", "--armd-url", armd_url]) == 0
        assert scene_cli_main(["upload", str(pol_dir), "--armd-url", armd_url]) == 0
        assert (
            scene_cli_main(
                [
                    "activate",
                    "--mode",
                    "wasm",
                    "--scene",
                    "sc1",
                    "--tool",
                    "suction",
                    "--policy",
                    "pol1",
                    "--armd-url",
                    armd_url,
                ]
            )
            == 0
        )
        assert scene_cli_main(["status", "--armd-url", armd_url]) == 0
        capsys.readouterr()

        client = ArmdSandboxClient(base_url=armd_url)
        assert client.catalog()["active_policy"] == "pol1"
        assert client.set_io(vacuum=True)["desired_io"]["vacuum"] is True
        assert client.reset(seed=7)["seed"] == 7
        assert any(p == "/api/v1/sandbox/policies" for _, p, _ in recorded)
    finally:
        srv.shutdown()
        srv.server_close()


def test_collect_pack_files_symlink_escape(tmp_path: Path):
    from rebot_companion.scenes.client import _collect_pack_files

    pack_dir = tmp_path / "my_pack"
    pack_dir.mkdir()
    (pack_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    # Internal symlink should succeed
    internal_target = pack_dir / "scene.xml"
    (pack_dir / "symlink_ok.xml").symlink_to(internal_target)
    files = _collect_pack_files(pack_dir)
    assert "scene.xml" in files
    assert "symlink_ok.xml" in files

    # Escaping symlink pointing outside pack_dir must raise ValueError
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret_data", encoding="utf-8")
    (pack_dir / "escaped.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="resolves outside pack directory"):
        _collect_pack_files(pack_dir)


def test_armd_sandbox_client_default_port_and_env(monkeypatch: pytest.MonkeyPatch):
    from rebot_companion.scenes.client import DEFAULT_ARMD_HTTP_URL, ArmdSandboxClient

    assert DEFAULT_ARMD_HTTP_URL == "http://127.0.0.1:8765"

    monkeypatch.delenv("REBOT_ARMD_HTTP", raising=False)
    monkeypatch.delenv("REBOT_HOST", raising=False)
    c1 = ArmdSandboxClient()
    assert c1.base_url == "http://127.0.0.1:8765"

    monkeypatch.setenv("REBOT_HOST", "192.168.42.1")
    c2 = ArmdSandboxClient()
    assert c2.base_url == "http://192.168.42.1:8765"

    monkeypatch.setenv("REBOT_ARMD_HTTP", "http://remote.local:9000")
    c3 = ArmdSandboxClient()
    assert c3.base_url == "http://remote.local:9000"

    c4 = ArmdSandboxClient("10.0.0.5:8765")
    assert c4.base_url == "http://10.0.0.5:8765"


def test_coffee_kiosk_evaluation_stops_on_tracking_stall(monkeypatch: pytest.MonkeyPatch):
    from examples.scenes.coffee_kiosk import train_and_eval_demo
    from rebot import ActionFailed

    reset_calls: list[int] = []
    episode_calls: list[int] = []

    class FakeSim:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def load_scene(self, scene: str, tool: str, seed: int):
            return {
                "scene": scene,
                "tool": tool,
                "task": {"stages": {"cup_grasped": False, "cup_on_tray": False}},
            }

        def reset(self, seed: int):
            reset_calls.append(seed)
            return {"task": {"stages": {}}}

        def state(self):
            return {"task": {"completed": False, "stages": {}}}

    class FakeArm:
        def __init__(self, host: str):
            self.host = host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def guard(self, timeout: float):
            pass

        def feed(self):
            pass

    def fake_run_episode(arm: Any, use_vision: bool, speed_scale: float):
        episode_calls.append(len(episode_calls))
        raise ActionFailed(code="TRACKING_STALL", message="arm tracking stalled")

    monkeypatch.setattr(train_and_eval_demo, "SimSceneClient", FakeSim)
    monkeypatch.setattr(train_and_eval_demo, "Arm", FakeArm)
    monkeypatch.setattr(train_and_eval_demo, "run_episode", fake_run_episode)

    ret = train_and_eval_demo.main(["--episodes", "5", "--seed-start", "10"])
    assert ret == 1
    assert reset_calls == [10]  # Evaluation terminated after seed 10, did not run later seeds
    assert len(episode_calls) == 1


def test_coffee_kiosk_evaluation_continues_on_not_holding(monkeypatch: pytest.MonkeyPatch):
    from examples.scenes.coffee_kiosk import train_and_eval_demo
    from rebot import ActionFailed

    reset_calls: list[int] = []
    episode_calls: list[int] = []

    class FakeSim:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def load_scene(self, scene: str, tool: str, seed: int):
            return {
                "scene": scene,
                "tool": tool,
                "task": {"stages": {"cup_grasped": False, "cup_on_tray": False}},
            }

        def reset(self, seed: int):
            reset_calls.append(seed)
            return {"task": {"stages": {}}}

        def state(self):
            return {"task": {"completed": False, "stages": {}}}

    class FakeArm:
        def __init__(self, host: str):
            self.host = host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def guard(self, timeout: float):
            pass

        def feed(self):
            pass

    def fake_run_episode(arm: Any, use_vision: bool, speed_scale: float):
        episode_calls.append(len(episode_calls))
        raise ActionFailed(code="NOT_HOLDING", message="grasp missed")

    monkeypatch.setattr(train_and_eval_demo, "SimSceneClient", FakeSim)
    monkeypatch.setattr(train_and_eval_demo, "Arm", FakeArm)
    monkeypatch.setattr(train_and_eval_demo, "run_episode", fake_run_episode)

    ret = train_and_eval_demo.main(["--episodes", "3", "--seed-start", "0"])
    assert ret == 1
    assert reset_calls == [0, 1, 2]  # All seeds ran because NOT_HOLDING is expected
    assert len(episode_calls) == 3
