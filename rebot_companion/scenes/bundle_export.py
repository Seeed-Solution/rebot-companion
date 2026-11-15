"""Static Bundle Exporter for reBot Cyclops Playground.

Generates self-contained static playground bundles conforming to `rebot-playground-bundle/v1`
with pre-composed MJCF XML using armd's Rust composer (`armd.wasm` via Node or shared fallback),
embedded manifests, and zero hardcoded `/assets/wasm/` or `/api/v1/` URLs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .spec import (
    SceneManifest,
    ToolManifest,
    resolve_builtin_path,
    validate_scene_dir,
    validate_tool_dir,
)

# armd checkout (rebot-armd) used to find armd.wasm and the base MuJoCo scenes; unset = none.
ARMD_DIR_ENV = "REBOT_ARMD_DIR"
ARMD_WASM_CANDIDATES = (
    "target/wasm32-unknown-unknown/release/armd.wasm",
    "dist/armd.wasm",
    "target/wasm32-unknown-unknown/debug/armd.wasm",
)
BASE_XML_SUBDIR = "sim/mujoco/models"


def _armd_dir() -> Path | None:
    raw = os.environ.get(ARMD_DIR_ENV)
    return Path(raw).expanduser() if raw else None

NODE_WASM_RUNNER = r"""
const fs = require('fs');
const wasmPath = process.argv[1];

let inputData = '';
process.stdin.setEncoding('utf-8');
process.stdin.on('data', chunk => { inputData += chunk; });
process.stdin.on('end', () => {
  try {
    const wasmBytes = fs.readFileSync(wasmPath);
    const wasmModule = new WebAssembly.Module(wasmBytes);
    const instance = new WebAssembly.Instance(wasmModule, {
      env: {
        abort: () => {},
      }
    });
    const { armd_alloc, armd_call_json, armd_free_result, armd_dealloc, memory } = instance.exports;
    if (!armd_call_json || !armd_alloc || !memory) {
      throw new Error('armd.wasm missing required exports (armd_call_json, armd_alloc, memory)');
    }

    const reqBuf = Buffer.from(inputData, 'utf-8');
    const reqLen = reqBuf.length;
    const reqPtr = armd_alloc(reqLen);
    const memView = new Uint8Array(memory.buffer);
    memView.set(reqBuf, reqPtr);

    const packed = armd_call_json(reqPtr, reqLen);
    const resPtr = Number(BigInt(packed) >> 32n);
    const resLen = Number(BigInt(packed) & 0xFFFFFFFFn);

    const resBytes = Buffer.from(memory.buffer, resPtr, resLen);
    const resText = resBytes.toString('utf-8');

    if (armd_free_result) {
      armd_free_result(resPtr, resLen);
    }
    if (armd_dealloc) {
      armd_dealloc(reqPtr, reqLen);
    }

    process.stdout.write(resText);
  } catch (err) {
    process.stderr.write(String(err && err.stack ? err.stack : err));
    process.exit(1);
  }
});
"""


def resolve_armd_wasm(explicit_path: str | Path | None = None) -> Path | None:
    """Find armd.wasm binary on disk or return None if not yet compiled."""
    if explicit_path is not None:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified armd.wasm not found: {p}")

    env_path = os.environ.get("REBOT_ARMD_WASM")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p

    armd_dir = _armd_dir()
    if armd_dir is None:
        return None
    for rel in ARMD_WASM_CANDIDATES:
        c = armd_dir / rel
        if c.is_file():
            return c
    return None


def compose_mjcf_fallback(
    base_xml: str,
    scene_world_xml: str | None = None,
    tool_manifest: dict[str, Any] | ToolManifest | None = None,
    tool_xml: str | None = None,
    tool_preset: str | None = None,
) -> str:
    """Pure-Python fallback composition strictly matching armd Rust mjcf_compose."""
    root = ET.fromstring(base_xml)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # 1. Remove cube if scene is active
    if scene_world_xml and scene_world_xml.strip():
        for body in list(worldbody.findall("body")):
            if body.get("name") == "cube":
                worldbody.remove(body)
        scene_root = ET.fromstring(scene_world_xml)
        scene_wb = scene_root.find("worldbody") if scene_root.tag != "worldbody" else scene_root
        if scene_wb is not None:
            for child in list(scene_wb):
                worldbody.append(child)

    # 2. Tool composition
    tm = tool_manifest.to_dict() if isinstance(tool_manifest, ToolManifest) else tool_manifest
    tool_id = (tm or {}).get("id") or tool_preset or "default"
    if tool_id != "default" and tool_xml and tool_xml.strip():
        tool_root = ET.fromstring(tool_xml)

        # Elements to replace
        replace_bodies = set((tm or {}).get("replace_bodies", ["finger_left", "finger_right"]))
        replace_geoms = set((tm or {}).get("replace_geoms", ["palm", "motor_housing"]))
        replace_equalities = set((tm or {}).get("replace_equalities", ["jaw_coupling"]))
        replace_actuators = set((tm or {}).get("replace_actuators", ["m7"]))

        # Remove replaced elements from base
        attach_body_name = (tm or {}).get("attach_body", "end_link")
        attach_node = None
        for elem in root.iter():
            if elem.tag == "body" and elem.get("name") == attach_body_name:
                attach_node = elem
            for child in list(elem):
                if child.tag == "body" and child.get("name") in replace_bodies:
                    elem.remove(child)
                elif child.tag == "geom" and child.get("name") in replace_geoms:
                    elem.remove(child)
                elif elem.tag == "equality" and child.get("name") in replace_equalities:
                    elem.remove(child)
                elif elem.tag == "actuator" and child.get("name") in replace_actuators:
                    elem.remove(child)

        # Inject tool elements
        tool_wb = tool_root.find("worldbody") if tool_root.tag != "worldbody" else tool_root
        if tool_wb is not None and attach_node is not None:
            for child in list(tool_wb):
                attach_node.append(child)

        tool_act = tool_root.find("actuator")
        if tool_act is not None:
            base_act = root.find("actuator")
            if base_act is None:
                base_act = ET.SubElement(root, "actuator")
            for child in list(tool_act):
                base_act.append(child)

        tool_eq = tool_root.find("equality")
        if tool_eq is not None:
            base_eq = root.find("equality")
            if base_eq is None:
                base_eq = ET.SubElement(root, "equality")
            for child in list(tool_eq):
                base_eq.append(child)

        tool_asset = tool_root.find("asset")
        if tool_asset is not None:
            base_asset = root.find("asset")
            if base_asset is None:
                base_asset = ET.SubElement(root, "asset")
            for child in list(tool_asset):
                base_asset.append(child)

    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def compose_mjcf_via_wasm(
    wasm_path: str | Path | None = None,
    base_xml: str | None = None,
    scene_world_xml: str | None = None,
    tool_manifest: dict[str, Any] | ToolManifest | None = None,
    tool_xml: str | None = None,
    tool_preset: str | None = None,
    allow_fallback: bool = False,
) -> str:
    """Compose complete MJCF model XML using armd.wasm via Node.js (or fallback)."""
    resolved_wasm = resolve_armd_wasm(wasm_path)

    if resolved_wasm is None:
        if allow_fallback and base_xml:
            return compose_mjcf_fallback(
                base_xml=base_xml,
                scene_world_xml=scene_world_xml,
                tool_manifest=tool_manifest,
                tool_xml=tool_xml,
                tool_preset=tool_preset,
            )
        where = wasm_path or f"${ARMD_DIR_ENV} ({os.environ.get(ARMD_DIR_ENV) or 'unset'})"
        raise FileNotFoundError(f"armd.wasm not found at {where}")

    node_bin = shutil.which("node")
    if not node_bin:
        if allow_fallback and base_xml:
            return compose_mjcf_fallback(
                base_xml=base_xml,
                scene_world_xml=scene_world_xml,
                tool_manifest=tool_manifest,
                tool_xml=tool_xml,
                tool_preset=tool_preset,
            )
        raise RuntimeError("Node.js ('node') executable not found in PATH")

    tm = tool_manifest.to_dict() if isinstance(tool_manifest, ToolManifest) else tool_manifest
    req: dict[str, Any] = {
        "op": "compose_mjcf",
        "base_xml": base_xml,
        "scene_world_xml": scene_world_xml,
        "tool_manifest": tm,
        "tool_xml": tool_xml,
        "tool_preset": tool_preset,
    }

    proc = subprocess.run(
        [node_bin, "-e", NODE_WASM_RUNNER, str(resolved_wasm)],
        input=json.dumps(req, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or f"Node exited with code {proc.returncode}"
        raise RuntimeError(f"armd.wasm compose_mjcf invocation failed: {err}")

    try:
        reply = json.loads(proc.stdout)
    except Exception as exc:
        raise RuntimeError(f"armd.wasm returned non-JSON output: {proc.stdout!r}") from exc

    if not reply.get("ok"):
        code = reply.get("code", "ERROR")
        msg = reply.get("message", "MJCF composition failed")
        raise RuntimeError(f"armd compose_mjcf error [{code}]: {msg}")

    return str(reply["xml"])


def export_static_bundle(
    out_dir: str | Path,
    *,
    scene_id: str | None = None,
    tool_id: str | None = None,
    policy_dir: str | Path | None = None,
    robot: str = "b601dm",
    wasm_path: str | Path | None = None,
    base_xml: str | None = None,
    overwrite: bool = True,
    allow_fallback: bool = True,
) -> Path:
    """Export a self-contained static playground bundle conforming to rebot-playground-bundle/v1."""
    dest = Path(out_dir).resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Output directory {dest} already exists and overwrite is False")
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Resolve scene manifest & world XML
    scene_manifest: SceneManifest | None = None
    scene_world_xml: str | None = None
    eff_scene_id = scene_id or "default"

    if eff_scene_id not in ("default", "none"):
        p = Path(eff_scene_id).expanduser()
        if p.is_dir() and (p / "scene.json").is_file():
            scene_manifest = validate_scene_dir(p)
        else:
            builtin_p = resolve_builtin_path("scene", eff_scene_id)
            if builtin_p is not None:
                scene_manifest = validate_scene_dir(builtin_p)
            else:
                raise FileNotFoundError(f"Scene pack not found: {eff_scene_id!r}")
        eff_scene_id = scene_manifest.id
        if scene_manifest.source_dir:
            scene_world_xml = (scene_manifest.source_dir / scene_manifest.world_xml).read_text(encoding="utf-8")

    # 2. Resolve tool manifest & tool XML
    eff_tool_id = tool_id or (scene_manifest.default_tool if scene_manifest else "default")
    tool_manifest: ToolManifest | None = None
    tool_xml: str | None = None
    tool_preset: str | None = None

    if eff_tool_id not in ("default", "none"):
        tp = Path(eff_tool_id).expanduser()
        if tp.is_dir() and (tp / "tool.json").is_file():
            tool_manifest = validate_tool_dir(tp)
            eff_tool_id = tool_manifest.id
        else:
            builtin_tp = resolve_builtin_path("tool", eff_tool_id)
            if builtin_tp is not None:
                tool_manifest = validate_tool_dir(builtin_tp)
                eff_tool_id = tool_manifest.id
            else:
                tool_preset = eff_tool_id

        if tool_manifest and tool_manifest.source_dir and tool_manifest.tool_xml:
            tool_xml = (tool_manifest.source_dir / tool_manifest.tool_xml).read_text(encoding="utf-8")
    else:
        tool_preset = "default"

    # 3. Base robot XML
    base_xml_content = base_xml
    if base_xml_content is None:
        robot_model_file = "so_arm101_scene.xml" if robot == "so-arm101" else "rebot_b601dm_scene.xml"
        armd_dir = _armd_dir()
        candidate = armd_dir / BASE_XML_SUBDIR / robot_model_file if armd_dir is not None else None
        if candidate is not None and candidate.is_file():
            base_xml_content = candidate.read_text(encoding="utf-8")
        else:
            # Fallback bare minimal XML
            base_xml_content = (
                f'<mujoco model="{robot}"><worldbody><body name="end_link" pos="0 0 0.2"></body>'
                '<body name="cube" pos="0.25 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>'
                "</worldbody></mujoco>"
            )

    if robot == "so-arm101":
        base_xml_content = base_xml_content.replace('meshdir="../../../assets/so101"', 'meshdir="so101"')

    # 4. Compose MJCF via armd.wasm (or fallback)
    mjcf_xml = compose_mjcf_via_wasm(
        wasm_path=wasm_path,
        base_xml=base_xml_content,
        scene_world_xml=scene_world_xml,
        tool_manifest=tool_manifest,
        tool_xml=tool_xml,
        tool_preset=tool_preset,
        allow_fallback=allow_fallback,
    )

    if robot == "so-arm101":
        mjcf_xml = mjcf_xml.replace('meshdir="../../../assets/so101"', 'meshdir="so101"')

    mesh_files: list[str] = sorted(set(re.findall(r'<mesh\s+[^>]*file="([^"]+)"', base_xml_content)))

    # 5. Build tool packs dictionary for runtime browser hot-swapping
    tool_packs: dict[str, Any] = {"default": {"tool_preset": "default"}}
    candidate_tools: set[str] = set()
    if scene_manifest and scene_manifest.recommended_tools:
        candidate_tools.update(scene_manifest.recommended_tools)
    if robot == "b601dm":
        candidate_tools.update(["wide_cup_gripper", "suction", "screwdriver"])
    if eff_tool_id not in ("default", "none"):
        candidate_tools.add(eff_tool_id)

    candidate_tools.discard("default")
    candidate_tools.discard("none")

    for t_name in sorted(candidate_tools):
        t_manifest: ToolManifest | None = None
        p = Path(t_name).expanduser()
        if p.is_dir() and (p / "tool.json").is_file():
            t_manifest = validate_tool_dir(p)
        else:
            bp = resolve_builtin_path("tool", t_name)
            if bp is not None:
                t_manifest = validate_tool_dir(bp)

        if t_manifest is not None and t_manifest.source_dir and t_manifest.tool_xml:
            t_xml_text = (t_manifest.source_dir / t_manifest.tool_xml).read_text(encoding="utf-8")
            tool_packs[t_name] = {
                "tool_manifest": t_manifest.to_dict(),
                "tool_xml": t_xml_text,
            }

    # 6. Handle optional policy
    policy_manifest_dict: dict[str, Any] | None = None
    model_onnx_rel: str | None = None
    if policy_dir is not None:
        from ..onnx_policy import validate_policy_dir

        pdir = Path(policy_dir).resolve()
        policy_m = validate_policy_dir(pdir)
        policy_manifest_dict = policy_m.to_dict()
        model_src = pdir / policy_m.model_file
        model_dest = dest / "model.onnx"
        shutil.copy2(model_src, model_dest)
        model_onnx_rel = "./model.onnx"
        policy_manifest_dict["model_onnx_rel"] = model_onnx_rel

    bundle_id = f"bundle_{eff_scene_id}_{eff_tool_id}_{robot}"
    bundle_data: dict[str, Any] = {
        "format": "rebot-playground-bundle/v1",
        "bundle_id": bundle_id,
        "robot": robot,
        "scene_id": eff_scene_id,
        "tool_id": eff_tool_id,
        "base_xml": base_xml_content,
        "scene_world_xml": scene_world_xml,
        "mjcf_xml": mjcf_xml,
        "tool_packs": tool_packs,
        "scene_manifest": scene_manifest.to_dict() if scene_manifest else None,
        "tool_manifest": tool_manifest.to_dict() if tool_manifest else None,
        "policy_manifest": policy_manifest_dict,
        "model_onnx_rel": model_onnx_rel,
    }
    if robot == "so-arm101":
        bundle_data["mesh_files"] = mesh_files

    bundle_json_str = json.dumps(bundle_data, indent=2, ensure_ascii=False)

    # Hard invariant: zero hardcoded URL paths
    if "/assets/wasm/" in bundle_json_str or "/api/v1/" in bundle_json_str:
        raise ValueError("Static bundle must not contain hardcoded /assets/wasm/ or /api/v1/ URLs")

    (dest / "bundle.json").write_text(bundle_json_str, encoding="utf-8")
    return dest
