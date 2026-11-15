"""Deploy Profile Validator, Safe Applier, and Python 3.8 main.py Generator.

Schema: `rebot-deploy-profile/v1`.
Validates motion limits, keepout boundaries, pose definitions, safety configuration,
and renders stdlib-only Python 3.8 executable scripts for real arms and simulation.
"""

from __future__ import annotations

import ast
import json
import math
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rebot.client import ArmdClient, ArmSession

DEPLOY_PROFILE_FORMAT = "rebot-deploy-profile/v1"
SUPPORTED_ROBOTS = ("b601dm", "so-arm101")
POSE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_KEEPOUT_BOXES = 16
MAX_CYCLES = 1000
MAX_LINEAR_SPEED_MPS = 0.15
MIN_LINEAR_SPEED_MPS = 0.005


@dataclass
class KeepoutBox:
    name: str
    min_m: list[float]
    max_m: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min_m": list(self.min_m), "max_m": list(self.max_m)}


@dataclass
class DeployProfile:
    format: str
    robot: str
    scene_id: str
    tool_id: str
    poses: dict[str, list[float]]
    keepout_boxes: list[dict[str, Any]]
    motion: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    program: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "robot": self.robot,
            "scene_id": self.scene_id,
            "tool_id": self.tool_id,
            "poses": {k: list(v) for k, v in self.poses.items()},
            "keepout_boxes": list(self.keepout_boxes),
            "motion": dict(self.motion),
            "safety": dict(self.safety),
            "program": dict(self.program),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p


def validate_deploy_profile(raw: dict[str, Any], source_path: Path | None = None) -> DeployProfile:
    """Validate a raw dictionary against the rebot-deploy-profile/v1 specification."""
    if not isinstance(raw, dict):
        raise ValueError(f"Deploy profile must be a JSON object, got {type(raw).__name__}")

    fmt = raw.get("format")
    if fmt != DEPLOY_PROFILE_FORMAT:
        raise ValueError(f"Invalid format {fmt!r}; expected {DEPLOY_PROFILE_FORMAT!r}")

    robot = str(raw.get("robot") or "").strip().lower()
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of {SUPPORTED_ROBOTS}")

    scene_id = str(raw.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError("Deploy profile must specify a non-empty 'scene_id'")

    tool_id = str(raw.get("tool_id") or "").strip()
    if not tool_id:
        raise ValueError("Deploy profile must specify a non-empty 'tool_id'")

    # Poses validation
    raw_poses = raw.get("poses")
    if not isinstance(raw_poses, dict) or not raw_poses:
        raise ValueError("Deploy profile must specify non-empty 'poses' mapping")

    expected_dof = 6 if robot == "so-arm101" else 7
    validated_poses: dict[str, list[float]] = {}
    for name, positions in raw_poses.items():
        if not isinstance(name, str) or not POSE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid pose name {name!r}; must match ^[A-Za-z0-9_-]{{1,64}}$"
            )
        if not isinstance(positions, (list, tuple)):
            raise ValueError(f"Pose {name!r} positions must be a list of floats, got {type(positions).__name__}")
        if len(positions) != expected_dof:
            raise ValueError(f"Pose {name!r} has {len(positions)} joints; expected {expected_dof}")
        float_positions: list[float] = []
        for val in positions:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Pose {name!r} contains non-finite joint value: {val!r}")
            float_positions.append(float(val))
        validated_poses[name] = float_positions

    # Keepout boxes validation (with fallback to safety.keepout_boxes)
    raw_boxes = raw.get("keepout_boxes")
    if raw_boxes is None and isinstance(raw.get("safety"), dict):
        raw_boxes = raw["safety"].get("keepout_boxes")
    if raw_boxes is None:
        raw_boxes = []
    if not isinstance(raw_boxes, list):
        raise ValueError("keepout_boxes must be a list")
    if len(raw_boxes) > MAX_KEEPOUT_BOXES:
        raise ValueError(f"Too many keepout boxes ({len(raw_boxes)}); max is {MAX_KEEPOUT_BOXES}")

    validated_boxes: list[dict[str, Any]] = []
    for idx, b in enumerate(raw_boxes):
        if not isinstance(b, dict):
            raise ValueError(f"keepout_box[{idx}] must be a dictionary")
        box_name = str(b.get("name") or "").strip()
        if not box_name or len(box_name) > 64:
            raise ValueError(f"keepout_box[{idx}] has invalid name: {box_name!r}")

        min_coords = b.get("min_m") if "min_m" in b else b.get("min")
        max_coords = b.get("max_m") if "max_m" in b else b.get("max")
        if not isinstance(min_coords, (list, tuple)) or len(min_coords) != 3:
            raise ValueError(f"keepout_box {box_name!r} min_m must have 3 coordinates")
        if not isinstance(max_coords, (list, tuple)) or len(max_coords) != 3:
            raise ValueError(f"keepout_box {box_name!r} max_m must have 3 coordinates")

        min_f: list[float] = []
        max_f: list[float] = []
        for i in range(3):
            v_min, v_max = min_coords[i], max_coords[i]
            if not isinstance(v_min, (int, float)) or not math.isfinite(v_min):
                raise ValueError(f"keepout_box {box_name!r} min_m[{i}] is not a finite number: {v_min!r}")
            if not isinstance(v_max, (int, float)) or not math.isfinite(v_max):
                raise ValueError(f"keepout_box {box_name!r} max_m[{i}] is not a finite number: {v_max!r}")
            f_min, f_max = float(v_min), float(v_max)
            if f_min >= f_max:
                raise ValueError(
                    f"Inverted keepout box {box_name!r} along axis {i}: min ({f_min}) >= max ({f_max})"
                )
            min_f.append(f_min)
            max_f.append(f_max)

        validated_boxes.append({"name": box_name, "min_m": min_f, "max_m": max_f})

    # Motion limits
    motion = dict(raw.get("motion") or {})
    speed_scale = float(motion.get("speed_scale", 0.3))
    if not 0.1 <= speed_scale <= 1.0 or not math.isfinite(speed_scale):
        raise ValueError(f"speed_scale ({speed_scale}) must be within 0.1..1.0")
    motion["speed_scale"] = speed_scale

    linear_speed = float(motion.get("linear_speed_mps", 0.04))
    if not MIN_LINEAR_SPEED_MPS <= linear_speed <= MAX_LINEAR_SPEED_MPS or not math.isfinite(linear_speed):
        raise ValueError(
            f"linear_speed_mps ({linear_speed}) must be in {MIN_LINEAR_SPEED_MPS}..{MAX_LINEAR_SPEED_MPS}"
        )
    motion["linear_speed_mps"] = linear_speed

    # Safety settings
    safety = dict(raw.get("safety") or {})
    col_guard = bool(safety.get("collision_guard_enabled", False))
    col_sens = str(safety.get("collision_sensitivity", "medium")).lower()
    if col_sens not in ("low", "medium", "high"):
        raise ValueError(f"Invalid collision_sensitivity {col_sens!r}; expected low, medium, or high")
    safety["collision_guard_enabled"] = col_guard
    safety["collision_sensitivity"] = col_sens

    # Program settings
    program = dict(raw.get("program") or {})
    cycles = int(program.get("cycles", 1))
    if not 1 <= cycles <= MAX_CYCLES:
        raise ValueError(f"cycles ({cycles}) must be in 1..{MAX_CYCLES}")
    dwell_s = float(program.get("dwell_s", 0.5))
    if dwell_s < 0.0 or not math.isfinite(dwell_s):
        raise ValueError(f"dwell_s ({dwell_s}) must be non-negative and finite")

    vp_z_raw = program.get("via_point_z", 0.11)
    if isinstance(vp_z_raw, bool) or not isinstance(vp_z_raw, (int, float)):
        try:
            via_point_z = float(vp_z_raw)
        except (ValueError, TypeError):
            raise ValueError(f"via_point_z ({vp_z_raw!r}) must be a finite float")
    else:
        via_point_z = float(vp_z_raw)
    if not math.isfinite(via_point_z):
        raise ValueError(f"via_point_z ({via_point_z}) must be finite")

    program["cycles"] = cycles
    program["dwell_s"] = dwell_s
    program["via_point_z"] = via_point_z
    program["ng_sorting"] = bool(program.get("ng_sorting", False))
    program["policy_enabled"] = bool(program.get("policy_enabled", False))
    program["io_actions"] = list(program.get("io_actions", []))

    return DeployProfile(
        format=DEPLOY_PROFILE_FORMAT,
        robot=robot,
        scene_id=scene_id,
        tool_id=tool_id,
        poses=validated_poses,
        keepout_boxes=validated_boxes,
        motion=motion,
        safety=safety,
        program=program,
        source_path=source_path,
    )


def render_main_py(profile: DeployProfile) -> str:
    """Generate a clean, standalone Python 3.8 stdlib-only `main.py` script."""
    # Enforce first-run speed safety clamp
    speed_scale = min(float(profile.motion.get("speed_scale", 0.3)), 0.3)
    linear_speed_mps = min(float(profile.motion.get("linear_speed_mps", 0.04)), 0.04)
    max_cycles = int(profile.program.get("cycles", 1))
    dwell_s = float(profile.program.get("dwell_s", 0.5))
    via_point_z = float(profile.program.get("via_point_z", 0.11))
    ng_sorting = bool(profile.program.get("ng_sorting", False))
    policy_enabled = bool(profile.program.get("policy_enabled", False))

    scene_id_lower = str(profile.scene_id or "").lower()
    tool_id_lower = str(profile.tool_id or "").lower()
    robot_lower = str(profile.robot or "").lower()
    is_so101 = robot_lower == "so-arm101" or "so101" in scene_id_lower
    auto_yaw_param = ", auto_yaw=True" if is_so101 else ""

    is_coffee = any(k in scene_id_lower for k in ("coffee", "level1", "level2", "speed"))
    is_brew = (
        scene_id_lower in ("level1_coffee", "template_coffee")
        or (is_coffee and "speed" not in scene_id_lower)
    )
    is_pcb = any(k in scene_id_lower for k in ("pcb", "inspection", "level3", "level4", "level5"))
    is_inspection = (
        scene_id_lower in ("level4_inspection", "template_pcb")
        or "inspection" in scene_id_lower
    )
    is_policy = (
        "level5" in scene_id_lower
        or policy_enabled
    )
    is_screwdriver = "screwdriver" in scene_id_lower or tool_id_lower == "screwdriver"
    is_so101_pick = is_so101 or "so101" in scene_id_lower

    if is_coffee:
        brew_block = ""
        if is_brew:
            brew_block = f'''
    # 3. Brew button press
    run_step(arm, "Move above brew button", lambda: arm.move_to((0.25, 0.17, 0.145), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Close gripper for plunger", lambda: arm.gripper_close())
    run_step(arm, "Depress brew button", lambda: arm.move_to((0.25, 0.17, 0.093), speed=linear_speed{auto_yaw_param}))
    time.sleep(0.25)
    sim_io_url = str(CONFIG.get("sim_io_url") or "").rstrip("/")
    if sim_io_url or os.environ.get("REBOT_ALLOW_DRY_IO") == "1":
        set_tool_io("brew_button_pressed", True)
        set_tool_io("brew_led", True)
    run_step(arm, "Retract above brew button", lambda: arm.move_to((0.25, 0.17, 0.145), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Open gripper", lambda: arm.gripper_open())
'''
        cycle_body = f'''    # 1. Grasp cup from pickup table
    run_step(arm, "Move to scan pose", lambda: arm.move_to_pose("scan", speed_scale=speed_scale))
    run_step(arm, "Open gripper", lambda: arm.gripper_open())
    run_step(arm, "Approach cup above", lambda: arm.move_to((0.27, -0.10, 0.13), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Descend to cup", lambda: arm.move_to((0.27, -0.10, 0.042), speed=linear_speed{auto_yaw_param}))
    def _grasp_cup():
        res = arm.gripper_close()
        if not getattr(res, "holding", False):
            raise ActionFailed("NOT_HOLDING", "Gripper closed but cup workpiece not holding")
    run_step(arm, "Close gripper on cup", _grasp_cup)
    run_step(arm, "Lift cup", lambda: arm.move_to((0.27, -0.10, 0.13), speed=linear_speed{auto_yaw_param}))

    # 2. Transit to drip tray [0.315, 0.085]
    run_step(arm, "Transit intermediate", lambda: arm.move_to((0.25, 0.0, 0.13), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Approach drip tray", lambda: arm.move_to((0.315, 0.085, 0.115), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Lower cup onto drip tray", lambda: arm.move_to((0.315, 0.085, 0.050), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Release cup on tray", lambda: arm.gripper_open())
    run_step(arm, "Retract above tray", lambda: arm.move_to((0.315, 0.085, 0.115), speed=linear_speed{auto_yaw_param}))
{brew_block}
    # 4. Regrasp cup from drip tray
    run_step(arm, "Approach drip tray for regrasp", lambda: arm.move_to((0.315, 0.085, 0.115), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Descend onto cup at tray", lambda: arm.move_to((0.315, 0.085, 0.050), speed=linear_speed{auto_yaw_param}))
    def _regrasp_cup():
        res = arm.gripper_close()
        if not getattr(res, "holding", False):
            raise ActionFailed("NOT_HOLDING", "Failed to regrasp cup from drip tray")
    run_step(arm, "Regrasp cup", _regrasp_cup)
    run_step(arm, "Lift cup from tray", lambda: arm.move_to((0.315, 0.085, 0.115), speed=linear_speed{auto_yaw_param}))

    # 5. Place at pickup counter [0.25, -0.16]
    run_step(arm, "Transit to counter", lambda: arm.move_to((0.25, 0.0, 0.13), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Approach pickup counter", lambda: arm.move_to((0.25, -0.16, 0.115), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Lower cup to counter", lambda: arm.move_to((0.25, -0.16, 0.048), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Release cup at counter", lambda: arm.gripper_open())
    run_step(arm, "Retract to home", lambda: arm.move_to_pose("home", speed_scale=speed_scale))'''

    elif is_pcb:
        policy_block = ""
        if is_policy:
            policy_block = f'''
    # Policy guidance / detection check
    print("  [POLICY] Closed-loop ONNX vision guidance: run via 'rebot-policy --policy onnx:...'")
    try:
        detections = arm.detect(min_score=0.4)
        if detections:
            print("  [POLICY] Detected " + str(len(detections)) + " workpiece(s): best=" + str(detections[0].label) + " (" + str(round(detections[0].score, 2)) + ")")
    except Exception as det_err:
        print("  [POLICY] Vision detection check: " + str(det_err))
'''
        if is_inspection:
            inspection_block = f'''
    # 2. Place on test fixture and dwell
    run_step(arm, "Approach test fixture", lambda: arm.move_to((0.27, 0.01, via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Place on fixture", lambda: arm.move_to((0.27, 0.01, 0.030), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Release suction on fixture", lambda: set_tool_io("vacuum", False))
    if dwell_s > 0.0:
        time.sleep(dwell_s)

    # Query fixture test pass / fail
    sim_io_url = str(CONFIG.get("sim_io_url") or "").rstrip("/")
    if sim_io_url:
        io_state = get_sim_io()
        val = io_state.get("fixture_test_pass")
        is_ng = (val is False) or (str(val).lower() in ("false", "0", "ng"))
    else:
        env_res = os.environ.get("REBOT_FIXTURE_RESULT", "").strip().upper()
        if not env_res:
            raise RuntimeError(
                "Inspection fixture result missing: neither REBOT_SIM_IO_URL nor REBOT_FIXTURE_RESULT is set"
            )
        if env_res not in ("PASS", "NG"):
            raise RuntimeError("Unknown REBOT_FIXTURE_RESULT '" + env_res + "': must be 'PASS' or 'NG'")
        is_ng = (env_res == "NG")

    run_step(arm, "Regrasp tested PCB", lambda: set_tool_io("vacuum", True))
    run_step(arm, "Lift tested PCB", lambda: arm.move_to((0.27, 0.01, via_z), speed=linear_speed{auto_yaw_param}))

    if is_ng and not ng_sorting:
        raise ActionFailed(
            "INVALID_COMMAND",
            "Inspection fixture detected defective board (NG) but ng_sorting is False; stopping cycle",
        )

    if is_ng and ng_sorting:
        dest = (0.16, 0.14, 0.030)
        print("  -> Routing NG board to NG bin (0.16, 0.14)...")
    else:
        dest = (0.26, 0.14, 0.030)
        print("  -> Routing PASS board to PASS bin (0.26, 0.14)...")

    run_step(arm, "Transit to destination bin", lambda: arm.move_to((dest[0], dest[1], via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Lower into bin", lambda: arm.move_to(dest, speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Release into bin", lambda: set_tool_io("vacuum", False))
    run_step(arm, "Retract above bin", lambda: arm.move_to((dest[0], dest[1], via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Retract to home", lambda: arm.move_to_pose("home", speed_scale=speed_scale))'''
        else:
            inspection_block = f'''
    # Standard transfer across divider wall
    run_step(arm, "Transit across divider", lambda: arm.move_to((0.27, 0.01, via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Lower to target", lambda: arm.move_to((0.27, 0.01, 0.030), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Release vacuum", lambda: set_tool_io("vacuum", False))
    run_step(arm, "Retract above target", lambda: arm.move_to((0.27, 0.01, via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Retract to home", lambda: arm.move_to_pose("home", speed_scale=speed_scale))'''

        cycle_body = f'''    # 1. Approach infeed with divider clearance via_z
    run_step(arm, "Move to scan pose", lambda: arm.move_to_pose("scan", speed_scale=speed_scale))
    run_step(arm, "Approach infeed above divider", lambda: arm.move_to((0.23, -0.14, via_z), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Descend to PCB", lambda: arm.move_to((0.23, -0.14, 0.018), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Engage suction", lambda: set_tool_io("vacuum", True))
    run_step(arm, "Lift over divider", lambda: arm.move_to((0.23, -0.14, via_z), speed=linear_speed{auto_yaw_param}))
{policy_block}{inspection_block}'''

    elif is_screwdriver:
        cycle_body = f'''    run_step(arm, "Move to scan pose", lambda: arm.move_to_pose("scan", speed_scale=speed_scale))

    # Fasten screw 1 at (0.23, -0.025)
    run_step(arm, "Approach screw 1", lambda: arm.move_to((0.23, -0.025, 0.09), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Descend to screw 1", lambda: arm.move_to((0.23, -0.025, 0.022), speed=linear_speed{auto_yaw_param}))
    set_tool_io("spindle_rpm", 300)
    time.sleep(0.5)
    set_tool_io("spindle_rpm", 0)
    run_step(arm, "Retract from screw 1", lambda: arm.move_to((0.23, -0.025, 0.09), speed=linear_speed{auto_yaw_param}))

    # Fasten screw 2 at (0.27, 0.025)
    run_step(arm, "Approach screw 2", lambda: arm.move_to((0.27, 0.025, 0.09), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Descend to screw 2", lambda: arm.move_to((0.27, 0.025, 0.022), speed=linear_speed{auto_yaw_param}))
    set_tool_io("spindle_rpm", 300)
    time.sleep(0.5)
    set_tool_io("spindle_rpm", 0)
    run_step(arm, "Retract from screw 2", lambda: arm.move_to((0.27, 0.025, 0.09), speed=linear_speed{auto_yaw_param}))
    run_step(arm, "Retract to home", lambda: arm.move_to_pose("home", speed_scale=speed_scale))'''

    elif is_so101_pick:
        cycle_body = f'''    run_step(arm, "Move to scan pose", lambda: arm.move_to_pose("scan", speed_scale=speed_scale))
    run_step(arm, "Open gripper", lambda: arm.gripper_open())
    run_step(arm, "Approach cube above", lambda: arm.move_to((0.22, -0.05, 0.09), speed=linear_speed, auto_yaw=True))
    run_step(arm, "Descend to cube", lambda: arm.move_to((0.22, -0.05, 0.026), speed=linear_speed, auto_yaw=True))
    def _grasp_cube():
        res = arm.gripper_close()
        if not getattr(res, "holding", False):
            raise ActionFailed("NOT_HOLDING", "SO-ARM101 parallel jaws missed cube")
    run_step(arm, "Close gripper on cube", _grasp_cube)
    run_step(arm, "Lift cube", lambda: arm.move_to((0.22, -0.05, 0.09), speed=linear_speed, auto_yaw=True))
    run_step(arm, "Transit to place pose", lambda: arm.move_to((0.22, 0.08, 0.09), speed=linear_speed, auto_yaw=True))
    run_step(arm, "Lower cube into place", lambda: arm.move_to((0.22, 0.08, 0.035), speed=linear_speed, auto_yaw=True))
    run_step(arm, "Open gripper to release cube", lambda: arm.gripper_open())
    run_step(arm, "Retract to home", lambda: arm.move_to_pose("home", speed_scale=speed_scale))'''

    else:
        if not profile.poses:
            raise RuntimeError("Deploy profile contains no poses for custom cycle")
        cycle_body = f'''    if not poses:
        raise RuntimeError("No taught poses available for custom cycle")
    for pose_name in poses:
        run_step(arm, "Move to pose " + pose_name, lambda p=pose_name: arm.move_to_pose(p, speed_scale=speed_scale))
        if dwell_s > 0.0:
            time.sleep(dwell_s)'''

    code = f'''#!/usr/bin/env python3
"""Auto-generated reBot companion application script for {profile.scene_id}.

Safety constraints:
- Speed scaled down for safe initial execution (speed_scale <= 0.3).
- Bounded loop cycles with watchdog feeding.
- Halts and returns 2 on arm faults without auto-recovery; operator inspects workspace and clears in web Cockpit.
- Uses Python 3.8 standard library and `rebot` SDK only.
"""

from typing import Any, Dict, List, Optional
import json
import os
import sys
import time
import urllib.error
import urllib.request

from rebot import ActionFailed, Arm, ArmFaultError, RebotError

SCENE_ID = {profile.scene_id!r}
TOOL_ID = {profile.tool_id!r}

CONFIG = {{
    "host": os.environ.get("REBOT_HOST", "127.0.0.1"),
    "speed_scale": {speed_scale!r},
    "linear_speed_mps": {linear_speed_mps!r},
    "max_cycles": {max_cycles!r},
    "dwell_s": {dwell_s!r},
    "via_point_z": {via_point_z!r},
    "ng_sorting": {ng_sorting!r},
    "policy_enabled": {policy_enabled!r},
    "poses": {profile.poses!r},
    "sim_io_url": os.environ.get("REBOT_SIM_IO_URL", ""),
    "is_so101": {is_so101!r},
    "guard_timeout_s": 15.0,
}}


def run_step(arm: Arm, label: str, fn: Any) -> Any:
    """Execute a single motion/action step, retrying only this step once on COLLISION."""
    arm.feed()
    print("  -> " + label + "...")
    try:
        return fn()
    except ActionFailed as exc:
        if exc.code == "COLLISION":
            print("  [COLLISION] Guard stopped at '" + label + "'. Waiting for workspace to clear...")
            arm.wait_until_clear()
            print("  Workspace clear, retrying step '" + label + "' once...")
            arm.feed()
            return fn()
        raise


def set_tool_io(name: str, value: Any) -> None:
    """Post tool I/O state to simulation I/O proxy or allow explicit dry-run."""
    sim_io_url = str(CONFIG.get("sim_io_url") or "").rstrip("/")
    if sim_io_url:
        target_url = sim_io_url + "/sim/io"
        payload = json.dumps({{name: value}}).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=payload,
            headers={{"Content-Type": "application/json"}},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status >= 400:
                    raise RuntimeError("Failed to set sim I/O channel: HTTP " + str(resp.status))
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError("Failed to connect to sim I/O at " + target_url + ": " + str(exc)) from exc
    elif os.environ.get("REBOT_ALLOW_DRY_IO") == "1":
        print("  [DRY_IO] I/O '" + name + "' set to " + str(value) + " (REBOT_ALLOW_DRY_IO=1)")
    else:
        raise RuntimeError(
            "Tool requires digital I/O '" + name + "', but neither REBOT_SIM_IO_URL nor REBOT_ALLOW_DRY_IO=1 is set. "
            "Set REBOT_SIM_IO_URL or set REBOT_ALLOW_DRY_IO=1 for dry simulation."
        )


def get_sim_io() -> Dict[str, Any]:
    """Query current simulation I/O states (GET /sim/io), raising RuntimeError on error."""
    sim_io_url = str(CONFIG.get("sim_io_url") or "").rstrip("/")
    if not sim_io_url:
        raise RuntimeError("REBOT_SIM_IO_URL is not configured; cannot query simulation I/O")
    target_url = sim_io_url + "/sim/io"
    req = urllib.request.Request(target_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status >= 400:
                raise RuntimeError("Failed to query sim I/O: HTTP " + str(resp.status))
            raw_body = resp.read()
            return json.loads(raw_body.decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("Failed to connect to sim I/O at " + target_url + ": " + str(exc)) from exc
    except Exception as exc:
        raise RuntimeError("Failed to parse sim I/O response: " + str(exc)) from exc


def move_cartesian(arm: Arm, xyz: List[float], speed: float) -> None:
    """Move straight-line to cartesian target, with auto_yaw on 5-DOF SO-ARM101."""
    if bool(CONFIG.get("is_so101")):
        arm.move_to(xyz, speed=speed, auto_yaw=True)
    else:
        arm.move_to(xyz, speed=speed)


def run_single_cycle(arm: Arm) -> bool:
    """Execute one complete cycle for scene."""
    speed_scale = float(CONFIG["speed_scale"])
    linear_speed = float(CONFIG["linear_speed_mps"])
    via_z = float(CONFIG["via_point_z"])
    dwell_s = float(CONFIG["dwell_s"])
    ng_sorting = bool(CONFIG["ng_sorting"])
    poses: Dict[str, List[float]] = CONFIG["poses"]  # type: ignore

{cycle_body}
    return True


def main() -> int:
    host = str(CONFIG["host"])
    speed = float(CONFIG["speed_scale"])
    max_cycles = int(CONFIG["max_cycles"])

    print("=== Starting reBot Companion Runner [speed_scale=" + str(speed) + "] ===")
    print("Connecting to arm on host " + host + " (force=False)...")

    with Arm(host=host, force=False) as arm:
        try:
            safety = arm.safety()
            if not safety.get("hardware", {{}}).get("physical_estop", True):
                print("WARNING: Arm has no physical E-stop. Software stops only; operator must be ready to cut power.")
        except RebotError as exc:
            print("  [error] safety check failed: " + str(exc))
            return 1

        arm.home()
        arm.guard(float(CONFIG.get("guard_timeout_s", 15.0)))

        for cycle in range(1, max_cycles + 1):
            print("\\n--- Cycle " + str(cycle) + "/" + str(max_cycles) + " ---")
            arm.feed()
            try:
                run_single_cycle(arm)
            except ActionFailed as exc:
                print("  [error] action failed (" + str(exc.code) + "): " + str(exc))
                return 1
            except ArmFaultError as exc:
                print("  [error] arm fault (" + str(exc.code) + "): " + str(exc))
                print("  Please inspect the workspace and clear the fault in the web Cockpit.")
                return 2

            time.sleep(0.5)

    print("\\n=== All planned cycles completed safely. Arm parked. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    ast.parse(code, filename="main.py", mode="exec", feature_version=(3, 8))
    return code


def apply_deploy_profile(
    profile_path_or_dict: str | Path | dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    client: ArmdClient | None = None,
    apply_changes: bool = False,
    backup_path: str | Path | None = None,
    out_script: str | Path | None = None,
    clear_keepout: bool = False,
    apply_collision_guard: bool = False,
) -> dict[str, Any]:
    """Validate and optionally apply deploy profile poses and keepout boxes to armd."""
    if isinstance(profile_path_or_dict, (str, Path)):
        p = Path(profile_path_or_dict).resolve()
        raw = json.loads(p.read_text(encoding="utf-8"))
        profile = validate_deploy_profile(raw, source_path=p)
    else:
        profile = validate_deploy_profile(profile_path_or_dict)

    script_written: str | None = None
    if out_script is not None:
        script_code = render_main_py(profile)
        out_p = Path(out_script).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(script_code, encoding="utf-8")
        script_written = str(out_p)

    if client is None:
        client = ArmdClient(base_url=f"http://{host}:{port}")
    warnings: list[str] = []

    # Fail-safe pre-reads: When apply_changes=True, DO NOT swallow pre-flight exceptions.
    # Abort with RuntimeError BEFORE writing backup or opening ArmSession.
    if apply_changes:
        try:
            safety_desc = client.get_safety_describe()
        except Exception as exc:
            raise RuntimeError(f"Required pre-flight read failed (safety_describe): {exc}") from exc
        if not safety_desc.get("hardware", {}).get("physical_estop", True):
            warnings.append(
                "Arm hardware reports no physical E-stop! Ensure an operator is at the emergency power cut switch."
            )

        try:
            existing_poses = client.get_poses()
        except Exception as exc:
            raise RuntimeError(f"Required pre-flight read failed (poses): {exc}") from exc

        try:
            existing_keepout = client.get_keepout()
        except Exception as exc:
            raise RuntimeError(f"Required pre-flight read failed (keepout): {exc}") from exc

        existing_collision: dict[str, Any] = {}
        if apply_collision_guard:
            try:
                existing_collision = client.get_collision_guard()
            except Exception as exc:
                raise RuntimeError(f"Required pre-flight read failed (collision_guard): {exc}") from exc
        else:
            try:
                existing_collision = client.get_collision_guard()
            except Exception:
                pass
        keepout_read_ok = True
    else:
        # Dry-run: best-effort read with warnings
        try:
            safety_desc = client.get_safety_describe()
            if not safety_desc.get("hardware", {}).get("physical_estop", True):
                warnings.append(
                    "Arm hardware reports no physical E-stop! Ensure an operator is at the emergency power cut switch."
                )
        except Exception as exc:
            warnings.append(f"Could not inspect safety description: {exc}")

        existing_poses = []
        try:
            existing_poses = client.get_poses()
        except Exception as exc:
            warnings.append(f"Could not read device poses: {exc}")

        existing_keepout = {}
        keepout_read_ok = True
        try:
            existing_keepout = client.get_keepout()
        except Exception as exc:
            keepout_read_ok = False
            warnings.append(f"Could not read device keepout boxes: {exc}")

        existing_collision = {}
        try:
            existing_collision = client.get_collision_guard()
        except Exception:
            pass

    device_boxes = existing_keepout.get("boxes", [])

    # Keepout preservation determination
    if not keepout_read_ok:
        if not profile.keepout_boxes and not clear_keepout:
            keepout_preserved = True
            warnings.append(
                "Could not read device keep-out boxes; preserving device state and not applying empty keep-out list. "
                "Pass --clear-keepout to force overwrite."
            )
        else:
            keepout_preserved = False
    else:
        keepout_preserved = bool(not profile.keepout_boxes and device_boxes and not clear_keepout)
        if keepout_preserved:
            warnings.append(
                "Existing device keep-out boxes preserved because profile specifies none. "
                "Pass --clear-keepout to overwrite with empty list."
            )

    collision_guard_requested = bool(profile.safety.get("collision_guard_enabled", False))
    collision_sensitivity = str(profile.safety.get("collision_sensitivity", "medium"))
    if not apply_collision_guard and collision_guard_requested:
        warnings.append(
            "Collision guard requested in profile but not automatically applied. "
            "Pass --apply-collision-guard or enable in Cockpit."
        )

    diff: dict[str, Any] = {
        "poses_to_set": list(profile.poses.keys()),
        "poses_device_count": len(existing_poses),
        "keepout_boxes_to_set": [b["name"] for b in profile.keepout_boxes],
        "keepout_boxes_device": [b.get("name") for b in device_boxes if isinstance(b, dict)],
        "collision_guard_requested": collision_guard_requested,
        "collision_sensitivity": collision_sensitivity,
    }

    if not apply_changes:
        return {
            "applied": False,
            "dry_run": True,
            "diff": diff,
            "warnings": warnings,
            "out_script": script_written,
        }

    # Validate all pose vectors, keepout boxes, and collision_sensitivity before entering session
    expected_dof = 6 if profile.robot == "so-arm101" else 7
    for name, positions in profile.poses.items():
        if not isinstance(name, str) or not POSE_NAME_RE.match(name):
            raise ValueError(f"Invalid pose name: {name!r}")
        if len(positions) != expected_dof:
            raise ValueError(f"Pose {name!r} has {len(positions)} joints; expected {expected_dof}")
        for val in positions:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Pose {name!r} has non-finite joint value: {val!r}")

    for b in profile.keepout_boxes:
        if not isinstance(b, dict):
            raise ValueError(f"Invalid keepout box: {b!r}")
        b_name = b.get("name")
        if not b_name or len(b_name) > 64:
            raise ValueError(f"Invalid keepout box name: {b_name!r}")
        min_m = b.get("min_m", [])
        max_m = b.get("max_m", [])
        if len(min_m) != 3 or len(max_m) != 3:
            raise ValueError(f"Keepout box {b_name} coordinates must have length 3")
        for i in range(3):
            if min_m[i] >= max_m[i]:
                raise ValueError(f"Keepout box {b_name} inverted along axis {i}")

    if collision_sensitivity not in ("low", "medium", "high"):
        raise ValueError(f"Invalid collision_sensitivity: {collision_sensitivity!r}")

    # Backup prior state before mutation
    if backup_path is None:
        backup_file = Path(tempfile.gettempdir()) / f"rebot_armd_backup_{int(time.time())}.json"
    else:
        backup_file = Path(backup_path).resolve()
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "host": host,
                "port": port,
                "poses": existing_poses,
                "keepout": existing_keepout,
                "collision": existing_collision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Build prior poses lookup for rollback
    prior_poses_map: dict[str, list[float]] = {}
    if isinstance(existing_poses, dict):
        pose_items = existing_poses.get("poses", existing_poses)
        if isinstance(pose_items, dict):
            for k, v in pose_items.items():
                if isinstance(v, dict) and "positions" in v:
                    prior_poses_map[k] = list(v["positions"])
                elif isinstance(v, (list, tuple)):
                    prior_poses_map[k] = list(v)
        elif isinstance(pose_items, list):
            for p in pose_items:
                if isinstance(p, dict) and "name" in p and "positions" in p:
                    prior_poses_map[p["name"]] = list(p["positions"])
    elif isinstance(existing_poses, list):
        for p in existing_poses:
            if isinstance(p, dict) and "name" in p and "positions" in p:
                prior_poses_map[p["name"]] = list(p["positions"])

    prior_collision_cfg = dict(existing_collision) if isinstance(existing_collision, dict) else {}

    keepout_mutated = False
    poses_mutated: list[str] = []
    collision_mutated = False

    # Open ArmSession to mutate poses & keepout (& optionally collision guard)
    session = ArmSession(client, force=False)
    with session:
        try:
            if not keepout_preserved:
                session.set_keepout(profile.keepout_boxes)
                keepout_mutated = True

            for pose_name, positions in profile.poses.items():
                session.save_pose(pose_name, positions)
                poses_mutated.append(pose_name)

            if apply_collision_guard and "collision_guard_enabled" in profile.safety:
                session.set_collision_guard(
                    enabled=profile.safety.get("collision_guard_enabled"),
                    sensitivity=profile.safety.get("collision_sensitivity"),
                )
                collision_mutated = True

        except Exception as exc:
            rollback_actions: list[str] = []
            rollback_errors: list[str] = []

            # 1. Rollback collision guard if mutated
            if collision_mutated and prior_collision_cfg:
                try:
                    session.set_collision_guard(
                        enabled=prior_collision_cfg.get("enabled"),
                        sensitivity=prior_collision_cfg.get("sensitivity"),
                    )
                    rollback_actions.append("collision_guard_restored")
                except Exception as r_exc:
                    rollback_errors.append(f"collision_guard_restore_failed({r_exc})")

            # 2. Rollback poses in reverse order
            for pose_name in reversed(poses_mutated):
                if pose_name in prior_poses_map:
                    try:
                        session.save_pose(pose_name, prior_poses_map[pose_name])
                        rollback_actions.append(f"restored_pose({pose_name})")
                    except Exception as r_exc:
                        rollback_errors.append(f"failed_restoring_pose({pose_name}: {r_exc})")
                else:
                    try:
                        session.delete_pose(pose_name)
                        rollback_actions.append(f"deleted_added_pose({pose_name})")
                    except Exception as r_exc:
                        rollback_errors.append(f"failed_deleting_pose({pose_name}: {r_exc})")

            # 3. Rollback keepout if mutated
            if keepout_mutated:
                try:
                    session.set_keepout(existing_keepout.get("boxes", []))
                    rollback_actions.append("keepout_restored")
                except Exception as r_exc:
                    rollback_errors.append(f"keepout_restore_failed({r_exc})")

            if rollback_errors:
                rollback_note = f"partial ({', '.join(rollback_actions)}; errors: {', '.join(rollback_errors)})"
            else:
                rollback_note = f"success ({', '.join(rollback_actions)})" if rollback_actions else "none_needed"

            raise RuntimeError(
                f"Failed applying deploy profile mid-mutation ({exc}); rollback={rollback_note}; backup={backup_file}"
            ) from exc

    return {
        "applied": True,
        "dry_run": False,
        "diff": diff,
        "warnings": warnings,
        "backup_path": str(backup_file),
        "out_script": script_written,
    }
