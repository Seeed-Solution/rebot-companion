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

    # Keepout boxes validation
    raw_boxes = raw.get("keepout_boxes", [])
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
    program["cycles"] = cycles
    program["dwell_s"] = dwell_s
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

    code = f'''#!/usr/bin/env python3
"""Auto-generated reBot companion application script for {profile.scene_id}.

Safety constraints:
- Speed scaled down for safe initial execution (speed_scale <= 0.3).
- Bounded loop cycles with watchdog feeding.
- Automatic recovery on transient watchdog/deadline trips only; latches hard stops.
- Uses Python 3.8 standard library and `rebot` SDK only.
"""

from typing import Any, Dict, List, Optional
import json
import os
import sys
import time
import urllib.request

from rebot import ActionFailed, Arm, ArmFaultError, RebotError

CONFIG = {{
    "host": os.environ.get("REBOT_HOST", "127.0.0.1"),
    "speed_scale": {speed_scale!r},
    "linear_speed_mps": {linear_speed_mps!r},
    "max_cycles": {max_cycles!r},
    "dwell_s": {dwell_s!r},
    "poses": {profile.poses!r},
    "sim_io_url": os.environ.get("REBOT_SIM_IO_URL", ""),
}}


def set_tool_io(name: str, value: Any) -> None:
    """Set virtual simulator I/O or raise NotImplementedError on real arm."""
    sim_io_url = CONFIG.get("sim_io_url")
    if sim_io_url:
        target_url = str(sim_io_url).rstrip("/") + "/sim/io"
        payload = json.dumps({{name: value}}).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=payload,
            headers={{"Content-Type": "application/json"}},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status >= 400:
                raise RuntimeError("Failed to set sim I/O channel: HTTP " + str(resp.status))
    else:
        raise NotImplementedError(
            "Configure physical GPIO / Modbus / suction relay before running without REBOT_SIM_IO_URL"
        )


def main() -> int:
    host = str(CONFIG["host"])
    speed = float(CONFIG["speed_scale"])
    max_cycles = int(CONFIG["max_cycles"])
    dwell_s = float(CONFIG["dwell_s"])
    poses: Dict[str, List[float]] = CONFIG["poses"]  # type: ignore

    print("[rebot-app] Starting program on host " + host + " (speed_scale=" + str(speed) + ")")

    with Arm(host=host, force=False) as arm:
        arm.home()
        arm.guard(15.0)

        for cycle in range(max_cycles):
            arm.feed()
            print("[rebot-app] Cycle " + str(cycle + 1) + " / " + str(max_cycles))

            for pose_name in poses:
                arm.feed()
                print("  -> moving to pose: " + pose_name)
                try:
                    arm.move_to_pose(pose_name, speed_scale=speed)
                except ActionFailed as exc:
                    if exc.code == "COLLISION":
                        print("  [warning] collision guard triggered, waiting for clear...")
                        arm.wait_until_clear()
                        # Retry once after clear
                        arm.move_to_pose(pose_name, speed_scale=speed)
                    else:
                        raise
                except ArmFaultError as exc:
                    if exc.code in ("WATCHDOG_TIMEOUT", "CONTROL_DEADLINE", "HANDOVER_UNSETTLED"):
                        print("  [warning] recoverable fault " + str(exc.code) + "; recovering...")
                        arm.recover()
                        arm.move_to_pose(pose_name, speed_scale=speed)
                    else:
                        print("  [error] non-recoverable arm fault: " + str(exc))
                        raise

                if dwell_s > 0.0:
                    time.sleep(dwell_s)

    print("[rebot-app] Program completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    # Ensure valid AST under Python 3.8
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

    try:
        safety_desc = client.get_safety_describe()
        if not safety_desc.get("hardware", {}).get("physical_estop", True):
            warnings.append(
                "Arm hardware reports no physical E-stop! Ensure an operator is at the emergency power cut switch."
            )
    except Exception as exc:
        warnings.append(f"Could not inspect safety description: {exc}")

    existing_poses: list[dict[str, Any]] = []
    try:
        existing_poses = client.get_poses()
    except Exception as exc:
        warnings.append(f"Could not read device poses: {exc}")

    existing_keepout: dict[str, Any] = {}
    try:
        existing_keepout = client.get_keepout()
    except Exception as exc:
        warnings.append(f"Could not read device keepout boxes: {exc}")

    device_boxes = existing_keepout.get("boxes", [])
    diff: dict[str, Any] = {
        "poses_to_set": list(profile.poses.keys()),
        "poses_device_count": len(existing_poses),
        "keepout_boxes_to_set": [b["name"] for b in profile.keepout_boxes],
        "keepout_boxes_device": [b.get("name") for b in device_boxes if isinstance(b, dict)],
    }

    if not apply_changes:
        return {
            "applied": False,
            "dry_run": True,
            "diff": diff,
            "warnings": warnings,
            "out_script": script_written,
        }

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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Open ArmSession to mutate poses & keepout
    session = ArmSession(client, force=False)
    with session:
        session.set_keepout(profile.keepout_boxes)
        for pose_name, positions in profile.poses.items():
            session.save_pose(pose_name, positions)

    return {
        "applied": True,
        "dry_run": False,
        "diff": diff,
        "warnings": warnings,
        "backup_path": str(backup_file),
        "out_script": script_written,
    }
