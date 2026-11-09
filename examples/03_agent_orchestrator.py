#!/usr/bin/env python3
"""
Example 3: Stage 1 - Remote Agent Orchestrator.
Demonstrates an Agent / VLM tool calling pattern over reCamera and rebot-armd atomic APIs.

Pass ``--calib calibration.json`` (written by ``02_hand_eye_calib.py`` or exported from the
rebot-armd Web cockpit). Without it the tools use ``PLACEHOLDER_CAM_TO_BASE``, a made-up
transform: base-frame positions are not real, and ``plan_and_execute_pick`` refuses to run
unless ``--dry-run`` is set (detect and check only, the arm does not move).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from rebot_companion import (
    ArmdClient,
    B601IkAdapter,
    CameraError,
    CameraTriggerTimeout,
    HandEyeCalibration,
    HandEyeResult,
    ReCameraClient,
    Transform,
)
from rebot_companion.pipeline import DeterministicPickPlacePipeline, PipelineConfig

# PLACEHOLDER, not a calibration: camera at (+0.30, -0.40, +0.50) m, rpy (180, 0, 90) deg.
# Only for showing the tool-calling flow offline. Never use it for real motion.
PLACEHOLDER_CAM_TO_BASE = Transform.from_translation_euler([0.30, -0.40, 0.50], 180.0, 0.0, 90.0)


class RobotCompanionAgentTools:
    """Tools exposed to an LLM / VLM Agent (compatible with OpenAI Function Calling & Anthropic Tools).

    ``hand_eye`` is a :class:`HandEyeResult` from :meth:`HandEyeCalibration.load_result`. Without it
    the placeholder transform is used and motion is refused unless ``dry_run`` is true.
    ``dry_run`` runs the pick pipeline without an IK solver: detect + workspace check, no motion.
    """

    def __init__(
        self,
        camera_url: str = "http://192.168.42.1:8090",
        arm_url: str = "http://127.0.0.1:8765",
        hand_eye: HandEyeResult | None = None,
        ik_solver: Callable[[Transform], list[float]] | None = None,
        dry_run: bool = False,
    ):
        self.camera = ReCameraClient(camera_url)
        self.arm = ArmdClient(arm_url)
        self.calibrated = hand_eye is not None
        self.hand_eye = hand_eye or HandEyeResult("eye_to_hand", PLACEHOLDER_CAM_TO_BASE)
        self.dry_run = dry_run
        # pose in base frame -> joint targets; never used in a dry run
        self.ik_solver = None if dry_run else ik_solver

    def get_tool_definitions(self) -> list[dict]:
        """Returns JSON schema definitions of tools for the LLM."""
        return [
            {
                "name": "inspect_scene",
                "description": (
                    "Trigger reCamera to detect all visible fiducial markers and return their 3D positions "
                    "in the robot base frame."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wait_ms": {"type": "integer", "description": "Maximum wait time in ms", "default": 2500}
                    },
                },
            },
            {
                "name": "take_visual_snapshot",
                "description": "Capture a high-resolution JPEG snapshot from reCamera for multimodal visual reasoning.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_arm_state",
                "description": "Read the robot arm safety state, torque state and joint positions (radians).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "emergency_stop",
                "description": "Immediately latch the robot arm emergency stop.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "plan_and_execute_pick",
                "description": "Command the robot arm to pick an object with the specified marker ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"marker_id": {"type": "integer", "description": "Target marker ID to grasp"}},
                    "required": ["marker_id"],
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Dispatcher for tool calls from the Agent."""
        if name == "inspect_scene":
            wait_ms = arguments.get("wait_ms", 2500)
            try:
                detections = self.camera.trigger(wait_ms=wait_ms)
            except CameraTriggerTimeout as e:
                return json.dumps({"error": f"camera trigger timed out: {e}"})
            joints = self.arm.get_observation().positions if self.hand_eye.needs_joints else None
            cam_to_base = self.hand_eye.cam_to_base(joints)
            results = []
            for d in detections:
                in_base = cam_to_base @ d.pose
                results.append(
                    {
                        "id": d.id,
                        "position_base_xyz": [round(x, 4) for x in in_base.translation],
                        "rpy_deg": [round(x, 1) for x in in_base.to_euler_deg()],
                        "confidence": 1.0 - d.reprojection_error,
                    }
                )
            out = {"detected_markers": results, "calibrated": self.calibrated}
            if not self.calibrated:
                out["warning"] = "placeholder hand-eye transform: base-frame positions are not real"
            return json.dumps(out, indent=2)

        elif name == "take_visual_snapshot":
            try:
                img_bytes = self.camera.get_snapshot()  # needs ReCameraClient(snapshot_url=...)
            except CameraError as e:
                return json.dumps({"error": str(e)})
            return f"Captured {len(img_bytes)} bytes of JPEG snapshot for visual analysis."

        elif name == "get_arm_state":
            obs = self.arm.get_observation()
            return json.dumps(
                {
                    "safety_state": obs.safety_state,
                    "fault_reason": obs.fault_reason,
                    "is_torqued": obs.is_torqued,
                    "joint_positions_rad": [round(q, 4) for q in obs.positions],
                }
            )

        elif name == "emergency_stop":
            # REST e-stop: works without holding the control lease.
            return json.dumps(self.arm.e_stop())

        elif name == "plan_and_execute_pick":
            m_id = int(arguments["marker_id"])
            if not self.calibrated and not self.dry_run:
                return json.dumps(
                    {
                        "success": False,
                        "stage": "refused",
                        "error": "no hand-eye calibration loaded; pass --calib, or --dry-run to detect without motion",
                    }
                )
            # Same deterministic pipeline as Stage 2: trigger -> transform -> bounds check -> leased motion.
            pipeline = DeterministicPickPlacePipeline(
                self.camera,
                self.arm,
                self.hand_eye,
                PipelineConfig(target_marker_id=m_id),
                ik_solver=self.ik_solver,
            )
            res = pipeline.run_cycle()
            return json.dumps(
                {
                    "success": res.success,
                    "stage": res.stage,
                    "error": res.error_message,
                    "motion_executed": res.success and self.ik_solver is not None,
                    "calibrated": self.calibrated,
                    "target_base_xyz": [round(x, 4) for x in res.target_pose_base.translation]
                    if res.target_pose_base
                    else None,
                    "warnings": res.warnings,
                }
            )

        return f"Unknown tool: {name}"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Stage 1 agent tool-calling demo")
    parser.add_argument("--camera-url", default="http://192.168.42.1:8090")
    parser.add_argument("--arm-url", default="http://127.0.0.1:8765")
    parser.add_argument("--calib", type=Path, help="hand-eye calibration JSON (HandEyeCalibration.load_result)")
    parser.add_argument(
        "--dry-run", action="store_true", help="plan_and_execute_pick detects and checks only; the arm does not move"
    )
    args = parser.parse_args(argv)

    hand_eye = HandEyeCalibration.load_result(args.calib) if args.calib else None
    tools = RobotCompanionAgentTools(
        args.camera_url,
        args.arm_url,
        hand_eye=hand_eye,
        ik_solver=B601IkAdapter(),
        dry_run=args.dry_run,
    )
    print("=== Stage 1: Remote Agent Orchestration Interface ===")
    if hand_eye:
        print(f"[*] Loaded {hand_eye.setup} calibration from {args.calib}")
    else:
        print("[!] No --calib given: using PLACEHOLDER_CAM_TO_BASE; positions are not real.")
        if not args.dry_run:
            print("[!] plan_and_execute_pick will refuse to move the arm (pass --calib, or --dry-run).")
    print("Available tools for LLM / VLM Agent:")
    for t in tools.get_tool_definitions():
        print(f" - {t['name']}: {t['description']}")

    print("\nSimulating Agent calling 'inspect_scene':")
    # In real usage, the LLM outputs this JSON tool call:
    call = {"name": "inspect_scene", "arguments": {"wait_ms": 1000}}
    try:
        out = tools.execute_tool(call["name"], call["arguments"])
        print("Agent Tool Response:\n", out)
    except Exception as e:
        print(f"Execution error (offline): {e}")


if __name__ == "__main__":
    main()
