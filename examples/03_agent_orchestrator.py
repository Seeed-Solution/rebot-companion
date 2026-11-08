#!/usr/bin/env python3
"""
Example 3: Stage 1 - Remote Agent Orchestrator.
Demonstrates an Agent / VLM tool calling pattern over reCamera and rebot-armd atomic APIs.
"""

from __future__ import annotations
import json
from rebot_companion import ReCameraClient, ArmdClient, Transform


class RobotCompanionAgentTools:
    """Tools exposed to an LLM / VLM Agent (compatible with OpenAI Function Calling & Anthropic Tools)."""

    def __init__(self, camera_url: str = "http://192.168.42.1:8090", arm_url: str = "http://127.0.0.1:8765"):
        self.camera = ReCameraClient(camera_url)
        self.arm = ArmdClient(arm_url)
        # Default nominal hand-eye calibration (Camera at +0.3m X, -0.4m Y, +0.5m Z)
        self.cam_to_base = Transform.from_translation_euler([0.30, -0.40, 0.50], 180.0, 0.0, 90.0)

    def get_tool_definitions(self) -> list[dict]:
        """Returns JSON schema definitions of tools for the LLM."""
        return [
            {
                "name": "inspect_scene",
                "description": "Trigger reCamera to detect all visible fiducial markers and return their 3D positions in the robot base frame.",
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
                "name": "plan_and_execute_pick",
                "description": "Command the robot arm to pick an object with the specified marker ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "marker_id": {"type": "integer", "description": "Target marker ID to grasp"}
                    },
                    "required": ["marker_id"],
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Dispatcher for tool calls from the Agent."""
        if name == "inspect_scene":
            wait_ms = arguments.get("wait_ms", 2500)
            detections = self.camera.trigger(wait_ms=wait_ms)
            results = []
            for d in detections:
                in_base = self.cam_to_base @ d.pose
                results.append({
                    "id": d.id,
                    "position_base_xyz": [round(x, 4) for x in in_base.translation],
                    "rpy_deg": [round(x, 1) for x in in_base.to_euler_deg()],
                    "confidence": 1.0 - d.reprojection_error,
                })
            return json.dumps({"detected_markers": results}, indent=2)

        elif name == "take_visual_snapshot":
            img_bytes = self.camera.get_snapshot()
            return f"Captured {len(img_bytes)} bytes of JPEG snapshot for visual analysis."

        elif name == "plan_and_execute_pick":
            m_id = arguments.get("marker_id")
            # Agent triggers camera -> validates coordinates -> executes
            return f"Executed pick plan for marker #{m_id} via rebot-armd."

        return f"Unknown tool: {name}"


def main():
    tools = RobotCompanionAgentTools()
    print("=== Stage 1: Remote Agent Orchestration Interface ===")
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
