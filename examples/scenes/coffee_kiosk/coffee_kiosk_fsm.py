"""Mobile coffee kiosk FSM example using only `rebot.Arm` and Python stdlib.

Workflow executed per cycle:
  1. Home the arm and move to the overhead `scan` pose (or named pose if taught).
  2. Detect the `cup` (2D vision macro) and grasp it, or use the fixed fallback pose
     when `--no-vision` / `CONFIG["use_vision"] = False` is selected.
  3. Carry the cup via a clear approach waypoint onto the espresso drip tray,
     avoiding the espresso machine column.
  4. Close the gripper and press the spring-loaded brew switch vertically.
  5. Re-grasp the cup from the drip tray and place it on the pickup counter.

Reads `$REBOT_HOST` (defaults to `127.0.0.1`). Press Ctrl+C anytime to stop;
leaving the `with Arm() as arm:` block parks the arm and releases control.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TypeVar

from rebot import ActionFailed, Arm, ArmFaultError, RebotError

T = TypeVar("T")

CONFIG = {
    # Keep speed_scale <= 0.5 on initial hardware/sim runs.
    "speed_scale": 0.4,
    "linear_speed_mps": 0.05,
    "use_vision": True,
    "cup_label": "cup",
    "min_score": 0.5,
    "max_cycles": 1,
    "guard_timeout_s": 25.0,
    # Saved pose names in armd (used automatically if already taught, otherwise
    # the Cartesian waypoints below are executed directly).
    "poses": {
        "scan": "scan",
        "drip_tray": "drip_tray",
        "pickup_counter": "pickup_counter",
    },
    # Top-down wrist orientation (roll, pitch, yaw in rad).
    "down_rpy": [0.0, 1.5707963, 0.0],
    # Fallback / fixture waypoints in arm base frame (metres: x forward, y left, z up).
    "waypoints": {
        "scan": [0.24, -0.10, 0.22],
        "cup_approach": [0.27, -0.10, 0.12],
        "cup_fallback": [0.27, -0.10, 0.035],
        "drip_approach": [0.31, 0.10, 0.115],
        "drip_tray": [0.31, 0.10, 0.048],
        "brew_above": [0.25, 0.17, 0.145],
        "brew_press": [0.25, 0.17, 0.094],
        "pickup_approach": [0.25, -0.16, 0.115],
        "pickup_counter": [0.25, -0.16, 0.040],
    },
}


def retry_after_collision(arm: Arm, step_fn: Callable[[], T]) -> T:
    """Run one motion step; if collision guard pauses the arm, wait once and retry."""
    try:
        return step_fn()
    except ActionFailed as exc:
        if exc.code != "COLLISION":
            raise
        print("collision guard triggered; waiting for clearance to retry once...")
        arm.wait_until_clear(timeout_s=10.0)
        return step_fn()


def _move_waypoint(arm: Arm, xyz: list[float], rpy: list[float], speed_mps: float) -> None:
    retry_after_collision(
        arm,
        lambda: arm.move_to(xyz, rpy=rpy, speed=speed_mps, auto_yaw=True),
    )


def run_episode(arm: Arm, *, use_vision: bool | None = None, speed_scale: float | None = None) -> bool:
    """Execute one 4-stage coffee kiosk cycle on an active `Arm` session."""
    scale = float(CONFIG["speed_scale"] if speed_scale is None else speed_scale)
    scale = max(0.1, min(1.0, scale))
    linear_speed = float(CONFIG["linear_speed_mps"]) * (scale / 0.4)
    vision_enabled = bool(CONFIG["use_vision"] if use_vision is None else use_vision)
    rpy = list(CONFIG["down_rpy"])
    wp = CONFIG["waypoints"]

    has_poses = "poses" in arm.capabilities and arm.capabilities["poses"]
    known_poses = {p["name"] for p in arm.poses()} if has_poses else set()
    pose_map = CONFIG["poses"]

    # Wait safely after park/torque-off so wrist pose is well-defined before move_to.
    arm.home()

    # 1. Move to scan pose and acquire cup.
    arm.gripper_open()
    if pose_map["scan"] in known_poses:
        retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["scan"], speed_scale=scale))
    else:
        _move_waypoint(arm, wp["scan"], rpy, linear_speed)

    held = False
    if vision_enabled:
        detections = arm.detect(label=str(CONFIG["cup_label"]), min_score=float(CONFIG["min_score"]))
        if detections:
            res = retry_after_collision(arm, lambda: arm.grasp(detections[0]))
            held = bool(res.holding)
        else:
            print("vision found no cup; falling back to default cup waypoint")

    if not held:
        _move_waypoint(arm, wp["cup_approach"], rpy, linear_speed)
        _move_waypoint(arm, wp["cup_fallback"], rpy, linear_speed)
        held = bool(arm.gripper_close(grasp=True).holding)
        _move_waypoint(arm, wp["cup_approach"], rpy, linear_speed)

    def _move_drip_tray() -> None:
        if pose_map["drip_tray"] in known_poses:
            retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["drip_tray"], speed_scale=scale))
        else:
            _move_waypoint(arm, wp["drip_tray"], rpy, linear_speed)

    # 2. Place cup under the espresso spout, routing via drip_approach above the tray.
    _move_waypoint(arm, wp["drip_approach"], rpy, linear_speed)
    _move_drip_tray()
    arm.gripper_open()
    _move_waypoint(arm, wp["drip_approach"], rpy, linear_speed)

    # 3. Close jaws cleanly and press the spring brew button.
    arm.gripper_close(grasp=False)
    _move_waypoint(arm, wp["brew_above"], rpy, linear_speed)
    _move_waypoint(arm, wp["brew_press"], rpy, linear_speed * 0.6)
    _move_waypoint(arm, wp["brew_above"], rpy, linear_speed)

    # 4. Re-grasp cup from drip tray and serve at customer pickup mat.
    _move_waypoint(arm, wp["drip_approach"], rpy, linear_speed)
    arm.gripper_open()
    _move_drip_tray()
    arm.gripper_close(grasp=True)
    _move_waypoint(arm, wp["drip_approach"], rpy, linear_speed)

    _move_waypoint(arm, wp["pickup_approach"], rpy, linear_speed)
    if pose_map["pickup_counter"] in known_poses:
        retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["pickup_counter"], speed_scale=scale))
    else:
        _move_waypoint(arm, wp["pickup_counter"], rpy, linear_speed)
    arm.gripper_open()
    _move_waypoint(arm, wp["pickup_approach"], rpy, linear_speed)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run coffee kiosk FSM on reBot arm / rebot-sim")
    parser.add_argument("--no-vision", action="store_true", help="Skip 2D vision detect/grasp and use fixed poses")
    parser.add_argument("--cycles", type=int, default=int(CONFIG["max_cycles"]), help="Maximum cycles (bounded)")
    parser.add_argument("--speed-scale", type=float, default=float(CONFIG["speed_scale"]), help="Speed scale 0.1..1.0")
    args = parser.parse_args(argv)

    max_cycles = max(1, int(args.cycles))
    with Arm() as arm:
        safety = arm.safety() if "safety_describe" in arm.capabilities and arm.capabilities["safety_describe"] else {}
        if safety.get("hardware", {}).get("physical_estop") is False:
            print("note: arm reports no physical E-stop; keep web stop button or power cut within reach")

        arm.guard(float(CONFIG["guard_timeout_s"]))
        for cycle in range(1, max_cycles + 1):
            arm.feed()
            print(f"starting coffee_kiosk cycle {cycle}/{max_cycles}...")
            try:
                run_episode(arm, use_vision=not args.no_vision, speed_scale=args.speed_scale)
            except ArmFaultError as exc:
                fault = getattr(exc.status, "fault", None) or {}
                if fault.get("recoverable"):
                    print(f"recoverable fault ({exc.code}); calling arm.recover()...")
                    arm.recover()
                    continue
                print(f"unrecoverable arm fault ({exc.code}): inspect scene and clear fault in web cockpit")
                return 1
            except RebotError as exc:
                print(f"stopped on RebotError ({exc.code}): {exc.message}")
                return 1
            arm.feed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
