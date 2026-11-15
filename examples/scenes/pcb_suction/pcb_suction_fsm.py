"""PCB suction pick-and-place & test cell FSM example using `rebot.Arm` and `SimSceneClient`.

Workflow executed per cycle:
  1. Home the arm and move to overhead `scan` pose looking down at the infeed tray.
  2. Lower to `infeed_pick` and engage vacuum suction via virtual I/O (`vacuum: True`).
  3. Lift workpiece above 0.055 m (triggering stage `pcb_picked`) and travel over
     the static `divider_wall` via `divider_clear`.
  4. Lower workpiece onto `test_fixture` (triggering stage `pcb_in_fixture` and
     lighting `test_pass_led` green), then briefly disengage vacuum.
  5. Re-engage vacuum suction, lift to `fixture_approach`, and transfer to `outfeed_bin`.
  6. Lower to `outfeed_place`, release vacuum (triggering stage `pcb_in_outfeed`),
     and retreat safely above the bin.

Reads `$REBOT_HOST` (defaults to `127.0.0.1`). Press Ctrl+C anytime to stop;
leaving the `with Arm() as arm:` block parks the arm and releases control.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, TypeVar

from rebot import ActionFailed, Arm, ArmFaultError, RebotError

T = TypeVar("T")


class _SimClient:
    """Lightweight stdlib HTTP helper talking to rebot-sim (/sim/*) or armd (/api/v1/sandbox/*)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._sandbox_mode: str | None = None
        self._is_armd = "/api/v1/sandbox" in self.base_url
        if not self._is_armd:
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/v1/sandbox/state", timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        self._is_armd = True
                        if isinstance(data, dict) and isinstance(data.get("mode"), str):
                            self._sandbox_mode = data["mode"]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                self._is_armd = ":8080" in self.base_url

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def load_scene(self, scene: str, *, tool: str | None = None) -> dict[str, Any]:
        if self._is_armd:
            mode = "wasm" if self._sandbox_mode == "wasm" else "sim"
            return self._post("/api/v1/sandbox/activate", {"mode": mode, "scene": scene, "tool": tool})
        return self._post("/sim/scene", {"scene": scene, **({"tool": tool} if tool else {})})

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        path = "/api/v1/sandbox/reset" if self._is_armd else "/sim/reset"
        return self._post(path, {"seed": seed})

    def set_io(self, updates: dict[str, Any]) -> dict[str, Any]:
        if self._is_armd:
            return self._post("/api/v1/sandbox/io", {"updates": updates})
        return self._post("/sim/io", updates)


CONFIG = {
    # Keep speed_scale <= 0.5 on initial hardware/sim runs.
    "speed_scale": 0.4,
    "linear_speed_mps": 0.05,
    "max_cycles": 1,
    "guard_timeout_s": 25.0,
    # Saved pose names in armd (used automatically if taught, otherwise Cartesian waypoints).
    "poses": {
        "scan": "scan",
        "fixture_place": "fixture_place",
        "outfeed_place": "outfeed_place",
    },
    # Top-down wrist orientation (roll, pitch, yaw in rad).
    "down_rpy": [0.0, 1.5707963, 0.0],
    # Cartesian waypoints in arm base frame (metres: x forward, y left, z up).
    "waypoints": {
        "scan": [0.23, -0.14, 0.20],
        "infeed_approach": [0.23, -0.14, 0.08],
        "infeed_pick": [0.23, -0.14, 0.018],
        "divider_clear": [0.25, -0.065, 0.10],
        "fixture_approach": [0.27, 0.01, 0.08],
        "fixture_place": [0.27, 0.01, 0.024],
        "outfeed_approach": [0.23, 0.15, 0.08],
        "outfeed_place": [0.23, 0.15, 0.022],
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


def run_episode(
    arm: Arm,
    sim_client: _SimClient | None = None,
    *,
    speed_scale: float | None = None,
) -> bool:
    """Execute one 3-stage PCB suction pick-test-place cycle on an active `Arm` session."""
    scale = float(CONFIG["speed_scale"] if speed_scale is None else speed_scale)
    scale = max(0.1, min(1.0, scale))
    linear_speed = float(CONFIG["linear_speed_mps"]) * (scale / 0.4)
    rpy = list(CONFIG["down_rpy"])
    wp = CONFIG["waypoints"]

    has_poses = "poses" in arm.capabilities and arm.capabilities["poses"]
    known_poses = {p["name"] for p in arm.poses()} if has_poses else set()
    pose_map = CONFIG["poses"]

    def set_vacuum(active: bool) -> None:
        if sim_client is not None:
            try:
                sim_client.set_io({"vacuum": active})
            except Exception as exc:
                print(f"note: could not update simulation vacuum I/O: {exc}")

    # Wait safely after park/torque-off so wrist pose is well-defined before move_to.
    arm.home()
    set_vacuum(False)

    # 1. Move to scan pose above infeed tray.
    if pose_map["scan"] in known_poses:
        retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["scan"], speed_scale=scale))
    else:
        _move_waypoint(arm, wp["scan"], rpy, linear_speed)

    # 2. Lower onto PCB workpiece and engage vacuum suction.
    _move_waypoint(arm, wp["infeed_approach"], rpy, linear_speed)
    _move_waypoint(arm, wp["infeed_pick"], rpy, linear_speed * 0.7)
    set_vacuum(True)
    time.sleep(0.05)

    # 3. Lift PCB clear of tray and navigate over divider wall to fixture.
    _move_waypoint(arm, wp["infeed_approach"], rpy, linear_speed)
    _move_waypoint(arm, wp["divider_clear"], rpy, linear_speed)
    _move_waypoint(arm, wp["fixture_approach"], rpy, linear_speed)

    # 4. Place PCB on test fixture.
    if pose_map["fixture_place"] in known_poses:
        retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["fixture_place"], speed_scale=scale))
    else:
        _move_waypoint(arm, wp["fixture_place"], rpy, linear_speed * 0.7)

    # Release vacuum to let fixture seat board.
    set_vacuum(False)
    time.sleep(0.05)

    # 5. Re-engage suction and lift from fixture.
    set_vacuum(True)
    time.sleep(0.05)
    _move_waypoint(arm, wp["fixture_approach"], rpy, linear_speed)

    # 6. Transfer PCB into outfeed bin and release vacuum.
    _move_waypoint(arm, wp["outfeed_approach"], rpy, linear_speed)
    if pose_map["outfeed_place"] in known_poses:
        retry_after_collision(arm, lambda: arm.move_to_pose(pose_map["outfeed_place"], speed_scale=scale))
    else:
        _move_waypoint(arm, wp["outfeed_place"], rpy, linear_speed * 0.7)

    set_vacuum(False)
    time.sleep(0.05)
    _move_waypoint(arm, wp["outfeed_approach"], rpy, linear_speed)
    _move_waypoint(arm, wp["scan"], rpy, linear_speed)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PCB suction pick-and-place FSM on reBot arm / rebot-sim")
    parser.add_argument("--cycles", type=int, default=int(CONFIG["max_cycles"]), help="Maximum cycles (bounded)")
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=float(CONFIG["speed_scale"]),
        help="Speed scale 0.1..1.0",
    )
    parser.add_argument(
        "--sim-url",
        type=str,
        default="http://127.0.0.1:8090",
        help="Base URL for rebot-sim (http://127.0.0.1:8090) or armd sandbox (http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--reset-scene",
        action="store_true",
        help="Hot-swap and reset pcb_suction scene on rebot-sim / armd before starting",
    )
    args = parser.parse_args(argv)

    sim_client = _SimClient(base_url=args.sim_url)
    if args.reset_scene:
        try:
            print("resetting rebot-sim to pcb_suction scene with suction tool...")
            sim_client.load_scene("pcb_suction", tool="suction")
            sim_client.reset(seed=42)
        except Exception as exc:
            print(f"warning: could not reset rebot-sim scene: {exc}")

    max_cycles = max(1, int(args.cycles))
    with Arm() as arm:
        safety = arm.safety() if "safety_describe" in arm.capabilities and arm.capabilities["safety_describe"] else {}
        if safety.get("hardware", {}).get("physical_estop") is False:
            print("note: arm reports no physical E-stop; keep web stop button or power cut within reach")

        arm.guard(float(CONFIG["guard_timeout_s"]))
        for cycle in range(1, max_cycles + 1):
            arm.feed()
            print(f"starting pcb_suction cycle {cycle}/{max_cycles}...")
            try:
                run_episode(arm, sim_client=sim_client, speed_scale=args.speed_scale)
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
