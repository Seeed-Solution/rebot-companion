#!/usr/bin/env python3
"""Example 1: Calling reCamera and rebot-armd atomic APIs directly without complex dependencies."""

import argparse
import sys

from rebot_companion import ReCameraClient
from rebot_companion.arm_client import ArmdClient, ArmdError


def arm_demo(url: str, move: bool) -> int:
    """Lease-free reads over REST; motion over a WebSocket lease session."""
    arm = ArmdClient(url)
    try:
        obs = arm.get_observation()
        schema = arm.get_schema()
    except Exception as e:
        print(f"[!] Failed to reach rebot-armd at {url}: {e}")
        return 1
    print(f"[*] Arm: {schema.dof} DOF {schema.joint_names}")
    print(
        f"[*] State={obs.safety_state}{f' ({obs.fault_reason})' if obs.fault_reason else ''} "
        f"epoch={obs.epoch_id} torqued={obs.is_torqued} q={[round(q, 3) for q in obs.positions]}"
    )
    if not move:
        print("    (pass --move to arm torque, home, nudge joint 1 and unload)")
        return 0
    try:
        with arm.session() as s:  # acquires the control lease and starts the heartbeat thread
            print(f"[*] Lease generation {s.generation}")
            if obs.is_stopped:
                print(f"[*] clear_fault: {s.clear_fault()['safety_state']}")
            s.torque(True)
            homed = s.home(duration=2.0)
            # Start from the measured home pose (J6 home is pi/2 on the B601-DM) and change joint 1 only.
            target = list(homed.positions)
            target[0] += 0.2
            done = s.move_to(target, duration=1.5)
            print(f"[*] Arrived: q={[round(q, 3) for q in done.positions]}")
            s.home(duration=1.5)
            s.unload()  # park + torque off, so no watchdog fault after the lease is released
            print("[*] Unloaded, lease released")
    except ArmdError as e:
        print(f"[!] Arm motion failed: {e}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="reCamera / rebot-armd Atomic API Demo")
    parser.add_argument("--url", default="http://192.168.42.1:8090", help="reCamera base URL")
    parser.add_argument("--arm-url", default=None, help="rebot-armd base URL, e.g. http://127.0.0.1:8765")
    parser.add_argument("--move", action="store_true", help="also run a short motion sequence on the arm")
    args = parser.parse_args()

    if args.arm_url:
        return arm_demo(args.arm_url, args.move)

    client = ReCameraClient(base_url=args.url)
    print(f"Connecting to reCamera at {args.url}...")

    # 1. Query current state (should be standby: 1 fps, inference disabled)
    try:
        state = client.get_inference_state()
        print(
            f"[*] Current State: enabled={state.get('enabled')}, fps={state.get('fps')}, "
            f"idle_fps={state.get('idle_fps')}"
        )
    except Exception as e:
        print(f"[!] Failed to connect to reCamera: {e}")
        print("    (If running in simulation or offline, use mock client)")
        return 1

    # 2. Atomic one-shot trigger
    print("[*] Sending atomic trigger request (POST /api/trigger)...")
    detections = client.trigger(wait_ms=2500)
    print(f"[*] Detected {len(detections)} markers:")
    for m in detections:
        t = [round(x, 4) for x in m.pose.translation]
        r, p, y = [round(x, 1) for x in m.pose.to_euler_deg()]
        print(f"    - Marker #{m.id}: pos={t}m, rpy_deg=[{r}, {p}, {y}], error={m.reprojection_error:.4f}")

    # 3. Verify state returned to standby
    post_state = client.get_inference_state()
    print(f"[*] Post-trigger State: enabled={post_state.get('enabled')} (CPU safely in standby)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
