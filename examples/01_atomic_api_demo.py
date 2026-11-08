#!/usr/bin/env python3
"""Example 1: Calling reCamera Atomic APIs directly without complex dependencies."""

import argparse
import sys
from rebot_companion import ReCameraClient


def main():
    parser = argparse.ArgumentParser(description="reCamera Atomic API Demo")
    parser.add_argument("--url", default="http://192.168.42.1:8090", help="reCamera base URL")
    args = parser.parse_args()

    client = ReCameraClient(base_url=args.url)
    print(f"Connecting to reCamera at {args.url}...")

    # 1. Query current state (should be standby: 1 fps, inference disabled)
    try:
        state = client.get_inference_state()
        print(f"[*] Current State: enabled={state.get('enabled')}, fps={state.get('fps')}, idle_fps={state.get('idle_fps')}")
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
