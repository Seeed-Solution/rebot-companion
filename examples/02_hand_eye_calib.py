#!/usr/bin/env python3
"""Example 2: Interactive Hand-Eye Calibration Wizard."""

import argparse
from pathlib import Path
from rebot_companion import ReCameraClient, ArmdClient, HandEyeCalibration, Transform


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration Wizard")
    parser.add_argument("--camera-url", default="http://192.168.42.1:8090")
    parser.add_argument("--arm-url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", default="eye_to_hand", choices=["eye_to_hand", "eye_in_hand"])
    parser.add_argument("--out", default="calibration.json", help="Output calibration file")
    args = parser.parse_args()

    calib = HandEyeCalibration(mode=args.mode)
    print(f"=== Starting {args.mode.upper()} Calibration Wizard ===")
    print("Move the robot arm through at least 4 distinct non-parallel poses (recommended 6~9).")

    # In actual usage, user moves arm or calls armd teleop. Here is the collection loop:
    print("\nPress Enter to capture current pose pair, or type 'done' to solve:")
    sample_idx = 1
    while True:
        cmd = input(f"Capture Sample #{sample_idx} [Enter=Capture, d=Done, q=Quit]: ").strip().lower()
        if cmd == "q":
            print("Aborted.")
            return
        if cmd in ("d", "done"):
            break

        # In real operation, read gripper pose from arm and marker pose from camera
        print(f"[*] Triggering reCamera for Sample #{sample_idx}...")
        # (Placeholder for real capture if hardware connected)
        break

    print("\nCalibration guide completed.")


if __name__ == "__main__":
    main()
