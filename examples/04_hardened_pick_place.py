#!/usr/bin/env python3
"""
Example 4: Stage 2 - Hardened Deterministic Local Pipeline.
Runs a repeatable, high-reliability pick and place cycle directly on the factory/lab client.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from rebot_companion import (
    ReCameraClient,
    ArmdClient,
    Transform,
    HandEyeCalibration,
    DeterministicPickPlacePipeline,
)
from rebot_companion.pipeline import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Hardened Pick & Place Loop")
    parser.add_argument("--camera-url", default="http://192.168.42.1:8090")
    parser.add_argument("--arm-url", default="http://127.0.0.1:8765")
    parser.add_argument("--calib", default="calibration.json", help="Path to hand-eye calibration JSON")
    parser.add_argument("--marker-id", type=int, default=0)
    args = parser.parse_args()

    # Load calibration matrix
    calib_path = Path(args.calib)
    if calib_path.exists():
        print(f"[*] Loading calibration matrix from {calib_path}...")
        cam_to_base = HandEyeCalibration.load_from_file(calib_path)
    else:
        print("[!] No calibration.json found, using nominal baseline transform.")
        cam_to_base = Transform.from_translation_euler([0.30, -0.40, 0.50], 180.0, 0.0, 90.0)

    camera = ReCameraClient(args.camera_url)
    arm = ArmdClient(args.arm_url)
    config = PipelineConfig(target_marker_id=args.marker_id)

    pipeline = DeterministicPickPlacePipeline(camera, arm, cam_to_base, config)
    print(f"=== Stage 2: Hardened Local Pick & Place Pipeline (Target Marker #{args.marker_id}) ===")
    print("[*] Running closed-loop execution cycle...")

    result = pipeline.run_cycle()
    if result.success:
        t = [round(x, 4) for x in result.target_pose_base.translation]
        print(f"[+] Cycle Completed Successfully in {result.elapsed_ms:.1f}ms!")
        print(f"    Grasp Pose in Arm Base: {t}m")
    else:
        print(f"[-] Cycle Failed at Stage '{result.stage}': {result.error_message}")


if __name__ == "__main__":
    main()
