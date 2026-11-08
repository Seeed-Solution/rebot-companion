#!/usr/bin/env python3
"""
Example 4: Stage 2 - Hardened Deterministic Local Pipeline.
Runs a repeatable, high-reliability pick and place cycle directly on the factory/lab client.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rebot_companion import (
    ArmdClient,
    DeterministicPickPlacePipeline,
    HandEyeCalibration,
    ReCameraClient,
    Transform,
)
from rebot_companion.arm_client import ArmdError
from rebot_companion.pipeline import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Hardened Pick & Place Loop")
    parser.add_argument("--camera-url", default="http://192.168.42.1:8090")
    parser.add_argument("--arm-url", default="http://127.0.0.1:8765")
    parser.add_argument("--calib", default="calibration.json", help="Path to hand-eye calibration JSON")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1, help="cycles to run under one control lease")
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

    obs = arm.get_observation()
    print(f"[*] Arm state: {obs.safety_state} (epoch {obs.epoch_id}, torqued={obs.is_torqued})")
    print(f"=== Stage 2: Hardened Local Pick & Place Pipeline (Target Marker #{args.marker_id}) ===")

    # Supply an IK solver (pose in base frame -> joint targets) to enable motion;
    # without one the pipeline stops after the workspace check.
    ik_solver = None
    try:
        # One lease for all cycles: the session heartbeats in the background, and
        # armd stops motion if this process dies or the connection drops.
        with arm.session() as session:
            pipeline = DeterministicPickPlacePipeline(
                camera, arm, cam_to_base, config, ik_solver=ik_solver, session=session
            )
            for i in range(args.cycles):
                print(f"[*] Cycle {i + 1}/{args.cycles}...")
                result = pipeline.run_cycle()
                if result.success:
                    t = [round(x, 4) for x in result.target_pose_base.translation]
                    print(f"[+] Cycle completed in {result.elapsed_ms:.1f}ms, grasp pose in arm base: {t}m")
                    for w in result.warnings:
                        print(f"    warning: {w}")
                else:
                    print(f"[-] Cycle failed at stage '{result.stage}': {result.error_message}")
                    break
    except KeyboardInterrupt:
        print("[!] Interrupted: sending e-stop")
        arm.e_stop()
    except ArmdError as e:
        print(f"[!] rebot-armd error: {e}")


if __name__ == "__main__":
    main()
