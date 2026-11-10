#!/usr/bin/env python3
"""
Example 4: Stage 2 - Hardened Deterministic Local Pipeline.
Runs a repeatable, high-reliability pick and place cycle directly on the factory/lab client.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rebot import ArmdError
from rebot_companion import (
    ArmdClient,
    B601IkAdapter,
    DeterministicPickPlacePipeline,
    HandEyeCalibration,
    HandEyeResult,
    ReCameraClient,
    Transform,
)
from rebot_companion.pipeline import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Hardened Pick & Place Loop")
    parser.add_argument("--camera-url", default="http://192.168.42.1:8090")
    parser.add_argument("--arm-url", default="http://127.0.0.1:8765")
    parser.add_argument("--calib", default="calibration.json", help="Path to hand-eye calibration JSON")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1, help="cycles to run under one control lease")
    parser.add_argument(
        "--dry-run", action="store_true", help="detect and check the target only; no IK, the arm does not move"
    )
    args = parser.parse_args()

    # Load calibration (setup + transform). eye_in_hand files are combined with FK at detection time.
    calib_path = Path(args.calib)
    if calib_path.exists():
        hand_eye = HandEyeCalibration.load_result(calib_path)
        print(f"[*] Loaded {hand_eye.setup} calibration from {calib_path}")
    elif args.dry_run:
        # PLACEHOLDER, not a calibration: base-frame targets are not real. Dry runs only.
        print(f"[!] {calib_path} not found: dry run with a placeholder transform; target positions are not real.")
        hand_eye = HandEyeResult("eye_to_hand", Transform.from_translation_euler([0.30, -0.40, 0.50], 180.0, 0.0, 90.0))
    else:
        parser.error(f"{calib_path} not found: run examples/02_hand_eye_calib.py first, or pass --dry-run")

    camera = ReCameraClient(args.camera_url)
    arm = ArmdClient(args.arm_url)
    config = PipelineConfig(target_marker_id=args.marker_id)

    obs = arm.get_observation()
    print(f"[*] Arm state: {obs.safety_state} (epoch {obs.epoch_id}, torqued={obs.is_torqued})")
    print(f"=== Stage 2: Hardened Local Pick & Place Pipeline (Target Marker #{args.marker_id}) ===")

    # B601IkAdapter: J1-J4 from analytic IK (top-down tool), J5 = 0, J6 = home, J7 = gripper.
    ik_solver = None if args.dry_run else B601IkAdapter()
    session = arm.session()
    try:
        # One lease for all cycles: the session heartbeats in the background, and
        # armd stops motion if this process dies or the connection drops. On an
        # exception or Ctrl-C the session unloads a torqued arm before releasing the lease.
        with session:
            pipeline = DeterministicPickPlacePipeline(
                camera, arm, hand_eye, config, ik_solver=ik_solver, session=session
            )
            for i in range(args.cycles):
                print(f"[*] Cycle {i + 1}/{args.cycles}...")
                result = pipeline.run_cycle()
                t = [round(x, 4) for x in result.target_pose_base.translation] if result.target_pose_base else None
                if result.success:
                    print(f"[+] Cycle completed in {result.elapsed_ms:.1f}ms, grasp pose in arm base: {t}m")
                elif result.stage == "dry_run":
                    print(f"[=] Dry run: target in arm base {t}m passed the workspace check; arm not moved")
                else:
                    print(f"[-] Cycle failed at stage '{result.stage}': {result.error_message}")
                for w in result.warnings:
                    print(f"    warning: {w}")
                if not result.success:
                    break
    except KeyboardInterrupt:
        outcome = session.safe_exit_outcome
        print(f"[!] Interrupted; safe exit: {outcome.action + ' ' + outcome.message if outcome else 'not run'}")
    except ArmdError as e:
        print(f"[!] rebot-armd error: {e}")


if __name__ == "__main__":
    main()
