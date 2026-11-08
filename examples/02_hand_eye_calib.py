#!/usr/bin/env python3
"""Example 2: interactive hand-eye calibration wizard.

Each capture reads the arm joints from rebot-armd (REST, no lease), triggers one reCamera ArUco
detection, computes the TCP pose with the B601-DM forward kinematics and stores the pair. Samples are
written to ``--samples`` after every capture, so an interrupted session can be resumed (``--resume``).

Methods (see README "Hand-eye calibration"):
  point  eye-to-hand, marker centre at the TCP. Same result as the rebot-armd Web cockpit.
  axxb   AX=XB (Park & Martin). Marker anywhere rigid on the gripper (eye_to_hand) or fixed in the
         scene (eye_in_hand). Needs orientation changes about >= 2 axes (J1 yaw AND wrist pitch).

Re-solve a Web cockpit export:  uv run python examples/02_hand_eye_calib.py --from-web-export rebot_hand_eye_*.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from rebot_companion import ArmdClient, CalibrationError, HandEyeCalibration, ReCameraClient, kinematics
from rebot_companion.calibration import rotation_spread_deg

GUIDE = """
Pose the arm by hand or with teleop (e.g. the rebot-armd Web cockpit, or torque off and move it
manually), hold it still, then press Enter to capture. Recommendations:
  * 9+ samples spread over the camera view and over depth (near / far), not on one line or plane.
  * keep the marker fully visible and facing the camera (< 60 deg tilt).
  * method=point: the marker centre must sit on the TCP (gripper tip); keep J5 at 0.
  * method=axxb : change J1 yaw AND wrist pitch between samples (>= 15 deg each), not only position.
"""


def parse_csv_floats(text: str | None, n: int, scale: float = 1.0) -> list[float] | None:
    if not text:
        return None
    vals = [float(v) * scale for v in text.split(",")]
    if len(vals) != n:
        raise SystemExit(f"expected {n} comma-separated values, got {text!r}")
    return vals


def print_result(calib: HandEyeCalibration) -> None:
    r = calib.result
    assert r is not None
    has_rot = r.rot_residuals_deg is not None
    print("\n  #   TCP x,y,z [mm]          Zc [m]   pos res [mm]" + ("   rot res [deg]" if has_rot else ""))
    for i, s in enumerate(calib.samples):
        res = r.residuals[i]
        bad = res > 2.5 * r.rmse and res > 0.005
        line = (
            f"{i + 1:3d}   {','.join(f'{v * 1000:7.1f}' for v in s.p_base):24s} {s.p_cam[2]:6.3f}   {res * 1000:10.2f}"
        )
        if has_rot:
            line += f"   {r.rot_residuals_deg[i]:13.2f}"
        print(line + ("   <-- outlier, consider re-capturing" if bad else ""))
    T = r.transform
    name = "T_cam_to_base" if calib.mode == "eye_to_hand" else "T_cam_to_gripper"
    print(f"\n{name} (p_target = R p_cam + t):")
    print(np.array2string(T.matrix, precision=5, suppress_small=True))
    print("translation [mm]: " + ", ".join(f"{v * 1000:.1f}" for v in T.translation))
    print("rpy [deg] (R = Rz(yaw) Ry(pitch) Rx(roll)): " + ", ".join(f"{v:.2f}" for v in T.to_euler_deg()))
    print(f"RMSE {r.rmse * 1000:.2f} mm, max {r.max_error * 1000:.2f} mm, n = {r.n}")
    if has_rot:
        print(f"rotation RMSE {r.rot_rmse_deg:.2f} deg")
        if r.marker_offset is not None:
            what = "marker -> gripper offset" if calib.mode == "eye_to_hand" else "marker pose in base"
            print(f"{what}: t [mm] = " + ", ".join(f"{v * 1000:.1f}" for v in r.marker_offset.translation))
    if r.rmse > 0.01:
        print("WARNING: RMSE > 10 mm. Check the marker size / intrinsics on the camera, TCP assumption and outliers.")


def solve_and_save(calib: HandEyeCalibration, out: Path) -> int:
    try:
        calib.solve()
    except CalibrationError as exc:
        print(f"[!] cannot solve: {exc}")
        return 1
    print_result(calib)
    calib.save_to_file(out)
    print(f"\n[+] saved {out}")
    return 0


def velocities_of(obs) -> list[float]:
    v = getattr(obs, "joint_velocities", None)
    if v is None:
        v = getattr(obs, "velocities", [])
    return list(v or [])


def capture(args, arm: ArmdClient, cam: ReCameraClient, calib: HandEyeCalibration, sign, offset) -> str:
    """Capture one sample. Returns a status message."""
    obs = arm.get_observation()
    pos = list(obs.joint_positions)
    if len(pos) < 4:
        return f"rejected: observation has {len(pos)} joints (< 4); is armd connected to the arm?"
    if any(abs(v) > args.vel_threshold for v in velocities_of(obs)):
        return "rejected: arm is moving; hold it still and retry"
    if len(pos) > 4 and abs(pos[4]) > math.radians(1.0) and not args.allow_j5:
        return (
            f"rejected: J5 = {math.degrees(pos[4]):.1f} deg; the FK model assumes J5 = 0 (use --allow-j5 to override)"
        )

    markers = cam.trigger(wait_ms=args.trigger_wait_ms)
    if args.marker_id is not None:
        markers = [m for m in markers if m.id == args.marker_id]
    elif calib.marker_id is not None:
        markers = [m for m in markers if m.id == calib.marker_id]
    if not markers:
        want = args.marker_id if args.marker_id is not None else calib.marker_id
        return "rejected: no marker detected" + (f" with id {want}" if want is not None else "")
    m = markers[0]

    obs2 = arm.get_observation()
    pos2 = list(obs2.joint_positions)
    if len(pos2) < 4 or max(abs(a - b) for a, b in zip(pos[:4], pos2[:4])) > 0.002:
        return "rejected: arm moved during detection; retry"
    if any(abs(v) > args.vel_threshold for v in velocities_of(obs2)):
        return "rejected: arm is moving; hold it still and retry"

    G = kinematics.tcp_pose_from_joints(pos, sign, offset)
    for i, s in enumerate(calib.samples):
        if np.linalg.norm(s.p_base - G.translation) < 0.01:
            return f"rejected: TCP within 10 mm of sample #{i + 1}; move further"
    if calib.marker_id is None:
        calib.marker_id = m.id
    tcp = kinematics.fk(kinematics.to_model(pos, sign, offset))
    calib.add_sample(
        G,
        m.pose,
        joints=pos,
        marker_id=m.id,
        tcp_pitch=tcp.pitch,
        marker_reprojection_error=m.reprojection_error,
        corners=m.corners,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    calib.save_to_file(args.samples, allow_unsolved=True)
    s = calib.samples[-1]
    return (
        f"captured #{len(calib.samples)}: marker {m.id} at Zc {s.p_cam[2]:.3f} m, "
        f"TCP [mm] {', '.join(f'{v * 1000:.0f}' for v in s.p_base)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Hand-eye calibration wizard (reCamera + rebot-armd)")
    ap.add_argument("--camera-url", default="http://192.168.42.1:8090", help="reCamera aruco-detector HTTP API")
    ap.add_argument("--arm-url", default="http://127.0.0.1:8765", help="rebot-armd REST API")
    ap.add_argument("--mode", default="eye_to_hand", choices=["eye_to_hand", "eye_in_hand"])
    ap.add_argument("--method", default="point", choices=["point", "axxb"])
    ap.add_argument("--marker-id", type=int, default=None, help="ArUco id to use (default: first one seen)")
    ap.add_argument("--target", type=int, default=9, help="recommended number of samples")
    ap.add_argument("--samples", type=Path, default=Path("calib_samples.json"), help="incremental sample file")
    ap.add_argument("--resume", action="store_true", help="continue from the samples in --samples")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"), help="output calibration file")
    ap.add_argument("--from-web-export", type=Path, default=None, help="re-solve samples from a Web cockpit export")
    ap.add_argument("--vel-threshold", type=float, default=0.02, help="rad/s; reject capture above this")
    ap.add_argument("--trigger-wait-ms", type=int, default=2500)
    ap.add_argument("--allow-j5", action="store_true", help="accept samples with J5 != 0 (FK ignores J5)")
    ap.add_argument("--iksign", default=None, help="J1..J4 motor sign, e.g. 1,1,1,1 (same as Web ?iksign=)")
    ap.add_argument("--ikoff", default=None, help="J1..J4 motor offset in degrees (same as Web ?ikoff=)")
    args = ap.parse_args()

    if args.method == "point" and args.mode != "eye_to_hand":
        ap.error("--method point supports --mode eye_to_hand only; use --method axxb")
    sign = parse_csv_floats(args.iksign, 4) or list(kinematics.DEFAULT_SIGN)
    offset = parse_csv_floats(args.ikoff, 4, math.pi / 180) or list(kinematics.DEFAULT_OFFSET)

    if args.from_web_export:
        calib = HandEyeCalibration.load_samples(args.from_web_export, args.method, args.mode, sign, offset)
        print(f"[*] {len(calib.samples)} samples from {args.from_web_export} (method {args.method})")
        return solve_and_save(calib, args.out)

    if args.resume and args.samples.exists():
        calib = HandEyeCalibration.load_samples(args.samples, args.method, args.mode, sign, offset)
        print(f"[*] resumed {len(calib.samples)} samples from {args.samples}")
    else:
        calib = HandEyeCalibration(mode=args.mode, method=args.method)

    arm = ArmdClient(args.arm_url)
    cam = ReCameraClient(args.camera_url)
    try:
        obs = arm.get_observation()
        print(f"[*] armd OK, {len(obs.joint_positions)} joints")
        cam.get_inference_state()
        print("[*] reCamera OK")
    except Exception as exc:  # noqa: BLE001 - surface any connectivity error to the user
        print(f"[!] connectivity check failed: {exc}")
        return 1

    print(f"=== {args.mode} / {args.method} calibration ===")
    print(GUIDE)
    while True:
        n = len(calib.samples)
        try:
            cmd = input(f"[{n}/{args.target}] Enter=capture  u=undo  s=solve  d=done  q=quit > ").strip().lower()
        except EOFError:
            cmd = "q"
        if cmd == "q":
            print(f"Samples kept in {args.samples}")
            return 0
        if cmd == "u":
            if calib.samples:
                calib.samples.pop()
                calib.result = None
                calib.save_to_file(args.samples, allow_unsolved=True)
                print("removed last sample")
            continue
        if cmd in ("s", "d", "done"):
            if n < 3:
                print("need at least 3 samples")
                continue
            if args.method == "axxb":
                print(f"gripper rotation spread: {rotation_spread_deg(calib.samples):.1f} deg")
            if cmd == "s":
                try:
                    calib.solve()
                    print_result(calib)
                except CalibrationError as exc:
                    print(f"[!] {exc}")
                continue
            if n < args.target:
                print(f"note: {n} < {args.target} recommended samples")
            return solve_and_save(calib, args.out)
        if cmd == "":
            try:
                print(capture(args, arm, cam, calib, sign, offset))
            except Exception as exc:  # noqa: BLE001 - keep the wizard alive on transient HTTP errors
                print(f"[!] capture failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
