"""rebot-gripper-calib: interactive gripper calibration CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence

from rebot import Arm, ArmdClient


def check_monotonic(points: Sequence[Sequence[float]]) -> bool:
    """Check that jaw opening points are strictly monotonic and valid.

    points: list of [j7_rad, opening_m], assumed sorted by j7.
    Requires at least 2 points. All values must be finite.
    Openings must be in [0, 0.2] m.
    j7 differences must have the same sign (and abs >= 1e-6).
    Opening differences must have the same sign (and abs >= 1e-6).
    """
    if len(points) < 2:
        return False
    for p in points:
        if len(p) < 2:
            return False
        j7, w = p[0], p[1]
        if not math.isfinite(j7) or not math.isfinite(w):
            return False
        if not (0.0 <= w <= 0.2):
            return False

    d_j7_0 = points[1][0] - points[0][0]
    d_w_0 = points[1][1] - points[0][1]
    if abs(d_j7_0) < 1e-6 or abs(d_w_0) < 1e-6:
        return False

    for i in range(len(points) - 1):
        dj = points[i + 1][0] - points[i][0]
        dw = points[i + 1][1] - points[i][1]
        if abs(dj) < 1e-6 or abs(dw) < 1e-6:
            return False
        if (dj > 0.0) != (d_j7_0 > 0.0) or (dw > 0.0) != (d_w_0 > 0.0):
            return False

    return True


def fit_opening_curve(points: Sequence[Sequence[float]]) -> dict:
    """Fit a linear curve opening_m = slope * j7_rad + intercept.

    Returns dict with:
        slope: float
        intercept: float
        residuals_mm: list of float (absolute error in mm)
        max_residual_mm: float
        max_opening_m: float
        suggested_max_grasp_width_m: float (max_opening_m - 0.015)
        warn_residual: bool (max_residual_mm > 2.0)
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required for fitting")

    j7_vals = [float(p[0]) for p in points]
    w_vals = [float(p[1]) for p in points]
    n = len(points)

    mean_x = sum(j7_vals) / n
    mean_y = sum(w_vals) / n
    var_x = sum((x - mean_x) ** 2 for x in j7_vals)
    if var_x < 1e-12:
        raise ValueError("Cannot fit line: j7 values have zero variance")

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(j7_vals, w_vals))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    residuals_mm = [abs(w - (slope * j7 + intercept)) * 1000.0 for j7, w in zip(j7_vals, w_vals)]
    max_residual_mm = max(residuals_mm)
    max_opening_m = max(w_vals)
    suggested_max_grasp_width_m = max(0.0, max_opening_m - 0.015)

    return {
        "slope": slope,
        "intercept": intercept,
        "residuals_mm": residuals_mm,
        "max_residual_mm": max_residual_mm,
        "max_opening_m": max_opening_m,
        "suggested_max_grasp_width_m": suggested_max_grasp_width_m,
        "warn_residual": max_residual_mm > 2.0,
    }


def parse_caliper_input(text: str) -> tuple[str, float | None]:
    """Parse caliper input string.

    Returns (status, value_mm):
    - ("ok", value_mm) if float in [0.0, 200.0]
    - ("skip", None) if 's' or 'S'
    - ("abort", None) if 'q' or 'Q'
    - ("invalid", None) otherwise
    """
    s = text.strip()
    if not s:
        return "invalid", None
    if s.lower() == "s":
        return "skip", None
    if s.lower() == "q":
        return "abort", None
    try:
        val = float(s)
        if 0.0 <= val <= 200.0:
            return "ok", val
        return "invalid", None
    except ValueError:
        return "invalid", None


def compute_fractions(n: int) -> list[float]:
    """Compute N opening fractions from 1.0 down to 0.0.

    Default for N=6: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    """
    if n <= 1:
        return [1.0]
    return [round(1.0 - i / (n - 1), 4) for i in range(n)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rebot-gripper-calib",
        description="Interactive gripper zero check and jaw opening calibration for reBot B601-DM.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="armd host or URL (default: $REBOT_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=6,
        help="Number of calibration positions from open (1.0) to closed (0.0) (default: 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform measurements and fit without storing calibration to device",
    )

    args = parser.parse_args(argv)

    host = args.host or os.environ.get("REBOT_HOST") or "127.0.0.1"
    url = f"http://{host}:8765" if "://" not in host and ":" not in host else host

    print("=== reBot Gripper Calibration ===")
    print(f"Connecting to armd at {url}...")

    # Step 1: Connect and show current calibration & observation.gripper_zero
    try:
        client = ArmdClient(url)
        calib = client.get_gripper_calibration()
        print("\nCurrent calibration (GET /api/v1/gripper/calibration):")
        print(json.dumps(calib, indent=2))
    except Exception as exc:
        print(f"\nCould not fetch current calibration: {exc}")

    try:
        obs = client.get_observation()
        print(f"\nobservation.gripper_zero: {obs.gripper_zero}")
    except Exception as exc:
        print(f"\nCould not fetch observation: {exc}")

    print("\nSafety notes:")
    print("  * The arm body does not move during this procedure.")
    print("  * Keep fingers and tools clear of the gripper jaws.")
    print("  * Press Ctrl+C at any time to stop the arm and release control.")
    print("  * Nothing is stored without explicit confirmation.")

    ans = input("\nContinue? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted.")
        return 0

    # Step 2: Take control with Arm()
    print("\nConnecting with control lease...")
    chosen_offset: float | None = None
    with Arm(host=args.host, enable=True) as arm:
        # Step 3: Zero check
        print("\n--- Step 1: Gripper zero check ---")
        print("Running mechanical zero check (empty jaws)...")
        res = arm.gripper_zero_check(store_offset=False, threshold_rad=0.10)
        stop_rad = res.stop_rad
        offset_rad = res.offset_rad
        is_pass = res.passed
        print(f"Zero check result: stop_rad={stop_rad:.4f} rad, offset_rad={offset_rad:.4f} rad, pass={is_pass}")

        ans = input("Store this offset? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            chosen_offset = offset_rad
            if args.dry_run:
                print("[dry-run] Skipping offset write.")
            else:
                print(f"Storing offset {offset_rad:.4f} rad...")
                arm.set_gripper_calibration(offset_rad=chosen_offset)
                print("Offset stored.")

        # Step 4: Opening table
        print("\n--- Step 2: Jaw opening calibration ---")
        fractions = compute_fractions(args.points)
        raw_points: list[list[float]] = []

        for i, frac in enumerate(fractions):
            print(f"\nMoving gripper to fraction {frac:.2f} ({i + 1}/{len(fractions)})...")
            arm.gripper_move(frac)
            time.sleep(0.5)
            st = arm.state()
            j7_rad = st.positions[6]
            print(f"Position reached. Measured J7 = {j7_rad:.4f} rad")
            while True:
                prompt = input("Caliper opening at the jaw tips in mm [number / s (skip) / q (abort)]: ").strip()
                status, val_mm = parse_caliper_input(prompt)
                if status == "ok":
                    assert val_mm is not None
                    opening_m = val_mm / 1000.0
                    raw_points.append([j7_rad, opening_m])
                    break
                elif status == "skip":
                    print(f"Skipped fraction {frac:.2f}.")
                    break
                elif status == "abort":
                    print("Aborted by user.")
                    return 0
                else:
                    print("Invalid input. Enter a number in mm (0-200), 's' to skip, or 'q' to abort.")

        # Step 5: Fit and validate
        if len(raw_points) < 2:
            print("\nError: At least 2 points are required for calibration. Aborting without storing.")
            return 1

        sorted_points = sorted(raw_points, key=lambda p: p[0])

        if not check_monotonic(sorted_points):
            print("\nError: Measured jaw openings are not strictly monotonic. Aborting without storing.")
            return 1

        try:
            fit = fit_opening_curve(sorted_points)
        except Exception as exc:
            print(f"\nError during curve fitting: {exc}")
            return 1

        print("\n=== Calibration Summary Table ===")
        print(f"{'#':<3} {'J7 (rad)':<12} {'Measured (mm)':<15} {'Fitted (mm)':<15} {'Residual (mm)':<15}")
        print("-" * 65)
        for idx, (p, r_mm) in enumerate(zip(sorted_points, fit["residuals_mm"])):
            j7_v, w_m = p[0], p[1]
            w_meas_mm = w_m * 1000.0
            w_fit_mm = (fit["slope"] * j7_v + fit["intercept"]) * 1000.0
            print(f"{idx + 1:<3} {j7_v:<12.4f} {w_meas_mm:<15.2f} {w_fit_mm:<15.2f} {r_mm:<15.2f}")

        print(f"\nMax residual: {fit['max_residual_mm']:.2f} mm")
        if fit["warn_residual"]:
            print(f"WARNING: Max residual ({fit['max_residual_mm']:.2f} mm) exceeds 2.0 mm!")

        print(f"Max opening: {fit['max_opening_m'] * 1000.0:.1f} mm ({fit['max_opening_m']:.4f} m)")
        print(
            f"Suggested max_grasp_width_m: {fit['suggested_max_grasp_width_m'] * 1000.0:.1f} mm "
            f"({fit['suggested_max_grasp_width_m']:.4f} m)"
        )

        # Step 6: Confirm and store
        ans = input("\nStore this calibration? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            if args.dry_run:
                print("[dry-run] Skipping calibration write.")
            else:
                print("Writing calibration to device...")
                arm.set_gripper_calibration(points=sorted_points, offset_rad=chosen_offset)
                print("Calibration written. Reading back from device...")
                readback = arm.gripper_calibration()
                print(f"Device calibration:\n{json.dumps(readback, indent=2)}")
        else:
            print("Calibration not stored.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
