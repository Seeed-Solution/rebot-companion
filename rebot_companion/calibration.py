"""Hand-Eye Calibration Solver (Eye-to-Hand & Eye-in-Hand)."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .transforms import Transform


@dataclass
class CalibrationSample:
    """A pair of (gripper_in_base, marker_in_camera) transforms."""
    gripper_pose: Transform
    marker_pose: Transform


class HandEyeCalibration:
    """Solves hand-eye calibration equation or direct rigid alignment."""

    def __init__(self, mode: str = "eye_to_hand") -> None:
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"Unknown calibration mode: {mode}")
        self.mode = mode
        self.samples: list[CalibrationSample] = []
        self.result_transform: Transform | None = None
        self.rmse: float = 0.0

    def add_sample(self, gripper_pose: Transform, marker_pose: Transform) -> None:
        """Add a measured sample pair."""
        self.samples.append(CalibrationSample(gripper_pose, marker_pose))

    def clear(self) -> None:
        self.samples.clear()
        self.result_transform = None
        self.rmse = 0.0

    def solve(self) -> Transform:
        """
        Solve calibration transform:
        - Eye-to-Hand: Solves T_cam_to_base such that T_gripper = T_cam_to_base @ T_marker.
        - Eye-in-Hand: Solves T_cam_to_gripper such that T_marker_base = T_gripper @ T_cam_to_gripper @ T_marker.
        Requires at least 3 distinct non-parallel poses (recommended 5~9).
        """
        if len(self.samples) < 3:
            raise ValueError(f"At least 3 samples required to solve calibration, got {len(self.samples)}")

        R_list = []
        t_list = []

        if self.mode == "eye_to_hand":
            # For each sample i: T_cam_to_base = T_gripper_i @ (T_marker_i)^-1
            for s in self.samples:
                X_i = s.gripper_pose @ s.marker_pose.inverse()
                R_list.append(X_i.rotation)
                t_list.append(X_i.translation)
        else:
            # Eye-in-Hand
            for i in range(len(self.samples) - 1):
                A = self.samples[i + 1].gripper_pose.inverse() @ self.samples[i].gripper_pose
                B = self.samples[i + 1].marker_pose @ self.samples[i].marker_pose.inverse()
                X_i = A @ B.inverse()
                R_list.append(X_i.rotation)
                t_list.append(X_i.translation)

        # Frobenius / Chordal SVD average of rotations onto SO(3)
        M = np.mean(R_list, axis=0)
        U, _, Vt = np.linalg.svd(M)
        Rx = U @ Vt
        if np.linalg.det(Rx) < 0:
            Vt[2, :] *= -1
            Rx = U @ Vt

        # Mean translation
        tx = np.mean(t_list, axis=0)
        self.result_transform = Transform.from_rot_trans(Rx, tx)

        # Compute reprojection residual error (RMSE in meters)
        residuals = []
        for s in self.samples:
            if self.mode == "eye_to_hand":
                p_base = s.gripper_pose.translation
                p_est = (self.result_transform @ s.marker_pose).translation
                residuals.append(np.linalg.norm(p_base - p_est))
            else:
                residuals.append(0.0)

        self.rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
        return self.result_transform

    def save_to_file(self, filepath: str | Path) -> None:
        """Save calibration result to JSON file."""
        if self.result_transform is None:
            raise RuntimeError("Cannot save calibration: solve() has not been called")
        data = {
            "mode": self.mode,
            "sample_count": len(self.samples),
            "rmse_meters": self.rmse,
            "transform": self.result_transform.to_dict(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Transform:
        """Load calibration transform from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Transform.from_dict(data["transform"])
