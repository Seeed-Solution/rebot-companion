"""Rigid body 3D transformation utilities (SE3)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Same convention as the reCamera device (aruco-detector rot_to_euler) and the rebot-armd Web cockpit.
EULER_CONVENTION = "R = Rz(yaw) Ry(pitch) Rx(roll)"


class Transform:
    """Represents a 3D rigid body transformation SE(3) with a 4x4 homogeneous matrix."""

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        if matrix is None:
            self.matrix = np.eye(4, dtype=np.float64)
        else:
            m = np.asarray(matrix, dtype=np.float64)
            if m.shape != (4, 4):
                raise ValueError(f"Transform matrix must be 4x4, got {m.shape}")
            self.matrix = m

    @classmethod
    def from_translation_euler(
        cls,
        translation: Sequence[float],
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> Transform:
        """Create SE(3) transform from translation [x, y, z] and Euler angles in degrees (roll-pitch-yaw Z-Y-X)."""
        r = math.radians(roll_deg)
        p = math.radians(pitch_deg)
        y = math.radians(yaw_deg)

        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = translation
        return cls(mat)

    @classmethod
    def from_rot_trans(cls, R: np.ndarray, t: np.ndarray) -> Transform:
        """Create SE(3) from 3x3 rotation matrix and 3D translation vector."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = np.asarray(t).reshape(3)
        return cls(mat)

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]] | np.ndarray) -> Transform:
        """Create from a 4x4 homogeneous matrix; the rotation block is re-orthonormalised (SVD)."""
        m = np.array(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {m.shape}")
        U, _, Vt = np.linalg.svd(m[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, 2] *= -1
            R = U @ Vt
        return cls.from_rot_trans(R, m[:3, 3])

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    def to_euler_deg(self) -> tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
        R = self.rotation
        r20 = R[2, 0]
        pitch = math.asin(-np.clip(r20, -1.0, 1.0))
        if abs(r20) < 0.999999:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            yaw = 0.0
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def inverse(self) -> Transform:
        """Calculate inverse transform T^-1 = [R^T, -R^T * t]."""
        R = self.rotation
        t = self.translation
        inv_mat = np.eye(4, dtype=np.float64)
        inv_mat[:3, :3] = R.T
        inv_mat[:3, 3] = -R.T @ t
        return Transform(inv_mat)

    def __matmul__(self, other: Transform) -> Transform:
        """Compose two transforms: T_composed = T1 @ T2."""
        return Transform(self.matrix @ other.matrix)

    def apply_point(self, point: Sequence[float]) -> np.ndarray:
        """Transform a 3D point (x, y, z) into the target coordinate frame."""
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
        res = self.matrix @ p
        return res[:3]

    def to_dict(self) -> dict:
        r, p, y = self.to_euler_deg()
        return {
            "translation": self.translation.tolist(),
            "euler_deg": [r, p, y],
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transform:
        if "matrix" in data:
            return cls(np.array(data["matrix"]))
        t = data.get("translation", [0.0, 0.0, 0.0])
        e = data.get("euler_deg", [0.0, 0.0, 0.0])
        return cls.from_translation_euler(t, e[0], e[1], e[2])

    def __repr__(self) -> str:
        t = [round(x, 4) for x in self.translation]
        r, p, y = [round(x, 2) for x in self.to_euler_deg()]
        return f"Transform(t={t}, rpy_deg=[{r}, {p}, {y}])"
