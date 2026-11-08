"""Hand-eye calibration solvers.

Frames and notation (``T_a_b`` maps coordinates in frame ``b`` into frame ``a``):

* ``G_i = T_base_gripper`` -- gripper (TCP) pose in the arm base frame, from forward kinematics.
* ``M_i = T_cam_marker``  -- ArUco marker pose in the camera frame, from the reCamera detector.

Methods
-------
``method="point"`` (default, eye-to-hand only)
    Assumes the marker centre coincides with the TCP. Solves ``p_base = R p_cam + t`` over the point
    pairs with Horn's closed-form quaternion method -- the same algorithm and result as the rebot-armd
    Web cockpit ``solveRigid``. Only marker/TCP positions are used, so marker orientation noise does not
    matter. Result: ``X = T_base_cam``.

``method="axxb"``
    Park & Martin (1994) solution of ``A X = X B`` over all sample pairs: rotation by least squares on
    the rotation logarithms, translation by stacked linear least squares. Uses full poses, so the
    marker may sit anywhere rigidly on the gripper (eye-to-hand) or anywhere fixed in the scene
    (eye-in-hand). Requires rotation about at least two non-parallel axes.

    * eye_to_hand: ``X = T_base_cam``; marker rigid on the gripper with unknown offset
      ``Y = T_gripper_marker`` (estimated and returned as ``marker_offset``).
      ``A = G_i G_j^-1``, ``B = M_i M_j^-1``.
    * eye_in_hand: ``X = T_gripper_cam``; marker fixed in the base frame, ``Y = T_base_marker``
      (returned as ``marker_offset``). ``A = G_j^-1 G_i``, ``B = M_j M_i^-1``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from . import kinematics
from .transforms import EULER_CONVENTION, Transform

MODES = ("eye_to_hand", "eye_in_hand")
METHODS = ("point", "axxb")
METHOD_NAMES = {"point": "point_registration_horn", "axxb": "axxb_park_martin"}
TRANSFORM_KEYS = {"eye_to_hand": "T_cam_to_base", "eye_in_hand": "T_cam_to_gripper"}
OFFSET_KEYS = {"eye_to_hand": "T_marker_to_gripper", "eye_in_hand": "T_marker_to_base"}
ASSUMPTIONS = {
    ("eye_to_hand", "point"): "ArUco marker centre coincides with the TCP; TCP from forward kinematics (URDF b601dm)",
    ("eye_to_hand", "axxb"): "ArUco marker rigidly attached to the gripper at an unknown offset (estimated)",
    ("eye_in_hand", "axxb"): "ArUco marker fixed in the base frame; camera rigidly attached to the gripper",
}


@dataclass(frozen=True)
class HandEyeResult:
    """A solved hand-eye transform together with the setup it belongs to.

    * ``eye_to_hand``: ``transform`` is ``T_cam_to_base`` (camera fixed in the scene).
    * ``eye_in_hand``: ``transform`` is ``T_cam_to_gripper`` (camera on the gripper); the camera
      pose in the base frame depends on the arm pose, ``T_base_gripper(q) @ transform``.
    """

    setup: str
    transform: Transform

    def __post_init__(self) -> None:
        if self.setup not in MODES:
            raise ValueError(f"setup must be one of {MODES}, got {self.setup!r}")

    @property
    def needs_joints(self) -> bool:
        return self.setup == "eye_in_hand"

    def cam_to_base(
        self,
        joints: Sequence[float] | None = None,
        sign: Sequence[float] = kinematics.DEFAULT_SIGN,
        offset: Sequence[float] = kinematics.DEFAULT_OFFSET,
    ) -> Transform:
        """``T_cam_to_base``. For eye_in_hand, ``joints`` are the raw armd positions when the image was taken."""
        if self.setup == "eye_to_hand":
            return self.transform
        if joints is None:
            raise ValueError("eye_in_hand calibration needs the arm joints at detection time")
        return gripper_pose(joints, sign, offset) @ self.transform


def gripper_pose(
    joints: Sequence[float],
    sign: Sequence[float] = kinematics.DEFAULT_SIGN,
    offset: Sequence[float] = kinematics.DEFAULT_OFFSET,
) -> Transform:
    """``T_base_gripper`` from raw armd joint positions. Single FK entry point used for eye_in_hand."""
    return kinematics.tcp_pose_from_joints(joints, sign, offset)


class CalibrationError(ValueError):
    """Raised when samples are insufficient or degenerate for the selected method."""


@dataclass
class CalibrationSample:
    """One measurement: gripper pose in base and marker pose in camera.

    ``has_rotation`` is False when only positions are known (e.g. Web samples without ``R_cam``);
    such samples can be used by ``method="point"`` only.
    """

    gripper_pose: Transform
    marker_pose: Transform
    joints: list[float] | None = None
    marker_id: int | None = None
    has_rotation: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def p_base(self) -> np.ndarray:
        return self.gripper_pose.translation

    @property
    def p_cam(self) -> np.ndarray:
        return self.marker_pose.translation

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.extra)
        d.update(
            {
                "joints": self.joints,
                "p_base": self.p_base.tolist(),
                "p_cam": self.p_cam.tolist(),
                "R_cam": self.marker_pose.rotation.tolist() if self.has_rotation else None,
                "rpy_cam_deg": list(self.marker_pose.to_euler_deg()) if self.has_rotation else None,
                "marker_id": self.marker_id,
                "T_base_gripper": self.gripper_pose.matrix.tolist(),
                "T_cam_marker": self.marker_pose.matrix.tolist() if self.has_rotation else None,
            }
        )
        return d

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        sign: Sequence[float] = kinematics.DEFAULT_SIGN,
        offset: Sequence[float] = kinematics.DEFAULT_OFFSET,
    ) -> CalibrationSample:
        """Parse an SDK sample or a rebot-armd Web export sample.

        Gripper rotation comes from ``T_base_gripper`` or, for Web samples, from FK of ``joints``;
        the translation is always the stored ``p_base`` so a re-solve reproduces the Web result.
        """
        joints = [float(v) for v in d["joints"]] if d.get("joints") is not None else None
        if d.get("T_base_gripper") is not None:
            G = Transform.from_matrix(d["T_base_gripper"])
        elif joints is not None and len(joints) >= 4:
            G = Transform.from_rot_trans(kinematics.tcp_pose_from_joints(joints, sign, offset).rotation, d["p_base"])
        else:
            G = Transform.from_rot_trans(np.eye(3), d["p_base"])
        has_rot = True
        if d.get("T_cam_marker") is not None:
            M = Transform.from_matrix(d["T_cam_marker"])
        elif d.get("R_cam") is not None:
            M = Transform.from_matrix(_hom(np.asarray(d["R_cam"], dtype=float), d["p_cam"]))
        elif d.get("rpy_cam_deg") is not None:
            r, p, y = d["rpy_cam_deg"]
            M = Transform.from_translation_euler(d["p_cam"], r, p, y)
        else:
            M = Transform.from_rot_trans(np.eye(3), d["p_cam"])
            has_rot = False
        if d.get("p_base") is not None:
            G = Transform.from_rot_trans(G.rotation, d["p_base"])
        known = {"joints", "p_base", "p_cam", "R_cam", "rpy_cam_deg", "marker_id", "T_base_gripper", "T_cam_marker"}
        extra = {k: v for k, v in d.items() if k not in known}
        mid = d.get("marker_id")
        return cls(G, M, joints, int(mid) if mid is not None else None, has_rot, extra)


@dataclass
class CalibrationResult:
    mode: str
    method: str
    transform: Transform
    rmse: float  # m; point method: 3D point residual; axxb: marker position prediction residual
    max_error: float  # m
    residuals: list[float]  # m, per sample
    rot_residuals_deg: list[float] | None = None  # axxb only, per sample
    rot_rmse_deg: float | None = None
    marker_offset: Transform | None = None  # axxb only, see module docstring
    n: int = 0
    solved_at: str = ""


def _hom(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return m


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def _max_triangle_area(P: np.ndarray) -> float:
    best = 0.0
    for i, j, k in itertools.combinations(range(len(P)), 3):
        best = max(best, float(np.linalg.norm(np.cross(P[j] - P[i], P[k] - P[i]))) / 2)
    return best


def _horn_quaternion(Ac: np.ndarray, Bc: np.ndarray) -> np.ndarray:
    """Horn (1987): rotation R minimising sum |R a_i - b_i|^2 for centred point sets."""
    M = Ac.T @ Bc
    (Sxx, Sxy, Sxz), (Syx, Syy, Syz), (Szx, Szy, Szz) = M
    N = np.array(
        [
            [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
            [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
            [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
            [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
        ]
    )
    vals, vecs = np.linalg.eigh(N)
    w, x, y, z = vecs[:, int(np.argmax(vals))]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def solve_point_registration(p_cam: Sequence[Sequence[float]], p_base: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Rigid fit ``p_base = R p_cam + t`` (Horn). Mirrors the Web ``solveRigid`` incl. its checks."""
    cam = np.asarray(p_cam, dtype=float)
    base = np.asarray(p_base, dtype=float)
    n = len(cam)
    if n < 3 or len(base) != n:
        raise CalibrationError(f"point registration needs >= 3 paired points, got {n}")
    ca, cb = cam.mean(axis=0), base.mean(axis=0)
    Ac, Bc = cam - ca, base - cb
    if _max_triangle_area(Ac) < 1e-4:  # same threshold as the Web: 1 cm^2
        raise CalibrationError("degenerate point set: camera points are (nearly) collinear")
    R = _horn_quaternion(Ac, Bc)
    t = cb - R @ ca
    res = np.linalg.norm(cam @ R.T + t - base, axis=1)
    return {"R": R, "t": t, "residuals": res, "rmse": float(np.sqrt(np.mean(res**2))), "max_error": float(res.max())}


def _check_rotation_diversity(rotvecs: np.ndarray, min_rotation_deg: float) -> None:
    angles = np.degrees(np.linalg.norm(rotvecs, axis=1))
    big = rotvecs[angles >= min_rotation_deg]
    if len(big) < 2:
        raise CalibrationError(
            f"insufficient rotation: fewer than 2 relative motions rotate >= {min_rotation_deg} deg; "
            "vary the gripper orientation (J1 yaw and wrist pitch) between samples"
        )
    axes = big / np.linalg.norm(big, axis=1, keepdims=True)
    s = np.linalg.svd(axes, compute_uv=False)
    if s[1] / s[0] < 0.1:
        raise CalibrationError(
            "insufficient rotation diversity: all relative rotations are about (nearly) parallel axes; "
            "rotate about at least two different axes (e.g. change both J1 yaw and wrist pitch)"
        )


def solve_axxb(A_list: Sequence[Transform], B_list: Sequence[Transform], min_rotation_deg: float = 5.0) -> Transform:
    """Park & Martin AX = XB. Rotation: Kabsch on (log B_k, log A_k); translation: stacked LS."""
    if len(A_list) < 2:
        raise CalibrationError("AX=XB needs at least 2 relative motions (3 samples)")
    alpha = np.array([Rotation.from_matrix(A.rotation).as_rotvec() for A in A_list])
    beta = np.array([Rotation.from_matrix(B.rotation).as_rotvec() for B in B_list])
    _check_rotation_diversity(alpha, min_rotation_deg)
    # alpha_k = Rx beta_k  ->  orthogonal Procrustes (works with rank-2 data)
    H = beta.T @ alpha
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rx = Vt.T @ D @ U.T
    C = np.vstack([A.rotation - np.eye(3) for A in A_list])
    d = np.concatenate([Rx @ B.translation - A.translation for A, B in zip(A_list, B_list)])
    tx, *_ = np.linalg.lstsq(C, d, rcond=None)
    return Transform.from_rot_trans(Rx, tx)


def _mean_transform(Ts: Sequence[Transform]) -> Transform:
    R = Rotation.from_matrix(np.array([T.rotation for T in Ts])).mean().as_matrix()
    return Transform.from_rot_trans(R, np.mean([T.translation for T in Ts], axis=0))


class HandEyeCalibration:
    """Collects samples and solves the hand-eye transform. See the module docstring for conventions."""

    def __init__(self, mode: str = "eye_to_hand", method: str = "point", min_rotation_deg: float = 5.0) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown calibration mode: {mode}")
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method: {method}")
        if method == "point" and mode != "eye_to_hand":
            raise ValueError("method='point' supports eye_to_hand only (marker centre at the TCP); use 'axxb'")
        self.mode = mode
        self.method = method
        self.min_rotation_deg = min_rotation_deg
        self.samples: list[CalibrationSample] = []
        self.result: CalibrationResult | None = None
        self.marker_id: int | None = None

    # -- compatibility accessors -------------------------------------------------
    @property
    def result_transform(self) -> Transform | None:
        return self.result.transform if self.result else None

    @property
    def rmse(self) -> float:
        return self.result.rmse if self.result else 0.0

    def add_sample(
        self,
        gripper_pose: Transform,
        marker_pose: Transform,
        joints: Sequence[float] | None = None,
        marker_id: int | None = None,
        **extra: Any,
    ) -> CalibrationSample:
        """Add a measured pair: ``gripper_pose = T_base_gripper``, ``marker_pose = T_cam_marker``."""
        s = CalibrationSample(
            gripper_pose, marker_pose, list(joints) if joints is not None else None, marker_id, True, dict(extra)
        )
        self.samples.append(s)
        self.result = None
        return s

    def clear(self) -> None:
        self.samples.clear()
        self.result = None

    def solve(self) -> Transform:
        if len(self.samples) < 3:
            raise CalibrationError(f"At least 3 samples required to solve calibration, got {len(self.samples)}")
        if self.method == "point":
            r = solve_point_registration([s.p_cam for s in self.samples], [s.p_base for s in self.samples])
            self.result = CalibrationResult(
                mode=self.mode,
                method=self.method,
                transform=Transform.from_rot_trans(r["R"], r["t"]),
                rmse=r["rmse"],
                max_error=r["max_error"],
                residuals=r["residuals"].tolist(),
            )
        else:
            self.result = self._solve_axxb()
        self.result.n = len(self.samples)
        self.result.solved_at = datetime.now(timezone.utc).isoformat()
        return self.result.transform

    def _solve_axxb(self) -> CalibrationResult:
        missing = [i + 1 for i, s in enumerate(self.samples) if not s.has_rotation]
        if missing:
            raise CalibrationError(f"axxb needs marker orientation; samples {missing} have position only")
        G = [s.gripper_pose for s in self.samples]
        M = [s.marker_pose for s in self.samples]
        A_list, B_list = [], []
        for i, j in itertools.combinations(range(len(G)), 2):
            if self.mode == "eye_to_hand":
                A_list.append(G[i] @ G[j].inverse())
                B_list.append(M[i] @ M[j].inverse())
            else:
                A_list.append(G[j].inverse() @ G[i])
                B_list.append(M[j] @ M[i].inverse())
        X = solve_axxb(A_list, B_list, self.min_rotation_deg)
        if self.mode == "eye_to_hand":
            Y = _mean_transform([Gi.inverse() @ X @ Mi for Gi, Mi in zip(G, M)])  # T_gripper_marker
            pred = [X.inverse() @ Gi @ Y for Gi in G]
        else:
            Y = _mean_transform([Gi @ X @ Mi for Gi, Mi in zip(G, M)])  # T_base_marker
            pred = [(Gi @ X).inverse() @ Y for Gi in G]
        # residual = predicted vs measured marker pose in the camera frame
        t_res = [float(np.linalg.norm(P.translation - Mi.translation)) for P, Mi in zip(pred, M)]
        r_res = [_rot_angle_deg(P.rotation.T @ Mi.rotation) for P, Mi in zip(pred, M)]
        return CalibrationResult(
            mode=self.mode,
            method=self.method,
            transform=X,
            rmse=float(np.sqrt(np.mean(np.square(t_res)))),
            max_error=float(max(t_res)),
            residuals=t_res,
            rot_residuals_deg=r_res,
            rot_rmse_deg=float(np.sqrt(np.mean(np.square(r_res)))),
            marker_offset=Y,
        )

    # -- persistence -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise in a superset of the rebot-armd Web export format (``kind: hand_eye_calibration``)."""
        out: dict[str, Any] = {
            "version": 1,
            "kind": "hand_eye_calibration",
            "setup": self.mode,
            "method": METHOD_NAMES[self.method],
            "assumption": ASSUMPTIONS.get((self.mode, self.method), ""),
            "units": {"length": "m", "angle": "deg"},
            "created": datetime.now(timezone.utc).isoformat(),
            "marker_id": self.marker_id,
            "result": None,
            "samples": [s.to_dict() for s in self.samples],
        }
        r = self.result
        if r is not None:
            T = r.transform
            res: dict[str, Any] = {
                TRANSFORM_KEYS[self.mode]: T.matrix.tolist(),
                "translation_m": T.translation.tolist(),
                "euler_rpy_deg": list(T.to_euler_deg()),
                "euler_convention": EULER_CONVENTION,
                "rmse_m": r.rmse,
                "max_error_m": r.max_error,
                "residuals_m": r.residuals,
                "samples": r.n,
                "solved_at": r.solved_at,
            }
            if r.rot_residuals_deg is not None:
                res["rot_residuals_deg"] = r.rot_residuals_deg
                res["rot_rmse_deg"] = r.rot_rmse_deg
            if r.marker_offset is not None:
                res[OFFSET_KEYS[self.mode]] = r.marker_offset.matrix.tolist()
            out["result"] = res
            # legacy SDK block (pre-0.2 readers)
            out["mode"] = self.mode
            out["rmse_meters"] = r.rmse
            out["transform"] = T.to_dict()
        return out

    def save_to_file(self, filepath: str | Path, allow_unsolved: bool = False) -> None:
        """Write calibration JSON. With ``allow_unsolved`` the samples are written even without a result."""
        if self.result is None and not allow_unsolved:
            raise RuntimeError("Cannot save calibration: solve() has not been called")
        path = Path(filepath)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read(filepath: str | Path) -> dict[str, Any]:
        return json.loads(Path(filepath).read_text(encoding="utf-8"))

    @classmethod
    def load_result(cls, filepath: str | Path) -> HandEyeResult:
        """Load a solved calibration (SDK file or rebot-armd Web export) together with its setup.

        Use :meth:`HandEyeResult.cam_to_base` to get ``T_cam_to_base``; for an eye_in_hand file it
        needs the arm joints at the time the marker was detected.
        """
        data = cls._read(filepath)
        declared = data.get("setup") or data.get("mode")
        if declared is not None and declared not in MODES:
            raise ValueError(f"{filepath}: unknown setup {declared!r}, expected one of {MODES}")
        res = data.get("result")
        if isinstance(res, dict):
            for setup, key in TRANSFORM_KEYS.items():
                if res.get(key) is None:
                    continue
                if declared is not None and declared != setup:
                    raise ValueError(f"{filepath}: setup {declared!r} does not match result key {key!r}")
                return HandEyeResult(setup=setup, transform=Transform.from_matrix(res[key]))
        if isinstance(data.get("transform"), dict):
            if declared is None:
                raise ValueError(
                    f"{filepath}: legacy file has no 'mode'/'setup'; cannot tell eye_to_hand from eye_in_hand"
                )
            return HandEyeResult(setup=declared, transform=Transform.from_dict(data["transform"]))
        if data.get("kind") == "hand_eye_calibration":
            raise ValueError(
                f"{filepath}: file has no solved result; use HandEyeCalibration.load_samples() and solve()"
            )
        raise ValueError(f"{filepath}: not a hand-eye calibration file")

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Transform:
        """Load ``T_cam_to_base`` from an **eye_to_hand** calibration file.

        An eye_in_hand file stores ``T_cam_to_gripper``, which is not a camera-to-base transform;
        it raises :class:`ValueError`. Use :meth:`load_result`, which returns the setup with the
        transform, for code that has to handle both.
        """
        result = cls.load_result(filepath)
        if result.setup != "eye_to_hand":
            raise ValueError(
                f"{filepath}: eye_in_hand calibration (T_cam_to_gripper) cannot be used as cam_to_base; "
                "use HandEyeCalibration.load_result() and HandEyeResult.cam_to_base(joints)"
            )
        return result.transform

    @classmethod
    def load_samples(
        cls,
        filepath: str | Path,
        method: str = "point",
        mode: str | None = None,
        sign: Sequence[float] = kinematics.DEFAULT_SIGN,
        offset: Sequence[float] = kinematics.DEFAULT_OFFSET,
    ) -> HandEyeCalibration:
        """Create a calibration populated with the samples of an SDK file or Web export (not solved)."""
        data = cls._read(filepath)
        mode = mode or data.get("setup") or data.get("mode") or "eye_to_hand"
        calib = cls(mode=mode, method=method)
        calib.marker_id = data.get("marker_id")
        calib.samples = [CalibrationSample.from_dict(s, sign, offset) for s in data.get("samples", [])]
        return calib


def rotation_spread_deg(samples: Sequence[CalibrationSample]) -> float:
    """Largest relative gripper rotation between any two samples (diagnostic for axxb)."""
    best = 0.0
    for a, b in itertools.combinations(samples, 2):
        best = max(best, _rot_angle_deg(a.gripper_pose.rotation.T @ b.gripper_pose.rotation))
    return best
