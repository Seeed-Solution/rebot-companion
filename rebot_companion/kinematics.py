"""Forward / inverse kinematics for the reBot B601-DM arm.

Two models, both in the URDF base frame (metres, radians, x forward / y left / z up):

* **Full FK** (:func:`tcp_pose`): the URDF chain ``base_link -> joint1 .. joint6 -> end_joint -> end_link``
  of ``reBot-DevArm_gripper.urdf`` (rebot-armd ``assets/``; a copy is in ``tests/fixtures/``), with
  the joint origins / axes embedded below. ``end_link`` x is the tool axis. URDF joint5 = wrist swing
  (axis vertical at the park pose), joint6 = tool roll (axis along the tool). Used for hand-eye
  calibration.
* **Planar FK/IK** (:func:`fk`, :func:`solve_ik`): port of the rebot-armd Web cockpit kinematics
  (``src/api/index.html``, block ``IK-BEGIN`` .. ``IK-END`` plus ``toModel`` / ``toMotor``). J1 yaw,
  J2/J3/J4 pitch chain, J5 held at 0, J6 / gripper ignored; constants rounded to 0.1 mm and the
  joint1 x offset (-0.084 mm) ignored, so its TCP position differs from the full FK by < 0.2 mm.
  ``pitch`` is the angle of the J4->TCP vector below horizontal, i.e. ``Ry`` angle minus ``DW``.

Motor -> URDF mapping (armd joint vector index 0..6), armd's ``model_joint_map``
(``GET /api/v1/observation/schema``): URDF angle = ``sign * motor + offset_rad`` per joint
(:class:`JointMap`). B601-DM (:data:`B601DM_JOINT_MAP`): J1..J5 motor = URDF angle, J6 URDF angle =
motor - pi/2. Directions checked on the arm 2026-11-09 (+0.2 rad on index 4 swings the wrist sideways,
on index 5 rolls the tool); the J6 zero from photos of the arm at the rest pose 2026-11-12 (motor J6 =
pi/2: jaws open left-right, camera holder on top, which FK gives at URDF J6 = 0). J1..J4 can also go
through the Web ``CFG.cart.sign`` / ``CFG.cart.off`` overrides (``model = (motor - offset) / sign``,
defaults 1 / 0). Index 6 is the gripper and is not part of the FK.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .transforms import Transform

# Geometry, (r, z) in the arm plane -- identical to the Web constants.
S0 = (0.0200, 0.1402)  # J2 axis (r, z) at J1 = 0
A = (-0.264, 0.0)  # J2 -> J3 at q = 0, (x, z)
B = (0.2426, 0.054)  # J3 -> J4
W = (0.2617, -0.0025)  # J4 -> TCP (end_link) with J5 = 0
LA = math.hypot(*A)
LB = math.hypot(*B)
LW = math.hypot(*W)
DB = math.atan2(B[1], B[0])
DW = math.atan2(W[1], W[0])
LIM = ((-math.pi, math.pi), (-2.5, 2.5), (-3.0, 3.0), (-math.pi, math.pi))  # kinematic bounds (model rad)
TAU = 2.0 * math.pi

# Full URDF chain (reBot-DevArm_gripper.urdf): (joint, origin xyz, origin rpy, axis or None if fixed).
URDF_CHAIN = (
    ("joint1", (-8.416e-05, 0.0, 0.08465), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("joint2", (0.020084, 0.031625, 0.05555), (-1.5708, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("joint3", (-0.264, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("joint4", (0.2426, -0.054, -0.001625), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("joint5", (0.078308, -0.0375, -0.03), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("joint6", (0.028008, 0.0, 0.04), (0.0, 1.5708, 0.0), (0.0, 0.0, 1.0)),
    ("end_joint", (0.0, 0.0, 0.15539), (0.0, -1.5708, 3.1415), None),
)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: ``Rz(y) @ Ry(p) @ Rx(r)``."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_rot(axis: np.ndarray, a: float) -> np.ndarray:
    """Rotation by ``a`` about the unit ``axis`` (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


_CHAIN = tuple(
    (
        np.vstack([np.column_stack([_rpy(*rpy), xyz]), [0.0, 0.0, 0.0, 1.0]]),
        None if axis is None else np.asarray(axis, dtype=float),
    )
    for _, xyz, rpy, axis in URDF_CHAIN
)

# Web defaults for CFG.cart.sign / CFG.cart.off (motor rad = sign * model rad + offset), J1..J4.
DEFAULT_SIGN = (1.0, 1.0, 1.0, 1.0)
DEFAULT_OFFSET = (0.0, 0.0, 0.0, 0.0)

# armd kinematics::B601DM_MODEL_OFFSETS_RAD: URDF angle = motor angle + offset, J1..J6.
B601DM_MODEL_OFFSETS_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 2)


@dataclass(frozen=True)
class JointMap:
    """armd joint -> URDF joint angle, J1..J6: ``model = sign[i] * motor + offset_rad[i]``.

    Same convention as armd's ``model_joint_map`` (observation schema) and ``chain.joints[]``.
    """

    sign: tuple[float, ...] = (1.0,) * 6
    offset_rad: tuple[float, ...] = B601DM_MODEL_OFFSETS_RAD

    def to_model(self, positions: Sequence[float]) -> list[float]:
        """Motor positions (index 0..5; missing = 0) -> URDF angles J1..J6."""
        return [
            self.sign[i] * (float(positions[i]) if i < len(positions) else 0.0) + self.offset_rad[i] for i in range(6)
        ]

    def to_motor(self, q: Sequence[float]) -> list[float]:
        """URDF angles J1..J6 -> motor positions."""
        return [(float(q[i]) - self.offset_rad[i]) / self.sign[i] for i in range(6)]

    @classmethod
    def from_schema(cls, schema) -> JointMap:
        """From an observation schema (dict or :class:`rebot.client.ObservationSchema`).

        Falls back to :data:`B601DM_JOINT_MAP` when the schema has no ``model_joint_map``
        (armd before 2026-11-12) or a malformed one.
        """
        entries = (
            schema.get("model_joint_map") if isinstance(schema, dict) else getattr(schema, "model_joint_map", None)
        )
        try:
            if entries and len(entries) >= 6:
                sign = tuple(float(e["sign"]) for e in entries[:6])
                off = tuple(float(e["offset_rad"]) for e in entries[:6])
                if all(s in (1.0, -1.0) for s in sign) and all(math.isfinite(o) for o in off):
                    return cls(sign, off)
        except (KeyError, TypeError, ValueError):
            pass
        return B601DM_JOINT_MAP


B601DM_JOINT_MAP = JointMap()


@dataclass(frozen=True)
class TcpPoint:
    """Web FK output: TCP position (m) and tool pitch below horizontal (rad)."""

    x: float
    y: float
    z: float
    pitch: float


@dataclass(frozen=True)
class IkSolution:
    ok: bool
    q: tuple[float, float, float, float] | None = None
    branch: int | None = None
    dist: float | None = None
    err: str | None = None  # 'reach' | 'singular' | 'limit'


def _rot(v: Sequence[float], p: float) -> tuple[float, float]:
    """Rotate (x, z) by phi about +y (positive phi tips +x downward)."""
    c, s = math.cos(p), math.sin(p)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)


def _near(a: float, ref: float) -> float:
    # JS Math.round rounds .5 toward +inf; floor(x + 0.5) reproduces that.
    return a + TAU * math.floor((ref - a) / TAU + 0.5)


def to_model(
    positions: Sequence[float],
    sign: Sequence[float] = DEFAULT_SIGN,
    offset: Sequence[float] = DEFAULT_OFFSET,
) -> list[float]:
    """Raw motor joint positions (rad, J1..J4 used) -> URDF model angles. Mirrors Web ``toModel``."""
    if len(positions) < 4:
        raise ValueError(f"need at least 4 joint positions, got {len(positions)}")
    return [(float(positions[i]) - offset[i]) / sign[i] for i in range(4)]


def to_motor(
    q: Sequence[float],
    sign: Sequence[float] = DEFAULT_SIGN,
    offset: Sequence[float] = DEFAULT_OFFSET,
) -> list[float]:
    """URDF model angles -> motor joint positions. Mirrors Web ``toMotor``."""
    return [sign[i] * float(x) + offset[i] for i, x in enumerate(q)]


def fk(q: Sequence[float]) -> TcpPoint:
    """Web ``IK.fk``: model angles [q1..q4] -> TCP position and pitch."""
    p2 = -q[1]
    p3 = p2 + q[2]
    p4 = p3 + q[3]
    a, b, w = _rot(A, p2), _rot(B, p3), _rot(W, p4)
    r = S0[0] + a[0] + b[0] + w[0]
    z = S0[1] + a[1] + b[1] + w[1]
    return TcpPoint(x=r * math.cos(q[0]), y=r * math.sin(q[0]), z=z, pitch=p4 - DW)


def tcp_pose(q: Sequence[float]) -> Transform:
    """Full URDF FK ``T_base_tcp`` (``end_link`` in base) from model angles [q1..q4] or [q1..q6].

    Missing q5 / q6 are taken as 0. With q5 = q6 = 0 the rotation equals the planar model's
    ``Rz(q1) @ Ry(-q2 + q3 + q4)`` to < 1e-4.
    """
    if len(q) < 4:
        raise ValueError(f"need at least 4 joint angles, got {len(q)}")
    qs = [float(v) for v in q[:6]] + [0.0] * (6 - min(len(q), 6))
    M = np.eye(4)
    i = 0
    for fixed, axis in _CHAIN:
        M = M @ fixed
        if axis is not None:
            M[:3, :3] = M[:3, :3] @ _axis_rot(axis, qs[i])
            i += 1
    return Transform.from_matrix(M)


def tcp_pose_from_joints(
    positions: Sequence[float],
    sign: Sequence[float] = DEFAULT_SIGN,
    offset: Sequence[float] = DEFAULT_OFFSET,
    joint_map: JointMap = B601DM_JOINT_MAP,
) -> Transform:
    """Raw motor positions (as returned by armd observation) -> ``T_base_tcp``.

    J1..J4 go through ``joint_map`` and then the Web ``sign`` / ``offset`` overrides
    (:func:`to_model`; defaults leave them unchanged); J5 / J6 (indices 4, 5) through ``joint_map``
    (B601-DM: URDF joint6 = motor - pi/2), 0 motor when absent; the gripper (index 6) is ignored.
    """
    if len(positions) < 4:
        raise ValueError(f"need at least 4 joint positions, got {len(positions)}")
    q = joint_map.to_model(positions)
    return tcp_pose(to_model(q[:4], sign, offset) + q[4:6])


def reach(p: TcpPoint) -> float:
    """Distance from the shoulder axis to the TCP (workspace sphere)."""
    r = math.hypot(p.x, p.y)
    return math.hypot(r - S0[0], p.z - S0[1])


def solve_ik(
    target: TcpPoint,
    prev: Sequence[float],
    branch: int | None = None,
    in_limits: Callable[[Sequence[float]], bool] | None = None,
    r_min: float = 0.03,
    cos_max: float = 0.998,
) -> IkSolution:
    """Web ``IK.solve``: analytic IK for position + pitch, choosing the solution nearest ``prev``."""
    yaw = math.atan2(target.y, target.x)
    rr = math.hypot(target.x, target.y)
    if rr < r_min:  # over the base axis
        yaw = prev[0]
        rr = target.x * math.cos(yaw) + target.y * math.sin(yaw)
    aw = -target.pitch  # CCW angle of the J4 -> TCP vector
    best: tuple[list[float], int, float] | None = None
    err = "reach"
    for k in range(2):  # k = 1: reach over the back
        q1 = _near(yaw + math.pi if k else yaw, prev[0])
        r = -rr if k else rr
        px = r - LW * math.cos(aw) - S0[0]
        pz = target.z - LW * math.sin(aw) - S0[1]
        c = (px * px + pz * pz - LA * LA - LB * LB) / (2 * LA * LB)
        if abs(c) > 1:
            continue
        if abs(c) > cos_max:
            if err == "reach":
                err = "singular"
            continue
        for sg in (1, -1):
            g = sg * math.acos(c)
            aa = math.atan2(pz, px) - math.atan2(LB * math.sin(g), LA + LB * math.cos(g))
            ab = aa + g
            q2 = aa - math.pi
            p3 = DB - ab
            q = [q1, _near(q2, prev[1]), 0.0, 0.0]
            q[2] = _near(p3 + q[1], prev[2])
            q[3] = _near(DW - aw - (-q[1] + q[2]), prev[3])
            if in_limits is not None:
                lim = bool(in_limits(q))
            else:
                lim = all(LIM[i][0] <= q[i] <= LIM[i][1] for i in range(4))
            if not lim:
                err = "limit"
                continue
            d = sum((q[i] - prev[i]) ** 2 for i in range(4))
            br = k * 2 + (0 if sg > 0 else 1)
            if br == branch:
                d *= 0.8  # hysteresis: prefer staying on the current branch
            if best is None or d < best[2]:
                best = (q, br, d)
    if best is None:
        return IkSolution(ok=False, err=err)
    q, br, d = best
    return IkSolution(ok=True, q=(q[0], q[1], q[2], q[3]), branch=br, dist=math.sqrt(d))
