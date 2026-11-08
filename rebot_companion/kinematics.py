"""Forward / inverse kinematics for the reBot B601-DM arm.

Port of the rebot-armd Web cockpit kinematics (``src/api/index.html``, block ``IK-BEGIN`` .. ``IK-END``
plus ``toModel`` / ``toMotor``). Model angles follow the URDF ``rebot-arm-workspace/assets/urdf_b601dm``:
metres, radians, base frame x forward / y left / z up.

Joint layout used by the Web FK/IK: J1 yaw, J2/J3/J4 pitch chain, J5 held at 0, gripper ignored.
Lateral link offsets (< 0.1 mm) are ignored, exactly as in the Web code.

The full TCP orientation is ``R = Rz(q1) @ Ry(q2' + q3 + q4)`` with ``q2' = -q2``; this equals the
URDF ``end_link`` frame (x = tool axis) when the wrist joints are at 0 (checked numerically against
the URDF chain: < 0.03 mm position, < 1e-4 rotation difference). ``pitch`` as reported by the Web FK
is the angle of the J4->TCP vector below horizontal, i.e. ``Ry`` angle minus ``DW`` (~ -0.55 deg).
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

# Web defaults for CFG.cart.sign / CFG.cart.off (motor rad = sign * model rad + offset).
DEFAULT_SIGN = (1.0, 1.0, 1.0, 1.0)
DEFAULT_OFFSET = (0.0, 0.0, 0.0, 0.0)


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
    """Full SE(3) TCP pose ``T_base_tcp`` (TCP expressed in base frame) from model angles [q1..q4]."""
    p = fk(q)
    p4 = p.pitch + DW
    cy, sy = math.cos(q[0]), math.sin(q[0])
    cp, sp = math.cos(p4), math.sin(p4)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return Transform.from_rot_trans(Rz @ Ry, [p.x, p.y, p.z])


def tcp_pose_from_joints(
    positions: Sequence[float],
    sign: Sequence[float] = DEFAULT_SIGN,
    offset: Sequence[float] = DEFAULT_OFFSET,
) -> Transform:
    """Raw motor positions (as returned by armd observation) -> ``T_base_tcp``."""
    return tcp_pose(to_model(positions, sign, offset))


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
