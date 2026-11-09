"""IK adapter: base-frame target pose -> the full 7-element joint vector rebot-armd accepts.

armd rejects targets whose length differs from its DOF (``safety.rs`` dimensionality check); the
reBot B601-DM has 7 (J1..J6 arm, J7 gripper). :func:`kinematics.solve_ik` covers J1..J4 only, so:

* J1..J4: analytic IK (position + tool pitch), converted to motor angles with ``sign`` / ``offset``.
* J5 = 0 (the IK model assumes it).
* J6: a fixed value (default armd home, pi/2, ``model.rs`` home_positions) or, with ``j6=None``,
  the seed's current J6.
* J7: gripper target (``GRIPPER_OPEN`` .. ``GRIPPER_CLOSED``; armd limit -5.75 .. 0.10).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import kinematics
from .transforms import Transform

DOF = 7
GRIPPER_INDEX = 6
HOME_POSITIONS = (0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963, 0.0)  # armd model.rs B601-DM home
HOME_J6 = HOME_POSITIONS[5]
GRIPPER_OPEN = -5.7  # armd Web CFG.gripOpen (mechanical stop about -5.87, soft limit -5.75)
GRIPPER_CLOSED = 0.0  # armd Web CFG.gripClosed (soft limit 0.10)
TOP_DOWN_PITCH = math.pi / 2  # tool axis pointing straight down


class IkError(ValueError):
    """No IK solution for the requested pose."""

    def __init__(self, message: str, err: str | None = None) -> None:
        super().__init__(message)
        self.err = err


def pitch_from_rotation(rotation: np.ndarray) -> float:
    """Tool pitch below horizontal (Web FK convention) from a TCP rotation whose x axis is the tool axis."""
    x = np.asarray(rotation)[:, 0]
    return math.atan2(-x[2], math.hypot(x[0], x[1])) - kinematics.DW


class B601IkAdapter:
    """Callable ``pose -> list[7]`` usable as ``DeterministicPickPlacePipeline(ik_solver=...)``.

    ``pitch``: tool pitch below horizontal in rad (default top-down); ``None`` derives it from the
    pose rotation (x axis = tool axis), which only makes sense for a TCP orientation, not a marker
    orientation. Solutions are chosen nearest the seed, and each solution becomes the next seed,
    so consecutive waypoints stay on one IK branch. Call :meth:`set_seed` with the current armd
    joint positions before planning.
    """

    def __init__(
        self,
        pitch: float | None = TOP_DOWN_PITCH,
        j6: float | None = HOME_J6,
        gripper: float = GRIPPER_OPEN,
        sign: Sequence[float] = kinematics.DEFAULT_SIGN,
        offset: Sequence[float] = kinematics.DEFAULT_OFFSET,
        seed: Sequence[float] | None = None,
    ) -> None:
        self.pitch = pitch
        self.j6 = j6
        self.gripper = float(gripper)
        self.sign = tuple(sign)
        self.offset = tuple(offset)
        self.branch: int | None = None
        self.seed: list[float] = list(HOME_POSITIONS)
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, joints: Sequence[float]) -> None:
        """Seed from raw armd joint positions (>= 4 entries; missing J5..J7 are taken from home)."""
        joints = [float(x) for x in joints]
        if len(joints) < 4:
            raise ValueError(f"need at least 4 joint positions, got {len(joints)}")
        self.seed = joints[:DOF] + list(HOME_POSITIONS[len(joints) :])
        self.branch = None

    def solve(self, pose: Transform, gripper: float | None = None) -> list[float]:
        t = pose.translation
        pitch = self.pitch if self.pitch is not None else pitch_from_rotation(pose.rotation)
        prev = kinematics.to_model(self.seed, self.sign, self.offset)
        sol = kinematics.solve_ik(kinematics.TcpPoint(float(t[0]), float(t[1]), float(t[2]), pitch), prev, self.branch)
        if not sol.ok or sol.q is None:
            raise IkError(f"no IK solution for {np.round(t, 4).tolist()} pitch {pitch:.3f} rad ({sol.err})", sol.err)
        j1_4 = kinematics.to_motor(sol.q, self.sign, self.offset)
        j6 = self.j6 if self.j6 is not None else self.seed[5]
        q = [*j1_4, 0.0, float(j6), self.gripper if gripper is None else float(gripper)]
        self.seed, self.branch = list(q), sol.branch
        return q

    __call__ = solve


__all__ = [
    "B601IkAdapter",
    "IkError",
    "pitch_from_rotation",
    "DOF",
    "GRIPPER_INDEX",
    "HOME_POSITIONS",
    "HOME_J6",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
    "TOP_DOWN_PITCH",
]
