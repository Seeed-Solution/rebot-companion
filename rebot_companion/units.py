"""Bidirectional unit conversions between armd radians and LeRobot / SO-100 / SO-101 formats.

Supported modes:
* ``rad``: passthrough armd internal radians.
* ``so101_norm``:
  - Non-gripper arm joints (``i != gripper_joint``): linearly mapped between joint limits
    ``[limit.min_position, limit.max_position]`` (fallback ``[-pi, pi]``) and ``[-100.0, 100.0]``
    (midpoint = 0.0).
  - Gripper joint (``i == gripper_joint``): linearly mapped between ``[closed_rad, open_rad]``
    (default ``[0.0, 1.5]``) and ``[0.0, 100.0]`` (0.0 = closed, 100.0 = open), even when
    ``open_rad < closed_rad`` (as on B601-DM).
* ``deg``:
  - Non-gripper arm joints (``i != gripper_joint``): radians <-> degrees.
  - Gripper joint (``i == gripper_joint``): LeRobot convention mapping ``[closed_rad, open_rad]``
    <-> ``[0.0, 100.0]`` normalized openness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

SUPPORTED_UNIT_MODES = ("rad", "so101_norm", "deg")


def _extract_limit(lim: Any) -> tuple[float, float]:
    if lim is None:
        return (-math.pi, math.pi)
    if hasattr(lim, "min_position") and hasattr(lim, "max_position"):
        return (float(lim.min_position), float(lim.max_position))
    if isinstance(lim, dict):
        return (float(lim.get("min_position", -math.pi)), float(lim.get("max_position", math.pi)))
    if isinstance(lim, (tuple, list)) and len(lim) >= 2:
        return (float(lim[0]), float(lim[1]))
    return (-math.pi, math.pi)


def _resolve_gripper_bounds(
    limits: Sequence[Any] | None,
    gripper_joint: int | None,
    closed_rad: float | None,
    open_rad: float | None,
) -> tuple[float, float]:
    c = 0.0 if closed_rad is None else float(closed_rad)
    if open_rad is not None:
        return c, float(open_rad)
    fallback = -5.7 if gripper_joint == 6 else 1.5
    if limits is not None and gripper_joint is not None and 0 <= gripper_joint < len(limits):
        lo, hi = _extract_limit(limits[gripper_joint])
        o = hi if abs(hi - c) >= abs(lo - c) else lo
        if o == c:
            o = fallback
    else:
        o = fallback
    if o == c:
        o = 1.5 if c == -5.7 else -5.7
    return c, o


def resolve_unit_mode(requested: str, info_meta: dict[str, Any] | None = None) -> str:
    """Resolve unit mode from user requested string and optional dataset/policy metadata.

    When ``requested == "auto"``:
    - Inspects ``info_meta`` for ``unit_mode`` or ``rebot_unit_mode``.
    - Checks heuristic range on ``stats`` / ``observation.state`` if available.
    - Defaults to ``"rad"``.
    """
    if requested != "auto":
        if requested not in SUPPORTED_UNIT_MODES:
            raise ValueError(f"unsupported unit mode {requested!r}; choose from {SUPPORTED_UNIT_MODES} or 'auto'")
        return requested

    if isinstance(info_meta, dict):
        mode = info_meta.get("unit_mode") or info_meta.get("rebot_unit_mode")
        if mode in SUPPORTED_UNIT_MODES:
            return str(mode)
        # Heuristic check on stats
        stats = info_meta.get("stats")
        if isinstance(stats, dict):
            state_stats = stats.get("observation.state")
            if isinstance(state_stats, dict):
                max_vals = state_stats.get("max")
                min_vals = state_stats.get("min")
                if max_vals is not None and min_vals is not None:
                    try:
                        max_arr = [float(v) for v in max_vals]
                        min_arr = [float(v) for v in min_vals]
                        combined = max_arr + min_arr
                        if any(abs(v) > 100.0 for v in combined):
                            return "deg"
                        if any(v > 10.0 for v in max_arr) or any(v < -10.0 for v in min_arr):
                            return "so101_norm"
                    except (ValueError, TypeError):
                        pass

    return "rad"


def from_armd(
    values: Sequence[float],
    mode: str,
    *,
    limits: Sequence[Any] | None = None,
    gripper_joint: int | None = None,
    closed_rad: float | None = None,
    open_rad: float | None = None,
) -> list[float]:
    """Convert armd internal radians to requested unit mode ('rad', 'so101_norm', 'deg')."""
    if mode == "rad":
        return [float(x) for x in values]

    if mode == "so101_norm":
        out = []
        for i, x in enumerate(values):
            if gripper_joint is not None and i == gripper_joint:
                c, o = _resolve_gripper_bounds(limits, gripper_joint, closed_rad, open_rad)
                lo_c, hi_c = (c, o) if c <= o else (o, c)
                x_clamped = min(max(float(x), lo_c), hi_c)
                if o == c:
                    norm = 0.0
                else:
                    norm = 100.0 * (x_clamped - c) / (o - c)
                out.append(float(min(max(norm, 0.0), 100.0)))
            else:
                lim = limits[i] if limits is not None and 0 <= i < len(limits) else None
                lo, hi = _extract_limit(lim)
                if hi == lo:
                    out.append(0.0)
                else:
                    x_clamped = min(max(float(x), lo), hi)
                    norm = 200.0 * (x_clamped - lo) / (hi - lo) - 100.0
                    out.append(float(min(max(norm, -100.0), 100.0)))
        return out

    if mode == "deg":
        out = []
        for i, x in enumerate(values):
            if gripper_joint is not None and i == gripper_joint:
                c, o = _resolve_gripper_bounds(limits, gripper_joint, closed_rad, open_rad)
                lo_c, hi_c = (c, o) if c <= o else (o, c)
                x_clamped = min(max(float(x), lo_c), hi_c)
                if o == c:
                    norm = 0.0
                else:
                    norm = 100.0 * (x_clamped - c) / (o - c)
                out.append(float(min(max(norm, 0.0), 100.0)))
            else:
                out.append(float(math.degrees(float(x))))
        return out

    raise ValueError(f"unsupported unit mode {mode!r}; choose from {SUPPORTED_UNIT_MODES}")


def to_armd(
    values: Sequence[float],
    mode: str,
    *,
    limits: Sequence[Any] | None = None,
    gripper_joint: int | None = None,
    closed_rad: float | None = None,
    open_rad: float | None = None,
) -> list[float]:
    """Convert requested unit mode ('rad', 'so101_norm', 'deg') back to armd internal radians."""
    if mode == "rad":
        return [float(x) for x in values]

    if mode == "so101_norm":
        out = []
        for i, v in enumerate(values):
            if gripper_joint is not None and i == gripper_joint:
                c, o = _resolve_gripper_bounds(limits, gripper_joint, closed_rad, open_rad)
                norm = min(max(float(v), 0.0), 100.0)
                rad = c + (norm / 100.0) * (o - c)
                out.append(float(rad))
            else:
                lim = limits[i] if limits is not None and 0 <= i < len(limits) else None
                lo, hi = _extract_limit(lim)
                norm = min(max(float(v), -100.0), 100.0)
                rad = lo + (norm + 100.0) / 200.0 * (hi - lo)
                out.append(float(rad))
        return out

    if mode == "deg":
        out = []
        for i, v in enumerate(values):
            if gripper_joint is not None and i == gripper_joint:
                c, o = _resolve_gripper_bounds(limits, gripper_joint, closed_rad, open_rad)
                norm = min(max(float(v), 0.0), 100.0)
                rad = c + (norm / 100.0) * (o - c)
                out.append(float(rad))
            else:
                out.append(float(math.radians(float(v))))
        return out

    raise ValueError(f"unsupported unit mode {mode!r}; choose from {SUPPORTED_UNIT_MODES}")
