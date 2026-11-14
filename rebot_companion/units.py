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

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("rebot_companion.units")

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
    if open_rad is not None:
        o = float(open_rad)
    else:
        fallback = -5.7 if gripper_joint == 6 else 1.5
        if limits is not None and gripper_joint is not None and 0 <= gripper_joint < len(limits):
            lo, hi = _extract_limit(limits[gripper_joint])
            if gripper_joint == 6:
                # Preserve B601-DM GRIPPER_OPEN = -5.7 safety margin when within limits
                if lo <= -5.7 <= hi:
                    o = -5.7
                else:
                    c_ref = 0.0 if closed_rad is None else float(closed_rad)
                    o = hi if abs(hi - c_ref) >= abs(lo - c_ref) else lo
            else:
                c_ref = 0.0 if closed_rad is None else float(closed_rad)
                o = hi if abs(hi - c_ref) >= abs(lo - c_ref) else lo
            if o == (0.0 if closed_rad is None else float(closed_rad)):
                o = fallback
        else:
            o = fallback

    if closed_rad is not None:
        c = float(closed_rad)
    else:
        if limits is not None and gripper_joint is not None and 0 <= gripper_joint < len(limits):
            lo, hi = _extract_limit(limits[gripper_joint])
            if lo <= 0.0 <= hi and abs(o - 0.0) > 1e-6:
                c = 0.0
            else:
                c = lo if abs(lo - o) >= abs(hi - o) else hi
        else:
            c = 0.0

    if o == c:
        o = 1.5 if c == -5.7 else -5.7
    return c, o


def _as_finite_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def resolve_schema_gripper_bounds(
    gripper_policy: dict[str, Any] | None,
    limits: Sequence[Any] | None,
    gripper_joint: int | None,
) -> tuple[float, float]:
    """Resolve (closed_rad, open_rad) bounds from schema gripper_policy and limits.

    When ``gripper_policy`` omits ``"closed_rad"`` and ``limits`` is provided with
    ``0 <= gripper_joint < len(limits)``, falls back to the limit endpoint (``min_position``
    or ``max_position``) farthest from ``open_rad``, else ``0.0``.
    """
    closed_rad: float | None = None
    open_rad: float | None = None
    if isinstance(gripper_policy, dict):
        closed_rad = _as_finite_float(gripper_policy.get("closed_rad"))
        open_rad = _as_finite_float(gripper_policy.get("open_rad"))
    return _resolve_gripper_bounds(limits, gripper_joint, closed_rad, open_rad)


def resolve_unit_mode(
    requested: str,
    info_meta: dict[str, Any] | None = None,
    *,
    require_metadata: bool = False,
) -> str:
    """Resolve unit mode from user requested string and optional dataset/policy metadata.

    When ``requested == "auto"``:
    - Inspects ``info_meta`` for ``unit_mode`` or ``rebot_unit_mode``.
    - If ``require_metadata=True`` and ``info_meta`` has neither explicit ``unit_mode`` nor
      ``rebot_unit_mode``, raises ``ValueError``.
    - If ``require_metadata=False``, checks heuristic range on ``stats`` (both ``observation.state``
      and ``action``). If non-radian values are present without explicit metadata, raises ``ValueError``.
    - Defaults to ``"rad"``.
    """
    if requested != "auto":
        if requested not in SUPPORTED_UNIT_MODES:
            raise ValueError(f"unsupported unit mode {requested!r}; choose from {SUPPORTED_UNIT_MODES} or 'auto'")
        return requested

    if isinstance(info_meta, dict):
        raw_mode = info_meta.get("unit_mode") or info_meta.get("rebot_unit_mode")
        if raw_mode is not None:
            mode_str = str(raw_mode).strip()
            if mode_str:
                if mode_str in SUPPORTED_UNIT_MODES:
                    return mode_str
                raise ValueError(f"unsupported unit mode {mode_str!r} in metadata; choose from {SUPPORTED_UNIT_MODES}")

    if require_metadata:
        raise ValueError(
            "cannot infer unit mode 'auto' without explicit 'unit_mode' in metadata; "
            "pass --action-units / --state-units explicitly ('rad', 'so101_norm', or 'deg')"
        )

    # Heuristic stats check for require_metadata=False
    if isinstance(info_meta, dict):
        stats = info_meta.get("stats")
        if isinstance(stats, dict):
            threshold = 2 * math.pi + 0.25
            for key in ("observation.state", "action"):
                sub_stats = stats.get(key)
                if isinstance(sub_stats, dict):
                    for bound_key in ("min", "max"):
                        raw_vals = sub_stats.get(bound_key)
                        if raw_vals is None:
                            continue
                        if isinstance(raw_vals, (list, tuple)) or (
                            hasattr(raw_vals, "__iter__") and not isinstance(raw_vals, (str, bytes, dict))
                        ):
                            items = raw_vals
                        else:
                            items = [raw_vals]
                        for v in items:
                            try:
                                fv = float(v)
                                if math.isfinite(fv):
                                    if abs(fv) > threshold:
                                        raise ValueError(
                                            "cannot infer unit mode 'auto' from non-radian stats without 'unit_mode' "
                                            "in metadata; pass --action-units / --state-units explicitly ('so101_norm' or 'deg')"
                                        )
                            except (TypeError, ValueError) as exc:
                                if "cannot infer unit mode" in str(exc):
                                    raise

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
    """Convert armd internal radians to requested unit mode ('rad', 'so101_norm', 'deg').

    Note: ``so101_norm`` normalizes against armd's configured joint limits ([min_position, max_position])
    and gripper bounds for round-trip compatibility with rebot-record, rather than external LeRobot motor tick ranges.
    """
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
                    lo_b, hi_b = (lo, hi) if lo <= hi else (hi, lo)
                    x_clamped = min(max(float(x), lo_b), hi_b)
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
    """Convert requested unit mode ('rad', 'so101_norm', 'deg') back to armd internal radians.

    Note: ``so101_norm`` normalizes against armd's configured joint limits ([min_position, max_position])
    and gripper bounds for round-trip compatibility with rebot-record, rather than external LeRobot motor tick ranges.
    """
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


def load_lerobot_norm_bounds(
    calib_path: str | Path,
    schema: Any = None,
) -> tuple[list[tuple[float, float]], float | None, float | None]:
    """Parse LeRobot SO-100/SO-101 calibration JSON and convert motor tick ranges to armd radians.

    When ``homing_applied_in_hw=False`` in follower ``feetech_joints``, ``zero_step + homing_offset_steps``
    matches armd's ``FeetechJointCalib::steps_to_rad`` (``src/protocol/feetech.rs:483-487``).

    Returns:
        (limits, closed_rad, open_rad) where limits has 6 elements [(lo_j1, hi_j1), ..., (closed_rad, open_rad)].
    """
    p = Path(calib_path).expanduser()
    if not p.is_file():
        raise ValueError(f"calibration file not found: {p}")

    try:
        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
    except Exception as exc:
        raise ValueError(f"failed to read/parse calibration file {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"calibration file {p} must contain a JSON object, got {type(data).__name__}")

    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    feetech_joints = (
        getattr(schema, "feetech_joints", None)
        if schema is not None
        else None
    )
    if feetech_joints is None and isinstance(schema, dict):
        feetech_joints = schema.get("feetech_joints")

    if schema is None or feetech_joints is None:
        log.warning(
            "load_lerobot_norm_bounds called without follower feetech_joints schema; "
            "assuming default sign=+1 and zero_step=2048 for joints lacking hardware calibration"
        )

    limits: list[tuple[float, float]] = []
    for i, name in enumerate(joint_names):
        alt_name = f"joint_{i + 1}"
        entry = data.get(name) or data.get(alt_name)
        if entry is None or not isinstance(entry, dict):
            raise ValueError(
                f"missing required joint key '{name}' or '{alt_name}' in calibration file {p}"
            )
        if "range_min" not in entry or "range_max" not in entry:
            raise ValueError(f"{name}: missing required 'range_min' or 'range_max' in {entry}")

        try:
            range_min = float(entry["range_min"])
            range_max = float(entry["range_max"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: invalid range_min/range_max in {entry}") from exc

        if not (math.isfinite(range_min) and math.isfinite(range_max) and 0 <= range_min < range_max <= 4095):
            raise ValueError(
                f"{name}: tick bounds must satisfy 0 <= range_min < range_max <= 4095, got [{range_min}, {range_max}]"
            )

        dm = entry.get("drive_mode", 0)
        drive_mode = bool(dm in (1, True, "1"))

        tick_neg100 = range_max if drive_mode else range_min
        tick_pos100 = range_min if drive_mode else range_max

        fj = (
            feetech_joints[i]
            if feetech_joints and i < len(feetech_joints) and isinstance(feetech_joints[i], dict)
            else None
        )
        if fj is not None:
            zero_step = float(fj.get("zero_step", 2048))
            homing_applied = bool(fj.get("homing_applied_in_hw", False))
            homing_offset = float(fj.get("homing_offset_steps", 0))
            # When homing_applied_in_hw=False, zero_step + homing_offset_steps matches
            # armd's FeetechJointCalib::steps_to_rad (src/protocol/feetech.rs:483-487).
            effective_zero = zero_step if homing_applied else zero_step + homing_offset
            armd_sign = -1.0 if fj.get("drive_mode_invert") else 1.0
        else:
            effective_zero = float(entry.get("zero_step", entry.get("center", 2048)))
            armd_sign = 1.0

        rad_neg100 = armd_sign * (tick_neg100 - effective_zero) * (2.0 * math.pi / 4096.0)
        rad_pos100 = armd_sign * (tick_pos100 - effective_zero) * (2.0 * math.pi / 4096.0)

        limits.append((float(rad_neg100), float(rad_pos100)))

    # Gripper
    grip_entry = data.get("gripper")
    if grip_entry is None or not isinstance(grip_entry, dict):
        raise ValueError(f"missing required key 'gripper' in calibration file {p}")
    if "range_min" not in grip_entry or "range_max" not in grip_entry:
        raise ValueError(f"gripper: missing required 'range_min' or 'range_max' in {grip_entry}")

    try:
        g_min = float(grip_entry["range_min"])
        g_max = float(grip_entry["range_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gripper: invalid range_min/range_max in {grip_entry}") from exc

    if not (math.isfinite(g_min) and math.isfinite(g_max) and 0 <= g_min < g_max <= 4095):
        raise ValueError(
            f"gripper: tick bounds must satisfy 0 <= range_min < range_max <= 4095, got [{g_min}, {g_max}]"
        )

    g_dm = grip_entry.get("drive_mode", 0)
    g_drive_mode = bool(g_dm in (1, True, "1"))

    g_tick_closed = g_max if g_drive_mode else g_min
    g_tick_open = g_min if g_drive_mode else g_max
    if "closed_tick" in grip_entry or "closed" in grip_entry:
        g_tick_closed = float(grip_entry.get("closed_tick", grip_entry.get("closed")))
    if "open_tick" in grip_entry or "open" in grip_entry:
        g_tick_open = float(grip_entry.get("open_tick", grip_entry.get("open")))

    if not (
        math.isfinite(g_tick_closed)
        and math.isfinite(g_tick_open)
        and 0 <= g_tick_closed <= 4095
        and 0 <= g_tick_open <= 4095
        and g_tick_closed != g_tick_open
    ):
        raise ValueError(
            f"gripper: tick bounds must satisfy 0 <= closed <= 4095, 0 <= open <= 4095 and closed != open, "
            f"got closed={g_tick_closed}, open={g_tick_open}"
        )

    fj_grip = (
        feetech_joints[5]
        if feetech_joints and len(feetech_joints) > 5 and isinstance(feetech_joints[5], dict)
        else None
    )
    if fj_grip is not None:
        g_zero_step = float(fj_grip.get("zero_step", 2048))
        g_homing_applied = bool(fj_grip.get("homing_applied_in_hw", False))
        g_homing_offset = float(fj_grip.get("homing_offset_steps", 0))
        # When homing_applied_in_hw=False, zero_step + homing_offset_steps matches
        # armd's FeetechJointCalib::steps_to_rad (src/protocol/feetech.rs:483-487).
        g_effective_zero = g_zero_step if g_homing_applied else g_zero_step + g_homing_offset
        g_armd_sign = -1.0 if fj_grip.get("drive_mode_invert") else 1.0
    else:
        g_effective_zero = float(grip_entry.get("zero_step", grip_entry.get("center", 2048)))
        g_armd_sign = 1.0

    closed_rad = float(g_armd_sign * (g_tick_closed - g_effective_zero) * (2.0 * math.pi / 4096.0))
    open_rad = float(g_armd_sign * (g_tick_open - g_effective_zero) * (2.0 * math.pi / 4096.0))
    limits.append((closed_rad, open_rad))
    return limits, closed_rad, open_rad
