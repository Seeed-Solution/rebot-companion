"""rebot-armd client.

rebot-armd binds actuator authority to the WebSocket connection that holds the
control lease. REST is used only for lease-free calls (observation, schema,
e-stop); every actuator command (torque, motion, stop, clear_fault) goes over
``/api/v1/ws`` inside an :class:`ArmSession`.

Standard library only (Python >= 3.8). Most programs use the :class:`rebot.Arm` facade;
this module is the protocol layer underneath it.

Typical use::

    arm = ArmdClient("http://127.0.0.1:8765")
    print(arm.get_observation().safety_state)
    with arm.session() as s:          # opens WS, acquires lease, starts heartbeat
        s.torque(True)
        s.home()
        s.move_to([0.2, 0.1, 0.0, 0.0, 0.0, 0.0], duration=2.0)
        s.move_pose([0.25, 0.0, 0.25])   # Cartesian action (armd with /api/v1/action/*)
        s.gripper("close")
        s.unload()                    # park + torque off (if armd supports it)
    arm.e_stop()                      # REST, never needs the lease

Protocol summary (armd ``src/api/ws.rs``):

* Server pushes one observation JSON (no ``type`` field) per control tick.
* ``acquire_lease`` -> ``lease_granted{generation}`` / ``lease_rejected{error}``.
* ``heartbeat`` must arrive well inside ``lease_timeout_ms`` (default 2500 ms),
  carrying the current ``generation`` and ``epoch_id``. A heartbeat with a stale
  epoch is answered with ``epoch_sync`` and does not renew the lease.
* ``command{request_id, path, body, generation, epoch_id}`` ->
  ``command_ack`` / ``command_reject{code}``; ``request_id`` must strictly increase.
* ``servo{sequence_id, positions, generation, epoch_id}`` streams targets; armd
  stops following the stream 200 ms after the last frame.
* ``estop`` needs no lease and advances the epoch.
* ``/api/v1/action/{move_to,move_relative,gripper}`` (and jog/home/unload) acks carry
  ``action_id`` + ``expected_duration_s``; the observation's ``action`` field reports
  ``running -> done | failed | cancelled`` for the latest action, ``previous_action`` the one
  before it for 2 s.
* ``/api/v1/ws?obs_hz=N`` (capability ``obs_rate``) limits observations to N per second.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from . import _http, _ws
from ._log import _log
from .errors import (
    ActionFailed,
    ArmCommandRejected,
    ArmConnectionError,
    ArmdError,
    ArmFaultError,
    LeaseError,
    MotionTimeout,
    RebotError,
    ServoJumpError,
    ServoRejected,
    UnloadFailed,
    UnsupportedError,
)

log = _log

SERVO_MAX_STEP = 0.35  # rad; armd drops frames jumping further from its commanded position
SERVO_MIN_RATE_HZ = 10.0  # armd motion_frame_timeout is 200 ms
GRIPPER_INDEX = 6  # 7-DOF B601-DM: joints 0..5 arm, 6 gripper
SAFE_EXIT_POLICIES = ("off", "error", "always")
ACTION_MAX_TIMEOUT_S = 60.0
TASK_UNSUPPORTED = "TASK_UNSUPPORTED"  # armd: the loaded vision model/depth cannot serve the grasp task
CAPABILITIES_DEVPLAT = 6  # capabilities.version with obs_rate / fault_codes / poses / keepout / ...
_GRIPPER_UNSET = object()
_SCHEMA_RETRY_COOLDOWN_S = 5.0


@dataclass
class SafeExitOutcome:
    """Result of :meth:`ArmSession.safe_exit`.

    ``action``: ``not_needed`` (torque already off), ``unloaded``, ``unsupported`` (armd has no
    resting-pose profile; arm left torqued), ``estop`` (E-stop latched; not cleared automatically),
    ``fault`` (non-recoverable fault; not cleared automatically),
    ``no_lease`` (lease lost; nothing can be commanded) or ``failed``.
    """

    action: str
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.action in ("not_needed", "unloaded")


# --------------------------------------------------------------------------- data


def _parse_safety_state(value: Any) -> tuple[str, str | None, Any]:
    """armd serialises ``SafetyState`` with serde's external tagging.

    Unit variants are plain strings (``"Armed"``); ``Fault(reason)`` is
    ``{"Fault": "WatchdogTimeout"}`` or ``{"Fault": {"Other": "..."}}``.
    Returns ``(name, fault_reason, raw_detail)``.
    """
    if isinstance(value, str):
        return value, None, None
    if isinstance(value, dict) and len(value) == 1:
        name, detail = next(iter(value.items()))
        if isinstance(detail, str):
            return name, detail, detail
        if isinstance(detail, dict) and len(detail) == 1:
            kind, inner = next(iter(detail.items()))
            return name, f"{kind}: {inner}" if isinstance(inner, str) else kind, detail
        return name, json.dumps(detail), detail
    return "Unknown", None, value


ACTION_TERMINAL_STATES = ("done", "failed", "cancelled")


def _opt_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        vf = float(v)
        return vf if math.isfinite(vf) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _req_int(v: Any, name: str = "value") -> int:
    if v is None or isinstance(v, bool):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    try:
        if isinstance(v, float):
            if not (math.isfinite(v) and v.is_integer()):
                raise ValueError(f"{name} must be an integer, got {v!r}")
            return int(v)
        return int(v)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer, got {v!r}") from exc


def _req_float(v: Any, name: str = "value") -> float:
    if v is None or isinstance(v, bool):
        raise ValueError(f"{name} must be a float, got {v!r}")
    try:
        vf = float(v)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a float, got {v!r}") from exc
    if not math.isfinite(vf):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return vf


@dataclass
class ActionStatus:
    """``ArmObservation.action``: the latest high-level action (rebot-armd agent action API).

    ``kind``: move_to | move_relative | gripper | joint_move | home | park.
    ``state``: running | done | failed | cancelled; ``code`` is set for failed / cancelled.
    """

    id: int
    kind: str
    state: str
    tag: str | None = None
    code: str | None = None
    message: str | None = None
    progress: float = 0.0
    started_at: float = 0.0
    finished_at: float | None = None
    holding: bool | None = None
    gripper_gap_rad: float | None = None
    settled: bool | None = None
    phase: str | None = None  # grasp only
    stop_rad: float | None = None
    offset_rad: float | None = None
    pass_: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ActionStatus:
        holding, settled = d.get("holding"), d.get("settled")
        pass_val = d.get("pass") if "pass" in d else d.get("pass_")
        return cls(
            id=int(d.get("id", 0)),
            kind=str(d.get("kind", "")),
            state=str(d.get("state", "")),
            tag=d.get("tag"),
            code=d.get("code"),
            message=d.get("message"),
            progress=float(d.get("progress") or 0.0),
            started_at=float(d.get("started_at") or 0.0),
            finished_at=_opt_float(d.get("finished_at")),
            holding=None if holding is None else bool(holding),
            gripper_gap_rad=_opt_float(d.get("gripper_gap_rad")),
            settled=None if settled is None else bool(settled),
            phase=d.get("phase"),
            stop_rad=_opt_float(d.get("stop_rad")),
            offset_rad=_opt_float(d.get("offset_rad")),
            pass_=None if pass_val is None else bool(pass_val),
        )

    @property
    def passed(self) -> bool | None:
        return self.pass_

    @property
    def terminal(self) -> bool:
        return self.state in ACTION_TERMINAL_STATES

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["pass"] = self.pass_
        return d


@dataclass
class ActionResult:
    """Outcome of :meth:`ArmSession.action`: the terminal action record and the observation that carried it.

    ``action`` is None only for :meth:`ArmSession.home_action` / :meth:`ArmSession.park_action`
    against an armd that does not report action records.
    """

    action: ActionStatus | None
    status: ArmStatus
    ack: dict
    duration_s: float

    @property
    def holding(self) -> bool | None:
        """Gripper/grasp verdict: True when an object is held, None when the action does not report it."""
        return self.action.holding if self.action is not None else None

    @property
    def stop_rad(self) -> float | None:
        return self.action.stop_rad if self.action is not None else None

    @property
    def offset_rad(self) -> float | None:
        return self.action.offset_rad if self.action is not None else None

    @property
    def pass_(self) -> bool | None:
        return self.action.pass_ if self.action is not None else None

    @property
    def passed(self) -> bool | None:
        return self.action.pass_ if self.action is not None else None

    @property
    def ok(self) -> bool:
        return self.action is None or self.action.state == "done"


def _checked_float_vec(raw: Any, expected_len: int) -> list[float] | None:
    if raw is None or not isinstance(raw, (list, tuple)):
        return None
    if len(raw) != expected_len:
        return None
    res: list[float] = []
    for x in raw:
        if x is None or isinstance(x, bool):
            return None
        try:
            xf = float(x)
            if not math.isfinite(xf):
                return None
            res.append(xf)
        except (ValueError, TypeError, OverflowError):
            return None
    return res


@dataclass
class TcpPose:
    """End-effector pose in arm base frame (metres and radians / quaternion [w, x, y, z])."""

    xyz_m: list[float]
    rpy_rad: list[float] | None = None
    quat_wxyz: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> TcpPose:
        if not isinstance(d, dict):
            raise ValueError(f"tcp pose must be a dict, got {d!r}")
        xyz = _checked_float_vec(d.get("xyz_m"), 3)
        if xyz is None:
            raise ValueError("invalid xyz_m vector")
        return cls(
            xyz_m=xyz,
            rpy_rad=_checked_float_vec(d.get("rpy_rad"), 3),
            quat_wxyz=_checked_float_vec(d.get("quat_wxyz"), 4),
        )

    def to_dict(self) -> dict:
        return {
            "xyz_m": self.xyz_m,
            "rpy_rad": self.rpy_rad,
            "quat_wxyz": self.quat_wxyz,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) and getattr(self, item) is not None


@dataclass
class GripperStatus:
    """Gripper state reported by armd."""

    joint: int
    rad: float
    fraction: float
    opening_m: float | None = None
    stalled: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> GripperStatus:
        if not isinstance(d, dict):
            raise ValueError(f"gripper status must be a dict, got {d!r}")
        joint = _req_int(d.get("joint"), "joint")
        rad = _req_float(d.get("rad"), "rad")
        fraction = _req_float(d.get("fraction"), "fraction")
        return cls(
            joint=joint,
            rad=rad,
            fraction=fraction,
            opening_m=_opt_float(d.get("opening_m")),
            stalled=bool(d.get("stalled", False)),
        )

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "rad": self.rad,
            "fraction": self.fraction,
            "opening_m": self.opening_m,
            "stalled": self.stalled,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


@dataclass
class ArmStatus:
    """One ``ArmObservation`` as emitted by armd (REST ``GET /api/v1/observation`` or WS telemetry)."""

    timestamp: float  # seconds since armd start
    positions: list[float]
    velocities: list[float]
    torques: list[float]
    temperatures: list[float]
    is_torqued: bool
    safety_state: str  # Booting | Idle | Armed | Moving | EStop | Fault
    epoch_id: int
    fault_reason: str | None = None
    fault_detail: Any = None
    gravity_compensated: bool = False
    gravity_torques: list[float] = field(default_factory=list)
    driver_capabilities: dict = field(default_factory=dict)
    enable_state: str = "Unknown"
    feedback_age_ms: list[int | None] = field(default_factory=list)
    disable_pending: bool = False
    feedback_valid: bool = False
    commanded_positions: list[float] = field(default_factory=list)  # empty on armd without the field
    action: ActionStatus | None = None
    previous_action: ActionStatus | None = None
    fault: dict | None = None  # {"code", "message", "recoverable"} (capability fault_codes)
    residual_torques: list[float] | None = None  # capability residual_torques
    loads: list[float] | None = None
    collision: dict | None = None  # capability collision_guard; None while the guard is off or unavailable
    motion_limit: dict | None = None  # capability servo_guard: {"reason", "joint", "since_ms"} or None
    gripper_zero: dict | None = None  # capability gripper_zero: {"state", "j7", "offset_rad"}
    tcp: TcpPose | None = None
    gripper: GripperStatus | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ArmStatus:
        name, reason, detail = _parse_safety_state(d.get("safety_state"))
        floats = lambda key: [float(x) for x in d.get(key) or []]  # noqa: E731
        tcp = None
        if isinstance(d.get("tcp"), dict):
            try:
                tcp = TcpPose.from_dict(d["tcp"])
            except (TypeError, ValueError) as exc:
                log.debug("ignoring malformed tcp payload: %s", exc)
                tcp = None

        gripper = None
        if isinstance(d.get("gripper"), dict):
            try:
                gripper = GripperStatus.from_dict(d["gripper"])
            except (TypeError, ValueError) as exc:
                log.debug("ignoring malformed gripper payload: %s", exc)
                gripper = None

        return cls(
            timestamp=float(d.get("timestamp", 0.0)),
            positions=floats("positions"),
            velocities=floats("velocities"),
            torques=floats("torques"),
            temperatures=floats("temperatures"),
            is_torqued=bool(d.get("is_torqued", False)),
            safety_state=name,
            epoch_id=int(d.get("epoch_id", 0)),
            fault_reason=reason,
            fault_detail=detail,
            gravity_compensated=bool(d.get("gravity_compensated", False)),
            gravity_torques=floats("gravity_torques"),
            driver_capabilities=dict(d.get("driver_capabilities") or {}),
            enable_state=str(d.get("enable_state", "Unknown")),
            feedback_age_ms=list(d.get("feedback_age_ms") or []),
            disable_pending=bool(d.get("disable_pending", False)),
            feedback_valid=bool(d.get("feedback_valid", False)),
            commanded_positions=floats("commanded_positions"),
            action=ActionStatus.from_dict(d["action"]) if isinstance(d.get("action"), dict) else None,
            previous_action=(
                ActionStatus.from_dict(d["previous_action"]) if isinstance(d.get("previous_action"), dict) else None
            ),
            fault=dict(d["fault"]) if isinstance(d.get("fault"), dict) else None,
            residual_torques=floats("residual_torques") if isinstance(d.get("residual_torques"), list) else None,
            loads=floats("loads") if isinstance(d.get("loads"), list) else None,
            collision=dict(d["collision"]) if isinstance(d.get("collision"), dict) else None,
            motion_limit=dict(d["motion_limit"]) if isinstance(d.get("motion_limit"), dict) else None,
            gripper_zero=dict(d["gripper_zero"]) if isinstance(d.get("gripper_zero"), dict) else None,
            tcp=tcp,
            gripper=gripper,
        )

    @property
    def e_stopped(self) -> bool:
        return self.safety_state == "EStop"

    @property
    def faulted(self) -> bool:
        return self.safety_state == "Fault"

    @property
    def is_stopped(self) -> bool:
        """EStop or Fault: motion is rejected until ``clear_fault``."""
        return self.safety_state in ("EStop", "Fault")

    @property
    def is_operational(self) -> bool:
        """Armed or Moving: motion commands can be accepted."""
        return self.safety_state in ("Armed", "Moving")

    @property
    def joint_positions(self) -> list[float]:
        return self.positions

    @property
    def fault_code(self) -> str | None:
        """armd's fault code (``fault.code``), or None (no fault, or armd without fault codes)."""
        return self.fault.get("code") if self.fault else None

    @property
    def recoverable(self) -> bool:
        """armd marks the fault as safe to clear automatically (``fault.recoverable``)."""
        return bool(self.fault and self.fault.get("recoverable") is True)


ArmObservation = ArmStatus


@dataclass
class JointLimit:
    name: str
    min_position: float
    max_position: float
    max_velocity: float
    max_acceleration: float


@dataclass
class ObservationSchema:
    """``GET /api/v1/observation/schema``."""

    dof: int
    joint_names: list[str]
    limits: list[JointLimit]
    fields: list[dict]
    capabilities: dict = field(default_factory=dict)  # empty on armd without capabilities
    speed_cap: dict | None = None
    # armd joint -> URDF angle per arm joint, [{sign, offset_rad}] (model = sign * motor + offset_rad);
    # empty on armd before 2026-11-12 (rebot_companion.kinematics.JointMap.from_schema falls back)
    model_joint_map: list[dict] = field(default_factory=list)
    profile_id: str | None = None
    arm_dof: int | None = None
    gripper_joint: int | None = None
    orientation_dof: int | None = None
    gripper_policy: dict | None = None
    feetech_joints: list[dict] | None = None
    servo_max_lead_rad: float | None = None

    @property
    def capabilities_version(self) -> int:
        return int(self.capabilities.get("version") or 0)

    def require_grasp_tasks(self) -> None:
        """Raise :class:`UnsupportedError` unless armd reports the top-level ``grasp_tasks: true``."""
        if self.capabilities.get("grasp_tasks") is not True:
            raise UnsupportedError(
                "Grasp tasks need an armd that reports capability 'grasp_tasks'. Update rebot-armd."
            )

    def has(self, capability: str, min_version: int = CAPABILITIES_DEVPLAT) -> bool:
        """``capabilities[capability]`` is true and ``capabilities.version >= min_version``."""
        return self.capabilities_version >= min_version and self.capabilities.get(capability) is True

    @classmethod
    def from_dict(cls, d: dict) -> ObservationSchema:
        dof = int(d["dof"])
        raw_arm_dof = d.get("arm_dof")
        if raw_arm_dof is not None:
            arm_dof = int(raw_arm_dof)
        else:
            arm_dof = 6 if dof == 7 else (dof if dof < 6 else None)

        if "gripper_joint" in d:
            raw_gripper_joint = d["gripper_joint"]
            gripper_joint = int(raw_gripper_joint) if raw_gripper_joint is not None else None
        else:
            gripper_joint = 6 if dof == 7 and arm_dof != 7 else None

        if gripper_joint is not None and not (0 <= gripper_joint < dof):
            raise ValueError(f"gripper_joint {gripper_joint} out of range for dof={dof}")

        raw_orientation_dof = d.get("orientation_dof")
        if raw_orientation_dof is not None:
            orientation_dof = int(raw_orientation_dof)
        else:
            orientation_dof = 3 if arm_dof == 6 else (2 if arm_dof == 5 else None)

        gripper_policy = dict(d["gripper_policy"]) if isinstance(d.get("gripper_policy"), dict) else None
        profile_id = str(d["profile_id"]) if d.get("profile_id") is not None else None
        feetech_joints = list(d["feetech_joints"]) if isinstance(d.get("feetech_joints"), list) else None
        raw_lead = d.get("servo_max_lead_rad")
        servo_max_lead_rad = float(raw_lead) if raw_lead is not None else None

        return cls(
            dof=dof,
            joint_names=list(d.get("joint_names") or []),
            limits=[
                JointLimit(**{k: lim[k] for k in JointLimit.__dataclass_fields__}) for lim in d.get("limits") or []
            ],
            fields=list(d.get("fields") or []),
            capabilities=dict(d.get("capabilities") or {}),
            speed_cap=d.get("speed_cap") if isinstance(d.get("speed_cap"), dict) else None,
            model_joint_map=[m for m in d.get("model_joint_map") or [] if isinstance(m, dict)],
            profile_id=profile_id,
            arm_dof=arm_dof,
            gripper_joint=gripper_joint,
            orientation_dof=orientation_dof,
            gripper_policy=gripper_policy,
            feetech_joints=feetech_joints,
            servo_max_lead_rad=servo_max_lead_rad,
        )


# --------------------------------------------------------------------------- REST client


class ArmdClient:
    """Entry point for rebot-armd. Lease-free REST calls plus :meth:`session` for control.

    ``transport`` replaces the HTTP layer (see :mod:`rebot._http`), e.g. an in-process fake in tests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        timeout_sec: float = 5.0,
        transport: _http.Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._transport = transport

    def request(
        self, method: str, path: str, json: Any = None, params: dict | None = None, timeout: float | None = None
    ):
        """One HTTP call to armd; returns :class:`rebot._http.HttpResponse` (any status)."""
        return _http.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            params=params,
            timeout=self.timeout_sec if timeout is None else timeout,
            transport=self._transport,
        )

    def _get(self, path: str, timeout: float | None = None) -> Any:
        res = self.request("GET", path, timeout=timeout)
        res.raise_for_status()
        return res.json()

    def _post(self, path: str, json: Any = None, timeout: float | None = None) -> Any:
        res = self.request("POST", path, json=json, timeout=timeout)
        res.raise_for_status()
        return res.json()

    def get_gripper_calibration(self) -> dict:
        """``GET /api/v1/gripper/calibration`` (no lease required)."""
        return dict(self._get("/api/v1/gripper/calibration") or {})

    def grasp_dry_run(
        self,
        label: str,
        box_norm: Sequence[float],
        frame_id: int | None = None,
        ts_ms: float | None = None,
        overrides: dict | None = None,
        tag: str | None = None,
        timeout: float | None = None,
        task: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """Check grasp reachability over REST (no lease required).

        With ``task`` (needs capability ``grasp_tasks``, else :class:`UnsupportedError`) an unsupported task
        returns ``(False, "TASK_UNSUPPORTED", reason)``.
        """
        if task is not None:
            self.get_schema().require_grasp_tasks()
        body: dict = {
            "label": str(label),
            "box_norm": [float(v) for v in box_norm],
            "dry_run": True,
        }
        if frame_id is not None:
            body["frame_id"] = int(frame_id)
        if ts_ms is not None:
            body["ts_ms"] = float(ts_ms)
        if overrides is not None:
            body["overrides"] = dict(overrides)
        if tag is not None:
            body["tag"] = str(tag)
        if task is not None:
            body["task"] = str(task)
        try:
            data = self._post("/api/v1/action/grasp", json=body, timeout=timeout)
        except _http.HttpStatusError as exc:
            data = _json_dict(exc.response)
            if data.get("code") != TASK_UNSUPPORTED:
                raise
        return _dry_run_tuple(data)

    def get_observation(self) -> ArmStatus:
        """``GET /api/v1/observation`` (no lease required)."""
        return ArmStatus.from_dict(self._get("/api/v1/observation"))

    def get_schema(self, timeout: float | None = None) -> ObservationSchema:
        """``GET /api/v1/observation/schema`` (no lease required): DOF, joint names, limits, capabilities."""
        return ObservationSchema.from_dict(self._get("/api/v1/observation/schema", timeout=timeout))

    def require(self, capability: str, what: str, schema: ObservationSchema | None = None) -> ObservationSchema:
        """Raise :class:`UnsupportedError` unless armd reports ``capability`` (capabilities version >= 6)."""
        schema = schema or self.get_schema()
        if not schema.has(capability):
            raise UnsupportedError(
                f"{what} needs an armd that reports capability '{capability}' (capabilities version "
                f">= {CAPABILITIES_DEVPLAT}); this armd reports version {schema.capabilities_version}. "
                "Update rebot-armd."
            )
        return schema

    def get_poses(self) -> list[dict]:
        """``GET /api/v1/poses`` (capability ``poses``): ``[{name, positions, tcp, saved_at_ms}]``."""
        self.require("poses", "Named poses")
        return list((self._get("/api/v1/poses") or {}).get("poses") or [])

    def get_keepout(self) -> dict:
        """``GET /api/v1/workspace/keepout`` (capability ``keepout``): ``{boxes, source}``."""
        self.require("keepout", "Keep-out boxes")
        return dict(self._get("/api/v1/workspace/keepout") or {})

    def get_collision_guard(self) -> dict:
        """``GET /api/v1/safety/collision`` (capability ``collision_guard``): effective guard configuration."""
        self.require("collision_guard", "The collision guard")
        return dict(self._get("/api/v1/safety/collision") or {})

    def get_safety_describe(self) -> dict:
        """``GET /api/v1/safety/describe`` (capability ``safety_describe``): stops, faults, limits, timing, hardware."""
        self.require("safety_describe", "The safety description")
        return dict(self._get("/api/v1/safety/describe") or {})

    def get_tick_stats(self) -> dict:
        """``GET /api/v1/diagnostics/tick`` (capability ``tick_stats``): control-loop timing."""
        self.require("tick_stats", "Tick statistics")
        return dict(self._get("/api/v1/diagnostics/tick") or {})

    def e_stop(self, retries: int = 2) -> dict:
        """``POST /api/v1/safety/e_stop``. Needs no lease; latches EStop and advances the epoch.

        Returns ``{"safety_state": ..., "epoch_id": ...}``.
        """
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                res = self.request("POST", "/api/v1/safety/e_stop", timeout=min(self.timeout_sec, 1.0))
                res.raise_for_status()
                return res.json()
            except _http.HttpError as exc:
                last = exc
                time.sleep(0.1)
        raise ArmConnectionError(f"E-stop request failed: {last}")

    def session(self, client_id: str | None = None, **kwargs: Any) -> ArmSession:
        """Create (not yet opened) control session. Use as a context manager."""
        return ArmSession(self, client_id=client_id, **kwargs)


def _dry_run_tuple(data: dict) -> tuple[bool, str | None, str | None]:
    """``(reachable, code, message)`` from a dry-run reply; ``task_supported: false`` -> TASK_UNSUPPORTED."""
    if data.get("task_supported") is False:
        return False, TASK_UNSUPPORTED, data.get("task_reason") or data.get("message")
    return bool(data.get("reachable", False)), data.get("code"), data.get("message")


def _json_dict(response: Any) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _vec(v: Sequence[float] | None, n: int, name: str) -> list[float]:
    if v is None:
        raise ValueError(f"{name} cannot be None; needs {n} values")
    try:
        items = list(v)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable, got {type(v).__name__}") from exc
    out = []
    for idx, x in enumerate(items):
        if isinstance(x, bool):
            raise ValueError(f"{name}[{idx}] must be numeric, got bool: {x!r}")
        try:
            xf = float(x)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}[{idx}] must be numeric, got {x!r}") from exc
        if not math.isfinite(xf):
            raise ValueError(f"{name}[{idx}] must be finite, got {x!r}")
        out.append(xf)
    if len(out) != n:
        raise ValueError(f"{name} needs {n} values, got {len(out)}")
    return out


def _put_speed(body: dict, linear: float | None, angular: float | None) -> None:
    speed = {}
    for k, v in (("linear", linear), ("angular", angular)):
        if v is not None:
            try:
                vf = float(v)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"speed {k} must be numeric, got {v!r}") from exc
            if not math.isfinite(vf):
                raise ValueError(f"speed {k} must be finite, got {v!r}")
            if vf <= 0:
                raise ValueError(f"speed {k} must be positive (> 0), got {vf}")
            speed[k] = vf
    if speed:
        body["speed"] = speed


# --------------------------------------------------------------------------- WS session


class _Pending:
    __slots__ = ("event", "reply")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reply: dict | None = None


class ArmSession:
    """One WebSocket connection holding the armd control lease.

    A background thread reads telemetry and replies; another sends heartbeats
    every ``heartbeat_interval`` seconds (armd's own UI uses 150 ms against the
    2500 ms lease timeout). Closing the session releases the lease, and armd
    stops any motion in progress.

    Note: armd's watchdog latches ``Fault(WatchdogTimeout)`` (position held,
    torque on) once no lease owner has renewed it for ``lease_timeout_ms`` while
    the arm is torqued. Call :meth:`unload` before closing if the arm should end
    parked with torque off; otherwise the next session has to ``clear_fault()``
    and ``torque(True)`` before moving.

    ``safe_exit`` controls what leaving the ``with`` block does while the arm is
    torqued (see :meth:`safe_exit`):

    * ``"error"`` (default): on an exception (including ``KeyboardInterrupt``)
      clear a Fault, unload (park at the resting pose, then torque off) with a
      bounded ``safe_exit_timeout``, then release the lease.
    * ``"always"``: do the same on a normal exit too.
    * ``"off"``: only release the lease.

    The outcome is stored in :attr:`safe_exit_outcome`. A latched EStop is never
    cleared automatically, and torque is never disabled away from the resting
    pose (armd refuses that).
    """

    def __init__(
        self,
        client: ArmdClient,
        client_id: str | None = None,
        force: bool = False,
        heartbeat_interval: float = 0.15,
        command_timeout: float = 4.0,
        connect_timeout: float = 5.0,
        safe_exit: str = "error",
        safe_exit_timeout: float = 30.0,
        obs_hz: int | None = None,
        send_timeout: float = 10.0,
    ) -> None:
        if safe_exit not in SAFE_EXIT_POLICIES:
            raise ValueError(f"safe_exit must be one of {SAFE_EXIT_POLICIES}")
        self.client = client
        self.client_id = client_id or f"rebot-companion-{os.getpid()}-{os.urandom(4).hex()}"
        if not 0 < len(self.client_id) <= 128:
            raise ValueError("client_id must be 1..128 characters")
        self.force = force
        self.heartbeat_interval = heartbeat_interval
        self.command_timeout = command_timeout
        self.connect_timeout = connect_timeout
        self.safe_exit_policy = safe_exit
        self.safe_exit_timeout = safe_exit_timeout
        self.safe_exit_outcome: SafeExitOutcome | None = None
        self.obs_hz = None if obs_hz is None else max(1, int(obs_hz))
        self.send_timeout = float(send_timeout)  # bounds one blocking socket send

        self._ws = None
        # Reentrant: _request() holds it while it allocates a request_id, registers the reply and sends,
        # so ids reach armd in increasing order (armd rejects a request_id lower than the last one).
        self._send_lock = threading.RLock()
        self._state = threading.Condition()
        self._pending: dict[tuple[str, int], _Pending] = {}
        self._request_ids = itertools.count(1)
        self._lease_seqs = itertools.count(1)
        self._servo_seq = 0
        self._generation: int | None = None
        self._last_generation = 0  # highest generation granted on this session
        self._epoch: int | None = None
        self._obs: ArmStatus | None = None
        self._obs_rx: float | None = None  # time.monotonic() when self._obs arrived
        self._frames = 0
        self._lease_lost: dict | None = None
        self._servo_error: dict | None = None
        self._closed_reason: str | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._schema: ObservationSchema | None = None
        self._gripper_joint: int | None | object = _GRIPPER_UNSET
        self._last_schema_attempt: float | None = None
        self._unload_pending = False
        self._last_servo: tuple[float, list[float]] | None = None  # (monotonic time, positions)

    # ---- lifecycle -----------------------------------------------------------

    @property
    def ws_url(self) -> str:
        base = self.client.base_url
        if base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        query = f"?obs_hz={self.obs_hz}" if self.obs_hz else ""
        return f"{base}/api/v1/ws{query}"

    def open(self) -> ArmSession:
        if self._ws is not None:
            return self
        try:
            # Origin http://<host>:<port> equals Host, which armd's same-origin guard accepts.
            self._ws = _ws.connect(self.ws_url, timeout=self.connect_timeout, send_timeout=self.send_timeout)
        except Exception as exc:
            raise ArmConnectionError(f"Cannot open {self.ws_url}: {exc}") from exc
        self._stop.clear()
        reader = threading.Thread(target=self._reader, name="armd-ws-reader", daemon=True)
        reader.start()
        self._threads = [reader]
        try:
            self.wait_for_observation(timeout=self.connect_timeout)
            if self._schema is None and self._gripper_joint is _GRIPPER_UNSET:
                try:
                    self.schema(timeout=min(self.connect_timeout, 1.5))
                except Exception:
                    pass
            self._acquire_lease()
            heartbeat = threading.Thread(target=self._heartbeat_loop, name="armd-heartbeat", daemon=True)
            heartbeat.start()
            self._threads.append(heartbeat)
        except BaseException:
            self._shutdown()
            raise
        return self

    def close(self, release: bool = True) -> None:
        """Release the lease (armd stops motion) and close the socket.

        Releasing the lease cancels a pending unload in armd, so an unload started
        with ``unload(wait=False)`` is waited for first (up to ``safe_exit_timeout``).
        """
        if self._ws is None:
            return
        try:
            if self._unload_pending and self.has_lease:
                try:
                    self._wait_unload(self._frames, self.safe_exit_timeout)
                except ArmdError as exc:
                    _log.warning("Unload started with wait=False did not finish before close: %s", exc)
            if release and self._generation is not None and self._closed_reason is None:
                try:
                    self._request(
                        "lease_released",
                        0,
                        {"type": "release_lease", "client_id": self.client_id, "generation": self._generation},
                        timeout=1.0,
                    )
                except ArmdError:
                    pass  # closing the socket releases the lease on the server side as well
                self._generation = None
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2.0)
        self._threads = []
        self._fail_pending()

    def __enter__(self) -> ArmSession:
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self.safe_exit_policy == "always" or (self.safe_exit_policy == "error" and exc_type is not None):
                self.safe_exit()
        except BaseException as err:  # never mask the original exception
            _log.warning("safe exit failed: %s", err)
        finally:
            self.close()

    def safe_exit(self, timeout: float | None = None) -> SafeExitOutcome:
        """Bring a torqued arm to rest with torque off before the lease is released.

        Steps: if torque is off, nothing to do. A latched EStop is reported and left
        alone. Only recoverable faults are cleared and re-armed (torque on, no motion)
        before running :meth:`unload` with ``timeout`` (default ``safe_exit_timeout``);
        unrecoverable faults and E-stop remain latched and automatic unloading is skipped.
        If armd has no resting-pose profile (``UNSUPPORTED``) the arm stays torqued and
        armd latches ``Fault(WatchdogTimeout)`` once the lease is released; this is
        reported, because torque cannot be disabled away from the resting pose.
        """
        outcome = self._safe_exit(self.safe_exit_timeout if timeout is None else timeout)
        self.safe_exit_outcome = outcome
        (_log.info if outcome.ok else _log.warning)("safe exit: %s %s", outcome.action, outcome.message)
        return outcome

    def _safe_exit(self, timeout: float) -> SafeExitOutcome:
        if self._ws is None or self._closed_reason:
            return SafeExitOutcome("failed", self._closed_reason or "session is not open")
        if not self.has_lease:
            return SafeExitOutcome("no_lease", "control lease not held; cannot unload")
        try:
            obs = self.wait_for_observation(timeout=1.0)
            if not obs.is_torqued:
                return SafeExitOutcome("not_needed", f"torque off ({obs.safety_state})")
            if obs.e_stopped:
                return SafeExitOutcome("estop", "EStop is latched; not cleared automatically, arm left holding")
            if obs.faulted:
                def _is_rec(o: ArmStatus) -> bool:
                    if o.fault is not None:
                        return bool(o.recoverable)
                    if self._schema is not None and self._schema.has("fault_codes"):
                        return False
                    return o.fault_reason in ("WatchdogTimeout", "ControlDeadline", "HandoverUnsettled")

                if not _is_rec(obs):
                    return SafeExitOutcome(
                        "fault",
                        f"Fault ({obs.fault_code or obs.fault_reason or obs.safety_state}) is not recoverable; not cleared automatically, arm left holding",
                    )
                self.clear_fault(
                    obs,
                    recheck=lambda o: o.faulted and not o.e_stopped and _is_rec(o),
                )
                obs = self.wait_for_observation(timeout=1.0)
            if not obs.is_operational:
                self.torque(True)  # re-arm after clear_fault; plans no motion
            self.unload(wait=True, timeout=timeout)
            return SafeExitOutcome("unloaded", "parked at the resting pose, torque off")
        except ArmdError as exc:
            if isinstance(exc, ArmCommandRejected) and exc.code == "UNSUPPORTED":
                return SafeExitOutcome(
                    "unsupported",
                    "armd has no resting-pose profile for automatic unload; arm left torqued and armd "
                    "will latch Fault(WatchdogTimeout) after the lease is released",
                )
            return SafeExitOutcome("failed", str(exc))

    # ---- state ---------------------------------------------------------------

    @property
    def generation(self) -> int | None:
        return self._generation

    @property
    def epoch_id(self) -> int | None:
        return self._epoch

    @property
    def has_lease(self) -> bool:
        return self._generation is not None and self._lease_lost is None

    @property
    def lease_lost(self) -> dict | None:
        """The ``lease_lost`` message (or reject) that ended the lease, if it was lost."""
        return self._lease_lost

    @property
    def closed_reason(self) -> str | None:
        """Why the WebSocket closed, or None while it is open."""
        return self._closed_reason

    def latest_observation(self) -> ArmStatus | None:
        """Newest observation (None before the first). Raises :class:`ArmConnectionError` once the
        connection has closed: the last frame no longer describes the arm."""
        with self._state:
            if self._closed_reason is not None:
                raise ArmConnectionError(self._closed_reason)
            return self._obs

    def latest_observation_timed(self) -> tuple[float, ArmStatus] | None:
        """``(time.monotonic() at receipt, observation)`` of the newest observation, or None before the
        first. Raises :class:`ArmConnectionError` once the connection has closed."""
        with self._state:
            if self._closed_reason is not None:
                raise ArmConnectionError(self._closed_reason)
            return None if self._obs is None else (self._obs_rx, self._obs)

    def wait_for_observation(self, timeout: float = 2.0) -> ArmStatus:
        """Block until the next telemetry frame arrives; :class:`ArmConnectionError` when closed."""
        with self._state:
            start = self._frames
            if not self._state.wait_for(lambda: self._frames > start or self._closed_reason, timeout):
                raise ArmConnectionError("No telemetry from armd")
            if self._closed_reason is not None:
                raise ArmConnectionError(self._closed_reason)
            return self._obs

    def schema(self, timeout: float | None = None) -> ObservationSchema:
        if self._schema is None:
            try:
                self._schema = self.client.get_schema(timeout=timeout)
                self._last_schema_attempt = None
            except Exception:
                self._last_schema_attempt = time.monotonic()
                raise
        return self._schema

    def set_schema(self, schema: ObservationSchema) -> None:
        self._schema = schema
        self._gripper_joint = schema.gripper_joint
        self._last_schema_attempt = None

    def _cached_gripper_joint(self, pos_len: int | None = None) -> int | None:
        if self._schema is not None:
            return self._schema.gripper_joint
        if self._gripper_joint is not _GRIPPER_UNSET:
            return self._gripper_joint  # type: ignore[return-value]
        obs = None
        try:
            obs = self.latest_observation()
        except ArmConnectionError:
            pass
        if obs is not None:
            if len(obs.positions) == 7:
                return 6
            return None
        if pos_len is not None:
            if pos_len == 7:
                return 6
            return None
        return None

    @property
    def gripper_joint(self) -> int | None:
        if self._schema is not None:
            return self._schema.gripper_joint
        if self._gripper_joint is not _GRIPPER_UNSET:
            return self._gripper_joint  # type: ignore[return-value]
        now = time.monotonic()
        if self._last_schema_attempt is None or (now - self._last_schema_attempt >= _SCHEMA_RETRY_COOLDOWN_S):
            try:
                self._gripper_joint = self.schema(timeout=0.5).gripper_joint
                return self._gripper_joint  # type: ignore[return-value]
            except Exception:
                pass
        return self._cached_gripper_joint()

    # ---- background threads --------------------------------------------------

    def _send(self, message: dict) -> None:
        ws = self._ws
        if ws is None:
            raise ArmConnectionError(self._closed_reason or "session is not open")
        try:
            with self._send_lock:
                ws.send(json.dumps(message, allow_nan=False))
        except _ws.WebSocketClosed as exc:
            raise ArmConnectionError(f"WebSocket closed: {exc}") from exc

    def _reader(self) -> None:
        ws = self._ws
        try:
            while not self._stop.is_set():
                text = ws.recv()
                try:
                    msg = json.loads(text)
                except (TypeError, ValueError):
                    continue
                if isinstance(msg, dict):
                    self._dispatch(msg)
        except _ws.WebSocketClosed as exc:
            reason = f"WebSocket closed ({exc})"
        except Exception as exc:  # protocol error, bad UTF-8, malformed observation, ...
            reason = f"reader failed: {type(exc).__name__}: {exc}"
        else:
            reason = "session closed"
        with self._state:
            self._closed_reason = reason
            self._state.notify_all()
        # The session is dead: stop heartbeats and close the socket so armd's lease watchdog takes over
        # instead of a heartbeat thread renewing a lease nobody reads replies for.
        self._stop.set()
        try:
            ws.close()
        except Exception:
            pass
        self._fail_pending()

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        with self._state:
            if kind is None:
                if "positions" in msg and "safety_state" in msg:
                    self._obs = ArmStatus.from_dict(msg)
                    self._obs_rx = time.monotonic()
                    self._advance_epoch(self._obs.epoch_id)
                    self._frames += 1
                    self._state.notify_all()
                return
            if isinstance(msg.get("epoch_id"), int):
                # armd acks persisted commands (poses, keep-out, collision, grasp setup) after the
                # disk write, so an ack can carry an epoch older than telemetry already received.
                self._advance_epoch(msg["epoch_id"])
            pending: _Pending | None = None
            if kind in ("command_ack", "command_reject"):
                pending = self._pending.pop(("command", msg.get("request_id")), None)
            elif kind in ("lease_granted", "lease_rejected"):
                pending = self._pending.pop(("lease", msg.get("seq")), None)
            elif kind == "lease_released":
                pending = self._pending.pop(("lease_released", 0), None)
            elif kind == "lease_lost":
                gen = msg.get("generation")
                if self._generation is not None and (gen is None or gen == self._generation):
                    self._lease_lost = msg
                    self._generation = None
                    for key in [k for k in self._pending if k[0] == "command"]:
                        p = self._pending.pop(key)
                        p.reply = {"type": "command_reject", "code": "LEASE_LOST", "message": "lease lost", **msg}
                        p.event.set()
            elif kind == "servo_reject":
                self._servo_error = msg
            self._state.notify_all()
        if pending is not None:
            pending.reply = msg
            pending.event.set()

    def _advance_epoch(self, epoch: int) -> None:
        """Epochs only move forward: a delayed message never lowers the epoch (caller holds _state)."""
        if self._epoch is None or epoch > self._epoch:
            self._epoch = epoch

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            if self._closed_reason is not None:
                return
            gen, epoch = self._generation, self._epoch
            if gen is None or epoch is None:
                continue
            try:
                self._send(
                    {
                        "type": "heartbeat",
                        "client_id": self.client_id,
                        "generation": gen,
                        "epoch_id": epoch,
                        "timestamp_ms": int(time.time() * 1000),
                    }
                )
            except ArmdError:
                return

    def _fail_pending(self) -> None:
        with self._state:
            pending, self._pending = list(self._pending.values()), {}
        for p in pending:
            p.reply = None
            p.event.set()

    # ---- request/response ----------------------------------------------------

    def _request(self, kind: str, key: Any, message: dict, timeout: float, id_field: str | None = None) -> dict:
        """Send ``message`` and wait for its reply. With ``id_field``, ``key`` is ignored and the next
        request id is allocated under the send lock and written to ``message[id_field]``."""
        p = _Pending()
        with self._send_lock:
            if id_field is not None:
                key = message[id_field] = next(self._request_ids)
            with self._state:
                if self._closed_reason:
                    raise ArmConnectionError(self._closed_reason)
                self._pending[(kind, key)] = p
            try:
                self._send(message)
            except BaseException:
                with self._state:
                    self._pending.pop((kind, key), None)
                raise
        if not p.event.wait(timeout):
            with self._state:
                self._pending.pop((kind, key), None)
            raise ArmConnectionError(f"No reply to {message.get('type')} within {timeout:.1f}s")
        if p.reply is None:
            raise ArmConnectionError(self._closed_reason or "session closed")
        return p.reply

    def _acquire_lease(self) -> None:
        seq = next(self._lease_seqs)
        reply = self._request(
            "lease",
            seq,
            {"type": "acquire_lease", "seq": seq, "client_id": self.client_id, "force": self.force},
            timeout=self.command_timeout,
        )
        if reply.get("type") != "lease_granted" or not isinstance(reply.get("generation"), int):
            reason = str(reply.get("error", "rejected"))
            extra = f" (retry in {reply['remaining_ms']} ms)" if "remaining_ms" in reply else ""
            raise LeaseError(f"armd rejected the lease: {reason}{extra}", reason=reason, raw=reply)
        with self._state:
            if reply["generation"] < self._last_generation:  # armd's lease generations only increase
                raise LeaseError("armd answered with an older lease generation", reason="stale", raw=reply)
            self._generation = self._last_generation = reply["generation"]
            self._lease_lost = None
            self._servo_seq = 0

    def _check_lease(self) -> int:
        if self._lease_lost is not None:
            raise LeaseError(
                "Control lease was lost"
                + (f" ({self._lease_lost.get('reason')})" if self._lease_lost.get("reason") else ""),
                reason=str(self._lease_lost.get("reason", "lost")),
                raw=self._lease_lost,
            )
        if self._generation is None:
            raise LeaseError("Session does not hold the control lease", reason="not_held")
        if self._epoch is None:
            raise ArmConnectionError("Epoch unknown: no telemetry yet")
        return self._generation

    def command(
        self, path: str, body: dict | None = None, timeout: float | None = None, epoch_id: int | None = None
    ) -> dict:
        """Send one actuator command over the lease-holding socket; returns the ``command_ack``.

        Raises :class:`ArmCommandRejected` on ``command_reject``. A rejected
        ``EPOCH_MISMATCH`` is not retried: the epoch advanced because of an
        e-stop or fault, and the command was issued before it. ``epoch_id`` binds
        the command to that epoch instead of the latest one seen.
        """
        generation = self._check_lease()
        reply = self._request(
            "command",
            None,
            {
                "type": "command",
                "request_id": None,  # allocated under the send lock
                "path": path,
                "body": body or {},
                "generation": generation,
                "epoch_id": self._epoch if epoch_id is None else epoch_id,
                "client_id": self.client_id,
            },
            timeout=timeout or self.command_timeout,
            id_field="request_id",
        )
        if reply.get("type") != "command_ack":
            if reply.get("code") == "LEASE_LOST":
                with self._state:
                    self._lease_lost = self._lease_lost or reply
                    self._generation = None
                raise LeaseError(reply.get("message", "lease lost"), reason="LEASE_LOST", raw=reply)
            raise ArmCommandRejected(reply)
        return reply

    # ---- actuator API --------------------------------------------------------

    def torque(self, enable: bool) -> dict:
        """Enable/disable torque.

        armd refuses ``enable=False`` (``ARM_NOT_AT_RESTING_POSE``) unless the arm is
        settled at its resting pose, and in EStop/Fault; use :meth:`unload` to park
        and cut torque in one step.
        """
        return self.command("/api/v1/torque", {"enable": bool(enable)})

    def stop(self) -> dict:
        """Stop the active trajectory/servo stream (decelerate and hold; torque stays on).

        Also cancels a pending unload.
        """
        self._unload_pending = False
        return self.command("/api/v1/motion/stop")

    def clear_fault(
        self,
        observation: ArmStatus | None = None,
        recheck: Callable[[ArmStatus], bool] | None = None,
        attempts: int = 3,
    ) -> dict:
        """Clear the EStop/Fault seen in ``observation`` back to Idle (advances the epoch).

        The command carries ``observation.epoch_id`` (default: the latest observation), so armd
        rejects it with ``EPOCH_MISMATCH`` when a newer stop latched after that observation; a
        newer E-stop is never cleared by a stale decision. On ``EPOCH_MISMATCH`` a fresh
        observation is fetched and ``recheck(obs)`` decides whether that new stop may be cleared
        too; without ``recheck`` (or when it says no) the rejection or :class:`ArmFaultError` is
        raised. Call :meth:`torque` (True) before moving again.
        """
        obs = observation or self.latest_observation() or self.wait_for_observation()
        for attempt in range(max(1, int(attempts))):
            try:
                return self.command("/api/v1/safety/clear_fault", epoch_id=obs.epoch_id)
            except ArmCommandRejected as exc:
                if exc.code != "EPOCH_MISMATCH" or recheck is None or attempt == attempts - 1:
                    raise
            obs = self.wait_for_observation(timeout=2.0)
            if not obs.is_stopped:
                return {"type": "command_ack", "safety_state": obs.safety_state, "epoch_id": obs.epoch_id}
            if not recheck(obs):
                raise ArmFaultError(obs, f"a newer {obs.safety_state} latched before clear_fault; not cleared")
        raise AssertionError("unreachable")  # pragma: no cover

    def e_stop(self) -> dict:
        """Emergency stop: sent on this socket and via REST. Neither path requires the lease."""
        try:
            self._send({"type": "estop"})
        except ArmdError:
            pass
        return self.client.e_stop()

    def move_to(
        self,
        target_positions: Sequence[float],
        duration: float | None = None,
        wait: bool = True,
        timeout: float | None = None,
        tolerance: float | Sequence[float | None] | None = None,
    ) -> ArmStatus | None:
        """Plan a quintic trajectory to joint targets (``/api/v1/motion/jog``) and wait for arrival.

        ``duration`` is a request; armd stretches it to respect velocity and
        acceleration limits. Arrival means armd reports the trajectory finished
        (state left ``Moving``) and measured positions are within ``tolerance``.
        """
        target = []
        for idx, x in enumerate(target_positions):
            try:
                xf = float(x)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"target_positions[{idx}] must be numeric, got {x!r}") from exc
            if not math.isfinite(xf):
                raise ValueError(f"target_positions[{idx}] must be finite, got {x!r}")
            target.append(xf)
        body: dict = {"target_positions": target}
        if duration is not None:
            try:
                df = float(duration)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"duration must be numeric, got {duration!r}") from exc
            if not math.isfinite(df) or df <= 0:
                raise ValueError(f"duration must be positive and finite, got {duration!r}")
            body["duration_secs"] = df
        frames_at_ack = self._frames_after(lambda: self.command("/api/v1/motion/jog", body))
        if not wait:
            return None
        return self._wait_motion(frames_at_ack, target, self._timeout_for(duration, timeout), tolerance)

    jog = move_to

    def home(self, duration: float | None = None, wait: bool = True, timeout: float | None = None) -> ArmStatus | None:
        """Move to armd's configured home positions."""
        body = {}
        if duration is not None:
            try:
                df = float(duration)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"duration must be numeric, got {duration!r}") from exc
            if not math.isfinite(df) or df <= 0:
                raise ValueError(f"duration must be positive and finite, got {duration!r}")
            body["duration_secs"] = df
        frames_at_ack = self._frames_after(lambda: self.command("/api/v1/motion/home", body))
        if not wait:
            return None
        return self._wait_motion(frames_at_ack, None, self._timeout_for(duration, timeout), None)

    def unload(self, wait: bool = True, timeout: float = 30.0) -> ArmStatus | None:
        """Normal unload: return to the resting pose, settle, then cut torque (state ends ``Idle``).

        armd answers ``UNSUPPORTED`` when the follower lacks a resting-pose
        profile or the required feedback. Raises :class:`UnloadFailed` as soon as
        armd reports the motion finished with torque still on (failed torque cut).

        With ``wait=False``, :meth:`close` waits for the unload before releasing
        the lease, because a lease release cancels a pending unload in armd.
        """
        frames_at_ack = self._frames_after(lambda: self.command("/api/v1/motion/unload"))
        self._unload_pending = True
        if not wait:
            return None
        return self._wait_unload(frames_at_ack, timeout)

    def _wait_unload(self, frames_at_ack: int, timeout: float) -> ArmStatus:
        def done(o: ArmStatus) -> bool:
            if not o.is_torqued and o.safety_state == "Idle":
                return True
            if o.is_torqued and o.safety_state == "Armed":
                # armd keeps Moving through the unload and settle phase; Armed with torque
                # on means set_torque(false) failed after the settle (supervisor.rs:1430).
                raise UnloadFailed(o)
            return False

        try:
            return self._wait_until(frames_at_ack, done, timeout, "unload to finish")
        finally:
            self._unload_pending = False

    # ---- high-level actions (armd /api/v1/action/*) --------------------------

    def action(self, path: str, body: dict | None = None, timeout: float | None = None) -> ActionResult:
        """Send an action command and block until its record in the observation is terminal.

        armd answers with ``command_ack{action_id, expected_duration_s}`` and reports progress in
        ``observation.action``. Default ``timeout`` is ``2 * expected_duration_s + 5`` capped at
        60 s; on timeout ``stop`` is sent and :class:`MotionTimeout` raised. A ``failed`` or
        ``cancelled`` action raises :class:`ActionFailed` with armd's code. Rejections before
        the action starts (``IK_UNREACHABLE``, ``WORKSPACE_VIOLATION``, ``MOTION_BUSY``, ...)
        raise :class:`ArmCommandRejected`.
        """
        return self._action(path, body, timeout, legacy_wait=None)

    def move_pose(
        self,
        position: Sequence[float],
        rpy: Sequence[float] | None = None,
        quat: Sequence[float] | None = None,
        linear_speed: float | None = None,
        angular_speed: float | None = None,
        tag: str | None = None,
        timeout: float | None = None,
        auto_yaw: Optional[bool] = None,
    ) -> ActionResult:
        """Straight-line Cartesian move of ``end_link`` to ``position`` (m, base frame).

        Orientation: ``rpy`` (rad, URDF fixed-axis ``Rz @ Ry @ Rx``) or ``quat`` ``[w, x, y, z]``;
        neither keeps the current orientation. armd defaults: 0.05 m/s, 0.5 rad/s.
        (:meth:`move_to` is the joint-space jog.)
        ``auto_yaw``: when True on 5-DOF arms (orient_dof=2), armd solves yaw from position.
        """
        if auto_yaw is not None and not isinstance(auto_yaw, bool):
            raise ValueError(f"auto_yaw must be bool or None, got {type(auto_yaw).__name__}: {auto_yaw!r}")
        if auto_yaw is not None and getattr(self, "_schema", None) is None:
            now = time.monotonic()
            last_att = getattr(self, "_last_schema_attempt", None)
            if last_att is None or (now - last_att >= _SCHEMA_RETRY_COOLDOWN_S):
                if hasattr(self, "schema") and callable(self.schema):
                    try:
                        self.schema(timeout=1.0)
                    except Exception:
                        pass
        caps = getattr(getattr(self, "_schema", None), "capabilities", None) or {}
        has_auto_yaw_cap = bool(isinstance(caps, dict) and caps.get("auto_yaw") is True)
        if not has_auto_yaw_cap:
            if auto_yaw is True:
                raise UnsupportedError("armd does not advertise auto_yaw capability", code="UNSUPPORTED")
            if auto_yaw is False:
                auto_yaw = None
        if rpy is not None and quat is not None:
            raise ValueError("give rpy or quat, not both")
        body: dict = {"position": _vec(position, 3, "position")}
        body["orientation"] = (
            {"rpy": _vec(rpy, 3, "rpy")} if rpy is not None else {"quat": _vec(quat, 4, "quat")} if quat else None
        )
        _put_speed(body, linear_speed, angular_speed)
        if tag is not None:
            body["tag"] = str(tag)
        if auto_yaw is not None:
            body["auto_yaw"] = bool(auto_yaw)
        return self._action("/api/v1/action/move_to", body, timeout, legacy_wait=None)

    def move_relative(
        self,
        frame: str = "base",
        d: Sequence[float] = (0.0, 0.0, 0.0),
        dyaw: float | None = None,
        dpitch: float | None = None,
        droll: float | None = None,
        level: bool | None = None,
        linear_speed: float | None = None,
        angular_speed: float | None = None,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> ActionResult:
        """Relative Cartesian move in ``frame`` (``base`` | ``tool`` | ``camera``); rotations in rad.

        Rotations are centred on the ``end_link`` origin. In the ``camera`` frame translation uses
        OpenCV axes (x right, y down, z optical); ``dpitch > 0`` looks down, ``dyaw > 0`` looks
        right, and ``level`` (armd default true) keeps the image level.
        """
        if frame not in ("base", "tool", "camera"):
            raise ValueError("frame must be base, tool or camera")
        body: dict = {"frame": frame, "d": _vec(d, 3, "d")}
        for key, val in (("dyaw", dyaw), ("dpitch", dpitch), ("droll", droll)):
            if val is not None:
                body[key] = float(val)
        if level is not None:
            body["level"] = bool(level)
        _put_speed(body, linear_speed, angular_speed)
        if tag is not None:
            body["tag"] = str(tag)
        return self._action("/api/v1/action/move_relative", body, timeout, legacy_wait=None)

    def gripper(
        self,
        command: str,
        fraction: float | None = None,
        grasp: bool | None = None,
        speed: float | None = None,
        timeout_s: float | None = None,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> ActionResult:
        """Gripper action: ``open``, ``close`` (``grasp`` default true) or ``move`` to ``fraction``.

        ``fraction``: 0 closed .. 1 open (``move`` only). ``speed`` rad/s (armd default 1.0, max 2.0);
        ``timeout_s`` is armd's own grasp timeout (default 6). The result's ``action.holding`` and
        ``action.gripper_gap_rad`` report the grasp verdict.
        """
        if command not in ("open", "close", "move"):
            raise ValueError("command must be open, close or move")
        body: dict = {"command": command}
        if command == "move":
            if fraction is None or not 0.0 <= float(fraction) <= 1.0:
                raise ValueError("move needs fraction in [0, 1]")
            body["fraction"] = float(fraction)
        if grasp is not None:
            body["grasp"] = bool(grasp)
        if speed is not None:
            body["speed"] = float(speed)
        if timeout_s is not None:
            body["timeout_s"] = float(timeout_s)
        if tag is not None:
            body["tag"] = str(tag)
        if timeout is None and timeout_s is not None:
            timeout = float(timeout_s) + 5.0
        return self._action("/api/v1/action/gripper", body, timeout, legacy_wait=None)

    def gripper_zero_check(
        self,
        store_offset: bool = False,
        threshold_rad: float = 0.10,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> ActionResult:
        """Run mechanical zero check of the gripper."""
        body: dict = {
            "store_offset": bool(store_offset),
            "threshold_rad": float(threshold_rad),
        }
        if tag is not None:
            body["tag"] = str(tag)
        return self._action("/api/v1/action/gripper_zero_check", body, timeout, legacy_wait=None)

    def home_action(self, timeout: float | None = None) -> ActionResult:
        """:meth:`home` as an action (armd reports kind ``home``); falls back to the state-based wait."""
        return self._action(
            "/api/v1/motion/home",
            {},
            timeout,
            legacy_wait=lambda frames, t: self._wait_motion(frames, None, t, None),
        )

    def park_action(self, timeout: float | None = None) -> ActionResult:
        """:meth:`unload` as an action (kind ``park``): rest pose, then torque off."""
        self._unload_pending = True
        try:
            return self._action(
                "/api/v1/motion/unload",
                {},
                timeout or 30.0,
                legacy_wait=lambda frames, t: self._wait_unload(frames, t),
            )
        finally:
            self._unload_pending = False

    def grasp(
        self,
        label: str,
        box_norm: Sequence[float],
        frame_id: int | None = None,
        ts_ms: float | None = None,
        overrides: dict | None = None,
        tag: str | None = None,
        timeout: float | None = None,
        task: str | None = None,
    ) -> ActionResult:
        """``/api/v1/action/grasp``: armd centres on the box, descends, closes and lifts.

        ``box_norm`` is ``[x0, y0, x1, y1]`` normalised to the detector image. The result's
        ``action.holding`` tells whether an object is in the gripper. Default timeout 120 s.
        ``task`` names a stored grasp task (capability ``grasp_tasks``, else :class:`UnsupportedError`);
        armd rejects it with ``TASK_UNSUPPORTED`` when the loaded vision model cannot serve it.
        """
        if task is not None:
            self.schema().require_grasp_tasks()
        body: dict = {"label": str(label), "box_norm": _vec(box_norm, 4, "box_norm")}
        if frame_id is not None:
            body["frame_id"] = int(frame_id)
        if ts_ms is not None:
            body["ts_ms"] = ts_ms
        if overrides:
            body["overrides"] = dict(overrides)
        if tag is not None:
            body["tag"] = str(tag)
        if task is not None:
            body["task"] = str(task)
        return self._action("/api/v1/action/grasp", body, 120.0 if timeout is None else timeout, legacy_wait=None)

    def grasp_dry_run(
        self,
        label: str,
        box_norm: Sequence[float],
        frame_id: int | None = None,
        ts_ms: float | None = None,
        overrides: dict | None = None,
        tag: str | None = None,
        timeout: float | None = None,
        task: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """Check grasp reachability over WebSocket (returns (reachable, code, message)).

        With ``task`` (capability ``grasp_tasks``, else :class:`UnsupportedError`) an unsupported task
        returns ``(False, "TASK_UNSUPPORTED", reason)``.
        """
        if task is not None:
            self.schema().require_grasp_tasks()
        body: dict = {
            "label": str(label),
            "box_norm": _vec(box_norm, 4, "box_norm"),
            "dry_run": True,
        }
        if frame_id is not None:
            body["frame_id"] = int(frame_id)
        if ts_ms is not None:
            body["ts_ms"] = float(ts_ms)
        if overrides:
            body["overrides"] = dict(overrides)
        if tag is not None:
            body["tag"] = str(tag)
        if task is not None:
            body["task"] = str(task)
        try:
            ack = self.command("/api/v1/action/grasp", body, timeout=timeout)
        except ArmCommandRejected as exc:
            if exc.code != TASK_UNSUPPORTED:
                raise
            return False, TASK_UNSUPPORTED, str(exc.raw.get("message") or "grasp task not supported")
        return _dry_run_tuple(ack)

    # ---- developer platform (capabilities version 6) -------------------------

    def require(self, capability: str, what: str) -> None:
        """Raise :class:`UnsupportedError` unless armd reports ``capability``."""
        self.client.require(capability, what, self.schema())

    def save_pose(self, name: str, positions: Sequence[float] | None = None) -> dict:
        """Store a named pose (default: the measured joint positions). Capability ``poses``."""
        self.require("poses", "Saving poses")
        body: dict = {"name": str(name)}
        if positions is not None:
            body["positions"] = [float(x) for x in positions]
        return self.command("/api/v1/poses/save", body)

    def delete_pose(self, name: str) -> dict:
        """Delete a named pose (``POSE_NOT_FOUND`` when unknown). Capability ``poses``."""
        self.require("poses", "Deleting poses")
        return self.command("/api/v1/poses/delete", {"name": str(name)})

    def move_to_pose(
        self,
        name: str,
        speed_scale: float | None = None,
        include_gripper: bool | None = None,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> ActionResult:
        """Joint-space move to a named pose (action kind ``move_to_pose``). Capability ``poses``."""
        self.require("poses", "Moving to a named pose")
        body: dict = {"name": str(name)}
        if speed_scale is not None:
            if not 0.1 <= float(speed_scale) <= 1.0:
                raise ValueError("speed_scale must be within 0.1..1.0")
            body["speed_scale"] = float(speed_scale)
        if include_gripper is not None:
            body["include_gripper"] = bool(include_gripper)
        if tag is not None:
            body["tag"] = str(tag)
        return self._action("/api/v1/action/move_to_pose", body, timeout, legacy_wait=None)

    def place(
        self,
        pose: str | None = None,
        position: Sequence[float] | None = None,
        rpy: Sequence[float] | None = None,
        approach_height_m: float = 0.08,
        release: bool = True,
        retreat: bool = True,
        speed_scale: float = 1.0,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> ActionResult:
        """``/api/v1/action/place``: lift, approach above the target, descend, open, retreat. Capability ``place``.

        Exactly one of ``pose`` (a named pose; its stored TCP is the place point) or ``position``
        (x, y, z in metres). ``rpy``: orientation in radians, default the pose's / the current one.
        """
        if (pose is None) == (position is None):
            raise ValueError("place() needs exactly one of pose= or position=")
        if not 0.02 <= float(approach_height_m) <= 0.30:
            raise ValueError("approach_height_m must be within 0.02..0.30")
        if not 0.1 <= float(speed_scale) <= 1.0:
            raise ValueError("speed_scale must be within 0.1..1.0")
        self.require("place", "Placing")
        body: dict = {
            "approach_height_m": float(approach_height_m),
            "release": bool(release),
            "retreat": bool(retreat),
            "speed_scale": float(speed_scale),
        }
        if pose is not None:
            body["pose"] = str(pose)
        else:
            body["position"] = _vec(position, 3, "position")
        if rpy is not None:
            body["orientation"] = {"rpy": _vec(rpy, 3, "rpy")}
        if tag is not None:
            body["tag"] = str(tag)
        return self._action("/api/v1/action/place", body, timeout, legacy_wait=None)

    def set_collision_guard(self, enabled: bool | None = None, sensitivity: str | None = None) -> dict:
        """``/api/v1/safety/collision`` (capability ``collision_guard``): switch the guard or change its sensitivity."""
        self.require("collision_guard", "The collision guard")
        body: dict = {}
        if enabled is not None:
            body["enabled"] = bool(enabled)
        if sensitivity is not None:
            body["sensitivity"] = sensitivity
        return self.command("/api/v1/safety/collision", body)

    def set_keepout(self, boxes: Sequence[dict]) -> dict:
        """Replace the runtime keep-out boxes (``[{name, min_m, max_m}]``, at most 16). Capability ``keepout``."""
        self.require("keepout", "Keep-out boxes")
        return self.command("/api/v1/workspace/keepout", {"boxes": [dict(b) for b in boxes]})

    def set_gripper_calibration(
        self,
        points: Sequence[Sequence[float]] | dict | None = None,
        offset_rad: float | None = None,
    ) -> dict:
        """Write gripper calibration over WS (requires lease)."""
        body: dict = {}
        if offset_rad is not None:
            body["offset_rad"] = float(offset_rad)
        if points is not None:
            if isinstance(points, dict):
                body["jaw_opening"] = points
            else:
                body["jaw_opening"] = {
                    "points": [[float(p[0]), float(p[1])] for p in points],
                    "calibrated": True,
                }
        else:
            try:
                cur = self.client.get_gripper_calibration()
                jo = cur.get("jaw_opening") or (cur.get("policy") or {}).get("jaw_opening")
            except Exception:
                jo = None
            if not jo:
                jo = {"points": [[0.0, 0.0], [-5.7, 0.085]], "calibrated": False}
            body["jaw_opening"] = jo
        return self.command("/api/v1/gripper/calibration", body)

    def _action(self, path: str, body: dict | None, timeout: float | None, legacy_wait: Any) -> ActionResult:
        t0 = time.monotonic()
        ack: dict = {}

        def send() -> None:
            ack.update(self.command(path, body or {}))

        frames_at_ack = self._frames_after(send)
        action_id = ack.get("action_id")
        expected = ack.get("expected_duration_s")
        if timeout is None:
            timeout = min(2.0 * float(expected) + 5.0, ACTION_MAX_TIMEOUT_S) if expected else ACTION_MAX_TIMEOUT_S
        if not isinstance(action_id, int):
            if legacy_wait is None:
                try:
                    self.stop()
                except ArmdError:
                    pass
                raise ArmdError(f"armd accepted {path} but returned no action_id (armd too old?); stop sent")
            obs = legacy_wait(frames_at_ack, timeout)
            return ActionResult(None, obs, ack, time.monotonic() - t0)

        def finished(o: ArmStatus) -> bool:
            a = o.action
            prev = o.previous_action
            if a is not None and a.id > action_id and prev is not None and prev.id == action_id and prev.terminal:
                a = prev  # replaced (or observations decimated by obs_hz): armd keeps it for 2 s
            if a is not None and a.id == action_id and a.terminal:
                if a.state != "done":
                    raise ActionFailed(a.code or a.state.upper(), a.message or "", a, o)
                return True
            if a is not None and a.id > action_id:
                # The observation carries only the latest record: a newer action replaced this one.
                raise ActionFailed("SUPERSEDED", f"replaced by {a.kind} action {a.id}", None, o)
            if o.is_stopped:  # armd reconciles the action in the same tick; this is a fallback
                code = "ESTOP_ACTIVE" if o.e_stopped else "FAULT"
                raise ActionFailed(code, o.fault_reason or "", a if a is not None and a.id == action_id else None, o)
            return False

        obs = self._wait_until(frames_at_ack, finished, timeout, f"action {action_id} to finish", check_stopped=False)
        record = obs.action
        if (record is None or record.id != action_id) and obs.previous_action and obs.previous_action.id == action_id:
            record = obs.previous_action
        return ActionResult(record, obs, ack, time.monotonic() - t0)

    def servo(self, positions: Sequence[float]) -> None:
        """Send one streaming servo frame (no reply on success).

        armd follows the latest frame with per-joint velocity limits and drops
        the stream 200 ms after the last frame, so send at >= 10 Hz (armd's UI
        uses 50 Hz). armd discards, without a reply, frames more than 0.35 rad
        from its commanded position (gripper exempt). This method raises
        :class:`ServoJumpError` instead of sending such a frame, measured from
        the previous frame (if sent within 200 ms) or from the observed
        positions. armd's commanded position trails the stream under velocity
        limits, so keep steps well below 0.35 rad. A ``servo_reject`` received
        since the previous call is raised here as :class:`ServoRejected`.
        """
        self._raise_servo_error()
        if positions is None:
            raise ValueError("servo positions cannot be None")
        try:
            raw_positions = list(positions)
        except TypeError as exc:
            raise ValueError(f"servo positions must be an iterable, got {type(positions).__name__}") from exc
        if not raw_positions:
            raise ValueError("servo positions cannot be empty")
        target = []
        for idx, x in enumerate(raw_positions):
            if isinstance(x, bool):
                raise ValueError(f"servo position[{idx}] must be numeric, got {x!r}")
            try:
                xf = float(x)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"servo position[{idx}] must be numeric, got {x!r}") from exc
            if not math.isfinite(xf):
                raise ValueError(f"servo position[{idx}] must be finite, got {x!r}")
            target.append(xf)

        if self._schema is not None and getattr(self._schema, "dof", None) is not None:
            expected_dof = int(self._schema.dof)
            if len(target) != expected_dof:
                raise ValueError(f"servo target length ({len(target)}) does not match schema DOF ({expected_dof})")

        now = time.monotonic()
        last = self._last_servo
        if last is not None and now - last[0] <= 0.2:
            ref = last[1]
        else:
            obs = self.latest_observation()
            ref = obs.positions if obs is not None else None
        gj = self._cached_gripper_joint(len(target))
        if ref is not None and len(ref) == len(target):
            for i, (t, r) in enumerate(zip(target, ref)):
                if gj is not None and i == gj:
                    continue
                if not math.isfinite(r):
                    raise ServoJumpError(
                        f"servo reference position joint {i + 1} is not finite ({r!r}); cannot verify jump safety"
                    )
                if abs(t - r) > SERVO_MAX_STEP:
                    raise ServoJumpError(
                        f"servo frame joint {i + 1} steps {abs(t - r):.3f} rad (> {SERVO_MAX_STEP} rad) from the "
                        "last commanded target; armd would drop it without a reply"
                    )
        generation = self._check_lease()
        self._servo_seq += 1
        self._send(
            {
                "type": "servo",
                "client_id": self.client_id,
                "generation": generation,
                "sequence_id": self._servo_seq,
                "epoch_id": self._epoch,
                "positions": target,
            }
        )
        self._last_servo = (now, target)

    def servo_stream(self, frames: Iterable[Sequence[float]], rate_hz: float = 50.0) -> None:
        """Stream frames at ``rate_hz`` (>= 10 Hz; armd drops the stream 200 ms after the last frame).

        Each frame is checked like :meth:`servo` (0.35 rad step limit).
        """
        if rate_hz < SERVO_MIN_RATE_HZ:
            raise ValueError(f"rate_hz must be >= {SERVO_MIN_RATE_HZ:g} Hz (armd motion_frame_timeout is 200 ms)")
        period = 1.0 / rate_hz
        next_t = time.monotonic()
        for frame in frames:
            self.servo(frame)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        time.sleep(min(0.05, period))
        self._raise_servo_error()

    # ---- helpers -------------------------------------------------------------

    def _raise_servo_error(self) -> None:
        with self._state:
            err, self._servo_error = self._servo_error, None
        if err is not None:
            raise ServoRejected(err)

    def _frames_after(self, fn: Any) -> int:
        fn()
        with self._state:
            return self._frames

    @staticmethod
    def _timeout_for(duration: float | None, timeout: float | None) -> float:
        if timeout is not None:
            return timeout
        return 30.0 if duration is None else 2.0 * float(duration) + 5.0

    def _tolerances(self, target: Sequence[float], tolerance: Any) -> list[float | None]:
        if tolerance is None:
            tol: list[float | None] = [0.03] * len(target)
            if self._schema is not None:
                gj = self._schema.gripper_joint
            elif self._gripper_joint is not _GRIPPER_UNSET:
                gj = self._gripper_joint  # type: ignore[assignment]
            else:
                now = time.monotonic()
                if self._last_schema_attempt is None or (now - self._last_schema_attempt >= _SCHEMA_RETRY_COOLDOWN_S):
                    try:
                        gj = self.schema(timeout=0.5).gripper_joint
                    except Exception:
                        gj = self._cached_gripper_joint(len(target))
                else:
                    gj = self._cached_gripper_joint(len(target))
            if gj is not None and len(target) > gj:
                tol[gj] = None
            elif self._schema is None and gj is None and len(target) >= 7:
                tol[6] = None
            return tol
        if isinstance(tolerance, (int, float)):
            return [float(tolerance)] * len(target)
        tol = list(tolerance)
        if len(tol) != len(target):
            raise ValueError("tolerance length must match target length")
        return tol

    def _wait_motion(self, frames_at_ack: int, target: list[float] | None, timeout: float, tolerance: Any) -> ArmStatus:
        tol = self._tolerances(target, tolerance) if target is not None else None

        def arrived(o: ArmStatus) -> bool:
            if o.safety_state == "Moving":
                return False
            if target is None or tol is None:
                return o.safety_state == "Armed"
            return len(o.positions) == len(target) and all(
                t is None or abs(p - q) <= t for p, q, t in zip(o.positions, target, tol)
            )

        return self._wait_until(frames_at_ack, arrived, timeout, "motion to finish")

    def _wait_until(
        self, frames_at_ack: int, predicate: Any, timeout: float, what: str, check_stopped: bool = True
    ) -> ArmStatus:
        # Skip frames that may have been computed before the command took effect.
        min_frame = frames_at_ack + 2
        deadline = time.monotonic() + timeout
        with self._state:
            while True:
                obs = self._obs
                if self._closed_reason:
                    raise ArmConnectionError(self._closed_reason)
                if self._lease_lost is not None:
                    break
                if obs is not None and self._frames >= min_frame:
                    if check_stopped and obs.is_stopped:
                        raise ArmFaultError(obs)
                    if predicate(obs):
                        return obs
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._state.wait(min(remaining, 0.1))
        self._check_lease()
        try:
            self.stop()
        except ArmdError:
            pass
        raise MotionTimeout(f"Timed out after {timeout:.1f}s waiting for {what}")


__all__ = [
    "RebotError",
    "UnsupportedError",
    "ArmdClient",
    "ArmSession",
    "ArmStatus",
    "ArmObservation",
    "TcpPose",
    "GripperStatus",
    "ActionStatus",
    "ActionResult",
    "ActionFailed",
    "ObservationSchema",
    "JointLimit",
    "ArmdError",
    "ArmConnectionError",
    "LeaseError",
    "ArmCommandRejected",
    "ServoRejected",
    "ArmFaultError",
    "MotionTimeout",
    "ServoJumpError",
    "UnloadFailed",
    "SafeExitOutcome",
    "GRIPPER_INDEX",
]
