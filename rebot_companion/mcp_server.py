"""stdio MCP server exposing the reBot arm, camera snapshots and detections to an agent.

Launch (needs the ``agent`` extra)::

    uv run --directory <repo> --extra agent rebot-mcp \\
        --armd http://192.168.42.1:8765 --camera-ws ws://192.168.42.1:8001/

Logs go to stderr; stdout is the MCP channel.

Lease rules (rebot-armd binds actuator authority to one WebSocket lease):

* At startup the server tries a non-force lease. If the web cockpit or another client holds it,
  the server stays up without it and actuator tools return ``LEASE_BUSY``. The server never
  sends ``force=true`` on its own.
* While it holds the lease, heartbeats every 150 ms keep it; the cockpit's non-force retry
  cannot take it. If the cockpit preempts (its confirm button), tools return ``LEASE_LOST``
  and control comes back only through ``take_control``.
* By default the agent cannot take control from the cockpit or clear faults: ``take_control``
  only acquires a free lease and ``clear_fault`` is not exposed, so after an E-stop or fault a
  person clears it in the web cockpit. ``--allow-preempt`` adds ``take_control(force=true)``
  and ``--allow-clear-fault`` adds the ``clear_fault`` tool.

Built on the mcp 2.x ``MCPServer`` API (the 1.x ``FastMCP`` class was renamed).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

import anyio
from mcp.server.mcpserver import Image, MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation

from rebot.client import (
    GRIPPER_INDEX,
    ActionFailed,
    ActionResult,
    ArmCommandRejected,
    ArmConnectionError,
    ArmdClient,
    ArmFaultError,
    ArmSession,
    ArmStatus,
    LeaseError,
    MotionTimeout,
)
from rebot.detection import DetectionClient, DetectorUnavailable, parse_detections
from rebot.errors import UnsupportedError

from . import kinematics
from .snapshot import (
    CameraUnavailable,
    SnapshotError,
    SnapshotTimeout,
    VideoGrabber,
    annotated_snapshot,
)

log = logging.getLogger("rebot_mcp")

GRIP_CLOSED_RAD = 0.0  # armd GripperPolicy::rebot_dm closed_rad (cockpit CFG.gripClosed)
GRIP_OPEN_RAD = -5.7  # open_rad (cockpit CFG.gripOpen)
MAX_LINEAR_MPS = 0.15
MAX_STEP_M = 0.30
MAX_ROT_DEG = 90.0

# Tool argument types (module level: annotations are evaluated from module globals).
Speed = Annotated[float | None, Field(gt=0, le=MAX_LINEAR_MPS, description="linear speed, m/s (default 0.05)")]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class KeepoutBox(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    min_m: Vec3
    max_m: Vec3


Step = Annotated[float | None, Field(ge=-MAX_STEP_M, le=MAX_STEP_M)]
Rot = Annotated[float | None, Field(ge=-MAX_ROT_DEG, le=MAX_ROT_DEG)]

HINTS = {
    "LEASE_BUSY": "the web cockpit or another client holds control; ask the user to release it, "
    "or call take_control(force=true) after the user confirms",
    "LEASE_LOST": "control was taken by another client",
    "NO_LEASE": "this server does not hold control; call take_control",
    "PREEMPT_COOLDOWN": "armd refuses another takeover this soon; retry after remaining_ms",
    "ESTOP_ACTIVE": "check the scene with the user, then clear_fault and enable",
    "NOT_ARMED": "call enable",
    "WORKSPACE_VIOLATION": "choose a closer or higher target",
    "IK_UNREACHABLE": "choose a closer or higher target",
    "IK_BRANCH_FLIP": "choose a closer or higher target",
    "MOTION_TIMEOUT": "stop was sent",
    "TRACKING_STALL": "arm may be blocked; check with snapshot",
    "EPOCH_MISMATCH": "the arm was e-stopped or faulted meanwhile; check get_state",
    "BUSY": "another motion tool call is still running",
    "POSE_NOT_FOUND": "call list_poses; teach the pose with save_pose or in the web cockpit",
    "PLACE_TIMEOUT": "the place did not finish; stop was sent; check the scene with snapshot",
    "UNSUPPORTED": "update rebot-armd on the device",
    "COLLISION": "the arm stopped on contact and resumes by itself; check the workspace, call get_state until "
    "collision.active is false, then retry once",
    "COLLISION_HOLD": "the arm is holding after a collision; wait until get_state shows collision.active false, "
    "then retry",
}

POSE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

INSTRUCTIONS_BASE = """Controls a reBot B601-DM arm through rebot-armd, plus the wrist/scene camera.
Coordinates: metres in the arm base frame (x forward, y left, z up); poses are of end_link.
Use snapshot / snapshot_annotated / detect to look before moving. stop and e_stop always work.
Named poses (list_poses, save_pose, move_to_pose) and place need a recent armd; save_pose and
set_keepout change the device's saved configuration, so use them only when the user asks.
recover clears only faults armd marks recoverable; a COLLISION fault is never cleared by recover.
The collision guard (collision_guard_status, set_collision_guard) is off by default and its thresholds
are uncalibrated placeholders. set_collision_guard changes the device's saved configuration, so use it
only when the user asks."""


def build_instructions(allow_clear_fault: bool, allow_preempt: bool) -> str:
    lease = (
        "Control lease: the web cockpit and this server share one control lease. If a tool returns "
        "LEASE_BUSY, ask the user to release control in the cockpit"
        + (
            ", or call take_control(force=true) only after the user explicitly confirms. Never retry force on your own."
            if allow_preempt
            else ", then call take_control. This server cannot take control from the cockpit."
        )
    )
    fault = (
        "After an E-stop or fault, check the scene with the user before clear_fault and enable."
        if allow_clear_fault
        else "After an E-stop or fault, stop and ask the user to check the arm and clear the fault in the "
        "web cockpit; this server cannot clear faults. Then call enable."
    )
    return f"{INSTRUCTIONS_BASE}\n{lease}\n{fault}"


INSTRUCTIONS = build_instructions(False, False)

HUMAN_CLEAR_HINT = (
    "ask the user to check the arm and clear the fault in the web cockpit (this server cannot clear faults), "
    "then call enable"
)
HUMAN_RELEASE_HINT = (
    "the web cockpit or another client holds control; ask the user to release it in the cockpit, then call take_control"
)


def apply_policy(fail: ToolFail, allow_clear_fault: bool, allow_preempt: bool) -> ToolFail:
    """Rewrite hints that point at tools this server does not expose."""
    if not allow_clear_fault and fail.code in ("ESTOP_ACTIVE", "FAULT", "FAULT_ACTIVE"):
        fail.hint = HUMAN_CLEAR_HINT
    if not allow_preempt and fail.code == "LEASE_BUSY":
        fail.hint = HUMAN_RELEASE_HINT
    return fail


class ToolFail(Exception):
    """Anticipated tool failure, returned to the agent as ``is_error`` JSON."""

    def __init__(self, code: str, message: str = "", hint: str | None = None, **extra: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.hint = hint if hint is not None else HINTS.get(code)
        self.extra = extra

    def payload(self, safety_state: str | None) -> dict:
        d: dict = {"code": self.code, "message": self.message}
        if self.hint:
            d["hint"] = self.hint
        d["safety_state"] = safety_state
        d.update(self.extra)
        return d


def map_exception(exc: BaseException) -> ToolFail | None:
    """SDK / snapshot exceptions -> ToolFail; None for unexpected errors."""
    if isinstance(exc, ToolFail):
        return exc
    if isinstance(exc, UnsupportedError):
        return ToolFail("UNSUPPORTED", str(exc))
    if isinstance(exc, ActionFailed):
        hint = None
        if exc.code == "FAULT":
            hint = f"armd fault: {exc.message}; call clear_fault then enable"
        return ToolFail(exc.code, exc.message, hint, action_id=exc.status.id if exc.status else None)
    if isinstance(exc, ArmCommandRejected):
        message = str(exc.raw.get("message") or exc.code)
        hint = None
        if exc.code == "FAULT_ACTIVE":
            hint = "the arm is faulted; check get_state.fault_reason, then clear_fault and enable"
        return ToolFail(exc.code, message, hint)
    if isinstance(exc, MotionTimeout):
        return ToolFail("MOTION_TIMEOUT", str(exc))
    if isinstance(exc, ArmFaultError):
        if exc.status.e_stopped:
            return ToolFail("ESTOP_ACTIVE", str(exc))
        if exc.code == "COLLISION":
            return ToolFail(
                "COLLISION",
                str(exc),
                "repeated collisions latched a fault; ask the user to check the workspace and clear it in the web cockpit",
            )
        return ToolFail("FAULT", str(exc), f"armd fault: {exc.status.fault_reason}; call clear_fault then enable")
    if isinstance(exc, LeaseError):
        if exc.reason == "controller_busy":
            return ToolFail("LEASE_BUSY", str(exc))
        if exc.reason == "preemption_cooldown":
            return ToolFail("PREEMPT_COOLDOWN", str(exc), remaining_ms=exc.raw.get("remaining_ms"))
        return ToolFail("LEASE_LOST", str(exc))
    if isinstance(exc, ArmConnectionError):  # includes HTTP errors (rebot._http.HttpError)
        return ToolFail("ARMD_UNAVAILABLE", str(exc), "check that armd is reachable")
    if isinstance(exc, SnapshotTimeout):
        return ToolFail("SNAPSHOT_TIMEOUT", str(exc))
    if isinstance(exc, CameraUnavailable):
        return ToolFail("CAMERA_UNAVAILABLE", str(exc))
    if isinstance(exc, SnapshotError):
        return ToolFail("SNAPSHOT_FAILED", str(exc))
    if isinstance(exc, DetectorUnavailable):
        return ToolFail("DETECTOR_UNAVAILABLE", str(exc))
    if isinstance(exc, ValueError):
        return ToolFail("INVALID_ARGUMENT", str(exc))
    return None


def _r(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def tcp_dict(positions: list[float]) -> dict | None:
    """Host FK of ``end_link`` (``kinematics.tcp_pose_from_joints``): xyz m, rpy deg, quat [w, x, y, z]."""
    if len(positions) < 6:
        return None
    T = kinematics.tcp_pose_from_joints(positions)
    x, y, z, w = Rotation.from_matrix(T.rotation).as_quat()
    return {
        "xyz_m": [_r(v) for v in T.translation],
        "rpy_deg": [_r(v, 2) for v in T.to_euler_deg()],
        "quat": [_r(w, 5), _r(x, 5), _r(y, 5), _r(z, 5)],
    }


def gripper_fraction(rad: float) -> float:
    return max(0.0, min(1.0, (rad - GRIP_CLOSED_RAD) / (GRIP_OPEN_RAD - GRIP_CLOSED_RAD)))


class Controller:
    """Lease and SDK state behind the MCP tools (synchronous; tools call it from worker threads)."""

    def __init__(
        self,
        client: ArmdClient,
        grabber: VideoGrabber,
        detector: DetectionClient,
        client_id: str = "rebot-mcp",
        auto_lease: bool = True,
    ) -> None:
        self.client = client
        self.grabber = grabber
        self.detector = detector
        self.client_id = client_id
        self.auto_lease = auto_lease
        self.session: ArmSession | None = None
        self.released = False  # release_control was called: no automatic re-acquire
        self._lock = threading.RLock()

    # ---- lease ---------------------------------------------------------------

    def start(self) -> None:
        if not self.auto_lease:
            log.info("auto lease disabled; call take_control to acquire")
            return
        try:
            self.acquire(force=False)
            log.info("control lease acquired")
        except ToolFail as exc:
            log.warning("starting without the control lease: %s %s", exc.code, exc.message)

    def acquire(self, force: bool) -> dict:
        with self._lock:
            s = self.session
            if s is not None and s.has_lease and not s.closed_reason:
                return self.lease_info()
            if s is not None:
                self._close_session(None)
            new = self.client.session(client_id=self.client_id, force=force, safe_exit="error")
            try:
                new.open()
            except (LeaseError, ArmConnectionError) as exc:
                raise map_exception(exc) from exc
            self.session = new
            self.released = False
            return self.lease_info()

    def release(self) -> dict:
        with self._lock:
            self._close_session(None)
            self.released = True
        return {"released": True}

    def _close_session(self, exc_info: tuple | None) -> None:
        s, self.session = self.session, None
        if s is None:
            return
        try:
            if exc_info is not None:
                s.__exit__(*exc_info)  # safe_exit="error": unload on an exception
            else:
                s.close()
        except Exception as err:  # pragma: no cover - best effort on shutdown
            log.warning("closing session: %s", err)

    def shutdown(self, exc: BaseException | None = None) -> None:
        with self._lock:
            self._close_session((type(exc), exc, exc.__traceback__) if exc is not None else None)

    def lease_info(self) -> dict:
        s = self.session
        held = s is not None and s.has_lease and not s.closed_reason
        if held:
            status = "held"
        elif s is not None and s.lease_lost is not None:
            status = "lost"
        elif self.released:
            status = "released"
        else:
            status = "not_held"
        return {"held": held, "generation": s.generation if held else None, "status": status}

    def actuator(self) -> ArmSession:
        """The lease-holding session, or ToolFail(LEASE_BUSY | LEASE_LOST | NO_LEASE)."""
        with self._lock:
            s = self.session
            if s is not None and s.has_lease and not s.closed_reason:
                return s
            if s is not None and s.lease_lost is not None:
                raise ToolFail("LEASE_LOST", f"control lease lost ({s.lease_lost.get('reason', 'lost')})")
            if self.auto_lease and not self.released:
                self.acquire(force=False)  # non-force only; raises LEASE_BUSY if someone holds it
                return self.session  # type: ignore[return-value]
            raise ToolFail("NO_LEASE", "this server does not hold the control lease")

    def observation(self) -> ArmStatus:
        s = self.session
        if s is not None and not s.closed_reason:
            obs = s.latest_observation()
            if obs is not None:
                return obs
        return self.client.get_observation()

    def safety_state(self) -> str | None:
        try:
            s = self.session
            obs = s.latest_observation() if s is not None else None
            return obs.safety_state if obs is not None else self.client.get_observation().safety_state
        except Exception:
            return None

    # ---- tools ---------------------------------------------------------------

    def get_state(self) -> dict:
        obs = self.observation()
        grip = obs.positions[GRIPPER_INDEX] if len(obs.positions) > GRIPPER_INDEX else None
        return {
            "safety_state": obs.safety_state,
            "fault_reason": obs.fault_reason,
            "is_torqued": obs.is_torqued,
            "lease": self.lease_info(),
            "joints": {
                "positions": [_r(v) for v in obs.positions],
                "commanded": [_r(v) for v in obs.commanded_positions],
                "velocities": [_r(v) for v in obs.velocities],
                "torques": [_r(v) for v in obs.torques],
                "temperatures": [_r(v, 1) for v in obs.temperatures],
            },
            "fault": obs.fault,
            "tcp": tcp_dict(obs.positions),
            "gripper": None if grip is None else {"rad": _r(grip), "fraction": _r(gripper_fraction(grip), 3)},
            "action": obs.action.to_dict() if obs.action else None,
            "feedback_valid": obs.feedback_valid,
            "epoch_id": obs.epoch_id,
            "collision": obs.collision,
        }

    @staticmethod
    def _action_dict(res: ActionResult) -> dict:
        a = res.action
        return {
            "action_id": a.id if a else None,
            "kind": a.kind if a else None,
            "state": a.state if a else "done",
            "settled": a.settled if a else None,
            "tcp_after": tcp_dict(res.status.positions),
            "duration_s": _r(res.duration_s, 3),
            "safety_state": res.status.safety_state,
            "is_torqued": res.status.is_torqued,
        }

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        roll_deg: float | None,
        pitch_deg: float | None,
        yaw_deg: float | None,
        quat: list[float] | None,
        speed_mps: float | None,
        timeout_s: float | None,
    ) -> dict:
        s = self.actuator()
        rpy = None
        if quat is not None and any(v is not None for v in (roll_deg, pitch_deg, yaw_deg)):
            raise ToolFail("INVALID_ARGUMENT", "give roll/pitch/yaw or quat, not both")
        if any(v is not None for v in (roll_deg, pitch_deg, yaw_deg)):
            obs = s.latest_observation()
            cur = (tcp_dict(obs.commanded_positions or obs.positions) or {}).get("rpy_deg", [0.0, 0.0, 0.0])
            rpy = [math.radians(v if v is not None else c) for v, c in zip((roll_deg, pitch_deg, yaw_deg), cur)]
        res = s.move_pose([x, y, z], rpy=rpy, quat=quat, linear_speed=speed_mps, timeout=timeout_s)
        return self._action_dict(res)

    def move_relative(self, frame: str, d: list[float], rot_deg: dict, level: bool | None, speed_mps) -> dict:
        s = self.actuator()
        rad = {k: math.radians(v) for k, v in rot_deg.items() if v is not None}
        res = s.move_relative(frame, d, level=level, linear_speed=speed_mps, **rad)
        return self._action_dict(res)

    def gripper(self, command: str, fraction: float | None, grasp: bool | None) -> dict:
        s = self.actuator()
        res = s.gripper(command, fraction=fraction, grasp=grasp)
        obs, a = res.status, res.action
        grip = obs.positions[GRIPPER_INDEX] if len(obs.positions) > GRIPPER_INDEX else 0.0
        return {
            "state": a.state if a else "done",
            "holding": a.holding if a else None,
            "gap_rad": a.gripper_gap_rad if a else None,
            "torque_nm": _r(obs.torques[GRIPPER_INDEX], 3) if len(obs.torques) > GRIPPER_INDEX else None,
            "fraction": _r(gripper_fraction(grip), 3),
        }

    def home(self) -> dict:
        return self._action_dict(self.actuator().home_action())

    def park(self) -> dict:
        return self._action_dict(self.actuator().park_action())

    def stop(self) -> dict:
        s = self.actuator()
        s.stop()
        return {"state": s.wait_for_observation(timeout=1.0).safety_state}

    def stop_quiet(self) -> None:
        s = self.session
        if s is not None and s.has_lease:
            try:
                s.stop()
            except Exception as exc:
                log.warning("stop after cancellation failed: %s", exc)

    def e_stop(self) -> dict:
        s = self.session
        res = s.e_stop() if s is not None and not s.closed_reason else self.client.e_stop()
        return {"safety_state": res.get("safety_state"), "epoch_id": res.get("epoch_id")}

    def enable(self) -> dict:
        obs = self.observation()
        if obs.e_stopped:
            raise ToolFail("ESTOP_ACTIVE", "E-stop is latched")
        if obs.faulted:
            raise ToolFail(
                "FAULT_ACTIVE", f"arm is faulted ({obs.fault_reason})", "call clear_fault, then enable again"
            )
        s = self.actuator()
        ack = s.torque(True)
        return {"safety_state": ack.get("safety_state"), "is_torqued": ack.get("is_torqued")}

    def clear_fault(self) -> dict:
        ack = self.actuator().clear_fault()
        return {"safety_state": ack.get("safety_state"), "epoch_id": ack.get("epoch_id")}

    # ---- poses, place, keep-out, recovery (armd capabilities version 6) -------

    def list_poses(self) -> dict:
        poses = self.client.get_poses()
        return {
            "count": len(poses),
            "poses": [
                {"name": p["name"], "positions": [_r(v) for v in p["positions"]], "tcp": p.get("tcp")} for p in poses
            ],
        }

    def save_pose(self, name: str) -> dict:
        if not POSE_NAME.match(name):
            raise ToolFail("INVALID_ARGUMENT", "name must be 1-64 characters: letters, digits, _ and -")
        s = self.actuator()
        s.save_pose(name)
        return {"saved": name, "positions": [_r(v) for v in self.observation().positions]}

    def move_to_pose(self, name: str, speed_scale: float) -> dict:
        return self._action_dict(self.actuator().move_to_pose(name, speed_scale=speed_scale))

    def place(
        self,
        pose: str | None,
        x: float | None,
        y: float | None,
        z: float | None,
        rpy_deg: list[float] | None,
        approach_height_m: float,
        release: bool,
        retreat: bool,
        speed_scale: float,
        timeout_s: float | None,
    ) -> dict:
        xyz = [x, y, z]
        if pose is not None and any(v is not None for v in xyz):
            raise ToolFail("INVALID_ARGUMENT", "give pose or x, y, z, not both")
        if pose is None and any(v is None for v in xyz):
            raise ToolFail("INVALID_ARGUMENT", "give a pose name, or all of x, y, z")
        s = self.actuator()
        res = s.place(
            pose=pose,
            position=None if pose is not None else xyz,
            rpy=None if rpy_deg is None else [math.radians(v) for v in rpy_deg],
            approach_height_m=approach_height_m,
            release=release,
            retreat=retreat,
            speed_scale=speed_scale,
            timeout=timeout_s,
        )
        return self._action_dict(res)

    def get_keepout(self) -> dict:
        return self.client.get_keepout()

    def set_keepout(self, boxes: list[dict]) -> dict:
        self.actuator().set_keepout(boxes)
        return self.client.get_keepout()

    def recover(self) -> dict:
        """Clear a fault armd marks recoverable and re-enable torque; refuses everything else."""
        s = self.actuator()
        obs = s.wait_for_observation(timeout=2.0)
        if not obs.is_stopped:
            return {"recovered": False, "safety_state": obs.safety_state, "note": "no fault to recover from"}
        human = "ask the user to check the arm and clear the fault in the web cockpit"
        if obs.fault_code == "COLLISION":
            raise ToolFail(
                "NOT_RECOVERABLE",
                "COLLISION: repeated contact stopped the arm and the fault is latched",
                "ask the user to check the workspace for what the arm hit, then clear the fault in the web cockpit",
            )
        if not s.schema().has("fault_codes"):
            raise ToolFail(
                "NOT_RECOVERABLE", f"armd reports no fault codes, so {obs.safety_state} is not cleared here", human
            )
        if not obs.recoverable:
            msg = (obs.fault or {}).get("message") or obs.safety_state
            raise ToolFail("NOT_RECOVERABLE", f"{obs.fault_code or obs.safety_state}: {msg}", human)
        code = obs.fault_code
        s.clear_fault()
        s._wait_until(
            s._frames, lambda o: not o.is_stopped and o.feedback_valid, 3.0, "fresh feedback", check_stopped=False
        )
        ack = s.torque(True)
        return {"recovered": True, "cleared": code, "safety_state": ack.get("safety_state")}

    def describe_safety(self) -> dict:
        d = self.client.get_safety_describe()
        stops = [x.get("id") for x in d.get("stops") or [] if x.get("available", True)]
        faults = d.get("faults") or []
        rec = [f["code"] for f in faults if f.get("recoverable")]
        hw = d.get("hardware") or {}
        spd = (d.get("limits") or {}).get("action_speed_max") or {}
        d["summary"] = (
            f"Stops: {', '.join(stops) or 'none'}. Recoverable faults: {', '.join(rec) or 'none'} "
            f"(the other {len(faults) - len(rec)} need a person). Physical e-stop: "
            f"{'present' if hw.get('physical_estop') else 'ABSENT (software stops only)'}. "
            f"Max action speed: {spd.get('linear_mps')} m/s linear, {spd.get('angular_rps')} rad/s angular."
        )
        return d

    def collision_guard_status(self) -> dict:
        return self.client.get_collision_guard()

    def set_collision_guard(self, enabled: bool | None, sensitivity: str | None) -> dict:
        if enabled is None and sensitivity is None:
            raise ToolFail("INVALID_ARGUMENT", "give enabled and/or sensitivity")
        self.actuator().set_collision_guard(enabled, sensitivity)
        return self.client.get_collision_guard()

    def snapshot(self, max_width: int, quality: int) -> list:
        snap = self.grabber.snapshot(jpeg_quality=quality, max_width=max_width)
        meta = {
            "width": snap.width,
            "height": snap.height,
            "device_ts_ms": snap.device_ts_ms,
            "age_ms": int(time.time() * 1000) - snap.host_rx_ms,
        }
        return [Image(data=snap.jpeg, format="jpeg"), json.dumps(meta)]

    def snapshot_annotated(self, min_score: float, max_width: int, quality: int) -> list:
        snap, meta = annotated_snapshot(
            self.grabber, self.detector, min_score=min_score, jpeg_quality=quality, max_width=max_width
        )
        return [Image(data=snap.jpeg, format="jpeg"), json.dumps(meta)]

    def detect(self, wait_ms: int, min_score: float, pipelines: str | None = None) -> dict:
        wanted = None if pipelines is None else (["yolo", "aruco"] if pipelines == "both" else [pipelines])
        res = self.detector.detect_once(wait_ms=wait_ms, pipelines=wanted)
        dets = [d for d in parse_detections(res["raw"]) if d.score >= min_score]
        out = {
            "frame_id": res["frame_id"],
            "ts_ms": res["ts_ms"],
            "pipelines": res.get("pipelines"),
            "count": len(dets),
            "detections": [
                {k: v for k, v in d.to_dict().items() if k != "corners_norm" or v is not None} for d in dets
            ],
        }
        if res["timeout"]:
            out["timeout"] = True
        return out


# --------------------------------------------------------------------------- MCP wiring


def build_server(ctrl: Controller, allow_clear_fault: bool = False, allow_preempt: bool = False) -> MCPServer:
    server = MCPServer("rebot", instructions=build_instructions(allow_clear_fault, allow_preempt))
    motion_lock = anyio.Lock()

    async def run(fn: Callable[..., Any], *args: Any) -> Any:
        return await anyio.to_thread.run_sync(fn, *args, abandon_on_cancel=True)

    async def motion(fn: Callable[..., Any], *args: Any) -> Any:
        if motion_lock.locked():
            raise ToolFail("BUSY", "another motion tool call is running")
        async with motion_lock:
            try:
                return await run(fn, *args)
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(ctrl.stop_quiet)
                raise

    def tool(description: str, **kw: Any):
        def deco(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    fail = map_exception(exc)
                    if fail is None:
                        raise
                    apply_policy(fail, allow_clear_fault, allow_preempt)
                    state = await anyio.to_thread.run_sync(ctrl.safety_state)
                    text = json.dumps(fail.payload(state))
                    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)

            return server.tool(description=description, structured_output=False, **kw)(wrapper)

        return deco

    @tool(
        "Arm state: safety_state, fault, torque, lease, joints, end_link pose (host FK), gripper, "
        "latest action. No lease needed."
    )
    async def get_state() -> dict:
        return await run(ctrl.get_state)

    @tool("JPEG snapshot from the camera stream, plus JSON {width, height, device_ts_ms, age_ms}.")
    async def snapshot(
        max_width: Annotated[int, Field(ge=64, le=1920)] = 800,
        quality: Annotated[int, Field(ge=10, le=95)] = 80,
    ) -> list:
        return await run(ctrl.snapshot, max_width, quality)

    @tool(
        "JPEG snapshot with the detector's latest boxes drawn, plus JSON {detections, det_frame_id, "
        "det_age_vs_frame_ms, note?}. Boxes come from the latest result and may lag the frame."
    )
    async def snapshot_annotated(
        min_score: Annotated[float, Field(ge=0, le=1)] = 0.3,
        max_width: Annotated[int, Field(ge=64, le=1920)] = 800,
        quality: Annotated[int, Field(ge=10, le=95)] = 80,
    ) -> list:
        return await run(ctrl.snapshot_annotated, min_score, max_width, quality)

    @tool(
        "Run one detector inference on a fresh frame. Boxes are normalised [x0, y0, x1, y1] in the "
        "image (0..1). pipelines: yolo (objects), aruco (markers) or both (default); honoured by "
        "rebot-vision >= 1.3, older detectors run their current mode. The result's pipelines lists "
        "what ran (null when the detector does not report it)."
    )
    async def detect(
        wait_ms: Annotated[int, Field(ge=0, le=5000)] = 2500,
        min_score: Annotated[float, Field(ge=0, le=1)] = 0.3,
        pipelines: Literal["yolo", "aruco", "both"] | None = None,
    ) -> dict:
        return await run(ctrl.detect, wait_ms, min_score, pipelines)

    @tool(
        "Straight-line move of end_link to (x, y, z) m in the base frame. Orientation: roll/pitch/yaw "
        "in degrees (URDF fixed-axis; missing angles keep the current value) or quat [w, x, y, z]; "
        "omit both to keep the current orientation. Blocks until done."
    )
    async def move_to(
        x: float,
        y: float,
        z: float,
        roll_deg: float | None = None,
        pitch_deg: float | None = None,
        yaw_deg: float | None = None,
        quat: Annotated[list[float] | None, Field(min_length=4, max_length=4)] = None,
        speed_mps: Speed = None,
        timeout_s: Annotated[float | None, Field(gt=0, le=60)] = None,
    ) -> dict:
        return await motion(ctrl.move_to, x, y, z, roll_deg, pitch_deg, yaw_deg, quat, speed_mps, timeout_s)

    @tool(
        "Relative move of end_link. frame: base (base axes), tool (end_link axes) or camera (x right, "
        "y down, z forward along the view; dpitch > 0 looks down, dyaw > 0 looks right; level keeps "
        "the image level, default true). Translations in m (each <= 0.30), rotations in degrees "
        "(each <= 90), centred on end_link. Blocks until done."
    )
    async def move_relative(
        frame: Literal["base", "tool", "camera"],
        dx: Step = None,
        dy: Step = None,
        dz: Step = None,
        dyaw_deg: Rot = None,
        dpitch_deg: Rot = None,
        droll_deg: Rot = None,
        level: bool | None = None,
        speed_mps: Speed = None,
    ) -> dict:
        d = [dx or 0.0, dy or 0.0, dz or 0.0]
        rot = {"dyaw": dyaw_deg, "dpitch": dpitch_deg, "droll": droll_deg}
        return await motion(ctrl.move_relative, frame, d, rot, level, speed_mps)

    @tool(
        "Gripper: open, close (grasp=true by default: stops on contact and reports holding) or move "
        "to fraction (0 closed .. 1 open). Returns {holding, gap_rad, torque_nm, fraction, state}."
    )
    async def gripper(
        command: Literal["open", "close", "move"],
        fraction: Annotated[float | None, Field(ge=0, le=1)] = None,
        grasp: bool | None = None,
    ) -> dict:
        return await motion(ctrl.gripper, command, fraction, grasp)

    @tool("Move to the home pose. Blocks until done.")
    async def home() -> dict:
        return await motion(ctrl.home)

    @tool("Park at the resting pose and turn torque off (ends Idle). Blocks until done.")
    async def park() -> dict:
        return await motion(ctrl.park)

    @tool("Stop the current motion and hold position (torque stays on). Works while another tool runs.")
    async def stop() -> dict:
        return await run(ctrl.stop)

    @tool(
        "Emergency stop. Needs no lease; latches EStop until "
        + ("clear_fault." if allow_clear_fault else "a person clears it in the web cockpit.")
    )
    async def e_stop() -> dict:
        return await run(ctrl.e_stop)

    @tool("Turn torque on (arm holds its pose). Rejected while E-stopped or faulted.")
    async def enable() -> dict:
        return await motion(ctrl.enable)

    if allow_clear_fault:

        @tool("Clear EStop / Fault back to Idle; call enable before moving.")
        async def clear_fault() -> dict:
            return await motion(ctrl.clear_fault)

    if allow_preempt:

        @tool(
            "Acquire the control lease. force=true takes control from the web cockpit; only call after "
            "the user explicitly confirms."
        )
        async def take_control(force: bool = False) -> dict:
            return await run(ctrl.acquire, force)

    else:

        @tool("Acquire the control lease when no other client holds it. Cannot take control from the web cockpit.")
        async def take_control() -> dict:
            return await run(ctrl.acquire, False)

    # -- named poses, place, keep-out, recovery. Each needs an armd with the matching capability and
    # returns UNSUPPORTED (with an update hint) otherwise.

    @tool("Saved poses: {count, poses[{name, positions, tcp}]}. No lease needed. Needs armd capability 'poses'.")
    async def list_poses() -> dict:
        return await run(ctrl.list_poses)

    @tool(
        "CONFIGURATION CHANGE: save the arm's current joint positions under a name (letters, digits, _ and -, "
        "max 64), replacing a pose with that name. Move the arm there first (the user can hand-guide it "
        "or use the cockpit). Persisted on the device. Needs capability 'poses'."
    )
    async def save_pose(name: str) -> dict:
        return await run(ctrl.save_pose, name)

    @tool("Joint-space move to a saved pose; speed_scale 0.1..1.0 of the planned speed. Blocks until done.")
    async def move_to_pose(name: str, speed_scale: Annotated[float, Field(ge=0.1, le=1.0)] = 0.5) -> dict:
        return await motion(ctrl.move_to_pose, name, speed_scale)

    @tool(
        "Put the held object down: lift, move above the target, descend, open the gripper, retreat. Target: "
        "a saved pose name (its gripper position is the place point) or x, y, z in m (base frame). "
        "rpy_deg optionally sets the orientation [roll, pitch, yaw]. release=false keeps the object, "
        "retreat=false stays at the target. Fails with IK_UNREACHABLE / WORKSPACE_VIOLATION (also for "
        "keep-out boxes) / TRACKING_STALL / PLACE_TIMEOUT. Needs capability 'place'. Blocks until done."
    )
    async def place(
        pose: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        rpy_deg: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = None,
        approach_height_m: Annotated[float, Field(ge=0.02, le=0.30)] = 0.08,
        release: bool = True,
        retreat: bool = True,
        speed_scale: Annotated[float, Field(ge=0.1, le=1.0)] = 0.5,
        timeout_s: Annotated[float | None, Field(gt=0, le=120)] = None,
    ) -> dict:
        return await motion(
            ctrl.place, pose, x, y, z, rpy_deg, approach_height_m, release, retreat, speed_scale, timeout_s
        )

    @tool(
        "Keep-out boxes the arm must not enter: {boxes[{name, min_m, max_m}], source}. "
        "No lease needed. Needs capability 'keepout'."
    )
    async def get_keepout() -> dict:
        return await run(ctrl.get_keepout)

    @tool(
        "CONFIGURATION CHANGE: replace the runtime keep-out boxes (max 16; min_m < max_m per axis, metres, base "
        "frame). Motions into a box fail with WORKSPACE_VIOLATION. Pass [] to remove all. Persisted on the "
        "device. Show the user the boxes before setting them. Needs capability 'keepout'."
    )
    async def set_keepout(boxes: Annotated[list[KeepoutBox], Field(max_length=16)]) -> dict:
        return await motion(ctrl.set_keepout, [b.model_dump() for b in boxes])

    @tool(
        "Clear a fault that armd marks recoverable (WATCHDOG_TIMEOUT, CONTROL_DEADLINE, HANDOVER_UNSETTLED) "
        "and turn torque on again. Any other fault or an E-stop returns NOT_RECOVERABLE and stays latched "
        "for a person to clear in the web cockpit. Needs capability 'fault_codes'."
    )
    async def recover() -> dict:
        return await motion(ctrl.recover)

    @tool(
        "What the arm's safety system does and does not do: stops, faults (with recoverable flags), speed and "
        "joint limits, timing, and hardware facts such as whether a physical e-stop exists. Plus a one-line "
        "'summary'. Read it before planning motions. No lease needed. Needs capability 'safety_describe'."
    )
    async def describe_safety() -> dict:
        return await run(ctrl.describe_safety)

    @tool(
        "Collision guard configuration: {enabled, sensitivity, thresholds_nm}. get_state.collision shows a "
        "trip in progress (active, joint, residual_nm, resume_in_s). The guard is off by default and its "
        "thresholds are uncalibrated placeholders. No lease needed. Needs capability 'collision_guard'."
    )
    async def collision_guard_status() -> dict:
        return await run(ctrl.collision_guard_status)

    @tool(
        "CONFIGURATION CHANGE: switch the collision guard on or off and/or set its sensitivity (low, medium, "
        "high). Persisted on the device. With the guard on, contact stops the arm; it resumes by itself when "
        "the contact is gone and latches a COLLISION fault after repeated trips. Thresholds are uncalibrated "
        "placeholders. Use only when the user asks. Needs capability 'collision_guard'."
    )
    async def set_collision_guard(
        enabled: bool | None = None, sensitivity: Literal["low", "medium", "high"] | None = None
    ) -> dict:
        return await motion(ctrl.set_collision_guard, enabled, sensitivity)

    @tool("Release the control lease (the web cockpit will acquire it automatically).")
    async def release_control() -> dict:
        return await run(ctrl.release)

    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rebot-mcp", description=__doc__.split("\n\n")[0])
    p.add_argument("--armd", default="http://192.168.42.1:8765", help="rebot-armd base URL")
    p.add_argument("--camera-ws", default="ws://192.168.42.1:8001/", help="camera debug stream WebSocket")
    p.add_argument("--client-id", default="rebot-mcp", help="lease label shown to other clients")
    p.add_argument("--no-auto-lease", action="store_true", help="do not acquire the lease until take_control")
    p.add_argument("--decoder", default="auto", choices=("auto", "pyav", "ffmpeg"))
    p.add_argument(
        "--allow-clear-fault",
        action="store_true",
        help="expose clear_fault to the agent (default: a person clears faults in the web cockpit)",
    )
    p.add_argument(
        "--allow-preempt",
        action="store_true",
        help="let take_control(force=true) take control from the web cockpit (default: not exposed)",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s")

    ctrl = Controller(
        ArmdClient(args.armd),
        VideoGrabber(args.camera_ws, backend=args.decoder),
        DetectionClient(args.armd),
        client_id=args.client_id,
        auto_lease=not args.no_auto_lease,
    )
    ctrl.start()
    server = build_server(ctrl, allow_clear_fault=args.allow_clear_fault, allow_preempt=args.allow_preempt)
    try:
        server.run("stdio")
    except BaseException as exc:
        ctrl.shutdown(exc)
        raise
    ctrl.shutdown()


if __name__ == "__main__":
    main()
