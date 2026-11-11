"""In-process fake rebot-armd enforcing armd's WS control protocol (shared by the SDK and MCP tests).

The fake mirrors rebot-armd ``src/api/ws.rs`` and ``src/supervisor.rs``:
lease required for every actuator command, generation + epoch checks,
strictly increasing request_id / sequence_id, heartbeat-driven lease expiry,
watchdog Fault(WatchdogTimeout) while torqued without heartbeats, 200 ms servo
frame timeout, and lease-free e-stop over WS and REST.

Safety rules modelled because the SDK depends on them (armd source in brackets):

* 7-DOF B601-DM layout: home J6 = pi/2, gripper J7 limit -5.75..0.10 [model.rs:215-218];
  any target of another length is rejected [safety.rs:232].
* Servo frames jumping > 0.35 rad from the commanded position (gripper exempt) are dropped
  silently, without a reply [supervisor.rs:1450-1471].
* torque(False) away from the resting pose is refused (ARM_NOT_AT_RESTING_POSE)
  [supervisor.rs:462, model.rs RestingPosePolicy::rebot_dm].
* unload without a resting-pose profile answers UNSUPPORTED [api/ws.rs:68]; releasing the
  lease or stop cancels a pending unload [supervisor.rs:229 stop_motion].
* A failed post-unload torque cut leaves the arm Armed and torqued [supervisor.rs:1430].

High-level actions follow the contract in rebot-armd
``docs/dev_flow_spec_agent_api_foundation.md`` §2.1 / §2.3 (not yet checked against armd):

* ``/api/v1/action/{move_to,move_relative,gripper}`` acks carry ``action_id`` and
  ``expected_duration_s``; jog / home / unload also create records (``joint_move`` / ``home`` /
  ``park``). The observation carries ``action`` (latest record) and ``commanded_positions``.
* Preconditions: operational, no trajectory or servo stream -> else ``MOTION_BUSY`` 409.
  ``gripper`` with ``width_m`` -> ``UNSUPPORTED`` 422.
* Reconciliation per tick: EStop -> failed ESTOP_ACTIVE, Fault -> failed FAULT (WATCHDOG for the
  watchdog), lease released / preempted -> cancelled LEASE_LOST, stop -> cancelled STOPPED, a new
  jog / preset / action -> cancelled SUPERSEDED. Park: done when Idle with torque off, else failed
  UNLOAD_FAILED.
* Developer platform (armd capabilities version 6, ``devplat = True``): ``?obs_hz=`` decimation,
  ``fault{code,message,recoverable}``, ``previous_action``, named poses + ``move_to_pose``.
  ``/api/v1/action/grasp`` closes the gripper on ``gripper_contact`` and reports ``holding``.
  The detector proxy answers ``trigger`` with ``detections`` (perception API 2).
* Scripted knobs: ``action_duration``, ``action_failure = (code, message)`` (fails at half the
  path), ``action_reject = (status, code)`` (rejects the command), ``gripper_contact`` (jaw angle
  where an object stops a close, None = empty), ``actions_supported = False`` (pre-action armd).
"""

from __future__ import annotations

import json
import pathlib
import threading
import time

import pytest
from http_mock import MockTransport, Request, Response
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from rebot import Arm
from rebot.client import ArmdClient

DOF = 7
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963, 0.0]
LIMITS = [(-3.14, 3.14)] * 6 + [(-5.75, 0.10)]
REST_TOL = [0.15, 0.08, 0.08, 0.08, 0.08, 0.08]  # RestingPosePolicy::rebot_dm (gripper excluded)
SERVO_JUMP = 0.35
TICK = 0.02  # 50 Hz like armd's default control_frequency_hz
GRIP_OPEN = -5.7  # GripperPolicy::rebot_dm open_rad
GRIP_CLOSED = 0.0


def q(*first: float) -> list[float]:
    """HOME with the leading joints replaced, e.g. ``q(0.1)`` nudges only J1."""
    out = list(HOME)
    out[: len(first)] = [float(x) for x in first]
    return out


class FakeArmd:
    def __init__(self, lease_timeout: float = 0.6) -> None:
        self.lock = threading.RLock()
        self.lease_timeout = lease_timeout
        self.frame_timeout = 0.2
        self.state = "Idle"
        self.fault: str | None = None
        self.epoch = 1
        self.is_torqued = False
        self.pending_rearm = False
        self.positions = list(HOME)
        self.home = list(HOME)
        self.resting_pose = True  # False: no resting-pose profile -> unload UNSUPPORTED
        self.fail_torque_cut = False  # True: post-unload torque cut fails (arm stays Armed, torqued)
        self.dropped_servo_frames = 0
        self.traj: tuple[list[float], list[float], float, float] | None = None
        self.servo_target: list[float] | None = None
        self.last_motion_frame = time.monotonic()
        self.last_heartbeat = time.monotonic()
        self.lease: dict | None = None  # {conn, generation, acquired_at}
        self.gen_counter = 0
        self.freeze_motion = False
        self.commands: list[str] = []
        self.heartbeats = 0
        self.conns: dict[int, tuple] = {}
        # high-level actions
        self.actions_supported = True
        self.action_duration = 0.3
        self.action_failure: tuple[str, str] | None = None
        self.action_reject: tuple[int, str] | None = None
        self.gripper_contact: float | None = -2.0
        self.action: dict | None = None
        self.action_gen: int | None = None
        self.traj_action_id: int | None = None
        self.next_action_id = 1
        self.action_bodies: list[tuple[str, dict]] = []
        self.ack_extra: dict = {}
        self.cancel_reason = "STOPPED"
        self.acquires: list[tuple[str, bool]] = []  # (client_id, force) per acquire_lease
        # developer platform (capabilities version 6)
        self.devplat = False
        self.place_cap = True
        self.keepout: list[dict] = []
        self.poses: dict[str, dict] = {}
        self.prev_action: dict | None = None
        self.prev_action_at = 0.0
        self.conn_hz: dict[int, tuple[int, float]] = {}  # conn -> (obs_hz, last sent)
        self.ws_paths: list[str] = []
        self.detections: list = []
        # collision guard (capability collision_guard, armd contract §11)
        self.collision_cap = True
        self.describe_cap = True  # capability safety_describe (armd contract §12)
        self.collision_cfg = {"enabled": False, "sensitivity": "medium"}
        self.collision_state: dict | None = None  # active trip: joint, residual_nm, since, resume_at
        self.collision_trips: list[float] = []
        self.collision_max_auto_resumes = 3
        self.collision_bodies: list[dict] = []
        self.collision_next_actions = 0  # trip halfway through each of the next N actions (resume after 0.2 s)
        self.trigger_bodies: list = []
        self.ack_delays: dict[str, float] = {}  # command path -> seconds the ack is held (persisted commands)
        self._stop = threading.Event()
        self._next_conn = 1

    # ---- simulation --------------------------------------------------------
    def observation(self) -> dict:
        state = {"Fault": self.fault} if self.state == "Fault" else self.state
        return {
            "timestamp": time.monotonic(),
            "positions": list(self.positions),
            "velocities": [0.0] * DOF,
            "torques": [0.0] * DOF,
            "temperatures": [25.0] * DOF,
            "is_torqued": self.is_torqued,
            "safety_state": state,
            "epoch_id": self.epoch,
            "gravity_compensated": False,
            "gravity_torques": [],
            "driver_capabilities": {},
            "enable_state": "Enabled" if self.is_torqued else "Disabled",
            "feedback_age_ms": [0] * DOF,
            "disable_pending": False,
            "feedback_valid": True,
            "commanded_positions": list(self.positions),
            "action": self._public(self.action),
            **(self.devplat_fields() if self.devplat else {}),
        }

    @staticmethod
    def _public(action: dict | None) -> dict | None:
        return {k: v for k, v in action.items() if not k.startswith("_")} if action else None

    def devplat_fields(self) -> dict:
        fault = None
        if self.state == "EStop":
            fault = {"code": "ESTOP", "message": "E-stop latched", "recoverable": False}
        elif self.state == "Fault":
            code = {
                "WatchdogTimeout": "WATCHDOG_TIMEOUT",
                "FollowingError": "FOLLOWING_ERROR",
                "Collision": "COLLISION",
            }.get(self.fault, "OTHER")
            fault = {"code": code, "message": str(self.fault), "recoverable": code == "WATCHDOG_TIMEOUT"}
        prev = self.prev_action if time.monotonic() - self.prev_action_at <= 2.0 else None
        return {
            "fault": fault,
            "previous_action": self._public(prev),
            "residual_torques": [0.0] * 6,
            "collision": self.collision_field(),
        }

    def collision_field(self) -> dict | None:
        if not (self.collision_cap and self.collision_cfg["enabled"]):
            return None
        trip = self.collision_state
        now = time.monotonic()
        return {
            "active": trip is not None,
            "joint": trip["joint"] if trip else None,
            "residual_nm": trip["residual_nm"] if trip else None,
            "threshold_nm": 6.0 if trip else None,
            "since_s": round(now - trip["since"], 2) if trip else None,
            "resume_in_s": round(max(0.0, trip["resume_at"] - now), 2) if trip else None,
            "trips_in_window": len(self.collision_trips),
            "enabled": True,
            "sensitivity": self.collision_cfg["sensitivity"],
        }

    def trip_collision(self, joint: int = 2, residual_nm: float = 7.1, resume_after_s: float = 0.3) -> None:
        """Test hook: a protective stop. Cancels the running action (code COLLISION), holds, resumes by itself
        after ``resume_after_s``; the (max_auto_resumes+1)-th trip escalates to a Fault with code COLLISION."""
        with self.lock:
            now = time.monotonic()
            self.collision_trips = [t for t in self.collision_trips if now - t < 60.0] + [now]
            if self.action_running():
                self.finish_action("failed", "COLLISION", f"joint J{joint} blocked ({residual_nm:.1f} N·m)")
            self.stop_motion()
            if len(self.collision_trips) > self.collision_max_auto_resumes:
                self.collision_state = None
                self.trigger_fault("Collision")
                return
            if self.state == "Moving":
                self.state = "Armed"
            self.collision_state = {
                "joint": joint,
                "residual_nm": residual_nm,
                "since": now,
                "resume_at": now + resume_after_s,
            }

    def capabilities(self) -> dict:
        caps = {"version": 5, "cartesian": True, "gripper": True, "grasp": True}
        if self.devplat:
            caps.update(version=6, obs_rate=True, fault_codes=True, poses=True, keepout=True)
            caps.update(place=self.place_cap, collision_guard=self.collision_cap)
            caps.update(safety_describe=self.describe_cap)
            caps.update(speed_cap=False, residual_torques=True, tick_stats=False)
        return caps

    def trigger_fault(self, reason: str) -> None:
        self.state, self.fault = "Fault", reason
        self.epoch += 1
        self.stop_motion()

    def tick(self) -> None:
        with self.lock:
            now = time.monotonic()
            if (
                self.is_torqued
                and self.state in ("Armed", "Moving")
                and not self.pending_unload
                and now - self.last_heartbeat > self.lease_timeout
            ):
                self.trigger_fault("WatchdogTimeout")
            if self.collision_state and now >= self.collision_state["resume_at"]:
                self.collision_state = None
            if self.traj and not self.freeze_motion:
                start, end, t0, dur = self.traj
                a = min(1.0, (now - t0) / dur)
                self.positions = [s + (e - s) * a for s, e in zip(start, end)]
                running = self.action_running() and self.traj_action_id == self.action["id"]
                if running and self.action_failure and a >= 0.5 and self.action["kind"] != "park":
                    code, message = self.action_failure
                    self.finish_action("failed", code, message)
                    self.stop_motion()
                    self.state = "Armed"
                elif running and self.collision_next_actions > 0 and a >= 0.5 and self.action["kind"] != "park":
                    self.collision_next_actions -= 1
                    self.trip_collision(resume_after_s=0.2)
                elif a >= 1.0:
                    self.traj = None
                    if self.state == "Moving":
                        self.state = "Armed"
                    if self.pending_unload:
                        self.pending_unload = False
                        if self.fail_torque_cut or self.resting_violation():
                            self.state = "Armed"  # set_torque(false) failed -> mark_motion_finished
                        else:
                            self.is_torqued = False
                            self.state = "Idle"
                    if running:
                        self.complete_action()
                elif running:
                    self.action["progress"] = round(a, 3)
            elif self.servo_target is not None:
                if now - self.last_motion_frame > self.frame_timeout:
                    self.servo_target = None
                else:
                    step = 2.0 * TICK
                    self.positions = [
                        p + max(-step, min(step, t - p)) for p, t in zip(self.positions, self.servo_target)
                    ]
            self.reconcile_action()
            text = json.dumps(self.observation())
        for conn_id in list(self.conns):
            hz, last = self.conn_hz.get(conn_id, (0, 0.0))
            if hz:
                if now - last < 1.0 / hz:
                    continue
                self.conn_hz[conn_id] = (hz, now)
            self.send(conn_id, text)

    def resting_violation(self) -> bool:
        return any(abs(p - h) > t for p, h, t in zip(self.positions, self.home, REST_TOL))

    # ---- actions -------------------------------------------------------------
    def action_running(self) -> bool:
        return self.action is not None and self.action["state"] == "running"

    def finish_action(self, state: str, code: str | None = None, message: str | None = None, **extra) -> None:
        self.action.update(state=state, code=code, message=message, finished_at=time.monotonic(), **extra)
        if state == "done":
            self.action["progress"] = 1.0
        self.traj_action_id = None

    def complete_action(self) -> None:
        kind = self.action["kind"]
        if kind == "park":
            if self.state == "Idle" and not self.is_torqued:
                self.finish_action("done")
            else:
                self.finish_action("failed", "UNLOAD_FAILED", "torque cut failed after unload")
        elif kind in ("gripper", "grasp") and self.action.get("_grasp"):
            contact = self.gripper_contact
            gap = GRIP_CLOSED - self.positions[6]
            holding = contact is not None and gap >= 0.30
            self.finish_action("done", holding=holding, gripper_gap_rad=round(gap, 4), settled=True)
        else:
            self.finish_action("done", settled=True)
        self.action.pop("_grasp", None)

    def reconcile_action(self) -> None:
        """armd §2.3: one place that ends a running action whose trajectory is gone."""
        if not self.action_running() or self.traj_action_id == self.action["id"] and self.traj is not None:
            return
        if self.state == "EStop":
            self.finish_action("failed", "ESTOP_ACTIVE", "EStop")
        elif self.state == "Fault":
            if self.fault == "WatchdogTimeout":
                self.finish_action("failed", "WATCHDOG", "watchdog park")
            else:
                self.finish_action("failed", "FAULT", str(self.fault))
        elif self.lease is None or self.lease["generation"] != self.action_gen:
            self.finish_action("cancelled", "LEASE_LOST", "lease released or taken over")
        else:
            self.finish_action("cancelled", self.cancel_reason, "")
        self.action.pop("_grasp", None)

    def start_action(self, kind: str, target: list[float], duration: float, tag=None, grasp=False) -> None:
        if self.action_running():
            self.finish_action("cancelled", "SUPERSEDED", "replaced by a new motion")
        if self.action is not None:
            self.prev_action, self.prev_action_at = dict(self.action), time.monotonic()
        aid = self.next_action_id
        self.next_action_id += 1
        self.action = {
            "id": aid,
            "kind": kind,
            "tag": tag,
            "state": "running",
            "code": None,
            "message": None,
            "progress": 0.0,
            "started_at": time.monotonic(),
            "finished_at": None,
            "holding": None,
            "gripper_gap_rad": None,
            "settled": None,
            "_grasp": grasp,
        }
        self.action_gen = self.lease["generation"] if self.lease else None
        self.start_traj(target, duration)
        self.traj_action_id = aid
        self.ack_extra = {"action_id": aid, "expected_duration_s": float(duration)}

    def execute_action(self, path: str, body: dict):
        if not self.actions_supported:
            return 400, "INVALID_COMMAND"
        err = self.motion_gate(self.positions)
        if err is not None:
            return err
        if self.traj is not None or self.servo_target is not None:
            return 409, "MOTION_BUSY"
        self.action_bodies.append((path, body))
        tag = body.get("tag")
        kind = path.rsplit("/", 1)[-1]
        if kind == "move_to":
            pos = body.get("position")
            if not isinstance(pos, list) or len(pos) != 3:
                return 400, "INVALID_COMMAND"
        elif kind == "move_relative":
            if body.get("frame") not in ("base", "tool", "camera") or len(body.get("d") or []) != 3:
                return 400, "INVALID_COMMAND"
        elif kind == "move_to_pose":
            if not self.devplat:
                return 400, "INVALID_COMMAND"
            pose = self.poses.get(body.get("name"))
            if pose is None:
                return 404, "POSE_NOT_FOUND"
        elif kind == "place":
            if not (self.devplat and self.place_cap):
                return 400, "INVALID_COMMAND"
            if (body.get("pose") is None) == (body.get("position") is None):
                return 400, "INVALID_COMMAND"
            if body.get("pose") is not None and body["pose"] not in self.poses:
                return 404, "POSE_NOT_FOUND"
        elif kind == "grasp":
            if not isinstance(body.get("label"), str) or len(body.get("box_norm") or []) != 4:
                return 400, "INVALID_COMMAND"
        elif kind == "gripper":
            if "width_m" in body:
                return 422, "UNSUPPORTED"
            cmd = body.get("command")
            if cmd not in ("open", "close", "move"):
                return 400, "INVALID_COMMAND"
        else:
            return 400, "INVALID_COMMAND"
        if self.action_reject is not None:
            return self.action_reject
        target = list(self.positions)
        grasp = False
        if kind == "move_to_pose":
            target = list(pose["positions"])
            if not body.get("include_gripper"):
                target[6] = self.positions[6]
        elif kind == "place":
            target[6] = GRIP_OPEN if body.get("release", True) else self.positions[6]
        elif kind == "grasp":
            grasp = True
            contact = self.gripper_contact
            target[6] = contact + 0.15 if contact is not None else -0.05
        elif kind == "gripper":
            if cmd == "open":
                target[6] = GRIP_OPEN
            elif cmd == "move":
                target[6] = GRIP_CLOSED + float(body.get("fraction", 0.0)) * (GRIP_OPEN - GRIP_CLOSED)
            else:
                grasp = body.get("grasp", True)
                contact = self.gripper_contact if grasp else None
                target[6] = contact + 0.15 if contact is not None else -0.05  # squeeze past contact
        self.start_action(kind, target, self.action_duration, tag, grasp=grasp)
        return None

    def stop_motion(self) -> None:
        self.traj = None
        self.servo_target = None
        self.pending_unload = False

    def run_ticker(self) -> None:
        while not self._stop.wait(TICK):
            self.tick()

    pending_unload = False

    # ---- transport ---------------------------------------------------------
    def send(self, conn_id: int, text: str) -> None:
        entry = self.conns.get(conn_id)
        if entry is None:
            return
        ws, lock = entry
        try:
            with lock:
                ws.send(text)
        except Exception:
            self.conns.pop(conn_id, None)

    def lease_valid(self, conn: int, generation: int) -> bool:
        lease = self.lease
        return (
            lease is not None
            and lease["conn"] == conn
            and lease["generation"] == generation
            and time.monotonic() - lease["acquired_at"] < self.lease_timeout
        )

    def handler(self, ws) -> None:
        with self.lock:
            conn = self._next_conn
            self._next_conn += 1
            self.conns[conn] = (ws, threading.Lock())
            path = ws.request.path
            self.ws_paths.append(path)
            if "obs_hz=" in path:
                self.conn_hz[conn] = (int(path.split("obs_hz=")[1].split("&")[0]), 0.0)
        local_gen = 0
        label: str | None = None
        last_seq: int | None = None
        last_req: int | None = None
        try:
            for text in ws:
                msg = json.loads(text)
                kind = msg.get("type", "")
                with self.lock:
                    reply, local_gen, label, last_seq, last_req = self.handle(
                        conn, msg, kind, local_gen, label, last_seq, last_req
                    )
                delay = self.ack_delays.get(msg.get("path", "")) if kind == "command" else None
                if reply is not None and delay:
                    # armd acks persisted commands after the disk write: the ack (with the epoch at
                    # processing time) can arrive after newer telemetry and after later requests' acks
                    threading.Timer(delay, self.send, (conn, json.dumps(reply))).start()
                elif reply is not None:
                    self.send(conn, json.dumps(reply))
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.conns.pop(conn, None)
                if local_gen and self.lease and self.lease["conn"] == conn and self.lease["generation"] == local_gen:
                    self.lease = None
                    self.stop_motion()

    def handle(self, conn, msg, kind, local_gen, label, last_seq, last_req):
        if kind in ("estop", "e_stop"):
            self.do_estop()
            return (
                {"type": "epoch_sync", "epoch_id": self.epoch, "safety_state": self.state},
                local_gen,
                label,
                last_seq,
                last_req,
            )
        if kind == "acquire_lease":
            client_id = msg.get("client_id") or ""
            self.acquires.append((client_id, bool(msg.get("force"))))
            seq = msg.get("seq")
            if not client_id or len(client_id) > 128:
                return (
                    {"type": "lease_rejected", "error": "invalid_client_id", "seq": seq},
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            lease = self.lease
            active = lease is not None and time.monotonic() - lease["acquired_at"] < self.lease_timeout
            if active and lease["conn"] != conn and not msg.get("force"):
                return (
                    {"type": "lease_rejected", "error": "controller_busy", "seq": seq},
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            evicted = lease["conn"] if active and lease["conn"] != conn else None
            if not (active and lease["conn"] == conn):
                self.gen_counter += 1
                self.lease = {"conn": conn, "generation": self.gen_counter, "acquired_at": time.monotonic()}
                self.stop_motion()
            if evicted is not None:
                self.send(
                    evicted,
                    json.dumps({"type": "lease_lost", "generation": lease["generation"], "reason": "preempted"}),
                )
            gen = self.lease["generation"]
            return (
                {"type": "lease_granted", "client_id": client_id, "generation": gen, "seq": seq},
                gen,
                client_id,
                None,
                last_req,
            )

        generation = msg.get("generation") or 0
        label_ok = "client_id" not in msg or msg["client_id"] == label
        authorized = (
            local_gen != 0
            and (generation == local_gen or (kind == "heartbeat" and generation == 0))
            and label_ok
            and self.lease_valid(conn, local_gen)
        )
        if kind == "servo" and msg.get("generation") is None:
            return {"type": "servo_reject", "reason": "missing_generation"}, local_gen, label, last_seq, last_req
        if not authorized:
            if kind == "command":
                return (
                    {
                        "type": "command_reject",
                        "request_id": msg.get("request_id"),
                        "status": 409,
                        "code": "LEASE_LOST",
                        "error": "LEASE_LOST",
                        "message": "This connection does not own an active lease",
                        "epoch_id": self.epoch,
                    },
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            if kind == "release_lease":
                return {"type": "lease_released", "success": False}, local_gen, label, last_seq, last_req
            return {"type": "lease_lost", "generation": local_gen}, local_gen, label, last_seq, last_req
        if kind == "release_lease":
            self.lease = None
            self.stop_motion()
            return {"type": "lease_released", "success": True}, 0, None, last_seq, last_req
        if msg.get("epoch_id") != self.epoch:
            if kind == "command":
                return (
                    {
                        "type": "command_reject",
                        "request_id": msg.get("request_id"),
                        "status": 409,
                        "code": "EPOCH_MISMATCH",
                        "error": "EPOCH_MISMATCH",
                        "message": "Command epoch is no longer current",
                        "epoch_id": self.epoch,
                    },
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            return (
                {"type": "epoch_sync", "epoch_id": self.epoch, "safety_state": self.state},
                local_gen,
                label,
                last_seq,
                last_req,
            )
        if kind == "heartbeat":
            self.heartbeats += 1
            self.lease["acquired_at"] = time.monotonic()
            self.last_heartbeat = time.monotonic()
            return None, local_gen, label, last_seq, last_req
        if kind == "command":
            rid = msg.get("request_id")
            if not isinstance(rid, int) or (last_req is not None and rid <= last_req):
                return (
                    {
                        "type": "command_reject",
                        "request_id": rid,
                        "status": 400,
                        "code": "INVALID_COMMAND",
                        "error": "INVALID_COMMAND",
                        "message": "request_id must be present and strictly increase",
                    },
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            last_req = rid
            path = msg.get("path", "")
            self.commands.append(path)
            self.ack_extra = {}
            err = self.execute(path, msg.get("body") or {})
            if err is None:
                return (
                    {
                        "type": "command_ack",
                        "request_id": rid,
                        "status": "accepted",
                        "epoch_id": self.epoch,
                        "safety_state": self.state,
                        "is_torqued": self.is_torqued,
                        **self.ack_extra,
                    },
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            return (
                {
                    "type": "command_reject",
                    "request_id": rid,
                    "status": err[0],
                    "code": err[1],
                    "error": err[1],
                    "message": err[1],
                    "epoch_id": self.epoch,
                },
                local_gen,
                label,
                last_seq,
                last_req,
            )
        if kind == "servo":
            seq = msg.get("sequence_id")
            if seq is None:
                return {"type": "servo_reject", "reason": "missing_sequence_id"}, local_gen, label, last_seq, last_req
            if last_seq is not None and not 0 < seq - last_seq <= 1000:
                return {"type": "servo_reject", "reason": "invalid_sequence_id"}, local_gen, label, last_seq, last_req
            last_seq = seq
            err = self.motion_gate(msg.get("positions") or [], servo=True)
            if err is not None:
                return (
                    {
                        "type": "servo_reject",
                        "status": err[0],
                        "code": err[1],
                        "error": err[1],
                        "message": err[1],
                        "epoch_id": self.epoch,
                    },
                    local_gen,
                    label,
                    last_seq,
                    last_req,
                )
            target = list(msg["positions"])
            if any(abs(t - p) > SERVO_JUMP for i, (t, p) in enumerate(zip(target, self.positions)) if i != 6):
                self.dropped_servo_frames += 1  # armd discards the frame without a reply
                self.servo_target = None
                return None, local_gen, label, last_seq, last_req
            self.servo_target = target
            self.last_motion_frame = self.last_heartbeat = time.monotonic()
            self.state = "Moving"
            return None, local_gen, label, last_seq, last_req
        return None, local_gen, label, last_seq, last_req

    def motion_gate(self, targets, servo=False):
        if self.collision_state and self.state in ("Armed", "Moving"):
            return 409, "COLLISION_HOLD"
        if self.state == "EStop":
            return 409, "ESTOP_ACTIVE"
        if self.state == "Fault":
            return 409, "FAULT_ACTIVE"
        if self.state not in ("Armed", "Moving") or self.pending_rearm or (servo and self.traj is not None):
            return 412, "NOT_ARMED"
        if len(targets) != DOF:
            return 400, "SAFETY_ERROR"
        if any(not lo <= t <= hi for t, (lo, hi) in zip(targets, LIMITS)):
            return 400, "SOFT_LIMIT_EXCEEDED"
        return None

    def start_motion(self, kind, target, duration):
        if self.actions_supported:
            self.start_action(kind, list(target), max(float(duration or 0.3), 0.05))
        else:
            self.start_traj(target, duration)

    def start_traj(self, target, duration):
        self.traj = (list(self.positions), list(target), time.monotonic(), max(float(duration or 0.3), 0.05))
        self.servo_target = None
        self.state = "Moving"
        self.last_heartbeat = time.monotonic()

    def execute(self, path, body):
        if path == "/api/v1/torque":
            enable = body.get("enable")
            if not isinstance(enable, bool):
                return 400, "INVALID_COMMAND"
            if self.state == "EStop":
                return 409, "ESTOP_ACTIVE"
            if self.state == "Fault":
                return 409, "FAULT_ACTIVE"
            if not enable and (not self.resting_pose or self.resting_violation()):
                return 409, "ARM_NOT_AT_RESTING_POSE"
            self.stop_motion()
            self.is_torqued = enable
            self.state = "Armed" if enable else "Idle"
            self.pending_rearm = False
            self.last_heartbeat = time.monotonic()
            return None
        if path == "/api/v1/safety/clear_fault":
            if self.state in ("EStop", "Fault"):
                self.state, self.fault = "Idle", None
                self.collision_state, self.collision_trips = None, []
                self.epoch += 1
                self.pending_rearm = True
            return None
        if path == "/api/v1/safety/collision":
            if not (self.devplat and self.collision_cap):
                return 400, "INVALID_COMMAND"
            self.collision_bodies.append(dict(body))
            if "enabled" in body:
                self.collision_cfg["enabled"] = bool(body["enabled"])
            if body.get("sensitivity") in ("low", "medium", "high"):
                self.collision_cfg["sensitivity"] = body["sensitivity"]
            return None
        if path.startswith("/api/v1/action/"):
            return self.execute_action(path, body)
        if path == "/api/v1/motion/jog":
            err = self.motion_gate(body.get("target_positions") or [])
            if err is None:
                self.start_motion("joint_move", body["target_positions"], body.get("duration_secs"))
            return err
        if path == "/api/v1/motion/home":
            err = self.motion_gate(self.home)
            if err is None:
                self.start_motion("home", self.home, body.get("duration_secs"))
            return err
        if path == "/api/v1/motion/unload":
            if not self.resting_pose:
                return 422, "UNSUPPORTED"
            if self.state not in ("Armed", "Moving"):
                return 412, "NOT_ARMED"
            self.start_motion("park", self.home, 0.3)
            self.pending_unload = True
            return None
        if path in ("/api/v1/poses/save", "/api/v1/poses/delete"):
            if not self.devplat:
                return 400, "INVALID_COMMAND"
            name = body.get("name")
            if not isinstance(name, str) or not name:
                return 400, "INVALID_COMMAND"
            if path.endswith("delete"):
                return None if self.poses.pop(name, None) else (404, "POSE_NOT_FOUND")
            positions = list(body.get("positions") or self.positions)
            if len(positions) != DOF:
                return 400, "INVALID_COMMAND"
            self.poses[name] = {"name": name, "positions": positions, "tcp": None, "saved_at_ms": 1}
            return None
        if path == "/api/v1/workspace/keepout":
            boxes = body.get("boxes")
            if not self.devplat or not isinstance(boxes, list) or len(boxes) > 16:
                return 400, "INVALID_COMMAND"
            self.keepout = [dict(b) for b in boxes]
            return None
        if path == "/api/v1/motion/stop":
            self.cancel_reason = "STOPPED"
            self.stop_motion()
            if self.state == "Moving":
                self.state = "Armed"
            return None
        return 400, "INVALID_COMMAND"

    def do_estop(self) -> None:
        self.state, self.fault = "EStop", None
        self.epoch += 1
        self.stop_motion()

    # ---- REST (served through MockTransport) -------------------------
    def rest(self, request: Request) -> Response:
        path, method = request.url.path, request.method
        with self.lock:
            if method == "GET" and path == "/api/v1/observation":
                return Response(200, json=self.observation())
            if method == "GET" and path == "/api/v1/observation/schema":
                return Response(
                    200,
                    json={
                        "dof": DOF,
                        "joint_names": [f"joint_{i + 1}" for i in range(DOF)],
                        "fields": [],
                        "capabilities": self.capabilities(),
                        "limits": [
                            {
                                "name": f"joint_{i + 1}",
                                "min_position": lo,
                                "max_position": hi,
                                "max_velocity": 2.0,
                                "max_acceleration": 5.0,
                            }
                            for i, (lo, hi) in enumerate(LIMITS)
                        ],
                    },
                )
            if method == "GET" and path == "/api/v1/poses" and self.devplat:
                return Response(200, json={"poses": list(self.poses.values())})
            if method == "GET" and path == "/api/v1/safety/collision" and self.devplat and self.collision_cap:
                scale = {"low": 1.5, "medium": 1.0, "high": 0.7}[self.collision_cfg["sensitivity"]]
                return Response(
                    200,
                    json={
                        **self.collision_cfg,
                        "thresholds_nm": [round(x * scale, 2) for x in [6, 6, 6, 2.5, 2.5, 2.5]],
                    },
                )
            if method == "GET" and path == "/api/v1/safety/describe" and self.devplat and self.describe_cap:
                return Response(
                    200,
                    json=json.loads((pathlib.Path(__file__).parent / "fixtures" / "safety_describe.json").read_text()),
                )
            if method == "GET" and path == "/api/v1/workspace/keepout" and self.devplat:
                return Response(200, json={"boxes": self.keepout, "source": "runtime" if self.keepout else "profile"})
            if method == "GET" and path == "/api/v1/vision/mode":
                return Response(200, json={"api_version": 2, "mode": "off", "per_request_pipelines": True})
            if method == "POST" and path == "/api/v1/detection/trigger":
                self.trigger_bodies.append(json.loads(request.content) if request.content else None)
                return Response(
                    200,
                    json={
                        "api_version": 2,
                        "success": True,
                        "frame_id": 7,
                        "ts_ms": 1000,
                        "detections": self.detections,
                    },
                )
            if method == "POST" and path == "/api/v1/safety/e_stop":
                self.do_estop()
                return Response(200, json={"safety_state": self.state, "epoch_id": self.epoch})
            if method == "POST":
                code = "LEASE_ACTIVE" if self.lease else "LEASE_REQUIRED"
                return Response(409, json={"error": code, "code": code, "message": "Use the WebSocket lease"})
        return Response(404)

    def stop(self) -> None:
        self._stop.set()


def open_arm(armd, **kw) -> Arm:
    return Arm(armd.url, transport=armd.transport, **kw)


@pytest.fixture
def armd():
    fake = FakeArmd()
    server = serve(fake.handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=fake.run_ticker, daemon=True).start()
    fake.url = f"http://127.0.0.1:{port}"
    fake.transport = MockTransport(fake.rest)
    fake.client = ArmdClient(fake.url, transport=fake.transport)
    yield fake
    fake.stop()
    server.shutdown()
