"""ArmSession against an in-process fake armd that enforces armd's WS control protocol.

The fake mirrors rebot-armd ``src/api/ws.rs`` and ``src/supervisor.rs``:
lease required for every actuator command, generation + epoch checks,
strictly increasing request_id / sequence_id, heartbeat-driven lease expiry,
watchdog Fault(WatchdogTimeout) while torqued without heartbeats, 200 ms servo
frame timeout, and lease-free e-stop over WS and REST.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect
from websockets.sync.server import serve

from rebot_companion.arm_client import (
    ArmCommandRejected,
    ArmdClient,
    ArmFaultError,
    ArmStatus,
    LeaseError,
    MotionTimeout,
    ServoRejected,
)

DOF = 6
LIMITS = [(-3.14, 3.14)] * DOF
TICK = 0.02  # 50 Hz like armd's default control_frequency_hz


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
        self.positions = [0.0] * DOF
        self.home = [0.0] * DOF
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
        }

    def trigger_fault(self, reason: str) -> None:
        self.state, self.fault = "Fault", reason
        self.epoch += 1
        self.traj = None
        self.servo_target = None

    def tick(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.is_torqued and self.state in ("Armed", "Moving") and now - self.last_heartbeat > self.lease_timeout:
                self.trigger_fault("WatchdogTimeout")
            if self.traj and not self.freeze_motion:
                start, end, t0, dur = self.traj
                a = min(1.0, (now - t0) / dur)
                self.positions = [s + (e - s) * a for s, e in zip(start, end)]
                if a >= 1.0:
                    self.traj = None
                    if self.state == "Moving":
                        self.state = "Armed"
                    if self.pending_unload:
                        self.pending_unload = False
                        self.is_torqued = False
                        self.state = "Idle"
            elif self.servo_target is not None:
                if now - self.last_motion_frame > self.frame_timeout:
                    self.servo_target = None
                else:
                    step = 2.0 * TICK
                    self.positions = [
                        p + max(-step, min(step, t - p)) for p, t in zip(self.positions, self.servo_target)
                    ]
            text = json.dumps(self.observation())
        for conn_id in list(self.conns):
            self.send(conn_id, text)

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
                if reply is not None:
                    self.send(conn, json.dumps(reply))
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.conns.pop(conn, None)
                if local_gen and self.lease and self.lease["conn"] == conn and self.lease["generation"] == local_gen:
                    self.lease = None
                    self.traj = None
                    self.servo_target = None

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
                self.traj = None
                self.servo_target = None
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
            self.traj = None
            self.servo_target = None
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
            self.servo_target = list(msg["positions"])
            self.last_motion_frame = self.last_heartbeat = time.monotonic()
            self.state = "Moving"
            return None, local_gen, label, last_seq, last_req
        return None, local_gen, label, last_seq, last_req

    def motion_gate(self, targets, servo=False):
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
            self.is_torqued = enable
            self.state = "Armed" if enable else "Idle"
            self.pending_rearm = False
            self.last_heartbeat = time.monotonic()
            return None
        if path == "/api/v1/safety/clear_fault":
            if self.state in ("EStop", "Fault"):
                self.state, self.fault = "Idle", None
                self.epoch += 1
                self.pending_rearm = True
            return None
        if path == "/api/v1/motion/jog":
            err = self.motion_gate(body.get("target_positions") or [])
            if err is None:
                self.start_traj(body["target_positions"], body.get("duration_secs"))
            return err
        if path == "/api/v1/motion/home":
            err = self.motion_gate(self.home)
            if err is None:
                self.start_traj(self.home, body.get("duration_secs"))
            return err
        if path == "/api/v1/motion/unload":
            if self.state not in ("Armed", "Moving"):
                return 412, "NOT_ARMED"
            self.start_traj(self.home, 0.3)
            self.pending_unload = True
            return None
        if path == "/api/v1/motion/stop":
            self.traj = None
            self.servo_target = None
            if self.state == "Moving":
                self.state = "Armed"
            return None
        return 400, "INVALID_COMMAND"

    def do_estop(self) -> None:
        self.state, self.fault = "EStop", None
        self.epoch += 1
        self.traj = None
        self.servo_target = None

    # ---- REST (served through httpx.MockTransport) -------------------------
    def rest(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        with self.lock:
            if method == "GET" and path == "/api/v1/observation":
                return httpx.Response(200, json=self.observation())
            if method == "GET" and path == "/api/v1/observation/schema":
                return httpx.Response(
                    200,
                    json={
                        "dof": DOF,
                        "joint_names": [f"joint_{i + 1}" for i in range(DOF)],
                        "fields": [],
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
            if method == "POST" and path == "/api/v1/safety/e_stop":
                self.do_estop()
                return httpx.Response(200, json={"safety_state": self.state, "epoch_id": self.epoch})
            if method == "POST":
                code = "LEASE_ACTIVE" if self.lease else "LEASE_REQUIRED"
                return httpx.Response(409, json={"error": code, "code": code, "message": "Use the WebSocket lease"})
        return httpx.Response(404)

    def stop(self) -> None:
        self._stop.set()


@pytest.fixture
def armd():
    fake = FakeArmd()
    server = serve(fake.handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=fake.run_ticker, daemon=True).start()
    fake.url = f"http://127.0.0.1:{port}"
    fake.client = ArmdClient(fake.url, transport=httpx.MockTransport(fake.rest))
    yield fake
    fake.stop()
    server.shutdown()


# --------------------------------------------------------------------------- tests


def test_observation_parsing_matches_armd_serde():
    obs = ArmStatus.from_dict(
        {
            "timestamp": 1.5,
            "positions": [0.1],
            "velocities": [0],
            "torques": [0],
            "temperatures": [25],
            "is_torqued": True,
            "epoch_id": 7,
            "safety_state": {"Fault": {"Other": "Persistent joint tracking error"}},
        }
    )
    assert obs.safety_state == "Fault" and obs.faulted and obs.is_stopped
    assert obs.fault_reason == "Other: Persistent joint tracking error"
    assert obs.epoch_id == 7 and obs.positions == [0.1]
    wd = ArmStatus.from_dict({"safety_state": {"Fault": "WatchdogTimeout"}, "epoch_id": 2})
    assert wd.fault_reason == "WatchdogTimeout"
    assert ArmStatus.from_dict({"safety_state": "EStop", "epoch_id": 3}).e_stopped


def test_rest_observation_and_schema(armd):
    obs = armd.client.get_observation()
    assert obs.safety_state == "Idle" and obs.epoch_id == 1 and len(obs.positions) == DOF
    schema = armd.client.get_schema()
    assert schema.dof == DOF and schema.limits[0].max_velocity == 2.0


def test_raw_command_without_lease_is_rejected(armd):
    """The fake enforces the lease exactly like ws.rs: no acquire_lease -> LEASE_LOST."""
    with ws_connect(armd.url.replace("http", "ws") + "/api/v1/ws") as ws:
        ws.send(
            json.dumps(
                {
                    "type": "command",
                    "request_id": 1,
                    "path": "/api/v1/torque",
                    "body": {"enable": True},
                    "generation": 1,
                    "epoch_id": 1,
                }
            )
        )
        while True:
            msg = json.loads(ws.recv(timeout=2))
            if msg.get("type") == "command_reject":
                break
    assert msg["code"] == "LEASE_LOST"
    assert armd.commands == []


def test_full_motion_cycle(armd):
    with armd.client.session(client_id="test") as s:
        assert s.generation == 1 and s.epoch_id == 1
        ack = s.torque(True)
        assert ack["is_torqued"] is True
        t0 = time.monotonic()
        done = s.move_to([0.3, -0.2, 0.1, 0, 0, 0], duration=0.4)
        assert time.monotonic() - t0 >= 0.35  # blocked until the trajectory finished
        assert done.safety_state == "Armed"
        assert done.positions == pytest.approx([0.3, -0.2, 0.1, 0, 0, 0], abs=1e-6)
        done = s.home(duration=0.2)
        assert done.positions == pytest.approx([0.0] * DOF, abs=1e-6)
        s.servo_stream([[0.01 * i, 0, 0, 0, 0, 0] for i in range(1, 11)], rate_hz=50)
        time.sleep(0.15)
        assert s.latest_observation().positions[0] == pytest.approx(0.1, abs=0.02)
        done = s.unload()
        assert done.safety_state == "Idle" and not done.is_torqued
    assert armd.lease is None  # released on exit
    assert armd.commands == ["/api/v1/torque", "/api/v1/motion/jog", "/api/v1/motion/home", "/api/v1/motion/unload"]


def test_second_client_refused_then_granted_after_release(armd):
    with armd.client.session(client_id="a"):
        with pytest.raises(LeaseError) as exc:
            armd.client.session(client_id="b").open()
        assert exc.value.reason == "controller_busy"
    with armd.client.session(client_id="b") as s:
        assert s.generation == 2


def test_heartbeat_keeps_lease_past_timeout(armd):
    with armd.client.session(client_id="hb") as s:
        s.torque(True)
        time.sleep(armd.lease_timeout * 3)
        obs = s.wait_for_observation()
        assert obs.safety_state == "Armed"
        assert armd.heartbeats >= 5
        s.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)


def test_heartbeat_starvation_trips_watchdog_and_rejects_motion(armd):
    s = armd.client.session(client_id="starved", heartbeat_interval=60.0).open()
    try:
        s.torque(True)
        time.sleep(armd.lease_timeout * 2)
        assert armd.client.get_observation().fault_reason == "WatchdogTimeout"
        with pytest.raises(LeaseError):
            s.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)
    finally:
        s.close()


def test_epoch_change_invalidates_commands_and_clear_fault_recovers(armd):
    with armd.client.session(client_id="ep") as s:
        s.torque(True)
        res = armd.client.e_stop()  # REST e-stop while another connection holds the lease
        assert res == {"safety_state": "EStop", "epoch_id": 2}
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)
        assert exc.value.code in ("EPOCH_MISMATCH", "ESTOP_ACTIVE")
        ack = s.clear_fault()
        assert ack["epoch_id"] == 3 and ack["safety_state"] == "Idle"
        with pytest.raises(ArmCommandRejected) as exc:  # clear_fault leaves the arm needing re-arm
            s.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)
        assert exc.value.code == "NOT_ARMED"
        s.torque(True)
        s.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)
        assert s.has_lease  # heartbeats followed the epoch change


def test_e_stop_needs_no_lease(armd):
    with armd.client.session(client_id="holder"):
        assert armd.client.e_stop()["safety_state"] == "EStop"
    assert armd.client.get_observation().e_stopped
    # session-level e-stop sends over WS too
    with armd.client.session(client_id="x") as s:
        s.clear_fault()
        s.e_stop()
        assert s.wait_for_observation().e_stopped


def test_estop_during_motion_raises_fault(armd):
    with armd.client.session(client_id="mv") as s:
        s.torque(True)
        threading.Timer(0.15, armd.client.e_stop).start()
        with pytest.raises(ArmFaultError) as exc:
            s.move_to([1.0, 0, 0, 0, 0, 0], duration=1.0)
        assert exc.value.status.e_stopped


def test_motion_timeout_sends_stop(armd):
    armd.freeze_motion = True
    with armd.client.session(client_id="to") as s:
        s.torque(True)
        with pytest.raises(MotionTimeout):
            s.move_to([0.5, 0, 0, 0, 0, 0], duration=0.1, timeout=0.4)
    assert armd.commands[-1] == "/api/v1/motion/stop"


def test_preemption_surfaces_lease_lost(armd):
    a = armd.client.session(client_id="a").open()
    try:
        a.torque(True)
        with armd.client.session(client_id="b", force=True) as b:
            assert b.generation == 2
            time.sleep(0.1)
            assert not a.has_lease
            with pytest.raises(LeaseError):
                a.move_to([0.1, 0, 0, 0, 0, 0], duration=0.1)
    finally:
        a.close()


def test_soft_limit_and_servo_reject(armd):
    with armd.client.session(client_id="lim") as s:
        s.torque(True)
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_to([9.0, 0, 0, 0, 0, 0], duration=0.1)
        assert exc.value.code == "SOFT_LIMIT_EXCEEDED"
        s.move_to([0.5, 0, 0, 0, 0, 0], duration=0.5, wait=False)
        s.servo([0.4, 0, 0, 0, 0, 0])  # servo while a trajectory runs -> servo_reject NOT_ARMED
        time.sleep(0.1)
        with pytest.raises(ServoRejected) as exc:
            s.servo([0.4, 0, 0, 0, 0, 0])
        assert exc.value.code == "NOT_ARMED"


def test_servo_stream_rejects_low_rate(armd):
    with armd.client.session(client_id="rate") as s:
        with pytest.raises(ValueError):
            s.servo_stream([[0] * DOF], rate_hz=4)
