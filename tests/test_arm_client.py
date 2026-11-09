"""ArmSession against the in-process fake armd (``tests/fake_armd.py``)."""

from __future__ import annotations

import json
import threading
import time

import pytest
from fake_armd import DOF, HOME, q
from websockets.sync.client import connect as ws_connect

from rebot_companion.arm_client import (
    ArmCommandRejected,
    ArmFaultError,
    ArmStatus,
    LeaseError,
    MotionTimeout,
    SafeExitOutcome,
    ServoJumpError,
    ServoRejected,
    UnloadFailed,
)

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
        done = s.move_to(q(0.3, -0.2, 0.1), duration=0.4)
        assert time.monotonic() - t0 >= 0.35  # blocked until the trajectory finished
        assert done.safety_state == "Armed"
        assert done.positions == pytest.approx(q(0.3, -0.2, 0.1), abs=1e-6)
        done = s.home(duration=0.2)
        assert done.positions == pytest.approx(HOME, abs=1e-6)
        s.servo_stream([q(0.01 * i) for i in range(1, 11)], rate_hz=50)
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
        s.move_to(q(0.1), duration=0.1)


def test_heartbeat_starvation_trips_watchdog_and_rejects_motion(armd):
    s = armd.client.session(client_id="starved", heartbeat_interval=60.0).open()
    try:
        s.torque(True)
        time.sleep(armd.lease_timeout * 2)
        assert armd.client.get_observation().fault_reason == "WatchdogTimeout"
        with pytest.raises(LeaseError):
            s.move_to(q(0.1), duration=0.1)
    finally:
        s.close()


def test_epoch_change_invalidates_commands_and_clear_fault_recovers(armd):
    with armd.client.session(client_id="ep") as s:
        s.torque(True)
        res = armd.client.e_stop()  # REST e-stop while another connection holds the lease
        assert res == {"safety_state": "EStop", "epoch_id": 2}
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_to(q(0.1), duration=0.1)
        assert exc.value.code in ("EPOCH_MISMATCH", "ESTOP_ACTIVE")
        ack = s.clear_fault()
        assert ack["epoch_id"] == 3 and ack["safety_state"] == "Idle"
        with pytest.raises(ArmCommandRejected) as exc:  # clear_fault leaves the arm needing re-arm
            s.move_to(q(0.1), duration=0.1)
        assert exc.value.code == "NOT_ARMED"
        s.torque(True)
        s.move_to(q(0.1), duration=0.1)
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
            s.move_to(q(1.0), duration=1.0)
        assert exc.value.status.e_stopped


def test_motion_timeout_sends_stop(armd):
    armd.freeze_motion = True
    with armd.client.session(client_id="to") as s:
        s.torque(True)
        with pytest.raises(MotionTimeout):
            s.move_to(q(0.5), duration=0.1, timeout=0.4)
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
                a.move_to(q(0.1), duration=0.1)
    finally:
        a.close()


def test_soft_limit_and_servo_reject(armd):
    with armd.client.session(client_id="lim") as s:
        s.torque(True)
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_to(q(9.0), duration=0.1)
        assert exc.value.code == "SOFT_LIMIT_EXCEEDED"
        s.move_to(q(0.5), duration=0.5, wait=False)
        s.servo(q(0.2))  # servo while a trajectory runs -> servo_reject NOT_ARMED
        time.sleep(0.1)
        with pytest.raises(ServoRejected) as exc:
            s.servo(q(0.2))
        assert exc.value.code == "NOT_ARMED"


def test_servo_stream_rejects_low_rate(armd):
    with armd.client.session(client_id="rate") as s:
        with pytest.raises(ValueError):
            s.servo_stream([HOME], rate_hz=9)


# ------------------------------------------------------------- armd safety rules ---


def test_targets_must_have_seven_joints(armd):
    with armd.client.session(client_id="len") as s:
        s.torque(True)
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_to([0.0] * 6, duration=0.1)
        assert exc.value.code == "SAFETY_ERROR"


def test_torque_off_refused_away_from_rest(armd):
    with armd.client.session(client_id="rest", safe_exit="off") as s:
        s.torque(True)
        s.move_to(q(0.5), duration=0.2)
        with pytest.raises(ArmCommandRejected) as exc:
            s.torque(False)
        assert exc.value.code == "ARM_NOT_AT_RESTING_POSE"
        assert s.latest_observation().is_torqued
        s.unload()


def test_servo_jump_raises_instead_of_silent_drop(armd):
    with armd.client.session(client_id="jump") as s:
        s.torque(True)
        with pytest.raises(ServoJumpError, match="joint 1"):
            s.servo(q(0.5))  # 0.5 rad from the observed position
        s.servo(q(0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963, -3.0))  # gripper is exempt
        s.servo_stream([q(0.05 * i) for i in range(1, 5)], rate_hz=50)
        with pytest.raises(ServoJumpError):
            s.servo_stream([q(0.2), q(0.6)], rate_hz=50)  # step from the previous frame
        assert armd.dropped_servo_frames == 0  # nothing reached armd that it would drop


def test_fake_armd_drops_large_servo_jump_without_reply(armd):
    with armd.client.session(client_id="raw") as s:
        s.torque(True)
        s._send(
            {
                "type": "servo",
                "client_id": s.client_id,
                "generation": s.generation,
                "sequence_id": 1,
                "epoch_id": s.epoch_id,
                "positions": q(0.5),
            }
        )
        time.sleep(0.1)
        assert armd.dropped_servo_frames == 1
        assert s._servo_error is None  # no servo_reject
        assert s.latest_observation().positions[0] == pytest.approx(0.0)


def test_unload_unsupported_without_resting_pose(armd):
    armd.resting_pose = False
    with armd.client.session(client_id="nrp", safe_exit="off") as s:
        s.torque(True)
        with pytest.raises(ArmCommandRejected) as exc:
            s.unload()
        assert exc.value.code == "UNSUPPORTED"


def test_unload_reports_failed_torque_cut_without_waiting_full_timeout(armd):
    armd.fail_torque_cut = True
    with armd.client.session(client_id="cut", safe_exit="off") as s:
        s.torque(True)
        s.move_to(q(0.2), duration=0.1)
        t0 = time.monotonic()
        with pytest.raises(UnloadFailed) as exc:
            s.unload(timeout=10.0)
        assert time.monotonic() - t0 < 2.0
        assert exc.value.status.is_torqued and exc.value.status.safety_state == "Armed"


def test_unload_without_wait_is_finished_before_close(armd):
    with armd.client.session(client_id="nowait", safe_exit="off") as s:
        s.torque(True)
        s.move_to(q(0.3), duration=0.1)
        s.unload(wait=False)
    # release would have cancelled the unload; close waited for it instead
    assert armd.state == "Idle" and not armd.is_torqued
    assert armd.positions == pytest.approx(HOME, abs=1e-6)


# ------------------------------------------------------------------- safe exit ---


def _assert_parked(armd):
    time.sleep(armd.lease_timeout * 1.5)  # past the watchdog window: no Fault latched
    obs = armd.client.get_observation()
    assert obs.safety_state == "Idle" and not obs.is_torqued
    assert obs.positions == pytest.approx(HOME, abs=1e-6)


def test_safe_exit_unloads_on_exception(armd):
    with pytest.raises(RuntimeError):
        with armd.client.session(client_id="err") as s:
            s.torque(True)
            s.move_to(q(0.4), duration=0.1)
            raise RuntimeError("cycle failed")
    assert s.safe_exit_outcome == SafeExitOutcome("unloaded", "parked at the resting pose, torque off")
    assert armd.commands[-1] == "/api/v1/motion/unload"
    assert armd.lease is None
    _assert_parked(armd)


def test_safe_exit_on_keyboard_interrupt(armd):
    with pytest.raises(KeyboardInterrupt):
        with armd.client.session(client_id="kbi") as s:
            s.torque(True)
            s.move_to(q(0.4), duration=0.5, wait=False)
            raise KeyboardInterrupt
    assert s.safe_exit_outcome.action == "unloaded"
    _assert_parked(armd)


def test_safe_exit_clears_fault_then_unloads(armd):
    with pytest.raises(ArmFaultError):
        with armd.client.session(client_id="flt") as s:
            s.torque(True)
            threading.Timer(0.1, armd.trigger_fault, args=("Other",)).start()
            s.move_to(q(0.6), duration=1.0)
    assert s.safe_exit_outcome.action == "unloaded"
    assert armd.commands[-3:] == ["/api/v1/safety/clear_fault", "/api/v1/torque", "/api/v1/motion/unload"]
    _assert_parked(armd)


def test_safe_exit_unsupported_is_reported_and_never_disables_torque(armd):
    armd.resting_pose = False
    with pytest.raises(RuntimeError):
        with armd.client.session(client_id="uns") as s:
            s.torque(True)
            s.move_to(q(0.4), duration=0.1)
            raise RuntimeError("boom")
    assert s.safe_exit_outcome.action == "unsupported"
    assert "resting-pose" in s.safe_exit_outcome.message
    assert armd.commands.count("/api/v1/torque") == 1  # only the enable
    assert armd.is_torqued


def test_safe_exit_leaves_estop_latched(armd):
    with pytest.raises(ArmFaultError):
        with armd.client.session(client_id="es") as s:
            s.torque(True)
            threading.Timer(0.1, armd.client.e_stop).start()
            s.move_to(q(0.6), duration=1.0)
    assert s.safe_exit_outcome.action == "estop"
    assert "/api/v1/safety/clear_fault" not in armd.commands
    assert armd.state == "EStop"


def test_safe_exit_policies(armd):
    with armd.client.session(client_id="off", safe_exit="off") as s:
        s.torque(True)
    assert s.safe_exit_outcome is None and armd.is_torqued  # lease released, arm left holding
    time.sleep(armd.lease_timeout * 1.5)
    assert armd.client.get_observation().fault_reason == "WatchdogTimeout"
    with armd.client.session(client_id="always", safe_exit="always") as s:
        s.clear_fault()
        s.torque(True)
        s.move_to(q(0.3), duration=0.1)
    assert s.safe_exit_outcome.action == "unloaded"
    _assert_parked(armd)
    with armd.client.session(client_id="idle", safe_exit="always") as s:
        pass
    assert s.safe_exit_outcome.action == "not_needed"
    with pytest.raises(ValueError):
        armd.client.session(safe_exit="sometimes")


def test_heartbeats_keep_flowing_through_jog_unload_cycles_with_a_polling_thread(armd):
    """Real-arm script shape (armd fce83ff): torque, jog, unload, torque, jog, while a
    second thread polls latest_observation() every 20 ms. Heartbeats must not stall."""
    beats: list[float] = []
    orig = armd.handle

    def handle(conn, msg, kind, *rest):
        if kind == "heartbeat":
            beats.append(time.monotonic())
        return orig(conn, msg, kind, *rest)

    armd.handle = handle
    stop = threading.Event()
    with armd.client.session(client_id="cycles") as s:

        def poll():
            while not stop.is_set():
                s.latest_observation()
                time.sleep(0.02)

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        try:
            for _ in range(2):
                s.torque(True)
                s.move_to(q(0.3), duration=0.3)
                s.unload(timeout=5)
            s.torque(True)
            s.move_to(q(0.3), duration=0.3)
            assert s.has_lease
        finally:
            stop.set()
            th.join(1)
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    assert len(beats) > 5 and max(gaps) < armd.lease_timeout / 2, max(gaps)
