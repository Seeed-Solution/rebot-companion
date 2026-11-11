"""ArmSession high-level actions against the fake armd's action model (tests/fake_armd.py)."""

from __future__ import annotations

import threading
import time

import pytest
from fake_armd import HOME, q

from rebot.client import (
    ActionFailed,
    ArmCommandRejected,
    ArmdError,
    ArmStatus,
    LeaseError,
    MotionTimeout,
)


def test_status_parses_commanded_positions_and_action():
    obs = ArmStatus.from_dict(
        {
            "safety_state": "Moving",
            "epoch_id": 1,
            "positions": [0.0] * 7,
            "commanded_positions": [0.1] * 7,
            "action": {
                "id": 4,
                "kind": "gripper",
                "tag": "t",
                "state": "done",
                "code": None,
                "progress": 1.0,
                "started_at": 1.0,
                "finished_at": 2.5,
                "holding": True,
                "gripper_gap_rad": 1.85,
                "settled": None,
            },
        }
    )
    assert obs.commanded_positions == [0.1] * 7
    a = obs.action
    assert a.id == 4 and a.kind == "gripper" and a.terminal and a.holding is True
    assert a.gripper_gap_rad == pytest.approx(1.85) and a.finished_at == 2.5 and a.settled is None
    old = ArmStatus.from_dict({"safety_state": "Idle", "epoch_id": 1})
    assert old.commanded_positions == [] and old.action is None


def test_move_pose_blocks_until_done(armd):
    with armd.client.session(client_id="act") as s:
        s.torque(True)
        t0 = time.monotonic()
        res = s.move_pose([0.25, 0.0, 0.25], rpy=[0.0, 0.5, 0.0], linear_speed=0.05, tag="pick")
        assert time.monotonic() - t0 >= armd.action_duration - 0.05
        assert res.action.state == "done" and res.action.kind == "move_to" and res.action.settled
        assert res.action.id == res.ack["action_id"] == 1
        assert res.ack["expected_duration_s"] == pytest.approx(armd.action_duration)
        assert res.status.safety_state == "Armed"
    path, body = armd.action_bodies[0]
    assert path == "/api/v1/action/move_to"
    assert body == {
        "position": [0.25, 0.0, 0.25],
        "orientation": {"rpy": [0.0, 0.5, 0.0]},
        "speed": {"linear": 0.05},
        "tag": "pick",
    }


def test_move_pose_keep_orientation_and_quat(armd):
    with armd.client.session(client_id="q") as s:
        s.torque(True)
        s.move_pose([0.2, 0.0, 0.3])
        s.move_pose([0.2, 0.0, 0.3], quat=[1, 0, 0, 0])
        with pytest.raises(ValueError):
            s.move_pose([0.2, 0.0, 0.3], rpy=[0, 0, 0], quat=[1, 0, 0, 0])
    assert armd.action_bodies[0][1]["orientation"] is None
    assert armd.action_bodies[1][1]["orientation"] == {"quat": [1.0, 0.0, 0.0, 0.0]}


def test_move_relative_body(armd):
    with armd.client.session(client_id="rel") as s:
        s.torque(True)
        res = s.move_relative("camera", (0.0, 0.0, 0.05), dpitch=0.1, level=True)
        assert res.action.kind == "move_relative"
        with pytest.raises(ValueError):
            s.move_relative("world", (0, 0, 0))
    assert armd.action_bodies[0][1] == {"frame": "camera", "d": [0.0, 0.0, 0.05], "dpitch": 0.1, "level": True}


def test_gripper_close_reports_holding(armd):
    with armd.client.session(client_id="grip") as s:
        s.torque(True)
        res = s.gripper("close")
        assert res.action.holding is True and res.action.gripper_gap_rad > 0.3
        armd.gripper_contact = None  # nothing between the jaws
        res = s.gripper("close")
        assert res.action.holding is False
        res = s.gripper("move", fraction=0.5)
        assert res.status.positions[6] == pytest.approx(-2.85, abs=1e-6)
        with pytest.raises(ValueError):
            s.gripper("move")
        with pytest.raises(ArmCommandRejected) as exc:
            s.command("/api/v1/action/gripper", {"command": "move", "width_m": 0.04})
        assert exc.value.code == "UNSUPPORTED"


def test_failed_action_raises_action_failed(armd):
    armd.action_failure = ("TRACKING_STALL", "J2 blocked")
    with armd.client.session(client_id="stall") as s:
        s.torque(True)
        with pytest.raises(ActionFailed) as exc:
            s.move_pose([0.25, 0.0, 0.25])
        assert exc.value.code == "TRACKING_STALL" and exc.value.message == "J2 blocked"
        assert exc.value.status.state == "failed"
        assert s.latest_observation().safety_state == "Armed"  # no Fault latched


def test_rejected_action_raises_command_rejected(armd):
    armd.action_reject = (422, "IK_UNREACHABLE")
    with armd.client.session(client_id="ik") as s:
        s.torque(True)
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_pose([1.5, 0.0, 0.25])
        assert exc.value.code == "IK_UNREACHABLE" and exc.value.status == 422
        assert armd.positions == pytest.approx(HOME)


def test_motion_busy_while_trajectory_runs(armd):
    with armd.client.session(client_id="busy") as s:
        s.torque(True)
        s.move_to(q(0.3), duration=0.5, wait=False)
        with pytest.raises(ArmCommandRejected) as exc:
            s.move_pose([0.25, 0.0, 0.25])
        assert exc.value.code == "MOTION_BUSY"


def test_timeout_sends_stop_and_cancels(armd):
    armd.freeze_motion = True
    with armd.client.session(client_id="to") as s:
        s.torque(True)
        with pytest.raises(MotionTimeout):
            s.move_pose([0.25, 0.0, 0.25], timeout=0.4)
        assert armd.commands[-1] == "/api/v1/motion/stop"
        obs = s.wait_for_observation()
        assert obs.action.state == "cancelled" and obs.action.code == "STOPPED"


def test_default_timeout_from_expected_duration(armd, monkeypatch):
    seen = {}
    with armd.client.session(client_id="dt") as s:
        s.torque(True)
        real = s._wait_until

        def spy(frames, pred, timeout, what, check_stopped=True):
            seen["timeout"] = timeout
            return real(frames, pred, timeout, what, check_stopped)

        monkeypatch.setattr(s, "_wait_until", spy)
        s.move_pose([0.25, 0.0, 0.25])
        assert seen["timeout"] == pytest.approx(2 * armd.action_duration + 5)
        armd.action_duration = 40.0  # 2*40+5 > 60 -> capped; stop it right away
        threading.Timer(0.2, lambda: s.stop()).start()
        with pytest.raises(ActionFailed) as exc:
            s.move_pose([0.25, 0.0, 0.25])
        assert seen["timeout"] == 60.0 and exc.value.code == "STOPPED"


def test_estop_mid_action(armd):
    armd.action_duration = 1.0
    with armd.client.session(client_id="es") as s:
        s.torque(True)
        threading.Timer(0.15, armd.client.e_stop).start()
        with pytest.raises(ActionFailed) as exc:
            s.move_pose([0.25, 0.0, 0.25])
        assert exc.value.code == "ESTOP_ACTIVE"


def test_jog_supersedes_running_action(armd):
    armd.action_duration = 1.0
    with armd.client.session(client_id="sup") as s:
        s.torque(True)
        errors = []

        def run():
            try:
                s.move_pose([0.25, 0.0, 0.25])
            except ActionFailed as exc:
                errors.append(exc.code)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.2)
        s.move_to(q(0.1), duration=0.1)  # jog over a running action is allowed
        t.join(3)
        assert errors == ["SUPERSEDED"]


def test_preemption_cancels_with_lease_lost(armd):
    armd.action_duration = 1.0
    a = armd.client.session(client_id="a", safe_exit="off").open()
    try:
        a.torque(True)
        result = {}

        def run():
            try:
                a.move_pose([0.25, 0.0, 0.25])
            except ArmdError as exc:
                result["exc"] = exc

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.2)
        with armd.client.session(client_id="b", force=True, safe_exit="off"):
            t.join(3)
        assert isinstance(result["exc"], LeaseError)  # lease_lost arrives before armd's next tick
        time.sleep(0.1)
        assert armd.action["state"] == "cancelled" and armd.action["code"] == "LEASE_LOST"
    finally:
        a.close()


def test_home_and_park_actions(armd):
    with armd.client.session(client_id="hp", safe_exit="off") as s:
        s.torque(True)
        s.move_to(q(0.3), duration=0.1)
        res = s.home_action()
        assert res.action.kind == "home" and res.action.state == "done"
        res = s.park_action()
        assert res.action.kind == "park" and res.status.safety_state == "Idle" and not res.status.is_torqued


def test_park_action_reports_unload_failed(armd):
    armd.fail_torque_cut = True
    with armd.client.session(client_id="uf", safe_exit="off") as s:
        s.torque(True)
        with pytest.raises(ActionFailed) as exc:
            s.park_action()
        assert exc.value.code == "UNLOAD_FAILED"


def test_home_and_park_fall_back_without_action_records(armd):
    armd.actions_supported = False
    with armd.client.session(client_id="old", safe_exit="off") as s:
        s.torque(True)
        s.move_to(q(0.3), duration=0.1)
        res = s.home_action()
        assert res.action is None and res.status.positions == pytest.approx(HOME, abs=1e-6)
        res = s.park_action()
        assert res.action is None and res.status.safety_state == "Idle"
        with pytest.raises(ArmCommandRejected) as exc:  # no /api/v1/action/* on this armd
            s.move_pose([0.25, 0.0, 0.25])
        assert exc.value.code == "INVALID_COMMAND"
