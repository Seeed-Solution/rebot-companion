"""rebot.Arm against the in-process fake armd (``tests/fake_armd.py``)."""

from __future__ import annotations

import time

import pytest
from fake_armd import HOME

from rebot import (
    Arm,
    ArmCommandRejected,
    ArmConnectionError,
    ArmFaultError,
    ArmStatus,
    GuardTimeout,
    RebotError,
    UnsupportedError,
)
from rebot.arm import _base_url


def open_arm(armd, **kw) -> Arm:
    return Arm(armd.url, transport=armd.transport, **kw)


def test_lifecycle_enable_move_park_release(armd):
    with open_arm(armd) as arm:
        assert arm.state().safety_state == "Armed" and armd.is_torqued
        arm.home()
        res = arm.move_to((0.25, 0.0, 0.2))
        assert res.ok and res.action.kind == "move_to"
        assert armd.action_bodies[-1][1]["position"] == [0.25, 0.0, 0.2]
        arm.move_relative(dz=0.05)
        assert armd.action_bodies[-1][1]["d"] == [0.0, 0.0, 0.05]
    assert armd.state == "Idle" and not armd.is_torqued
    assert armd.lease is None
    assert arm.exit_outcome == "parked (torque off)"
    assert "obs_hz" not in armd.ws_paths[-1]  # capabilities version 5: no obs_rate


def test_exception_inside_block_parks_and_propagates(armd):
    with pytest.raises(KeyError):
        with open_arm(armd):
            raise KeyError("boom")
    assert armd.state == "Idle" and not armd.is_torqued and armd.lease is None


def test_park_then_motion_re_enables(armd):
    with open_arm(armd) as arm:
        arm.park()
        assert not armd.is_torqued
        arm.home()
        assert armd.is_torqued and "/api/v1/torque" in armd.commands[-3:]


def test_readable_reject_message(armd):
    armd.action_reject = (422, "IK_UNREACHABLE")
    with open_arm(armd) as arm:
        with pytest.raises(ArmCommandRejected) as info:
            arm.move_to((0.9, 0, 0.1))
        armd.action_reject = None
    err = info.value
    assert isinstance(err, RebotError) and err.code == "IK_UNREACHABLE"
    assert str(err).startswith("IK_UNREACHABLE: target (0.9, 0, 0.1) is out of reach")


def test_new_features_are_gated_on_capabilities(armd):
    with open_arm(armd) as arm:
        for call in (
            arm.poses,
            lambda: arm.save_pose("a"),
            lambda: arm.move_to_pose("a"),
            lambda: arm.delete_pose("a"),
        ):
            with pytest.raises(UnsupportedError) as info:
                call()
            assert info.value.code == "UNSUPPORTED" and "reports version 5" in info.value.message
    assert not any("/poses" in c or "move_to_pose" in c for c in armd.commands)


def test_poses_with_devplat_armd(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        assert "obs_hz=10" in armd.ws_paths[-1]
        arm.save_pose("bin_a", positions=[0.2, 0, 0, 0, 0, 1.5707963, 0])
        assert [p["name"] for p in arm.poses()] == ["bin_a"]
        res = arm.move_to_pose("bin_a", speed_scale=0.5)
        assert res.action.kind == "move_to_pose" and abs(armd.positions[0] - 0.2) < 1e-6
        assert armd.action_bodies[-1][1] == {"name": "bin_a", "speed_scale": 0.5}
        with pytest.raises(ArmCommandRejected) as info:
            arm.move_to_pose("nope")
        assert info.value.code == "POSE_NOT_FOUND"
        arm.delete_pose("bin_a")
        assert arm.poses() == []
        with pytest.raises(ValueError):
            arm.move_to_pose("bin_a", speed_scale=2.0)


def test_obs_hz_decimates_observations(armd):
    armd.devplat = True
    with open_arm(armd, obs_hz=5) as arm:
        n0 = arm.session._frames
        time.sleep(1.0)
        n = arm.session._frames - n0
    assert 3 <= n <= 7  # 50 Hz fake decimated to 5 Hz


def test_detect_and_grasp(armd):
    armd.detections = [
        {"format": "xywh_center", "class_name": "bottle", "confidence": 0.8, "x": 0.2, "y": 0.5, "w": 0.1, "h": 0.2},
        {"format": "xywh_center", "class_name": "cup", "confidence": 0.6, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
        {"format": "xywh_center", "class_name": "cup", "confidence": 0.9, "x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1},
        {"format": "xywh_center", "class_name": "cup", "confidence": 0.1, "x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1},
    ]
    with open_arm(armd) as arm:
        cups = arm.detect("Cup")
        assert [round(c.score, 2) for c in cups] == [0.9, 0.6]
        assert armd.trigger_bodies[-1] == {"pipelines": ["yolo"]}
        assert cups[0].frame_id == 7 and cups[0].ts_ms == 1000
        assert len(arm.detect(min_score=0.0)) == 4
        res = arm.grasp(cups[0])
        assert res.holding is True and res.action.kind == "grasp"
        body = armd.action_bodies[-1][1]
        assert body["label"] == "cup" and body["frame_id"] == 7
        assert body["box_norm"] == pytest.approx([0.35, 0.35, 0.45, 0.45])
        arm.grasp("cup", box=(0.1, 0.1, 0.2, 0.2))
        with pytest.raises(ValueError):
            arm.grasp("cup")
        with pytest.raises(ValueError):
            arm.detect(pipelines=("video",))


def test_gripper_helpers(armd):
    with open_arm(armd) as arm:
        arm.gripper_open()
        assert armd.positions[6] == pytest.approx(-5.7)
        assert arm.gripper_close().holding is True
        arm.gripper_move(0.5)
        assert armd.action_bodies[-1][1] == {"command": "move", "fraction": 0.5}


def test_guard_releases_lease_when_loop_stops_feeding(armd):
    arm = open_arm(armd)
    arm.guard(0.3)
    for _ in range(4):
        time.sleep(0.1)
        arm.feed()
    assert armd.lease is not None
    time.sleep(0.6)
    assert armd.lease is None
    with pytest.raises(GuardTimeout) as info:
        arm.home()
    assert info.value.code == "GUARD_TIMEOUT"
    with pytest.raises(GuardTimeout):
        arm.feed()
    arm.close()  # no error after a trip


def test_guard_does_not_trip_during_a_long_call(armd):
    armd.action_duration = 1.0
    with open_arm(armd) as arm:
        arm.guard(0.3)
        arm.move_to((0.25, 0.0, 0.2))
        arm.feed()
        assert armd.lease is not None
        arm.guard(None)
        time.sleep(0.5)
        assert armd.lease is not None


def test_recover_clears_only_recoverable_faults(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        armd.trigger_fault("WatchdogTimeout")
        time.sleep(0.3)
        with pytest.raises(ArmFaultError) as info:
            arm.home()
        assert info.value.code == "WATCHDOG_TIMEOUT" and "arm.recover()" in info.value.message
        obs = arm.recover()
        assert obs.is_operational and armd.state == "Armed"
        assert armd.commands.count("/api/v1/safety/clear_fault") == 1

        armd.trigger_fault("FollowingError")
        time.sleep(0.3)
        with pytest.raises(ArmFaultError) as info:
            arm.recover(max_attempts=3)
        assert info.value.code == "FOLLOWING_ERROR" and "web cockpit" in info.value.message
        assert armd.commands.count("/api/v1/safety/clear_fault") == 1
    assert arm.exit_outcome.startswith("left holding: FOLLOWING_ERROR")
    assert armd.commands.count("/api/v1/safety/clear_fault") == 1


def test_recover_never_clears_on_armd_without_fault_codes(armd):
    with open_arm(armd, park_on_exit=False) as arm:
        armd.trigger_fault("WatchdogTimeout")
        time.sleep(0.2)
        with pytest.raises(ArmFaultError) as info:
            arm.recover()
        assert "no fault codes" in info.value.message
    assert "/api/v1/safety/clear_fault" not in armd.commands


def test_recover_refuses_estop(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        arm.e_stop()
        time.sleep(0.2)
        with pytest.raises(ArmFaultError) as info:
            arm.recover()
        assert info.value.code == "ESTOP"
    assert "/api/v1/safety/clear_fault" not in armd.commands


def test_fault_fields_parse():
    obs = ArmStatus.from_dict(
        {
            "safety_state": {"Fault": "WatchdogTimeout"},
            "epoch_id": 1,
            "fault": {"code": "WATCHDOG_TIMEOUT", "message": "lease", "recoverable": True},
            "previous_action": {"id": 3, "kind": "move_to", "state": "done"},
            "residual_torques": [0.1, 0, 0, 0, 0, 0],
        }
    )
    assert obs.fault_code == "WATCHDOG_TIMEOUT" and obs.recoverable
    assert obs.previous_action.id == 3 and obs.previous_action.terminal
    assert obs.residual_torques[0] == 0.1
    old = ArmStatus.from_dict({"safety_state": "Armed", "epoch_id": 1})
    assert old.fault is None and not old.recoverable and old.residual_torques is None


def test_action_completes_from_previous_action(armd):
    """With decimated observations the terminal record may only be visible as previous_action."""
    armd.devplat = True
    with open_arm(armd) as arm:
        s = arm.session
        orig = armd.complete_action

        def complete_and_replace():
            orig()
            armd.prev_action, armd.prev_action_at = dict(armd.action), time.monotonic()
            armd.action = {**armd.action, "id": armd.action["id"] + 1, "kind": "gripper", "state": "running"}
            armd.next_action_id += 1

        armd.complete_action = complete_and_replace
        res = s.move_pose([0.25, 0.0, 0.2])
        assert res.action.kind == "move_to" and res.action.state == "done"
        armd.complete_action = orig
        armd.action = None


def test_base_url(monkeypatch):
    monkeypatch.delenv("REBOT_HOST", raising=False)
    assert _base_url(None, None) == "http://127.0.0.1:8765"
    assert _base_url("192.168.42.1", None) == "http://192.168.42.1:8765"
    assert _base_url("10.0.0.2:9000", None) == "http://10.0.0.2:9000"
    assert _base_url("10.0.0.2", 9000) == "http://10.0.0.2:9000"
    assert _base_url("http://h:1/", None) == "http://h:1"
    monkeypatch.setenv("REBOT_HOST", "10.1.1.1")
    assert _base_url(None, None) == "http://10.1.1.1:8765"


def test_unreachable_armd_is_a_clear_error():
    with pytest.raises(ArmConnectionError) as info:
        Arm("127.0.0.1", port=9, timeout=1.0)
    assert "cannot reach armd at http://127.0.0.1:9" in str(info.value)


def test_lease_busy(armd):
    with open_arm(armd):
        with pytest.raises(RebotError) as info:
            open_arm(armd)
        assert info.value.code == "LEASE_BUSY"
    assert armd.positions[:6] == pytest.approx(HOME[:6], abs=0.01)


def test_place_pose_and_position(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        arm.save_pose("bin_a")
        res = arm.place(pose="bin_a", approach_height_m=0.1, speed_scale=0.5)
        assert res.ok and res.action.kind == "place"
        assert armd.action_bodies[-1] == (
            "/api/v1/action/place",
            {"pose": "bin_a", "approach_height_m": 0.1, "release": True, "retreat": True, "speed_scale": 0.5},
        )
        arm.place(position=(0.2, 0.1, 0.05), rpy=(0, 1.5, 0), release=False, retreat=False)
        body = armd.action_bodies[-1][1]
        assert body["position"] == [0.2, 0.1, 0.05] and body["orientation"] == {"rpy": [0.0, 1.5, 0.0]}
        assert body["release"] is False and body["retreat"] is False
        with pytest.raises(ArmCommandRejected) as info:
            arm.place(pose="nope")
        assert info.value.code == "POSE_NOT_FOUND"
        with pytest.raises(ValueError):
            arm.place()
        with pytest.raises(ValueError):
            arm.place(pose="bin_a", position=(0, 0, 0))
        with pytest.raises(ValueError):
            arm.place(pose="bin_a", approach_height_m=0.5)


def test_place_unsupported(armd):
    with open_arm(armd) as arm:  # capabilities version 5
        with pytest.raises(UnsupportedError):
            arm.place(position=(0.2, 0, 0.1))
    armd.devplat = True
    armd.place_cap = False
    with open_arm(armd) as arm:
        with pytest.raises(UnsupportedError):
            arm.place(pose="x")


def test_sigterm_handler_installed_and_restored(armd):
    import signal

    before = signal.getsignal(signal.SIGTERM)
    arm = open_arm(armd)
    handler = signal.getsignal(signal.SIGTERM)
    assert handler is not before
    with pytest.raises(SystemExit) as info:
        handler(signal.SIGTERM, None)
    assert info.value.code == 143
    arm.close()
    assert signal.getsignal(signal.SIGTERM) is before
    with open_arm(armd, handle_sigterm=False):
        assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_not_installed_off_main_thread(armd):
    import signal
    import threading

    before = signal.getsignal(signal.SIGTERM)
    out = []

    def run():
        with open_arm(armd):
            out.append(signal.getsignal(signal.SIGTERM) is before)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert out == [True]
