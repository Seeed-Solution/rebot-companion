"""Collision guard: parsing, wait_until_clear, configuration, readable errors (capability collision_guard)."""

from __future__ import annotations

import threading
import time

import pytest
from fake_armd import open_arm

from rebot import ArmCommandRejected, ArmFaultError, MotionTimeout, UnsupportedError
from rebot.client import ArmStatus
from rebot.errors import ActionFailed


def test_status_parses_collision():
    base = {"safety_state": "Armed"}
    assert ArmStatus.from_dict(base).collision is None
    c = {"active": True, "joint": 2, "residual_nm": 7.1, "enabled": True}
    assert ArmStatus.from_dict({**base, "collision": c}).collision == c


def test_gated_on_capability(armd):
    with open_arm(armd) as arm:  # armd without devplat
        with pytest.raises(UnsupportedError):
            arm.set_collision_guard(enabled=True)
        with pytest.raises(UnsupportedError):
            arm.collision_guard()
        assert arm.state().collision is None
        assert arm.wait_until_clear(timeout_s=1).collision is None  # nothing to wait for
    assert not any("collision" in c for c in armd.commands)


def test_set_and_read_guard(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        assert arm.state().collision is None  # off by default
        cfg = arm.set_collision_guard(enabled=True, sensitivity="high")
        assert cfg["enabled"] is True and cfg["sensitivity"] == "high"
        assert cfg["thresholds_nm"][0] == pytest.approx(4.2)
        assert armd.collision_bodies == [{"enabled": True, "sensitivity": "high"}]
        assert arm.collision_guard()["sensitivity"] == "high"
        with pytest.raises(ValueError):
            arm.set_collision_guard(sensitivity="extreme")


def _move_in_thread(arm, out):
    def run():
        try:
            out.append(arm.move_to((0.25, 0.0, 0.25)))
        except Exception as exc:  # noqa: BLE001
            out.append(exc)

    t = threading.Thread(target=run)
    t.start()
    return t


def test_collision_during_motion_then_auto_resume(armd):
    armd.devplat = True
    armd.action_duration = 1.0
    with open_arm(armd) as arm:
        arm.set_collision_guard(enabled=True)
        out: list = []
        t = _move_in_thread(arm, out)
        time.sleep(0.3)
        armd.trip_collision(joint=2, residual_nm=7.1, resume_after_s=0.4)
        t.join(5)
        err = out[0]
        assert isinstance(err, ActionFailed) and err.code == "COLLISION"
        assert "joint J2 blocked (7.1 N·m)" in err.message and "resumes by itself" in err.message
        with pytest.raises(ArmCommandRejected) as rej:
            arm.move_joints(list(armd.positions))
        assert rej.value.code == "COLLISION_HOLD"
        st = arm.wait_until_clear(timeout_s=5)
        assert not (st.collision or {}).get("active")
        arm.home()  # accepted again


def test_wait_until_clear_times_out(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        arm.set_collision_guard(enabled=True)
        armd.trip_collision(resume_after_s=30)
        with pytest.raises(MotionTimeout):
            arm.wait_until_clear(timeout_s=0.5)


def test_escalation_raises_fault_with_collision_code(armd):
    armd.devplat = True
    with open_arm(armd) as arm:
        arm.set_collision_guard(enabled=True)
        for _ in range(3):
            armd.trip_collision(resume_after_s=0.05)
        armd.trip_collision(resume_after_s=0.05)  # 4th trip within the window
        with pytest.raises(ArmFaultError) as err:
            arm.wait_until_clear(timeout_s=3)
        assert err.value.code == "COLLISION"
        with pytest.raises(ArmFaultError) as rec:
            arm.recover()
        assert "not recoverable" in rec.value.message
