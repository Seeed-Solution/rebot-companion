"""rebot.Arm against a real armd binary with the mock follower.

Runs only when ``ARMD_BIN`` points to an armd build, e.g.::

    ARMD_BIN=/path/to/target/release/armd ARMD_PROFILE=/path/to/rebot-b601dm.json \
        uv run pytest tests/test_integration_armd.py

``ARMD_PROFILE`` (the B601-DM profile, 7 joints with kinematics) defaults to
``../rebot-armd-public/kits/recamera-b601dm/profile/rebot-b601dm.json``. Without it the mock
follower uses a generic 6-joint profile without Cartesian actions.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from rebot import Arm, ArmCommandRejected, ArmdClient, DetectorUnavailable, RebotError, UnsupportedError

ARMD_BIN = os.environ.get("ARMD_BIN")
pytestmark = pytest.mark.skipif(not ARMD_BIN, reason="set ARMD_BIN to an armd binary to run")

DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2] / "rebot-armd-public/kits/recamera-b601dm/profile/rebot-b601dm.json"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def armd_url(tmp_path_factory):
    profile = Path(os.environ.get("ARMD_PROFILE") or DEFAULT_PROFILE)
    if not profile.is_file():
        pytest.skip(f"B601-DM profile not found at {profile}; set ARMD_PROFILE")
    work = tmp_path_factory.mktemp("armd")
    port = _free_port()
    env = dict(
        os.environ,
        PORT=str(port),
        ARMD_LOG_DIR="off",
        ARMD_BLACKBOX_DIR="off",
        ARMD_GRASP_SETUP=str(work / "grasp_setup.json"),
        ARMD_POSES=str(work / "poses.json"),
        ARMD_KEEPOUT=str(work / "keepout.json"),
        ARMD_DETECTOR_URL="http://127.0.0.1:9",  # nothing listens: detections fail deterministically
    )
    log = open(work / "armd.log", "wb")
    proc = subprocess.Popen(
        [ARMD_BIN, "--follower", "mock", "--config", str(profile)], env=env, cwd=work, stdout=log, stderr=log
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while True:
        try:
            ArmdClient(url, timeout_sec=0.5).get_schema()
            break
        except RebotError:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"armd did not start; log: {(work / 'armd.log').read_text()[-2000:]}")
            time.sleep(0.2)
    yield url
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def _home_tcp() -> list[float]:
    from rebot_companion import kinematics  # host FK (numpy); only to pick a reachable target

    home = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963, 0.0]
    return [float(v) for v in kinematics.tcp_pose_from_joints(home).translation]


def test_connect_enable_move_home_park_release(armd_url):
    schema = ArmdClient(armd_url).get_schema()
    print(f"\narmd capabilities version {schema.capabilities_version}, dof {schema.dof}")
    with Arm(armd_url) as arm:
        obs = arm.state()
        print(f"connected: {obs.safety_state}, torqued={obs.is_torqued}, ws={arm.session.ws_url}")
        assert obs.is_operational and obs.is_torqued
        x, y, z = _home_tcp()
        target = (round(x, 3), round(y + 0.03, 3), round(z - 0.03, 3))
        res = arm.move_to(target, speed=0.1)
        print(f"move_to {target}: {res.action.state} in {res.duration_s:.2f}s")
        assert res.ok and res.action.kind == "move_to"
        res = arm.home()
        print(f"home: {res.action.state if res.action else 'done'} in {res.duration_s:.2f}s")
        assert res.ok
        res = arm.park()
        print(f"park: {res.action.state if res.action else 'done'}, torqued={arm.state().is_torqued}")
        assert not arm.state().is_torqued
    print(f"exit: {arm.exit_outcome}")
    obs = ArmdClient(armd_url).get_observation()
    assert not obs.is_torqued
    raw = ArmdClient(armd_url)._get("/api/v1/observation")
    print(f"after release: safety_state={obs.safety_state}, lease_holder={raw.get('lease_holder')}")
    assert raw.get("lease_holder") in (None, {}, "")


def test_errors_are_readable(armd_url):
    with Arm(armd_url) as arm:
        with pytest.raises(ArmCommandRejected) as info:
            arm.move_to((0.9, 0.0, 0.1))
        print(f"\nunreachable: {info.value}")
        assert info.value.code in ("IK_UNREACHABLE", "IK_BRANCH_FLIP", "WORKSPACE_VIOLATION")
        assert "(0.9, 0, 0.1)" in str(info.value)
        with pytest.raises(DetectorUnavailable) as det:
            arm.detect("cup")
        print(f"detect: {det.value}")
        if not arm.schema.has("poses"):
            with pytest.raises(UnsupportedError) as uns:
                arm.poses()
            print(f"poses: {uns.value}")
        else:
            arm.save_pose("it_pose")
            assert "it_pose" in [p["name"] for p in arm.poses()]
            arm.move_to_pose("it_pose")
            arm.delete_pose("it_pose")


def test_exit_parks_a_torqued_arm(armd_url):
    with Arm(armd_url) as arm:
        arm.move_relative(dz=0.02)
        assert arm.state().is_torqued
    print(f"\nexit: {arm.exit_outcome}")
    assert arm.exit_outcome == "parked (torque off)"
    assert not ArmdClient(armd_url).get_observation().is_torqued
