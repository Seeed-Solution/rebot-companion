"""Tests for gripper calibration SDK and CLI against a real armd mock."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rebot import (
    ActionFailed,
    Arm,
    ArmdClient,
    Detection,
    DetectorUnavailable,
    RebotError,
)
from rebot_companion.gripper_calib import (
    check_monotonic,
    compute_fractions,
    fit_opening_curve,
    parse_caliper_input,
)
from rebot_companion.gripper_calib import (
    main as cli_main,
)

# Runs only when ARMD_BIN points to an armd build with the gripper calibration API (rebot-armd >= feat/grasp-borrow),
# as in test_integration_armd.py; ARMD_PROFILE overrides the B601-DM profile path.
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
def armd_mock_server(tmp_path_factory) -> Iterator[dict]:
    profile = Path(os.environ.get("ARMD_PROFILE") or DEFAULT_PROFILE)
    if not profile.is_file():
        pytest.skip(f"Profile not found at {profile}")

    work = tmp_path_factory.mktemp("armd_calib")
    port = _free_port()
    calib_file = work / "gripper_calib.json"
    env = dict(
        os.environ,
        PORT=str(port),
        ARMD_LOG_DIR="off",
        ARMD_BLACKBOX_DIR="off",
        ARMD_GRASP_SETUP=str(work / "grasp_setup.json"),
        ARMD_POSES=str(work / "poses.json"),
        ARMD_KEEPOUT=str(work / "keepout.json"),
        ARMD_GRIPPER_CALIB=str(calib_file),
        ARMD_DETECTOR_URL="http://127.0.0.1:9",  # nothing listens: detect() fails cleanly with DetectorUnavailable
    )
    log = open(work / "armd.log", "wb")
    proc = subprocess.Popen(
        [ARMD_BIN, "--follower", "mock", "--config", str(profile)],
        env=env,
        cwd=work,
        stdout=log,
        stderr=log,
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

    yield {"url": url, "calib_file": calib_file, "work": work}

    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


# ==============================================================================
# Pure unit tests
# ==============================================================================


def test_pure_compute_fractions():
    assert compute_fractions(6) == [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    assert compute_fractions(2) == [1.0, 0.0]
    assert compute_fractions(1) == [1.0]
    assert compute_fractions(3) == [1.0, 0.5, 0.0]


def test_pure_parse_caliper_input():
    assert parse_caliper_input("45.2") == ("ok", 45.2)
    assert parse_caliper_input("  0.0  ") == ("ok", 0.0)
    assert parse_caliper_input("200.0") == ("ok", 200.0)
    assert parse_caliper_input("-1.0") == ("invalid", None)
    assert parse_caliper_input("205.0") == ("invalid", None)
    assert parse_caliper_input("s") == ("skip", None)
    assert parse_caliper_input("S") == ("skip", None)
    assert parse_caliper_input("q") == ("abort", None)
    assert parse_caliper_input("Q") == ("abort", None)
    assert parse_caliper_input("abc") == ("invalid", None)
    assert parse_caliper_input("") == ("invalid", None)


def test_pure_check_monotonic():
    # Strictly increasing J7 and strictly decreasing opening
    valid_dec = [[-5.7, 0.08], [-4.0, 0.06], [-2.0, 0.03], [0.0, 0.0]]
    assert check_monotonic(valid_dec) is True

    # Strictly increasing J7 and strictly increasing opening
    valid_inc = [[0.0, 0.0], [2.0, 0.03], [4.0, 0.06], [5.7, 0.08]]
    assert check_monotonic(valid_inc) is True

    # Non-monotonic opening (goes down then up)
    non_mono = [[-5.7, 0.08], [-4.0, 0.03], [-2.0, 0.05], [0.0, 0.0]]
    assert check_monotonic(non_mono) is False

    # Equal openings (delta < 1e-6)
    equal_openings = [[-5.7, 0.08], [-4.0, 0.08], [0.0, 0.0]]
    assert check_monotonic(equal_openings) is False

    # Fewer than 2 points
    assert check_monotonic([[-5.7, 0.08]]) is False
    assert check_monotonic([]) is False

    # Opening out of bounds (> 0.2 m)
    out_of_bounds = [[-5.7, 0.25], [0.0, 0.0]]
    assert check_monotonic(out_of_bounds) is False

    # Non-finite values
    assert check_monotonic([[-5.7, float("nan")], [0.0, 0.0]]) is False


def test_pure_fit_opening_curve():
    # Perfect linear line: w = -0.014 * j7 + 0.0002
    points = [[-5.0, 0.070], [-3.0, 0.042], [-1.0, 0.014], [0.0, 0.0]]
    fit = fit_opening_curve(points)
    assert fit["max_residual_mm"] < 1.0
    assert fit["warn_residual"] is False
    assert fit["max_opening_m"] == pytest.approx(0.070)
    assert fit["suggested_max_grasp_width_m"] == pytest.approx(0.070 - 0.015)

    # Nonlinear data with large residual (> 2mm)
    nonlinear = [[-5.0, 0.080], [-2.5, 0.020], [0.0, 0.0]]
    fit_nl = fit_opening_curve(nonlinear)
    assert fit_nl["max_residual_mm"] > 2.0
    assert fit_nl["warn_residual"] is True


# ==============================================================================
# Integration tests against real armd mock
# ==============================================================================


def test_zero_check_result_fields(armd_mock_server):
    url = armd_mock_server["url"]
    with Arm(url) as arm:
        res = arm.gripper_zero_check(store_offset=False, threshold_rad=0.10)
        assert res.ok is True
        assert res.stop_rad is not None
        assert isinstance(res.stop_rad, float)
        assert res.offset_rad is not None
        assert isinstance(res.offset_rad, float)
        assert res.passed is True
        assert res.pass_ is True

        # Action status fields
        assert res.action is not None
        assert res.action.kind == "gripper_zero_check"
        assert res.action.stop_rad == res.stop_rad
        assert res.action.offset_rad == res.offset_rad
        assert res.action.passed is True


def test_calibration_write_and_read_back(armd_mock_server):
    url = armd_mock_server["url"]
    with Arm(url) as arm:
        # Initial calibration read via GET
        calib0 = arm.gripper_calibration()
        assert isinstance(calib0, dict)

        # Write calibration
        pts = [[-5.7, 0.08], [0.0, 0.0]]
        ack = arm.set_gripper_calibration(points=pts, offset_rad=0.035)
        assert ack.get("persisted") is True

        # Read back via GET
        calib1 = arm.gripper_calibration()
        assert calib1.get("offset_rad") == pytest.approx(0.035)
        assert calib1.get("calibrated") is True
        ret_pts = calib1.get("jaw_opening", {}).get("points")
        assert len(ret_pts) == 2
        assert ret_pts[0][0] == pytest.approx(-5.7 + 0.035)
        assert ret_pts[0][1] == pytest.approx(0.08)

        # Observation field observation.gripper_zero
        time.sleep(0.15)
        st = arm.state()
        assert st.gripper_zero is not None
        assert st.gripper_zero.get("offset_rad") == pytest.approx(0.035)


def test_grasp_dry_run_fields(armd_mock_server):
    url = armd_mock_server["url"]
    with Arm(url) as arm:
        det = Detection(label="cup", score=0.9, box_norm=(0.3, 0.3, 0.5, 0.5))
        reachable, code, message = arm.grasp_dry_run(det)
        assert isinstance(reachable, bool)
        if not reachable:
            assert code is None or isinstance(code, str)
            assert message is None or isinstance(message, str)

        # Also via (label, box)
        reachable2, code2, message2 = arm.grasp_dry_run("cup", box=(0.3, 0.3, 0.5, 0.5))
        assert isinstance(reachable2, bool)


def test_grasp_any_behavior_and_errors(armd_mock_server, monkeypatch):
    url = armd_mock_server["url"]
    with Arm(url) as arm:
        # 1. No detector available (detector proxy unreachable at port 9) -> raises DetectorUnavailable
        with pytest.raises(DetectorUnavailable):
            arm.grasp_any("cup")

        # 2. Detector returns empty list -> raises ActionFailed with IK_UNREACHABLE
        monkeypatch.setattr(arm, "detect", lambda **kwargs: [])
        with pytest.raises(ActionFailed) as exc_info:
            arm.grasp_any("cup")
        assert exc_info.value.code == "IK_UNREACHABLE"

        # 3. Detector returns candidates but none reachable -> raises ActionFailed with last dry_run code
        d1 = Detection(label="cup", score=0.9, box_norm=(0.1, 0.1, 0.2, 0.2))
        d2 = Detection(label="cup", score=0.8, box_norm=(0.2, 0.2, 0.3, 0.3))
        monkeypatch.setattr(arm, "detect", lambda **kwargs: [d1, d2])

        codes = iter([(False, "IK_BRANCH_FLIP", "branch flip"), (False, "WORKSPACE_VIOLATION", "out of bounds")])
        monkeypatch.setattr(arm, "grasp_dry_run", lambda *args, **kwargs: next(codes))

        with pytest.raises(ActionFailed) as exc_info:
            arm.grasp_any("cup")
        assert exc_info.value.code == "WORKSPACE_VIOLATION"
        assert "out of bounds" in exc_info.value.message


def test_cli_flow_stores_and_reads_back(armd_mock_server, monkeypatch):
    url = armd_mock_server["url"]

    # Scripted input sequence:
    # 1. Continue safety check? -> 'y'
    # 2. Store offset? -> 'y'
    # 3. 6 caliper opening points (fraction 1.0 -> 80.0 mm down to 0.0 -> 0.0 mm)
    # 4. Store calibration? -> 'y'
    inputs = iter([
        "y",  # Continue
        "y",  # Store offset
        "80.0",  # point 1 (1.0)
        "64.0",  # point 2 (0.8)
        "48.0",  # point 3 (0.6)
        "32.0",  # point 4 (0.4)
        "16.0",  # point 5 (0.2)
        "0.0",  # point 6 (0.0)
        "y",  # Store calibration
    ])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    ret = cli_main(["--host", url, "--points", "6"])
    assert ret == 0

    # Verify calibration written and read back
    client = ArmdClient(url)
    calib = client.get_gripper_calibration()
    assert calib.get("calibrated") is True
    points = calib.get("jaw_opening", {}).get("points")
    assert len(points) == 6
    assert points[0][1] == pytest.approx(0.080)  # sorted by j7 (-5.70 rad -> 80.0 mm)
    assert points[-1][1] == pytest.approx(0.0)  # -0.00 rad -> 0.0 mm


def test_cli_dry_run_does_not_write(armd_mock_server, monkeypatch):
    url = armd_mock_server["url"]

    inputs = iter([
        "y",  # Continue
        "y",  # Store offset
        "75.0",  # point 1 (1.0)
        "0.0",  # point 2 (0.0)
        "y",  # Store calibration
    ])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    ret = cli_main(["--host", url, "--points", "2", "--dry-run"])
    assert ret == 0

    # The previous 6 points from test_cli_flow_stores_and_reads_back remain intact
    client = ArmdClient(url)
    calib = client.get_gripper_calibration()
    points = calib.get("jaw_opening", {}).get("points")
    assert len(points) == 6
