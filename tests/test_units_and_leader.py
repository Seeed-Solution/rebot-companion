"""Unit tests for Stage 4: units conversion, recorder/policy unit integration,
PC-side leader teleop bridge, and auto_yaw polish.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rebot.client import ArmdClient, ArmSession, ArmStatus, JointLimit, ObservationSchema, _GRIPPER_UNSET
from rebot_companion import units
from rebot_companion.leader_teleop import (
    DEFAULT_WRIST_ROLL_LIMITS,
    HARD_MAX_GAP_RAD,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    FeetechSerialBus,
    GripperCalib,
    JointCalib,
    LeaderCalibration,
    LeaderError,
    LeaderTeleopBridge,
    LeaderTransport,
    PosixSerialTransport,
    compute_checksum,
    decode_packet,
    encode_packet,
)
from rebot_companion.lerobot_recorder import LeRobotRecorder
from rebot_companion.mcp_server import Controller
from rebot_companion.policy_runner import PolicyRunner, RunnerError, parse_policy, pretrained_info

# --------------------------------------------------------------------------- units.py tests


def test_units_rad_passthrough():
    vals = [0.1, -0.5, 1.2, -2.0, 0.0, 0.7]
    assert units.from_armd(vals, "rad") == vals
    assert units.to_armd(vals, "rad") == vals


def test_units_so101_norm_6dof_roundtrip():
    # 6-DOF: 5 arm joints + 1 gripper joint (index 5)
    limits = [
        JointLimit("j1", -3.14, 3.14, 2.0, 5.0),
        JointLimit("j2", -1.57, 1.57, 2.0, 5.0),
        JointLimit("j3", -2.0, 2.0, 2.0, 5.0),
        JointLimit("j4", -1.8, 1.8, 2.0, 5.0),
        JointLimit("j5", -2.74, 2.74, 2.0, 5.0),
        JointLimit("gripper", 0.0, -1.5, 2.0, 5.0),
    ]
    gripper_j = 5
    closed_rad = 0.0
    open_rad = -1.5

    # Test center/zeros
    arm_zeros = [0.0, 0.0, 0.0, 0.0, 0.0, closed_rad]
    norm_zeros = units.from_armd(
        arm_zeros, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    # Symmetric limits -> 0.0 rad maps to 0.0
    for v in norm_zeros[:5]:
        assert abs(v) < 1e-4
    # Closed gripper maps to 0.0
    assert abs(norm_zeros[5] - 0.0) < 1e-4

    # Roundtrip back to rad
    recovered_zeros = units.to_armd(
        norm_zeros, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    for orig, rec in zip(arm_zeros, recovered_zeros):
        assert abs(orig - rec) < 1e-4

    # Test full open gripper
    arm_open = [1.0, -0.5, 0.8, -1.2, 2.0, open_rad]
    norm_open = units.from_armd(
        arm_open, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    assert abs(norm_open[5] - 100.0) < 1e-4

    recovered_open = units.to_armd(
        norm_open, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    for orig, rec in zip(arm_open, recovered_open):
        assert abs(orig - rec) < 1e-4


def test_units_b601_dm_inverted_gripper_roundtrip():
    # 7-DOF B601-DM: gripper at index 6, closed=0.0, open=-5.7 (open_rad < closed_rad)
    limits = [JointLimit(f"j{i}", -3.14, 3.14, 2.0, 5.0) for i in range(7)]
    gripper_j = 6
    closed_rad = 0.0
    open_rad = -5.7

    rad_vals = [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, -5.7]
    norm_vals = units.from_armd(
        rad_vals, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    assert abs(norm_vals[6] - 100.0) < 1e-4

    rec_vals = units.to_armd(
        norm_vals, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    for orig, rec in zip(rad_vals, rec_vals):
        assert abs(orig - rec) < 1e-4


def test_units_b601_inverted_gripper_fallback_from_limits():
    # Inverted B601 limits [-5.7, 0.0] with closed_rad=None, open_rad=None
    limits = [JointLimit(f"j{i}", -3.14, 3.14, 2.0, 5.0) for i in range(6)]
    limits.append(JointLimit("gripper", -5.7, 0.0, 2.0, 5.0))
    gripper_j = 6

    # When closed_rad=None and open_rad=None, should resolve closed=0.0 and open=-5.7
    rad_closed = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0]
    norm_closed = units.from_armd(
        rad_closed, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=None, open_rad=None
    )
    assert abs(norm_closed[6] - 0.0) < 1e-4

    rad_open = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -5.7]
    norm_open = units.from_armd(
        rad_open, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=None, open_rad=None
    )
    assert abs(norm_open[6] - 100.0) < 1e-4

    rec_open = units.to_armd(
        norm_open, "so101_norm", limits=limits, gripper_joint=gripper_j, closed_rad=None, open_rad=None
    )
    for orig, rec in zip(rad_open, rec_open):
        assert abs(orig - rec) < 1e-4

    # Also test when limits is None, gripper_joint=6 fallback
    norm_open_no_limits = units.from_armd(
        rad_open, "so101_norm", limits=None, gripper_joint=gripper_j, closed_rad=None, open_rad=None
    )
    assert abs(norm_open_no_limits[6] - 100.0) < 1e-4


def test_units_deg_roundtrip():
    limits = [JointLimit(f"j{i}", -math.pi, math.pi, 2.0, 5.0) for i in range(6)]
    gripper_j = 5
    closed_rad = 0.0
    open_rad = 1.5

    rad_vals = [0.0, math.pi / 2, -math.pi / 4, math.pi, -math.pi, 0.75]
    deg_vals = units.from_armd(
        rad_vals, "deg", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    assert abs(deg_vals[0] - 0.0) < 1e-4
    assert abs(deg_vals[1] - 90.0) < 1e-4
    assert abs(deg_vals[2] - (-45.0)) < 1e-4
    assert abs(deg_vals[3] - 180.0) < 1e-4
    assert abs(deg_vals[4] - (-180.0)) < 1e-4
    assert abs(deg_vals[5] - 50.0) < 1e-4  # 0.75 is mid-point of [0.0, 1.5]

    rec_rad = units.to_armd(
        deg_vals, "deg", limits=limits, gripper_joint=gripper_j, closed_rad=closed_rad, open_rad=open_rad
    )
    for orig, rec in zip(rad_vals, rec_rad):
        assert abs(orig - rec) < 1e-4


def test_resolve_unit_mode():
    assert units.resolve_unit_mode("rad") == "rad"
    assert units.resolve_unit_mode("so101_norm") == "so101_norm"
    assert units.resolve_unit_mode("deg") == "deg"
    with pytest.raises(ValueError, match="unsupported unit mode"):
        units.resolve_unit_mode("invalid")

    # Auto resolution with explicit metadata
    assert units.resolve_unit_mode("auto", {"unit_mode": "so101_norm"}) == "so101_norm"
    assert units.resolve_unit_mode("auto", {"unit_mode": "deg"}) == "deg"
    assert units.resolve_unit_mode("auto", {"rebot_unit_mode": "so101_norm"}) == "so101_norm"
    assert units.resolve_unit_mode("auto", {}) == "rad"
    assert units.resolve_unit_mode("auto", None) == "rad"

    # Non-radian stats without explicit unit_mode must raise ValueError
    stats_norm = {
        "stats": {
            "observation.state": {
                "max": [95.0, 80.0, 70.0, 50.0, 40.0, 100.0],
                "min": [-95.0, -80.0, -70.0, -50.0, -40.0, 0.0],
            }
        }
    }
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' from non-radian stats without 'unit_mode'"):
        units.resolve_unit_mode("auto", stats_norm)

    stats_deg = {
        "stats": {
            "observation.state": {
                "max": [180.0, 90.0, 120.0, 150.0, 180.0, 100.0],
                "min": [-180.0, -90.0, -120.0, -150.0, -180.0, 0.0],
            }
        }
    }
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' from non-radian stats without 'unit_mode'"):
        units.resolve_unit_mode("auto", stats_deg)

    # When explicit unit_mode is provided alongside stats, it is respected
    assert units.resolve_unit_mode("auto", {"unit_mode": "so101_norm", **stats_norm}) == "so101_norm"
    assert units.resolve_unit_mode("auto", {"unit_mode": "deg", **stats_deg}) == "deg"

    # Radian range stats default to rad
    stats_rad = {
        "stats": {
            "observation.state": {
                "max": [3.14, 1.57, 2.0, 1.8, 2.74, 0.0],
                "min": [-3.14, -1.57, -2.0, -1.8, -2.74, -1.5],
            }
        }
    }
    assert units.resolve_unit_mode("auto", stats_rad) == "rad"

    # Action stats with non-radian values also trigger the guard
    stats_action_non_rad = {
        "stats": {
            "action": {
                "max": [50.0, 10.0, 20.0, 5.0, 0.0, 100.0],
                "min": [-50.0, -10.0, -20.0, -5.0, 0.0, 0.0],
            }
        }
    }
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' from non-radian stats without 'unit_mode'"):
        units.resolve_unit_mode("auto", stats_action_non_rad)


# --------------------------------------------------------------------------- lerobot_recorder.py tests


def test_recorder_unit_mode_so101_norm(tmp_path: Path):
    root = tmp_path / "rec_norm"
    rec = LeRobotRecorder(
        root=root,
        repo_id="test/so101_dataset",
        fps=10,
        camera_url=None,
        armd_host=None,
        task="test task",
        unit_mode="so101_norm",
    )
    # Set mock schema / DOF for 6-DOF SO-ARM101
    rec.arm_dof = 5
    rec.gripper_joint = 5
    rec.profile_id = "so-arm101"
    rec.closed_rad = 0.0
    rec.open_rad = -1.5
    rec.limits = [
        JointLimit(f"j{i}", -3.14, 3.14, 2.0, 5.0) for i in range(5)
    ] + [JointLimit("gripper", 0.0, -1.5, 2.0, 5.0)]

    rec.start_episode(sample=False)
    # Add frame with zero arm angles and closed gripper (0.0 rad)
    rec.add_frame([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Add frame with open gripper (-1.5 rad)
    rec.add_frame([0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [0.0, 0.0, 0.0, 0.0, 0.0, -1.5])
    rec.stop_episode()
    rec.save()

    info_path = root / "meta" / "info.json"
    assert info_path.is_file()
    meta = json.loads(info_path.read_text())
    assert meta["unit_mode"] == "so101_norm"
    assert meta["profile_id"] == "so-arm101"
    assert meta["arm_dof"] == 5
    assert meta["gripper_joint"] == 5

    # Check that data contains normalized values (0..100 for gripper)
    import pyarrow.parquet as pq

    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    states = table.column("observation.state").to_pylist()
    assert len(states) == 2
    # First frame: gripper was 0.0 rad -> 0.0 norm
    assert abs(states[0][5] - 0.0) < 1e-3
    # Second frame: gripper was -1.5 rad -> 100.0 norm
    assert abs(states[1][5] - 100.0) < 1e-3


# --------------------------------------------------------------------------- policy_runner.py tests


def test_policy_runner_so101_6dof_so101_norm(tmp_path: Path):
    from rebot_companion.policy_runner import Policy

    class Mock6DofPolicy(Policy):
        action_dim = 6
        info = {"unit_mode": "so101_norm"}

        def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
            # Observes normalized values and returns normalized action
            st = obs["observation.state"]
            # Action: echo state with small offset
            return np.array(st, dtype=np.float64)

    mock_client = MagicMock()
    mock_client.get_safety_describe.return_value = {"device": {"follower": "mock"}}
    schema_dict = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "orientation_dof": 2,
        "profile_id": "so-arm101",
        "gripper_policy": {"closed_rad": 0.0, "open_rad": -1.5},
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(5)
        ]
        + [
            {
                "name": "gripper",
                "min_position": -1.5,
                "max_position": 0.0,
                "max_velocity": 2.0,
                "max_acceleration": 5.0,
            }
        ],
    }
    mock_res = MagicMock()
    mock_res.json.return_value = schema_dict
    mock_client.request.return_value = mock_res

    pol = Mock6DofPolicy()
    runner = PolicyRunner(mock_client, pol, camera=None, action_units="auto", state_units="auto", fps=10)
    runner._read_arm()

    assert runner.dof == 6
    assert runner.gripper_joint == 5
    assert runner.action_units == "so101_norm"
    assert runner.state_units == "so101_norm"
    assert runner.closed_rad == 0.0
    assert runner.open_rad == -1.5

    # Check observation conversion
    mock_status = MagicMock()
    mock_status.positions = [0.0, 0.0, 0.0, 0.0, 0.0, -1.5]
    obs = runner._observation(mock_status)
    # Gripper at -1.5 rad converts to 100.0 norm
    assert abs(obs["observation.state"][5] - 100.0) < 1e-4


def test_policy_runner_dof_mismatch():
    mock_client = MagicMock()
    mock_client.get_safety_describe.return_value = {"device": {"follower": "mock"}}
    # Arm schema has 6 DOF
    schema_dict = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    mock_res = MagicMock()
    mock_res.json.return_value = schema_dict
    mock_client.request.return_value = mock_res

    from rebot_companion.policy_runner import Policy

    class Mock7DofPolicy(Policy):
        action_dim = 7

    pol = Mock7DofPolicy()
    runner = PolicyRunner(mock_client, pol, camera=None, fps=10)
    with pytest.raises(RunnerError, match="policy action dimension \\(7\\) does not match arm DOF \\(6\\)"):
        runner._read_arm()


# --------------------------------------------------------------------------- leader_teleop.py tests


class MockLeaderTransport(LeaderTransport):
    """In-memory mock transport for testing Feetech serial communication."""

    def __init__(self) -> None:
        self.written = bytearray()
        self.to_read = bytearray()

    def write(self, data: bytes) -> None:
        self.written.extend(data)

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        chunk = self.to_read[:n]
        self.to_read = self.to_read[n:]
        return bytes(chunk)

    def flush(self) -> None:
        pass


def test_feetech_packet_encode_decode():
    # Encode ping packet for servo 1: ID=1, Length=2, Instruction=0x01
    pkt = encode_packet(1, 0x01, [])
    # 0xFF, 0xFF, 0x01, 0x02, 0x01, checksum: ~(1 + 2 + 1) = ~4 = 0xFB
    assert pkt == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])

    # Decode response packet: 0xFF, 0xFF, ID=1, Length=4, Error=0x00, Param=0x12, 0x34, Checksum
    params = [0x12, 0x34]
    resp_body = [1, 4, 0x00] + params
    chk = compute_checksum(resp_body)
    resp = bytes([0xFF, 0xFF] + resp_body + [chk])

    sid, err, dec_params, rem = decode_packet(resp)
    assert sid == 1
    assert err == 0
    assert dec_params == bytes(params)
    assert rem == b""

    # Test checksum failure
    corrupt_resp = bytes([0xFF, 0xFF] + resp_body + [(chk + 1) & 0xFF])
    sid, err, dec_params, rem = decode_packet(corrupt_resp)
    assert sid is None


def test_leader_calibration():
    # 5 arm joints + 1 gripper
    calib = LeaderCalibration(
        joints=[JointCalib(center=2048, sign=1) for _ in range(5)],
        gripper=GripperCalib(closed_tick=2048, open_tick=3072),
        servo_ids=[1, 2, 3, 4, 5, 6],
    )
    # At center ticks: arm angles should be 0.0 rad, gripper closed (0.0)
    ticks = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
    rads = calib.ticks_to_rad(ticks)
    for r in rads:
        assert abs(r) < 1e-4

    # Test wrist_roll clamping (servo 5, index 4): 4096 is +pi rad, should clamp to 2.74
    # Default follower_open_rad is 1.5, so gripper at 3072 ticks is fully open (+1.5 rad)
    ticks_wrist = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 4095, 6: 3072}
    rads_wrist = calib.ticks_to_rad(ticks_wrist)
    assert abs(rads_wrist[4] - DEFAULT_WRIST_ROLL_LIMITS[1]) < 1e-4
    assert abs(rads_wrist[5] - 1.5) < 1e-4

    # Explicitly test follower_open_rad=-1.5 backward compatibility
    rads_wrist_legacy = calib.ticks_to_rad(ticks_wrist, follower_closed_rad=0.0, follower_open_rad=-1.5)
    assert abs(rads_wrist_legacy[4] - DEFAULT_WRIST_ROLL_LIMITS[1]) < 1e-4
    assert abs(rads_wrist_legacy[5] - (-1.5)) < 1e-4


def test_leader_teleop_bridge_lifecycle_and_ramp():
    from rebot_companion.leader_teleop import FeetechSerialBus

    transport = MockLeaderTransport()
    bus = FeetechSerialBus(transport)
    mock_session = MagicMock()
    mock_obs = MagicMock()
    # Follower starts at zero angles
    mock_obs.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mock_obs.safety_state = "Armed"
    mock_obs.is_stopped = False
    mock_obs.is_torqued = True
    mock_obs.is_operational = True
    mock_obs.collision = None
    mock_session.wait_for_observation.return_value = mock_obs
    mock_session.latest_observation.return_value = mock_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
        "gripper_policy": {"closed_rad": 0.0, "open_rad": -1.5},
    })

    # Prepare response for bus.read_positions_sync
    # Pack positions for 6 servos: 2048 for all -> 0.0 rad
    sync_resp = bytearray()
    for sid in range(1, 7):
        body = [sid, 4, 0x00, 0x00, 0x08]  # 0x0800 = 2048
        chk = compute_checksum(body)
        sync_resp.extend([0xFF, 0xFF] + body + [chk])
    transport.to_read = sync_resp * 100

    bridge = LeaderTeleopBridge(
        bus,
        mock_session,
        schema,
        fps=50,
        ramp_seconds=0.2,  # 10 steps
        max_gap_rad=0.5,
        park_on_exit=True,
    )

    bridge.align_and_ramp()
    assert bridge.frames_sent == 10
    assert mock_session.servo.call_count == 10

    # Test shutdown
    bridge.close()
    mock_session.unload.assert_called_once_with(wait=True, timeout=15.0)


def test_read_positions_sync_partial_packet_no_hang():
    from rebot_companion.leader_teleop import FeetechSerialBus

    transport = MockLeaderTransport()
    bus = FeetechSerialBus(transport)
    # Feed truncated 5-byte partial frame
    transport.to_read = bytearray(b"\xff\xff\x01\x04\x00")
    # Must terminate immediately without hanging and fallback cleanly
    positions = bus.read_positions_sync([1, 2, 3, 4, 5, 6])
    assert positions == {}


def test_leader_bridge_rejects_missing_servos():
    bus = MagicMock()
    # Missing servo 6
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048}
    mock_session = MagicMock()
    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })
    bridge = LeaderTeleopBridge(bus, mock_session, schema)
    with pytest.raises(LeaderError, match=r"missing leader readings for servo IDs \[6\]"):
        bridge.read_leader_target()


def test_leader_bridge_arms_idle_follower_before_ramp():
    bus = MagicMock()
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}

    idle_obs = MagicMock()
    idle_obs.positions = [0.0] * 6
    idle_obs.safety_state = "Idle"
    idle_obs.is_stopped = False
    idle_obs.is_torqued = False
    idle_obs.is_operational = False
    idle_obs.collision = None

    armed_obs = MagicMock()
    armed_obs.positions = [0.0] * 6
    armed_obs.safety_state = "Armed"
    armed_obs.is_stopped = False
    armed_obs.is_torqued = True
    armed_obs.is_operational = True
    armed_obs.collision = None

    mock_session = MagicMock()
    mock_session.wait_for_observation.side_effect = [idle_obs, armed_obs]
    mock_session.latest_observation.return_value = armed_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    bridge = LeaderTeleopBridge(
        bus,
        mock_session,
        schema,
        fps=10,
        ramp_seconds=0.2,  # 2 steps
        max_gap_rad=0.5,
    )
    bridge.align_and_ramp()
    mock_session.torque.assert_called_once_with(True)
    assert mock_session.servo.call_count == 2


def test_posix_serial_custom_baud_darwin(monkeypatch):
    import struct
    import termios
    import tty

    mock_fd = 42
    monkeypatch.setattr("os.open", lambda *a, **k: mock_fd)
    monkeypatch.setattr("os.close", lambda fd: None)
    monkeypatch.setattr("termios.tcgetattr", lambda fd: [0, 0, 0, 0, 0, 0, [0] * 32])
    monkeypatch.setattr("termios.tcsetattr", lambda fd, when, mode: None)
    monkeypatch.delattr("termios.B1000000", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")

    ioctl_calls = []

    def fake_ioctl(fd, request, arg):
        ioctl_calls.append((fd, request, bytes(arg) if isinstance(arg, (bytes, bytearray)) else arg))
        return 0

    monkeypatch.setattr("fcntl.ioctl", fake_ioctl)

    # 1000000 baud is not a standard B1000000 in termios
    transport = PosixSerialTransport("/dev/fake_port", baud=1000000)
    assert transport.fd == mock_fd

    # Verify IOSSIOSPEED (0x80085402) called with packed 8-byte baud
    expected_buf = struct.pack("L", 1000000)
    assert len(ioctl_calls) == 1
    assert ioctl_calls[0][0] == mock_fd
    assert ioctl_calls[0][1] == 0x80085402
    assert ioctl_calls[0][2] == expected_buf

    # Test error handling on darwin ioctl failure
    def failing_ioctl(fd, request, arg):
        raise OSError("ioctl failure")

    monkeypatch.setattr("fcntl.ioctl", failing_ioctl)
    with pytest.raises(LeaderError, match=r"failed to set custom baud 1000000 on macOS for /dev/fake_port"):
        PosixSerialTransport("/dev/fake_port", baud=1000000)

    # Test unsupported platform
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(LeaderError, match=r"custom baud 1000000 is unsupported on win32 without pyserial"):
        PosixSerialTransport("/dev/fake_port", baud=1000000)

    # Test linux failure raising LeaderError
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("fcntl.ioctl", failing_ioctl)
    with pytest.raises(LeaderError, match=r"failed to set custom baud 1000000 on Linux for /dev/fake_port"):
        PosixSerialTransport("/dev/fake_port", baud=1000000)


def test_leader_bridge_rejects_7dof_before_torque_or_servo():
    bus = MagicMock()
    mock_session = MagicMock()
    mock_obs = MagicMock()
    mock_obs.is_stopped = False
    mock_obs.collision = None
    mock_obs.positions = [0.0] * 7  # 7-DOF positions
    mock_obs.is_torqued = False
    mock_obs.is_operational = False
    mock_session.wait_for_observation.return_value = mock_obs

    schema = ObservationSchema.from_dict({
        "dof": 7,
        "arm_dof": 6,
        "gripper_joint": 6,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(7)
        ],
    })

    bridge = LeaderTeleopBridge(bus, mock_session, schema)

    with pytest.raises(LeaderError, match="expected 6 DOF for SO-ARM101"):
        bridge.align_and_ramp()

    # Crucial safety check: torque and servo MUST NOT have been called
    mock_session.torque.assert_not_called()
    mock_session.servo.assert_not_called()
    bus.read_positions_sync.assert_not_called()


def test_leader_teleop_main_rejects_7dof_before_session(monkeypatch, capsys):
    from rebot_companion import leader_teleop

    mock_transport = MagicMock()
    monkeypatch.setattr(leader_teleop, "open_serial_transport", lambda *a, **k: mock_transport)

    mock_schema = ObservationSchema.from_dict({
        "dof": 7,
        "arm_dof": 6,
        "gripper_joint": 6,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(7)
        ],
    })
    mock_client = MagicMock()
    mock_client.get_schema.return_value = mock_schema
    monkeypatch.setattr(leader_teleop, "ArmdClient", lambda *a, **k: mock_client)

    session_created = False

    def fake_arm_session(*a, **k):
        nonlocal session_created
        session_created = True
        return MagicMock()

    monkeypatch.setattr(leader_teleop, "ArmSession", fake_arm_session)

    rc = leader_teleop.main(["--leader-port", "/dev/null", "--arm-host", "127.0.0.1"])
    assert rc == 1
    assert not session_created
    mock_transport.close.assert_called_once()
    err = capsys.readouterr().err
    assert "unsupported arm schema (dof=7, arm_dof=6)" in err


def test_recorder_fetch_schema_normalizes_plain_host(tmp_path, monkeypatch):
    from rebot._http import host_url

    # Verify host_url argument order
    assert host_url("http", "192.168.16.1", 8765) == "http://192.168.16.1:8765"

    recorded_client_hosts = []
    recorded_requests = []

    class FakeArmdClient:
        def __init__(self, host_or_url: str):
            recorded_client_hosts.append(host_or_url)

        def get_schema(self):
            return ObservationSchema.from_dict({
                "dof": 6,
                "arm_dof": 5,
                "gripper_joint": 5,
                "limits": [
                    {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
                    for i in range(6)
                ],
            })

    def fake_request(method, url, **kwargs):
        recorded_requests.append((method, url))
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"points": []}
        return resp

    monkeypatch.setattr("rebot.client.ArmdClient", FakeArmdClient)
    monkeypatch.setattr("rebot._http.request", fake_request)

    recorder = LeRobotRecorder(tmp_path / "rec", "test/repo", armd_host="192.168.16.1", fps=10)
    recorder._fetch_schema()

    assert recorded_client_hosts == ["http://192.168.16.1:8765"]


def test_lerobot_recorder_fetch_schema_7dof_fallback(monkeypatch, tmp_path):
    # Joint 6 has B601 limits [-5.7, 0.0]
    limits_b601 = [
        {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
        for i in range(6)
    ] + [{"name": "gripper", "min_position": -5.7, "max_position": 0.0, "max_velocity": 2.0, "max_acceleration": 5.0}]

    mock_schema = ObservationSchema.from_dict({
        "dof": 7,
        "arm_dof": 6,
        "gripper_joint": 6,
        "joint_names": [f"j{i}" for i in range(7)],
        "gripper_policy": None,
        "limits": limits_b601,
    })

    current_schema = mock_schema

    class FakeArmdClient:
        def __init__(self, host_url):
            pass

        def get_schema(self):
            return current_schema

    monkeypatch.setattr("rebot.client.ArmdClient", FakeArmdClient)

    rec = LeRobotRecorder(tmp_path / "rec", "test/repo", armd_host="192.168.16.1", fps=10)
    rec._fetch_schema()
    assert rec.closed_rad == 0.0
    assert rec.open_rad == -5.7

    # Partial gripper_policy with closed_rad only on B601 resolves open_rad from limits
    mock_schema_partial = ObservationSchema.from_dict({
        "dof": 7,
        "arm_dof": 6,
        "gripper_joint": 6,
        "gripper_policy": {"closed_rad": 0.12},
        "limits": limits_b601,
    })
    current_schema = mock_schema_partial
    rec2 = LeRobotRecorder(tmp_path / "rec2", "test/repo2", armd_host="192.168.16.1", fps=10)
    rec2._fetch_schema()
    assert rec2.closed_rad == 0.12
    assert rec2.open_rad == -5.7

    # Partial gripper_policy on so_arm101 never leaves open_rad at -5.7
    so_limits = [
        {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
        for i in range(5)
    ] + [{"name": "gripper", "min_position": 0.0, "max_position": 1.745329, "max_velocity": 2.0, "max_acceleration": 5.0}]
    mock_schema_so = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "gripper_policy": {"closed_rad": 0.0},
        "limits": so_limits,
    })
    current_schema = mock_schema_so
    rec3 = LeRobotRecorder(tmp_path / "rec3", "test/repo3", armd_host="192.168.16.1", fps=10)
    rec3._fetch_schema()
    assert rec3.closed_rad == 0.0
    assert rec3.open_rad == 1.745329


def test_pretrained_info_resolves_so101_norm_for_lerobot_policy(tmp_path):
    dataset_dir = tmp_path / "ds"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(json.dumps({"unit_mode": "so101_norm", "fps": 30}))
    (dataset_dir / "meta" / "stats.json").write_text(json.dumps({"observation.state": {"min": [0.0], "max": [1.0]}}))

    policy_dir = tmp_path / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "train_config.json").write_text(json.dumps({"dataset": {"root": str(dataset_dir)}}))

    info = pretrained_info(policy_dir)
    assert info is not None
    assert info["unit_mode"] == "so101_norm"
    assert "stats" in info

    from rebot_companion.policy_runner import Policy

    class DummyPolicy(Policy):
        action_dim = 6
        fps = 30

        def __init__(self, info_dict):
            self.info = info_dict

    policy = DummyPolicy(info)
    mock_client = MagicMock()
    mock_client.get_safety_describe.return_value = {"device": {"follower": "mock"}}
    mock_res = MagicMock()
    mock_res.json.return_value = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    mock_client.request.return_value = mock_res

    runner = PolicyRunner(mock_client, policy, camera=None, action_units="auto", state_units="auto", operator_present=True)
    runner._read_arm()
    assert runner.action_units == "so101_norm"
    assert runner.state_units == "so101_norm"


# --------------------------------------------------------------------------- mcp_server.py tests


def test_mcp_server_auto_yaw_on_orient_dof_2():
    mock_client = MagicMock()
    # Arm schema with orient_dof == 2 (5-DOF arm)
    mock_schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "orientation_dof": 2,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })
    mock_client.get_schema.return_value = mock_schema

    ctrl = Controller(mock_client, grabber=MagicMock(), detector=MagicMock())
    mock_session = MagicMock()
    mock_session.has_lease = True
    mock_session.lease_lost = None
    mock_session.closed_reason = None
    ctrl.session = mock_session
    ctrl.observation = MagicMock(return_value=None)
    mock_obs = MagicMock()
    mock_obs.tcp = None
    mock_obs.positions = [0.0] * 6
    mock_obs.commanded_positions = None
    mock_session.latest_observation.return_value = mock_obs
    mock_session.move_pose.return_value = MagicMock(
        code="OK", message="", position=[0.2, 0.0, 0.2], rpy=[0.0, 0.0, 0.0],
        success=True, duration_s=1.0, positions=[0.0]*6, collision=False, error=None
    )

    # Calling move_to with roll/pitch and yaw_deg=None -> should pass auto_yaw=True and yaw=0.0
    ctrl.move_to(0.2, 0.0, 0.2, roll_deg=10.0, pitch_deg=20.0, yaw_deg=None, quat=None, speed_mps=None, timeout_s=None)
    _, kwargs = mock_session.move_pose.call_args
    assert kwargs.get("auto_yaw") is True
    assert kwargs.get("rpy")[2] == 0.0

    # Calling move_to with explicit yaw_deg on orient_dof=2 -> auto_yaw=None
    ctrl.move_to(0.2, 0.0, 0.2, roll_deg=10.0, pitch_deg=20.0, yaw_deg=30.0, quat=None, speed_mps=None, timeout_s=None)
    _, kwargs = mock_session.move_pose.call_args
    assert kwargs.get("auto_yaw") is None
    assert abs(kwargs.get("rpy")[2] - math.radians(30.0)) < 1e-4


def test_mcp_server_schema_fallback_7dof():
    mock_client = MagicMock()
    # Schema endpoint raises Exception -> trigger fallback
    mock_client.get_schema.side_effect = Exception("schema not supported")

    ctrl = Controller(mock_client, grabber=MagicMock(), detector=MagicMock())
    mock_obs = MagicMock()
    mock_obs.positions = [0.0] * 7  # 7-DOF legacy mock
    ctrl.observation = MagicMock(return_value=mock_obs)

    schema = ctrl.schema()
    assert schema.dof == 7
    assert schema.gripper_joint == 6
    assert ctrl.gripper_joint() == 6
    assert mock_client.get_schema.call_count == 1
    # Cooldown prevents hammering get_schema on immediate retry
    ctrl.schema()
    assert mock_client.get_schema.call_count == 1
    # Must NOT permanently cache synthetic schema
    assert ctrl._schema is None

    # When cooldown expires and get_schema recovers, next call returns real schema
    real_schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [],
    })
    mock_client.get_schema.side_effect = None
    mock_client.get_schema.return_value = real_schema
    ctrl._last_schema_attempt = 0.0  # simulate cooldown expiry
    schema2 = ctrl.schema()
    assert schema2.dof == 6
    assert ctrl._schema is real_schema


def test_arm_session_does_not_permanently_cache_failed_schema():
    mock_client = MagicMock(spec=ArmdClient)
    mock_client.get_schema.side_effect = Exception("schema fetch failed")
    session = ArmSession(mock_client, safe_exit="off")
    mock_obs = MagicMock()
    mock_obs.positions = [0.0] * 7
    session.latest_observation = MagicMock(return_value=mock_obs)

    # When get_schema fails, gripper_joint falls back to 6 for 7-DOF obs
    gj = session.gripper_joint
    assert gj == 6
    assert session._schema is None
    assert session._gripper_joint is _GRIPPER_UNSET
    assert mock_client.get_schema.call_count == 1

    # Immediate call within cooldown does not hit get_schema again
    assert session.gripper_joint == 6
    assert mock_client.get_schema.call_count == 1

    # When cooldown expires and get_schema later succeeds, it sets real schema and real gripper_joint
    real_schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [],
    })
    mock_client.get_schema.side_effect = None
    mock_client.get_schema.return_value = real_schema
    session._last_schema_attempt = 0.0  # simulate cooldown expiry

    gj2 = session.gripper_joint
    assert gj2 == 5
    assert session._schema is real_schema
    assert session._gripper_joint == 5


def test_leader_feetech_sync_opcodes():
    assert INST_SYNC_READ == 0x82
    assert INST_SYNC_WRITE == 0x83
    mock_transport = MagicMock()
    bus = FeetechSerialBus(mock_transport)
    mock_transport.read.return_value = b""
    bus.read_positions_sync([1, 2])
    pkt = mock_transport.write.call_args_list[0][0][0]
    assert pkt[4] == 0x82


def test_decode_packet_invalid_length_guard():
    # length byte = 0 (< 2)
    bad_buf = b"\xff\xff\x01\x00\x00\x00"
    sid, err, params, rem = decode_packet(bad_buf)
    assert sid is None and err is None and params is None
    assert rem == bad_buf[2:]

    # length byte = 1 (< 2) followed by valid packet
    valid_pkt = encode_packet(1, 0x01, [0x00])
    combined = b"\xff\xff\x01\x01" + valid_pkt
    sid, err, params, rem = decode_packet(combined)
    assert sid is None
    sid2, err2, params2, _ = decode_packet(rem)
    assert sid2 == 1


def test_leader_calibration_validation_and_parsing():
    # Valid calibration with drive_mode (0 -> +1, 1 -> -1)
    cal_dict = {
        "joints": [
            {"center": 2048, "drive_mode": 0},
            {"center": 1000, "drive_mode": 1},
            {"center": 2048, "sign": 1},
            {"center": 2048, "sign": -1},
            {"center": 2048},
        ],
        "gripper": {"closed_tick": 2000, "open_tick": 3000},
        "servo_ids": [1, 2, 3, 4, 5, 6],
    }
    cal = LeaderCalibration.load(cal_dict)
    assert cal.joints[0].sign == 1
    assert cal.joints[1].sign == -1
    assert cal.joints[2].sign == 1
    assert cal.joints[3].sign == -1
    assert cal.joints[4].sign == 1

    # drive_mode with boolean
    cal_bool = LeaderCalibration.load({
        "joints": [
            {"center": 2048, "drive_mode": False},
            {"center": 2048, "drive_mode": True},
            {"center": 2048},
            {"center": 2048},
            {"center": 2048},
        ]
    })
    assert cal_bool.joints[0].sign == 1
    assert cal_bool.joints[1].sign == -1

    # Invalid drive_mode
    with pytest.raises(LeaderError, match="invalid 'drive_mode'"):
        LeaderCalibration.load({"joints": [{"center": 2048, "drive_mode": 2}] + [{"center": 2048}] * 4})

    # Invalid sign
    with pytest.raises(LeaderError, match="'sign' must be 1 or -1"):
        LeaderCalibration.load({"joints": [{"center": 2048, "sign": 0}] + [{"center": 2048}] * 4})

    # homing_offset without center tick must be rejected
    with pytest.raises(LeaderError, match="raw EEPROM 'homing_offset' without 'center'"):
        LeaderCalibration.load({"joints": [{"homing_offset": 500}] + [{"center": 2048}] * 4})

    # joints list length != 5 must be rejected
    with pytest.raises(LeaderError, match="'joints' list must contain exactly 5"):
        LeaderCalibration.load({"joints": [{"center": 2048}] * 4})

    # servo_ids length != 6 or duplicate
    with pytest.raises(LeaderError, match="servo_ids must contain exactly 6"):
        LeaderCalibration(servo_ids=[1, 2, 3, 4, 5])
    with pytest.raises(LeaderError, match="servo_ids must be 6 unique integers"):
        LeaderCalibration(servo_ids=[1, 1, 3, 4, 5, 6])

    # center tick out of range [0, 4096)
    with pytest.raises(LeaderError, match="center must be in"):
        LeaderCalibration(joints=[JointCalib(center=-1)] + [JointCalib()] * 4)
    with pytest.raises(LeaderError, match="center must be in"):
        LeaderCalibration(joints=[JointCalib(center=4096)] + [JointCalib()] * 4)


def test_leader_bridge_preflight_failure_no_unload():
    bus = MagicMock()
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
    mock_session = MagicMock()
    mock_obs = MagicMock()
    mock_obs.positions = [1.0] * 6  # Gap > 0.5 rad
    mock_obs.safety_state = "Armed"
    mock_obs.is_stopped = False
    mock_obs.is_torqued = True
    mock_obs.is_operational = True
    mock_obs.collision = None
    mock_session.wait_for_observation.return_value = mock_obs
    mock_session.latest_observation.return_value = mock_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    bridge = LeaderTeleopBridge(
        bus, mock_session, schema, fps=50, ramp_seconds=0.2, max_gap_rad=0.5, operator_present=False
    )
    assert bridge._activated is False
    with pytest.raises(LeaderError, match="alignment gap .* exceeds --max-gap-rad"):
        bridge.align_and_ramp()

    assert bridge._activated is False
    bridge.close()
    mock_session.unload.assert_not_called()


def test_leader_bridge_hard_gap_ceiling():
    bus = MagicMock()
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
    mock_session = MagicMock()
    mock_obs = MagicMock()
    mock_obs.positions = [3.5] * 6  # Gap > pi
    mock_obs.safety_state = "Armed"
    mock_obs.is_stopped = False
    mock_obs.is_torqued = True
    mock_obs.is_operational = True
    mock_obs.collision = None
    mock_session.wait_for_observation.return_value = mock_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -4.0, "max_position": 4.0, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    bridge = LeaderTeleopBridge(
        bus,
        mock_session,
        schema,
        fps=50,
        ramp_seconds=0.2,
        max_gap_rad=0.5,
        operator_present=True,
        allow_large_gap=True,
    )
    with pytest.raises(LeaderError, match="exceeds hard limit"):
        bridge.align_and_ramp()
    assert bridge._activated is False
    bridge.close()
    mock_session.unload.assert_not_called()


def test_leader_bridge_follower_fault_and_stale_telemetry():
    bus = MagicMock()
    mock_session = MagicMock()
    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })
    bridge = LeaderTeleopBridge(bus, mock_session, schema, fps=10, ramp_seconds=0.2)

    # Collision active raises
    coll_obs = MagicMock()
    coll_obs.is_stopped = False
    coll_obs.collision = {"active": True, "joint": 1}
    with pytest.raises(LeaderError, match="collision hold active"):
        bridge._check_follower_healthy(coll_obs)

    # Motion limit blocked/torque raises
    ml_obs = MagicMock()
    ml_obs.is_stopped = False
    ml_obs.collision = None
    ml_obs.motion_limit = {"reason": "blocked", "joint": 2}
    with pytest.raises(LeaderError, match="follower motion limit active: blocked"):
        bridge._check_follower_healthy(ml_obs)

    # Stale telemetry raises when unchanged longer than threshold
    healthy_obs = MagicMock()
    healthy_obs.is_stopped = False
    healthy_obs.collision = None
    healthy_obs.motion_limit = None
    healthy_obs.is_torqued = True
    healthy_obs.is_operational = True
    healthy_obs.timestamp = 10.0
    mock_session._frames = 100

    bridge._check_follower_healthy(healthy_obs)
    # Fast forward receipt time beyond max(0.5, 5.0 / 10) = 0.5s
    bridge._last_obs_time = time.monotonic() - 0.6
    with pytest.raises(LeaderError, match="follower telemetry is stale"):
        bridge._check_follower_healthy(healthy_obs)


def test_pretrained_info_checkpoint_local_meta(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoint_001000"
    meta_dir = ckpt_dir / "meta"
    meta_dir.mkdir(parents=True)
    info_json = meta_dir / "info.json"
    info_json.write_text(json.dumps({"unit_mode": "so101_norm", "fps": 50}))
    stats_json = meta_dir / "stats.json"
    stats_json.write_text(json.dumps({"observation.state": {"min": [-100.0] * 6, "max": [100.0] * 6}}))

    info = pretrained_info(ckpt_dir)
    assert info is not None
    assert info["unit_mode"] == "so101_norm"
    assert "stats" in info
    assert info["stats"]["observation.state"]["max"] == [100.0] * 6


def test_recorder_and_policy_runner_partial_policy_match(tmp_path: Path, monkeypatch):
    so_limits = [
        {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
        for i in range(5)
    ] + [{"name": "gripper", "min_position": 0.0, "max_position": 1.745329, "max_velocity": 2.0, "max_acceleration": 5.0}]
    raw_schema = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "profile_id": "so-arm101",
        "gripper_policy": {"closed_rad": 0.0},
        "limits": so_limits,
    }
    schema = ObservationSchema.from_dict(raw_schema)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_schema(self):
            return schema

        def get_safety_describe(self):
            return {"device": {"follower": "mock"}}

        def request(self, method, path, **kwargs):
            resp = MagicMock()
            resp.ok = True
            resp.json.return_value = raw_schema
            return resp

    monkeypatch.setattr("rebot.client.ArmdClient", FakeClient)

    rec = LeRobotRecorder(tmp_path / "rec", "test/repo", armd_host="127.0.0.1", fps=10)
    rec._fetch_schema()

    runner = PolicyRunner(FakeClient(), parse_policy("sine:1:0.1:2.0"))
    runner._read_arm()

    assert rec.closed_rad == runner.closed_rad == 0.0
    assert rec.open_rad == runner.open_rad == 1.745329


def test_arm_status_loads_parsing():
    raw_with_loads = {
        "timestamp": 123.456,
        "positions": [0.0] * 6,
        "loads": [0.0, 0.12, -0.05, 0.02, 0.0, 0.01],
    }
    status = ArmStatus.from_dict(raw_with_loads)
    assert status.loads == [0.0, 0.12, -0.05, 0.02, 0.0, 0.01]

    raw_without_loads = {
        "timestamp": 123.456,
        "positions": [0.0] * 6,
    }
    status_none = ArmStatus.from_dict(raw_without_loads)
    assert status_none.loads is None


# --------------------------------------------------------------------------- Round 2 review regression tests


def test_replay_policy_loads_stats_json_when_missing_in_info(tmp_path: Path):
    ds = tmp_path / "ds_no_info_stats"
    (ds / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (ds / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    (ds / "meta" / "info.json").write_text(json.dumps(info))

    stats = {
        "observation.state": {
            "min": [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
            "max": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        "action": {
            "min": [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
            "max": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
    }
    (ds / "meta" / "stats.json").write_text(json.dumps(stats))

    import pyarrow as pa
    import pyarrow.parquet as pq

    ep_tbl = pa.table({
        "episode_index": [0],
        "data/chunk_index": [0],
        "data/file_index": [0],
    })
    pq.write_table(ep_tbl, ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    data_tbl = pa.table({
        "episode_index": [0, 0],
        "action": [[0.0] * 6, [10.0] * 6],
    })
    pq.write_table(data_tbl, ds / "data" / "chunk-000" / "file-000.parquet")

    from rebot_companion.policy_runner import ReplayPolicy

    pol = ReplayPolicy(ds, 0)
    assert "stats" in pol.info
    assert pol.info["stats"]["observation.state"]["min"][0] == -100.0

    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' from non-radian stats"):
        units.resolve_unit_mode("auto", pol.info)


def test_pretrained_info_finds_stats_without_info_json(tmp_path: Path):
    ds = tmp_path / "ds_stats_only"
    (ds / "meta").mkdir(parents=True)
    stats = {"observation.state": {"min": [-3.14], "max": [3.14]}}
    (ds / "meta" / "stats.json").write_text(json.dumps(stats))

    info = pretrained_info(ds)
    assert info is not None
    assert "stats" in info
    assert info["stats"]["observation.state"]["min"] == [-3.14]


def test_resolve_unit_mode_require_metadata():
    # Backward compatibility: require_metadata=False defaults to "rad"
    assert units.resolve_unit_mode("auto", {}) == "rad"
    assert units.resolve_unit_mode("auto", None) == "rad"
    assert units.resolve_unit_mode("auto", {}, require_metadata=False) == "rad"

    # require_metadata=True fails closed if metadata/stats missing
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' without metadata or stats"):
        units.resolve_unit_mode("auto", {}, require_metadata=True)
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' without metadata or stats"):
        units.resolve_unit_mode("auto", None, require_metadata=True)

    # require_metadata=True succeeds when explicit unit_mode is present
    assert units.resolve_unit_mode("auto", {"unit_mode": "so101_norm"}, require_metadata=True) == "so101_norm"
    assert units.resolve_unit_mode("auto", {"rebot_unit_mode": "deg"}, require_metadata=True) == "deg"

    # require_metadata=True succeeds when usable radian stats are present
    usable_meta = {"stats": {"observation.state": {"min": [-2.0], "max": [2.0]}}}
    assert units.resolve_unit_mode("auto", usable_meta, require_metadata=True) == "rad"


def test_policy_runner_fails_closed_on_so_arm_when_metadata_missing():
    from rebot_companion.policy_runner import ReplayPolicy

    class MockReplay(ReplayPolicy):
        def __init__(self):
            self.name = "replay:mock:0"
            self.fps = 10
            self.action_dim = 6
            self.actions = np.zeros((2, 6))
            self.info = {}  # No unit_mode, no stats

    raw_schema = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "profile_id": "so-arm101",
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    schema = ObservationSchema.from_dict(raw_schema)

    class FakeClient:
        def get_schema(self):
            return schema

        def request(self, method, path, **kwargs):
            resp = MagicMock()
            resp.ok = True
            resp.json.return_value = raw_schema
            return resp

        def get_safety_describe(self):
            return {"device": {"follower": "mock"}}

    runner = PolicyRunner(FakeClient(), MockReplay())
    with pytest.raises(ValueError, match="cannot infer unit mode 'auto' without metadata or stats"):
        runner._read_arm()


def test_recorder_fetch_schema_retry_and_strict_validation(tmp_path: Path, monkeypatch):
    class FailingClient:
        def __init__(self, host_or_url: str):
            pass

        def get_schema(self):
            raise ConnectionRefusedError("armd down")

    monkeypatch.setattr("rebot.client.ArmdClient", FailingClient)

    rec_rad = LeRobotRecorder(tmp_path / "rec_rad", "test/repo", armd_host="192.168.1.1", fps=10, unit_mode="rad", camera_url=None)
    rec_rad._fetch_schema()
    assert rec_rad._schema_fetched is False

    rec_norm = LeRobotRecorder(tmp_path / "rec_norm", "test/repo", armd_host="192.168.1.1", fps=10, unit_mode="so101_norm", camera_url=None)
    with pytest.raises(RuntimeError, match="failed to fetch schema from armd"):
        rec_norm._fetch_schema()
    assert rec_norm._schema_fetched is False

    class NoLimitsClient:
        def __init__(self, host_or_url: str):
            pass

        def get_schema(self):
            return ObservationSchema.from_dict({"dof": 6, "arm_dof": 5, "gripper_joint": 5, "limits": []})

    monkeypatch.setattr("rebot.client.ArmdClient", NoLimitsClient)
    rec_norm2 = LeRobotRecorder(tmp_path / "rec_norm2", "test/repo", armd_host="192.168.1.1", fps=10, unit_mode="so101_norm", camera_url=None)
    with pytest.raises(RuntimeError, match="joint limits required for non-radian unit_mode"):
        rec_norm2._fetch_schema()
    assert rec_norm2._schema_fetched is False

    class NoGripperClient:
        def __init__(self, host_or_url: str):
            pass

        def get_schema(self):
            return ObservationSchema.from_dict({
                "dof": 5,
                "arm_dof": 5,
                "gripper_joint": None,
                "limits": [
                    {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
                    for i in range(5)
                ],
            })

    monkeypatch.setattr("rebot.client.ArmdClient", NoGripperClient)
    rec_norm3 = LeRobotRecorder(tmp_path / "rec_norm3", "test/repo", armd_host="192.168.1.1", fps=10, unit_mode="so101_norm", camera_url=None)
    with pytest.raises(RuntimeError, match="gripper_joint required for unit_mode 'so101_norm'"):
        rec_norm3._fetch_schema()


def test_recorder_add_frame_strict_limit_validation(tmp_path: Path):
    rec = LeRobotRecorder(tmp_path / "rec_strict", "test/repo", fps=10, unit_mode="deg", armd_host=None, camera_url=None)
    rec.start_episode(sample=False)
    with pytest.raises(RuntimeError, match="requires joint limits matching state dim"):
        rec.add_frame([0.0] * 6, [0.0] * 6)


def test_leader_teleop_exit_codes_and_fault_code_reporting():
    bus = MagicMock()
    session = MagicMock()
    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    bridge = LeaderTeleopBridge(bus, session, schema, fps=10, ramp_seconds=0.2)

    # 1. Fault code reported correctly from dict obs.fault
    obs_fault = MagicMock()
    obs_fault.is_stopped = True
    obs_fault.safety_state = "Fault"
    obs_fault.fault = {"code": "THERMAL_TRIP", "message": "Motor overheated"}
    with pytest.raises(LeaderError, match=r"follower is in Fault \(THERMAL_TRIP\)"):
        bridge._check_follower_healthy(obs_fault)

    # 2. bridge.run() records error on align_and_ramp failure
    session.wait_for_observation.side_effect = Exception("connection lost")
    bridge.run()
    assert bridge._error is not None
    assert "connection lost" in str(bridge._error)

    # 3. Clean completion without error
    bridge._error = None
    bridge._stop = True
    bridge.align_and_ramp = MagicMock()
    bridge.run()
    assert bridge._error is None

    # 4. KeyboardInterrupt propagates without setting bridge._error
    bridge._error = None
    bridge.align_and_ramp = MagicMock(side_effect=KeyboardInterrupt("Ctrl-C"))
    with pytest.raises(KeyboardInterrupt):
        bridge.run()
    assert bridge._error is None


def test_schema_cooldown_initial_sentinel_allows_early_monotonic(monkeypatch):
    """When time.monotonic() is small (e.g. 1.0 in a fresh container), first schema call must succeed."""
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)

    # 1. ArmSession
    client = MagicMock()
    schema = ObservationSchema.from_dict({"dof": 7, "arm_dof": 6, "gripper_joint": 6})
    client.get_schema.return_value = schema

    session = ArmSession(client)
    assert session._last_schema_attempt is None
    # First call must make the client call even though monotonic() == 1.0 < 5.0
    res = session.schema()
    assert res == schema
    assert client.get_schema.call_count == 1
    assert session._last_schema_attempt == 1.0

    # 2. Controller in mcp_server
    from rebot_companion.mcp_server import Controller
    mcp_client = MagicMock()
    mcp_client.get_schema.return_value = schema
    controller = Controller(mcp_client, MagicMock(), MagicMock())
    assert controller._last_schema_attempt is None
    res_mcp = controller.schema()
    assert res_mcp == schema
    assert mcp_client.get_schema.call_count == 1
    assert controller._last_schema_attempt == 1.0


def test_operator_present_enforces_max_gap_unless_allow_large_gap():
    bus = MagicMock()
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
    mock_session = MagicMock()
    mock_obs = MagicMock()
    mock_obs.positions = [0.8] * 6  # gap 0.8 rad: > max_gap_rad (0.5), < HARD_MAX_GAP_RAD (pi)
    mock_obs.safety_state = "Armed"
    mock_obs.is_stopped = False
    mock_obs.is_torqued = True
    mock_obs.is_operational = True
    mock_obs.collision = None
    mock_session.wait_for_observation.return_value = mock_obs
    mock_session.latest_observation.return_value = mock_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    # 1. operator_present=True alone MUST NOT bypass max_gap_rad
    bridge1 = LeaderTeleopBridge(
        bus, mock_session, schema, fps=50, ramp_seconds=0.2, max_gap_rad=0.5, operator_present=True, allow_large_gap=False
    )
    with pytest.raises(LeaderError, match="alignment gap .* exceeds --max-gap-rad"):
        bridge1.align_and_ramp()
    assert bridge1._activated is False

    # 2. allow_large_gap=True permits gap exceeding max_gap_rad up to pi
    bridge2 = LeaderTeleopBridge(
        bus, mock_session, schema, fps=50, ramp_seconds=0.2, max_gap_rad=0.5, operator_present=False, allow_large_gap=True
    )
    bridge2.align_and_ramp()
    assert bridge2._activated is True

    # 3. allow_large_gap=True STILL enforces HARD_MAX_GAP_RAD (pi)
    mock_obs.positions = [3.5] * 6
    bridge3 = LeaderTeleopBridge(
        bus, mock_session, schema, fps=50, ramp_seconds=0.2, max_gap_rad=0.5, operator_present=True, allow_large_gap=True
    )
    with pytest.raises(LeaderError, match="exceeds hard limit"):
        bridge3.align_and_ramp()
    assert bridge3._activated is False


def test_leader_bridge_post_torque_gap_growth_fails_before_ramp():
    bus = MagicMock()
    # Leader starts at 0.1 rad
    bus.read_positions_sync.return_value = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}

    idle_obs = MagicMock()
    idle_obs.positions = [0.0] * 6  # initial gap: 0.1 rad <= 0.5 rad (passes)
    idle_obs.safety_state = "Idle"
    idle_obs.is_stopped = False
    idle_obs.is_torqued = False
    idle_obs.is_operational = False
    idle_obs.collision = None

    armed_obs = MagicMock()
    armed_obs.positions = [0.8] * 6  # post-torque sag: gap grows to 0.8 rad > 0.5 rad!
    armed_obs.safety_state = "Armed"
    armed_obs.is_stopped = False
    armed_obs.is_torqued = True
    armed_obs.is_operational = True
    armed_obs.collision = None

    mock_session = MagicMock()
    mock_session.wait_for_observation.side_effect = [idle_obs, armed_obs]
    mock_session.latest_observation.return_value = armed_obs

    schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    bridge = LeaderTeleopBridge(
        bus, mock_session, schema, fps=10, ramp_seconds=0.2, max_gap_rad=0.5, allow_large_gap=False
    )
    with pytest.raises(LeaderError, match="post-torque alignment gap .* exceeds --max-gap-rad"):
        bridge.align_and_ramp()

    # Torque was requested, but servo was NOT called due to post-torque gap failure
    mock_session.torque.assert_called_once_with(True)
    mock_session.servo.assert_not_called()


def test_auto_detect_leader_port_and_ping_ids(monkeypatch):
    from rebot_companion import leader_teleop
    from rebot_companion.leader_teleop import auto_detect_leader_port

    # 1. FeetechSerialBus ping & ping_ids
    transport = MockLeaderTransport()
    bus = FeetechSerialBus(transport)

    # Response packet for servo: FF FF ID 02 00 Checksum
    def make_ping_resp(sid):
        body = [sid, 2, 0x00]
        return bytes([0xFF, 0xFF] + body + [compute_checksum(body)])

    transport.to_read = bytearray(make_ping_resp(1))
    assert bus.ping(1) is True
    transport.to_read = bytearray(make_ping_resp(2))
    assert bus.ping(1) is False  # ID mismatch

    # Feed pings for IDs 1..6
    transport.to_read = bytearray(b"".join(make_ping_resp(i) for i in range(1, 7)))
    assert bus.ping_ids([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]

    # Feed pings missing ID 4
    transport.to_read = bytearray(b"".join(make_ping_resp(i) for i in [1, 2, 3, 5, 6]))
    assert bus.ping_ids([1, 2, 3, 4, 5, 6]) == [1, 2, 3]

    # 2. auto_detect_leader_port selecting correct port
    monkeypatch.setattr(leader_teleop.glob, "glob", lambda pat: ["/dev/ttyACM0", "/dev/ttyUSB0"] if "ACM" in pat else [])

    class FakeProbeTransport(MockLeaderTransport):
        def __init__(self, port, **kwargs):
            super().__init__()
            self.port = port
            if port == "/dev/ttyUSB0":
                self.to_read = bytearray(b"".join(make_ping_resp(i) for i in range(1, 7)) * 5)
            else:
                self.to_read = bytearray()

        def close(self):
            pass

    monkeypatch.setattr(leader_teleop, "open_serial_transport", lambda port, **k: FakeProbeTransport(port))
    selected = auto_detect_leader_port(servo_ids=[1, 2, 3, 4, 5, 6])
    assert selected == "/dev/ttyUSB0"

    # 3. main() rejecting non-responding bus before bridge.run() / disable_torque
    mock_client = MagicMock()
    mock_client.get_schema.return_value = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [{"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0} for i in range(6)],
    })
    mock_client.get_safety_describe.return_value = {"device": {"follower": "mock"}}
    monkeypatch.setattr(leader_teleop, "ArmdClient", lambda *a, **k: mock_client)

    class NonRespondingTransport(MockLeaderTransport):
        def close(self):
            pass

    monkeypatch.setattr(leader_teleop, "open_serial_transport", lambda *a, **k: NonRespondingTransport())
    rc = leader_teleop.main(["--leader-port", "/dev/ttyUSB0", "--arm-host", "127.0.0.1"])
    assert rc == 1


def test_mcp_server_gripper_fraction_schema_bounds():
    from rebot_companion.mcp_server import gripper_fraction

    # Direct function testing
    assert gripper_fraction(0.0) == pytest.approx(0.0)
    assert gripper_fraction(-5.7) == pytest.approx(1.0)
    assert gripper_fraction(-2.85) == pytest.approx(0.5)
    assert gripper_fraction(0.0, 0.0, 1.5) == pytest.approx(0.0)
    assert gripper_fraction(1.5, 0.0, 1.5) == pytest.approx(1.0)
    assert gripper_fraction(0.0, 0.0, 0.0) == 0.0  # near-zero span safe fallback

    # Controller.get_state fallback
    schema_so101 = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(5)
        ] + [{"name": "gripper", "min_position": 0.0, "max_position": 1.5, "max_velocity": 2.0, "max_acceleration": 5.0}],
    })
    client = MagicMock()
    client.get_schema.return_value = schema_so101
    obs = MagicMock()
    obs.gripper = None
    obs.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5]
    obs.safety_state = "Armed"
    obs.fault_reason = None
    obs.is_torqued = True
    obs.commanded_positions = obs.positions
    obs.velocities = [0.0] * 6
    obs.torques = [0.0] * 6
    obs.temperatures = [30.0] * 6
    obs.fault = None
    obs.tcp = None
    client.get_observation.return_value = obs

    ctrl = Controller(client, MagicMock(), MagicMock())
    state = ctrl.get_state()
    assert state["gripper"]["fraction"] == 1.0


def test_lerobot_recorder_gripperless_6dof_so101_norm(tmp_path):
    schema_6dof = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 6,
        "gripper_joint": None,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })
    rec = LeRobotRecorder(tmp_path / "rec", "test/repo", unit_mode="so101_norm", fps=10, armd_host=None, camera_url=None)
    rec.schema = schema_6dof
    rec.arm_dof = 6
    rec.gripper_joint = None
    rec.limits = schema_6dof.limits
    rec._schema_fetched = True

    rec.start_episode(task="test")
    rec.add_frame([0.0] * 6, [0.0] * 6)
    assert len(rec._ep.state) == 1

    # Arm with arm_dof=5 and gripper_joint=None must raise
    rec5 = LeRobotRecorder(tmp_path / "rec5", "test/repo", unit_mode="so101_norm", fps=10, armd_host=None, camera_url=None)
    rec5.schema = ObservationSchema.from_dict({
        "dof": 5,
        "arm_dof": 5,
        "gripper_joint": None,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(5)
        ],
    })
    rec5.arm_dof = 5
    rec5.gripper_joint = None
    rec5.limits = schema_6dof.limits[:5]
    rec5._schema_fetched = True
    rec5.start_episode(task="test5")
    with pytest.raises(RuntimeError, match="gripper_joint"):
        rec5.add_frame([0.0] * 5, [0.0] * 5)


def test_units_resolve_schema_gripper_bounds_b601_safety_margin():
    limits_b601 = [
        {"min_position": -3.14, "max_position": 3.14} for _ in range(6)
    ] + [{"min_position": -5.75, "max_position": 0.10}]

    # 1. B601 schema without open_rad in gripper_policy must preserve -5.7
    closed, opened = units.resolve_schema_gripper_bounds(None, limits_b601, 6)
    assert (closed, opened) == (0.0, -5.7)

    # 2. Explicit gripper_policy overrides
    closed, opened = units.resolve_schema_gripper_bounds({"open_rad": -5.6}, limits_b601, 6)
    assert (closed, opened) == (0.0, -5.6)

    # 3. SO-ARM schema (gripper_joint=5) returns dynamic limits
    limits_so = [
        {"min_position": -3.14, "max_position": 3.14} for _ in range(5)
    ] + [{"min_position": -0.1, "max_position": 1.5}]
    closed, opened = units.resolve_schema_gripper_bounds(None, limits_so, 5)
    assert (closed, opened) == (0.0, 1.5)


def test_sine_policy_schema_resolution():
    schema_so = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })

    # "gripper" resolves to index 5
    p_grip = parse_policy("sine:gripper:0.1:2.0")
    p_grip.bind_schema(schema_so)
    assert p_grip.joint == 5

    # "wrist_roll" resolves to index 4
    p_wrist = parse_policy("sine:wrist_roll:0.1:2.0")
    p_wrist.bind_schema(schema_so)
    assert p_wrist.joint == 4

    # Unknown joint name raises RunnerError
    p_bad = parse_policy("sine:unknown_joint:0.1:2.0")
    with pytest.raises(RunnerError, match="unknown joint name"):
        p_bad.bind_schema(schema_so)

    # Gripper on gripperless arm raises RunnerError
    schema_no_grip = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 6,
        "gripper_joint": None,
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    })
    p_grip2 = parse_policy("sine:gripper:0.1:2.0")
    with pytest.raises(RunnerError, match="no gripper joint"):
        p_grip2.bind_schema(schema_no_grip)

    # PolicyRunner pre-seeds session._schema
    client = MagicMock()
    client.get_schema.return_value = schema_so
    client.get_safety_describe.return_value = {"device": {"follower": "mock"}}
    client.request.return_value.json.return_value = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    client.request.return_value.raise_for_status = lambda: None

    runner = PolicyRunner(client, parse_policy("sine:1:0.05:1"), max_seconds=0.01)
    mock_sess = MagicMock()
    mock_sess._schema = None
    runner._loop = lambda: None
    runner._prepare = lambda: None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("rebot_companion.policy_runner.ArmSession.open", lambda self: mock_sess)
        runner.run()
    assert mock_sess._schema == runner.schema

