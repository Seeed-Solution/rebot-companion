"""Unit tests for Stage 4: units conversion, recorder/policy unit integration,
PC-side leader teleop bridge, and auto_yaw polish.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rebot.client import ArmStatus, JointLimit, ObservationSchema
from rebot_companion import units
from rebot_companion.leader_teleop import (
    DEFAULT_WRIST_ROLL_LIMITS,
    GripperCalib,
    JointCalib,
    LeaderCalibration,
    LeaderError,
    LeaderTeleopBridge,
    LeaderTransport,
    compute_checksum,
    decode_packet,
    encode_packet,
)
from rebot_companion.lerobot_recorder import LeRobotRecorder
from rebot_companion.mcp_server import Controller
from rebot_companion.policy_runner import PolicyRunner, RunnerError, pretrained_info

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

    # Heuristic check from stats
    stats_norm = {
        "stats": {
            "observation.state": {
                "max": [95.0, 80.0, 70.0, 50.0, 40.0, 100.0],
                "min": [-95.0, -80.0, -70.0, -50.0, -40.0, 0.0],
            }
        }
    }
    assert units.resolve_unit_mode("auto", stats_norm) == "so101_norm"

    stats_deg = {
        "stats": {
            "observation.state": {
                "max": [180.0, 90.0, 120.0, 150.0, 180.0, 100.0],
                "min": [-180.0, -90.0, -120.0, -150.0, -180.0, 0.0],
            }
        }
    }
    assert units.resolve_unit_mode("auto", stats_deg) == "deg"


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
        ramp_seconds=0.1,  # 5 steps
        max_gap_rad=0.5,
        park_on_exit=True,
    )

    bridge.align_and_ramp()
    assert bridge.frames_sent == 5
    assert mock_session.servo.call_count == 5

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
        fps=50,
        ramp_seconds=0.04,  # 2 steps
        max_gap_rad=0.5,
    )
    bridge.align_and_ramp()
    mock_session.torque.assert_called_once_with(True)
    assert mock_session.servo.call_count == 2


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
    assert any(
        method == "GET" and url == "http://192.168.16.1:8765/api/v1/gripper/calibration"
        for method, url in recorded_requests
    )


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
