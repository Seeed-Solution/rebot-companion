"""Unit and integration tests for SO-ARM101 (5+1 DOF) and generic arm support (Stage 1B).

Covers:
- Schema parsing, profile_id, arm_dof, gripper_joint, orientation_dof, gripper_policy, and fallbacks.
- ArmObservation / ArmStatus parsing of tcp and gripper (dataclass and dict-like access).
- ArmSession _tolerances, servo jump exemption, and move_pose orientation for 5+1, 6+0, and 6+1 arms.
- gripper_calib.py handling of zero_check_supported=False and gripper_joint=5.
- mcp_server.py get_state, gripper, and move_to with 2-DOF orientation and obs.tcp/obs.gripper.
- Refusal of local B601 IK on non-B601 / non-6-DOF arms in calibration and pipeline.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from rebot import (
    Arm,
    ArmdClient,
    ArmObservation,
    ArmSession,
    ArmStatus,
    GripperStatus,
    ObservationSchema,
    ServoJumpError,
    TcpPose,
    UnsupportedError,
)
from rebot_companion.calibration import HandEyeResult, check_b601_compatibility, gripper_pose
from rebot_companion.camera_client import MarkerDetection, ReCameraClient
from rebot_companion.gripper_calib import main as gripper_calib_main
from rebot_companion.ik import B601IkAdapter
from rebot_companion.mcp_server import Controller, ToolFail
from rebot_companion.pipeline import DeterministicPickPlacePipeline, PipelineConfig
from rebot_companion.transforms import Transform

# ==============================================================================
# 1. Schema Parsing & Fallback Tests
# ==============================================================================


def test_schema_parsing_5_plus_1_so_arm101():
    raw = {
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "orientation_dof": 2,
        "profile_id": "so-arm101",
        "gripper_policy": {
            "closed_rad": 0.0,
            "open_rad": -1.5,
            "zero_check_supported": False,
            "holding_mode": "current",
        },
        "capabilities": {"gripper": True, "gripper_zero": False},
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    schema = ObservationSchema.from_dict(raw)
    assert schema.dof == 6
    assert schema.arm_dof == 5
    assert schema.gripper_joint == 5
    assert schema.orientation_dof == 2
    assert schema.profile_id == "so-arm101"
    assert schema.gripper_policy is not None
    assert schema.gripper_policy["zero_check_supported"] is False


def test_schema_parsing_6_plus_0_b601_rs():
    raw = {
        "dof": 6,
        "arm_dof": 6,
        "gripper_joint": None,
        "orientation_dof": 3,
        "profile_id": "rebot-b601rs",
        "capabilities": {"gripper": False},
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(6)
        ],
    }
    schema = ObservationSchema.from_dict(raw)
    assert schema.dof == 6
    assert schema.arm_dof == 6
    assert schema.gripper_joint is None
    assert schema.orientation_dof == 3
    assert schema.profile_id == "rebot-b601rs"
    assert schema.gripper_policy is None


def test_schema_parsing_6_plus_1_b601_dm():
    raw = {
        "dof": 7,
        "arm_dof": 6,
        "gripper_joint": 6,
        "orientation_dof": 3,
        "profile_id": "rebot-b601dm",
        "gripper_policy": {
            "closed_rad": 0.0,
            "open_rad": -5.7,
            "zero_check_supported": True,
        },
        "capabilities": {"gripper": True, "gripper_zero": True},
        "limits": [
            {"name": f"j{i}", "min_position": -3.14, "max_position": 3.14, "max_velocity": 2.0, "max_acceleration": 5.0}
            for i in range(7)
        ],
    }
    schema = ObservationSchema.from_dict(raw)
    assert schema.dof == 7
    assert schema.arm_dof == 6
    assert schema.gripper_joint == 6
    assert schema.orientation_dof == 3
    assert schema.profile_id == "rebot-b601dm"


def test_schema_fallback_when_fields_omitted():
    # Legacy armd with 7-DOF:
    legacy_7 = ObservationSchema.from_dict({"dof": 7, "limits": []})
    assert legacy_7.arm_dof == 6
    assert legacy_7.gripper_joint == 6
    assert legacy_7.orientation_dof == 3
    assert legacy_7.profile_id is None
    assert legacy_7.gripper_policy is None

    # Legacy armd with 6-DOF (dof=6, omitted fields):
    legacy_6 = ObservationSchema.from_dict({"dof": 6, "limits": []})
    assert legacy_6.arm_dof is None
    assert legacy_6.gripper_joint is None  # fallback for dof != 7 is None
    assert legacy_6.orientation_dof is None

    # Legacy armd with 5-DOF (dof=5, omitted fields):
    legacy_5 = ObservationSchema.from_dict({"dof": 5, "limits": []})
    assert legacy_5.arm_dof == 5
    assert legacy_5.gripper_joint is None
    assert legacy_5.orientation_dof == 2


# ==============================================================================
# 2. ArmObservation / ArmStatus: tcp & gripper Parsing Tests
# ==============================================================================


def test_observation_tcp_and_gripper_parsing():
    raw = {
        "timestamp": 12.34,
        "positions": [0.1, 0.2, 0.3, 0.4, 0.5, -0.8],
        "velocities": [0.0] * 6,
        "torques": [0.0] * 6,
        "temperatures": [30.0] * 6,
        "is_torqued": True,
        "safety_state": "Armed",
        "epoch_id": 2,
        "tcp": {
            "xyz_m": [0.25, 0.05, 0.15],
            "rpy_rad": [0.1, 0.5, 0.0],
            "quat_wxyz": [0.9689, 0.0383, 0.2447, -0.0097],
        },
        "gripper": {
            "joint": 5,
            "rad": -0.8,
            "fraction": 0.53,
            "opening_m": 0.045,
            "stalled": False,
        },
    }
    obs = ArmStatus.from_dict(raw)
    assert isinstance(obs, ArmObservation)
    assert obs.tcp is not None
    assert isinstance(obs.tcp, TcpPose)
    assert obs.tcp.xyz_m == [0.25, 0.05, 0.15]
    assert obs.tcp["xyz_m"] == [0.25, 0.05, 0.15]
    assert obs.tcp.get("rpy_rad") == [0.1, 0.5, 0.0]
    assert "quat_wxyz" in obs.tcp
    assert obs.tcp.to_dict()["xyz_m"] == [0.25, 0.05, 0.15]

    assert obs.gripper is not None
    assert isinstance(obs.gripper, GripperStatus)
    assert obs.gripper.joint == 5
    assert obs.gripper.fraction == 0.53
    assert obs.gripper["rad"] == -0.8
    assert obs.gripper.opening_m == 0.045
    assert obs.gripper.stalled is False
    assert obs.gripper.to_dict()["joint"] == 5


def test_observation_omitted_tcp_and_gripper():
    raw = {
        "timestamp": 1.0,
        "positions": [0.0] * 6,
        "velocities": [0.0] * 6,
        "torques": [0.0] * 6,
        "temperatures": [25.0] * 6,
        "is_torqued": False,
        "safety_state": "Idle",
        "epoch_id": 1,
    }
    obs = ArmStatus.from_dict(raw)
    assert obs.tcp is None
    assert obs.gripper is None


# ==============================================================================
# 3. ArmSession: _tolerances, servo, and move_pose for 5+1, 6+0, 6+1
# ==============================================================================


def _mock_session(schema: ObservationSchema) -> ArmSession:
    client = MagicMock(spec=ArmdClient)
    client.get_schema.return_value = schema
    session = ArmSession(client)
    session._schema = schema
    session._obs = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * schema.dof,
        velocities=[0.0] * schema.dof,
        torques=[0.0] * schema.dof,
        temperatures=[25.0] * schema.dof,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )
    session._generation = 1
    session._epoch = 1
    session._send = MagicMock()
    return session


def test_tolerances_and_servo_5_plus_1_so_arm101():
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 5,
            "gripper_joint": 5,
            "orientation_dof": 2,
            "profile_id": "so-arm101",
            "limits": [],
        }
    )
    session = _mock_session(schema)
    assert session.gripper_joint == 5

    # _tolerances check: joint 5 must have tol[5] = None, 0..4 = 0.03
    target = [0.1, 0.2, -0.3, 0.4, 0.5, -1.2]
    tol = session._tolerances(target, None)
    assert len(tol) == 6
    assert tol[:5] == [0.03, 0.03, 0.03, 0.03, 0.03]
    assert tol[5] is None  # gripper exempt from settling tolerance

    # servo jump check: gripper joint 5 jumping 1.0 rad (> 0.35 rad) must NOT raise
    session.servo([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert session._send.called

    # servo jump on arm joint 0 jumping 0.5 rad (> 0.35 rad) MUST raise ServoJumpError
    with pytest.raises(ServoJumpError):
        session.servo([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_tolerances_and_servo_6_plus_0_arm():
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 6,
            "gripper_joint": None,
            "orientation_dof": 3,
            "profile_id": "rebot-b601rs",
            "limits": [],
        }
    )
    session = _mock_session(schema)
    assert session.gripper_joint is None

    # _tolerances: all 6 joints must be 0.03 (no gripper)
    target = [0.1, 0.2, -0.3, 0.4, 0.5, 0.6]
    tol = session._tolerances(target, None)
    assert tol == [0.03] * 6

    # servo jump: joint 5 is an arm joint, jumping 0.5 rad MUST raise ServoJumpError
    with pytest.raises(ServoJumpError):
        session.servo([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])


def test_tolerances_and_servo_6_plus_1_b601_dm():
    schema = ObservationSchema.from_dict(
        {
            "dof": 7,
            "arm_dof": 6,
            "gripper_joint": 6,
            "orientation_dof": 3,
            "profile_id": "rebot-b601dm",
            "limits": [],
        }
    )
    session = _mock_session(schema)
    assert session.gripper_joint == 6

    target = [0.0] * 7
    tol = session._tolerances(target, None)
    assert tol[:6] == [0.03] * 6
    assert tol[6] is None

    # gripper joint 6 jumping 1.0 rad does not raise
    session.servo([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
    assert session._send.called


def test_move_pose_orientation_none_when_rpy_omitted():
    schema = ObservationSchema.from_dict(
        {"dof": 6, "arm_dof": 5, "gripper_joint": 5, "orientation_dof": 2, "limits": []}
    )
    session = _mock_session(schema)
    session._action = MagicMock()

    session.move_pose([0.25, 0.0, 0.15], rpy=None)
    session._action.assert_called_once()
    path, body = session._action.call_args[0][:2]
    assert path == "/api/v1/action/move_to"
    assert body["position"] == [0.25, 0.0, 0.15]
    assert body["orientation"] is None


# ==============================================================================
# 4. gripper_calib.py Tests: zero_check_supported=False & gripper_joint=5
# ==============================================================================


def test_gripper_calib_aborts_when_no_gripper():
    schema = ObservationSchema.from_dict(
        {"dof": 6, "arm_dof": 6, "gripper_joint": None, "capabilities": {"gripper": False}, "limits": []}
    )
    with patch("rebot_companion.gripper_calib.ArmdClient") as mock_client_cls:
        client = mock_client_cls.return_value
        client.get_schema.return_value = schema
        ret = gripper_calib_main(["--host", "127.0.0.1"])
        assert ret == 1


def test_gripper_calib_skips_zero_check_for_so_arm101(capsys):
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 5,
            "gripper_joint": 5,
            "orientation_dof": 2,
            "profile_id": "so-arm101",
            "gripper_policy": {"zero_check_supported": False},
            "capabilities": {"gripper": True, "gripper_zero": False},
            "limits": [],
        }
    )
    obs_open = ArmStatus(
        timestamp=1.0,
        positions=[0.0, 0.0, 0.0, 0.0, 0.0, -1.2],
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )
    obs_closed = ArmStatus(
        timestamp=1.0,
        positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )
    with (
        patch("rebot_companion.gripper_calib.ArmdClient") as mock_client_cls,
        patch("rebot_companion.gripper_calib.Arm") as mock_arm_cls,
        patch("builtins.input", side_effect=["y", "40.0", "10.0", "y"]),
    ):
        client = mock_client_cls.return_value
        client.get_schema.return_value = schema
        client.get_gripper_calibration.return_value = {}
        client.get_observation.return_value = obs_open

        arm = MagicMock()
        mock_arm_cls.return_value.__enter__.return_value = arm
        arm.state.side_effect = [obs_open, obs_closed]
        arm.gripper_calibration.return_value = {"jaw_opening": {"points": [[-1.2, 0.04], [0.0, 0.01]]}}

        ret = gripper_calib_main(["--host", "127.0.0.1", "--points", "2"])
        assert ret == 0

        # Step 1 gripper_zero_check must NOT have been called!
        arm.gripper_zero_check.assert_not_called()

        out = capsys.readouterr().out
        assert "Mechanical zero check is not supported" in out
        assert "Measured J6 = -1.2000 rad" in out
        assert "J6 (rad)" in out


# ==============================================================================
# 5. mcp_server.py: obs.tcp / obs.gripper and 2-DOF orientation Tests
# ==============================================================================


def test_mcp_server_reads_obs_tcp_and_gripper_on_5_plus_1():
    client = MagicMock(spec=ArmdClient)
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 5,
            "gripper_joint": 5,
            "orientation_dof": 2,
            "profile_id": "so-arm101",
            "limits": [],
        }
    )
    client.get_schema.return_value = schema
    obs = ArmStatus(
        timestamp=1.0,
        positions=[0.1, 0.2, 0.3, 0.4, 0.5, -0.9],
        velocities=[0.0] * 6,
        torques=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        temperatures=[30.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
        tcp=TcpPose(xyz_m=[0.22, -0.05, 0.18], rpy_rad=[0.0, 0.7854, 0.0], quat_wxyz=[0.9239, 0.0, 0.3827, 0.0]),
        gripper=GripperStatus(joint=5, rad=-0.9, fraction=0.45, opening_m=0.038, stalled=False),
    )
    client.get_observation.return_value = obs

    ctrl = Controller(client, MagicMock(), MagicMock(), auto_lease=False)
    state = ctrl.get_state()

    assert state["tcp"] is not None
    assert state["tcp"]["xyz_m"] == [0.22, -0.05, 0.18]
    assert state["tcp"]["rpy_deg"] == [0.0, 45.0, 0.0]
    assert state["tcp"]["quat"] == [0.9239, 0.0, 0.3827, 0.0]

    assert state["gripper"] is not None
    assert state["gripper"]["joint"] == 5
    assert state["gripper"]["rad"] == -0.9
    assert state["gripper"]["fraction"] == 0.45
    assert state["gripper"]["opening_m"] == 0.038
    assert state["gripper"]["stalled"] is False


def test_mcp_server_6_plus_0_has_no_gripper():
    client = MagicMock(spec=ArmdClient)
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 6,
            "gripper_joint": None,
            "orientation_dof": 3,
            "profile_id": "rebot-b601rs",
            "capabilities": {"gripper": False},
            "limits": [],
        }
    )
    client.get_schema.return_value = schema
    obs = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * 6,
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )
    client.get_observation.return_value = obs
    ctrl = Controller(client, MagicMock(), MagicMock(), auto_lease=False)
    state = ctrl.get_state()
    assert state["gripper"] is None

    # Attempting to call gripper tool must fail with UNSUPPORTED
    with pytest.raises(ToolFail) as exc_info:
        ctrl.gripper("open", fraction=None, grasp=None)
    assert exc_info.value.code == "UNSUPPORTED"


def test_mcp_server_move_to_2_dof_orientation():
    client = MagicMock(spec=ArmdClient)
    schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 5,
            "gripper_joint": 5,
            "orientation_dof": 2,
            "profile_id": "so-arm101",
            "limits": [],
        }
    )
    client.get_schema.return_value = schema
    obs = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * 6,
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
        tcp=TcpPose(xyz_m=[0.20, 0.0, 0.10], rpy_rad=[0.0, 0.5, 0.0], quat_wxyz=[1, 0, 0, 0]),
    )
    client.get_observation.return_value = obs

    ctrl = Controller(client, MagicMock(), MagicMock(), auto_lease=False)
    session = MagicMock(spec=ArmSession)
    session.closed_reason = None
    session.has_lease = True
    session.latest_observation.return_value = obs
    action_res = MagicMock()
    action_res.action = None
    action_res.status = obs
    action_res.duration_s = 1.0
    session.move_pose.return_value = action_res
    ctrl.session = session

    # 1. All angles omitted -> rpy must be None
    ctrl.move_to(0.25, 0.0, 0.15, None, None, None, None, None, None)
    session.move_pose.assert_called_with([0.25, 0.0, 0.15], rpy=None, quat=None, linear_speed=None, timeout=None)

    # 2. Only pitch_deg specified, yaw_deg omitted on 2-DOF arm -> does not crash, yaw derives from current
    ctrl.move_to(0.25, 0.0, 0.15, None, 45.0, None, None, None, None)
    called_rpy = session.move_pose.call_args[1]["rpy"]
    assert len(called_rpy) == 3
    assert math.isclose(called_rpy[1], math.radians(45.0), abs_tol=1e-4)


# ==============================================================================
# 6. Refusing Local B601 IK on non-B601 Arms
# ==============================================================================


def test_refuse_b601_kinematics_in_calibration():
    # Compatible schema (b601, arm_dof=6)
    b601_schema = ObservationSchema.from_dict({"dof": 7, "arm_dof": 6, "profile_id": "rebot-b601dm", "limits": []})
    check_b601_compatibility(b601_schema)  # does not raise

    # Incompatible: profile_id="so-arm101"
    so_schema = ObservationSchema.from_dict({"dof": 6, "arm_dof": 5, "profile_id": "so-arm101", "limits": []})
    with pytest.raises(ValueError, match="Local B601 kinematics"):
        check_b601_compatibility(so_schema)

    with pytest.raises(ValueError, match="Local B601 kinematics"):
        gripper_pose([0.0] * 6, schema=so_schema)


def test_pipeline_refuses_b601_ik_on_non_b601_arm():
    so_schema = ObservationSchema.from_dict(
        {
            "dof": 6,
            "arm_dof": 5,
            "gripper_joint": 5,
            "profile_id": "so-arm101",
            "limits": [],
        }
    )
    arm = MagicMock(spec=ArmdClient)
    arm.get_schema.return_value = so_schema
    arm.get_observation.return_value = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * 6,
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )

    camera = MagicMock(spec=ReCameraClient)
    marker = Transform.from_translation_euler([0.2, 0.1, 0.3], 0.0, 0.0, 0.0)
    camera.trigger.return_value = [MarkerDetection(id=0, center=[0, 0], corners=[], pose=marker, reprojection_error=0)]

    pipeline = DeterministicPickPlacePipeline(
        camera,
        arm,
        Transform(),
        PipelineConfig(max_allowed_distance=1.0),
        ik_solver=B601IkAdapter(),
    )
    res = pipeline.run_cycle()
    assert res.success is False
    assert res.stage == "ik_solve"
    assert "Refusing local B601 IK" in res.error_message


def test_pipeline_eye_in_hand_fails_closed_before_cam_to_base_on_so_arm101():
    so_schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "profile_id": "so-arm101",
        "limits": [],
    })
    arm = MagicMock(spec=ArmdClient)
    arm.get_schema.return_value = so_schema
    arm.get_observation.return_value = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * 6,
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )

    camera = MagicMock(spec=ReCameraClient)
    marker = Transform.from_translation_euler([0.2, 0.1, 0.3], 0.0, 0.0, 0.0)
    camera.trigger.return_value = [MarkerDetection(id=0, center=[0, 0], corners=[], pose=marker, reprojection_error=0)]

    hand_eye = MagicMock(spec=HandEyeResult)
    hand_eye.is_eye_in_hand = True
    hand_eye.setup = "eye_in_hand"

    pipeline = DeterministicPickPlacePipeline(
        camera,
        arm,
        hand_eye,
        PipelineConfig(max_allowed_distance=1.0),
    )
    res = pipeline.run_cycle()
    assert res.success is False
    assert res.stage == "frame_transform"
    assert "Camera pose unavailable: Local B601 kinematics cannot be used for arm with profile_id 'so-arm101'" in res.error_message
    hand_eye.cam_to_base.assert_not_called()


def test_pipeline_fails_closed_when_get_schema_fails():
    arm = MagicMock(spec=ArmdClient)
    arm.get_schema.side_effect = Exception("schema network timeout")
    arm.get_observation.return_value = ArmStatus(
        timestamp=1.0,
        positions=[0.0] * 6,
        velocities=[0.0] * 6,
        torques=[0.0] * 6,
        temperatures=[25.0] * 6,
        is_torqued=True,
        safety_state="Armed",
        epoch_id=1,
    )
    camera = MagicMock(spec=ReCameraClient)
    marker = Transform.from_translation_euler([0.2, 0.1, 0.3], 0.0, 0.0, 0.0)
    camera.trigger.return_value = [MarkerDetection(id=0, center=[0, 0], corners=[], pose=marker, reprojection_error=0)]

    # 1. eye_in_hand -> stage "frame_transform"
    hand_eye = MagicMock(spec=HandEyeResult)
    hand_eye.is_eye_in_hand = True
    hand_eye.setup = "eye_in_hand"
    pipeline_eih = DeterministicPickPlacePipeline(camera, arm, hand_eye, PipelineConfig(max_allowed_distance=1.0))
    res_eih = pipeline_eih.run_cycle()
    assert res_eih.success is False and res_eih.stage == "frame_transform"
    assert "schema network timeout" in res_eih.error_message
    hand_eye.cam_to_base.assert_not_called()

    # 2. is_b601_ik -> stage "ik_solve"
    pipeline_ik = DeterministicPickPlacePipeline(
        camera, arm, Transform(), PipelineConfig(max_allowed_distance=1.0), ik_solver=B601IkAdapter()
    )
    res_ik = pipeline_ik.run_cycle()
    assert res_ik.success is False and res_ik.stage == "ik_solve"
    assert "schema network timeout" in res_ik.error_message


def test_schema_fallback_unlabeled_6dof_fails_b601_compatibility():
    # Legacy armd with 6-DOF (dof=6, omitted fields) yields arm_dof=None
    sch = ObservationSchema.from_dict({"dof": 6, "limits": []})
    assert sch.arm_dof is None
    with pytest.raises(ValueError, match="Local B601 kinematics cannot be used for unlabeled non-7-DOF arm"):
        check_b601_compatibility(sch)

    # Dictionary with dof=6 and no profile_id/arm_dof also fails
    with pytest.raises(ValueError, match="Local B601 kinematics cannot be used for unlabeled non-7-DOF arm"):
        check_b601_compatibility({"dof": 6})

    # Unlabeled 7-DOF arm is allowed
    check_b601_compatibility({"dof": 7})


def test_gripper_zero_check_raises_unsupported_on_so_arm101():
    # 1. capabilities.gripper_zero is False
    sch_disabled_zero = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [],
        "capabilities": {"gripper_zero": False},
    })
    mock_session = MagicMock()
    mock_session._schema = sch_disabled_zero
    mock_session.schema.return_value = sch_disabled_zero

    with patch.object(ArmdClient, "get_schema", return_value=sch_disabled_zero), patch.object(ArmSession, "open"):
        arm = Arm(host="127.0.0.1", port=8765, enable=False)
    arm.session = mock_session
    with pytest.raises(UnsupportedError) as exc_info:
        arm.gripper_zero_check()
    assert exc_info.value.code == "UNSUPPORTED"
    assert "gripper mechanical zero check is not supported" in str(exc_info.value)

    # 2. gripper_policy zero_check_supported is False
    sch_disabled_policy = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "limits": [],
        "capabilities": {},
        "gripper_policy": {"zero_check_supported": False},
    })
    mock_session._schema = sch_disabled_policy
    with pytest.raises(UnsupportedError) as exc_info2:
        arm.gripper_zero_check()
    assert exc_info2.value.code == "UNSUPPORTED"


def test_armsession_open_heartbeat_starts_immediately_after_acquire_lease():
    import threading
    import time
    from unittest.mock import patch
    import rebot._ws

    client = MagicMock(spec=ArmdClient)
    client.base_url = "http://127.0.0.1:8765"
    session = ArmSession(client, connect_timeout=1.0)

    events = []

    def fake_wait_for_obs(timeout=None):
        events.append("wait_for_observation")

    def fake_schema(timeout=None):
        events.append(f"schema(timeout={timeout})")
        return MagicMock()

    def fake_acquire_lease():
        events.append("acquire_lease")

    session.wait_for_observation = fake_wait_for_obs
    session.schema = fake_schema
    session._acquire_lease = fake_acquire_lease

    with patch("rebot._ws.connect", return_value=MagicMock()):
        with patch("threading.Thread") as mock_thread_cls:
            created_threads = []

            def make_thread(target=None, name=None, daemon=None):
                t = MagicMock()
                t.name = name
                t.start.side_effect = lambda: events.append(f"start_{name}")
                created_threads.append(t)
                return t

            mock_thread_cls.side_effect = make_thread

            session.open()

    # Pre-lease schema was fetched with bounded timeout <= 1.5s
    assert "schema(timeout=1.0)" in events
    # Heartbeat thread started immediately after acquire_lease
    idx_lease = events.index("acquire_lease")
    idx_hb = events.index("start_armd-heartbeat")
    assert idx_lease < idx_hb
    # Shutdown on error tears down threads
    with patch.object(session, "_shutdown") as mock_shutdown:
        session._ws = None  # force re-open
        session._acquire_lease = MagicMock(side_effect=RuntimeError("lease failed"))
        with patch("rebot._ws.connect", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="lease failed"):
                session.open()
        mock_shutdown.assert_called_once()


def test_mcp_server_auto_yaw_unsupported_when_explicitly_false():
    mock_client = MagicMock()
    mock_schema = ObservationSchema.from_dict({
        "dof": 6,
        "arm_dof": 5,
        "gripper_joint": 5,
        "orientation_dof": 2,
        "capabilities": {"version": 6, "auto_yaw": False},
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
    mock_obs = MagicMock()
    mock_obs.tcp = None
    mock_obs.positions = [0.0] * 6
    mock_obs.commanded_positions = None
    mock_session.latest_observation.return_value = mock_obs

    # yaw_deg is None on orient_dof == 2 with auto_yaw: False must raise ToolFail("UNSUPPORTED")
    with pytest.raises(ToolFail) as exc_info:
        ctrl.move_to(0.2, 0.0, 0.2, roll_deg=10.0, pitch_deg=20.0, yaw_deg=None, quat=None, speed_mps=None, timeout_s=None)
    assert exc_info.value.code == "UNSUPPORTED"
    assert "armd does not advertise auto_yaw capability" in exc_info.value.message
