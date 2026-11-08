from unittest.mock import MagicMock

import numpy as np
import pytest

from rebot_companion.arm_client import ArmCommandRejected, ArmdClient, ArmSession, ArmStatus
from rebot_companion.camera_client import MarkerDetection, ReCameraClient
from rebot_companion.pipeline import (
    DeterministicPickPlacePipeline,
    PipelineConfig,
)
from rebot_companion.transforms import Transform


def status(state="Idle", torqued=False) -> ArmStatus:
    return ArmStatus.from_dict(
        {
            "timestamp": 1.0,
            "positions": [0.0] * 6,
            "velocities": [0.0] * 6,
            "torques": [0.0] * 6,
            "temperatures": [35.0] * 6,
            "is_torqued": torqued,
            "safety_state": state,
            "epoch_id": 1,
        }
    )


def marker_scene():
    camera = MagicMock(spec=ReCameraClient)
    marker_pose = Transform.from_translation_euler([0.1, 0.2, 0.4], 0.0, 0.0, 0.0)
    camera.trigger.return_value = [
        MarkerDetection(id=0, center=[320.0, 180.0], corners=[], pose=marker_pose, reprojection_error=0.001)
    ]
    return camera


def session_mock(state="Idle"):
    session = MagicMock(spec=ArmSession)
    session.latest_observation.return_value = status(state, torqued=state != "Idle")
    return session


def test_pipeline_success():
    camera = MagicMock(spec=ReCameraClient)
    arm = MagicMock(spec=ArmdClient)

    # Arm healthy
    arm.get_observation.return_value = status("Idle")

    # Camera detects marker 0 at [0.1, 0.2, 0.4] in camera frame
    marker_pose = Transform.from_translation_euler([0.1, 0.2, 0.4], 0.0, 0.0, 0.0)
    camera.trigger.return_value = [
        MarkerDetection(id=0, center=[320.0, 180.0], corners=[], pose=marker_pose, reprojection_error=0.001)
    ]

    # Camera is mounted at [0.1, -0.1, 0.1] relative to arm base
    cam_to_base = Transform.from_translation_euler([0.1, -0.1, 0.1], 0.0, 0.0, 0.0)

    config = PipelineConfig(target_marker_id=0, max_allowed_distance=1.0)
    pipeline = DeterministicPickPlacePipeline(camera, arm, cam_to_base, config)

    res = pipeline.run_cycle()
    assert res.success is True
    assert res.stage == "completed"
    assert res.target_pose_base is not None
    # Expected translation: [0.1+0.1, -0.1+0.2, 0.1+0.4] = [0.2, 0.1, 0.5]
    np.testing.assert_allclose(res.target_pose_base.translation, [0.2, 0.1, 0.5], atol=1e-5)


def test_pipeline_estop_aborts():
    camera = MagicMock(spec=ReCameraClient)
    arm = MagicMock(spec=ArmdClient)

    arm.get_observation.return_value = status("EStop")  # E-STOP ACTIVE

    cam_to_base = Transform()
    pipeline = DeterministicPickPlacePipeline(camera, arm, cam_to_base)
    res = pipeline.run_cycle()

    assert res.success is False
    assert res.stage == "health_check"
    assert "E-STOPPED" in res.error_message
    camera.trigger.assert_not_called()


def test_pipeline_marker_not_found():
    camera = MagicMock(spec=ReCameraClient)
    arm = MagicMock(spec=ArmdClient)

    arm.get_observation.return_value = status("Idle")
    camera.trigger.return_value = []  # No markers

    cam_to_base = Transform()
    config = PipelineConfig(target_marker_id=42)
    pipeline = DeterministicPickPlacePipeline(camera, arm, cam_to_base, config)
    res = pipeline.run_cycle()

    assert res.success is False
    assert res.stage == "marker_detection"
    assert "#42 not detected" in res.error_message


def test_pipeline_fault_aborts():
    camera = MagicMock(spec=ReCameraClient)
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status({"Fault": "WatchdogTimeout"})
    res = DeterministicPickPlacePipeline(camera, arm, Transform()).run_cycle()
    assert res.success is False and res.stage == "health_check"
    assert "WatchdogTimeout" in res.error_message
    camera.trigger.assert_not_called()


def test_pipeline_motion_uses_lease_session():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Idle")
    session = session_mock("Idle")
    arm.session.return_value.__enter__.return_value = session
    ik = MagicMock(side_effect=lambda pose: [float(pose.translation[2])] + [0.0] * 5)
    cam_to_base = Transform.from_translation_euler([0.1, -0.1, 0.1], 0.0, 0.0, 0.0)
    pipeline = DeterministicPickPlacePipeline(
        marker_scene(), arm, cam_to_base, PipelineConfig(max_allowed_distance=1.0), ik_solver=ik
    )
    res = pipeline.run_cycle()
    assert res.success is True, res.error_message
    arm.session.assert_called_once()  # pipeline-owned session (lease) per cycle
    session.torque.assert_called_once_with(True)  # Idle arm gets armed
    targets = [c.args[0][0] for c in session.move_to.call_args_list]
    assert targets == pytest.approx([0.58, 0.5, 0.58, 0.05])  # approach, grasp, lift, place (z as joint 0)
    session.unload.assert_called_once()
    assert not hasattr(ArmdClient, "servo_joints")


def test_pipeline_external_session_and_unload_unsupported():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Armed", torqued=True)
    session = session_mock("Armed")
    session.unload.side_effect = ArmCommandRejected(
        {"type": "command_reject", "code": "UNSUPPORTED", "message": "no resting pose"}
    )
    pipeline = DeterministicPickPlacePipeline(
        marker_scene(),
        arm,
        Transform(),
        PipelineConfig(max_allowed_distance=1.0),
        ik_solver=lambda pose: [0.0] * 6,
        session=session,
    )
    res = pipeline.run_cycle()
    assert res.success is True
    arm.session.assert_not_called()
    session.torque.assert_not_called()  # already Armed
    assert session.move_to.call_count == 4
    assert any("unsupported" in w for w in res.warnings)


def test_pipeline_ik_failure_moves_nothing():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Idle")
    pipeline = DeterministicPickPlacePipeline(
        marker_scene(),
        arm,
        Transform(),
        PipelineConfig(max_allowed_distance=1.0),
        ik_solver=MagicMock(side_effect=ValueError("unreachable")),
    )
    res = pipeline.run_cycle()
    assert res.success is False and res.stage == "ik_solve"
    arm.session.assert_not_called()


def test_pipeline_motion_rejection_reports_stage():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Idle")
    session = session_mock("Idle")
    session.move_to.side_effect = ArmCommandRejected({"code": "SOFT_LIMIT_EXCEEDED", "message": "joint_2"})
    arm.session.return_value.__enter__.return_value = session
    pipeline = DeterministicPickPlacePipeline(
        marker_scene(), arm, Transform(), PipelineConfig(max_allowed_distance=1.0), ik_solver=lambda pose: [0.0] * 6
    )
    res = pipeline.run_cycle()
    assert res.success is False and res.stage == "motion_execution"
    assert "SOFT_LIMIT_EXCEEDED" in res.error_message
