from unittest.mock import MagicMock
import numpy as np
import pytest

from rebot_companion.camera_client import ReCameraClient, MarkerDetection
from rebot_companion.arm_client import ArmdClient, ArmStatus
from rebot_companion.transforms import Transform
from rebot_companion.pipeline import (
    DeterministicPickPlacePipeline,
    PipelineConfig,
)


def test_pipeline_success():
    camera = MagicMock(spec=ReCameraClient)
    arm = MagicMock(spec=ArmdClient)

    # Arm healthy
    arm.get_observation.return_value = ArmStatus(
        connected=True, e_stopped=False,
        joint_positions=[0.0] * 6, joint_velocities=[0.0] * 6,
        joint_temperatures=[35.0] * 6, timestamp_ms=1000
    )

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

    arm.get_observation.return_value = ArmStatus(
        connected=True, e_stopped=True,  # E-STOP ACTIVE
        joint_positions=[0.0] * 6, joint_velocities=[0.0] * 6,
        joint_temperatures=[35.0] * 6, timestamp_ms=1000
    )

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

    arm.get_observation.return_value = ArmStatus(
        connected=True, e_stopped=False,
        joint_positions=[0.0] * 6, joint_velocities=[0.0] * 6,
        joint_temperatures=[35.0] * 6, timestamp_ms=1000
    )
    camera.trigger.return_value = []  # No markers

    cam_to_base = Transform()
    config = PipelineConfig(target_marker_id=42)
    pipeline = DeterministicPickPlacePipeline(camera, arm, cam_to_base, config)
    res = pipeline.run_cycle()

    assert res.success is False
    assert res.stage == "marker_detection"
    assert "#42 not detected" in res.error_message
