import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from rebot.client import ArmCommandRejected, ArmdClient, ArmSession, ArmStatus, SafeExitOutcome
from rebot_companion import kinematics as K
from rebot_companion.calibration import HandEyeResult
from rebot_companion.camera_client import MarkerDetection, ReCameraClient
from rebot_companion.ik import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_POSITIONS, B601IkAdapter, IkError
from rebot_companion.pipeline import (
    DeterministicPickPlacePipeline,
    PipelineConfig,
)
from rebot_companion.transforms import Transform


def status(state="Idle", torqued=False, positions=None) -> ArmStatus:
    positions = [0.0] * 6 if positions is None else list(positions)
    n = len(positions)
    return ArmStatus.from_dict(
        {
            "timestamp": 1.0,
            "positions": positions,
            "velocities": [0.0] * n,
            "torques": [0.0] * n,
            "temperatures": [35.0] * n,
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


def session_mock(state="Idle", positions=None):
    session = MagicMock(spec=ArmSession)
    session.latest_observation.return_value = status(state, torqued=state != "Idle", positions=positions)
    return session


def test_pipeline_without_ik_is_dry_run():
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
    assert res.success is False  # no IK solver: nothing moved, never reported as completed
    assert res.stage == "dry_run"
    assert "arm not moved" in res.error_message
    arm.session.assert_not_called()
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
    assert any("gripper not commanded" in w for w in res.warnings)


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


# ------------------------------------------------------------------ gripper + IK ---
HOME7 = list(HOME_POSITIONS)


def test_pipeline_commands_gripper_open_close_open():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Idle", positions=HOME7)
    session = session_mock("Idle", positions=HOME7)
    arm.session.return_value.__enter__.return_value = session
    ik = MagicMock(side_effect=lambda pose: [float(pose.translation[2]), 0, 0, 0, 0, 1.5707963, 0.0])
    cam_to_base = Transform.from_translation_euler([0.1, -0.1, 0.1], 0.0, 0.0, 0.0)
    res = DeterministicPickPlacePipeline(
        marker_scene(), arm, cam_to_base, PipelineConfig(max_allowed_distance=1.0), ik_solver=ik
    ).run_cycle()
    assert res.success is True and res.stage == "completed", res.error_message
    targets = [c.args[0] for c in session.move_to.call_args_list]
    assert [t[6] for t in targets] == [GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_CLOSED] + [
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
    ]
    assert targets[0][:6] == HOME7[:6]  # open in place first
    assert [t[0] for t in targets[1:]] == pytest.approx([0.58, 0.5, 0.5, 0.58, 0.05, 0.05])
    assert all(len(t) == 7 for t in targets)
    assert res.warnings == []


def test_ik_adapter_returns_full_armd_vector():
    ik = B601IkAdapter()
    target = Transform.from_rot_trans(np.eye(3), [0.30, 0.05, 0.10])
    q = ik(target)
    assert len(q) == 7
    assert q[4] == 0.0 and q[5] == pytest.approx(1.5707963) and q[6] == GRIPPER_OPEN
    tcp = K.fk(K.to_model(q))
    assert [tcp.x, tcp.y, tcp.z] == pytest.approx([0.30, 0.05, 0.10], abs=1e-9)
    assert tcp.pitch == pytest.approx(math.pi / 2, abs=1e-9)
    assert ik.solve(target, gripper=GRIPPER_CLOSED)[6] == GRIPPER_CLOSED
    keep = B601IkAdapter(j6=None, seed=[0.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.0])
    assert keep(target)[5] == pytest.approx(0.9)
    with pytest.raises(IkError):
        ik(Transform.from_rot_trans(np.eye(3), [2.0, 0.0, 0.1]))


def test_pipeline_with_ik_adapter_moves_seven_joint_targets():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Armed", torqued=True, positions=HOME7)
    session = session_mock("Armed", positions=HOME7)
    marker = Transform.from_translation_euler([0.3, 0.05, 0.4], 0.0, 0.0, 0.0)
    camera = MagicMock(spec=ReCameraClient)
    camera.trigger.return_value = [MarkerDetection(id=0, center=[0, 0], corners=[], pose=marker, reprojection_error=0)]
    cam_to_base = Transform.from_translation_euler([0.0, 0.0, -0.3], 0.0, 0.0, 0.0)
    res = DeterministicPickPlacePipeline(
        camera, arm, cam_to_base, PipelineConfig(place_position_base=(0.25, 0.15, 0.1)), B601IkAdapter(), session
    ).run_cycle()
    assert res.success is True, res.error_message
    targets = [c.args[0] for c in session.move_to.call_args_list]
    assert len(targets) == 7 and all(len(t) == 7 for t in targets)
    grasp = K.fk(K.to_model(targets[2]))
    assert [grasp.x, grasp.y, grasp.z] == pytest.approx([0.3, 0.05, 0.1], abs=1e-9)
    assert all(t[5] == pytest.approx(1.5707963) and t[4] == 0.0 for t in targets[1:])


# -------------------------------------------------------------------- eye_in_hand ---
X_EIH = Transform.from_translation_euler([0.05, 0.0, 0.04], 0.0, -90.0, 180.0)  # T_cam_to_gripper
Q_DET = [0.3, 0.4, -0.5, 0.2, 0.0, 1.5707963, 0.0]


def test_eye_in_hand_uses_arm_pose_at_detection():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Armed", torqued=True, positions=Q_DET)
    camera = marker_scene()
    res = DeterministicPickPlacePipeline(
        camera, arm, HandEyeResult("eye_in_hand", X_EIH), PipelineConfig(max_allowed_distance=5.0)
    ).run_cycle()
    assert res.stage in ("dry_run", "safety_check")
    marker = camera.trigger.return_value[0].pose
    expected = K.tcp_pose_from_joints(Q_DET) @ X_EIH @ marker
    np.testing.assert_allclose(res.target_pose_base.matrix, expected.matrix, atol=1e-12)
    # T_cam_to_gripper is never applied directly as cam_to_base
    assert not np.allclose(res.target_pose_base.matrix, (X_EIH @ marker).matrix)


def test_eye_in_hand_rejects_arm_motion_during_detection():
    arm = MagicMock(spec=ArmdClient)
    moved = list(Q_DET)
    moved[0] += 0.05
    arm.get_observation.side_effect = [status("Armed", True, Q_DET), status("Armed", True, moved)]
    res = DeterministicPickPlacePipeline(marker_scene(), arm, HandEyeResult("eye_in_hand", X_EIH)).run_cycle()
    assert res.success is False and res.stage == "frame_transform"
    assert "moved" in res.error_message


def test_eye_in_hand_rejects_moving_arm():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Moving", True, Q_DET)
    camera = marker_scene()
    res = DeterministicPickPlacePipeline(camera, arm, HandEyeResult("eye_in_hand", X_EIH)).run_cycle()
    assert res.stage == "health_check"
    camera.trigger.assert_not_called()


def test_motion_failure_on_external_session_runs_safe_exit():
    arm = MagicMock(spec=ArmdClient)
    arm.get_observation.return_value = status("Armed", torqued=True)
    session = session_mock("Armed")
    session.move_to.side_effect = [None, ArmCommandRejected({"code": "SOFT_LIMIT_EXCEEDED", "message": "joint_2"})]
    session.safe_exit.return_value = SafeExitOutcome("unloaded", "parked at the resting pose, torque off")
    res = DeterministicPickPlacePipeline(
        marker_scene(), arm, Transform(), PipelineConfig(max_allowed_distance=1.0), lambda p: [0.0] * 6, session
    ).run_cycle()
    assert res.stage == "motion_execution"
    session.safe_exit.assert_called_once()
    assert res.safe_exit.action == "unloaded" and "safe exit: unloaded" in res.error_message
