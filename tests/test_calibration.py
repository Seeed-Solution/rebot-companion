import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from rebot_companion.calibration import HandEyeCalibration
from rebot_companion.transforms import Transform


def test_hand_eye_calibration_synthetic(tmp_path: Path):
    # Set up ground truth Eye-to-Hand transform: Camera is at [0.4, -0.3, 0.6] with 45 deg yaw
    X_gt = Transform.from_translation_euler([0.40, -0.30, 0.60], 0.0, 0.0, 45.0)

    # In Eye-to-Hand: p_base = X_gt @ p_cam
    # For multiple distinct poses, generate gripper poses and marker observations
    calib = HandEyeCalibration(mode="eye_to_hand")

    test_positions = [
        ([0.2, 0.1, 0.3], 0.0, 10.0, 20.0),
        ([0.3, 0.0, 0.25], 15.0, -10.0, 30.0),
        ([0.25, -0.15, 0.35], -20.0, 15.0, -10.0),
        ([0.15, 0.2, 0.4], 10.0, 20.0, -25.0),
        ([0.35, 0.1, 0.2], -15.0, -15.0, 40.0),
    ]

    for t, r, p, y in test_positions:
        T_gripper = Transform.from_translation_euler(t, r, p, y)
        # Marker pose in camera frame: T_marker_cam = X_gt^-1 @ T_gripper
        T_marker_cam = X_gt.inverse() @ T_gripper
        calib.add_sample(T_gripper, T_marker_cam)

    X_solved = calib.solve()

    # Verify solved transform matches ground truth
    np.testing.assert_allclose(X_solved.translation, X_gt.translation, atol=1e-4)
    np.testing.assert_allclose(X_solved.rotation, X_gt.rotation, atol=1e-4)
    assert calib.rmse < 1e-4

    # Test saving and loading
    calib_file = tmp_path / "test_calib.json"
    calib.save_to_file(calib_file)
    assert calib_file.exists()

    loaded = HandEyeCalibration.load_from_file(calib_file)
    np.testing.assert_allclose(loaded.matrix, X_solved.matrix, atol=1e-6)
