import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rebot_companion import kinematics as K
from rebot_companion.calibration import (
    CalibrationError,
    HandEyeCalibration,
    HandEyeResult,
    gripper_pose,
    solve_point_registration,
)
from rebot_companion.transforms import Transform

# Web solveRigid reference (tools/gen_web_fixtures.mjs): inputs and outputs
RIGID_CAM = [
    [0.03882242217659951, -0.06022056341171264, 0.512687599658966],
    [0.1479574203491211, 0.12459797859191894, 0.3723679065704346],
    [0.19710404872894288, -0.13228976726531982, 0.4652193397283554],
    [0.1696053981781006, 0.09287363290786743, 0.5723671078681946],
    [-0.11844053268432618, 0.11769569814205169, 0.370673930644989],
    [0.086595319211483, 0.07163636684417725, 0.6089638471603394],
    [-0.18552083969116212, -0.04829713404178619, 0.39508217573165894],
    [0.11607437133789063, -0.056133270263671875, 0.6123346328735352],
    [0.029623937606811524, -0.025464284420013427, 0.4395009160041809],
]
RIGID_BASE = [
    [0.48043654597414903, -0.2903603947277844, 1.1428044114112854],
    [0.444885347937406, -0.08054688706672729, 1.0011413251161576],
    [0.6440979137835443, -0.24591490654814246, 1.09591407674551],
    [0.4781453785386111, -0.08923105541411436, 1.2023509421348573],
    [0.24480096393033396, -0.25769747410065313, 0.9987827526330948],
    [0.4313528208471359, -0.1580798405948339, 1.2399338760375975],
    [0.29985512171064366, -0.42692318143448693, 1.0242798228263854],
    [0.5351729671891499, -0.2393536764338888, 1.242182655453682],
    [0.45042324408544826, -0.26998872908108007, 1.0689139785766602],
]
RIGID_JS_R = [
    [0.7654421927025623, -0.6434791228343605, -0.005733071342261457],
    [0.643500460023151, 0.7654390116083396, 0.0032058474762856224],
    [0.0023254205397196864, -0.006143124967808834, 0.9999784269847741],
]
RIGID_JS_T = [0.4132896441181777, -0.2718548244159633, 0.6296220270863755]
RIGID_JS_RMSE = 0.0017208849001885567
RIGID_JS_MAX = 0.0027693439003313247
RIGID_JS_RES = [
    0.0018848474459602126,
    0.000907138696601349,
    0.0027693439003313247,
    0.002077395245771269,
    0.0010362052533720756,
    0.0024555702481183,
    0.0003714425794766311,
    0.001317561504678258,
    0.0011467920140047854,
]


def test_point_registration_matches_web_solve_rigid():
    r = solve_point_registration(RIGID_CAM, RIGID_BASE)
    np.testing.assert_allclose(r["R"], RIGID_JS_R, atol=1e-9)
    np.testing.assert_allclose(r["t"], RIGID_JS_T, atol=1e-9)
    assert r["rmse"] == pytest.approx(RIGID_JS_RMSE, abs=1e-10)
    assert r["max_error"] == pytest.approx(RIGID_JS_MAX, abs=1e-10)
    np.testing.assert_allclose(r["residuals"], RIGID_JS_RES, atol=1e-10)


def test_point_registration_degenerate_like_web():
    with pytest.raises(CalibrationError, match="collinear"):
        solve_point_registration([[0, 0, 0.5], [0.1, 0, 0.5], [0.2, 0, 0.5]], [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])


def test_point_method_eye_to_hand(tmp_path: Path):
    X_gt = Transform.from_translation_euler([0.40, -0.30, 0.60], 0.0, 0.0, 45.0)
    calib = HandEyeCalibration(mode="eye_to_hand")
    poses = [
        ([0.2, 0.1, 0.3], 0.0, 10.0, 20.0),
        ([0.3, 0.0, 0.25], 15.0, -10.0, 30.0),
        ([0.25, -0.15, 0.35], -20.0, 15.0, -10.0),
        ([0.15, 0.2, 0.4], 10.0, 20.0, -25.0),
        ([0.35, 0.1, 0.2], -15.0, -15.0, 40.0),
    ]
    for t, r, p, y in poses:
        G = Transform.from_translation_euler(t, r, p, y)
        calib.add_sample(G, X_gt.inverse() @ G)
    X = calib.solve()
    np.testing.assert_allclose(X.matrix, X_gt.matrix, atol=1e-9)
    assert calib.rmse < 1e-9
    assert len(calib.result.residuals) == 5

    f = tmp_path / "calib.json"
    calib.save_to_file(f)
    np.testing.assert_allclose(HandEyeCalibration.load_from_file(f).matrix, X.matrix, atol=1e-12)
    data = json.loads(f.read_text())
    assert data["result"]["euler_convention"] == "R = Rz(yaw) Ry(pitch) Rx(roll)"
    # re-solve from the saved samples
    again = HandEyeCalibration.load_samples(f)
    np.testing.assert_allclose(again.solve().matrix, X.matrix, atol=1e-9)


def _web_export(tmp_path: Path, with_result: bool) -> Path:
    joints_list = [
        [0.0, -1.0, -1.6, 0.6],
        [0.3, -0.9, -1.8, 0.9],
        [-0.3, -1.2, -1.4, 0.4],
        [0.15, -0.7, -2.0, 1.0],
        [-0.2, -1.1, -1.7, 0.2],
    ]
    X_gt = Transform.from_translation_euler([0.45, 0.05, 0.55], 170.0, 5.0, -95.0)
    samples = []
    for j in joints_list:
        tcp = K.fk(j)
        p_base = [tcp.x, tcp.y, tcp.z]
        p_cam = X_gt.inverse().apply_point(p_base).tolist()
        samples.append(
            {
                "joints": j + [0.0, 0.0, 0.0],
                "p_base": p_base,
                "tcp_pitch": tcp.pitch,
                "marker_id": 3,
                "p_cam": p_cam,
                "R_cam": None,
                "rpy_cam_deg": None,
                "corners": None,
                "ts": "2026-11-01T00:00:00Z",
            }
        )
    r = solve_point_registration([s["p_cam"] for s in samples], [s["p_base"] for s in samples])
    result = None
    if with_result:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = r["R"], r["t"]
        result = {
            "T_cam_to_base": T.tolist(),
            "translation_m": r["t"].tolist(),
            "euler_rpy_deg": [0, 0, 0],
            "euler_convention": "R = Rz(yaw) Ry(pitch) Rx(roll)",
            "rmse_m": r["rmse"],
            "max_error_m": r["max_error"],
            "residuals_m": r["residuals"].tolist(),
            "samples": 5,
        }
    doc = {
        "version": 1,
        "kind": "hand_eye_calibration",
        "setup": "eye_to_hand",
        "method": "point_registration_horn",
        "units": {"length": "m", "angle": "deg"},
        "marker_id": 3,
        "result": result,
        "samples": samples,
    }
    f = tmp_path / "web.json"
    f.write_text(json.dumps(doc))
    return f


def test_load_web_export(tmp_path: Path):
    f = _web_export(tmp_path, with_result=True)
    X_gt = Transform.from_translation_euler([0.45, 0.05, 0.55], 170.0, 5.0, -95.0)
    np.testing.assert_allclose(HandEyeCalibration.load_from_file(f).matrix, X_gt.matrix, atol=1e-9)
    calib = HandEyeCalibration.load_samples(f)
    assert calib.marker_id == 3 and len(calib.samples) == 5
    assert not calib.samples[0].has_rotation
    # gripper rotation reconstructed from joints via FK
    s1 = calib.samples[1]
    np.testing.assert_allclose(s1.gripper_pose.rotation, K.tcp_pose_from_joints(s1.joints).rotation, atol=1e-12)
    np.testing.assert_allclose(calib.solve().matrix, X_gt.matrix, atol=1e-9)
    # position-only samples cannot feed AX=XB
    with pytest.raises(CalibrationError, match="orientation"):
        HandEyeCalibration.load_samples(f, method="axxb").solve()


def test_load_web_export_without_result(tmp_path: Path):
    f = _web_export(tmp_path, with_result=False)
    with pytest.raises(ValueError, match="no solved result"):
        HandEyeCalibration.load_from_file(f)


def test_legacy_sdk_format(tmp_path: Path):
    T = Transform.from_translation_euler([0.1, 0.2, 0.3], 10.0, 20.0, 30.0)
    f = tmp_path / "legacy.json"
    f.write_text(
        json.dumps(
            {"mode": "eye_to_hand", "transform": {"translation": [0.1, 0.2, 0.3], "euler_deg": [10.0, 20.0, 30.0]}}
        )
    )
    np.testing.assert_allclose(HandEyeCalibration.load_from_file(f).matrix, T.matrix, atol=1e-12)


# ---------------------------------------------------------------- AX = XB ---


def _random_poses(rng, n):
    """Gripper poses in a realistic workspace with yaw/pitch/roll diversity."""
    out = []
    for _ in range(n):
        R = Rotation.from_euler("ZYX", rng.uniform([-60, 20, -30], [60, 80, 30]), degrees=True).as_matrix()
        t = rng.uniform([0.15, -0.25, 0.05], [0.40, 0.25, 0.35])
        out.append(Transform.from_rot_trans(R, t))
    return out


def _noisy(T: Transform, rng, rot_deg: float, trans_m: float) -> Transform:
    dR = Rotation.from_rotvec(rng.normal(size=3) * np.radians(rot_deg)).as_matrix()
    return Transform.from_rot_trans(dR @ T.rotation, T.translation + rng.normal(size=3) * trans_m)


def _errors(X: Transform, X_gt: Transform):
    rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(X.rotation.T @ X_gt.rotation).as_rotvec())))
    return rot, float(np.linalg.norm(X.translation - X_gt.translation))


AXXB_TOL = {"rot_deg": 1.0, "trans_m": 0.005}  # noise below gives ~0.5 deg per-sample rotation error
NOISE = {"rot_deg": 0.3, "trans_m": 0.001}


def test_axxb_eye_to_hand_noise(capsys):
    rng = np.random.default_rng(1)
    X_gt = Transform.from_translation_euler([0.55, -0.35, 0.60], 175.0, -5.0, 120.0)  # T_base_cam
    Y_gt = Transform.from_translation_euler([0.03, -0.01, 0.05], 0.0, 90.0, 0.0)  # T_gripper_marker
    calib = HandEyeCalibration(mode="eye_to_hand", method="axxb")
    for G in _random_poses(rng, 12):
        M = X_gt.inverse() @ G @ Y_gt
        calib.add_sample(_noisy(G, rng, 0.05, 0.0005), _noisy(M, rng, NOISE["rot_deg"], NOISE["trans_m"]))
    X = calib.solve()
    rot_err, t_err = _errors(X, X_gt)
    y_rot, y_t = _errors(calib.result.marker_offset, Y_gt)
    with capsys.disabled():
        print(
            f"\n[axxb eye_to_hand] X err: {rot_err:.4f} deg, {t_err * 1000:.3f} mm | "
            f"marker offset err: {y_rot:.4f} deg, {y_t * 1000:.3f} mm | rmse {calib.rmse * 1000:.3f} mm, "
            f"rot rmse {calib.result.rot_rmse_deg:.3f} deg"
        )
    assert rot_err < AXXB_TOL["rot_deg"] and t_err < AXXB_TOL["trans_m"]
    assert y_rot < AXXB_TOL["rot_deg"] and y_t < AXXB_TOL["trans_m"]
    assert len(calib.result.residuals) == 12 and len(calib.result.rot_residuals_deg) == 12
    assert 0 < calib.rmse < 0.01


def test_axxb_eye_in_hand_noise(tmp_path: Path, capsys):
    rng = np.random.default_rng(2)
    X_gt = Transform.from_translation_euler([0.05, 0.0, 0.04], 0.0, -90.0, 180.0)  # T_gripper_cam
    Y_gt = Transform.from_translation_euler([0.35, 0.05, 0.0], 0.0, 0.0, 30.0)  # T_base_marker (on table)
    calib = HandEyeCalibration(mode="eye_in_hand", method="axxb")
    for G in _random_poses(rng, 12):
        M = (G @ X_gt).inverse() @ Y_gt
        calib.add_sample(_noisy(G, rng, 0.05, 0.0005), _noisy(M, rng, NOISE["rot_deg"], NOISE["trans_m"]))
    X = calib.solve()
    rot_err, t_err = _errors(X, X_gt)
    with capsys.disabled():
        print(
            f"\n[axxb eye_in_hand] X err: {rot_err:.4f} deg, {t_err * 1000:.3f} mm | rmse "
            f"{calib.rmse * 1000:.3f} mm, rot rmse {calib.result.rot_rmse_deg:.3f} deg"
        )
    assert rot_err < AXXB_TOL["rot_deg"] and t_err < AXXB_TOL["trans_m"]
    f = tmp_path / "eih.json"
    calib.save_to_file(f)
    data = json.loads(f.read_text())
    assert "T_cam_to_gripper" in data["result"] and "T_marker_to_base" in data["result"]
    loaded = HandEyeCalibration.load_result(f)
    assert loaded.setup == "eye_in_hand"
    np.testing.assert_allclose(loaded.transform.matrix, X.matrix, atol=1e-12)
    with pytest.raises(ValueError, match="eye_in_hand"):
        HandEyeCalibration.load_from_file(f)  # T_cam_to_gripper is never returned as cam_to_base
    again = HandEyeCalibration.load_samples(f, method="axxb")
    np.testing.assert_allclose(again.solve().matrix, X.matrix, atol=1e-9)


def test_axxb_exact_recovery_both_modes():
    rng = np.random.default_rng(3)
    X_gt = Transform.from_translation_euler([0.5, -0.3, 0.6], 170.0, 10.0, 100.0)
    Y_gt = Transform.from_translation_euler([0.02, 0.0, 0.03], 0.0, 80.0, 10.0)
    for mode in ("eye_to_hand", "eye_in_hand"):
        calib = HandEyeCalibration(mode=mode, method="axxb")
        for G in _random_poses(rng, 6):
            M = X_gt.inverse() @ G @ Y_gt if mode == "eye_to_hand" else (G @ X_gt).inverse() @ Y_gt
            calib.add_sample(G, M)
        np.testing.assert_allclose(calib.solve().matrix, X_gt.matrix, atol=1e-9)
        assert calib.rmse < 1e-9


def test_axxb_with_fk_poses_two_axes():
    """Real arm: orientation diversity only from J1 yaw and wrist pitch (two axes) -- still solvable."""
    X_gt = Transform.from_translation_euler([0.55, -0.35, 0.60], 175.0, -5.0, 120.0)
    Y_gt = Transform.from_translation_euler([0.0, 0.0, 0.02], 0.0, 0.0, 0.0)
    calib = HandEyeCalibration(mode="eye_to_hand", method="axxb")
    for q1, q2, q3, q4 in itertools.product([-0.4, 0.3], [-1.2, -0.8], [-1.6], [0.3, 1.0]):
        G = K.tcp_pose([q1, q2, q3, q4])
        calib.add_sample(G, X_gt.inverse() @ G @ Y_gt)
    np.testing.assert_allclose(calib.solve().matrix, X_gt.matrix, atol=1e-8)


def test_axxb_degenerate_motion():
    X_gt = Transform.from_translation_euler([0.5, -0.3, 0.6], 170.0, 10.0, 100.0)
    # pure translations -> no rotation at all
    calib = HandEyeCalibration(mode="eye_to_hand", method="axxb")
    for t in ([0.2, 0, 0.2], [0.3, 0.1, 0.2], [0.25, -0.1, 0.3], [0.3, 0.0, 0.1]):
        G = Transform.from_translation_euler(t, 0, 45, 0)
        calib.add_sample(G, X_gt.inverse() @ G)
    with pytest.raises(CalibrationError, match="insufficient rotation"):
        calib.solve()
    # rotations about a single axis (J1 yaw only) -> parallel axes
    calib = HandEyeCalibration(mode="eye_to_hand", method="axxb")
    for yaw in (-40, -10, 20, 50):
        G = K.tcp_pose([np.radians(yaw), -1.0, -1.6, 0.6])
        calib.add_sample(G, X_gt.inverse() @ G)
    with pytest.raises(CalibrationError, match="diversity"):
        calib.solve()


def test_mode_method_validation():
    with pytest.raises(ValueError):
        HandEyeCalibration(mode="eye_in_hand", method="point")
    with pytest.raises(ValueError):
        HandEyeCalibration(method="tsai")
    with pytest.raises(CalibrationError):
        HandEyeCalibration().solve()
    with pytest.raises(RuntimeError):
        HandEyeCalibration().save_to_file("/nonexistent/x.json")


def test_axxb_matches_opencv_park():
    """Cross-check against cv2.calibrateHandEye(PARK) (eye-in-hand convention)."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(4)
    X_gt = Transform.from_translation_euler([0.05, 0.0, 0.04], 0.0, -90.0, 180.0)
    Y_gt = Transform.from_translation_euler([0.35, 0.05, 0.0], 0.0, 0.0, 30.0)
    calib = HandEyeCalibration(mode="eye_in_hand", method="axxb")
    for G in _random_poses(rng, 10):
        calib.add_sample(_noisy(G, rng, 0.05, 0.0005), _noisy((G @ X_gt).inverse() @ Y_gt, rng, 0.3, 0.001))
    X = calib.solve()
    R, t = cv2.calibrateHandEye(
        [s.gripper_pose.rotation for s in calib.samples],
        [s.gripper_pose.translation for s in calib.samples],
        [s.marker_pose.rotation for s in calib.samples],
        [s.marker_pose.translation for s in calib.samples],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    np.testing.assert_allclose(X.rotation, R, atol=1e-6)
    np.testing.assert_allclose(X.translation, t.ravel(), atol=1e-6)


# ---------------------------------------------------------- setup-aware loading ---
def test_load_result_eye_to_hand(tmp_path: Path):
    X = Transform.from_translation_euler([0.3, -0.4, 0.5], 180.0, 0.0, 90.0)
    f = tmp_path / "e2h.json"
    f.write_text(
        json.dumps(
            {"kind": "hand_eye_calibration", "setup": "eye_to_hand", "result": {"T_cam_to_base": X.matrix.tolist()}}
        )
    )
    r = HandEyeCalibration.load_result(f)
    assert r.setup == "eye_to_hand" and not r.needs_joints
    np.testing.assert_allclose(r.cam_to_base().matrix, X.matrix, atol=1e-12)
    np.testing.assert_allclose(r.cam_to_base([0.3] * 7).matrix, X.matrix, atol=1e-12)  # joints ignored
    np.testing.assert_allclose(HandEyeCalibration.load_from_file(f).matrix, X.matrix, atol=1e-12)


def test_eye_in_hand_cam_to_base_uses_fk_at_detection(tmp_path: Path):
    X = Transform.from_translation_euler([0.05, 0.0, 0.04], 0.0, -90.0, 180.0)  # T_gripper_cam
    f = tmp_path / "eih.json"
    f.write_text(
        json.dumps(
            {"kind": "hand_eye_calibration", "setup": "eye_in_hand", "result": {"T_cam_to_gripper": X.matrix.tolist()}}
        )
    )
    r = HandEyeCalibration.load_result(f)
    assert r.needs_joints
    with pytest.raises(ValueError, match="joints"):
        r.cam_to_base()
    q_a = [0.2, 0.3, -0.4, 0.1, 0.0, 1.5707963, 0.0]
    q_b = [-0.5, 0.1, 0.2, -0.3, 0.0, 1.5707963, -2.0]
    for q in (q_a, q_b):
        expected = K.tcp_pose_from_joints(q) @ X
        np.testing.assert_allclose(r.cam_to_base(q).matrix, expected.matrix, atol=1e-12)
        np.testing.assert_allclose(gripper_pose(q).matrix, K.tcp_pose_from_joints(q).matrix, atol=1e-12)
    assert not np.allclose(r.cam_to_base(q_a).matrix, X.matrix)
    assert not np.allclose(r.cam_to_base(q_a).matrix, r.cam_to_base(q_b).matrix)


def test_load_result_rejects_ambiguous_files(tmp_path: Path):
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps({"transform": {"translation": [0.1, 0.2, 0.3], "euler_deg": [0, 0, 0]}}))
    with pytest.raises(ValueError, match="cannot tell"):
        HandEyeCalibration.load_result(f)
    f.write_text(json.dumps({"setup": "eye_to_hand", "result": {"T_cam_to_gripper": np.eye(4).tolist()}}))
    with pytest.raises(ValueError, match="does not match"):
        HandEyeCalibration.load_result(f)
    with pytest.raises(ValueError):
        HandEyeResult(setup="eye_on_base", transform=Transform())


# ------------------------------------------- FK poses with the wrist joints moving ---
def _wrist_joint_samples(rng, n):
    """Raw armd 7-joint vectors with J5 (swing) and J6 (roll) away from 0, gripper arbitrary."""
    out = []
    while len(out) < n:
        q = [
            rng.uniform(-0.8, 0.8),
            rng.uniform(-1.4, -0.6),
            rng.uniform(-2.0, -1.2),
            rng.uniform(0.0, 1.2),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-2.5, 2.5),
            rng.uniform(-5.7, 0.0),
        ]
        if abs(q[4]) > 0.2 and abs(q[5] - np.pi / 2) > 0.2:
            out.append(q)
    return out


@pytest.mark.parametrize("mode,method", [("eye_to_hand", "point"), ("eye_to_hand", "axxb"), ("eye_in_hand", "axxb")])
def test_recover_x_with_nonzero_j5_j6(mode, method, tmp_path: Path):
    rng = np.random.default_rng(42)
    X_gt = (
        Transform.from_translation_euler([0.55, -0.35, 0.60], 175.0, -5.0, 120.0)
        if mode == "eye_to_hand"
        else Transform.from_translation_euler([0.05, 0.0, 0.04], 0.0, -90.0, 180.0)
    )
    Y_gt = (
        Transform.from_matrix(np.eye(4))
        if method == "point"
        else Transform.from_translation_euler([0.03, -0.01, 0.05], 10.0, 80.0, -20.0)
        if mode == "eye_to_hand"
        else Transform.from_translation_euler([0.35, 0.05, 0.0], 0.0, 0.0, 30.0)
    )
    calib = HandEyeCalibration(mode=mode, method=method)
    for q in _wrist_joint_samples(rng, 10):
        G = gripper_pose(q)
        # J5 / J6 change the TCP pose: ignoring them would give a different gripper pose
        assert not np.allclose(G.matrix, gripper_pose(q[:4] + [0.0, 0.0, 0.0]).matrix, atol=1e-3)
        M = X_gt.inverse() @ G @ Y_gt if mode == "eye_to_hand" else (G @ X_gt).inverse() @ Y_gt
        calib.add_sample(G, M, joints=q)
    np.testing.assert_allclose(calib.solve().matrix, X_gt.matrix, atol=1e-8)
    assert calib.rmse < 1e-8
    # re-solve from the saved samples (gripper pose stored as T_base_gripper)
    f = tmp_path / "s.json"
    calib.save_to_file(f)
    np.testing.assert_allclose(HandEyeCalibration.load_samples(f, method=method).solve().matrix, X_gt.matrix, atol=1e-8)
