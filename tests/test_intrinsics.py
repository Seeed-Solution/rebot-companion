"""Offline intrinsics calibration on synthetic checkerboard images rendered with known K / distortion."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_spec = importlib.util.spec_from_file_location(
    "calibrate_intrinsics", Path(__file__).resolve().parents[1] / "tools" / "calibrate_intrinsics.py"
)
ci = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ci
_spec.loader.exec_module(ci)

W, H = 960, 540
K_GT = np.array([[820.0, 0, 487.0], [0, 818.0, 266.0], [0, 0, 1]])
D_GT = np.array([-0.22, 0.07, 0.0008, -0.0006, 0.0])
COLS, ROWS, SQ = 9, 6, 0.03


def render(rvec, tvec, ss=2):
    """Ray-cast a checkerboard through the distorted camera; ss x ss supersampling."""
    R, _ = cv2.Rodrigues(rvec)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    acc = np.zeros((H, W))
    n = R[:, 2]
    for du in offs:
        for dv in offs:
            pts = np.stack([u + du, v + dv], axis=-1).reshape(-1, 1, 2).astype(np.float64)
            xy = cv2.undistortPointsIter(
                pts, K_GT, D_GT, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
            ).reshape(-1, 2)
            d = np.hstack([xy, np.ones((len(xy), 1))])
            lam = (n @ tvec) / (d @ n)
            P = d * lam[:, None] - tvec
            bxy = P @ R[:, :2]  # board-frame X, Y
            X, Y = bxy[:, 0], bxy[:, 1]
            inside = (X > -2 * SQ) & (X < (COLS + 1) * SQ) & (Y > -2 * SQ) & (Y < (ROWS + 1) * SQ)
            sq = (X >= -SQ) & (X < COLS * SQ) & (Y >= -SQ) & (Y < ROWS * SQ)
            black = sq & ((np.floor(X / SQ) + np.floor(Y / SQ)) % 2 == 0)
            val = np.where(inside, np.where(black, 20.0, 235.0), 128.0)
            acc += val.reshape(H, W)
    return (acc / (ss * ss)).astype(np.uint8)


def make_views(n=12, seed=0):
    rng = np.random.default_rng(seed)
    board_c = np.array([(COLS - 1) * SQ / 2, (ROWS - 1) * SQ / 2, 0])
    out = []
    while len(out) < n:
        ang = np.radians(rng.uniform([-30, -30, -20], [30, 30, 20]))
        R, _ = cv2.Rodrigues(ang)
        z = rng.uniform(0.45, 0.75)
        c = np.array([rng.uniform(-0.12, 0.12) * z, rng.uniform(-0.07, 0.07) * z, z])
        tvec = c - R @ board_c
        proj, _ = cv2.projectPoints(_obj(), cv2.Rodrigues(R)[0], tvec, K_GT, D_GT)
        p = proj.reshape(-1, 2)
        if p[:, 0].min() < 40 or p[:, 0].max() > W - 40 or p[:, 1].min() < 40 or p[:, 1].max() > H - 40:
            continue
        out.append((cv2.Rodrigues(R)[0].ravel(), tvec))
    return out


def _obj():
    return ci.BoardSpec(cols=COLS, rows=ROWS, square=SQ).object_points()


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("views")
    for i, (r, t) in enumerate(make_views()):
        cv2.imwrite(str(d / f"view_{i:02d}.png"), render(r, t))
    return d


def test_offline_calibration_recovers_ground_truth(image_dir, capsys):
    board = ci.BoardSpec("checkerboard", COLS, ROWS, SQ)
    views, size, used = ci.collect_offline(image_dir, board)
    assert size == (W, H) and len(views) >= 10
    res = ci.calibrate(views, size)
    with capsys.disabled():
        print(
            f"\n[intrinsics] views {res.n_views} rms {res.rms:.4f}px "
            f"K {np.round(res.K[[0, 1, 0, 1], [0, 1, 2, 2]], 2)} "
            f"dist {np.round(res.dist, 4)}"
        )
    assert res.rms < 0.3
    assert res.K[0, 0] == pytest.approx(K_GT[0, 0], rel=0.01)
    assert res.K[1, 1] == pytest.approx(K_GT[1, 1], rel=0.01)
    assert res.K[0, 2] == pytest.approx(K_GT[0, 2], abs=4)
    assert res.K[1, 2] == pytest.approx(K_GT[1, 2], abs=4)
    assert res.dist[0] == pytest.approx(D_GT[0], abs=0.03)
    assert res.dist[1] == pytest.approx(D_GT[1], abs=0.08)


def test_cli_offline_writes_conf(image_dir, tmp_path):
    out = tmp_path / "intr.conf"
    rc = ci.main(
        [
            "--images",
            str(image_dir),
            "--cols",
            str(COLS),
            "--rows",
            str(ROWS),
            "--square",
            str(SQ),
            "--infer-size",
            "640x360",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    kv = dict(line.split("=", 1) for line in out.read_text().splitlines() if line and not line.startswith("#"))
    assert set(kv) == {"FX", "FY", "CX", "CY", "K1", "K2", "P1", "P2", "K3", "CAMERA_WIDTH", "CAMERA_HEIGHT"}
    assert kv["CAMERA_WIDTH"] == "640" and kv["CAMERA_HEIGHT"] == "360"
    s = 640 / 960
    assert float(kv["FX"]) == pytest.approx(K_GT[0, 0] * s, rel=0.01)
    assert float(kv["CX"]) == pytest.approx((K_GT[0, 2] + 0.5) * s - 0.5, abs=3)
    assert "reprojection RMS" in out.read_text()


def test_scale_intrinsics_half_resolution():
    K = np.array([[1000.0, 0, 639.5], [0, 1000.0, 359.5], [0, 0, 1]])
    fx, fy, cx, cy = ci.scale_intrinsics(K, (1280, 720), (640, 360))
    assert (fx, fy) == (500.0, 500.0)
    assert (cx, cy) == (319.5, 179.5)  # image centre maps to image centre


def test_charuco_board_detection():
    board = ci.BoardSpec("charuco", 5, 4, 0.04, 0.03, "DICT_4X4_50")
    img = board._charuco[0].generateImage((500, 400), marginSize=20)
    det = board.detect(img)
    assert det is not None and len(det[0]) == (5 - 1) * (4 - 1)
