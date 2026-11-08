#!/usr/bin/env python3
"""Camera intrinsics + distortion calibration for reCamera (OpenCV).

Two input modes:

* RTSP (interactive, default):  frames from ``rtsp://<host>:8554/live0`` (1280x720). A frame is captured
  automatically when the board is detected, has been still for a few frames and its position / size /
  tilt differs enough from the frames already captured. Captured frames are saved so the run can be
  repeated offline.
* Offline:  ``--images DIR`` with PNG/JPG files of the board.

Boards: chessboard (``--board checkerboard --cols/--rows`` = inner corners) or ChArUco
(``--board charuco --cols/--rows`` = squares, ``--marker`` = marker side, ``--dict``).

Output: a shell snippet for ``/etc/aruco-detector.conf`` with ``FX FY CX CY K1 K2 P1 P2 K3`` (OpenCV
order) at the inference resolution plus ``CAMERA_WIDTH CAMERA_HEIGHT``, and the reprojection RMS.

Resolution note: calibration runs at the stream resolution (1280x720). The detector infers on the
640x360 channel. Both come from the same sensor; this tool ASSUMES the same field of view (pure
scaling, no crop) and scales fx, fy, cx, cy by the width/height ratio. Verify on the device (e.g.
compare a marker distance measured by the detector against a tape measure) before relying on it.

Requires the optional extra:  uv sync --extra calib
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only without the extra
    sys.exit("OpenCV missing: install the optional extra with `uv sync --extra calib`")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
SCALING_NOTE = (
    "NOTE: intrinsics were scaled from the calibration resolution to the inference resolution assuming the "
    "same field of view (no crop). Verify this on the device."
)


@dataclass
class BoardSpec:
    kind: str = "checkerboard"  # checkerboard | charuco
    cols: int = 9  # checkerboard: inner corners per row; charuco: squares in x
    rows: int = 6  # checkerboard: inner corners per column; charuco: squares in y
    square: float = 0.025  # m
    marker: float = 0.018  # m, charuco only
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        self._charuco = None
        if self.kind == "charuco":
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
            board = cv2.aruco.CharucoBoard((self.cols, self.rows), self.square, self.marker, d)
            self._charuco = (board, cv2.aruco.CharucoDetector(board))
        elif self.kind != "checkerboard":
            raise ValueError(f"unknown board kind {self.kind!r}")

    def object_points(self) -> np.ndarray:
        """Checkerboard inner-corner coordinates in the board frame (z = 0), OpenCV order."""
        g = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return np.hstack([g * self.square, np.zeros((len(g), 1))]).astype(np.float32)

    def detect(self, gray: np.ndarray, fast: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (object points Nx3, image points Nx2) or None."""
        if self.kind == "checkerboard":
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            if fast:
                flags |= cv2.CALIB_CB_FAST_CHECK
            ok, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), flags)
            if not ok:
                return None
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
            return self.object_points(), corners.reshape(-1, 2).astype(np.float32)
        board, detector = self._charuco
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 6:
            return None
        obj, img = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < 6:
            return None
        return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


@dataclass
class IntrinsicsResult:
    K: np.ndarray
    dist: np.ndarray  # k1 k2 p1 p2 k3
    rms: float
    image_size: tuple[int, int]  # (w, h) at calibration
    n_views: int
    per_view_rms: list[float]


def calibrate(
    views: Sequence[tuple[np.ndarray, np.ndarray]], image_size: tuple[int, int], fix_k3: bool = False
) -> IntrinsicsResult:
    if len(views) < 4:
        raise ValueError(f"need >= 4 board views, got {len(views)}")
    obj = [v[0] for v in views]
    img = [v[1] for v in views]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, axis=1)))))
    d = np.zeros(5)
    flat = dist.ravel()
    d[: min(5, len(flat))] = flat[:5]
    return IntrinsicsResult(K, d, float(rms), image_size, len(views), per_view)


def scale_intrinsics(K: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> tuple[float, float, float, float]:
    """Scale fx, fy, cx, cy from resolution ``src`` (w, h) to ``dst`` assuming identical field of view.

    Uses the pixel-centre convention: ``c' = (c + 0.5) * s - 0.5``.
    """
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    return K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5


def to_conf(res: IntrinsicsResult, out_size: tuple[int, int]) -> str:
    fx, fy, cx, cy = scale_intrinsics(res.K, res.image_size, out_size)
    k1, k2, p1, p2, k3 = res.dist
    w0, h0 = res.image_size
    lines = [
        "# Camera intrinsics for /etc/aruco-detector.conf (generated by rebot-companion tools/calibrate_intrinsics.py)",
        f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')}; {res.n_views} views; reprojection RMS {res.rms:.4f} px "
        f"at {w0}x{h0}",
        f"# calibrated at {w0}x{h0}, scaled to {out_size[0]}x{out_size[1]} assuming the same field of view "
        "(verify on device)",
        f"CAMERA_WIDTH={out_size[0]}",
        f"CAMERA_HEIGHT={out_size[1]}",
        f"FX={fx:.4f}",
        f"FY={fy:.4f}",
        f"CX={cx:.4f}",
        f"CY={cy:.4f}",
        f"K1={k1:.8f}",
        f"K2={k2:.8f}",
        f"P1={p1:.8f}",
        f"P2={p2:.8f}",
        f"K3={k3:.8f}",
    ]
    return "\n".join(lines) + "\n"


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def collect_offline(directory: Path, board: BoardSpec) -> tuple[list, tuple[int, int], list[str]]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"no images ({', '.join(IMAGE_EXTS)}) in {directory}")
    views, size, used = [], None, []
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip {f.name}: unreadable")
            continue
        s = (gray.shape[1], gray.shape[0])
        if size is None:
            size = s
        elif s != size:
            raise ValueError(f"{f.name}: size {s} differs from {size}; all images must share one resolution")
        det = board.detect(gray)
        if det is None:
            print(f"  skip {f.name}: board not found")
            continue
        views.append(det)
        used.append(f.name)
    return views, size, used


def view_feature(img_pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Position, scale and perspective descriptor of a board detection (for diversity checks)."""
    w, h = size
    c = img_pts.mean(axis=0)
    hull = cv2.convexHull(img_pts.astype(np.float32))
    area = cv2.contourArea(hull)
    x, y, bw, bh = cv2.boundingRect(img_pts.astype(np.float32))
    # tilt proxy: how far the hull deviates from its bounding box and the box aspect ratio
    fill = area / max(bw * bh, 1.0)
    return np.array([c[0] / w, c[1] / h, math.sqrt(area / (w * h)), fill, bw / max(bh, 1.0) * 0.25])


def collect_rtsp(
    url: str,
    board: BoardSpec,
    target: int,
    save_dir: Path,
    stable_frames: int = 5,
    still_px: float = 1.0,
    min_diversity: float = 0.08,
) -> tuple[list, tuple[int, int]]:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {url}")
    save_dir.mkdir(parents=True, exist_ok=True)
    views, feats, size = [], [], None
    prev, still = None, 0
    last_msg = 0.0
    print(f"[*] streaming {url}; capturing {target} views into {save_dir} (Ctrl-C to stop early)")
    print("    move the board slowly across the whole image, near and far, tilted up/down/left/right;")
    print("    hold it still ~0.5 s at each spot.")
    try:
        while len(views) < target:
            ok, frame = cap.read()
            if not ok:
                print("[!] stream read failed; retrying")
                time.sleep(0.5)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            det = board.detect(gray, fast=True)
            status = "no board"
            if det is None:
                prev, still = None, 0
            else:
                pts = det[1]
                if prev is not None and len(prev) == len(pts):
                    still = still + 1 if float(np.mean(np.linalg.norm(prev - pts, axis=1))) < still_px else 0
                else:
                    still = 0
                prev = pts
                status = f"board ({len(pts)} pts), still {still}/{stable_frames}"
                if still >= stable_frames:
                    f = view_feature(pts, size)
                    dmin = min((float(np.linalg.norm(f - g)) for g in feats), default=1.0)
                    if dmin >= min_diversity:
                        views.append(det)
                        feats.append(f)
                        path = save_dir / f"view_{len(views):03d}.png"
                        cv2.imwrite(str(path), frame)
                        print(f"[+] captured {len(views)}/{target} ({path.name})")
                        still = 0
                    else:
                        status += " - too similar to a captured view, move/tilt the board"
            now = time.monotonic()
            if now - last_msg > 1.0:
                print(f"    {status}", flush=True)
                last_msg = now
    except KeyboardInterrupt:
        print("\n[*] stopped by user")
    finally:
        cap.release()
    if size is None:
        raise RuntimeError("no frames received")
    return views, size


def report(res: IntrinsicsResult, names: Sequence[str] | None = None) -> None:
    print(f"\nreprojection RMS: {res.rms:.4f} px over {res.n_views} views at {res.image_size[0]}x{res.image_size[1]}")
    worst = sorted(range(len(res.per_view_rms)), key=lambda i: -res.per_view_rms[i])[:5]
    for i in worst:
        label = names[i] if names else f"view {i + 1}"
        print(f"  {label}: {res.per_view_rms[i]:.3f} px")
    print("K =\n" + np.array2string(res.K, precision=3, suppress_small=True))
    print("dist (k1 k2 p1 p2 k3) = " + " ".join(f"{v:.6f}" for v in res.dist))
    if res.rms > 1.0:
        print("WARNING: RMS > 1 px - check board flatness, focus, motion blur and board dimensions.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="reCamera intrinsics calibration (OpenCV)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--images", type=Path, help="offline: directory of board images")
    src.add_argument("--host", default="192.168.42.1", help="reCamera host for RTSP (default 192.168.42.1)")
    ap.add_argument("--rtsp-url", default=None, help="override RTSP URL (default rtsp://<host>:8554/live0)")
    ap.add_argument("--board", choices=["checkerboard", "charuco"], default="checkerboard")
    ap.add_argument("--cols", type=int, default=9, help="checkerboard inner corners / charuco squares in x")
    ap.add_argument("--rows", type=int, default=6, help="checkerboard inner corners / charuco squares in y")
    ap.add_argument("--square", type=float, default=0.025, help="square side in metres")
    ap.add_argument("--marker", type=float, default=0.018, help="charuco marker side in metres")
    ap.add_argument("--dict", default="DICT_4X4_50", help="charuco ArUco dictionary name")
    ap.add_argument("--target", type=int, default=20, help="RTSP: number of views to capture")
    ap.add_argument("--save-dir", type=Path, default=None, help="RTSP: where to save captured frames")
    ap.add_argument("--infer-size", default="640x360", help="detector inference resolution WxH")
    ap.add_argument("--fix-k3", action="store_true", help="do not estimate k3 (recommended for < 15 views)")
    ap.add_argument("--out", type=Path, default=Path("aruco-intrinsics.conf"), help="conf snippet output")
    args = ap.parse_args(argv)

    board = BoardSpec(args.board, args.cols, args.rows, args.square, args.marker, args.dict)
    names = None
    if args.images:
        views, size, names = collect_offline(args.images, board)
        print(f"[*] board found in {len(views)} images")
    else:
        url = args.rtsp_url or f"rtsp://{args.host}:8554/live0"
        save_dir = args.save_dir or Path("intrinsics_frames") / time.strftime("%Y%m%d-%H%M%S")
        views, size = collect_rtsp(url, board, args.target, save_dir)
        print(f"[*] frames saved in {save_dir}; re-run offline with --images {save_dir}")
    if size is None or len(views) < 4:
        print(f"[!] need >= 4 views with the board, got {len(views)}")
        return 1

    res = calibrate(views, size, args.fix_k3)
    report(res, names)
    out_size = parse_size(args.infer_size)
    if abs(out_size[0] / out_size[1] - size[0] / size[1]) > 1e-3:
        print(
            f"WARNING: aspect ratio differs ({size[0]}x{size[1]} vs {out_size[0]}x{out_size[1]}): a crop is likely "
            "and plain scaling is wrong."
        )
    conf = to_conf(res, out_size)
    args.out.write_text(conf, encoding="utf-8")
    print("\n" + conf)
    print(SCALING_NOTE)
    print(f"[+] wrote {args.out}; merge these keys into /etc/aruco-detector.conf on the device and restart the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
