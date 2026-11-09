"""Snapshots from a fake camera WebSocket replaying a committed H.264 fixture.

``tests/fixtures/h264_320x240_3au.bin`` holds three stream messages, each prefixed with a 4-byte
little-endian length: a P access unit, an IDR access unit (SPS + PPS + IDR) and another P access
unit, each followed by the stream's 8-byte little-endian unix-ms tail. Generated with::

    ffmpeg -f lavfi -i testsrc=size=320x240:rate=10 -frames:v 4 -c:v libx264 -g 2 -bf 0 \\
        -pix_fmt yuv420p -x264-params aud=1:repeat-headers=1:scenecut=0 \\
        -bsf:v h264_mp4toannexb -f h264 raw.h264

then split on access unit delimiters, keeping AUs 2..4 (P, IDR, P) with tails 1762000000000 + 100*i.
"""

from __future__ import annotations

import shutil
import struct
import threading
import time
from pathlib import Path

import pytest
from websockets.sync.server import serve

from rebot_companion.snapshot import (
    CameraUnavailable,
    SnapshotTimeout,
    VideoGrabber,
    is_idr,
    nal_types,
    snapshot_once,
    split_message,
)

FIXTURE = Path(__file__).parent / "fixtures" / "h264_320x240_3au.bin"
IDR_TS = 1_762_000_000_100


def load_messages() -> list[bytes]:
    data = FIXTURE.read_bytes()
    out, i = [], 0
    while i < len(data):
        (n,) = struct.unpack_from("<I", data, i)
        out.append(data[i + 4 : i + 4 + n])
        i += 4 + n
    return out


class FakeCamera:
    """Sends the given messages (in order, repeated) to every client, one every ``period`` s."""

    def __init__(self, messages: list[bytes], period: float = 0.02, repeat: bool = False) -> None:
        self.messages = messages
        self.period = period
        self.repeat = repeat
        self.clients = 0
        self.stopped = threading.Event()

    def handler(self, ws) -> None:
        self.clients += 1
        try:
            while not self.stopped.is_set():
                for m in self.messages:
                    ws.send(m)
                    time.sleep(self.period)
                if not self.repeat:
                    self.stopped.wait()  # keep the socket open, send nothing more
        except Exception:
            pass


@pytest.fixture
def camera():
    servers = []

    def start(messages, **kw):
        cam = FakeCamera(messages, **kw)
        server = serve(cam.handler, "127.0.0.1", 0)
        servers.append((cam, server))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        cam.url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}/"
        return cam

    yield start
    for cam, server in servers:
        cam.stopped.set()
        server.shutdown()


def test_fixture_framing():
    msgs = load_messages()
    assert len(msgs) == 3
    parsed = [split_message(m) for m in msgs]
    assert [ts for _, ts in parsed] == [IDR_TS - 100, IDR_TS, IDR_TS + 100]
    assert [is_idr(au) for au, _ in parsed] == [False, True, False]
    types = nal_types(parsed[1][0])
    assert {7, 8, 5} <= set(types)
    assert split_message(b"12345678") is None  # 8 bytes: tail only


def _assert_jpeg(snap, w=320, h=240):
    assert snap.jpeg[:3] == b"\xff\xd8\xff" and snap.jpeg[-2:] == b"\xff\xd9"
    assert (snap.width, snap.height) == (w, h)
    assert snap.device_ts_ms == IDR_TS  # P frame before the IDR is skipped
    assert abs(snap.host_rx_ms - time.time() * 1000) < 5000


def test_snapshot_pyav(camera):
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    cam = camera(load_messages())
    _assert_jpeg(snapshot_once(cam.url, timeout=3.0, backend="pyav"))
    _assert_jpeg(snapshot_once(cam.url, timeout=3.0, backend="pyav", max_width=160, jpeg_quality=60), 160, 120)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_snapshot_ffmpeg(camera):
    cam = camera(load_messages())
    _assert_jpeg(snapshot_once(cam.url, timeout=3.0, backend="ffmpeg"))


def test_timeout_without_idr(camera):
    msgs = load_messages()
    cam = camera([msgs[0], msgs[2]], repeat=True)  # P frames only
    t0 = time.monotonic()
    with pytest.raises(SnapshotTimeout):
        snapshot_once(cam.url, timeout=0.5, backend="ffmpeg" if shutil.which("ffmpeg") else "auto")
    assert time.monotonic() - t0 < 2.0


def test_camera_unavailable():
    with pytest.raises(CameraUnavailable):
        snapshot_once("ws://127.0.0.1:9/", timeout=1.0, backend="ffmpeg" if shutil.which("ffmpeg") else "auto")


def test_persistent_grabber_returns_latest_frame(camera):
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    cam = camera(load_messages(), repeat=True)
    g = VideoGrabber(cam.url, backend="pyav")
    g.start()
    try:
        deadline = time.monotonic() + 3.0
        while g._latest is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert g._latest is not None
        clients = cam.clients
        snap = g.snapshot(max_age_ms=1000)
        assert snap.jpeg[:2] == b"\xff\xd8" and snap.width == 320
        assert cam.clients == clients  # served from the background decoder, no new connection
    finally:
        g.stop()
