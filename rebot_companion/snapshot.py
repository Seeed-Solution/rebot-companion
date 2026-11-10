"""JPEG snapshots from the reCamera H.264 debug stream, plus detection overlay.

The camera app's debug stream (``ws://<camera>:8001/``) sends one binary WebSocket message per
H.264 access unit: Annex-B NAL units (SPS + PPS prepended on keyframes) followed by an 8-byte
little-endian unix-ms timestamp (rebot-armd ``src/api/index.html`` ``setupWS``). A snapshot waits
for the first IDR access unit, decodes it and encodes a JPEG.

Decoders: PyAV (``pip install 'rebot-companion[agent]'``) or the ``ffmpeg`` binary. ``annotate``
needs Pillow (also in the ``agent`` extra).

Detections come from armd's detector proxy (``/api/v1/detection/*``, rebot-armd
``docs/detection_api.md``); :func:`parse_detections` reads the same payload formats as the
cockpit's ``parseDetections``.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from rebot import _ws
from rebot.detection import (  # noqa: F401  (re-exported)
    PIPELINES,
    Detection,
    DetectionClient,
    DetectorUnavailable,
    parse_detections,
)

log = logging.getLogger(__name__)

DEFAULT_CAMERA_WS = "ws://192.168.42.1:8001/"
NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8
TS_TAIL_BYTES = 8
PYAV_EXTRA_AUS = 5  # AUs fed after the IDR when the decoder has not produced a frame yet
FFMPEG_TIMEOUT_S = 2.0


# --------------------------------------------------------------------------- errors


class SnapshotError(Exception):
    """Snapshot could not be produced (no decoder, decode failure)."""


class SnapshotTimeout(SnapshotError):
    """No IDR access unit arrived within the timeout."""


class CameraUnavailable(SnapshotError):
    """The camera WebSocket could not be opened or closed before an IDR arrived."""


# --------------------------------------------------------------------------- data


@dataclass
class Snapshot:
    jpeg: bytes
    width: int
    height: int
    device_ts_ms: int  # 8-byte LE tail of the access unit (device clock, may be unsynced)
    host_rx_ms: int  # host time.time() at receipt


# --------------------------------------------------------------------------- framing


def split_message(msg: bytes) -> tuple[bytes, int] | None:
    """``(access_unit, device_ts_ms)`` for one stream message, or None if it is too short."""
    if not isinstance(msg, (bytes, bytearray)) or len(msg) <= TS_TAIL_BYTES:
        return None
    return bytes(msg[:-TS_TAIL_BYTES]), int.from_bytes(msg[-TS_TAIL_BYTES:], "little")


def nal_types(au: bytes) -> list[int]:
    """NAL unit types found after ``00 00 01`` start codes (covers ``00 00 00 01`` too)."""
    out = []
    i = au.find(b"\x00\x00\x01")
    while i >= 0 and i + 3 < len(au):
        out.append(au[i + 3] & 0x1F)
        i = au.find(b"\x00\x00\x01", i + 3)
    return out


def is_idr(au: bytes) -> bool:
    return NAL_IDR in nal_types(au)


# --------------------------------------------------------------------------- decoders


def _have_pyav() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str = "auto") -> str:
    """``pyav`` or ``ffmpeg``; ``auto`` prefers PyAV, then ffmpeg on PATH."""
    if backend == "pyav":
        if not _have_pyav():
            raise SnapshotError("PyAV is not installed: pip install 'rebot-companion[agent]'")
        return backend
    if backend == "ffmpeg":
        if shutil.which("ffmpeg") is None:
            raise SnapshotError("ffmpeg is not on PATH")
        return backend
    if backend != "auto":
        raise ValueError("backend must be auto, pyav or ffmpeg")
    if _have_pyav():
        return "pyav"
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    raise SnapshotError("no decoder: pip install 'rebot-companion[agent]' or install ffmpeg")


class _PyAvDecoder:
    def __init__(self) -> None:
        import av

        self._ctx = av.CodecContext.create("h264", "r")

    def feed(self, au: bytes | None):
        """Decode one AU (None flushes). Returns the last decoded ``av.VideoFrame`` or None."""
        frames = []
        try:
            if au is None:
                for pkt in self._ctx.parse(b""):
                    frames += self._ctx.decode(pkt)
                frames += self._ctx.decode(None)
            else:
                for pkt in self._ctx.parse(au):
                    frames += self._ctx.decode(pkt)
        except Exception as exc:  # av.error.* (invalid data, EOF)
            log.debug("pyav decode: %s", exc)
        return frames[-1] if frames else None


def _encode_jpeg(image, quality: int, max_width: int | None) -> tuple[bytes, int, int]:
    if max_width and image.width > max_width:
        h = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, h))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=int(quality))
    return buf.getvalue(), image.width, image.height


def _jpeg_size(jpeg: bytes) -> tuple[int, int]:
    """Width/height from the JPEG SOF marker (no Pillow needed)."""
    i = 2
    while i + 9 < len(jpeg):
        if jpeg[i] != 0xFF:
            i += 1
            continue
        marker = jpeg[i + 1]
        seg = int.from_bytes(jpeg[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(jpeg[i + 7 : i + 9], "big"), int.from_bytes(jpeg[i + 5 : i + 7], "big")
        i += 2 + seg
    return 0, 0


def _ffmpeg_decode(aus: Sequence[bytes], quality: int, max_width: int | None) -> bytes:
    q = max(2, min(31, round(2 + (100 - int(quality)) * 29 / 100)))  # mjpeg -q:v: 2 best .. 31 worst
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "h264", "-i", "pipe:0", "-frames:v", "1"]
    if max_width:
        cmd += ["-vf", f"scale='min({int(max_width)},iw)':-2"]
    cmd += ["-f", "image2", "-c:v", "mjpeg", "-q:v", str(q), "pipe:1"]
    try:
        res = subprocess.run(cmd, input=b"".join(aus), capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError("ffmpeg did not produce a frame within 2 s") from exc
    if not res.stdout.startswith(b"\xff\xd8"):
        raise SnapshotError(f"ffmpeg decode failed: {res.stderr.decode(errors='replace').strip()[:200]}")
    return res.stdout


def decode_snapshot(
    aus: Iterable[tuple[bytes, int]],
    backend: str = "auto",
    jpeg_quality: int = 85,
    max_width: int | None = None,
) -> Snapshot:
    """Decode the first IDR access unit of ``aus`` (``(au, device_ts_ms)`` pairs) to a JPEG.

    Access units before the first IDR are skipped. PyAV gets up to 5 following AUs if the IDR
    alone produced no frame, then a flush; ffmpeg gets the IDR AU on stdin.
    """
    backend = resolve_backend(backend)
    it = iter(aus)
    for au, ts in it:  # noqa: B007 - au/ts of the IDR are used after the loop
        if is_idr(au):
            break
    else:
        raise SnapshotTimeout("no IDR access unit in the stream")
    rx = int(time.time() * 1000)
    if backend == "ffmpeg":
        jpeg = _ffmpeg_decode([au], jpeg_quality, max_width)
        w, h = _jpeg_size(jpeg)
        return Snapshot(jpeg, w, h, ts, rx)
    dec = _PyAvDecoder()
    frame = dec.feed(au)
    extra = 0
    while frame is None and extra < PYAV_EXTRA_AUS:
        nxt = next(it, None)
        if nxt is None:
            break
        frame = dec.feed(nxt[0])
        extra += 1
    if frame is None:
        frame = dec.feed(None)
    if frame is None:
        raise SnapshotError("decoder produced no frame from the IDR access unit")
    jpeg, w, h = _encode_jpeg(frame.to_image(), jpeg_quality, max_width)
    return Snapshot(jpeg, w, h, ts, rx)


# --------------------------------------------------------------------------- grabbing


def _stream_aus(ws, deadline: float) -> Iterable[tuple[bytes, int]]:
    """Yield ``(au, ts)`` from an open socket until ``deadline``; raise SnapshotTimeout if no IDR came."""
    got_idr = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if not got_idr:
                raise SnapshotTimeout("no IDR access unit within the timeout")
            return
        try:
            msg = ws.recv(timeout=remaining)
        except TimeoutError:
            continue
        except _ws.WebSocketClosed as exc:
            if not got_idr:
                raise CameraUnavailable(f"camera stream closed before an IDR arrived: {exc}") from exc
            return
        parsed = split_message(msg)
        if parsed is None:
            continue
        if not got_idr and is_idr(parsed[0]):
            got_idr = True
        yield parsed


def snapshot_once(
    url: str = DEFAULT_CAMERA_WS,
    timeout: float = 6.0,
    backend: str = "auto",
    jpeg_quality: int = 85,
    max_width: int | None = None,
    open_timeout: float = 3.0,
) -> Snapshot:
    """Connect, wait for an IDR (up to ``timeout`` s), decode one frame, close."""
    backend = resolve_backend(backend)
    deadline = time.monotonic() + timeout
    try:
        ws = _ws.connect(url, timeout=min(open_timeout, timeout))
    except Exception as exc:
        raise CameraUnavailable(f"cannot open {url}: {exc}") from exc
    with ws:
        aus = _stream_aus(ws, deadline)
        # After the IDR, following AUs are only needed for PyAV; give them a short extra window.
        return decode_snapshot(_bounded_after_idr(aus, 1.0), backend, jpeg_quality, max_width)


def _bounded_after_idr(aus: Iterable[tuple[bytes, int]], window_s: float) -> Iterable[tuple[bytes, int]]:
    end: float | None = None
    for au, ts in aus:
        yield au, ts
        if end is None and is_idr(au):
            end = time.monotonic() + window_s
        elif end is not None and time.monotonic() > end:
            return


class VideoGrabber:
    """Snapshots from the camera stream, one-shot or from a persistent background decoder.

    Without :meth:`start`, each :meth:`snapshot` connects, waits for an IDR and disconnects.
    After :meth:`start`, a background thread decodes every access unit (PyAV only) and
    :meth:`snapshot` returns the latest frame if it is younger than ``max_age_ms``.
    """

    def __init__(
        self,
        url: str = DEFAULT_CAMERA_WS,
        backend: str = "auto",
        jpeg_quality: int = 85,
        max_width: int | None = None,
    ) -> None:
        self.url = url
        self.backend = backend
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self._latest: tuple[Any, int, int] | None = None  # (PIL image, device_ts_ms, host_rx_ms)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def snapshot(
        self,
        timeout: float = 6.0,
        max_age_ms: int = 500,
        jpeg_quality: int | None = None,
        max_width: int | None = None,
    ) -> Snapshot:
        quality = self.jpeg_quality if jpeg_quality is None else jpeg_quality
        width = self.max_width if max_width is None else max_width
        with self._lock:
            latest = self._latest
        if latest is not None and int(time.time() * 1000) - latest[2] <= max_age_ms:
            jpeg, w, h = _encode_jpeg(latest[0], quality, width)
            return Snapshot(jpeg, w, h, latest[1], latest[2])
        return snapshot_once(self.url, timeout, self.backend, quality, width)

    def start(self) -> None:
        if self._thread is not None:
            return
        if resolve_backend(self.backend) != "pyav":
            raise SnapshotError("persistent mode needs PyAV")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="video-grabber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=3.0)
        with self._lock:
            self._latest = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with _ws.connect(self.url, timeout=3.0) as ws:
                    dec = _PyAvDecoder()
                    started = False
                    while not self._stop.is_set():
                        try:
                            msg = ws.recv(timeout=0.5)
                        except TimeoutError:
                            continue
                        parsed = split_message(msg)
                        if parsed is None:
                            continue
                        au, ts = parsed
                        started = started or is_idr(au)
                        if not started:
                            continue
                        frame = dec.feed(au)
                        if frame is not None:
                            with self._lock:
                                self._latest = (frame.to_image(), ts, int(time.time() * 1000))
            except Exception as exc:
                self.last_error = str(exc)
                log.debug("video grabber: %s", exc)
                self._stop.wait(1.0)


# --------------------------------------------------------------------------- detections


def annotate(snap: Snapshot, dets: Sequence[Detection], min_score: float = 0.0, quality: int = 85) -> Snapshot:
    """Draw boxes and ``label score`` on the snapshot (Pillow). Returns a new Snapshot."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(snap.jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(min(w, h) / 160))
    for d in dets:
        if d.score < min_score:
            continue
        x0, y0, x1, y1 = d.box_norm
        box = (round(x0 * (w - 1)), round(y0 * (h - 1)), round(x1 * (w - 1)), round(y1 * (h - 1)))
        draw.rectangle(box, outline=(140, 198, 63), width=lw)
        text = f"{d.label} {d.score:.2f}"
        ty = box[1] - 12 if box[1] >= 12 else box[1] + lw
        tb = draw.textbbox((box[0] + lw, ty), text)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((box[0] + lw, ty), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    return replace(snap, jpeg=buf.getvalue(), width=w, height=h)


def annotated_snapshot(
    grabber: VideoGrabber,
    detector: DetectionClient,
    min_score: float = 0.0,
    timeout: float = 6.0,
    jpeg_quality: int | None = None,
    max_width: int | None = None,
) -> tuple[Snapshot, dict]:
    """Snapshot with the latest detections drawn on it, plus metadata.

    Metadata: ``detections``, ``det_frame_id``, ``det_ts_ms``, ``det_age_vs_frame_ms``
    (``snap.device_ts_ms - det.ts_ms``) and ``note`` when inference is disabled. Normalised boxes
    are only correct when the detector input has the stream's aspect ratio (needs verification).
    """
    snap = grabber.snapshot(timeout=timeout, jpeg_quality=jpeg_quality, max_width=max_width)
    det = detector.latest(wait_ms=0)
    dets = [d for d in parse_detections(det, (snap.width, snap.height)) if d.score >= min_score]
    quality = grabber.jpeg_quality if jpeg_quality is None else jpeg_quality
    out = annotate(snap, dets, min_score, quality=quality)
    meta: dict = {
        "width": out.width,
        "height": out.height,
        "device_ts_ms": out.device_ts_ms,
        "detections": [d.to_dict() for d in dets],
        "det_frame_id": det.get("frame_id"),
        "det_ts_ms": det.get("ts_ms"),
        "det_age_vs_frame_ms": (snap.device_ts_ms - det["ts_ms"]) if isinstance(det.get("ts_ms"), int) else None,
    }
    if det.get("enabled") is False:
        meta["note"] = "inference disabled; boxes may be stale or empty"
    return out, meta


__all__ = [
    "Snapshot",
    "Detection",
    "VideoGrabber",
    "DetectionClient",
    "SnapshotError",
    "SnapshotTimeout",
    "CameraUnavailable",
    "DetectorUnavailable",
    "snapshot_once",
    "decode_snapshot",
    "annotate",
    "annotated_snapshot",
    "parse_detections",
    "split_message",
    "nal_types",
    "is_idr",
    "resolve_backend",
]
