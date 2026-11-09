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

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

log = logging.getLogger(__name__)

DEFAULT_CAMERA_WS = "ws://192.168.42.1:8001/"
NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8
TS_TAIL_BYTES = 8
PYAV_EXTRA_AUS = 5  # AUs fed after the IDR when the decoder has not produced a frame yet
FFMPEG_TIMEOUT_S = 2.0
PIXEL_UNIT_THRESHOLD = 1.05  # cx|cy|w|h above this means pixels (cockpit parseDetections)


# --------------------------------------------------------------------------- errors


class SnapshotError(Exception):
    """Snapshot could not be produced (no decoder, decode failure)."""


class SnapshotTimeout(SnapshotError):
    """No IDR access unit arrived within the timeout."""


class CameraUnavailable(SnapshotError):
    """The camera WebSocket could not be opened or closed before an IDR arrived."""


class DetectorUnavailable(Exception):
    """armd's detector proxy answered 502 or could not be reached."""


# --------------------------------------------------------------------------- data


@dataclass
class Snapshot:
    jpeg: bytes
    width: int
    height: int
    device_ts_ms: int  # 8-byte LE tail of the access unit (device clock, may be unsynced)
    host_rx_ms: int  # host time.time() at receipt


@dataclass
class Detection:
    label: str
    score: float
    box_norm: tuple[float, float, float, float]  # x0, y0, x1, y1 in [0, 1], clamped to the frame
    track_id: Any = None
    center_norm: tuple[float, float] = (0.0, 0.0)  # detector centre, clamped to [0, 1]
    corners_norm: list[tuple[float, float]] | None = None  # marker corners (not clamped)

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "score": round(self.score, 4),
            "box_norm": [round(v, 4) for v in self.box_norm],
            "center_norm": [round(v, 4) for v in self.center_norm],
            "track_id": self.track_id,
        }
        if self.corners_norm is not None:
            d["corners_norm"] = [[round(u, 4), round(v, 4)] for u, v in self.corners_norm]
        return d


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
        except ConnectionClosed as exc:
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
        ws = ws_connect(url, open_timeout=min(open_timeout, timeout), max_size=None)
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
                with ws_connect(self.url, open_timeout=3.0, max_size=None) as ws:
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


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _flatten_xy(v: Any) -> list[Any]:
    if isinstance(v, dict):
        return [v.get("x"), v.get("y")]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _flatten_xy(e)]
    return [v]


def _corners(item: dict) -> list[tuple[float, float]] | None:
    """Four marker corners from ``corners`` / ``corner`` / ``polygon`` (also under ``pose`` / ``detail``)."""
    sources = [item.get(k) for k in ("pose", "pose3d", "pose_3d") if isinstance(item.get(k), dict)]
    det = item.get("detail") if isinstance(item.get("detail"), dict) else {}
    if isinstance(det.get("pose"), dict):
        sources.append(det["pose"])
    for src in [*sources[:1], item, det]:
        for k in ("corners", "corner", "polygon"):
            raw = src.get(k)
            if not isinstance(raw, list):
                continue
            flat = _flatten_xy(raw)
            nums = [_num(v) for v in flat[:8]]
            if len(nums) == 8 and all(n is not None for n in nums):
                return [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]  # type: ignore[misc]
    return None


def _items(raw: Any) -> tuple[list, float | None, float | None]:
    if isinstance(raw, list):
        if len(raw) >= 5 and isinstance(raw[0], (int, float)):
            return [raw], None, None
        return list(raw), None, None
    if not isinstance(raw, dict):
        return [], None, None
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    items: list = []
    if raw.get("format"):
        items = [raw]
    elif isinstance(raw.get("boxes"), list):
        items = raw["boxes"]
    elif isinstance(raw.get("detections"), list):
        items = raw["detections"]
    elif isinstance(data.get("boxes"), list):
        items = data["boxes"]
    elif isinstance(data.get("detections"), list):
        items = data["detections"]
    elif isinstance(raw.get("data"), list):
        items = raw["data"]
    elif isinstance(raw.get("results"), list):
        items = raw["results"]
    items = list(items)
    for k in ("markers", "aruco"):
        m = raw.get(k) if isinstance(raw.get(k), list) else data.get(k) if isinstance(data.get(k), list) else None
        if m:
            items += [{"label": "aruco", **x} if isinstance(x, dict) else x for x in m]
    ref_w = raw.get("width") or raw.get("w") or data.get("width") or data.get("w")
    ref_h = raw.get("height") or raw.get("h") or data.get("height") or data.get("h")
    for src in (raw, data):
        res = src.get("resolution")
        if not ref_w and isinstance(res, list) and len(res) >= 2:
            ref_w, ref_h = res[0], res[1]
    return items, _num(ref_w), _num(ref_h)


def _raw_box(item: Any, corners: list[tuple[float, float]] | None):
    """``(cx, cy, w, h, score, label, track_id)`` in the payload's units, or None to drop."""
    if isinstance(item, (list, tuple)) and len(item) >= 5:
        # SSCMA array [cx, cy, w, h, score, target(, track)], centre per the cockpit (index.html:2471);
        # needs verification against real yolo-detector output.
        vals = [_num(v) for v in item[:4]]
        if any(v is None for v in vals):
            return None
        cx, cy, w, h = vals  # type: ignore[misc]
        if w <= 0 or h <= 0:
            return None
        label = str(item[5]) if len(item) > 5 else "obj"
        return cx, cy, w, h, _num(item[4]), label, item[6] if len(item) > 6 else None
    if not isinstance(item, dict):
        return None
    fmt = item.get("format") or ("corners" if corners else None)
    if fmt == "corners":
        if not corners:
            return None
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w <= 0 or h <= 0:
            return None
        cx, cy = sum(xs) / 4, sum(ys) / 4
    elif fmt == "xywh_center":
        cx = _num(item.get("cx", item.get("x")))
        cy = _num(item.get("cy", item.get("y")))
        w = _num(item.get("w", item.get("width")))
        h = _num(item.get("h", item.get("height")))
        if None in (cx, cy, w, h) or w <= 0 or h <= 0:
            return None
    elif fmt == "xyxy":
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else item.get("box")
        bbox = bbox if isinstance(bbox, list) and len(bbox) >= 4 else None
        if "x1" in item and "x2" in item:
            x0, x1 = _num(item["x1"]), _num(item["x2"])
        elif "x0" in item and "x1" in item:
            x0, x1 = _num(item["x0"]), _num(item["x1"])
        elif bbox:
            x0, x1 = _num(bbox[0]), _num(bbox[2])
        else:
            return None
        if "y1" in item and "y2" in item:
            y0, y1 = _num(item["y1"]), _num(item["y2"])
        elif "y0" in item and "y1" in item:
            y0, y1 = _num(item["y0"]), _num(item["y1"])
        elif bbox:
            y0, y1 = _num(bbox[1]), _num(bbox[3])
        else:
            return None
        if None in (x0, y0, x1, y1) or x1 <= x0 or y1 <= y0:
            return None
        w, h, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    elif fmt == "xywh_top_left":
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, w, h = (_num(v) for v in bbox[:4])
        else:
            x0 = _num(item.get("x", item.get("x0")))
            y0 = _num(item.get("y", item.get("y0")))
            w = _num(item.get("w", item.get("width")))
            h = _num(item.get("h", item.get("height")))
        if None in (x0, y0, w, h) or w <= 0 or h <= 0:
            return None
        cx, cy = x0 + w / 2, y0 + h / 2
    else:
        return None
    score = item.get("score", item.get("confidence"))
    score = _num(score) if score is not None else (1.0 if corners else 0.0)  # marker decoders send no score
    label = next(
        (item[k] for k in ("label", "name", "target", "class_name") if item.get(k) is not None),
        "aruco" if corners else "obj",
    )
    track = item.get("track_id", item.get("id"))
    return cx, cy, w, h, score, str(label), track


def parse_detections(raw: Any, ref_res: tuple[int, int] | None = None) -> list[Detection]:
    """Detector payload -> normalised boxes (port of the cockpit's ``parseDetections``).

    Accepts the envelopes ``result`` / ``results`` / ``boxes`` / ``detections`` / ``data.*`` plus
    ``markers`` / ``aruco`` lists; items as SSCMA arrays ``[cx, cy, w, h, score, target(, track)]``
    or dicts with ``format`` ``xywh_center`` / ``xyxy`` / ``xywh_top_left`` / ``corners``. Values
    with ``cx|cy|w|h > 1.05`` are pixels and are divided by the payload ``width/height/resolution``,
    else by ``ref_res`` (usually the snapshot size); pixel items without a resolution are dropped.
    Scores are clamped to [0, 1]; boxes are clamped to the frame.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        inner = raw.get("result") or (raw.get("results") if isinstance(raw.get("results"), dict) else None)
        if isinstance(inner, dict) and not raw.get("boxes") and not raw.get("detections"):
            ref = ref_res or ((raw["width"], raw["height"]) if raw.get("width") and raw.get("height") else None)
            return parse_detections(inner, ref)
    items, ref_w, ref_h = _items(raw)
    if ref_res and len(ref_res) >= 2:
        ref_w = ref_w or _num(ref_res[0])
        ref_h = ref_h or _num(ref_res[1])
    has_ref = bool(ref_w and ref_h and ref_w > 0 and ref_h > 0)

    out: list[Detection] = []
    for item in items:
        corners = _corners(item) if isinstance(item, dict) else None
        box = _raw_box(item, corners)
        if box is None:
            continue
        cx, cy, w, h, score, label, track = box
        if score is None:
            continue
        score = max(0.0, min(1.0, score))
        pixel = max(cx, cy, w, h) > PIXEL_UNIT_THRESHOLD
        sx, sy = (ref_w, ref_h) if pixel else (1.0, 1.0)
        if pixel and not has_ref:
            continue
        clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
        x0, y0 = clamp((cx - w / 2) / sx), clamp((cy - h / 2) / sy)
        x1, y1 = clamp((cx + w / 2) / sx), clamp((cy + h / 2) / sy)
        if x1 - x0 <= 0.001 or y1 - y0 <= 0.001:
            continue
        out.append(
            Detection(
                label=label,
                score=score,
                box_norm=(x0, y0, x1, y1),
                track_id=track,
                center_norm=(clamp(cx / sx), clamp(cy / sy)),
                corners_norm=[(u / sx, v / sy) for u, v in corners] if corners else None,
            )
        )
    return out


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


class DetectionClient:
    """armd's detector proxy (``/api/v1/detection/*``, rebot-armd ``docs/detection_api.md``)."""

    def __init__(
        self,
        armd_url: str = "http://192.168.42.1:8765",
        timeout: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = armd_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self.trigger_supported: bool | None = None  # None until the first trigger call
        # perception API 2 (rebot-vision >= 1.3): triggers run the pipelines they name
        self.pipelines_supported: bool | None = None  # None until probed

    def _call(self, method: str, path: str, wait_ms: int = 0, **kw: Any) -> httpx.Response:
        # armd waits wait_ms (capped at 5000) + 2 s for the upstream.
        timeout = self.timeout + min(max(wait_ms, 0), 5000) / 1000.0 + 2.0
        try:
            with httpx.Client(timeout=timeout, transport=self._transport) as c:
                res = c.request(method, f"{self.base_url}{path}", **kw)
        except httpx.HTTPError as exc:
            raise DetectorUnavailable(f"armd detection proxy unreachable: {exc}") from exc
        if res.status_code == 502:
            raise DetectorUnavailable("detector unreachable (502 from armd)")
        return res

    def latest(self, after: int | None = None, wait_ms: int = 0) -> dict:
        params: dict = {"wait_ms": int(wait_ms)}
        if after is not None:
            params["after"] = int(after)
        res = self._call("GET", "/api/v1/detection/latest", wait_ms, params=params)
        res.raise_for_status()
        return res.json()

    def inference(self) -> dict:
        res = self._call("GET", "/api/v1/detection/inference")
        res.raise_for_status()
        return res.json()

    def set_inference(self, enabled: bool) -> dict:
        res = self._call("POST", "/api/v1/detection/inference", json={"enabled": bool(enabled)})
        res.raise_for_status()
        return res.json()

    def supports_pipelines(self) -> bool:
        """Whether the provider runs the pipelines a trigger names (perception API 2).

        Read once from ``GET /api/v1/vision/mode`` (``per_request_pipelines``); a provider
        without the route (404) or any other error counts as no.
        """
        if self.pipelines_supported is None:
            try:
                res = self._call("GET", "/api/v1/vision/mode")
                ok = res.status_code == 200 and res.json().get("per_request_pipelines") is True
            except (DetectorUnavailable, ValueError):
                return False  # not cached: the provider may come up later
            self.pipelines_supported = ok
        return self.pipelines_supported

    def detect_once(self, wait_ms: int = 2500, pipelines: Sequence[str] | None = None) -> dict:
        """One fresh inference result, normalised to ``{frame_id, ts_ms, raw, timeout, pipelines}``.

        Uses ``POST /api/v1/detection/trigger``. ``pipelines`` (``"yolo"`` / ``"aruco"``) is sent
        when the provider supports per-request pipelines (perception API 2); ``None`` then means
        both. Older providers run their current mode and ``pipelines`` is ignored. ``pipelines``
        in the result lists what ran on the frame when the provider reports it, else ``None``.
        Detector builds without the trigger answer 404; then inference is enabled, ``latest`` is
        long-polled for a newer frame, and inference is disabled again only if this call
        enabled it.
        """
        wait_ms = max(0, min(int(wait_ms), 5000))
        if pipelines is not None:
            pipelines = list(dict.fromkeys(pipelines))
            if not pipelines or any(p not in PIPELINES for p in pipelines):
                raise ValueError(f"pipelines must be a non-empty subset of {list(PIPELINES)}")
        if self.trigger_supported is not False:
            kw: dict = {"params": {"wait_ms": wait_ms}}
            if self.supports_pipelines():
                kw["json"] = {"pipelines": pipelines or list(PIPELINES)}
            res = self._call("POST", "/api/v1/detection/trigger", wait_ms, **kw)
            if res.status_code != 404:
                self.trigger_supported = True
                res.raise_for_status()
                body = res.json()
                return {
                    "frame_id": body.get("frame_id"),
                    "ts_ms": body.get("ts_ms"),
                    "timeout": bool(body.get("timeout")) or body.get("success") is False,
                    "pipelines": _ran(body),
                    "raw": body,
                }
            self.trigger_supported = False
        before = self.latest(wait_ms=0)
        enabled_here = False
        if not self.inference().get("enabled"):
            self.set_inference(True)
            enabled_here = True
        try:
            body = self.latest(after=before.get("frame_id") or 0, wait_ms=wait_ms)
        finally:
            if enabled_here:
                try:
                    self.set_inference(False)
                except (httpx.HTTPError, DetectorUnavailable) as exc:
                    log.warning("could not disable inference again: %s", exc)
        return {
            "frame_id": body.get("frame_id"),
            "ts_ms": body.get("ts_ms"),
            "timeout": bool(body.get("timeout")),
            "pipelines": _ran(body),
            "raw": body,
        }


PIPELINES = ("yolo", "aruco")


def _ran(body: dict) -> list[str] | None:
    """Pipelines that ran on a result frame (rebot-vision ``pipelines.*.ran``), else ``None``."""
    p = body.get("pipelines")
    if not isinstance(p, dict):
        return None
    return [k for k in PIPELINES if isinstance(p.get(k), dict) and p[k].get("ran") is True]


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
