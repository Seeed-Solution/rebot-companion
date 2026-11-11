"""Record reBot arm + reCamera episodes into a LeRobot dataset (``codebase_version`` v3.0).

Target format: what ``lerobot==0.6.1`` writes and reads (``docs/lerobot_dataset.md``). The
recorder needs PyAV and pyarrow (``pip install 'rebot-companion[record]'``), not lerobot itself.

Sources (both read-only, no control lease is taken):

* armd observations: ``ws://<armd>:8765/api/v1/ws?obs_hz=N``. Connecting without
  ``acquire_lease`` only receives the observation stream, so a teleop client or a program can
  hold the lease while this records.
* reCamera debug stream: ``ws://<camera>:8001/``, one H.264 Annex-B access unit per message plus
  an 8-byte little-endian unix-ms tail (:mod:`rebot_companion.snapshot`). Every access unit is
  decoded with PyAV, the newest decoded frame is kept.

Per dataset frame (fps ``F``):

* ``observation.state`` = ``positions`` (7: joints 1..6 + gripper, rad)
* ``action`` = ``commanded_positions`` (the position target armd is following; ``positions`` on an
  armd that does not report it)
* optional ``observation.velocity`` = ``velocities``, ``observation.effort`` = ``torques``
* ``observation.images.<name>`` = the camera frame, re-encoded to one mp4 frame per row

Clock model: every message gets the host ``time.monotonic()`` at receipt. Episode tick ``k`` is
due at ``T0 + k / F`` (host monotonic); row ``k`` takes the newest observation and the newest
decoded camera frame received at or before that instant (zero-order hold). armd's ``timestamp``
(seconds since armd start) and the camera's unix-ms tail come from other clocks and are not used
for alignment; they are kept in ``meta/rebot_episodes.jsonl`` with the per-episode sample ages.
Network, encode and decode latency is not compensated: a camera frame is older than its receipt
time by the camera pipeline delay (not measured here). A late tick (host busy) still produces one
row per period, so ``timestamp = frame_index / F`` holds exactly.

Scripted recording without the live sources: ``start_episode(sample=False)`` then
:meth:`LeRobotRecorder.add_frame` per step.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import queue
import shutil
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rebot import _ws
from rebot._http import host_url
from rebot.client import ArmStatus

log = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
LEROBOT_VERSION = "0.6.1"  # format reference; docs/lerobot_dataset.md
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
EXTRAS = {"velocity": ("observation.velocity", "velocities"), "effort": ("observation.effort", "torques")}
ARMD_PORT = 8765
CAMERA_PORT = 8001
STATS_IMAGE_EVERY_S = 0.5  # one camera frame per 0.5 s goes into the image stats


class RecorderError(RuntimeError):
    """Recording could not start or continue."""


def armd_url(host: str) -> str:
    """``192.168.16.1`` -> ``http://192.168.16.1:8765``; IPv6 literals are bracketed; full URLs pass through."""
    return host_url("http", host, ARMD_PORT)


def camera_url(host: str) -> str:
    """``192.168.16.1`` -> ``ws://192.168.16.1:8001/``; IPv6 literals are bracketed; full URLs pass through."""
    if "://" in host:
        return host
    return host_url("ws", host, CAMERA_PORT) + "/"


# --------------------------------------------------------------------------- sources


class _Stream:
    """Background WebSocket reader keeping the newest parsed messages with their host receive times.

    ``history`` recent items are kept so the sampler can pick, for a tick that runs late, the item
    received at or before the tick's due time instead of a newer one.
    """

    name = "stream"
    history = 128

    def __init__(self, url: str, connect_timeout: float = 5.0) -> None:
        self.url = url
        self.connect_timeout = connect_timeout
        self.error: str | None = None
        self.count = 0
        self._latest: tuple[float, Any] | None = None
        self._history: collections.deque = collections.deque(maxlen=self.history)
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._ws = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._ws = _ws.connect(self.url, timeout=self.connect_timeout)
        except Exception as exc:
            raise RecorderError(f"cannot open {self.url}: {exc}") from exc
        self._thread = threading.Thread(target=self._run, name=f"rec-{self.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    msg = self._ws.recv(timeout=0.5)
                except TimeoutError:
                    continue
                self.handle(msg, time.monotonic())
        except _ws.WebSocketClosed as exc:
            self.error = f"{self.url} closed: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"{self.url} reader failed: {exc}"
        with self._cond:
            self._cond.notify_all()

    def handle(self, msg: Any, rx: float) -> None:
        item = self.parse(msg)
        if item is None:
            return
        with self._cond:
            self._latest = (rx, item)
            self._history.append(self._latest)
            self.count += 1
            self._cond.notify_all()

    def parse(self, msg: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def latest(self) -> tuple[float, Any] | None:
        with self._cond:
            return self._latest

    def at_or_before(self, t: float) -> tuple[float, Any] | None:
        """Newest ``(rx, item)`` received at or before host time ``t``; None if the history is newer."""
        with self._cond:
            for entry in reversed(self._history):
                if entry[0] <= t:
                    return entry
        return None

    def wait_first(self, timeout: float) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None or self.error is not None, timeout)
        if self._latest is None:
            raise RecorderError(self.error or f"no data from {self.url} within {timeout:g} s")

    def close(self) -> None:
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)


class ObservationStream(_Stream):
    """armd ``/api/v1/ws`` observations without a lease; items are :class:`rebot.client.ArmStatus`."""

    name = "armd"

    def __init__(self, armd: str, obs_hz: int | None = None, connect_timeout: float = 5.0) -> None:
        base = armd_url(armd).replace("http://", "ws://")
        super().__init__(f"{base}/api/v1/ws" + (f"?obs_hz={int(obs_hz)}" if obs_hz else ""), connect_timeout)

    def parse(self, msg: Any) -> ArmStatus | None:
        if not isinstance(msg, str):
            return None
        try:
            d = json.loads(msg)
        except ValueError:
            return None
        if isinstance(d, dict) and "type" not in d and "positions" in d and "safety_state" in d:
            return ArmStatus.from_dict(d)
        return None


@dataclass
class CameraFrame:
    frame: Any  # av.VideoFrame
    device_ts_ms: int
    seq: int


class CameraStream(_Stream):
    """reCamera H.264 debug stream decoded with PyAV; items are :class:`CameraFrame`."""

    name = "camera"
    history = 32  # decoded frames are large; about 1 s at 30 fps

    def __init__(self, url: str, connect_timeout: float = 5.0) -> None:
        super().__init__(camera_url(url), connect_timeout)
        from .snapshot import _PyAvDecoder, resolve_backend

        resolve_backend("pyav")
        self._decoder = _PyAvDecoder()
        self._seq = 0

    def parse(self, msg: Any) -> CameraFrame | None:
        from .snapshot import split_message

        parsed = split_message(msg)
        if parsed is None:
            return None
        frame = self._decoder.feed(parsed[0])
        if frame is None:
            return None
        self._seq += 1
        return CameraFrame(frame, parsed[1], self._seq)


def _sample_at(src: Any, due: float, prev: tuple[float, Any] | None) -> tuple[float, Any] | None:
    """The item ``src`` received at or before ``due`` (never a later one, so ages are >= 0).

    Sources without ``at_or_before`` give their latest item when it is not newer than ``due``.
    When nothing old enough is left (the sampler stalled longer than the history), the previous
    row's item is held; for the first tick that is the item read just before the sampler clock
    started.
    """
    pick = getattr(src, "at_or_before", None)
    item = pick(due) if pick is not None else src.latest()
    if item is None or item[0] > due:
        return prev
    return item


# --------------------------------------------------------------------------- video


def pick_vcodec(vcodec: str = "auto") -> str:
    """``auto``: libsvtav1 (LeRobot's default) when PyAV has it, else h264 (libx264)."""
    import av

    if vcodec != "auto":
        return vcodec
    for name in ("libsvtav1", "h264"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RecorderError("PyAV has neither libsvtav1 nor an h264 encoder")


@dataclass
class VideoSettings:
    vcodec: str = "auto"
    pix_fmt: str = "yuv420p"
    g: int = 2  # LeRobot default: keyframe every 2 frames for fast random access
    crf: int = 30
    preset: int | None = None  # libsvtav1: 12 (LeRobot default)

    def options(self) -> dict[str, str]:
        opts = {"g": str(self.g), "crf": str(self.crf)}
        if self.vcodec == "libsvtav1":
            opts["preset"] = str(12 if self.preset is None else self.preset)
        elif self.preset is not None:
            opts["preset"] = str(self.preset)
        return opts


class _VideoWriter:
    """mp4 with frame ``i`` at ``pts = i / fps``; frames are encoded on a worker thread."""

    def __init__(self, path: Path, fps: int, settings: VideoSettings, size: tuple[int, int] | None) -> None:
        self.path = path
        self.fps = fps
        self.settings = settings
        self.size = size  # (width, height); None = size of the first frame
        self.frames = 0
        self.error: BaseException | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max(8, fps * 4))
        self._thread = threading.Thread(target=self._run, name="rec-encoder", daemon=True)
        self._thread.start()

    def put(self, frame: Any) -> None:
        """Queue an ``av.VideoFrame`` or an RGB ``uint8`` array (H, W, 3)."""
        if self.error is not None:
            raise RecorderError(f"video encoder failed: {self.error}")
        self._q.put(frame)
        self.frames += 1

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self.error is not None:
            raise RecorderError(f"video encoder failed: {self.error}")

    def _run(self) -> None:
        out = stream = None
        pts = 0
        while True:
            item = self._q.get()
            if item is None:
                break
            if self.error is not None:
                continue  # drain so put() never blocks
            try:
                if out is None:
                    out, stream = self._open(item)
                pts = self._encode(out, stream, item, pts)
            except BaseException as exc:  # reported by put()/close()
                self.error = exc
        try:
            if out is not None:
                if self.error is None:
                    for packet in stream.encode(None):
                        out.mux(packet)
                out.close()
        except BaseException as exc:
            self.error = self.error or exc

    def _open(self, item: Any):
        import av

        w, h = self.size or (
            (item.shape[1], item.shape[0]) if isinstance(item, np.ndarray) else (item.width, item.height)
        )
        self.size = (w, h)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = av.open(str(self.path), "w")
        stream = out.add_stream(self.settings.vcodec, rate=self.fps, options=self.settings.options())
        stream.width, stream.height, stream.pix_fmt = w, h, self.settings.pix_fmt
        return out, stream

    def _encode(self, out: Any, stream: Any, item: Any, pts: int) -> int:
        import av

        if isinstance(item, np.ndarray):
            item = av.VideoFrame.from_ndarray(np.ascontiguousarray(item[..., :3]), format="rgb24")
        w, h = self.size
        frame = item.reformat(width=w, height=h, format=self.settings.pix_fmt)
        frame.pts = pts
        for packet in stream.encode(frame):
            out.mux(packet)
        return pts + 1


def probe_video(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = s.average_rate or s.base_rate  # Fraction (PyAV < 17) or AVRational
        return {
            "video.height": s.height,
            "video.width": s.width,
            "video.codec": s.codec_context.codec.canonical_name,
            "video.pix_fmt": s.pix_fmt,
            "video.fps": round(rate.numerator / rate.denominator),
            "video.channels": 3,
            "has_audio": len(c.streams.audio) > 0,
            "frames": s.frames,
        }


# --------------------------------------------------------------------------- stats


def feature_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    """LeRobot per-feature stats over axis 0: min/max/mean/std/count/q01..q99."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    out = {
        "min": data.min(axis=0),
        "max": data.max(axis=0),
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "count": np.array([len(data)]),
    }
    for q in QUANTILES:
        out[f"q{int(round(q * 100)):02d}"] = np.quantile(data, q, axis=0)
    return out


def image_stats(images: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Per-channel stats of RGB uint8 images (H, W, 3), scaled to [0, 1], shape (3, 1, 1)."""
    px = np.concatenate([im.reshape(-1, 3) for im in images]).astype(np.float64) / 255.0
    stats = feature_stats(px)
    stats.pop("count")
    out = {k: v.reshape(3, 1, 1) for k, v in stats.items()}
    out["count"] = np.array([len(images)])
    return out


def aggregate_stats(per_episode: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    """LeRobot ``aggregate_stats``: count-weighted mean and quantiles, pooled variance."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in per_episode[0]:
        items = [ep[key] for ep in per_episode if key in ep]
        counts = np.stack([s["count"] for s in items]).astype(np.float64)
        total = counts.sum(axis=0)
        means = np.stack([s["mean"] for s in items])
        w = counts.reshape(counts.shape + (1,) * (means.ndim - counts.ndim))
        mean = (means * w).sum(axis=0) / total
        var = ((np.stack([s["std"] for s in items]) ** 2 + (means - mean) ** 2) * w).sum(axis=0) / total
        agg = {
            "min": np.min(np.stack([s["min"] for s in items]), axis=0),
            "max": np.max(np.stack([s["max"] for s in items]), axis=0),
            "mean": mean,
            "std": np.sqrt(var),
            "count": total.astype(np.int64),
        }
        for q in [k for k in items[0] if k.startswith("q")]:
            agg[q] = (np.stack([s[q] for s in items]) * w).sum(axis=0) / total
        out[key] = agg
    return out


def _jsonable(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    return {k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in stats.items()}


# --------------------------------------------------------------------------- recorder


@dataclass
class EpisodeSummary:
    frames: int
    duration_s: float
    task: str
    late_ticks: int = 0  # ticks sampled more than one period after their due time
    camera_repeats: int = 0  # rows that reuse the previous row's camera frame
    obs_age_ms_mean: float = 0.0
    obs_age_ms_max: float = 0.0
    camera_age_ms_mean: float = 0.0
    camera_age_ms_max: float = 0.0
    armd_timestamp: tuple[float, float] | None = None  # first/last armd clock value (s since armd start)
    camera_device_ms: tuple[int, int] | None = None  # first/last camera unix-ms tail
    host_start_unix: float = 0.0
    episode_index: int | None = None  # set by save()

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class _Episode:
    task: str
    tmp: Path
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    stat_images: list = field(default_factory=list)
    video: _VideoWriter | None = None
    summary: EpisodeSummary | None = None


class LeRobotRecorder:
    """Write episodes of state/action (+ one camera) to ``root`` as a LeRobot v3.0 dataset.

    ``camera_url`` / ``armd_host`` accept a host (default ports 8001 / 8765) or a full URL;
    ``camera_url=None`` records state and action only. ``obs_source`` / ``frame_source`` replace
    the network streams (objects with ``latest() -> (host_monotonic_s, item) | None``, optional
    ``start()``, ``wait_first(timeout)`` and ``close()``), e.g. in tests. ``with_camera=True``
    without a camera source keeps the video feature for images passed to :meth:`add_frame`.

    Episode flow: :meth:`start_episode` -> :meth:`stop_episode` -> :meth:`save` or :meth:`discard`.
    ``root`` must be empty or absent; metadata is rewritten after every :meth:`save`.
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        fps: int = 30,
        camera_url: str | None = "192.168.16.1",
        armd_host: str | None = "192.168.16.1",
        task: str = "",
        *,
        camera_key: str = "observation.images.recamera",
        extras: Sequence[str] = (),
        image_size: tuple[int, int] | None = None,
        video: VideoSettings | None = None,
        robot_type: str = "rebot_b601_dm",
        joint_names: Sequence[str] = JOINT_NAMES,
        obs_hz: int | None = None,
        obs_source: Any = None,
        frame_source: Any = None,
        connect_timeout: float = 5.0,
        with_camera: bool | None = None,
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise RecorderError(f"{self.root} exists and is not empty; choose a new dataset root")
        if fps <= 0:
            raise ValueError("fps must be positive")
        unknown = set(extras) - set(EXTRAS)
        if unknown:
            raise ValueError(f"unknown extras {sorted(unknown)}; choose from {sorted(EXTRAS)}")
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.camera_key = camera_key
        self.extras = list(extras)
        self.image_size = image_size
        self.video = video or VideoSettings()
        self.robot_type = robot_type
        self.joint_names = list(joint_names)
        self.connect_timeout = connect_timeout
        if obs_source is None and armd_host is not None:
            obs_source = ObservationStream(armd_host, obs_hz=obs_hz or self.fps * 2, connect_timeout=connect_timeout)
        if frame_source is None and camera_url is not None:
            frame_source = CameraStream(camera_url, connect_timeout=connect_timeout)
        self.obs_source = obs_source
        self.frame_source = frame_source
        self.has_video = frame_source is not None if with_camera is None else with_camera
        if self.has_video:
            import av  # noqa: F401  (fail early without the record extra)

            self.video.vcodec = pick_vcodec(self.video.vcodec)
        import pyarrow  # noqa: F401

        self._episodes: list[dict] = []  # meta/episodes rows
        self._stats: list[dict] = []  # per-episode stats
        self._tasks: dict[str, int] = {}
        self._video_info: dict | None = None
        self._ep: _Episode | None = None
        self._recording = False
        self._sampler: threading.Thread | None = None
        self._stop_sampling = threading.Event()
        self._sampler_error: BaseException | None = None
        self._opened = False
        self._warned_no_cmd = False

    # ---- lifecycle -------------------------------------------------------------

    def open(self) -> LeRobotRecorder:
        """Start the sources and wait for the first observation / camera frame."""
        if self._opened:
            return self
        started = []
        try:
            for src in (self.obs_source, self.frame_source):
                if src is not None and hasattr(src, "start"):
                    started.append(src)
                    src.start()
            for src in (self.obs_source, self.frame_source):
                if src is not None and hasattr(src, "wait_first"):
                    src.wait_first(self.connect_timeout)
        except BaseException:
            for src in reversed(started):  # roll back: no stream thread or socket outlives a failed open
                if hasattr(src, "close"):
                    try:
                        src.close()
                    except Exception as exc:
                        log.warning("closing %s after a failed open: %s", getattr(src, "name", src), exc)
            raise
        self._opened = True
        return self

    def close(self) -> None:
        if self._ep is not None:
            log.warning("closing with an unsaved episode; it is discarded")
            self.discard()
        for src in (self.obs_source, self.frame_source):
            if src is not None and hasattr(src, "close"):
                src.close()
        shutil.rmtree(self.root / ".recording", ignore_errors=True)
        self._opened = False

    def __enter__(self) -> LeRobotRecorder:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def recording(self) -> bool:
        return self._recording

    # ---- episode control -------------------------------------------------------

    def start_episode(self, task: str | None = None, sample: bool | None = None) -> None:
        """Begin an episode. ``sample`` (default: True when an observation source exists) starts the
        fixed-rate sampler; with ``sample=False`` the caller adds rows with :meth:`add_frame`."""
        if self._ep is not None:
            raise RecorderError("an episode is already active or waiting for save()/discard()")
        task = self.task if task is None else task
        sample = self.obs_source is not None if sample is None else sample
        if sample and self.obs_source is None:
            raise RecorderError("sampling needs an observation source")
        if sample and not self._opened:
            self.open()
        tmp = self.root / ".recording" / f"ep-{len(self._episodes):06d}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        ep = _Episode(task=task, tmp=tmp, extras={k: [] for k in self.extras})
        if self.has_video:
            ep.video = _VideoWriter(tmp / "video.mp4", self.fps, self.video, self.image_size)
        ep.summary = EpisodeSummary(frames=0, duration_s=0.0, task=task, host_start_unix=time.time())
        self._ep = ep
        self._recording = True
        if sample:
            self._stop_sampling.clear()
            self._sampler_error = None
            self._sampler = threading.Thread(target=self._sample_loop, name="rec-sampler", daemon=True)
            self._sampler.start()

    def add_frame(
        self,
        state: Sequence[float],
        action: Sequence[float],
        image: Any = None,
        velocity: Sequence[float] | None = None,
        effort: Sequence[float] | None = None,
    ) -> int:
        """Append one row (``timestamp = frame_index / fps``). ``image``: RGB uint8 (H, W, 3) or
        ``av.VideoFrame``; required when the dataset has a camera. Returns the frame index."""
        ep = self._require_recording()
        if self.has_video and image is None:
            raise ValueError("image is required: the dataset has a camera feature")
        i = len(ep.state)
        if i == 0:
            self._dof = len(state)
        if len(state) != self._dof or len(action) != self._dof:
            raise ValueError(f"state and action need {self._dof} values")
        ep.state.append([float(x) for x in state])
        ep.action.append([float(x) for x in action])
        for name, values in (("velocity", velocity), ("effort", effort)):
            if name in ep.extras:
                if values is None or len(values) != self._dof:
                    raise ValueError(f"{name} needs {self._dof} values")
                ep.extras[name].append([float(x) for x in values])
        if ep.video is not None:
            ep.video.put(image)
            every = max(1, round(self.fps * STATS_IMAGE_EVERY_S))
            if i % every == 0:
                ep.stat_images.append(_stats_image(image))
        ep.summary.frames = i + 1
        return i

    def stop_episode(self) -> EpisodeSummary:
        """End the active episode; it waits for :meth:`save` or :meth:`discard`."""
        ep = self._require_recording()
        self._recording = False
        if self._sampler is not None:
            self._stop_sampling.set()
            self._sampler.join()
            self._sampler = None
        if ep.video is not None:
            ep.video.close()
        if self._sampler_error is not None:
            err, self._sampler_error = self._sampler_error, None
            raise RecorderError(f"sampling failed: {err}") from err
        ep.summary.frames = len(ep.state)
        ep.summary.duration_s = len(ep.state) / self.fps
        return ep.summary

    def save(self) -> int:
        """Write the stopped episode into the dataset; returns its episode index."""
        ep = self._ep
        if ep is None or self._recording:
            raise RecorderError("no stopped episode to save (call stop_episode() first)")
        n = len(ep.state)
        if n == 0:
            raise RecorderError("episode has no frames; discard() it")
        import pyarrow as pa
        import pyarrow.parquet as pq

        idx = len(self._episodes)
        chunk, file = divmod(idx, CHUNKS_SIZE)
        start = self._episodes[-1]["dataset_to_index"] if self._episodes else 0
        task_index = self._tasks.setdefault(ep.task, len(self._tasks))
        frame_index = np.arange(n, dtype=np.int64)
        columns = {
            "observation.state": np.asarray(ep.state, np.float32),
            "action": np.asarray(ep.action, np.float32),
        }
        for name in self.extras:
            columns[EXTRAS[name][0]] = np.asarray(ep.extras[name], np.float32)
        columns |= {
            "timestamp": (frame_index / self.fps).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.full(n, idx, np.int64),
            "index": np.arange(start, start + n, dtype=np.int64),
            "task_index": np.full(n, task_index, np.int64),
        }
        if self._video_info is None and ep.video is not None:
            self._video_info = probe_video(ep.video.path)
        if ep.video is not None:
            frames = probe_video(ep.video.path)["frames"]
            if frames and frames != n:
                raise RecorderError(f"video has {frames} frames for {n} rows")
        features = self._features()

        arrays, fields = [], []
        for key, values in columns.items():
            if values.ndim == 2:
                arr = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])
            else:
                arr = pa.array(values)
            arrays.append(arr)
            fields.append(pa.field(key, arr.type))
        hf = {"info": {"features": {k: _hf_feature(features[k]) for k in columns}}}
        schema = pa.schema(fields, metadata={"huggingface": json.dumps(hf)})
        data_path = self.root / DATA_PATH.format(chunk_index=chunk, file_index=file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, schema=schema), data_path)

        stats = {k: feature_stats(v) for k, v in columns.items()}
        row: dict[str, Any] = {
            "episode_index": idx,
            "tasks": [ep.task],
            "length": n,
            "data/chunk_index": chunk,
            "data/file_index": file,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
        }
        if ep.video is not None:
            video_path = self.root / VIDEO_PATH.format(video_key=self.camera_key, chunk_index=chunk, file_index=file)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep.video.path), video_path)
            stats[self.camera_key] = image_stats(ep.stat_images)
            row |= {
                f"videos/{self.camera_key}/chunk_index": chunk,
                f"videos/{self.camera_key}/file_index": file,
                f"videos/{self.camera_key}/from_timestamp": 0.0,
                f"videos/{self.camera_key}/to_timestamp": n / self.fps,
            }
        for key in features:
            for s, v in stats[key].items():
                v = np.asarray(v)
                row[f"stats/{key}/{s}"] = v.astype(np.int64).tolist() if s == "count" else v.astype(float).tolist()
        row |= {"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        self._episodes.append(row)
        self._stats.append(stats)
        ep.summary.episode_index = idx
        self._write_meta(ep.summary)
        shutil.rmtree(ep.tmp, ignore_errors=True)
        self._ep = None
        return idx

    def discard(self) -> None:
        """Drop the active or stopped episode."""
        ep = self._ep
        if ep is None:
            return
        if self._recording:
            try:
                self.stop_episode()
            except RecorderError as exc:
                log.warning("discarding an episode that failed: %s", exc)
        shutil.rmtree(ep.tmp, ignore_errors=True)
        self._ep = None

    def check_sources(self) -> str | None:
        """Error text of a source that stopped (socket closed), or the sampler failure."""
        if self._sampler_error is not None:
            return str(self._sampler_error)
        for src in (self.obs_source, self.frame_source):
            err = getattr(src, "error", None)
            if err:
                return err
        return None

    # ---- internals -------------------------------------------------------------

    def _require_recording(self) -> _Episode:
        if self._ep is None or not self._recording:
            raise RecorderError("no active episode (call start_episode() first)")
        return self._ep

    def _sample_loop(self) -> None:
        ep = self._ep
        s = ep.summary
        period = 1.0 / self.fps
        # Seed the held items before the clock starts: every seed was received before t0, so the first
        # tick has an item at or before its due time even when a latest()-only source updates between
        # here and the first read.
        obs_ages, cam_ages = [], []
        last_seq = None
        try:
            prev_obs = self.obs_source.latest()
            prev_cam = self.frame_source.latest() if self.frame_source is not None else None
            t0 = time.monotonic()
            k = 0
            while not self._stop_sampling.is_set():
                due = t0 + k * period
                now = time.monotonic()
                if due > now:
                    if self._stop_sampling.wait(due - now):
                        break
                elif now - due > period:
                    s.late_ticks += 1
                obs_item = prev_obs = _sample_at(self.obs_source, due, prev_obs)
                if obs_item is None:
                    raise RecorderError("no observation received")
                rx, obs = obs_item
                obs_ages.append(due - rx)
                s.armd_timestamp = (s.armd_timestamp or (obs.timestamp, obs.timestamp))[:1] + (obs.timestamp,)
                action = obs.commanded_positions
                if len(action) != len(obs.positions):
                    if not self._warned_no_cmd:
                        log.warning("armd reports no commanded_positions; action = positions")
                        self._warned_no_cmd = True
                    action = obs.positions
                image = None
                if self.frame_source is not None:
                    cam = prev_cam = _sample_at(self.frame_source, due, prev_cam)
                    if cam is None:
                        raise RecorderError("no camera frame received")
                    crx, frame = cam
                    cam_ages.append(due - crx)
                    image = getattr(frame, "frame", frame)
                    seq = getattr(frame, "seq", None)
                    if seq is not None and seq == last_seq:
                        s.camera_repeats += 1
                    last_seq = seq
                    dev = getattr(frame, "device_ts_ms", None)
                    if dev is not None:
                        s.camera_device_ms = (s.camera_device_ms or (dev, dev))[:1] + (dev,)
                self.add_frame(obs.positions, action, image, velocity=obs.velocities, effort=obs.torques)
                k += 1
        except BaseException as exc:
            self._sampler_error = exc
            log.error("sampler stopped: %s", exc)
        for ages, mean_attr, max_attr in (
            (obs_ages, "obs_age_ms_mean", "obs_age_ms_max"),
            (cam_ages, "camera_age_ms_mean", "camera_age_ms_max"),
        ):
            if ages:
                setattr(s, mean_attr, round(1000 * float(np.mean(ages)), 2))
                setattr(s, max_attr, round(1000 * float(np.max(ages)), 2))

    def _features(self) -> dict[str, dict]:
        dof = getattr(self, "_dof", len(self.joint_names))
        names = self.joint_names if len(self.joint_names) == dof else [f"joint_{i + 1}" for i in range(dof)]
        feats: dict[str, dict] = {
            "observation.state": {"dtype": "float32", "shape": [dof], "names": list(names)},
            "action": {"dtype": "float32", "shape": [dof], "names": list(names)},
        }
        for name in self.extras:
            feats[EXTRAS[name][0]] = {"dtype": "float32", "shape": [dof], "names": list(names)}
        if self.has_video and self._video_info is not None:
            vi = self._video_info
            feats[self.camera_key] = {
                "dtype": "video",
                "shape": [vi["video.height"], vi["video.width"], 3],
                "names": ["height", "width", "channels"],
                "info": {
                    **{k: v for k, v in vi.items() if k != "frames"},
                    "video.g": self.video.g,
                    "video.crf": self.video.crf,
                    "video.preset": self.video.options().get("preset"),
                    "video.fast_decode": 0,
                    "video.video_backend": "pyav",
                    "video.extra_options": {},
                    "is_depth_map": False,
                },
            }
        feats |= json.loads(json.dumps(DEFAULT_FEATURES))
        return feats

    def _write_meta(self, summary: EpisodeSummary) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta = self.root / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        total_frames = self._episodes[-1]["dataset_to_index"]
        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": self.fps,
            "features": self._features(),
            "total_episodes": len(self._episodes),
            "total_frames": int(total_frames),
            "total_tasks": len(self._tasks),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.has_video else None,
            "robot_type": self.robot_type,
            "splits": {"train": f"0:{len(self._episodes)}"},
        }
        _write_json(meta / "info.json", info)
        _write_json(meta / "stats.json", _jsonable(aggregate_stats(self._stats)))
        pq.write_table(pa.Table.from_pylist(self._episodes), self.root / EPISODES_PATH)
        pq.write_table(_tasks_table(self._tasks), meta / "tasks.parquet")
        with open(meta / "rebot_episodes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"repo_id": self.repo_id, "fps": self.fps, **summary.to_dict()}) + "\n")


def _stats_image(image: Any) -> np.ndarray:
    """RGB uint8 (H, W, 3), downsampled like LeRobot's ``auto_downsample_height_width``."""
    arr = image if isinstance(image, np.ndarray) else image.to_ndarray(format="rgb24")
    h, w = arr.shape[:2]
    if max(h, w) >= 300:
        step = int((w if w > h else h) / 150)
        arr = arr[::step, ::step]
    return np.ascontiguousarray(arr[..., :3])


def _hf_feature(ft: dict) -> dict:
    if ft["shape"] == [1]:
        return {"dtype": ft["dtype"], "_type": "Value"}
    return {"feature": {"dtype": ft["dtype"], "_type": "Value"}, "length": ft["shape"][0], "_type": "List"}


def _tasks_table(tasks: dict[str, int]):
    """``meta/tasks.parquet``: ``task_index`` column, task string as the pandas index ``task``."""
    import pyarrow as pa

    pandas_meta = {
        "index_columns": ["task"],
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ],
        "attributes": {},
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    items = sorted(tasks.items(), key=lambda kv: kv[1])
    schema = pa.schema(
        [pa.field("task_index", pa.int64()), pa.field("task", pa.string())],
        metadata={"pandas": json.dumps(pandas_meta)},
    )
    return pa.table([[i for _, i in items], [t for t, _ in items]], schema=schema)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- CLI


def _stdin_lines(q: queue.Queue) -> None:
    for line in sys.stdin:
        q.put(line.strip().lower())
    q.put("q")  # EOF


def _wait_key(keys: queue.Queue, timeout: float | None = None) -> str | None:
    try:
        return keys.get(timeout=timeout)
    except queue.Empty:
        return None


def _record_one(rec: LeRobotRecorder, keys: queue.Queue, episode_s: float, prompt: bool) -> str:
    """Record one episode; returns ``saved``, ``discarded`` or ``quit``."""
    rec.start_episode()
    print(
        f"  recording up to {episode_s:g} s" + ("  [Enter] stop  [d] discard  [q] discard and quit" if prompt else "")
    )
    end = time.monotonic() + episode_s
    key = None
    while time.monotonic() < end:
        if rec.check_sources():
            print(f"  source error: {rec.check_sources()}")
            break
        key = _wait_key(keys, min(0.2, max(0.0, end - time.monotonic()))) if prompt else None
        if key is not None:
            break
        if not prompt:
            time.sleep(min(0.2, max(0.0, end - time.monotonic())))
    s = rec.stop_episode()
    print(
        f"  {s.frames} frames ({s.duration_s:.1f} s), late ticks {s.late_ticks}, camera repeats {s.camera_repeats}, "
        f"obs age max {s.obs_age_ms_max:.0f} ms, camera age max {s.camera_age_ms_max:.0f} ms"
    )
    if key in ("d", "q") or s.frames == 0:
        rec.discard()
        print("  discarded")
        return "quit" if key == "q" else "discarded"
    if prompt:
        print("  [Enter] save  [d] discard  [q] save and quit")
        key = _wait_key(keys)
        if key == "d":
            rec.discard()
            print("  discarded")
            return "discarded"
    idx = rec.save()
    print(f"  saved episode {idx}")
    return "quit" if key == "q" else "saved"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rebot-record",
        description="Record reBot arm + reCamera episodes into a LeRobot v3.0 dataset (lease-free observer).",
    )
    p.add_argument("--armd", default="192.168.16.1", help="armd host or URL (default port 8765)")
    p.add_argument("--camera", default="192.168.16.1", help="reCamera host or ws:// URL (default port 8001)")
    p.add_argument("--no-camera", action="store_true", help="record state and action only")
    p.add_argument("--repo-id", required=True, help="dataset id, e.g. user/pick_cup")
    p.add_argument("--root", help="dataset directory (default ~/.cache/huggingface/lerobot/<repo-id>)")
    p.add_argument("--task", required=True, help="task description stored with every episode")
    p.add_argument("--episodes", type=int, default=1, help="episodes to save (default 1)")
    p.add_argument("--episode-s", type=float, default=30.0, help="maximum episode length in seconds")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--extras", default="", help="comma list of extra features: velocity,effort")
    p.add_argument("--camera-key", default="observation.images.recamera")
    p.add_argument("--image-size", help="resize video to WxH (default: camera size)")
    p.add_argument("--vcodec", default="auto", help="auto (libsvtav1 if available, else h264), libsvtav1, h264")
    p.add_argument("--no-prompt", action="store_true", help="start/save episodes without waiting for Enter")
    p.add_argument("--reset-s", type=float, default=0.0, help="pause between episodes with --no-prompt")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    root = Path(args.root) if args.root else Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    size = tuple(int(v) for v in args.image_size.lower().split("x")) if args.image_size else None
    try:
        rec = LeRobotRecorder(
            root,
            args.repo_id,
            fps=args.fps,
            camera_url=None if args.no_camera else args.camera,
            armd_host=args.armd,
            task=args.task,
            camera_key=args.camera_key,
            extras=[e for e in args.extras.split(",") if e],
            image_size=size,
            video=VideoSettings(vcodec=args.vcodec),
        )
        print(f"connecting to armd {args.armd}" + ("" if args.no_camera else f" and camera {args.camera}") + " ...")
        rec.open()
    except (RecorderError, ValueError, ImportError) as exc:
        print(f"rebot-record: {exc}", file=sys.stderr)
        return 1
    keys: queue.Queue = queue.Queue()
    if not args.no_prompt:
        threading.Thread(target=_stdin_lines, args=(keys,), daemon=True).start()
    print(f"dataset {args.repo_id} -> {root}")
    try:
        while rec.num_episodes < args.episodes:
            n = rec.num_episodes + 1
            if args.no_prompt:
                if n > 1 and args.reset_s > 0:
                    print(f"reset: {args.reset_s:g} s")
                    time.sleep(args.reset_s)
                print(f"episode {n}/{args.episodes}")
            else:
                print(f"episode {n}/{args.episodes}: [Enter] start  [q] quit")
                if _wait_key(keys) == "q":
                    break
            if _record_one(rec, keys, args.episode_s, not args.no_prompt) == "quit":
                break
    except KeyboardInterrupt:
        print("\ninterrupted; the current episode is discarded")
    except RecorderError as exc:
        print(f"rebot-record: {exc}", file=sys.stderr)
    finally:
        rec.close()
    print(f"{rec.num_episodes} episode(s) in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
