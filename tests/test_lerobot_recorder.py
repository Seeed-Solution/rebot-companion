"""LeRobot recorder: synthetic observations + synthetic H.264 camera -> LeRobot v3.0 dataset.

The structural checks run with PyAV + pyarrow (dev group). Loading with ``LeRobotDataset`` needs
``lerobot`` (Python >= 3.12, torch); it runs in-process when importable, else through the
interpreter named by ``REBOT_LEROBOT_PYTHON`` (a venv with ``lerobot[dataset]`` 0.6.1 or 0.7.0), else skips.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pq = pytest.importorskip("pyarrow.parquet")

from rebot.client import ArmStatus  # noqa: E402
from rebot_companion.lerobot_recorder import (  # noqa: E402
    CameraStream,
    LeRobotRecorder,
    ObservationStream,
    RecorderError,
    armd_url,
    camera_url,
)

FPS = 30
CAM = "observation.images.recamera"
W, H = 64, 48


def level(i: int) -> int:
    """Gray level drawn into synthetic camera frame ``i``."""
    return 16 + (i * 8) % 224


def h264_aus(n: int, w: int = W, h: int = H) -> list[bytes]:
    """Annex-B access units of ``n`` uniform gray frames (libx264, no B-frames)."""
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = w, h, "yuv420p"
    ctx.time_base = Fraction(1, FPS)
    ctx.options = {"bf": "0", "g": "10", "tune": "zerolatency", "preset": "ultrafast"}
    aus = []
    for i in range(n):
        frame = av.VideoFrame.from_ndarray(np.full((h, w, 3), level(i), np.uint8), format="rgb24")
        frame.pts = i
        aus += [bytes(p) for p in ctx.encode(frame)]
    aus += [bytes(p) for p in ctx.encode(None)]
    return aus


def obs(t: float, n: int = 7) -> ArmStatus:
    pos = [round(t, 4) + j for j in range(n)]
    return ArmStatus.from_dict(
        {
            "timestamp": t,
            "positions": pos,
            "velocities": [0.1] * n,
            "torques": [0.2] * n,
            "commanded_positions": [p + 0.5 for p in pos],
            "safety_state": "Armed",
            "is_torqued": True,
        }
    )


class FakeObservations:
    """Observation source updated at 200 Hz; positions encode the host time since start."""

    def __init__(self) -> None:
        self._latest = None
        self._history = []
        self._stop = threading.Event()
        self.t0 = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.wait(0.005):
            now = time.monotonic()
            self._latest = (now, obs(now - self.t0))
            self._history = self._history[-99:] + [self._latest]

    def wait_first(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while self._latest is None and time.monotonic() < end:
            time.sleep(0.01)

    def latest(self):
        return self._latest

    def at_or_before(self, t: float):
        return next((e for e in reversed(self._history) if e[0] <= t), None)

    def close(self) -> None:
        self._stop.set()


class FakeCamera(CameraStream):
    """The real PyAV decode path, fed synthetic access units at 30 fps instead of a socket."""

    def __init__(self, aus: list[bytes]) -> None:
        super().__init__("127.0.0.1")
        self.aus = aus

    def start(self) -> None:
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self) -> None:
        for i, au in enumerate(self.aus):
            if self._stop.is_set():
                return
            self.handle(au + (1_700_000_000_000 + i * 33).to_bytes(8, "little"), time.monotonic())
            time.sleep(1 / FPS)

    def close(self) -> None:
        self._stop.set()


def decode_means(path: Path) -> list[float]:
    with av.open(str(path)) as c:
        return [float(f.to_ndarray(format="rgb24").mean()) for f in c.decode(video=0)]


def check_structure(root: Path, lengths: list[int], video: bool = True) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert info["total_episodes"] == len(lengths)
    assert info["total_frames"] == sum(lengths)
    assert info["splits"] == {"train": f"0:{len(lengths)}"}
    for key in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert key in info["features"]
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["length"] for e in eps] == lengths
    start = 0
    for i, e in enumerate(eps):
        assert e["episode_index"] == i
        assert (e["dataset_from_index"], e["dataset_to_index"]) == (start, start + lengths[i])
        data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=i)).to_pydict()
        assert data["index"] == list(range(start, start + lengths[i]))
        assert data["frame_index"] == list(range(lengths[i]))
        assert set(data["episode_index"]) == {i}
        np.testing.assert_allclose(data["timestamp"], np.arange(lengths[i]) / FPS, atol=1e-6)
        assert len(data["observation.state"][0]) == 7
        if video:
            vpath = root / info["video_path"].format(video_key=CAM, chunk_index=0, file_index=i)
            assert len(decode_means(vpath)) == lengths[i]
            assert e[f"videos/{CAM}/to_timestamp"] == pytest.approx(lengths[i] / FPS)
        start += lengths[i]
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [sum(lengths)]
    assert set(stats["action"]) == {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    tasks = pq.read_table(root / "meta/tasks.parquet")
    assert tasks.column_names == ["task_index", "task"]
    assert json.loads(tasks.schema.metadata[b"pandas"])["index_columns"] == ["task"]
    if video:
        vinfo = info["features"][CAM]
        assert vinfo["dtype"] == "video" and vinfo["shape"] == [H, W, 3]
        assert len(stats[CAM]["mean"]) == 3
    return info


# --------------------------------------------------------------------------- scripted (add_frame)


def write_scripted(root: Path, lengths=(20, 15)) -> None:
    rec = LeRobotRecorder(
        root, "test/scripted", fps=FPS, camera_url=None, armd_host=None, task="pick the cup", with_camera=True
    )
    for n in lengths:
        rec.start_episode(sample=False)
        for i in range(n):
            o = obs(i / FPS)
            rec.add_frame(o.positions, o.commanded_positions, np.full((H, W, 3), level(i), np.uint8))
        rec.stop_episode()
        rec.save()
    rec.close()


def test_scripted_dataset(tmp_path):
    root = tmp_path / "scripted"
    write_scripted(root)
    info = check_structure(root, [20, 15])
    means = decode_means(root / info["video_path"].format(video_key=CAM, chunk_index=0, file_index=0))
    assert means == pytest.approx([level(i) for i in range(20)], abs=4)
    data = pq.read_table(root / "data/chunk-000/file-001.parquet").to_pydict()
    assert data["action"][3][0] == pytest.approx(3 / FPS + 0.5, abs=1e-4)
    assert not (root / ".recording").exists()


def test_discard_keeps_indices_contiguous(tmp_path):
    rec = LeRobotRecorder(tmp_path / "d", "test/d", fps=FPS, camera_url=None, armd_host=None, task="t")
    for n, keep in ((5, True), (7, False), (4, True)):
        rec.start_episode(sample=False)
        for i in range(n):
            rec.add_frame([i] * 7, [i] * 7)
        rec.stop_episode()
        rec.save() if keep else rec.discard()
    rec.close()
    check_structure(tmp_path / "d", [5, 4], video=False)


def test_root_must_be_empty(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("")
    with pytest.raises(RecorderError):
        LeRobotRecorder(tmp_path / "x", "a/b", camera_url=None, armd_host=None)


# --------------------------------------------------------------------------- live sampler


def test_live_sampling_with_synthetic_streams(tmp_path):
    root = tmp_path / "live"
    cam = FakeCamera(h264_aus(90))
    rec = LeRobotRecorder(
        root,
        "test/live",
        fps=FPS,
        task="pick the cup",
        obs_source=FakeObservations(),
        frame_source=cam,
        extras=("velocity", "effort"),
    )
    with rec:
        summaries = []
        for _ in range(2):
            rec.start_episode()
            time.sleep(0.6)
            summaries.append(rec.stop_episode())
            rec.save()
    lengths = [s.frames for s in summaries]
    assert all(15 <= n <= 20 for n in lengths), lengths
    info = check_structure(root, lengths)
    assert info["features"]["observation.velocity"]["shape"] == [7]
    data = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    np.testing.assert_allclose(np.array(data["action"]) - np.array(data["observation.state"]), 0.5, atol=1e-5)
    # state encodes host time; consecutive rows are about one period apart (zero-order hold of a 200 Hz source)
    dt = np.diff(np.array(data["observation.state"])[:, 0])
    assert abs(np.median(dt) - 1 / FPS) < 0.01, dt
    s = summaries[0]
    assert s.obs_age_ms_max < 50 and s.camera_age_ms_max < 200
    assert s.camera_device_ms is not None and s.armd_timestamp is not None
    lines = (root / "meta/rebot_episodes.jsonl").read_text().splitlines()
    assert [json.loads(x)["episode_index"] for x in lines] == [0, 1]
    # camera frames come from the decoded stream: gray levels follow the synthetic sequence
    means = decode_means(root / "videos" / CAM / "chunk-000/file-000.mp4")
    assert all(any(abs(m - level(i)) < 6 for i in range(90)) for m in means)


def test_overdue_ticks_never_use_samples_received_after_their_due_time(tmp_path):
    src = FakeObservations()
    rec = LeRobotRecorder(tmp_path / "late", "test/late", fps=FPS, task="t", obs_source=src, camera_url=None)
    real_add = rec.add_frame
    calls = []

    def stalling_add(*a, **kw):
        calls.append(1)
        if len(calls) == 5:
            time.sleep(4 / FPS)  # the next ticks are overdue
        return real_add(*a, **kw)

    rec.add_frame = stalling_add
    with rec:
        rec.start_episode()
        started = time.monotonic()  # the sampler's t0 is no later than this
        time.sleep(0.5)
        summary = rec.stop_episode()
        rec.save()
    assert summary.late_ticks >= 1
    data = pq.read_table(tmp_path / "late/data/chunk-000/file-000.parquet").to_pydict()
    rx = np.array(data["observation.state"])[:, 0] + src.t0  # host receive time of each row's sample
    due_bound = started + np.arange(len(rx)) / FPS
    assert np.all(rx <= due_bound + 1e-3), (rx - due_bound)[rx > due_bound + 1e-3]
    assert summary.obs_age_ms_mean >= 0


def test_failed_open_closes_the_sources_it_started(tmp_path):
    class Source:
        def __init__(self, fail: str = "") -> None:
            self.fail, self.started, self.closed = fail, False, False

        def start(self) -> None:
            if self.fail == "start":
                raise RecorderError("cannot open camera")
            self.started = True

        def wait_first(self, timeout: float) -> None:
            if self.fail == "wait":
                raise RecorderError("no data")

        def close(self) -> None:
            self.closed = True

    for fail in ("start", "wait"):
        obs_src, cam = Source(), Source(fail)
        rec = LeRobotRecorder(
            tmp_path / fail, "t/x", camera_url=None, obs_source=obs_src, frame_source=cam, with_camera=False
        )
        with pytest.raises(RecorderError):
            with rec:
                pass
        assert obs_src.started and obs_src.closed, fail
        assert cam.closed or not cam.started, fail


def test_observation_stream_is_lease_free(armd, tmp_path):
    rec = LeRobotRecorder(tmp_path / "armd", "test/armd", fps=FPS, camera_url=None, armd_host=armd.url, task="t")
    with rec:
        rec.start_episode()
        time.sleep(0.3)
        summary = rec.stop_episode()
        rec.save()
    assert summary.frames >= 5
    assert armd.acquires == []
    assert any("obs_hz=60" in p for p in armd.ws_paths)
    check_structure(tmp_path / "armd", [summary.frames], video=False)


def test_observation_stream_parse():
    s = ObservationStream("10.0.0.2")
    assert s.url == "ws://10.0.0.2:8765/api/v1/ws"
    assert s.parse(json.dumps({"type": "lease_granted"})) is None
    o = s.parse(json.dumps({"positions": [1.0] * 7, "safety_state": "Idle", "timestamp": 3.0}))
    assert o.positions == [1.0] * 7


# --------------------------------------------------------------------------- LeRobotDataset


LOAD_SCRIPT = r"""
import json, sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, repo_id, cam = sys.argv[1], sys.argv[2], sys.argv[3]
ds = LeRobotDataset(repo_id, root=root)
out = {"len": len(ds), "episodes": ds.num_episodes, "fps": ds.fps, "frames": []}
for i in (0, 5, 19, 20, len(ds) - 1):
    item = ds[i]
    out["frames"].append({
        "i": i, "episode": int(item["episode_index"]), "frame": int(item["frame_index"]),
        "ts": float(item["timestamp"]), "action0": float(item["action"][0]),
        "image_shape": list(item[cam].shape), "image_mean": float(item[cam].mean()) * 255,
        "task": item["task"],
    })
print(json.dumps(out))
"""


def lerobot_python() -> str | None:
    try:
        import lerobot  # noqa: F401

        return sys.executable
    except ImportError:
        return os.environ.get("REBOT_LEROBOT_PYTHON") or None


def test_lerobot_dataset_loads(tmp_path):
    python = lerobot_python()
    if python is None:
        pytest.skip(
            "lerobot not importable (needs Python >= 3.12 and torch; not a dev dependency). "
            "Set REBOT_LEROBOT_PYTHON to a venv python with lerobot[dataset] 0.6.1 or 0.7.0"
        )
    root = tmp_path / "scripted"
    write_scripted(root)
    res = subprocess.run(
        [python, "-c", LOAD_SCRIPT, str(root), "test/scripted", CAM],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["len"] == 35 and out["episodes"] == 2 and out["fps"] == FPS
    for f in out["frames"]:
        assert f["image_shape"] == [3, H, W]
        assert f["ts"] == pytest.approx(f["frame"] / FPS, abs=1e-4)
        assert f["image_mean"] == pytest.approx(level(f["frame"]), abs=4)
        assert f["action0"] == pytest.approx(f["frame"] / FPS + 0.5, abs=1e-4)
        assert f["task"] == "pick the cup"
    assert [(f["episode"], f["frame"]) for f in out["frames"]] == [(0, 0), (0, 5), (0, 19), (1, 0), (1, 14)]


# --------------------------------------------------------------------------- CLI


def test_cli_no_prompt(armd, tmp_path, capsys):
    from rebot_companion.lerobot_recorder import main

    root = tmp_path / "cli"
    argv = ["--armd", armd.url, "--no-camera", "--repo-id", "t/cli", "--root", str(root), "--task", "t"]
    assert main(argv + ["--episodes", "2", "--episode-s", "0.3", "--no-prompt"]) == 0
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_episodes"] == 2
    assert "2 episode(s)" in capsys.readouterr().out


def test_cli_enter_driven(armd, tmp_path, monkeypatch, capsys):
    from rebot_companion.lerobot_recorder import main

    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))

    def typist() -> None:
        with os.fdopen(w, "w") as f:
            # start, stop, save | start, stop, discard | start, stop, save-and-quit
            for delay, line in [(0.3, ""), (0.4, ""), (0.1, ""), (0.1, ""), (0.3, ""), (0.1, "d"), (0.1, "")]:
                time.sleep(delay)
                f.write(line + "\n")
                f.flush()
            time.sleep(0.3)
            f.write("\n")
            f.flush()
            time.sleep(0.1)
            f.write("q\n")

    threading.Thread(target=typist, daemon=True).start()
    root = tmp_path / "enter"
    argv = ["--armd", armd.url, "--no-camera", "--repo-id", "t/e", "--root", str(root), "--task", "t"]
    assert main(argv + ["--episodes", "5", "--episode-s", "10"]) == 0
    out = capsys.readouterr().out
    assert "discarded" in out
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_episodes"] == 2


def test_host_urls_bracket_ipv6_literals():
    assert armd_url("192.168.16.1") == "http://192.168.16.1:8765"
    assert armd_url("h:9000") == "http://h:9000"
    assert armd_url("fd00::2") == "http://[fd00::2]:8765"
    assert armd_url("[fd00::2]:9000") == "http://[fd00::2]:9000"
    assert camera_url("fd00::2") == "ws://[fd00::2]:8001/"
    assert camera_url("192.168.16.1") == "ws://192.168.16.1:8001/"
    assert camera_url("ws://x/y") == "ws://x/y"
