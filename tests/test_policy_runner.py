"""rebot-policy against the fake armd: servo stream, clipping, stop conditions, park, replay round trip.

The learned-policy test builds a tiny randomly initialised ACT policy for the recorder's synthetic
dataset features and runs ``rebot-policy`` for 10 steps. It needs lerobot (Python >= 3.12, torch):
in-process when importable, else the interpreter named by ``REBOT_LEROBOT_PYTHON`` (a venv with
``lerobot[dataset]`` 0.6.1 / 0.7.0, PyAV, pyarrow and scipy), else it skips.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_armd import HOME, LIMITS
from http_mock import Request
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

pytest.importorskip("av")
pytest.importorskip("pyarrow")

from rebot.client import ArmSession  # noqa: E402
from rebot_companion.lerobot_recorder import LeRobotRecorder  # noqa: E402
from rebot_companion.policy_runner import (  # noqa: E402
    Policy,
    PolicyRunner,
    ReplayPolicy,
    RunnerError,
    SinePolicy,
    main,
    parse_policy,
)

REPO = Path(__file__).resolve().parent.parent
FPS = 30
CAM = "observation.images.recamera"
W, H = 64, 48


@pytest.fixture
def arm(armd):
    armd.devplat = True  # safety_describe -> follower "mock"
    return armd


def runner(armd, policy, **kw) -> PolicyRunner:
    kw.setdefault("max_seconds", 0.6)
    return PolicyRunner(armd.client, policy, **kw)


def run_with(armd, policy, action=None, after=0.3, **kw):
    """Run in a thread; call ``action()`` ``after`` seconds into the run."""
    r = runner(armd, policy, **kw)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("res", r.run()), daemon=True)
    t.start()
    if action is not None:
        time.sleep(after)
        action()
    t.join(timeout=20)
    assert not t.is_alive()
    return out["res"]


def assert_in_order(armd) -> None:
    seqs = [s for s, _, _ in armd.servo_log]
    assert seqs and seqs == list(range(seqs[0], seqs[0] + len(seqs)))


# --------------------------------------------------------------------------- stream + park


def test_sine_streams_frames_in_order_and_parks(arm):
    res = runner(arm, SinePolicy(0, 0.1, 1.0), max_seconds=1.0).run()
    assert res.reason == "max_seconds", res
    assert res.steps == FPS
    assert 40 <= res.frames <= 60  # 50 Hz for ~1 s
    assert_in_order(arm)
    assert len(arm.servo_log) == res.frames
    assert arm.dropped_servo_frames == 0
    j1 = [t[0] for _, t, _ in arm.servo_log]
    assert max(j1) == pytest.approx(0.1, abs=0.02) and min(j1) == pytest.approx(-0.1, abs=0.02)
    assert arm.commands[-2:] == ["/api/v1/motion/stop", "/api/v1/motion/unload"]
    assert res.parked and arm.state == "Idle" and not arm.is_torqued
    assert arm.lease is None
    for key in ("obs_age_ms", "inference_ms", "send_lag_ms", "step_lag_ms"):
        assert res.latency_ms[key]["n"] > 0
    assert res.latency_ms["obs_age_ms"]["max"] < 200


def test_interpolates_between_policy_steps(arm):
    """A 5 Hz policy step to +0.1 rad reaches armd as several 50 Hz frames, not one jump."""

    class Step(Policy):
        fps = 5

        def select(self, obs, step, t):
            return np.array(HOME) + np.array([0.1 * (step >= 1), 0, 0, 0, 0, 0, 0])

    res = runner(arm, Step(), max_seconds=0.8, park=False).run()
    assert res.reason == "max_seconds"
    j1 = [round(t[0], 4) for _, t, _ in arm.servo_log]
    rising = [v for v in j1 if 0.001 < v < 0.099]
    assert len(rising) >= 5, j1
    assert j1 == sorted(j1), j1


def test_no_park_holds(arm):
    res = runner(arm, SinePolicy(0, 0.05, 1.0), park=False).run()
    assert res.reason == "max_seconds"
    assert "/api/v1/motion/unload" not in arm.commands
    assert arm.state == "Armed" and arm.is_torqued
    assert not res.parked and "holding" in res.park_detail


def test_torques_on_an_idle_arm(arm):
    assert arm.state == "Idle"
    runner(arm, SinePolicy(0, 0.05, 1.0), max_seconds=0.2).run()
    assert arm.commands[0] == "/api/v1/torque"


# --------------------------------------------------------------------------- clipping


def test_lead_clip_keeps_frames_near_the_measured_position(arm):
    arm.servo_max_lead_rad = 0.05
    res = runner(arm, SinePolicy(0, 0.5, 0.5), max_seconds=1.0, park=False).run()
    assert res.reason == "max_seconds"
    assert res.lead_clipped_frames > 0
    assert res.lead_clips_per_joint[0] > 0 and sum(res.lead_clips_per_joint[1:]) == 0
    # the runner clips against an observation up to one tick old; the fake moves 0.04 rad per tick
    assert max(abs(t[0] - p[0]) for _, t, p in arm.servo_log) <= 0.05 + 0.045
    assert arm.dropped_servo_frames == 0


def test_joint_limit_clip_on_the_gripper(arm):
    res = runner(arm, SinePolicy(6, 0.5, 1.0), max_seconds=0.6, park=False).run()
    assert res.reason == "max_seconds", res
    assert res.limit_clipped_frames > 0
    assert max(t[6] for _, t, _ in arm.servo_log) <= LIMITS[6][1]
    assert res.lead_clips_per_joint[6] == 0  # armd exempts the gripper from the lead cap


# --------------------------------------------------------------------------- stop conditions


def test_stop_on_fault_does_not_park(arm):
    res = run_with(arm, SinePolicy(0, 0.05, 1.0), lambda: arm.trigger_fault("FollowingError"), max_seconds=5)
    assert res.reason == "fault" and res.detail == "FOLLOWING_ERROR"
    assert not res.parked and "Fault" in res.park_detail
    assert "/api/v1/motion/unload" not in arm.commands
    assert res.duration_s < 2


def test_stop_on_estop(arm):
    res = run_with(arm, SinePolicy(0, 0.05, 1.0), arm.do_estop, max_seconds=5)
    assert res.reason == "estop" and not res.parked


def test_stop_on_collision_hold(arm):
    arm.collision_cfg["enabled"] = True
    res = run_with(arm, SinePolicy(0, 0.05, 1.0), lambda: arm.trip_collision(resume_after_s=5), max_seconds=5)
    assert res.reason == "collision_hold"
    assert not res.parked and "collision" in res.park_detail


def test_stop_on_blocked_hold(arm):
    def block():
        arm.motion_limit = {"reason": "blocked", "joint": 1, "since_ms": 1}

    res = run_with(arm, SinePolicy(0, 0.05, 1.0), block, max_seconds=5)
    assert res.reason == "blocked"
    assert res.armd_motion_limits.get("blocked", 0) >= 1


def test_stop_on_lease_loss(arm):
    other = {}

    def preempt():
        other["s"] = ArmSession(arm.client, client_id="cockpit", force=True).open()

    res = run_with(arm, SinePolicy(0, 0.05, 1.0), preempt, max_seconds=5)
    other["s"].close()
    assert res.reason == "lease_lost"
    assert not res.parked and "lease" in res.park_detail


def test_stop_on_servo_rejection_then_parks(arm):
    def busy():
        arm.motion_gate = lambda targets, servo=False: (409, "MOTION_BUSY") if servo else None

    res = run_with(arm, SinePolicy(0, 0.05, 1.0), busy, max_seconds=5)
    assert res.reason == "servo_rejected" and "MOTION_BUSY" in res.detail
    assert res.parked


def test_stop_on_policy_stall(arm):
    class Slow(Policy):
        def select(self, obs, step, t):
            if step == 3:
                time.sleep(1.0)
            return np.array(obs["observation.state"], dtype=float)

    res = runner(arm, Slow(), max_seconds=5, stall_s=0.3).run()
    assert res.reason == "policy_stall"
    assert res.parked


def test_stop_on_bad_action(arm):
    class Short(Policy):
        def select(self, obs, step, t):
            return np.zeros(6)

    res = runner(arm, Short(), max_seconds=5).run()
    assert res.reason == "bad_action"


def test_policy_exception_still_parks(arm):
    class Boom(Policy):
        def select(self, obs, step, t):
            if step == 2:
                raise RuntimeError("cuda out of memory")
            return np.array(obs["observation.state"], dtype=float)

    res = runner(arm, Boom(), max_seconds=5).run()
    assert res.reason == "error" and "cuda" in res.detail
    assert res.parked


def test_stale_camera_stops():
    class Cam:
        def latest(self):
            return (time.monotonic() - 2.0, np.zeros((H, W, 3), np.uint8))

    class Img(Policy):
        image_key, image_size = CAM, (H, W)

        def select(self, obs, step, t):
            return obs["observation.state"]

    r = PolicyRunner(None, Img(), Cam())
    assert r._camera_image(time.monotonic()) is None
    assert r.result.reason == "camera_stale"


def test_image_policy_needs_a_camera():
    class Img(Policy):
        image_key, image_size = CAM, (H, W)

    with pytest.raises(RunnerError, match="camera"):
        PolicyRunner(None, Img())


# --------------------------------------------------------------------------- operator gate


def test_real_follower_needs_the_operator_flag(arm, monkeypatch):
    monkeypatch.setattr(arm.client, "get_safety_describe", lambda: {"device": {"follower": "rebot-dm"}})
    with pytest.raises(RunnerError, match="i-am-at-the-arm"):
        runner(arm, SinePolicy(0, 0.05, 1.0)).run()
    assert arm.acquires == []  # refused before taking the lease
    res = runner(arm, SinePolicy(0, 0.05, 1.0), operator_present=True, max_seconds=0.2).run()
    assert res.reason == "max_seconds" and res.follower == "rebot-dm"


def test_unknown_follower_needs_the_operator_flag(armd):
    armd.devplat = False  # no safety_describe -> follower unknown
    with pytest.raises(RunnerError, match="unknown"):
        runner(armd, SinePolicy(0, 0.05, 1.0)).run()


def test_refuses_an_arm_in_fault(arm):
    arm.trigger_fault("FollowingError")
    with pytest.raises(RunnerError, match="Fault"):
        runner(arm, SinePolicy(0, 0.05, 1.0)).run()
    assert arm.servo_log == []


# --------------------------------------------------------------------------- policy specs


def home_dataset(root: Path, n: int = 45, hold: int = 15, ramp: float = 0.3, image: bool = True) -> np.ndarray:
    """A recorder dataset at the fake's home pose: J1 ramps 0 -> ``ramp`` over ``n`` frames, then holds."""
    rec = LeRobotRecorder(
        root, "test/home", fps=FPS, camera_url=None, armd_host=None, task="pick the cup", with_camera=image
    )
    rec.start_episode(sample=False)
    actions = []
    for i in range(n + hold):
        a = np.array(HOME)
        a[0] = ramp * min(i, n) / n
        state = a - 0.01 * (i < n)
        img = np.full((H, W, 3), 16 + (i * 8) % 224, np.uint8) if image else None
        rec.add_frame(state.tolist(), a.tolist(), img)
        actions.append(a)
    rec.stop_episode()
    rec.save()
    rec.close()
    return np.array(actions)


def test_parse_policy(tmp_path):
    actions = home_dataset(tmp_path / "ds", image=False)
    p = parse_policy(f"replay:{tmp_path / 'ds'}:0")
    assert isinstance(p, ReplayPolicy) and p.fps == FPS
    np.testing.assert_allclose(p.actions, actions, atol=1e-6)
    assert isinstance(parse_policy(f"scripted:replay:{tmp_path / 'ds'}:0"), ReplayPolicy)
    s = parse_policy("sine:gripper:0.2:2")
    assert (s.joint, s.amp, s.period) == (6, 0.2, 2.0)
    assert parse_policy("scripted:sine:2").joint == 1
    with pytest.raises(RunnerError, match="episode 3"):
        parse_policy(f"replay:{tmp_path / 'ds'}:3")
    for bad in ("sine:9", "sine:1:2.0", "replay:x", "scripted:wave"):
        with pytest.raises(ValueError):
            parse_policy(bad)


def test_replay_round_trip(arm, tmp_path):
    """Recorder dataset -> replay through armd -> recorder observes the arm -> same trajectory."""
    actions = home_dataset(tmp_path / "a")
    rec = LeRobotRecorder(tmp_path / "b", "test/b", fps=FPS, camera_url=None, armd_host=arm.url, task="t")
    with rec:
        rec.start_episode()
        res = runner(arm, ReplayPolicy(tmp_path / "a", 0), max_seconds=10, park=False).run()
        time.sleep(0.2)
        rec.stop_episode()
        rec.save()
    assert res.reason == "policy_done" and res.steps == len(actions)
    assert res.lead_clipped_frames == 0 and arm.dropped_servo_frames == 0
    import pyarrow.parquet as pq

    state = np.array(pq.read_table(tmp_path / "b/data/chunk-000/file-000.parquet").to_pydict()["observation.state"])
    j1 = state[:, 0]
    assert j1[-1] == pytest.approx(actions[-1, 0], abs=0.01)
    assert j1.max() == pytest.approx(actions[:, 0].max(), abs=0.01)
    np.testing.assert_allclose(state[-1, 1:], actions[-1, 1:], atol=1e-3)
    # the arm follows the ramp: halfway through the replay J1 is near the middle of the ramp
    mid = np.argmin(np.abs(j1 - 0.15))
    assert 0.1 < mid / len(j1) < 0.9


# --------------------------------------------------------------------------- CLI over real HTTP


@pytest.fixture
def http_armd(arm):
    """The fake armd on one port for WS and REST GETs (the CLI and subprocesses use real HTTP)."""

    def process_request(conn, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        res = arm.rest(Request("GET", f"http://h{request.path}", None, dict(request.headers.raw_items())))
        return conn.respond(res.status, res.body.decode() if res.body else "")

    server = serve(arm.handler, "127.0.0.1", 0, process_request=process_request)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    arm.http_url = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
    yield arm
    server.shutdown()


def test_cli_sine(http_armd, tmp_path, capsys):
    log_path = tmp_path / "run.jsonl"
    argv = ["--armd", http_armd.http_url, "--policy", "sine:1:0.05:1", "--max-seconds", "0.5", "--log", str(log_path)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "stopped: max_seconds" in out and "parked" in out
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    assert lines[-1]["summary"]["steps"] == 15
    assert [x["step"] for x in lines[:-1]] == list(range(15))


def test_cli_rejects_without_camera(http_armd, capsys, monkeypatch):
    class Img(Policy):
        image_key, image_size = CAM, (H, W)

    import rebot_companion.policy_runner as pr

    monkeypatch.setattr(pr, "parse_policy", lambda *a, **k: Img())
    assert main(["--armd", http_armd.http_url, "--policy", "x"]) == 1
    assert "--camera" in capsys.readouterr().err


# --------------------------------------------------------------------------- learned policy (lerobot)


@pytest.fixture
def camera_server():
    """reCamera-style H.264 stream (Annex-B access unit + 8-byte unix-ms tail per message) at 30 fps."""
    from test_lerobot_recorder import h264_aus

    aus = h264_aus(90)

    def handler(ws):
        i = 0
        try:
            while True:
                ws.send(aus[i % len(aus)] + int(time.time() * 1000).to_bytes(8, "little"))
                i += 1
                time.sleep(1 / FPS)
        except ConnectionClosed:
            pass

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/"
    server.shutdown()


BUILD_ACT = r"""
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
root, out = sys.argv[1], sys.argv[2]
meta = LeRobotDatasetMetadata("test/home", root=root)
cfg = ACTConfig(device="cpu", chunk_size=10, n_action_steps=10, dim_model=32, dim_feedforward=64, n_heads=2,
                n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1, pretrained_backbone_weights=None,
                push_to_hub=False)
policy = make_policy(cfg, ds_meta=meta)
pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
for obj in (policy, pre, post):
    obj.save_pretrained(out)
print("ok")
"""


def lerobot_python() -> str | None:
    try:
        import lerobot.datasets  # noqa: F401  (lerobot[dataset])
        import scipy  # noqa: F401

        return sys.executable
    except ImportError:
        return os.environ.get("REBOT_LEROBOT_PYTHON") or None


def test_act_policy_runs_10_steps(http_armd, camera_server, tmp_path):
    python = lerobot_python()
    if python is None:
        pytest.skip(
            "lerobot not importable (needs Python >= 3.12 and torch). Set REBOT_LEROBOT_PYTHON to a venv python "
            "with lerobot[dataset] 0.6.1 or 0.7.0, PyAV, pyarrow and scipy"
        )
    home_dataset(tmp_path / "ds")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    res = subprocess.run(
        [python, "-c", BUILD_ACT, str(tmp_path / "ds"), str(tmp_path / "act")],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    log_path = tmp_path / "act.jsonl"
    cmd = [python, "-m", "rebot_companion.policy_runner", "--armd", http_armd.http_url, "--camera", camera_server]
    cmd += ["--policy", str(tmp_path / "act"), "--fps", str(FPS), "--max-seconds", str(10 / FPS)]
    cmd += ["--device", "cpu", "--task", "pick the cup", "--log", str(log_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    print(res.stdout)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    summary = lines[-1]["summary"]
    assert summary["reason"] == "max_seconds" and summary["steps"] == 10, summary
    assert summary["policy"].startswith("lerobot:act:")
    assert summary["frames"] > 0 and summary["parked"]
    steps = lines[:-1]
    assert len(steps) == 10 and all(len(s["action"]) == 7 for s in steps)
    assert all(s["camera_age_ms"] is not None for s in steps)
    assert "camera observation.images.recamera (48, 64)" in res.stdout
    assert_in_order(http_armd)
    assert http_armd.dropped_servo_frames == 0


# --------------------------------------------------------------------------- sender unit tests (stub session)


def obs_status(positions=None, state="Moving"):
    from rebot.client import ArmStatus

    return ArmStatus.from_dict(
        {
            "positions": list(HOME if positions is None else positions),
            "safety_state": state,
            "is_torqued": True,
            "epoch_id": 1,
        }
    )


class StubSession:
    """Synchronous stand-in for ArmSession: the sender loop runs in the test thread."""

    lease_lost = None
    has_lease = True

    def __init__(self, runner, obs=None, age=0.0, on_obs=None, max_frames=5):
        self.runner, self.obs, self.age, self.on_obs = runner, obs or obs_status(), age, on_obs
        self.max_frames = max_frames
        self.sent: list[list[float]] = []
        self.reads = 0

    def latest_observation_timed(self):
        self.reads += 1
        if self.on_obs is not None:
            self.on_obs(self)
        return (time.monotonic() - self.age, self.obs)

    def latest_observation(self):
        return self.obs

    def servo(self, positions):
        self.sent.append(list(positions))
        if len(self.sent) >= self.max_frames:
            self.runner.request_stop("test_done")


def stub_runner(policy=None, **kw):
    from rebot_companion.policy_runner import _Segment

    r = PolicyRunner(None, policy or SinePolicy(0, 0.05, 1.0), **kw)
    r.limits = list(LIMITS)
    r.result.lead_clips_per_joint = [0] * 7
    home = np.array(HOME)
    now = time.monotonic()
    r._segment = _Segment(home, home, now, 0.0)
    r._last_target_t = now + 3600  # no stall
    r._deadline = now + 3600
    return r


def test_no_servo_frame_after_a_stop_requested_during_the_tick():
    """A stop that arrives after the loop-boundary check (here: while reading telemetry) sends no frame."""
    r = stub_runner()
    r.session = StubSession(r, on_obs=lambda s: r.request_stop("fault", "injected"))
    r._send_loop()
    assert r.session.sent == []
    assert r.result.reason == "fault"


def test_request_stop_sets_the_event_before_anything_else():
    r = stub_runner()
    seen = {}
    import rebot_companion.policy_runner as pr

    def spy(*a, **k):
        seen["stop_set"] = r._stop.is_set()

    old = pr.log.warning
    pr.log.warning = spy
    try:
        r.request_stop("fault", "x")
    finally:
        pr.log.warning = old
    assert seen["stop_set"] is True


def test_a_hung_sender_fails_closed_with_e_stop():
    """If the sender does not end within sender_join_s (a send stuck in the socket), the runner e-stops over REST."""
    r = stub_runner(send_timeout_s=0.02, sender_join_s=0.05)
    release = threading.Event()
    r._sender = threading.Thread(target=release.wait, daemon=True)
    r._sender.start()
    calls = []

    class Client:
        def e_stop(self):
            calls.append("e_stop")
            return {}

    r.client = Client()
    r.session = StubSession(r)
    try:
        assert r._stop_sender() is False
    finally:
        release.set()
    assert calls == ["e_stop"]
    assert "sender" in r.result.park_detail


def test_request_stop_does_not_wait_forever_for_a_stuck_send():
    r = stub_runner(send_timeout_s=0.02, sender_join_s=0.05)
    held, release = threading.Event(), threading.Event()

    def stuck():
        with r._send_lock:
            held.set()
            release.wait()

    t = threading.Thread(target=stuck, daemon=True)
    t.start()
    held.wait()
    try:
        r.request_stop("fault", "x")  # returns after sender_join_s instead of blocking
        assert r._stop.is_set() and r.result.reason == "fault"
    finally:
        release.set()


# --------------------------------------------------------------------------- timing bounds


@pytest.mark.parametrize("name", ["fps", "servo_hz", "max_seconds", "stall_s", "max_frame_age_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_timing_parameters_must_be_finite_and_positive(name, value):
    with pytest.raises(ValueError, match=name):
        PolicyRunner(None, SinePolicy(0, 0.05, 1.0), **{name: value})


def test_the_sender_enforces_the_deadline():
    r = stub_runner()
    r._deadline = time.monotonic() - 1e-3
    r.session = StubSession(r)
    r._send_loop()
    assert r.session.sent == [] and r.result.reason == "max_seconds"


def test_max_seconds_holds_while_inference_blocks(arm):
    """Inference that blocks past the deadline does not extend the stream: the sender stops on time."""
    holder = {}

    class Blocking(Policy):
        def select(self, obs, step, t):
            if step == 1:
                holder["r"]._stop.wait(5.0)  # released by the sender's deadline stop
            return np.array(obs["observation.state"], dtype=float)

    r = runner(arm, Blocking(), max_seconds=0.1, stall_s=10, park=False)
    holder["r"] = r
    res = r.run()
    assert res.reason == "max_seconds", res
    assert res.frames <= 0.1 * 50 + 2
    assert res.duration_s < 1.0


# --------------------------------------------------------------------------- telemetry checks


def test_stale_telemetry_stops_the_sender():
    r = stub_runner()
    r.session = StubSession(r, age=0.2)  # newest observation received 200 ms ago
    r._send_loop()
    assert r.session.sent == [] and r.result.reason == "stale_observation"


@pytest.mark.parametrize("positions", [[float("nan")] + HOME[1:], HOME[:6], [float("inf")] * 7])
def test_malformed_telemetry_stops_the_sender(positions):
    r = stub_runner()
    r.session = StubSession(r, obs=obs_status(positions))
    r._send_loop()
    assert r.session.sent == [] and r.result.reason == "bad_observation"


@pytest.mark.parametrize("value", [float("nan"), 0, -0.1])
def test_max_obs_age_is_validated(value):
    with pytest.raises(ValueError, match="max_obs_age_s"):
        PolicyRunner(None, SinePolicy(0, 0.05, 1.0), max_obs_age_s=value)


def test_fresh_telemetry_keeps_streaming():
    r = stub_runner()
    r.session = StubSession(r, max_frames=3)
    r._send_loop()
    assert len(r.session.sent) == 3 and r.result.reason == "test_done"


# --------------------------------------------------------------------------- interpolation start


def test_a_new_segment_starts_from_the_last_sent_frame():
    """After lead clipping the last sent frame lags the nominal path; the next segment starts there,
    not at the nominal point, so it does not keep pushing toward the old target."""
    from rebot_companion.policy_runner import _Segment

    r = stub_runner()
    home = np.array(HOME)
    old_target = home.copy()
    old_target[0] = 1.0
    r._segment = _Segment(home, old_target, time.monotonic() - 1.0, 0.1)  # nominal point: 1.0 rad
    sent = home.copy()
    sent[0] = 0.19  # what the lead clip let through
    r._last_sent = sent
    new_target = home.copy()
    new_target[0] = 0.2
    r._publish(new_target)
    np.testing.assert_allclose(r._segment.start, sent)
    frames = [r._segment.at(r._segment.t0 + f * r._segment.duration) for f in np.linspace(0, 1, 6)]
    assert all(0.19 - 1e-9 <= f[0] <= 0.2 + 1e-9 for f in frames)


def test_the_first_segment_starts_from_the_measured_position(arm):
    """Before any frame is sent the reference is the measured position."""
    arm.positions = list(HOME)
    arm.positions[0] = 0.05
    r = runner(arm, SinePolicy(0, 0.05, 1.0), max_seconds=0.05, park=False)
    r.run()
    first = arm.servo_log[0][1]
    assert first[0] == pytest.approx(0.05, abs=0.02)


# --------------------------------------------------------------------------- limits and lead together


def test_lead_clip_cannot_push_a_frame_past_the_joint_limit():
    """Measured J1 beyond its limit by more than the lead: [limit] and [measured +- lead] do not overlap,
    so no frame is sent (the old order clipped to the limit, then the lead clip moved it back out)."""
    r = stub_runner()
    pos = list(HOME)
    pos[0] = LIMITS[0][1] + 0.3
    r.session = StubSession(r, obs=obs_status(pos))
    r._send_loop()
    assert r.session.sent == []
    assert r.result.reason == "outside_limits"


def test_frames_stay_in_the_intersection_of_limit_and_lead():
    r = stub_runner()
    pos = list(HOME)
    pos[0] = LIMITS[0][1] + 0.1  # beyond the limit, but within the lead of it
    r.session = StubSession(r, obs=obs_status(pos), max_frames=2)
    r._send_loop()
    lead = r.lead * 0.95
    for f in r.session.sent:
        assert pos[0] - lead - 1e-9 <= f[0] <= LIMITS[0][1]


@pytest.mark.parametrize(
    "limits",
    [
        LIMITS[:6],
        [(float("nan"), 1.0)] + LIMITS[1:],
        [(1.0, -1.0)] + LIMITS[1:],
        [(-1.0, float("inf"))] + LIMITS[1:],
    ],
)
def test_bad_schema_limits_are_refused_before_torque(arm, monkeypatch, limits):
    import fake_armd

    monkeypatch.setattr(fake_armd, "LIMITS", limits)
    with pytest.raises(RunnerError, match="limits"):
        runner(arm, SinePolicy(0, 0.05, 1.0)).run()
    assert arm.commands == [] and arm.acquires == []


def test_stop_reaches_the_e_stop_within_its_bound_while_a_send_is_stuck():
    """The real sender stuck inside servo() (holding whatever locks it holds while sending): request_stop,
    _publish and _stop_sender all return within their bounds and the REST e-stop goes out."""
    r = stub_runner(send_timeout_s=0.02, sender_join_s=0.05)
    entered, release = threading.Event(), threading.Event()

    class Stuck(StubSession):
        def servo(self, positions):
            entered.set()
            release.wait()

    calls = []

    class Client:
        def e_stop(self):
            calls.append("e_stop")
            return {}

    r.session, r.client = Stuck(r), Client()
    r._sender = threading.Thread(target=r._send_loop, daemon=True)
    r._sender.start()
    assert entered.wait(5)
    done = threading.Event()

    def stopper():
        r._publish(np.array(HOME))  # the policy thread must not block on the stuck send either
        r.request_stop("fault", "x")
        r._stop_sender()
        done.set()

    threading.Thread(target=stopper, daemon=True).start()
    try:
        assert done.wait(1.0), "request_stop/_stop_sender blocked behind the stuck send"
        assert calls == ["e_stop"] and r.result.reason == "fault"
    finally:
        release.set()


# --------------------------------------------------------------------------- blocked inference


@pytest.mark.parametrize(
    "kw,reason",
    [({"max_seconds": 0.3, "stall_s": 10}, "max_seconds"), ({"max_seconds": 10, "stall_s": 0.3}, "policy_stall")],
)
def test_inference_that_never_returns_still_stops_parks_and_releases(arm, kw, reason):
    never = threading.Event()

    class Forever(Policy):
        def select(self, obs, step, t):
            if step == 2:
                never.wait()  # blocks until the test ends
            return np.array(obs["observation.state"], dtype=float)

    out = {}
    t = threading.Thread(target=lambda: out.setdefault("res", runner(arm, Forever(), **kw).run()), daemon=True)
    t.start()
    t.join(5.0)
    never.set()
    assert not t.is_alive(), "run() blocked behind the policy's inference"
    res = out["res"]
    assert res.reason == reason and res.parked, res
    assert res.duration_s < 0.3 + 1.0
    assert "/api/v1/motion/unload" in arm.commands and arm.lease is None


@pytest.mark.parametrize("join", [0.5, 1.0, 1.49, float("nan")])
def test_sender_join_must_exceed_the_send_timeout(join):
    with pytest.raises(ValueError, match="sender_join_s"):
        PolicyRunner(None, SinePolicy(0, 0.05, 1.0), sender_join_s=join)  # default send timeout 1 s
    PolicyRunner(None, SinePolicy(0, 0.05, 1.0), sender_join_s=1.5)
    PolicyRunner(None, SinePolicy(0, 0.05, 1.0), send_timeout_s=0.02, sender_join_s=0.05)
