"""Run a policy on the reBot arm through armd's servo stream (``rebot-policy``).

The runner builds each observation the way ``rebot-record`` writes a dataset row
(:mod:`rebot_companion.lerobot_recorder`), asks the policy for the next joint target and streams
targets to armd as servo frames. armd's servo guard (speed cap, lead cap, blocked hold, collision
hold, faults) applies unchanged; the runner adds its own lead clip and stops on every guard event.

Observation (keys and units as in the dataset):

* ``observation.state`` = ``positions`` (7: joints 1..6 + gripper, rad, float32)
* ``observation.velocity`` = ``velocities``, ``observation.effort`` = ``torques`` when the policy
  lists them as inputs (``rebot-record --extras``)
* ``observation.images.<key>`` = the newest decoded reCamera frame, RGB uint8 (H, W, 3), resized
  with PyAV to the policy's input size. The recorder scales with the same PyAV call before encoding
  the mp4, so the only difference to a training frame is the video codec's loss.

The action (7 values) is the joint target: ``commanded_positions`` in the recorded data.

Timing: policy steps run at ``fps`` (default: the dataset's fps for ``replay``, else 30) on the
main thread. A sender thread streams servo frames at ``servo_hz`` (default 50 Hz, armd's control
rate): when a policy step publishes a target, the sender interpolates linearly from the frame it
last sent to the new target over one policy period, then holds the target. Every frame is clipped
to the joint limits and, for joints 1..6, to ``measured +- 0.95 x servo_max_lead_rad`` (schema
field, default 0.2 rad), so armd's lead cap does not have to act.

Stop conditions: E-stop, Fault, collision hold, servo ``blocked`` hold, torque off, lease loss,
servo rejection, connection loss, policy stall (no new target for ``stall_s``), stale camera frame,
non-finite or wrong-sized action, end of a replayed episode, ``max_steps``, ``max_seconds``,
Ctrl-C / SIGTERM.
After a stop the stream ends, the arm holds (``motion/stop``) and parks (``unload``) unless
``park=False`` or armd cannot park (Fault, E-stop, collision hold, lease lost).

Policies (``--policy``):

* ``replay:<dataset root>:<episode>``: the ``action`` column of a recorded episode (pyarrow).
* ``sine:<joint>:<amp rad>:<period s>``: ``start + amp * sin(2 pi t / period)`` on one joint
  (1..7 or a joint name); ``amp`` <= 0.5 rad.
* anything else: a LeRobot pretrained policy directory or Hub repo id (``PreTrainedConfig`` /
  ``PreTrainedPolicy.from_pretrained``, the saved pre/post-processors, ``predict_action``).
  Needs ``torch`` and ``lerobot`` 0.6.1 / 0.7.0 (Python >= 3.12), which are not dependencies of this package;
  e.g. ``uv run --python 3.12 --with 'lerobot[dataset]' rebot-policy ...``. Imported only for this case.

A ``scripted:`` prefix on ``replay:`` / ``sine:`` is accepted. When armd's follower is not ``mock``
or ``sim`` (``GET /api/v1/safety/describe`` ``device.follower``, unknown on armd without
``safety_describe``), the run needs ``operator_present=True`` (CLI ``--i-am-at-the-arm``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rebot.client import GRIPPER_INDEX, ArmdClient, ArmSession, ArmStatus
from rebot.errors import ArmConnectionError, LeaseError, RebotError, ServoRejected

from .lerobot_recorder import JOINT_NAMES, armd_url

log = logging.getLogger("rebot_companion.policy_runner")

SIM_FOLLOWERS = ("mock", "sim")
DEFAULT_LEAD_RAD = 0.2  # armd's servo_max_lead_rad default (control_safety_contract.md)
LEAD_FRACTION = 0.95  # clip inside armd's lead: the observation the clip uses is a few ms old
LIMIT_MARGIN = 1e-3  # rad inside the joint limits
WARMUP_TIMEOUT_S = 120.0  # first inference (model load, CUDA init) before the stream starts
SEND_TIMEOUT_S = 1.0  # bound on one blocking WebSocket send (servo frame)
STATE_KEY = "observation.state"
EXTRA_KEYS = {"observation.velocity": "velocities", "observation.effort": "torques"}
ROBOT_TYPE = "rebot_b601_dm"


class RunnerError(RuntimeError):
    """The run could not start."""


def _positive(name: str, value: Any) -> float:
    """``value`` as a finite float > 0, else ValueError naming the parameter."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


# --------------------------------------------------------------------------- policies


class Policy:
    """``select(obs, step, t)`` returns the next 7 joint targets, or None when the policy is finished.

    ``inputs``: observation keys the policy reads; ``image_key`` / ``image_size`` (H, W) name the
    camera input (None: no camera). ``fps``: the rate the policy was made for (None: unknown).
    """

    name = "policy"
    fps: int | None = None
    inputs: tuple[str, ...] = (STATE_KEY,)
    image_key: str | None = None
    image_size: tuple[int, int] | None = None
    warmup = False  # run one inference before the stream starts (first-call latency)

    def reset(self) -> None:
        pass

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        raise NotImplementedError


class ReplayPolicy(Policy):
    """Plays the ``action`` column of one episode of a LeRobot v3.0 dataset."""

    def __init__(self, root: str | Path, episode: int) -> None:
        import pyarrow.parquet as pq

        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        self.fps = int(info["fps"])
        rows = []
        for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
            rows += pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pylist()
        row = next((r for r in rows if r["episode_index"] == episode), None)
        if row is None:
            raise RunnerError(f"episode {episode} not in {root} (episodes: {[r['episode_index'] for r in rows]})")
        path = root / info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
        data = pq.read_table(path, columns=["episode_index", "action"]).to_pydict()
        actions = [a for e, a in zip(data["episode_index"], data["action"]) if e == episode]
        if not actions:
            raise RunnerError(f"episode {episode} has no rows in {path}")
        self.actions = np.asarray(actions, dtype=np.float64)
        self.name = f"replay:{root}:{episode}"

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        return self.actions[step] if step < len(self.actions) else None


class SinePolicy(Policy):
    """``start + amp * sin(2 pi t / period)`` on one joint; the other joints hold their start position."""

    def __init__(self, joint: int, amp: float, period: float) -> None:
        if not 0 <= joint < len(JOINT_NAMES):
            raise ValueError(f"joint must be 1..{len(JOINT_NAMES)} or one of {JOINT_NAMES}")
        if not 0 < amp <= 0.5:
            raise ValueError("amp must be in (0, 0.5] rad")
        if period < 0.5:
            raise ValueError("period must be >= 0.5 s")
        self.joint, self.amp, self.period = joint, float(amp), float(period)
        self.name = f"sine:{JOINT_NAMES[joint]}:{amp:g}:{period:g}"
        self._start: np.ndarray | None = None

    def reset(self) -> None:
        self._start = None

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        if self._start is None:
            self._start = np.asarray(obs[STATE_KEY], dtype=np.float64).copy()
        out = self._start.copy()
        out[self.joint] += self.amp * math.sin(2 * math.pi * t / self.period)
        return out


class LeRobotPolicy(Policy):
    """A LeRobot ``PreTrainedPolicy`` with its saved pre/post-processors (lerobot 0.6.1 / 0.7.0 API)."""

    warmup = True

    def __init__(self, path: str, device: str | None = None, task: str = "") -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.configs.types import FeatureType
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as exc:
            raise RunnerError(
                "learned policies need torch and lerobot in this Python (>= 3.12), e.g. "
                f"uv run --python 3.12 --with 'lerobot[dataset]' rebot-policy ...: {exc}"
            ) from exc
        try:
            from lerobot.common.control_utils import predict_action  # 0.6.1, 0.7.0
        except ImportError:  # pragma: no cover - older layout
            from lerobot.utils.control_utils import predict_action

        self._torch, self._predict = torch, predict_action
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        self.device = torch.device(device)
        self.task = task
        self.name = f"lerobot:{cfg.type}:{path}"
        inputs = dict(cfg.input_features or {})
        visual = [k for k, v in inputs.items() if v.type == FeatureType.VISUAL]
        if len(visual) > 1:
            raise RunnerError(f"policy expects {len(visual)} cameras {visual}; rebot-policy feeds one")
        unsupported = [k for k in inputs if k not in visual and k != STATE_KEY and k not in EXTRA_KEYS]
        if unsupported:
            raise RunnerError(f"policy inputs {unsupported} are not produced by rebot-policy")
        if visual:
            self.image_key = visual[0]
            c, h, w = inputs[self.image_key].shape
            self.image_size = (int(h), int(w))
        self.inputs = tuple(inputs)
        action = (cfg.output_features or {}).get("action")
        self.action_dim = int(action.shape[0]) if action is not None else None

    def reset(self) -> None:
        self.policy.reset()
        for p in (self.pre, self.post):
            if hasattr(p, "reset"):
                p.reset()

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        action = self._predict(
            obs, self.policy, self.device, self.pre, self.post, use_amp=False, task=self.task, robot_type=ROBOT_TYPE
        )
        return action.squeeze(0).to("cpu").numpy().astype(np.float64)


def parse_policy(spec: str, device: str | None = None, task: str = "") -> Policy:
    """``replay:<root>:<episode>``, ``sine:<joint>:<amp>:<period>`` (optional ``scripted:`` prefix), or a
    LeRobot pretrained path / repo id."""
    s = spec[len("scripted:") :] if spec.startswith("scripted:") else spec
    if s.startswith("replay:"):
        root, sep, ep = s[len("replay:") :].rpartition(":")
        if not sep or not root or not ep.isdigit():
            raise ValueError("replay policy: replay:<dataset root>:<episode index>")
        return ReplayPolicy(root, int(ep))
    if s.startswith("sine:"):
        parts = s[len("sine:") :].split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError("sine policy: sine:<joint>[:<amp rad>[:<period s>]]")
        j = parts[0]
        joint = int(j) - 1 if j.isdigit() else (JOINT_NAMES.index(j) if j in JOINT_NAMES else -1)
        amp = float(parts[1]) if len(parts) > 1 else 0.05
        period = float(parts[2]) if len(parts) > 2 else 4.0
        return SinePolicy(joint, amp, period)
    if spec.startswith("scripted:"):
        raise ValueError(f"unknown scripted policy {spec!r}: replay:<root>:<episode> or sine:<joint>:<amp>:<period>")
    return LeRobotPolicy(spec, device=device, task=task)


# --------------------------------------------------------------------------- images


def to_image(frame: Any, size: tuple[int, int] | None) -> np.ndarray:
    """Camera item (``CameraFrame``, ``av.VideoFrame`` or RGB array) -> RGB uint8 (H, W, 3) at ``size`` (H, W)."""
    frame = getattr(frame, "frame", frame)
    if isinstance(frame, np.ndarray):
        arr = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if size is None or arr.shape[:2] == tuple(size):
            return arr
        import av

        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
    if size is None:
        return frame.to_ndarray(format="rgb24")
    h, w = size
    return frame.reformat(width=w, height=h, format="rgb24").to_ndarray()


# --------------------------------------------------------------------------- runner


def _stats_ms(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "max": round(float(a.max()), 2),
        "n": len(a),
    }


@dataclass
class RunResult:
    reason: str  # max_seconds | max_steps | policy_done | interrupted | estop | fault | collision_hold | blocked | ...
    detail: str = ""
    policy: str = ""
    follower: str | None = None
    fps: int = 0
    servo_hz: float = 0.0
    duration_s: float = 0.0
    steps: int = 0
    late_steps: int = 0
    frames: int = 0
    late_frames: int = 0
    lead_clipped_frames: int = 0
    lead_clips_per_joint: list[int] = field(default_factory=list)
    limit_clipped_frames: int = 0
    camera_repeats: int = 0
    armd_motion_limits: dict[str, int] = field(default_factory=dict)  # observation.motion_limit reasons seen
    parked: bool = False
    park_detail: str = ""
    latency_ms: dict[str, dict | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reason in ("max_seconds", "max_steps", "policy_done")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class _InferenceWorker:
    """Runs ``policy.select`` on a daemon thread so the main loop can wait with a bound. A call that does
    not return in time is abandoned (the thread stays blocked; the runner stops and cleans up)."""

    def __init__(self, policy: Policy, wake: threading.Event) -> None:
        self.policy, self.wake = policy, wake
        self._req: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._out: tuple[Any, BaseException | None] = (None, None)
        self.busy = False
        threading.Thread(target=self._run, name="policy-inference", daemon=True).start()

    def _run(self) -> None:
        while True:
            args = self._req.get()
            try:
                self._out = (self.policy.select(*args), None)
            except BaseException as exc:
                self._out = (None, exc)
            self._done.set()
            self.wake.set()

    def call(self, args: tuple, timeout: float) -> tuple[bool, Any]:
        """``(True, action)`` or ``(False, None)`` on timeout / wake-up by a stop; re-raises the policy's error."""
        if self.busy:
            return False, None  # a previous call is still running
        self._done.clear()
        self.busy = True
        self._req.put(args)
        end = time.monotonic() + max(0.0, timeout)
        while not self._done.is_set():
            left = end - time.monotonic()
            if left <= 0 or not self.wake.wait(left):
                return False, None
            self.wake.clear()
            if not self._done.is_set():
                return False, None  # woken by request_stop
        self.busy = False
        value, exc = self._out
        if exc is not None:
            raise exc
        return True, value


class _Segment:
    """Linear interpolation from ``start`` to ``end`` over [t0, t0 + duration], then hold ``end``."""

    def __init__(self, start: np.ndarray, end: np.ndarray, t0: float, duration: float) -> None:
        self.start, self.end, self.t0, self.duration = start, end, t0, duration

    def at(self, t: float) -> np.ndarray:
        if self.duration <= 0:
            return self.end
        a = min(1.0, max(0.0, (t - self.t0) / self.duration))
        return self.start + (self.end - self.start) * a


class PolicyRunner:
    """One policy run on one armd. ``client`` is a :class:`rebot.client.ArmdClient`; ``camera`` is any
    source with ``latest() -> (host_monotonic_s, frame) | None`` (optional ``start()``,
    ``wait_first(timeout)``, ``close()``), e.g. :class:`rebot_companion.lerobot_recorder.CameraStream`.
    """

    def __init__(
        self,
        client: ArmdClient,
        policy: Policy,
        camera: Any = None,
        *,
        fps: int | None = None,
        servo_hz: float = 50.0,
        max_seconds: float = 60.0,
        max_steps: int | None = None,
        park: bool = True,
        operator_present: bool = False,
        stall_s: float = 1.0,
        max_frame_age_s: float = 0.5,
        max_obs_age_s: float | None = None,
        log_path: str | Path | None = None,
        client_id: str | None = None,
        connect_timeout: float = 5.0,
        send_timeout_s: float = SEND_TIMEOUT_S,
        sender_join_s: float = 2 * SEND_TIMEOUT_S,
    ) -> None:
        fps_v = _positive("fps", policy.fps or 30 if fps is None else fps)
        if fps_v != int(fps_v):
            raise ValueError(f"fps must be a whole number, got {fps!r}")
        self.fps = int(fps_v)
        self.servo_hz = _positive("servo_hz", servo_hz)
        if self.servo_hz < 10:
            raise ValueError("servo_hz must be >= 10 (armd drops a stream 200 ms after its last frame)")
        self.max_seconds = _positive("max_seconds", max_seconds)
        if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError(f"max_steps must be a whole number >= 1, got {max_steps!r}")
        self.max_steps = max_steps
        self.stall_s = _positive("stall_s", stall_s)
        self.max_frame_age_s = _positive("max_frame_age_s", max_frame_age_s)
        # None: max(0.1 s, 3 observation periods) at the rate armd actually sends, resolved at start
        self._obs_age_explicit = max_obs_age_s is not None
        self.max_obs_age_s = _positive("max_obs_age_s", max_obs_age_s) if self._obs_age_explicit else 0.1
        self.obs_hz = int(max(self.servo_hz, 50))  # requested observation rate
        self._control_hz: float | None = None
        self.client = client
        self.policy = policy
        self.camera = camera
        self.park = park
        self.operator_present = operator_present
        self.log_path = Path(log_path) if log_path else None
        self.client_id = client_id
        self.connect_timeout = connect_timeout
        # The sender's in-flight send ends within send_timeout_s; joining it must allow for that plus margin,
        # else a slow but healthy send would be treated as a stuck sender (REST e-stop).
        self.send_timeout_s = _positive("send_timeout_s", send_timeout_s)
        self.sender_join_s = _positive("sender_join_s", sender_join_s)
        if self.sender_join_s < 1.5 * self.send_timeout_s:
            raise ValueError(
                f"sender_join_s ({sender_join_s}) must be >= 1.5 x send_timeout_s ({self.send_timeout_s:g} s)"
            )
        if policy.image_key and camera is None:
            raise RunnerError(f"policy reads {policy.image_key}: a camera is required")

        self.session: ArmSession | None = None
        self.lead = DEFAULT_LEAD_RAD
        self.limits: list[tuple[float, float]] = []
        self.result = RunResult(reason="", policy=policy.name, fps=self.fps, servo_hz=self.servo_hz)
        self._stop = threading.Event()
        self._wake = threading.Event()  # set by request_stop and by a finished inference
        self._worker: _InferenceWorker | None = None
        # _lock guards the segment and the last committed frame; it is never held across I/O.
        self._lock = threading.RLock()
        # _send_lock is held across "stop requested?" + the servo send only. request_stop takes it with a
        # timeout, so after it returns no further frame is sent unless the sender is stuck in a send
        # (then _stop_sender fails closed). _reason_lock only guards the stop reason.
        self._send_lock = threading.RLock()
        self._reason_lock = threading.Lock()
        self._segment: _Segment | None = None
        self._last_sent: np.ndarray | None = None
        self._last_target_t = 0.0
        self._deadline = math.inf  # monotonic end of the stream, set when it starts
        self._sender: threading.Thread | None = None
        self._lat: dict[str, list[float]] = {
            "obs_age": [],
            "camera_age": [],
            "inference": [],
            "step_lag": [],
            "send_lag": [],
        }
        self._log_file = None
        self._last_seq: Any = None
        self._streaming = False

    # ---- stop bookkeeping ----------------------------------------------------

    def request_stop(self, reason: str, detail: str = "") -> None:
        """First reason wins. Cancels the stream first; returns after any in-flight frame has been written."""
        self._stop.set()
        self._wake.set()
        # Wait for an in-flight frame (bounded by the send timeout). If the lock is not free within
        # sender_join_s the sender is stuck; _stop_sender then fails closed.
        locked = self._send_lock.acquire(timeout=self.sender_join_s)
        try:
            with self._reason_lock:  # never held across anything that blocks
                first = not self.result.reason
                if first:
                    self.result.reason, self.result.detail = reason, detail
        finally:
            if locked:
                self._send_lock.release()
        if first:
            log.warning("stop: %s %s", reason, detail)

    def _check_telemetry(self, timed: tuple[float, ArmStatus] | None, now: float) -> tuple[str, str] | None:
        """Age and shape of the newest observation (before every servo frame and policy step)."""
        if timed is None:
            return "stale_observation", "no observation"
        rx, obs = timed
        age = now - rx
        if age > self.max_obs_age_s:
            return "stale_observation", f"newest observation is {age * 1000:.0f} ms old"
        pos = obs.positions
        if len(pos) != len(JOINT_NAMES) or not all(math.isfinite(p) for p in pos):
            return "bad_observation", f"positions {pos}"
        return None

    def _check(self, obs: ArmStatus, count: bool = False) -> tuple[str, str] | None:
        if obs.safety_state == "EStop":
            return "estop", "E-stop latched"
        if obs.safety_state == "Fault":
            return "fault", obs.fault_code or str(obs.fault_reason or "")
        if obs.collision and obs.collision.get("active"):
            return "collision_hold", f"joint {obs.collision.get('joint')}"
        ml = obs.motion_limit
        if ml:
            reason = str(ml.get("reason"))
            if count:  # once per servo frame
                self.result.armd_motion_limits[reason] = self.result.armd_motion_limits.get(reason, 0) + 1
            if reason == "blocked":
                return "blocked", f"armd held a blocked joint (index {ml.get('joint')})"
        if not obs.is_torqued or not obs.is_operational:
            return "not_armed", f"state {obs.safety_state}, torque {'on' if obs.is_torqued else 'off'}"
        s = self.session
        if s is not None and s.lease_lost is not None:
            return "lease_lost", str(s.lease_lost.get("reason", ""))
        return None

    # ---- setup ---------------------------------------------------------------

    def _read_arm(self) -> None:
        follower = None
        try:
            describe = self.client.get_safety_describe()
            follower = (describe.get("device") or {}).get("follower")
            hz = (describe.get("timing") or {}).get("control_hz")
            if isinstance(hz, (int, float)) and math.isfinite(hz) and hz > 0:
                self._control_hz = float(hz)
        except (RebotError, OSError, ValueError) as exc:
            log.info("no safety description (%s); follower unknown", exc)
        self.result.follower = follower
        if follower not in SIM_FOLLOWERS and not self.operator_present:
            raise RunnerError(
                f"armd follower is {follower or 'unknown'} (not mock/sim): the policy would move a real arm. "
                "Stand at the arm with a way to cut power and pass --i-am-at-the-arm."
            )
        res = self.client.request("GET", "/api/v1/observation/schema")
        res.raise_for_status()
        schema = res.json()
        lead = schema.get("servo_max_lead_rad")
        self.lead = float(lead) if isinstance(lead, (int, float)) and lead > 0 else DEFAULT_LEAD_RAD
        dof = int(schema.get("dof") or 0)
        if dof != len(JOINT_NAMES):
            raise RunnerError(f"armd reports {dof} joints; rebot-policy drives the 7-DOF B601-DM")
        try:
            limits = [(float(lim["min_position"]), float(lim["max_position"])) for lim in schema.get("limits") or []]
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(f"armd schema joint limits are malformed: {exc}") from exc
        if len(limits) != dof or not all(
            math.isfinite(lo) and math.isfinite(hi) and hi - lo > 2 * LIMIT_MARGIN for lo, hi in limits
        ):
            raise RunnerError(f"armd schema joint limits are unusable (need {dof} finite min < max): {limits}")
        self.limits = limits

    def _observation(self, obs: ArmStatus) -> dict[str, np.ndarray]:
        out = {STATE_KEY: np.asarray(obs.positions, dtype=np.float32)}
        for key, attr in EXTRA_KEYS.items():
            if key in self.policy.inputs:
                out[key] = np.asarray(getattr(obs, attr), dtype=np.float32)
        return out

    def _camera_image(self, now: float) -> np.ndarray | None:
        if not self.policy.image_key:
            return None
        item = self.camera.latest()
        if item is None:
            raise RunnerError("no camera frame")
        rx, frame = item
        age = now - rx
        self._lat["camera_age"].append(age)
        if age > self.max_frame_age_s:
            self.request_stop("camera_stale", f"newest frame is {age * 1000:.0f} ms old")
            return None
        seq = getattr(frame, "seq", None)
        if seq is not None and seq == self._last_seq:
            self.result.camera_repeats += 1
        self._last_seq = seq
        return to_image(frame, self.policy.image_size)

    # ---- servo sender --------------------------------------------------------

    def _shape(self, frame: np.ndarray, measured: Sequence[float]) -> tuple[np.ndarray | None, str]:
        """Clip ``frame`` per joint to [limit] intersected with [measured +- lead] (gripper: limit only).
        Returns ``(frame, "")``, or ``(None, detail)`` when a joint's interval is empty."""
        out = np.array(frame, dtype=np.float64)
        lead = self.lead * LEAD_FRACTION
        limit_clip = lead_clip = False
        for j in range(len(out)):
            lim_lo, lim_hi = self.limits[j][0] + LIMIT_MARGIN, self.limits[j][1] - LIMIT_MARGIN
            lo, hi = lim_lo, lim_hi
            if j != GRIPPER_INDEX:  # armd exempts the gripper from the lead cap
                m = measured[j]
                lo, hi = max(lo, m - lead), min(hi, m + lead)
                if lo > hi:
                    return None, f"joint {j + 1} measured {m:.3f} rad, limits [{lim_lo:.3f}, {lim_hi:.3f}]"
            x = out[j]
            v = min(max(x, lo), hi)
            if not lim_lo <= x <= lim_hi:
                limit_clip = True
            if v != min(max(x, lim_lo), lim_hi):
                self.result.lead_clips_per_joint[j] += 1
                lead_clip = True
            out[j] = v
        self.result.limit_clipped_frames += limit_clip
        self.result.lead_clipped_frames += lead_clip
        return out, ""

    def _send_loop(self) -> None:
        period = 1.0 / self.servo_hz
        next_t = time.monotonic()
        s = self.session
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= self._deadline:
                    self.request_stop("max_seconds", f"{self.max_seconds:g} s")
                    break
                self._lat["send_lag"].append(max(0.0, now - next_t))
                timed = s.latest_observation_timed()
                stop = self._check_telemetry(timed, time.monotonic())
                if stop:
                    self.request_stop(*stop)
                    break
                obs = timed[1]
                stop = self._check(obs, count=True)
                if stop:
                    self.request_stop(*stop)
                    break
                with self._lock:
                    last_target = self._last_target_t
                if now - last_target > self.stall_s:
                    self.request_stop("policy_stall", f"no new target for {now - last_target:.2f} s")
                    break
                # Compute and commit the frame under _lock (it becomes the next segment's start before it is
                # sent, so _publish never starts from an older frame), then send under _send_lock only.
                with self._lock:
                    frame, problem = self._shape(self._segment.at(now), obs.positions)
                    if frame is not None:
                        self._last_sent = frame
                if frame is None:
                    self.request_stop("outside_limits", problem)
                    break
                with self._send_lock:
                    if self._stop.is_set() or time.monotonic() >= self._deadline:
                        break
                    s.servo(frame.tolist())
                self.result.frames += 1
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    if self._stop.wait(delay):
                        break
                elif -delay > period:
                    self.result.late_frames += 1
                    next_t = time.monotonic()
            if time.monotonic() >= self._deadline:
                self.request_stop("max_seconds", f"{self.max_seconds:g} s")  # first reason wins
        except ServoRejected as exc:
            self.request_stop("servo_rejected", str(exc))
        except LeaseError as exc:
            self.request_stop("lease_lost", str(exc))
        except ArmConnectionError as exc:
            self.request_stop("connection_lost", str(exc))
        except RebotError as exc:  # ServoJumpError, ...
            self.request_stop(exc.code.lower() if exc.code else "armd_error", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.request_stop("sender_error", f"{type(exc).__name__}: {exc}")

    def _publish(self, target: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            # start from what armd last received (after clipping), not from the nominal path
            if self._last_sent is not None:
                start = self._last_sent.copy()
            else:
                start = self._segment.at(now) if self._segment is not None else target
            self._segment = _Segment(np.asarray(start, dtype=np.float64), target, now, 1.0 / self.fps)
            self._last_target_t = now

    # ---- run -----------------------------------------------------------------

    def run(self) -> RunResult:
        """Blocks until a stop condition; returns the result (``result.reason`` says why it stopped)."""
        self._read_arm()
        self.result.lead_clips_per_joint = [0] * len(JOINT_NAMES)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w", encoding="utf-8")
        cam_started = False
        t_start = time.monotonic()
        try:
            if self.camera is not None and self.policy.image_key:
                if hasattr(self.camera, "start"):
                    self.camera.start()
                cam_started = True
                if hasattr(self.camera, "wait_first"):
                    self.camera.wait_first(self.connect_timeout)
            self.session = ArmSession(
                self.client,
                client_id=self.client_id,
                safe_exit="off",
                connect_timeout=self.connect_timeout,
                obs_hz=self.obs_hz,
                send_timeout=self.send_timeout_s,
            ).open()
            try:
                self._prepare()
                t_start = time.monotonic()
                self._loop()
            except KeyboardInterrupt:
                self.request_stop("interrupted", "Ctrl-C / SIGTERM")
            except RunnerError:
                if not self._streaming:
                    raise
                self.request_stop("error", str(sys.exc_info()[1]))
            except Exception as exc:
                if not self._streaming:
                    raise
                self.request_stop("error", f"{type(exc).__name__}: {exc}")
            finally:
                self._stop.set()
                sender_done = self._stop_sender()
                self.result.duration_s = round(time.monotonic() - t_start, 3)
                if self._streaming and sender_done:
                    self._finish()
        finally:
            if self.session is not None:
                try:
                    self.session.close()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("closing the session failed: %s", exc)
            if cam_started and hasattr(self.camera, "close"):
                self.camera.close()
            self.result.latency_ms = {k + "_ms": _stats_ms(v) for k, v in self._lat.items()}
            if self._log_file is not None:
                self._log_file.write(json.dumps({"summary": self.result.to_dict()}) + "\n")
                self._log_file.close()
        return self.result

    def _resolve_obs_age(self) -> None:
        """Default observation age bound: max(0.1 s, 3 periods) at min(requested obs_hz, armd control_hz,
        the rate measured over the next 4 observations)."""
        if self._obs_age_explicit:
            return
        hz = float(self.obs_hz)
        if self._control_hz:
            hz = min(hz, self._control_hz)
        s, rx = self.session, []
        end = time.monotonic() + self.connect_timeout
        while len(rx) < 4 and time.monotonic() < end:
            s.wait_for_observation(self.connect_timeout)
            rx.append(s.latest_observation_timed()[0])
        if len(rx) >= 2 and rx[-1] > rx[0]:
            hz = min(hz, (len(rx) - 1) / (rx[-1] - rx[0]))
        self.max_obs_age_s = max(0.1, 3.0 / hz)
        log.info("observations at %.1f Hz; max observation age %.0f ms", hz, self.max_obs_age_s * 1000)

    def _infer(self, args: tuple, timeout: float) -> tuple[bool, Any]:
        if self._worker is None:
            self._worker = _InferenceWorker(self.policy, self._wake)
        return self._worker.call(args, timeout)

    def _stop_sender(self) -> bool:
        """Wait for the sender thread; True when it has ended. A sender still alive after ``sender_join_s``
        is stuck in a socket send, so the WebSocket cannot carry ``motion/stop``: fail closed with the
        lease-free REST e-stop (latched; a person clears it in the cockpit)."""
        t = self._sender
        if t is None:
            return True
        t.join(self.sender_join_s)
        if not t.is_alive():
            return True
        log.error("servo sender did not stop within %.1f s; sending e-stop", self.sender_join_s)
        try:
            self.client.e_stop()
            self.result.park_detail = "not parked: the servo sender did not stop; e-stop sent"
        except Exception as exc:
            self.result.park_detail = f"not parked: the servo sender did not stop; e-stop failed: {exc}"
        return False

    def _prepare(self) -> None:
        s = self.session
        obs = s.wait_for_observation(self.connect_timeout)
        if obs.is_stopped:
            raise RunnerError(
                f"arm is in {obs.safety_state} ({obs.fault_code or obs.fault_reason}); "
                "check it and clear the fault in the cockpit"
            )
        if obs.collision and obs.collision.get("active"):
            raise RunnerError("arm is holding after a collision; wait until it resumes")
        if not obs.is_torqued or not obs.is_operational:
            s.torque(True)
            deadline = time.monotonic() + self.connect_timeout
            while not (obs.is_torqued and obs.is_operational):
                if time.monotonic() > deadline:
                    raise RunnerError(f"torque on did not arm the arm (state {obs.safety_state})")
                obs = s.wait_for_observation(self.connect_timeout)
        self._resolve_obs_age()
        self.policy.reset()
        if self.policy.warmup:
            image = self._camera_image(time.monotonic()) if self.policy.image_key else None
            o = self._observation(obs)
            if image is not None:
                o[self.policy.image_key] = image
            t = time.perf_counter()
            ok, _ = self._infer((o, 0, 0.0), WARMUP_TIMEOUT_S)
            if not ok:
                raise RunnerError(f"warm-up inference did not return within {WARMUP_TIMEOUT_S:g} s")
            log.info("warm-up inference %.1f ms", (time.perf_counter() - t) * 1000)
            self.policy.reset()
        here = np.asarray(s.latest_observation().positions, dtype=np.float64)
        with self._lock:
            self._segment = _Segment(here, here, time.monotonic(), 0.0)
            self._last_sent = here.copy()  # reference for the first segment: the measured position
            self._last_target_t = time.monotonic()
        self._sender = threading.Thread(target=self._send_loop, name="policy-servo", daemon=True)
        # absolute end of the stream, enforced by the sender whatever the policy's inference time
        self._deadline = time.monotonic() + self.max_seconds
        self._streaming = True
        self._sender.start()

    def _loop(self) -> None:
        s = self.session
        period = 1.0 / self.fps
        t0 = time.monotonic()
        k = 0
        while not self._stop.is_set():
            due = t0 + k * period
            now = time.monotonic()
            if due > now:
                if self._stop.wait(due - now):
                    break
                now = time.monotonic()
            lag = now - due
            self._lat["step_lag"].append(lag)
            if lag > period:
                self.result.late_steps += 1
                t0 = now - k * period  # re-anchor: one step per tick, never a burst of catch-up steps
            if now >= self._deadline:
                self.request_stop("max_seconds", f"{self.max_seconds:g} s")
                break
            timed = s.latest_observation_timed()
            stop = self._check_telemetry(timed, time.monotonic())
            if stop:
                self.request_stop(*stop)
                break
            rx, obs = timed
            self._lat["obs_age"].append(now - rx)
            stop = self._check(obs)
            if stop:
                self.request_stop(*stop)
                break
            o = self._observation(obs)
            if self.policy.image_key:
                image = self._camera_image(now)
                if image is None:
                    break
                o[self.policy.image_key] = image
            t_inf = time.perf_counter()
            # bounded wait: inference that blocks past the deadline or stall_s does not block the stop
            ok, action = self._infer((o, k, k * period), min(self._deadline - time.monotonic(), self.stall_s))
            inf = time.perf_counter() - t_inf
            if not ok:
                if time.monotonic() >= self._deadline:
                    self.request_stop("max_seconds", f"{self.max_seconds:g} s")
                else:
                    self.request_stop("policy_stall", f"inference took longer than {self.stall_s:g} s")
                break
            self._lat["inference"].append(inf)
            if action is None:
                self.request_stop("policy_done", f"{k} steps")
                break
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] != len(JOINT_NAMES) or not np.all(np.isfinite(action)):
                self.request_stop("bad_action", f"policy returned {action.tolist()}")
                break
            self._publish(action)
            self.result.steps = k + 1
            if self._log_file is not None:
                self._log_file.write(
                    json.dumps(
                        {
                            "step": k,
                            "t": round(k * period, 4),
                            "obs_age_ms": round((now - rx) * 1000, 2),
                            "camera_age_ms": round(self._lat["camera_age"][-1] * 1000, 2)
                            if self.policy.image_key
                            else None,
                            "inference_ms": round(inf * 1000, 2),
                            "step_lag_ms": round(lag * 1000, 2),
                            "action": [round(float(x), 5) for x in action],
                            "state": [round(float(x), 5) for x in obs.positions],
                        }
                    )
                    + "\n"
                )
            k += 1
            if self.max_steps is not None and k >= self.max_steps:
                self.request_stop("max_steps", f"{k} steps")
                break

    def _finish(self) -> None:
        s = self.session
        if not self.result.reason:
            self.result.reason = "stopped"
        try:
            obs = s.latest_observation()
        except RebotError as exc:
            self.result.park_detail = f"not parked: {exc}"
            return
        if obs is None or s.lease_lost is not None or not s.has_lease:
            self.result.park_detail = "not parked: the control lease is gone"
            return
        if obs.is_stopped:
            self.result.park_detail = f"not parked: arm in {obs.safety_state}; clear it in the cockpit"
            return
        if obs.collision and obs.collision.get("active"):
            self.result.park_detail = "not parked: collision hold"
            return
        try:
            if obs.is_operational:
                s.stop()
            if not self.park:
                self.result.park_detail = "holding (--no-park)"
                return
            if not obs.is_torqued:
                self.result.park_detail = "not parked: torque is off"
                return
            s.unload(wait=True, timeout=30.0)
            self.result.parked = True
            self.result.park_detail = "parked"
        except RebotError as exc:
            self.result.park_detail = f"park failed: {exc}"


# --------------------------------------------------------------------------- CLI


def _print_summary(r: RunResult) -> None:
    print(f"stopped: {r.reason}" + (f" ({r.detail})" if r.detail else ""))
    print(f"policy {r.policy}, follower {r.follower or 'unknown'}, {r.fps} Hz policy, {r.servo_hz:g} Hz servo")
    print(
        f"{r.steps} steps ({r.late_steps} late), {r.frames} servo frames ({r.late_frames} late), {r.duration_s:.2f} s"
    )
    print(
        f"lead-clipped frames {r.lead_clipped_frames} (per joint {r.lead_clips_per_joint}), "
        f"limit-clipped frames {r.limit_clipped_frames}"
        + (f", armd motion limits {r.armd_motion_limits}" if r.armd_motion_limits else "")
    )
    for name, st in r.latency_ms.items():
        if st:
            cols = "  ".join(f"{k} {st[k]:7.2f}" for k in ("mean", "p50", "p95", "max"))
            print(f"  {name:16s} {cols}")
    print(r.park_detail)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rebot-policy",
        description="Run a LeRobot or scripted policy on the reBot arm through armd's servo stream.",
    )
    p.add_argument("--armd", default="192.168.16.1", help="armd host or URL (default port 8765)")
    p.add_argument("--camera", help="reCamera host or ws:// URL (default port 8001); needed by image policies")
    p.add_argument(
        "--policy",
        required=True,
        help="LeRobot pretrained dir / Hub repo id, replay:<dataset root>:<episode>, or sine:<joint>:<amp>:<period>",
    )
    p.add_argument("--fps", type=int, help="policy steps per second (default: dataset fps for replay, else 30)")
    p.add_argument("--servo-hz", type=float, default=50.0, help="servo frame rate (default 50)")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, help="stop after this many policy steps")
    p.add_argument("--task", default="", help="task text for language-conditioned policies")
    p.add_argument("--device", help="torch device (default cuda if available, else cpu)")
    p.add_argument("--no-park", action="store_true", help="hold position at the end instead of parking")
    p.add_argument("--stall-s", type=float, default=1.0, help="stop when the policy gives no target for this long")
    p.add_argument("--max-frame-age", type=float, default=0.5, help="stop when the newest camera frame is older (s)")
    p.add_argument(
        "--max-obs-age",
        type=float,
        help="stop when the newest armd observation is older (s; default max(0.1, 3 / observation rate))",
    )
    p.add_argument("--log", help="write per-step JSON lines (latency, action, state) and a summary")
    p.add_argument(
        "--i-am-at-the-arm",
        action="store_true",
        help="required when armd drives a real arm (follower not mock/sim): you watch it and can cut power",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    def _term(*_: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _term)
    except ValueError:  # not the main thread (tests)
        pass
    camera = None
    try:
        policy = parse_policy(args.policy, device=args.device, task=args.task)
        if policy.image_key:
            if not args.camera:
                raise RunnerError(f"policy reads {policy.image_key}: pass --camera")
            from .lerobot_recorder import CameraStream

            camera = CameraStream(args.camera)
        runner = PolicyRunner(
            ArmdClient(armd_url(args.armd)),
            policy,
            camera,
            fps=args.fps,
            servo_hz=args.servo_hz,
            max_seconds=args.max_seconds,
            max_steps=args.max_steps,
            park=not args.no_park,
            operator_present=args.i_am_at_the_arm,
            stall_s=args.stall_s,
            max_frame_age_s=args.max_frame_age,
            max_obs_age_s=args.max_obs_age,
            log_path=args.log,
        )
        print(
            f"policy {policy.name}" + (f", camera {policy.image_key} {policy.image_size}" if policy.image_key else "")
        )
        result = runner.run()
    except (RunnerError, ValueError, ImportError, RebotError, OSError) as exc:
        print(f"rebot-policy: {exc}", file=sys.stderr)
        return 1
    _print_summary(result)
    if result.reason == "interrupted":
        return 130
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
