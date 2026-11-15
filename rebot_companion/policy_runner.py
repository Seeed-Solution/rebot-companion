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

Timing: policy steps run at ``fps`` (default: the dataset's fps for ``replay``, the fps in a learned
policy's ``train_config.json`` or its training dataset, which is required otherwise; 30 for ``sine``) on the
main thread. A sender thread streams servo frames at ``servo_hz`` (default 50 Hz, armd's control
rate): when a policy step publishes a target, the sender interpolates linearly from the frame it
last sent to the new target over one policy period, then holds the target. When the policy ends
(end of a replay, ``max_steps``) the stream continues until the last segment has reached its target,
plus one policy period. Every frame is clipped
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
* ``onnx:<dir>`` or local directory containing ``policy.json``: an ONNX policy package
  (``rebot-onnx-policy/v1``) driven by ``onnxruntime``.
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
import os
import queue
import re
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rebot.client import ArmdClient, ArmSession, ArmStatus, ObservationSchema
from rebot.errors import ArmConnectionError, LeaseError, RebotError, ServoRejected

from . import units
from .lerobot_recorder import EXTRAS, JOINT_NAMES, armd_url
from .onnx_policy import validate_image_spec, validate_normalization_stats, validate_policy_dir

log = logging.getLogger("rebot_companion.policy_runner")

SIM_FOLLOWERS = ("mock", "sim")
DEFAULT_LEAD_RAD = 0.2  # armd's servo_max_lead_rad default (control_safety_contract.md)
LEAD_FRACTION = 0.95  # clip inside armd's lead: the observation the clip uses is a few ms old
LIMIT_MARGIN = 1e-3  # rad inside the joint limits
WARMUP_TIMEOUT_S = 120.0  # first inference (model load, CUDA init) before the stream starts
SEND_TIMEOUT_S = 1.0  # bound on one blocking WebSocket send (servo frame)
STATE_KEY = "observation.state"
EXTRA_KEYS = dict(EXTRAS.values())  # dataset key -> ArmStatus attribute, as rebot-record writes them
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
    camera input (None: no camera). ``fps``: the rate the policy was made for (None: unknown, the
    runner then needs an explicit ``fps``); scripted policies default to 30.
    """

    name = "policy"
    source = ""  # where a learned policy was loaded from (local directory / Hugging Face Hub)
    fps: int | None = 30
    fps_problem = ""  # why fps is None (shown when the runner asks for an explicit fps)
    inputs: tuple[str, ...] = (STATE_KEY,)
    image_key: str | None = None
    image_size: tuple[int, int] | None = None
    warmup = False  # run one inference before the stream starts (first-call latency)
    # set by the runner: the inference thread of the last run. While it is alive (a call abandoned after
    # a stall or the deadline is still inside select) the policy cannot be used by another run.
    _inference_thread: threading.Thread | None = None

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
        if "stats" not in info:
            for stats_file in (root / "meta" / "stats.json", root / "stats.json"):
                if stats_file.is_file():
                    try:
                        stats_data = json.loads(stats_file.read_text())
                        if isinstance(stats_data, dict):
                            info["stats"] = stats_data
                            break
                    except Exception as exc:
                        log.warning("failed to read %s: %s", stats_file, exc)
        self.fps = int(info["fps"])
        rows = []
        for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
            rows += pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pylist()
        row = next((r for r in rows if r["episode_index"] == episode), None)
        if row is None:
            raise RunnerError(f"episode {episode} not in {root} (episodes: {[r['episode_index'] for r in rows]})")
        path = self._data_file(root, info, row)
        data = pq.read_table(path, columns=["episode_index", "action"]).to_pydict()
        actions = [a for e, a in zip(data["episode_index"], data["action"]) if e == episode]
        if not actions:
            raise RunnerError(f"episode {episode} has no rows in {path}")
        self.actions = np.asarray(actions, dtype=np.float64)
        self.action_dim = int(self.actions.shape[1])
        self.info = info
        self.unit_mode = info.get("unit_mode")
        self.name = f"replay:{root}:{episode}"

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        return self.actions[step] if step < len(self.actions) else None

    @staticmethod
    def _data_file(root: Path, info: dict, row: dict) -> Path:
        """The episode's data file from ``info.json`` ``data_path``; it must resolve inside the dataset root."""
        template = info.get("data_path")
        if not isinstance(template, str) or not template:
            raise RunnerError(f"{root / 'meta' / 'info.json'} has no data_path")
        try:
            rel = template.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise RunnerError(f"info.json data_path {template!r} cannot be resolved: {exc!r}") from None
        base = root.resolve()
        path = (base / rel).resolve()
        if not path.is_relative_to(base):
            raise RunnerError(f"info.json data_path {template!r} resolves to {path}, outside the dataset {base}")
        return path


class SinePolicy(Policy):
    """``start + amp * sin(2 pi t / period)`` on one joint; the other joints hold their start position."""

    def __init__(self, joint: int | str, amp: float, period: float, target_joint: str | int | None = None) -> None:
        if not 0 < amp <= 0.5:
            raise ValueError("amp must be in (0, 0.5] rad")
        if period < 0.5:
            raise ValueError("period must be >= 0.5 s")

        self.amp, self.period = float(amp), float(period)
        self._start: np.ndarray | None = None

        if isinstance(joint, str):
            self.target_joint: str | int = joint
            if joint.isdigit():
                val = int(joint)
                if not 1 <= val <= len(JOINT_NAMES):
                    raise ValueError(f"joint must be 1..{len(JOINT_NAMES)} or one of {JOINT_NAMES}")
                self.joint = val - 1
            elif joint == "gripper":
                self.joint = 6
            elif joint in JOINT_NAMES:
                self.joint = JOINT_NAMES.index(joint)
            else:
                if not joint.isidentifier():
                    raise ValueError(f"invalid joint specification {joint!r}")
                self.joint = -1
        else:
            if not 0 <= joint < len(JOINT_NAMES):
                raise ValueError(f"joint must be 1..{len(JOINT_NAMES)} or one of {JOINT_NAMES}")
            self.joint = int(joint)
            self.target_joint = target_joint if target_joint is not None else (self.joint + 1)

        name_target = (
            str(self.target_joint)
            if not isinstance(self.target_joint, int)
            else (JOINT_NAMES[self.joint] if 0 <= self.joint < len(JOINT_NAMES) else str(self.target_joint))
        )
        self.name = f"sine:{name_target}:{self.amp:g}:{self.period:g}"

    def reset(self) -> None:
        self._start = None

    def bind_schema(self, schema: ObservationSchema) -> None:
        """Resolve symbolic joint names and validate joint target against arm schema."""
        dof = schema.dof
        gripper_joint = schema.gripper_joint
        joint_names = list(schema.joint_names) if schema.joint_names else []

        if self.target_joint == "gripper":
            if gripper_joint is None:
                raise RunnerError("sine:gripper specified but arm has no gripper joint in schema")
            if not 0 <= gripper_joint < dof:
                raise RunnerError(f"schema gripper_joint {gripper_joint} out of range for dof {dof}")
            self.joint = gripper_joint
            return

        if isinstance(self.target_joint, str) and not self.target_joint.isdigit():
            target = self.target_joint
            if joint_names:
                if target in joint_names:
                    self.joint = joint_names.index(target)
                else:
                    raise RunnerError(f"unknown joint name {target!r} in sine policy (schema joints: {joint_names})")
            elif target in JOINT_NAMES:
                idx = JOINT_NAMES.index(target)
                if idx >= dof or (gripper_joint is not None and idx == gripper_joint):
                    raise RunnerError(
                        f"unknown joint name {target!r} in sine policy (schema joints: {JOINT_NAMES[:dof]})"
                    )
                self.joint = idx
            else:
                raise RunnerError(f"unknown joint name {target!r} in sine policy (schema joints: {JOINT_NAMES[:dof]})")
            return

        # 1-based integer or 0-based joint index
        if self.joint < 0 or self.joint >= dof:
            one_based = self.joint + 1 if self.joint >= 0 else self.target_joint
            raise RunnerError(f"sine policy joint index {one_based} exceeds arm DOF {dof}")

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        if self._start is None:
            self._start = np.asarray(obs[STATE_KEY], dtype=np.float64).copy()
        out = self._start.copy()
        if not 0 <= self.joint < len(out):
            raise RunnerError(f"sine policy joint {self.joint} out of bounds for state dim {len(out)}")
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
        local = Path(path).expanduser()
        if local.is_dir():
            path = str(local)
            self.source = f"local directory {local.resolve()}"
        elif path.startswith(("/", ".", "~")) or path.count("/") > 1:
            raise RunnerError(f"policy directory {path} does not exist")
        else:
            self.source = f"Hugging Face Hub repo {path}"
        log.info("loading policy from %s", self.source)
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
            _, h, w = inputs[self.image_key].shape
            self.image_size = (int(h), int(w))
        self.inputs = tuple(inputs)
        self.info = pretrained_info(path)
        try:
            found = pretrained_fps(path)
        except ValueError as exc:
            found, self.fps_problem = None, str(exc)
            log.warning("policy fps unknown: %s", exc)
        else:
            self.fps_problem = "" if found else "no fps in its train_config.json or training dataset"
        self.fps = found[0] if found else None
        if found:
            log.info("policy fps %d from %s", *found)
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


class OnnxPolicy(Policy):
    """Unified ``rebot-onnx-policy/v1`` policy executed via ``onnxruntime``."""

    warmup = True

    def __init__(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        if p.is_file() and p.name == "policy.json":
            p = p.parent
        if not p.is_dir():
            raise RunnerError(f"ONNX policy directory {path} does not exist")

        try:
            manifest = validate_policy_dir(p)
        except FileNotFoundError as exc:
            raise RunnerError(str(exc)) from exc

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RunnerError("ONNX policies require onnxruntime, e.g. uv run --extra onnx rebot-policy ...") from exc

        model_path = p / manifest.model_file
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.manifest = manifest
        self.name = f"onnx:{manifest.name}"
        self.source = f"local directory {p.resolve()}"
        self.fps = manifest.fps
        self.action_dim = manifest.action_dim

        raw_unit_mode = None
        policy_json_path = p / "policy.json"
        if policy_json_path.is_file():
            try:
                raw_json = json.loads(policy_json_path.read_text(encoding="utf-8"))
                if isinstance(raw_json, dict):
                    raw_unit_mode = raw_json.get("unit_mode")
            except Exception:
                raw_unit_mode = None

        if raw_unit_mode is not None:
            self.unit_mode = manifest.unit_mode
            self.info = {"unit_mode": manifest.unit_mode, "fps": manifest.fps}
        else:
            self.unit_mode = None
            self.info = {"fps": manifest.fps}

        state_spec = manifest.inputs["state"]
        self._state_input_name = str(state_spec["name"])

        img_spec = manifest.inputs.get("image")
        if img_spec is not None:
            validate_image_spec(img_spec)
            self._image_input_name = str(img_spec["name"])
            self._image_layout = str(img_spec.get("layout", "NCHW")).upper()
            shape = [int(v) for v in img_spec["shape"]]
            if self._image_layout == "NCHW":
                self._image_channels = shape[1]
                h, w = shape[2], shape[3]
                mean_broadcast = (1, -1, 1, 1)
            else:
                self._image_channels = shape[3]
                h, w = shape[1], shape[2]
                mean_broadcast = (1, 1, 1, -1)
            self.image_size = (h, w)
            self.image_key = (
                self._image_input_name
                if self._image_input_name.startswith("observation.images.")
                else "observation.images.wrist"
            )
            self._image_scale = float(img_spec.get("scale", 1.0 / 255.0))
            mean = img_spec.get("mean")
            std = img_spec.get("std")
            self._image_mean = np.asarray(mean, dtype=np.float32).reshape(mean_broadcast) if mean is not None else None
            self._image_std = np.asarray(std, dtype=np.float32).reshape(mean_broadcast) if std is not None else None
            self.inputs = (STATE_KEY, self.image_key)
        else:
            self._image_input_name = None
            self._image_channels = 3
            self.image_key = None
            self.image_size = None
            self._image_layout = "NCHW"
            self._image_scale = 1.0 / 255.0
            self._image_mean = None
            self._image_std = None
            self.inputs = (STATE_KEY,)

        norm = manifest.normalization or {}
        state_shape = manifest.inputs.get("state", {}).get("shape", [])
        state_dim = int(state_shape[-1]) if state_shape else manifest.action_dim
        validate_normalization_stats(norm, state_dim=state_dim, action_dim=manifest.action_dim)

        self._state_mean = (
            np.asarray(norm["state_mean"], dtype=np.float32).reshape(-1) if norm.get("state_mean") is not None else None
        )
        self._state_std = (
            np.asarray(norm["state_std"], dtype=np.float32).reshape(-1) if norm.get("state_std") is not None else None
        )
        self._action_mean = (
            np.asarray(norm["action_mean"], dtype=np.float32).reshape(-1)
            if norm.get("action_mean") is not None
            else None
        )
        self._action_std = (
            np.asarray(norm["action_std"], dtype=np.float32).reshape(-1) if norm.get("action_std") is not None else None
        )

        self._action_output_name = str(manifest.outputs["action"]["name"])
        io_spec = manifest.outputs.get("io") or {}
        self._io_output_name = io_spec.get("name")
        self._io_channels = list(io_spec.get("channels") or ())
        self.last_io_updates: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_io_updates = {}

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        raw_state = obs[STATE_KEY] if STATE_KEY in obs else obs[self._state_input_name]
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if self._state_mean is not None and self._state_std is not None:
            if state.shape != self._state_mean.shape:
                raise ValueError(
                    f"Observation state shape {state.shape} does not match "
                    f"normalization state_mean shape {self._state_mean.shape}"
                )
            state = (state - self._state_mean) / self._state_std

        feed: dict[str, np.ndarray] = {self._state_input_name: np.ascontiguousarray(state.reshape(1, -1))}
        if self._image_input_name is not None and self.image_key is not None:
            raw_img = obs[self.image_key] if self.image_key in obs else obs[self._image_input_name]
            arr = np.asarray(raw_img)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if (
                arr.ndim == 3
                and arr.shape[-1] in (1, 3, 4)
                and self.image_size is not None
                and arr.shape[:2] != self.image_size
                and np.issubdtype(arr.dtype, np.integer)
            ):
                arr = to_image(arr, self.image_size)
            scale = self._image_scale if np.issubdtype(arr.dtype, np.integer) else 1.0
            arr_f = arr.astype(np.float32) * scale
            if arr_f.ndim == 3 and arr_f.shape[-1] in (1, 3, 4):
                if self._image_layout == "NCHW":
                    arr_f = np.transpose(arr_f[..., : self._image_channels], (2, 0, 1))
                else:
                    arr_f = arr_f[..., : self._image_channels]
            elif arr_f.ndim == 3 and arr_f.shape[0] in (1, 3, 4) and self._image_layout == "NHWC":
                arr_f = np.transpose(arr_f[: self._image_channels, ...], (1, 2, 0))
            batch = arr_f[np.newaxis, ...]
            if self._image_mean is not None and self._image_std is not None and batch.ndim == 4:
                batch = (batch - self._image_mean) / self._image_std
            feed[self._image_input_name] = np.ascontiguousarray(batch, dtype=np.float32)

        output_names = [self._action_output_name]
        if self._io_output_name:
            output_names.append(str(self._io_output_name))
        outs = self.session.run(output_names, feed)

        action = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if self._action_mean is not None and self._action_std is not None:
            if action.shape != self._action_mean.shape:
                raise ValueError(
                    f"Action output shape {action.shape} does not match "
                    f"normalization action_mean shape {self._action_mean.shape}"
                )
            action = action * self._action_std + self._action_mean

        if self._io_output_name and len(outs) > 1:
            io_vals = np.asarray(outs[1], dtype=np.float32).reshape(-1)
            updates: dict[str, Any] = {}
            for idx, ch in enumerate(self._io_channels):
                if idx >= io_vals.shape[0]:
                    break
                val = float(io_vals[idx])
                if isinstance(ch, dict):
                    ch_name = str(ch.get("name", ""))
                    if ch_name:
                        updates[ch_name] = bool(val > 0.5) if ch.get("kind") == "bool" else val
                elif isinstance(ch, str) and ch:
                    updates[ch] = bool(val > 0.5) if ch == "vacuum" else val
            self.last_io_updates = updates

        return action.astype(np.float64)


def _fps_field(container: Any, where: str) -> int | None:
    """``container["fps"]`` as a whole number > 0; None when absent. ValueError when ``container`` is not
    a JSON object or the fps is not a finite, positive, integral number."""
    if container is None:
        return None
    if not isinstance(container, dict):
        raise ValueError(f"{where} is not a JSON object")
    v = container.get("fps")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 or v != int(v):
        raise ValueError(f"{where} fps {v!r} is not a whole number > 0")
    return int(v)


def _load_train_config(path: str | Path) -> tuple[dict | None, Path | None]:
    """Load ``train_config.json`` from a local directory or download it from the Hub.

    Returns ``(cfg_dict, cfg_path)``, or ``(None, None)`` when the file is absent.
    Raises ValueError when the JSON is malformed or not a dict.
    """
    local = Path(path).expanduser()
    cfg_file = local / "train_config.json"
    if not local.is_dir():
        try:
            from huggingface_hub import hf_hub_download

            cfg_file = Path(hf_hub_download(str(path), "train_config.json"))
        except Exception as exc:  # no huggingface_hub, offline, no such file
            log.info("no train_config.json for %s: %s", path, exc)
            return None, None
    try:
        cfg = json.loads(cfg_file.read_text())
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        raise ValueError(f"{cfg_file} is not readable JSON: {exc}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_file} is not a JSON object")
    return cfg, cfg_file


def _resolve_dataset_roots(cfg: dict, cfg_file: Path | None = None) -> list[Path]:
    """Extract candidate dataset root paths from a train_config dict."""
    dataset = cfg.get("dataset") or {}
    for key in ("root", "repo_id"):
        if dataset.get(key) is not None and not isinstance(dataset[key], str):
            prefix = f"{cfg_file} " if cfg_file else ""
            raise ValueError(f"{prefix}dataset.{key} {dataset[key]!r} is not a string")
    roots = []
    if dataset.get("root"):
        roots.append(Path(dataset["root"]).expanduser())
    if dataset.get("repo_id"):
        home = os.environ.get("HF_LEROBOT_HOME") or Path.home() / ".cache" / "huggingface" / "lerobot"
        roots.append(Path(home) / dataset["repo_id"])
    return roots


def pretrained_fps(path: str | Path) -> tuple[int, str] | None:
    """``(fps, where it came from)`` for a LeRobot pretrained directory or Hub repo id, from its
    ``train_config.json``: ``env.fps``, ``dataset.fps``, else the training dataset's ``meta/info.json``
    (``dataset.root``, else ``$HF_LEROBOT_HOME/<dataset.repo_id>``). None when none of them is available;
    ValueError when one of them is malformed (not an object, fps not a whole number > 0)."""
    cfg, cfg_file = _load_train_config(path)
    if cfg is None:
        return None
    for section in ("env", "dataset"):
        fps = _fps_field(cfg.get(section), f"{cfg_file} {section}")
        if fps:
            return fps, f"{cfg_file} {section}.fps"
    roots = _resolve_dataset_roots(cfg, cfg_file)
    for root in roots:
        info = root / "meta" / "info.json"
        try:
            meta = json.loads(info.read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise ValueError(f"{info} is not readable JSON: {exc}") from None
        fps = _fps_field(meta, str(info))
        if fps:
            return fps, str(info)
    return None


def pretrained_info(path: str | Path) -> dict | None:
    """Read dataset and policy metadata (``meta/info.json``, ``stats.json``, configs) for a pretrained policy.

    Returns a dict with dataset info/stats/unit_mode, or None. Logs warnings on non-fatal errors.
    """
    local = Path(path).expanduser()
    try:
        cfg, cfg_file = _load_train_config(path)
    except Exception as exc:
        log.warning("failed to load train_config for %s: %s", path, exc)
        cfg, cfg_file = None, None

    policy_cfg: dict | None = None
    if local.is_dir():
        p_cfg = local / "config.json"
        if p_cfg.is_file():
            try:
                data = json.loads(p_cfg.read_text())
                if isinstance(data, dict):
                    policy_cfg = data
            except Exception as exc:
                log.warning("failed to read %s: %s", p_cfg, exc)

    roots: list[Path] = []
    if local.is_dir():
        roots.append(local)
    if cfg is not None:
        try:
            roots.extend(_resolve_dataset_roots(cfg, cfg_file))
        except Exception as exc:
            log.warning("failed to resolve dataset roots for %s: %s", path, exc)

    info_dict: dict | None = None
    info_path: Path | None = None
    for root in roots:
        candidates = [root / "meta" / "info.json", root / "info.json"]
        found = False
        for info_file in candidates:
            if info_file.is_file():
                try:
                    data = json.loads(info_file.read_text())
                    if isinstance(data, dict):
                        info_dict = data
                        info_path = info_file
                        found = True
                        break
                except Exception as exc:
                    log.warning("failed to read %s: %s", info_file, exc)
        if found:
            break

    stats_data: dict | None = None
    if info_dict is not None and "stats" in info_dict and isinstance(info_dict["stats"], dict):
        stats_data = info_dict["stats"]
    else:
        stats_candidates: list[Path] = []
        if info_path is not None:
            stats_candidates.append(info_path.parent / "stats.json")
        for root in roots:
            stats_candidates.append(root / "meta" / "stats.json")
            stats_candidates.append(root / "stats.json")
        if local.is_dir():
            stats_candidates.append(local / "meta" / "stats.json")
            stats_candidates.append(local / "stats.json")
        for stats_file in stats_candidates:
            if stats_file.is_file():
                try:
                    s_data = json.loads(stats_file.read_text())
                    if isinstance(s_data, dict):
                        stats_data = s_data
                        break
                except Exception as exc:
                    log.warning("failed to read %s: %s", stats_file, exc)

    if stats_data is not None:
        if info_dict is None:
            info_dict = {"stats": stats_data}
        else:
            info_dict["stats"] = stats_data

    cfg_unit_mode = None
    candidates = [cfg]
    if isinstance(cfg, dict) and isinstance(cfg.get("dataset"), dict):
        candidates.append(cfg["dataset"])
    if isinstance(policy_cfg, dict):
        candidates.append(policy_cfg)
    for candidate in candidates:
        if isinstance(candidate, dict):
            m = candidate.get("unit_mode") or candidate.get("rebot_unit_mode")
            if m:
                cfg_unit_mode = m
                break

    if cfg_unit_mode:
        if info_dict is None:
            info_dict = {"unit_mode": cfg_unit_mode}
        elif "unit_mode" not in info_dict and "rebot_unit_mode" not in info_dict:
            info_dict["unit_mode"] = cfg_unit_mode

    return info_dict


_PREFIX = re.compile(r"[A-Za-z][\w.+-]*:")


def parse_policy(spec: str, device: str | None = None, task: str = "") -> Policy:
    """``replay:<root>:<episode>``, ``sine:<joint>:<amp>:<period>`` (optional ``scripted:`` prefix),
    ``onnx:<dir>``, a local directory containing ``policy.json``, or a LeRobot pretrained path / repo id."""
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
        amp = float(parts[1]) if len(parts) > 1 else 0.05
        period = float(parts[2]) if len(parts) > 2 else 4.0
        return SinePolicy(j, amp, period)
    if s.startswith("onnx:") and not spec.startswith("scripted:"):
        return OnnxPolicy(s[len("onnx:") :])
    if spec.startswith("scripted:"):
        raise ValueError(f"unknown scripted policy {spec!r}: replay:<root>:<episode> or sine:<joint>:<amp>:<period>")
    local = Path(spec).expanduser()
    if local.is_dir() and (local / "policy.json").is_file():
        if not (local / "config.json").is_file() and not (local / "pretrained_model").is_dir():
            return OnnxPolicy(local)
    elif local.is_file() and local.name == "policy.json":
        return OnnxPolicy(local)
    # a Hub repo id ("org/name") has no colon: "<word>:..." is a mistyped prefix unless it is a local directory
    if _PREFIX.match(spec) and not local.is_dir():
        raise ValueError(
            f"unknown policy prefix in {spec!r}: use onnx:<dir>, replay:<root>:<episode>, sine:<joint>:<amp>:<period>, "
            "a LeRobot pretrained directory or a Hub repo id"
        )
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
    not return in time is abandoned (the thread stays blocked; the runner stops and cleans up).
    :meth:`close` ends the thread: at once when idle, else when the abandoned call returns."""

    def __init__(self, policy: Policy, wake: threading.Event) -> None:
        self.policy, self.wake = policy, wake
        self._req: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._out: tuple[Any, BaseException | None] = (None, None)
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)
        policy._inference_thread = self.thread
        self.thread.start()

    def close(self, timeout: float) -> bool:
        """Stop the thread (sentinel) and wait up to ``timeout``; True when it has ended. An abandoned
        call is not waited for: the thread ends when it returns."""
        self._req.put(None)
        if not self.busy:
            self.thread.join(timeout)
        self._out = (None, None)
        return not self.thread.is_alive()

    def _run(self) -> None:
        while True:
            args = self._req.get()
            if args is None:
                return
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
        action_units: str = "auto",
        state_units: str = "auto",
        lerobot_calib: str | Path | None = None,
        io_sender: Any = None,
    ) -> None:
        if fps is None and policy.fps is None:
            raise ValueError(f"fps of {policy.name} is unknown ({policy.fps_problem or 'not given'}): pass --fps")
        if fps is not None and policy.fps is not None and fps != policy.fps:
            log.warning("fps %s differs from the %d fps %s was made for", fps, policy.fps, policy.name)
        fps_v = _positive("fps", policy.fps if fps is None else fps)
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
        self.action_units_requested = action_units
        self.state_units_requested = state_units
        self.lerobot_calib = lerobot_calib
        if self.lerobot_calib is not None:
            if "deg" in (action_units, state_units):
                raise RunnerError(
                    "--lerobot-calib is incompatible with 'deg' unit mode; it calibrates 'so101_norm' [-100, 100] ranges"
                )
            p_info = getattr(policy, "info", None)
            if isinstance(p_info, dict):
                p_unit = p_info.get("unit_mode") or p_info.get("rebot_unit_mode")
                if p_unit == "deg" and (action_units == "auto" or state_units == "auto"):
                    raise RunnerError(
                        "--lerobot-calib is incompatible with 'deg' unit mode; it calibrates 'so101_norm' [-100, 100] ranges"
                    )

        p_info = getattr(policy, "info", None)
        if isinstance(p_info, dict):
            meta_unit = p_info.get("unit_mode") or p_info.get("rebot_unit_mode")
            if meta_unit is not None:
                meta_unit_str = str(meta_unit).strip()
                if meta_unit_str:
                    if action_units != "auto" and action_units != meta_unit_str:
                        log.warning(
                            "explicit action_units %r differs from policy metadata unit_mode %r",
                            action_units,
                            meta_unit_str,
                        )
                    if state_units != "auto" and state_units != meta_unit_str:
                        log.warning(
                            "explicit state_units %r differs from policy metadata unit_mode %r",
                            state_units,
                            meta_unit_str,
                        )
        self._lerobot_limits: list[tuple[float, float]] | None = None
        self._lerobot_closed_rad: float | None = None
        self._lerobot_open_rad: float | None = None
        self._recorded_limits: list[tuple[float, float]] | None = None
        self._recorded_closed_rad: float | None = None
        self._recorded_open_rad: float | None = None
        self.action_units = "rad"
        self.state_units = "rad"
        self.dof = len(JOINT_NAMES)
        self.gripper_joint: int | None = 6
        self.closed_rad: float = 0.0
        self.open_rad: float = -5.7
        self.schema: ObservationSchema | None = None
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
        self._ran = False  # a runner runs once
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
        self.io_sender = io_sender
        self._last_io_sent: dict[str, Any] = {}
        self._sandbox_io_checked: bool = False
        self._sandbox_io_enabled: bool = bool(io_sender is not None)
        base_url = getattr(client, "base_url", None)
        if isinstance(base_url, str) and base_url:
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            self.host = parsed.hostname or "127.0.0.1"
            self.port = parsed.port or 8765
        else:
            raw_host = getattr(client, "host", None)
            raw_port = getattr(client, "port", None)
            self.host = raw_host if isinstance(raw_host, str) else "127.0.0.1"
            self.port = raw_port if isinstance(raw_port, int) else 8765

        self._torque_sent = False  # torque-on was sent: the arm may be armed, so cleanup stops and parks
        self._stopping = False  # stop/cleanup in progress: handle_signal ignores further signals

    # ---- stop bookkeeping ----------------------------------------------------

    def handle_signal(self, signum: int, frame: Any = None) -> None:
        """SIGINT / SIGTERM handler (install on the main thread, which calls :meth:`run`). The first signal
        while the run is active raises KeyboardInterrupt, which stops the run on its normal path. Once a stop
        is in progress, later signals are ignored so stop, e-stop and park complete."""
        if self._stopping or self._stop.is_set():
            os.write(2, b"rebot-policy: stopping; finishing stop and park\n")  # print() is not reentrant
            return
        self._stopping = True
        raise KeyboardInterrupt

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
        if len(pos) != self.dof or not all(math.isfinite(p) for p in pos):
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
        raw_schema = res.json()
        schema = ObservationSchema.from_dict(raw_schema)
        self.schema = schema
        self.dof = schema.dof
        self.gripper_joint = schema.gripper_joint

        try:
            limits = [
                (float(lim["min_position"]), float(lim["max_position"])) for lim in raw_schema.get("limits") or []
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(f"armd schema joint limits are malformed: {exc}") from exc
        if len(limits) != self.dof or not all(
            math.isfinite(lo) and math.isfinite(hi) and hi - lo > 2 * LIMIT_MARGIN for lo, hi in limits
        ):
            raise RunnerError(f"armd schema joint limits are unusable (need {self.dof} finite min < max): {limits}")
        self.limits = limits

        self.closed_rad, self.open_rad = units.resolve_schema_gripper_bounds(
            schema.gripper_policy, limits, self.gripper_joint
        )

        if self.lerobot_calib is not None:
            if not (
                self.dof == 6 and self.gripper_joint == 5 and (self.schema is None or self.schema.arm_dof in (None, 5))
            ):
                raise RunnerError(
                    f"--lerobot-calib requires a 6-DOF arm with gripper_joint=5 (got dof={self.dof}, "
                    f"gripper_joint={self.gripper_joint}, arm_dof={getattr(self.schema, 'arm_dof', None)})"
                )
            if self.schema is None or getattr(self.schema, "feetech_joints", None) is None:
                log.warning(
                    "PolicyRunner: --lerobot-calib used but follower feetech_joints schema is missing; "
                    "assuming default direction and center ticks"
                )
            l_limits, l_closed, l_open = units.load_lerobot_norm_bounds(self.lerobot_calib, schema=self.schema)
            self._lerobot_limits = l_limits
            self._lerobot_closed_rad = l_closed
            self._lerobot_open_rad = l_open

        lead = raw_schema.get("servo_max_lead_rad")
        self.lead = float(lead) if isinstance(lead, (int, float)) and lead > 0 else DEFAULT_LEAD_RAD

        policy_dof = None
        if hasattr(self.policy, "action_dim") and self.policy.action_dim is not None:
            policy_dof = self.policy.action_dim
        elif (
            hasattr(self.policy, "actions")
            and hasattr(self.policy.actions, "shape")
            and len(self.policy.actions.shape) > 1
        ):
            policy_dof = int(self.policy.actions.shape[1])
        elif hasattr(self.policy, "dof"):
            policy_dof = self.policy.dof

        if policy_dof is not None and policy_dof != self.dof:
            raise RunnerError(f"policy action dimension ({policy_dof}) does not match arm DOF ({self.dof})")

        if isinstance(self.policy, SinePolicy):
            self.policy.bind_schema(schema)

        prof = (schema.profile_id or "").lower()
        is_so_arm = (
            prof.startswith("so_arm")
            or prof.startswith("so-arm")
            or schema.arm_dof == 5
            or (self.dof == 6 and self.gripper_joint is not None)
        )
        is_learned_or_replay = isinstance(self.policy, (LeRobotPolicy, ReplayPolicy, OnnxPolicy))
        require_meta = is_learned_or_replay and is_so_arm

        policy_info = getattr(self.policy, "info", None)
        if isinstance(self.policy, LeRobotPolicy) and policy_info is None:
            extracted_stats = None
            for proc in (getattr(self.policy, "pre", None), getattr(self.policy, "post", None)):
                if proc is not None and hasattr(proc, "stats") and isinstance(proc.stats, dict):
                    extracted_stats = proc.stats
                    break
            if extracted_stats:
                policy_info = {"stats": extracted_stats}
            elif not is_so_arm:
                log.warning("LeRobotPolicy metadata/stats unavailable; falling back to rad units on 7-DOF arm")

        if isinstance(self.policy, SinePolicy):
            if self.action_units_requested not in ("auto", "rad") or self.state_units_requested not in ("auto", "rad"):
                raise RunnerError("sine policy works in radians only; omit --state-units/--action-units or pass 'rad'")
            self.action_units = "rad"
            self.state_units = "rad"
        else:
            self.action_units = units.resolve_unit_mode(
                self.action_units_requested, policy_info, require_metadata=require_meta
            )
            self.state_units = units.resolve_unit_mode(
                self.state_units_requested, policy_info, require_metadata=require_meta
            )

        if self.lerobot_calib is not None and ("deg" in (self.action_units, self.state_units)):
            raise RunnerError(
                "--lerobot-calib is incompatible with 'deg' unit mode; it calibrates 'so101_norm' [-100, 100] ranges"
            )

        if isinstance(policy_info, dict):
            meta_unit = policy_info.get("unit_mode") or policy_info.get("rebot_unit_mode")
            if meta_unit is not None:
                meta_unit_str = str(meta_unit).strip()
                if meta_unit_str:
                    if self.action_units_requested != "auto" and self.action_units_requested != meta_unit_str:
                        log.warning(
                            "explicit action_units %r differs from policy metadata unit_mode %r",
                            self.action_units_requested,
                            meta_unit_str,
                        )
                    if self.state_units_requested != "auto" and self.state_units_requested != meta_unit_str:
                        log.warning(
                            "explicit state_units %r differs from policy metadata unit_mode %r",
                            self.state_units_requested,
                            meta_unit_str,
                        )

        if (
            any(m in ("so101_norm", "deg") for m in (self.action_units, self.state_units))
            and self.lerobot_calib is None
        ):
            norm_b = policy_info.get("normalization_bounds") if isinstance(policy_info, dict) else None
            if isinstance(norm_b, dict) and ("limits" in norm_b or "closed_rad" in norm_b or "open_rad" in norm_b):
                rec_lims = norm_b.get("limits")
                if rec_lims is not None:
                    if not isinstance(rec_lims, (list, tuple)) or len(rec_lims) != self.dof:
                        lims_len = len(rec_lims) if isinstance(rec_lims, (list, tuple)) else "invalid"
                        raise RunnerError(
                            f"normalization_bounds limits length ({lims_len}) does not match arm DOF ({self.dof})"
                        )
                    parsed_lims = []
                    for i, lim in enumerate(rec_lims):
                        if not isinstance(lim, (list, tuple)) or len(lim) != 2:
                            raise RunnerError(
                                f"normalization_bounds limits[{i}] must be sequence of length 2, got {lim!r}"
                            )
                        try:
                            lo = float(lim[0])
                            hi = float(lim[1])
                        except (ValueError, TypeError) as exc:
                            raise RunnerError(
                                f"normalization_bounds limits[{i}] values must be floats, got {lim!r}: {exc}"
                            ) from exc
                        if not (math.isfinite(lo) and math.isfinite(hi)):
                            raise RunnerError(
                                f"normalization_bounds limits[{i}] values must be finite floats, got ({lo}, {hi})"
                            )
                        if abs(hi - lo) <= 1e-6:
                            raise RunnerError(
                                f"normalization_bounds limits[{i}] range must satisfy "
                                f"abs(hi - lo) > 1e-6, got ({lo}, {hi})"
                            )
                        parsed_lims.append((lo, hi))
                    self._recorded_limits = parsed_lims

                if "closed_rad" in norm_b and norm_b.get("closed_rad") is not None:
                    val = norm_b["closed_rad"]
                    try:
                        c_rad = float(val)
                    except (ValueError, TypeError) as exc:
                        raise RunnerError(f"normalization_bounds closed_rad must be float, got {val!r}: {exc}") from exc
                    if not math.isfinite(c_rad):
                        raise RunnerError(f"normalization_bounds closed_rad must be finite, got {val!r}")
                    self._recorded_closed_rad = c_rad

                if "open_rad" in norm_b and norm_b.get("open_rad") is not None:
                    val = norm_b["open_rad"]
                    try:
                        o_rad = float(val)
                    except (ValueError, TypeError) as exc:
                        raise RunnerError(f"normalization_bounds open_rad must be float, got {val!r}: {exc}") from exc
                    if not math.isfinite(o_rad):
                        raise RunnerError(f"normalization_bounds open_rad must be finite, got {val!r}")
                    self._recorded_open_rad = o_rad

                if self._recorded_closed_rad is not None and self._recorded_open_rad is not None:
                    if abs(self._recorded_open_rad - self._recorded_closed_rad) <= 1e-6:
                        raise RunnerError(
                            f"normalization_bounds open_rad ({self._recorded_open_rad}) and "
                            f"closed_rad ({self._recorded_closed_rad}) must satisfy abs(open_rad - closed_rad) > 1e-6"
                        )
            elif (self.action_units == "so101_norm" or self.state_units == "so101_norm") and not (
                isinstance(policy_info, dict) and policy_info.get("rebot_unit_mode")
            ):
                log.warning(
                    "so101_norm is active without --lerobot-calib and without 'rebot_unit_mode' in metadata; "
                    "normalizing using armd schema joint limits (rebot-record convention). "
                    "If running an external LeRobot policy, pass --lerobot-calib <calib.json> to align motor ranges."
                )

    def _observation(self, obs: ArmStatus) -> dict[str, np.ndarray]:
        if self._lerobot_limits is not None:
            limits = self._lerobot_limits
            closed_rad = self._lerobot_closed_rad
            open_rad = self._lerobot_open_rad
        else:
            limits = self._recorded_limits if self._recorded_limits is not None else self.limits
            closed_rad = self._recorded_closed_rad if self._recorded_closed_rad is not None else self.closed_rad
            open_rad = self._recorded_open_rad if self._recorded_open_rad is not None else self.open_rad
        positions = units.from_armd(
            obs.positions,
            self.state_units,
            limits=limits,
            gripper_joint=self.gripper_joint,
            closed_rad=closed_rad,
            open_rad=open_rad,
        )
        out = {STATE_KEY: np.asarray(positions, dtype=np.float32)}
        for key, attr in EXTRA_KEYS.items():
            if key in self.policy.inputs:
                out[key] = np.asarray(getattr(obs, attr), dtype=np.float32)
        return out

    def _camera_stale(self, now: float, age: float | None = None) -> bool:
        """True (and a ``camera_stale`` stop) when the newest frame is older than ``max_frame_age_s``."""
        if age is None:
            item = self.camera.latest()
            if item is None:
                self.request_stop("camera_stale", "no camera frame")
                return True
            age = now - item[0]
        if age > self.max_frame_age_s:
            self.request_stop("camera_stale", f"newest frame is {age * 1000:.0f} ms old")
            return True
        return False

    def _camera_image(self, now: float) -> np.ndarray | None:
        item = self.camera.latest()
        if item is None:
            raise RunnerError("no camera frame")
        rx, frame = item
        age = now - rx
        self._lat["camera_age"].append(age)
        if self._camera_stale(now, age):
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
            if self.gripper_joint is None or j != self.gripper_joint:  # armd exempts the gripper from the lead cap
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
            # start from what armd last received (after clipping), not from the nominal path;
            # _prepare sets _last_sent to the measured position before the first step
            self._segment = _Segment(self._last_sent.copy(), target, now, 1.0 / self.fps)
            self._last_target_t = now

    def _policy_has_io(self) -> bool:
        if getattr(self.policy, "io", None):
            return True
        manifest = getattr(self.policy, "manifest", None)
        if manifest and hasattr(manifest, "outputs") and isinstance(manifest.outputs, dict):
            if manifest.outputs.get("io"):
                return True
        if getattr(self.policy, "_io_output_name", None) or getattr(self.policy, "_io_channels", None):
            return True
        if getattr(self.policy, "last_io_updates", None) is not None or hasattr(self.policy, "io_sequence"):
            return True
        return False

    def _check_sandbox_io(self) -> None:
        """Check armd sandbox mode and decide whether to enable sandbox IO forwarding."""
        if self._sandbox_io_checked:
            return
        self._sandbox_io_checked = True

        if self.io_sender is not None:
            self._sandbox_io_enabled = True
            return

        if not self._policy_has_io():
            self._sandbox_io_enabled = False
            return

        mode = None
        try:
            if hasattr(self.client, "request"):
                res = self.client.request("GET", "/api/v1/sandbox/state")
                status = getattr(res, "status_code", getattr(res, "status", 200))
                if isinstance(status, int) and status >= 400:
                    raise RuntimeError(f"HTTP {status}")
                if hasattr(res, "raise_for_status") and callable(res.raise_for_status):
                    res.raise_for_status()
                if hasattr(res, "json") and callable(res.json):
                    data = res.json()
                elif hasattr(res, "text") and isinstance(res.text, str):
                    import json

                    data = json.loads(res.text) if res.text else {}
                else:
                    data = {}
                mode = data.get("mode") if isinstance(data, dict) else None
            else:
                import json
                import urllib.error
                import urllib.request

                url = f"http://{self.host}:{self.port}/api/v1/sandbox/state"
                req = urllib.request.Request(
                    url,
                    headers={"Accept": "application/json"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw) if raw else {}
                    mode = data.get("mode") if isinstance(data, dict) else None
        except Exception as exc:
            log.warning(
                "Sandbox IO forwarding disabled: /api/v1/sandbox/state unavailable (%s). "
                "Running on real hardware or unconfigured sandbox.",
                exc,
            )
            self._sandbox_io_enabled = False
            return

        if mode in {"sim", "wasm"}:
            self._sandbox_io_enabled = True
            log.info("Sandbox IO forwarding enabled (armd sandbox mode: %s).", mode)
        else:
            self._sandbox_io_enabled = False
            log.warning(
                "Sandbox IO forwarding disabled: armd sandbox mode is %r (requires 'sim' or 'wasm').",
                mode,
            )

    def _reset_io_channels(self) -> None:
        """Best-effort reset of previously sent boolean IO channels back to False."""
        if not self._sandbox_io_enabled:
            return
        bool_resets = {k: False for k, v in self._last_io_sent.items() if isinstance(v, bool) and v is not False}
        if not bool_resets:
            bool_resets = {k: False for k, v in self._last_io_sent.items() if isinstance(v, bool)}
        if not bool_resets:
            return
        try:
            if self.io_sender is not None:
                self.io_sender(bool_resets)
            elif hasattr(self.client, "request"):
                res = self.client.request("POST", "/api/v1/sandbox/io", json={"updates": bool_resets})
                if hasattr(res, "raise_for_status") and callable(res.raise_for_status):
                    res.raise_for_status()
                elif hasattr(res, "status_code") and isinstance(res.status_code, int) and res.status_code >= 400:
                    raise RuntimeError(f"HTTP {res.status_code}")
                elif hasattr(res, "status") and isinstance(res.status, int) and res.status >= 400:
                    raise RuntimeError(f"HTTP {res.status}")
            else:
                import json
                import urllib.request

                url = f"http://{self.host}:{self.port}/api/v1/sandbox/io"
                data = json.dumps({"updates": bool_resets}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0):
                    pass
            self._last_io_sent.update(bool_resets)
        except Exception as exc:
            log.warning("Failed to reset boolean IO channels to False on stop: %s", exc)

    def _send_io_update(self, updates: dict[str, Any]) -> None:
        """Send changed IO outputs to armd POST /api/v1/sandbox/io."""
        if not updates:
            return
        if not self._sandbox_io_checked:
            self._check_sandbox_io()
        if not self._sandbox_io_enabled:
            return
        # deduplicate identical consecutive updates
        changed = {k: v for k, v in updates.items() if self._last_io_sent.get(k) != v}
        if not changed:
            return

        try:
            if self.io_sender is not None:
                self.io_sender(changed)
            elif hasattr(self.client, "request"):
                res = self.client.request("POST", "/api/v1/sandbox/io", json={"updates": changed})
                if hasattr(res, "raise_for_status") and callable(res.raise_for_status):
                    res.raise_for_status()
                elif hasattr(res, "status_code") and isinstance(res.status_code, int) and res.status_code >= 400:
                    raise RuntimeError(f"HTTP {res.status_code}")
                elif hasattr(res, "status") and isinstance(res.status, int) and res.status >= 400:
                    raise RuntimeError(f"HTTP {res.status}")
            else:
                import json
                import urllib.request

                url = f"http://{self.host}:{self.port}/api/v1/sandbox/io"
                data = json.dumps({"updates": changed}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5.0):
                    pass
            self._last_io_sent.update(changed)
        except Exception as exc:
            self.request_stop("io_delivery_failed", f"Failed to deliver IO updates: {exc}")
            raise

    # ---- run -----------------------------------------------------------------

    def run(self) -> RunResult:
        """Blocks until a stop condition; returns the result (``result.reason`` says why it stopped).
        A runner runs once; a policy whose inference from an earlier run is still blocked is refused."""
        if self._ran:
            raise RunnerError("a PolicyRunner runs once; create a new one")
        self._ran = True
        t = self.policy._inference_thread
        if t is not None and t.is_alive():
            raise RunnerError(
                f"policy {self.policy.name} is still inside an inference abandoned by an earlier run; "
                "wait for it or create a new policy"
            )
        self._read_arm()
        self._check_sandbox_io()
        self.result.lead_clips_per_joint = [0] * self.dof
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
            if getattr(self.session, "_schema", None) is None:
                if hasattr(self.session, "set_schema"):
                    self.session.set_schema(self.schema)
                else:
                    self.session._schema = self.schema
            try:
                self._prepare()
                t_start = time.monotonic()
                self._loop()
            except KeyboardInterrupt:
                self._stopping = True
                self.request_stop("interrupted", "Ctrl-C / SIGTERM")
            except Exception as exc:
                self._stopping = True
                if not self._streaming:
                    raise
                self.request_stop("error", str(exc) if isinstance(exc, RunnerError) else f"{type(exc).__name__}: {exc}")
            finally:
                self._stopping = True
                self._stop.set()
                self._reset_io_channels()
                sender_done = self._stop_sender()
                self.result.duration_s = round(time.monotonic() - t_start, 3)
                if (self._streaming or self._torque_sent) and sender_done:
                    self._finish()
        finally:
            self._stopping = True
            w, self._worker = self._worker, None
            if w is not None and not w.close(timeout=1.0):
                log.warning("policy inference is still running (abandoned); its thread ends when it returns")
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
            self._torque_sent = True  # before the call: an interrupt can arrive after armd applied it
            s.torque(True)
            deadline = time.monotonic() + self.connect_timeout
            while not (obs.is_torqued and obs.is_operational):
                if time.monotonic() > deadline:
                    raise RunnerError(f"torque on did not arm the arm (state {obs.safety_state})")
                obs = s.wait_for_observation(self.connect_timeout)
        self._resolve_obs_age()
        self.policy.reset()
        if self.policy.warmup:
            o = self._observation(obs)
            if self.policy.image_key:
                image = self._camera_image(time.monotonic())
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
                self._end_of_policy("policy_done", f"{k} steps")
                break
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] != self.dof or not np.all(np.isfinite(action)):
                self.request_stop("bad_action", f"policy returned {action.tolist()}")
                break
            if self.action_units != "rad":
                if self._lerobot_limits is not None:
                    limits = self._lerobot_limits
                    closed_rad = self._lerobot_closed_rad
                    open_rad = self._lerobot_open_rad
                else:
                    limits = self._recorded_limits if self._recorded_limits is not None else self.limits
                    closed_rad = self._recorded_closed_rad if self._recorded_closed_rad is not None else self.closed_rad
                    open_rad = self._recorded_open_rad if self._recorded_open_rad is not None else self.open_rad
                action_rad = units.to_armd(
                    action.tolist(),
                    self.action_units,
                    limits=limits,
                    gripper_joint=self.gripper_joint,
                    closed_rad=closed_rad,
                    open_rad=open_rad,
                )
                action = np.asarray(action_rad, dtype=np.float64).reshape(-1)
                if action.shape[0] != self.dof or not np.all(np.isfinite(action)):
                    self.request_stop("bad_action", f"post-conversion action returned {action.tolist()}")
                    break
            self._publish(action)
            io_updates = getattr(self.policy, "last_io_updates", None)
            if io_updates:
                try:
                    self._send_io_update(io_updates)
                except Exception:
                    break
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
                self._end_of_policy("max_steps", f"{k} steps")
                break

    def _end_of_policy(self, reason: str, detail: str) -> None:
        """No further target: keep streaming until the last segment has reached its target, plus one
        policy period of hold, then stop. During the hold the camera must stay fresh (``camera_stale``); reaching
        the deadline ends it as ``max_seconds``; any other stop meanwhile wins."""
        with self._lock:
            seg_end = self._segment.t0 + self._segment.duration
        hold_end = max(seg_end, time.monotonic()) + 1.0 / self.fps
        tick = 1.0 / self.servo_hz
        while not self._stop.is_set():
            now = time.monotonic()
            # the deadline outranks the end of the policy: a hold that reaches it ends as max_seconds
            if now >= self._deadline:
                self.request_stop("max_seconds", f"{self.max_seconds:g} s")
                return
            if now >= hold_end:
                self.request_stop(reason, detail)
                return
            if self.policy.image_key and self._camera_stale(now):  # same freshness rule as a policy step
                return
            self._stop.wait(min(hold_end, self._deadline, now + tick) - now)

    def _finish(self) -> None:
        s = self.session
        if not self.result.reason:
            self.result.reason = "stopped"
        try:
            # a fresh observation: after an interrupt during torque-on the latest one predates the ack
            obs = s.wait_for_observation(1.0)
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
    p.add_argument(
        "--fps",
        type=int,
        help="policy steps per second (default: dataset fps for replay, train_config.json fps for a learned policy "
        "(required when it has none), 30 for sine)",
    )
    p.add_argument("--servo-hz", type=float, default=50.0, help="servo frame rate (default 50)")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, help="stop after this many policy steps")
    p.add_argument("--task", default="", help="task text for language-conditioned policies")
    p.add_argument("--device", help="torch device (default cuda if available, else cpu)")
    p.add_argument("--no-park", action="store_true", help="hold position at the end instead of parking")
    p.add_argument(
        "--action-units",
        choices=["auto", "rad", "so101_norm", "deg"],
        default="auto",
        help=(
            "units for policy actions: auto (read from info.json, default rad; strictly requires unit_mode on SO-ARM), "
            "rad, so101_norm ([-100, 100], gripper [0, 100]), deg"
        ),
    )
    p.add_argument(
        "--state-units",
        choices=["auto", "rad", "so101_norm", "deg"],
        default="auto",
        help=(
            "units for policy observations: auto (read from info.json, default rad; strictly requires unit_mode on "
            "SO-ARM), rad, so101_norm, deg"
        ),
    )
    p.add_argument(
        "--lerobot-calib",
        help="path to LeRobot SO-100/SO-101 calibration JSON to align motor tick bounds for so101_norm",
    )
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

    saved: dict[int, Any] = {}

    def _handle(sig: int, handler: Any) -> None:
        try:
            old = signal.signal(sig, handler)
        except ValueError:  # not the main thread
            return
        saved.setdefault(sig, old)

    _handle(signal.SIGTERM, _term)
    try:
        return _main(args, _handle)
    finally:
        for sig, old in saved.items():
            signal.signal(sig, old)


def _main(args: argparse.Namespace, handle: Any) -> int:
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
            action_units=args.action_units,
            state_units=args.state_units,
            lerobot_calib=args.lerobot_calib,
        )
        print(
            f"policy {policy.name}"
            + (f" from {policy.source}" if policy.source else "")
            + (f", camera {policy.image_key} {policy.image_size}" if policy.image_key else "")
        )
        # from here a second Ctrl-C / SIGTERM during the stop and park is ignored instead of aborting it
        handle(signal.SIGINT, runner.handle_signal)
        handle(signal.SIGTERM, runner.handle_signal)
        result = runner.run()
    except (RunnerError, ValueError, ImportError, RebotError, OSError) as exc:
        print(f"rebot-policy: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # before the stream started (loading the policy, connecting)
        print("rebot-policy: interrupted", file=sys.stderr)
        return 130
    _print_summary(result)
    if result.reason == "interrupted":
        return 130
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
