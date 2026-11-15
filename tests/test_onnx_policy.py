"""Tests for OnnxPolicy normalization validation and PolicyRunner last_io_updates delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rebot_companion.onnx_policy import (
    build_linear_onnx_bytes,
    scaffold_onnx_policy,
    validate_image_spec,
    validate_normalization_stats,
    validate_policy_dir,
)
from rebot_companion.policy_runner import OnnxPolicy, Policy, PolicyRunner


def test_validate_normalization_stats_paired():
    # Only state_mean provided -> raises ValueError
    with pytest.raises(ValueError, match="Both state_mean and state_std must be provided together"):
        validate_normalization_stats({"state_mean": [0.0] * 7}, state_dim=7, action_dim=7)

    # Only state_std provided -> raises ValueError
    with pytest.raises(ValueError, match="Both state_mean and state_std must be provided together"):
        validate_normalization_stats({"state_std": [1.0] * 7}, state_dim=7, action_dim=7)

    # Only action_mean provided -> raises ValueError
    with pytest.raises(ValueError, match="Both action_mean and action_std must be provided together"):
        validate_normalization_stats({"action_mean": [0.0] * 7}, state_dim=7, action_dim=7)

    # Only action_std provided -> raises ValueError
    with pytest.raises(ValueError, match="Both action_mean and action_std must be provided together"):
        validate_normalization_stats({"action_std": [1.0] * 7}, state_dim=7, action_dim=7)


def test_validate_normalization_stats_lengths():
    # state_mean wrong length
    with pytest.raises(ValueError, match="state_mean length .* does not match state_dim"):
        validate_normalization_stats(
            {"state_mean": [0.0] * 6, "state_std": [1.0] * 7},
            state_dim=7,
            action_dim=7,
        )

    # action_std wrong length
    with pytest.raises(ValueError, match="action_std length .* does not match action_dim"):
        validate_normalization_stats(
            {"action_mean": [0.0] * 7, "action_std": [1.0] * 8},
            state_dim=7,
            action_dim=7,
        )


def test_validate_normalization_stats_finite_and_positive():
    # NaN in state_mean
    with pytest.raises(ValueError, match="All state_mean values must be finite"):
        validate_normalization_stats(
            {"state_mean": [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0], "state_std": [1.0] * 7},
            state_dim=7,
            action_dim=7,
        )

    # Inf in action_mean
    with pytest.raises(ValueError, match="All action_mean values must be finite"):
        validate_normalization_stats(
            {"action_mean": [float("inf")] * 7, "action_std": [1.0] * 7},
            state_dim=7,
            action_dim=7,
        )

    # Zero in state_std
    with pytest.raises(ValueError, match="strictly positive"):
        validate_normalization_stats(
            {"state_mean": [0.0] * 7, "state_std": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]},
            state_dim=7,
            action_dim=7,
        )

    # Negative in action_std
    with pytest.raises(ValueError, match="strictly positive"):
        validate_normalization_stats(
            {"action_mean": [0.0] * 7, "action_std": [1.0, 1.0, 1.0, 1.0, -0.5, 1.0, 1.0]},
            state_dim=7,
            action_dim=7,
        )


def test_onnx_policy_normalization_applied_and_shape_mismatch(tmp_path: Path):
    pol_dir = scaffold_onnx_policy(tmp_path / "norm_pol", policy_id="norm_pol", action_dim=7)

    # Update policy.json to include normalization
    manifest_dict = json.loads((pol_dir / "policy.json").read_text(encoding="utf-8"))
    state_mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    state_std = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    action_mean = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    action_std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    manifest_dict["normalization"] = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    (pol_dir / "policy.json").write_text(json.dumps(manifest_dict), encoding="utf-8")

    # Set model to identity: output = input @ I + 0
    w = np.eye(7, dtype=np.float32)
    b = np.zeros(7, dtype=np.float32)
    (pol_dir / "model.onnx").write_bytes(build_linear_onnx_bytes("observation.state", "action", w, b))

    pol = OnnxPolicy(pol_dir)

    raw_x = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    # Normalized state: (raw_x - state_mean) / state_std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Linear output: [1.0, ...]
    # Unnormalized action: norm_action * action_std + action_mean = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    out = pol.select({"observation.state": raw_x}, step=0, t=0.0)
    assert out is not None
    expected = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    assert np.allclose(out, expected, atol=1e-5)

    # Observation state shape mismatch raises ValueError instead of silently skipping
    with pytest.raises(ValueError, match="does not match normalization state_mean shape"):
        pol.select({"observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}, step=1, t=0.03)


def test_manifest_validation_rejects_invalid_normalization(tmp_path: Path):
    pol_dir = scaffold_onnx_policy(tmp_path / "bad_norm", policy_id="bad_norm", action_dim=7)
    manifest_dict = json.loads((pol_dir / "policy.json").read_text(encoding="utf-8"))
    manifest_dict["normalization"] = {
        "state_mean": [0.0] * 7,
        "state_std": [0.0] * 7,  # not strictly > 0
    }
    (pol_dir / "policy.json").write_text(json.dumps(manifest_dict), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly positive"):
        validate_policy_dir(pol_dir)

    with pytest.raises(ValueError, match="strictly positive"):
        OnnxPolicy(pol_dir)


class DummyPolicyWithIO(Policy):
    fps = 10
    image_key = None

    def __init__(self, io_sequence: list[dict[str, Any]]):
        self.io_sequence = io_sequence
        self.last_io_updates: dict[str, Any] = {}

    def select(self, obs: dict[str, np.ndarray], step: int, t: float) -> np.ndarray | None:
        if step >= len(self.io_sequence):
            return None
        self.last_io_updates = self.io_sequence[step]
        return np.zeros(7, dtype=np.float64)


class MockArmdClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", mode: str = "sim", status_code: int = 200):
        self.base_url = base_url
        self.mode = mode
        self.status_code = status_code
        self.recorded_requests: list[tuple[str, str, Any]] = []

    def request(self, method: str, path: str, json: Any = None, **kwargs: Any) -> Any:
        self.recorded_requests.append((method, path, json))
        status_code = self.status_code
        resp_data = {"mode": self.mode} if path == "/api/v1/sandbox/state" else {"status": "ok"}

        class MockResp:
            def __init__(self, sc: int, data: dict):
                self.status_code = sc
                self.data = data

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self.data

        return MockResp(status_code, resp_data)


def test_policy_runner_delivers_and_deduplicates_io_updates():
    mock_client = MockArmdClient()
    delivered_to_sender: list[dict[str, Any]] = []

    def test_sender(updates: dict[str, Any]):
        delivered_to_sender.append(dict(updates))

    policy = DummyPolicyWithIO(
        [
            {"vacuum": True},  # step 0: send
            {"vacuum": True},  # step 1: duplicate, should be skipped
            {"vacuum": False, "gripper_force": 0.5},  # step 2: changes, should send both/changes
            {"vacuum": False, "gripper_force": 0.5},  # step 3: duplicate, should be skipped
        ]
    )

    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
        io_sender=test_sender,
    )

    # Call _send_io_update through each step
    runner._send_io_update({"vacuum": True})
    assert len(delivered_to_sender) == 1
    assert delivered_to_sender[-1] == {"vacuum": True}

    # Consecutive duplicate update is deduplicated
    runner._send_io_update({"vacuum": True})
    assert len(delivered_to_sender) == 1

    # Changed updates sent
    runner._send_io_update({"vacuum": False, "gripper_force": 0.5})
    assert len(delivered_to_sender) == 2
    assert delivered_to_sender[-1] == {"vacuum": False, "gripper_force": 0.5}

    # Another duplicate update is deduplicated
    runner._send_io_update({"vacuum": False, "gripper_force": 0.5})
    assert len(delivered_to_sender) == 2


def test_policy_runner_io_delivery_via_client_request():
    mock_client = MockArmdClient(mode="sim")
    policy = DummyPolicyWithIO([{"vacuum": True}])
    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
    )

    runner._send_io_update({"vacuum": True})
    assert any(
        method == "POST" and path == "/api/v1/sandbox/io" and body == {"updates": {"vacuum": True}}
        for method, path, body in mock_client.recorded_requests
    )
    assert runner._sandbox_io_enabled is True


def test_policy_runner_sandbox_io_disabled_on_mode_none():
    mock_client = MockArmdClient(mode="none")
    policy = DummyPolicyWithIO([{"vacuum": True}])
    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
    )

    runner._send_io_update({"vacuum": True})
    assert runner._sandbox_io_enabled is False
    assert not any(path == "/api/v1/sandbox/io" for _, path, _ in mock_client.recorded_requests)
    assert runner.result.reason != "io_delivery_failed"


def test_policy_runner_sandbox_io_disabled_on_404_or_error():
    mock_client = MockArmdClient(status_code=404)
    policy = DummyPolicyWithIO([{"vacuum": True}])
    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
    )

    runner._send_io_update({"vacuum": True})
    assert runner._sandbox_io_enabled is False
    assert not any(path == "/api/v1/sandbox/io" for _, path, _ in mock_client.recorded_requests)
    assert runner.result.reason != "io_delivery_failed"


def test_policy_runner_resets_boolean_io_on_stop():
    mock_client = MockArmdClient(mode="sim")
    delivered_to_sender: list[dict[str, Any]] = []

    policy = DummyPolicyWithIO([{"vacuum": True, "gripper_force": 0.8}])
    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
        io_sender=lambda u: delivered_to_sender.append(dict(u)),
    )

    runner._send_io_update({"vacuum": True, "gripper_force": 0.8})
    assert delivered_to_sender[-1] == {"vacuum": True, "gripper_force": 0.8}

    # Stop cleanup resets boolean IO channels to False
    runner._reset_io_channels()
    assert delivered_to_sender[-1] == {"vacuum": False}
    assert "gripper_force" not in delivered_to_sender[-1]


def test_policy_runner_io_delivery_failure_stops_cleanly():
    mock_client = MockArmdClient()

    def failing_sender(updates: dict[str, Any]):
        raise ConnectionResetError("Connection refused by armd")

    policy = DummyPolicyWithIO([{"vacuum": True}])
    runner = PolicyRunner(
        client=mock_client,
        policy=policy,
        fps=10,
        servo_hz=20.0,
        io_sender=failing_sender,
    )

    with pytest.raises(ConnectionResetError):
        runner._send_io_update({"vacuum": True})

    assert runner.result.reason == "io_delivery_failed"


def test_onnx_policy_nhwc_image_preprocessing(tmp_path: Path):
    pol_dir = scaffold_onnx_policy(
        tmp_path / "nhwc_pol",
        policy_id="nhwc_pol",
        action_dim=7,
        include_image=True,
        image_layout="NHWC",
    )

    manifest_dict = json.loads((pol_dir / "policy.json").read_text(encoding="utf-8"))
    manifest_dict["inputs"]["image"]["scale"] = 1.0 / 255.0
    manifest_dict["inputs"]["image"]["mean"] = [0.5, 0.5, 0.5]
    manifest_dict["inputs"]["image"]["std"] = [0.5, 0.5, 0.5]
    (pol_dir / "policy.json").write_text(json.dumps(manifest_dict), encoding="utf-8")

    pol = OnnxPolicy(pol_dir)
    assert pol._image_layout == "NHWC"
    assert pol.image_size == (224, 224)
    assert pol._image_channels == 3
    assert pol._image_mean is not None and pol._image_mean.shape == (1, 1, 1, 3)
    assert pol._image_std is not None and pol._image_std.shape == (1, 1, 1, 3)

    raw_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    img_hwc = np.full((224, 224, 3), 128, dtype=np.uint8)
    out = pol.select({"observation.state": raw_x, "observation.images.wrist": img_hwc}, step=0, t=0.0)
    assert out is not None
    assert np.allclose(out, raw_x, atol=1e-5)


def test_manifest_validation_rejects_invalid_image_spec():
    # Invalid layout
    with pytest.raises(ValueError, match="layout must be 'NCHW' or 'NHWC'"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "layout": "INVALID"})

    # Invalid scale: non-positive or non-finite
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "scale": 0.0})
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "scale": -1.0})
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "scale": float("nan")})

    # Unpaired mean/std
    with pytest.raises(ValueError, match="Both image mean and std must be provided together"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "mean": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="Both image mean and std must be provided together"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "std": [1.0, 1.0, 1.0]})

    # std <= 0 rejected
    with pytest.raises(ValueError, match="strictly positive"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "mean": [0.0, 0.0, 0.0], "std": [1.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        validate_image_spec(
            {"name": "img", "shape": [1, 3, 224, 224], "mean": [0.0, 0.0, 0.0], "std": [1.0, -0.5, 1.0]}
        )

    # mean/std length mismatch with channels
    with pytest.raises(ValueError, match="match channel count"):
        validate_image_spec({"name": "img", "shape": [1, 3, 224, 224], "mean": [0.0, 0.0], "std": [1.0, 1.0]})

    # non-finite mean/std
    with pytest.raises(ValueError, match="finite"):
        validate_image_spec(
            {"name": "img", "shape": [1, 3, 224, 224], "mean": [0.0, float("nan"), 0.0], "std": [1.0, 1.0, 1.0]}
        )
