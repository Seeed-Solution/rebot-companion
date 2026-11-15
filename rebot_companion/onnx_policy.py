"""Unified ONNX Policy runtime, validator and scaffolder for reBot companion.

Contract specification: ``policy.json`` + ``model.onnx``.
Usable in both Python (onnxruntime) and browser / web cockpit (onnxruntime-web).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ONNX_POLICY_FORMAT = "rebot-onnx-policy/v1"
SUPPORTED_ROBOTS = ("b601dm", "so-arm101")
SUPPORTED_UNIT_MODES = ("rad", "so101_norm")
SUPPORTED_ACTION_DIMS = (6, 7)


def validate_image_spec(image_spec: dict[str, Any]) -> None:
    """Validate image input specification according to the rebot-onnx-policy contract.

    - Must declare a non-empty name string.
    - Shape must be a 4-element list with positive integers.
    - Layout must be 'NCHW' or 'NHWC'.
    - Scale, if provided, must be a finite positive number (> 0).
    - Mean and std, if provided, must be paired, have length equal to channel count (or 1),
      be finite numbers, and all std values must be strictly > 0.
    """
    if not isinstance(image_spec, dict):
        raise ValueError(f"inputs.image must be a dictionary, got {type(image_spec).__name__}")
    name = image_spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("inputs.image must have a non-empty 'name' string")

    shape = image_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError("inputs.image shape must be a 4-element list [batch, dim1, dim2, dim3]")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"inputs.image shape elements must be positive integers, got {shape!r}")

    layout_raw = image_spec.get("layout", "NCHW")
    if not isinstance(layout_raw, str) or layout_raw.upper() not in ("NCHW", "NHWC"):
        raise ValueError(f"inputs.image layout must be 'NCHW' or 'NHWC', got {layout_raw!r}")
    layout = layout_raw.upper()

    channels = shape[1] if layout == "NCHW" else shape[3]

    if "scale" in image_spec:
        scale = image_spec["scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"inputs.image scale must be a finite positive number (> 0), got {scale!r}")

    has_mean = "mean" in image_spec and image_spec["mean"] is not None
    has_std = "std" in image_spec and image_spec["std"] is not None
    if has_mean != has_std:
        raise ValueError("Both image mean and std must be provided together")

    if has_mean and has_std:
        mean = image_spec["mean"]
        std = image_spec["std"]
        if not isinstance(mean, (list, tuple, np.ndarray)) or not isinstance(std, (list, tuple, np.ndarray)):
            raise ValueError("inputs.image mean and std must be sequences")
        arr_mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        arr_std = np.asarray(std, dtype=np.float32).reshape(-1)
        if arr_mean.shape[0] not in (1, channels):
            raise ValueError(
                f"inputs.image mean length ({arr_mean.shape[0]}) must match channel count ({channels}) or 1"
            )
        if arr_std.shape[0] not in (1, channels):
            raise ValueError(f"inputs.image std length ({arr_std.shape[0]}) must match channel count ({channels}) or 1")
        if arr_mean.shape[0] != arr_std.shape[0]:
            raise ValueError(
                f"inputs.image mean and std lengths must match ({arr_mean.shape[0]} != {arr_std.shape[0]})"
            )
        if not np.all(np.isfinite(arr_mean)):
            raise ValueError("All inputs.image mean values must be finite numbers")
        if not np.all(np.isfinite(arr_std)):
            raise ValueError("All inputs.image std values must be finite numbers")
        if np.any(arr_std <= 0.0):
            raise ValueError(f"All inputs.image std values must be strictly positive (> 0), got {std!r}")


def validate_normalization_stats(
    normalization: dict[str, Any],
    state_dim: int,
    action_dim: int,
) -> None:
    """Validate normalization dictionary according to the rebot-onnx-policy contract.

    - Both mean and std must be provided together (not half-specified).
    - Lengths must match state_dim / action_dim.
    - All values must be finite.
    - All std values must be strictly > 0.
    """
    has_state_mean = "state_mean" in normalization and normalization["state_mean"] is not None
    has_state_std = "state_std" in normalization and normalization["state_std"] is not None
    if has_state_mean != has_state_std:
        raise ValueError("Both state_mean and state_std must be provided together in normalization")
    if has_state_mean:
        sm = normalization["state_mean"]
        ss = normalization["state_std"]
        if not isinstance(sm, (list, tuple, np.ndarray)) or not isinstance(ss, (list, tuple, np.ndarray)):
            raise ValueError("state_mean and state_std must be sequences")
        arr_sm = np.asarray(sm, dtype=np.float32).reshape(-1)
        arr_ss = np.asarray(ss, dtype=np.float32).reshape(-1)
        if arr_sm.shape[0] != state_dim:
            raise ValueError(f"state_mean length ({arr_sm.shape[0]}) does not match state_dim ({state_dim})")
        if arr_ss.shape[0] != state_dim:
            raise ValueError(f"state_std length ({arr_ss.shape[0]}) does not match state_dim ({state_dim})")
        if not np.all(np.isfinite(arr_sm)):
            raise ValueError("All state_mean values must be finite")
        if not np.all(np.isfinite(arr_ss)):
            raise ValueError("All state_std values must be finite")
        if np.any(arr_ss <= 0.0):
            raise ValueError("All state_std values must be strictly positive (> 0)")

    has_action_mean = "action_mean" in normalization and normalization["action_mean"] is not None
    has_action_std = "action_std" in normalization and normalization["action_std"] is not None
    if has_action_mean != has_action_std:
        raise ValueError("Both action_mean and action_std must be provided together in normalization")
    if has_action_mean:
        am = normalization["action_mean"]
        astd = normalization["action_std"]
        if not isinstance(am, (list, tuple, np.ndarray)) or not isinstance(astd, (list, tuple, np.ndarray)):
            raise ValueError("action_mean and action_std must be sequences")
        arr_am = np.asarray(am, dtype=np.float32).reshape(-1)
        arr_astd = np.asarray(astd, dtype=np.float32).reshape(-1)
        if arr_am.shape[0] != action_dim:
            raise ValueError(f"action_mean length ({arr_am.shape[0]}) does not match action_dim ({action_dim})")
        if arr_astd.shape[0] != action_dim:
            raise ValueError(f"action_std length ({arr_astd.shape[0]}) does not match action_dim ({action_dim})")
        if not np.all(np.isfinite(arr_am)):
            raise ValueError("All action_mean values must be finite")
        if not np.all(np.isfinite(arr_astd)):
            raise ValueError("All action_std values must be finite")
        if np.any(arr_astd <= 0.0):
            raise ValueError("All action_std values must be strictly positive (> 0)")


@dataclass
class OnnxPolicyManifest:
    """Metadata manifest matching the rebot-onnx-policy/v1 contract."""

    format: str = ONNX_POLICY_FORMAT
    name: str = ""
    robot: str = "b601dm"
    fps: int = 30
    action_dim: int = 7
    unit_mode: str = "rad"
    model_file: str = "model.onnx"
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    normalization: dict[str, list[float]] = field(default_factory=dict)
    description: str = ""
    action_space: str = "joint_positions"
    image_y_up: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a plain JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnnxPolicyManifest:
        """Create and validate an OnnxPolicyManifest from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Manifest data must be a dictionary, got {type(data).__name__}")

        fmt = data.get("format")
        if fmt != ONNX_POLICY_FORMAT:
            raise ValueError(f"Invalid format {fmt!r}, expected {ONNX_POLICY_FORMAT!r}")

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"name must be a non-empty string, got {name!r}")

        robot = data.get("robot")
        if robot not in SUPPORTED_ROBOTS:
            raise ValueError(f"robot must be one of {SUPPORTED_ROBOTS}, got {robot!r}")

        fps = data.get("fps", 30)
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {fps!r}")

        action_dim = data.get("action_dim", 7)
        if action_dim not in SUPPORTED_ACTION_DIMS:
            raise ValueError(f"action_dim must be one of {SUPPORTED_ACTION_DIMS}, got {action_dim!r}")

        unit_mode = data.get("unit_mode", "rad")
        if unit_mode not in SUPPORTED_UNIT_MODES:
            raise ValueError(f"unit_mode must be one of {SUPPORTED_UNIT_MODES}, got {unit_mode!r}")

        model_file = data.get("model_file", "model.onnx")
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValueError(f"model_file must be a non-empty string, got {model_file!r}")
        model_path = Path(model_file)
        if model_path.is_absolute() or any(part == ".." for part in model_path.parts):
            raise ValueError(f"model_file must be a safe relative path, got {model_file!r}")

        inputs = data.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dictionary")
        if "state" not in inputs or not isinstance(inputs["state"], dict):
            raise ValueError("inputs must declare a 'state' dictionary")
        state_in = inputs["state"]
        if "name" not in state_in or not isinstance(state_in["name"], str) or not state_in["name"].strip():
            raise ValueError("inputs.state must have a non-empty 'name' string")
        if "shape" not in state_in or not isinstance(state_in["shape"], (list, tuple)):
            raise ValueError("inputs.state must have a 'shape' list")

        if "image" in inputs:
            validate_image_spec(inputs["image"])

        outputs = data.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("outputs must be a dictionary")
        if "action" not in outputs or not isinstance(outputs["action"], dict):
            raise ValueError("outputs must declare an 'action' dictionary")
        act_out = outputs["action"]
        if "name" not in act_out or not isinstance(act_out["name"], str) or not act_out["name"].strip():
            raise ValueError("outputs.action must have a non-empty 'name' string")
        if "shape" not in act_out or not isinstance(act_out["shape"], (list, tuple)):
            raise ValueError("outputs.action must have a 'shape' list")

        if "io" in outputs:
            io_out = outputs["io"]
            if not isinstance(io_out, dict):
                raise ValueError("outputs.io must be a dictionary")
            if "channels" in io_out and not isinstance(io_out["channels"], (list, tuple)):
                raise ValueError("outputs.io channels must be a list")

        normalization = data.get("normalization", {})
        if not isinstance(normalization, dict):
            raise ValueError("normalization must be a dictionary")

        state_shape = state_in.get("shape", [])
        state_dim = int(state_shape[-1]) if state_shape else action_dim
        validate_normalization_stats(normalization, state_dim=state_dim, action_dim=action_dim)

        description = str(data.get("description", ""))
        action_space = str(data.get("action_space", "joint_positions"))
        image_y_up = bool(data.get("image_y_up", False))

        return cls(
            format=fmt,
            name=name,
            robot=robot,
            fps=fps,
            action_dim=action_dim,
            unit_mode=unit_mode,
            model_file=model_file,
            inputs=inputs,
            outputs=outputs,
            normalization=normalization,
            description=description,
            action_space=action_space,
            image_y_up=image_y_up,
        )

    def save(self, path: str | Path) -> Path:
        """Save manifest to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def load_onnx_manifest(path: str | Path) -> OnnxPolicyManifest:
    """Load and validate an OnnxPolicyManifest from a policy directory or JSON file.

    Rejects directory traversal in ``model_file`` and verifies file existence when
    given a directory.
    """
    p = Path(path).resolve()
    if p.is_dir():
        manifest_file = p / "policy.json"
        policy_dir = p
    elif p.is_file():
        manifest_file = p
        policy_dir = p.parent
    else:
        raise FileNotFoundError(f"Path does not exist: {p}")

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Policy manifest not found: {manifest_file}")

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read {manifest_file}: {exc}") from exc

    manifest = OnnxPolicyManifest.from_dict(data)

    # Verify model_file does not escape policy_dir
    model_path = (policy_dir / manifest.model_file).resolve()
    try:
        model_path.relative_to(policy_dir)
    except ValueError:
        raise ValueError(f"model_file {manifest.model_file!r} resolves outside policy dir {policy_dir}") from None

    if p.is_dir() and not model_path.is_file():
        raise FileNotFoundError(f"Model file declared in manifest not found: {model_path}")

    return manifest


def validate_policy_dir(path: str | Path) -> OnnxPolicyManifest:
    """Validate that a directory contains a valid ONNX policy package.

    Loads manifest, verifies ``model_file`` exists and is non-empty, and if
    ``onnxruntime`` is installed, checks that declared inputs and outputs exist
    in the ONNX model.
    """
    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    manifest = load_onnx_manifest(dir_path)
    model_path = dir_path / manifest.model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(f"Model file is empty: {model_path}")

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None

    if ort is not None:
        try:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ValueError(f"onnxruntime failed to load {model_path}: {exc}") from exc

        sess_in_names = {inp.name for inp in session.get_inputs()}
        sess_out_names = {out.name for out in session.get_outputs()}

        state_name = manifest.inputs["state"]["name"]
        if state_name not in sess_in_names:
            raise ValueError(f"Model inputs {sess_in_names} do not contain declared state input {state_name!r}")

        if "image" in manifest.inputs:
            img_name = manifest.inputs["image"]["name"]
            if img_name not in sess_in_names:
                raise ValueError(f"Model inputs {sess_in_names} do not contain declared image input {img_name!r}")

        action_name = manifest.outputs["action"]["name"]
        if action_name not in sess_out_names:
            raise ValueError(f"Model outputs {sess_out_names} do not contain declared action output {action_name!r}")

        if "io" in manifest.outputs:
            io_name = manifest.outputs["io"].get("name")
            if io_name and io_name not in sess_out_names:
                raise ValueError(f"Model outputs {sess_out_names} do not contain declared io output {io_name!r}")

    return manifest


# ---------------------------------------------------------------------------
# Pure-Python Minimal ONNX Protobuf Wire-Format Serializer
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a standard protobuf varint (64-bit safe)."""
    bits = value & 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        b = bits & 0x7F
        bits >>= 7
        if bits:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_int(field_number: int, val: int) -> bytes:
    return _encode_tag(field_number, 0) + _encode_varint(val)


def _encode_string(field_number: int, s: str) -> bytes:
    b = s.encode("utf-8")
    return _encode_tag(field_number, 2) + _encode_varint(len(b)) + b


def _encode_bytes(field_number: int, b: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(b)) + b


def _encode_message(field_number: int, msg: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(msg)) + msg


def _make_dim(d: int | str) -> bytes:
    # Dimension: 1=dim_value (int64), 2=dim_param (string)
    if isinstance(d, int):
        return _encode_int(1, d)
    return _encode_string(2, str(d))


def _make_shape(dims: Sequence[int | str]) -> bytes:
    # TensorShapeProto: repeated Dimension dim = 1
    return b"".join(_encode_message(1, _make_dim(d)) for d in dims)


def _make_tensor_type(elem_type: int, dims: Sequence[int | str]) -> bytes:
    # TypeProto.Tensor: elem_type=1, shape=2
    t = _encode_int(1, elem_type) + _encode_message(2, _make_shape(dims))
    # TypeProto: tensor_type = 1
    return _encode_message(1, t)


def _make_value_info(name: str, elem_type: int, dims: Sequence[int | str]) -> bytes:
    # ValueInfoProto: name=1, type=2
    return _encode_string(1, name) + _encode_message(2, _make_tensor_type(elem_type, dims))


def _make_tensor(name: str, array: np.ndarray) -> bytes:
    # TensorProto: dims=1, data_type=2 (FLOAT=1), name=8, raw_data=9
    arr_f32 = np.ascontiguousarray(array, dtype=np.float32)
    b_dims = b"".join(_encode_int(1, int(d)) for d in arr_f32.shape)
    b_dtype = _encode_int(2, 1)  # FLOAT
    b_name = _encode_string(8, name)
    b_raw = _encode_bytes(9, arr_f32.tobytes())
    return b_dims + b_dtype + b_name + b_raw


def _make_node(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str = "") -> bytes:
    # NodeProto: input=1, output=2, name=3, op_type=4
    parts = []
    for inp in inputs:
        parts.append(_encode_string(1, inp))
    for out in outputs:
        parts.append(_encode_string(2, out))
    if name:
        parts.append(_encode_string(3, name))
    parts.append(_encode_string(4, op_type))
    return b"".join(parts)


def build_linear_onnx_bytes(
    input_name: str,
    output_name: str,
    weight: np.ndarray,
    bias: np.ndarray,
    ir_version: int = 8,
    opset_version: int = 13,
) -> bytes:
    """Build a minimal valid ONNX ModelProto bytes representing ``output = input @ weight + bias``.

    Uses ONLY Python standard library + numpy with zero dependency on the onnx or torch
    packages. Produces a ModelProto compatible with both onnxruntime and onnxruntime-web.

    Args:
        input_name: Name of the input tensor (e.g. "observation.state").
        output_name: Name of the output tensor (e.g. "action").
        weight: 2D numpy array of shape (in_dim, out_dim).
        bias: 1D numpy array of shape (out_dim,).
        ir_version: ONNX IR version (default 8).
        opset_version: ONNX operator set version (default 13).
    """
    w = np.asarray(weight, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be a 2D array, got shape {w.shape}")
    in_dim, out_dim = w.shape
    if b.ndim == 2 and b.shape[0] == 1:
        b = b.squeeze(0)
    if b.ndim != 1 or b.shape[0] != out_dim:
        raise ValueError(f"bias shape must match out_dim {out_dim}, got {b.shape}")

    t_w = _make_tensor("W_weight", w)
    t_b = _make_tensor("B_bias", b)

    node_matmul = _make_node("MatMul", [input_name, "W_weight"], ["linear_mm_out"], "node_matmul")
    node_add = _make_node("Add", ["linear_mm_out", "B_bias"], [output_name], "node_add")

    vi_in = _make_value_info(input_name, 1, [1, in_dim])
    vi_out = _make_value_info(output_name, 1, [1, out_dim])

    # GraphProto: node=1, name=2, initializer=5, input=11, output=12
    graph_parts = [
        _encode_message(1, node_matmul),
        _encode_message(1, node_add),
        _encode_string(2, "linear_policy_graph"),
        _encode_message(5, t_w),
        _encode_message(5, t_b),
        _encode_message(11, vi_in),
        _encode_message(12, vi_out),
    ]
    graph = b"".join(graph_parts)

    # OperatorSetIdProto: domain=1, version=2
    opset = _encode_string(1, "") + _encode_int(2, opset_version)

    # ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8
    model = (
        _encode_int(1, ir_version)
        + _encode_string(2, "rebot-companion")
        + _encode_message(7, graph)
        + _encode_message(8, opset)
    )
    return model


def _build_multimodal_linear_onnx_bytes(
    state_input_name: str,
    image_input_name: str,
    output_name: str,
    action_dim: int,
    image_shape: tuple[int, int, int] = (3, 224, 224),
    image_layout: str = "NCHW",
    ir_version: int = 8,
    opset_version: int = 13,
) -> bytes:
    """Build a minimal valid multimodal ONNX ModelProto bytes with state and wrist image.

    Graph (NCHW):
        state_mm = MatMul(state, I)
        gap_out = GlobalAveragePool(image)
        flat_out = Flatten(gap_out)
        img_mm = MatMul(flat_out, zeros)
        state_b = Add(state_mm, zeros)
        action = Add(state_b, img_mm)

    Graph (NHWC):
        state_mm = MatMul(state, I)
        flat_out = Flatten(image)
        img_mm = MatMul(flat_out, zeros)
        state_b = Add(state_mm, zeros)
        action = Add(state_b, img_mm)
    """
    w_state = np.eye(action_dim, dtype=np.float32)
    b_state = np.zeros(action_dim, dtype=np.float32)

    is_nhwc = image_layout.upper() == "NHWC"
    if is_nhwc:
        in_img_features = int(np.prod(image_shape))
        w_img = np.zeros((in_img_features, action_dim), dtype=np.float32)
    else:
        w_img = np.zeros((image_shape[0], action_dim), dtype=np.float32)

    t_w = _make_tensor("W_state", w_state)
    t_b = _make_tensor("B_state", b_state)
    t_w_img = _make_tensor("W_img", w_img)

    n_state_mm = _make_node("MatMul", [state_input_name, "W_state"], ["state_mm"], "n_state_mm")
    if is_nhwc:
        n_flat = _make_node("Flatten", [image_input_name], ["flat_out"], "n_flat")
        img_nodes = [n_flat]
    else:
        n_gap = _make_node("GlobalAveragePool", [image_input_name], ["gap_out"], "n_gap")
        n_flat = _make_node("Flatten", ["gap_out"], ["flat_out"], "n_flat")
        img_nodes = [n_gap, n_flat]
    n_img_mm = _make_node("MatMul", ["flat_out", "W_img"], ["img_mm"], "n_img_mm")
    n_state_b = _make_node("Add", ["state_mm", "B_state"], ["state_b"], "n_state_b")
    n_out = _make_node("Add", ["state_b", "img_mm"], [output_name], "n_out")

    vi_state = _make_value_info(state_input_name, 1, [1, action_dim])
    vi_image = _make_value_info(image_input_name, 1, [1, *image_shape])
    vi_out = _make_value_info(output_name, 1, [1, action_dim])

    graph_parts = [
        _encode_message(1, n_state_mm),
        *[_encode_message(1, node) for node in img_nodes],
        _encode_message(1, n_img_mm),
        _encode_message(1, n_state_b),
        _encode_message(1, n_out),
        _encode_string(2, "multimodal_policy_graph"),
        _encode_message(5, t_w),
        _encode_message(5, t_b),
        _encode_message(5, t_w_img),
        _encode_message(11, vi_state),
        _encode_message(11, vi_image),
        _encode_message(12, vi_out),
    ]
    graph = b"".join(graph_parts)
    opset = _encode_string(1, "") + _encode_int(2, opset_version)

    model = (
        _encode_int(1, ir_version)
        + _encode_string(2, "rebot-companion")
        + _encode_message(7, graph)
        + _encode_message(8, opset)
    )
    return model


def scaffold_onnx_policy(
    target_dir: str | Path,
    policy_id: str,
    action_dim: int = 7,
    robot: str = "b601dm",
    include_image: bool = False,
    overwrite: bool = False,
    image_layout: str = "NCHW",
) -> Path:
    """Scaffold a new valid ONNX policy package directory with ``policy.json`` and ``model.onnx``.

    Args:
        target_dir: Directory where the policy package will be created.
        policy_id: Unique identifier / name for the policy.
        action_dim: Number of action degrees of freedom (6 or 7).
        robot: Target robot ("b601dm" or "so-arm101").
        include_image: Whether to wire a wrist camera image input into the manifest and model.
        overwrite: Whether to overwrite existing files if target_dir exists.
        image_layout: Tensor layout for the image input ("NCHW" or "NHWC").

    Returns:
        The Path to the created policy directory.
    """
    dest = Path(target_dir).resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Target directory {dest} already exists and overwrite is False")

    dest.mkdir(parents=True, exist_ok=True)

    state_input_name = "observation.state"
    image_input_name = "observation.images.wrist"
    action_output_name = "action"

    inputs: dict[str, dict[str, Any]] = {
        "state": {
            "name": state_input_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        }
    }
    if include_image:
        layout_upper = image_layout.upper()
        if layout_upper not in ("NCHW", "NHWC"):
            raise ValueError(f"image_layout must be 'NCHW' or 'NHWC', got {image_layout!r}")
        shape = [1, 3, 224, 224] if layout_upper == "NCHW" else [1, 224, 224, 3]
        inputs["image"] = {
            "name": image_input_name,
            "shape": shape,
            "dtype": "float32",
            "layout": layout_upper,
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }

    outputs: dict[str, dict[str, Any]] = {
        "action": {
            "name": action_output_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        }
    }

    manifest = OnnxPolicyManifest(
        format=ONNX_POLICY_FORMAT,
        name=policy_id,
        robot=robot,
        fps=30,
        action_dim=action_dim,
        unit_mode="rad",
        model_file="model.onnx",
        inputs=inputs,
        outputs=outputs,
        normalization={},
        description=f"Scaffolded ONNX policy {policy_id}",
    )
    manifest.save(dest / "policy.json")

    if include_image:
        onnx_bytes = _build_multimodal_linear_onnx_bytes(
            state_input_name=state_input_name,
            image_input_name=image_input_name,
            output_name=action_output_name,
            action_dim=action_dim,
            image_shape=(3, 224, 224) if layout_upper == "NCHW" else (224, 224, 3),
            image_layout=layout_upper,
        )
    else:
        w = np.eye(action_dim, dtype=np.float32)
        b = np.zeros(action_dim, dtype=np.float32)
        onnx_bytes = build_linear_onnx_bytes(state_input_name, action_output_name, w, b)

    (dest / "model.onnx").write_bytes(onnx_bytes)
    return dest


DEFAULT_CLOSED_LOOP_TARGET_Q = [0.08, -0.35, 0.55, 0.0, 1.15, 0.0, 0.0]
DEFAULT_CLOSED_LOOP_W_IMG = np.array(
    [
        [0.02, -0.02, 0.01, 0.0, 0.01, 0.0, 0.0],
        [-0.03, 0.03, 0.02, 0.0, -0.02, 0.0, 0.0],
        [0.01, -0.01, 0.0, 0.0, 0.01, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def build_closed_loop_multimodal_onnx_bytes(
    state_input_name: str = "observation.state",
    image_input_name: str = "observation.images.wrist",
    output_name: str = "action",
    action_dim: int = 7,
    target_q: Sequence[float] | None = None,
    state_gain: float = 0.88,
    image_layout: str = "NCHW",
    image_hw: tuple[int, int] = (64, 64),
    w_img: np.ndarray | None = None,
    ir_version: int = 8,
    opset_version: int = 13,
) -> bytes:
    """Build an ONNX ModelProto for closed-loop contraction + camera control.

    The model computes:
        action = state @ W_state + B_state + image_features @ W_img
    where W_state = state_gain * I, B_state = (1.0 - state_gain) * target_q.
    """
    if target_q is None:
        target_arr = np.array(DEFAULT_CLOSED_LOOP_TARGET_Q[:action_dim], dtype=np.float32)
    else:
        target_arr = np.asarray(target_q, dtype=np.float32).reshape(-1)
        if target_arr.shape[0] != action_dim:
            raise ValueError(f"target_q length ({target_arr.shape[0]}) must match action_dim ({action_dim})")

    w_state = (float(state_gain) * np.eye(action_dim)).astype(np.float32)
    b_state = ((1.0 - float(state_gain)) * target_arr).astype(np.float32)

    if w_img is None:
        base_w_img = DEFAULT_CLOSED_LOOP_W_IMG[:, :action_dim]
    else:
        base_w_img = np.asarray(w_img, dtype=np.float32)

    layout = image_layout.upper()
    h, w = int(image_hw[0]), int(image_hw[1])
    is_nhwc = layout == "NHWC"

    if is_nhwc:
        img_shape = (h, w, 3)
        w_img_tensor = np.tile(base_w_img / float(h * w), (h * w, 1)).astype(np.float32)
    else:
        img_shape = (3, h, w)
        w_img_tensor = base_w_img.astype(np.float32)

    t_w = _make_tensor("W_state", w_state)
    t_b = _make_tensor("B_state", b_state)
    t_w_img = _make_tensor("W_img", w_img_tensor)

    n_state_mm = _make_node("MatMul", [state_input_name, "W_state"], ["state_mm"], "n_state_mm")
    if is_nhwc:
        n_flat = _make_node("Flatten", [image_input_name], ["flat_out"], "n_flat")
        img_nodes = [n_flat]
    else:
        n_gap = _make_node("GlobalAveragePool", [image_input_name], ["gap_out"], "n_gap")
        n_flat = _make_node("Flatten", ["gap_out"], ["flat_out"], "n_flat")
        img_nodes = [n_gap, n_flat]
    n_img_mm = _make_node("MatMul", ["flat_out", "W_img"], ["img_mm"], "n_img_mm")
    n_state_b = _make_node("Add", ["state_mm", "B_state"], ["state_b"], "n_state_b")
    n_out = _make_node("Add", ["state_b", "img_mm"], [output_name], "n_out")

    vi_state = _make_value_info(state_input_name, 1, [1, action_dim])
    vi_image = _make_value_info(image_input_name, 1, [1, *img_shape])
    vi_out = _make_value_info(output_name, 1, [1, action_dim])

    graph_parts = [
        _encode_message(1, n_state_mm),
        *[_encode_message(1, node) for node in img_nodes],
        _encode_message(1, n_img_mm),
        _encode_message(1, n_state_b),
        _encode_message(1, n_out),
        _encode_string(2, "closed_loop_multimodal_policy_graph"),
        _encode_message(5, t_w),
        _encode_message(5, t_b),
        _encode_message(5, t_w_img),
        _encode_message(11, vi_state),
        _encode_message(11, vi_image),
        _encode_message(12, vi_out),
    ]
    graph = b"".join(graph_parts)
    opset = _encode_string(1, "") + _encode_int(2, opset_version)

    model = (
        _encode_int(1, ir_version)
        + _encode_string(2, "rebot-companion")
        + _encode_message(7, graph)
        + _encode_message(8, opset)
    )
    return model


def scaffold_closed_loop_demo_policy(
    target_dir: str | Path,
    policy_id: str = "pcb_visual_servo",
    action_dim: int = 7,
    robot: str = "b601dm",
    target_q: Sequence[float] | None = None,
    state_gain: float = 0.88,
    image_layout: str = "NCHW",
    image_hw: tuple[int, int] = (64, 64),
    overwrite: bool = False,
) -> Path:
    """Scaffold a Level-5 closed-loop multimodal policy with camera and proprioceptive state inputs."""
    dest = Path(target_dir).resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Target directory {dest} already exists and overwrite is False")

    dest.mkdir(parents=True, exist_ok=True)

    state_input_name = "observation.state"
    image_input_name = "observation.images.wrist"
    action_output_name = "action"

    layout_upper = image_layout.upper()
    if layout_upper not in ("NCHW", "NHWC"):
        raise ValueError(f"image_layout must be 'NCHW' or 'NHWC', got {image_layout!r}")
    h, w = int(image_hw[0]), int(image_hw[1])
    shape = [1, 3, h, w] if layout_upper == "NCHW" else [1, h, w, 3]

    inputs: dict[str, dict[str, Any]] = {
        "state": {
            "name": state_input_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        },
        "image": {
            "name": image_input_name,
            "shape": shape,
            "dtype": "float32",
            "layout": layout_upper,
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    }

    outputs: dict[str, dict[str, Any]] = {
        "action": {
            "name": action_output_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        }
    }

    manifest = OnnxPolicyManifest(
        format=ONNX_POLICY_FORMAT,
        name=policy_id,
        robot=robot,
        fps=30,
        action_dim=action_dim,
        unit_mode="rad",
        model_file="model.onnx",
        inputs=inputs,
        outputs=outputs,
        normalization={},
        description=f"Closed-loop Level 5 multimodal policy {policy_id}",
    )
    manifest.save(dest / "policy.json")

    onnx_bytes = build_closed_loop_multimodal_onnx_bytes(
        state_input_name=state_input_name,
        image_input_name=image_input_name,
        output_name=action_output_name,
        action_dim=action_dim,
        target_q=target_q,
        state_gain=state_gain,
        image_layout=layout_upper,
        image_hw=(h, w),
    )
    (dest / "model.onnx").write_bytes(onnx_bytes)
    return dest


def build_pcb_visual_servo_onnx_bytes(
    image_input_name: str = "observation.images.policy_cam",
    state_input_name: str = "observation.state",
    output_name: str = "action",
    action_dim: int = 7,
    cam_pos: tuple[float, float, float] = (0.23, -0.14, 0.34),
    fovy_deg: float = 45.0,
    plane_z: float = 0.011,
    image_hw: tuple[int, int] = (64, 64),
    image_y_up: bool = False,
    ir_version: int = 8,
    opset_version: int = 13,
) -> bytes:
    """Build an opset-13 ONNX ModelProto computing spatial visual-servo target from policy_cam.

    Computes:
      1) Green-excess per-pixel mask via 1x1 Conv (W=[[-0.6, 1.0, -0.6]], B=[-0.06]) -> Relu -> Flatten -> [1, 4096].
      2) Metric spatial coordinates from camera extrinsics:
         d = cam_z - plane_z = 0.34 - 0.011 = 0.329 m.
         S = d * tan(fovy / 2) = 0.13627638 m.
         dx = (col - 31.5) / 32.0 * S.
         dy = (31.5 - row) / 32.0 * S (when image_y_up is False).
         Centroid = (sum(M * dx) / (sum(M) + eps), sum(M * dy) / (sum(M) + eps)).
      3) Cartesian target [cam_x + dx, cam_y + dy, plane_z, 0, 0, 0, 1.0].
      4) Adds state @ 0 to link observation.state without altering action.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    n_pixels = h * w
    cam_x, cam_y, cam_z = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
    d = cam_z - float(plane_z)
    fovy_rad = math.radians(float(fovy_deg))
    s_half = d * math.tan(fovy_rad / 2.0)

    cols = np.tile(np.arange(w, dtype=np.float32), h)
    rows = np.repeat(np.arange(h, dtype=np.float32), w)

    half_w = float(w) / 2.0
    half_h = float(h) / 2.0
    dx_grid = ((cols - (half_w - 0.5)) / half_w * s_half).astype(np.float32)
    if image_y_up:
        dy_grid = ((rows - (half_h - 0.5)) / half_h * s_half).astype(np.float32)
    else:
        dy_grid = (((half_h - 0.5) - rows) / half_h * s_half).astype(np.float32)

    w_uv = np.column_stack([dx_grid, dy_grid]).astype(np.float32)
    w_ones = np.ones((n_pixels, 1), dtype=np.float32)

    # Conv 1x1: W=[1, 3, 1, 1], B=[1]
    w_conv = np.array([[[[-0.6]], [[1.0]], [[-0.6]]]], dtype=np.float32)
    b_conv = np.array([-0.06], dtype=np.float32)

    eps_val = np.array([[1e-6]], dtype=np.float32)

    w_act = np.zeros((2, action_dim), dtype=np.float32)
    w_act[0, 0] = 1.0
    w_act[1, 1] = 1.0

    b_act = np.zeros((action_dim,), dtype=np.float32)
    b_act[0] = cam_x
    b_act[1] = cam_y
    if action_dim > 2:
        b_act[2] = float(plane_z)
    if action_dim > 6:
        b_act[6] = 1.0  # vacuum tool activate

    w_state = np.zeros((action_dim, action_dim), dtype=np.float32)

    t_w_conv = _make_tensor("W_conv", w_conv)
    t_b_conv = _make_tensor("B_conv", b_conv)
    t_w_uv = _make_tensor("W_uv", w_uv)
    t_w_ones = _make_tensor("W_ones", w_ones)
    t_eps = _make_tensor("Eps", eps_val)
    t_w_act = _make_tensor("W_act", w_act)
    t_b_act = _make_tensor("B_act", b_act)
    t_w_state = _make_tensor("W_state", w_state)

    n_conv = _make_node("Conv", [image_input_name, "W_conv", "B_conv"], ["conv_out"], "n_conv")
    n_relu = _make_node("Relu", ["conv_out"], ["relu_out"], "n_relu")
    n_flat = _make_node("Flatten", ["relu_out"], ["M_flat"], "n_flat")

    n_num = _make_node("MatMul", ["M_flat", "W_uv"], ["dxy_num"], "n_num")
    n_den = _make_node("MatMul", ["M_flat", "W_ones"], ["den_raw"], "n_den")
    n_den_eps = _make_node("Add", ["den_raw", "Eps"], ["den_safe"], "n_den_eps")
    n_dxy = _make_node("Div", ["dxy_num", "den_safe"], ["dxy"], "n_dxy")

    n_act_mm = _make_node("MatMul", ["dxy", "W_act"], ["act_xy_dim"], "n_act_mm")
    n_act_base = _make_node("Add", ["act_xy_dim", "B_act"], ["act_base"], "n_act_base")

    n_state_mm = _make_node("MatMul", [state_input_name, "W_state"], ["state_dummy"], "n_state_mm")
    n_action = _make_node("Add", ["act_base", "state_dummy"], [output_name], "n_action")

    vi_img = _make_value_info(image_input_name, 1, [1, 3, h, w])
    vi_state = _make_value_info(state_input_name, 1, [1, action_dim])
    vi_action = _make_value_info(output_name, 1, [1, action_dim])

    graph_parts = [
        _encode_message(1, n_conv),
        _encode_message(1, n_relu),
        _encode_message(1, n_flat),
        _encode_message(1, n_num),
        _encode_message(1, n_den),
        _encode_message(1, n_den_eps),
        _encode_message(1, n_dxy),
        _encode_message(1, n_act_mm),
        _encode_message(1, n_act_base),
        _encode_message(1, n_state_mm),
        _encode_message(1, n_action),
        _encode_string(2, "pcb_visual_servo_graph"),
        _encode_message(5, t_w_conv),
        _encode_message(5, t_b_conv),
        _encode_message(5, t_w_uv),
        _encode_message(5, t_w_ones),
        _encode_message(5, t_eps),
        _encode_message(5, t_w_act),
        _encode_message(5, t_b_act),
        _encode_message(5, t_w_state),
        _encode_message(11, vi_img),
        _encode_message(11, vi_state),
        _encode_message(12, vi_action),
    ]
    graph = b"".join(graph_parts)
    opset = _encode_string(1, "") + _encode_int(2, opset_version)

    return (
        _encode_int(1, ir_version)
        + _encode_string(2, "rebot-companion")
        + _encode_message(7, graph)
        + _encode_message(8, opset)
    )


def scaffold_pcb_visual_servo_policy(
    target_dir: str | Path,
    policy_id: str = "pcb_visual_servo",
    action_dim: int = 7,
    robot: str = "b601dm",
    cam_pos: tuple[float, float, float] = (0.23, -0.14, 0.34),
    fovy_deg: float = 45.0,
    plane_z: float = 0.011,
    image_hw: tuple[int, int] = (64, 64),
    image_y_up: bool = False,
    overwrite: bool = False,
) -> Path:
    """Scaffold a visual-servo policy estimating PCB world centroid from overhead camera."""
    dest = Path(target_dir).resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Target directory {dest} already exists and overwrite is False")

    dest.mkdir(parents=True, exist_ok=True)

    image_input_name = "observation.images.policy_cam"
    state_input_name = "observation.state"
    action_output_name = "action"

    h, w = int(image_hw[0]), int(image_hw[1])
    inputs: dict[str, dict[str, Any]] = {
        "state": {
            "name": state_input_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        },
        "image": {
            "name": image_input_name,
            "shape": [1, 3, h, w],
            "dtype": "float32",
            "layout": "NCHW",
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    }

    outputs: dict[str, dict[str, Any]] = {
        "action": {
            "name": action_output_name,
            "shape": [1, action_dim],
            "dtype": "float32",
        }
    }

    manifest = OnnxPolicyManifest(
        format=ONNX_POLICY_FORMAT,
        name=policy_id,
        robot=robot,
        fps=30,
        action_dim=action_dim,
        unit_mode="rad",
        model_file="model.onnx",
        inputs=inputs,
        outputs=outputs,
        normalization={},
        description=f"Overhead visual servo policy {policy_id} for PCB centroid alignment",
        action_space="cartesian_xy_target",
        image_y_up=image_y_up,
    )
    manifest.save(dest / "policy.json")

    onnx_bytes = build_pcb_visual_servo_onnx_bytes(
        image_input_name=image_input_name,
        state_input_name=state_input_name,
        output_name=action_output_name,
        action_dim=action_dim,
        cam_pos=cam_pos,
        fovy_deg=fovy_deg,
        plane_z=plane_z,
        image_hw=(h, w),
        image_y_up=image_y_up,
    )
    (dest / "model.onnx").write_bytes(onnx_bytes)
    return dest


def export_pytorch_to_onnx(
    model: Any,
    dummy_inputs: Any,
    output_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, Any] | None = None,
    opset_version: int = 13,
    **kwargs: Any,
) -> Path:
    """Export a PyTorch model to an ONNX file.

    Requires PyTorch. Lazy imports torch so importing this module never requires PyTorch.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"PyTorch is required for ONNX export: {exc}") from exc

    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_inputs,
        str(dest),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs,
    )
    return dest
