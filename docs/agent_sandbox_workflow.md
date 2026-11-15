# Agent Sandbox, Training & ONNX Policy Workflow

This guide describes the API-first, Agent-driven workflow connecting the four execution tiers in the `rebot` ecosystem:

| Mode | Where physics runs | Timing | Camera | Intended use |
|---|---|---|---|---|
| **Physical arm** (`armd` serial/CAN) | Real hardware (`b601dm` / `so-arm101`) | Real-time (500 Hz) | Real USB / reCamera | Production deployment on RK + arm |
| **Real-time sim (`mode="sim"`)** | `rebot-sim` (`mujoco` Python process) + `armd --follower sim` | Real-time (500 Hz physics) | MuJoCo offscreen wrist + fixed RTSP/JPEG/seg | Pre-flight verification before running on hardware |
| **Browser WASM (`mode="wasm"`)** | User browser (`mujoco_wasm.js` + `onnxruntime-web` + Three.js `#v3-canvas`) | Best-effort `requestAnimationFrame` | Offscreen Three.js / WebGL wrist crop | Zero-backend preview, fast task iteration, ONNX browser smoke test |
| **Headless fast training (`SimEnv`)** | `rebot_sim.env.SimEnv` (Python `mujoco`) | Unpaced (faster than wall-clock, zero sleeps) | Optional `mujoco.Renderer` (`render_camera="wrist"`) | RL / behavioral cloning / dataset rollouts / policy evaluation |

No scene-editor modals or model-upload forms are built into the Web Cockpit. An AI coding assistant edits scene packs, custom tools, training loops, and ONNX policies locally, then pushes and switches them over HTTP using `rebot-scene` or `/api/v1/sandbox/*`.

---

## 1. Prerequisites & Runtime Assets

1. **Browser WASM runtimes** (stored outside the `armd` binary and `.deb` package):

```bash
cd /path/to/rebot-armd-public
node scripts/fetch_wasm_runtime.mjs
```

This downloads `mujoco-js@0.0.7` and `onnxruntime-web@1.21.0` from `https://registry.npmmirror.com` into `assets/wasm/` (`ARMD_WASM_DIR` can override this location). `armd` serves those files lazily on `GET /assets/wasm/{*path}`.

2. **Python ONNX inference extra** (needed only for `rebot-policy` ONNX execution and validation, without installing `torch` or `lerobot`):

```bash
cd /path/to/rebot-companion
uv sync --extra onnx
```

---

## 2. Built-in Scene Packs (Including `pcb_suction`)

List built-in scenes and tools:

```bash
rebot-scene list
```

Built-in scenes include:
- `coffee_kiosk`: 4-stage beverage cell (`default` parallel jaw or `wide_cup_gripper`).
- `pcb_suction`: 3-stage PCB handling & test workflow (`suction` tool default, `infeed_tray` -> `test_fixture` -> `outfeed_bin` over a `0.050 m` `divider_wall` with `pcb_picked`, `pcb_in_fixture`, `pcb_in_outfeed` task stages and `test_pass_led`).

Scaffold custom directories when creating a new task or tool:

```bash
rebot-scene init ./my_screw_cell --kind scene --id screw_cell
rebot-scene init ./my_suction_head --kind tool --id micro_suction
rebot-scene validate ./my_screw_cell
```

---

## 3. Non-Realtime Training with `rebot_sim.env.SimEnv`

Use `SimEnv` for high-throughput rollouts without real-time `time.sleep` pacing. Actions are motor joint targets in radians (`POS_VEL` control law with `qfrc_bias` gravity compensation matching `rebot_sim.server.Motors`).

```python
import numpy as np
from rebot_sim.env import SimEnv

with SimEnv(scene="pcb_suction", tool="suction", robot="b601dm") as env:
    obs, info = env.reset(seed=42)
    env.set_io({"vacuum": True})
    for _ in range(200):
        action = obs["state"].copy()
        action[1] = -0.35  # valid interior target inside [-2.1944, 0.0]
        obs, reward, done, info = env.step(action)
        if done:
            break
    print(info["task"])
```

Pass `render_camera="wrist"` to include `obs["image"]` (`uint8` `(H, W, 3)`) when training visuomotor policies.

---

## 4. Unified ONNX Policy Contract (`rebot-onnx-policy/v1`)

A portable policy directory contains `policy.json` and `model.onnx`:

```json
{
  "format": "rebot-onnx-policy/v1",
  "name": "pcb_suction_bc",
  "robot": "b601dm",
  "fps": 30,
  "action_dim": 7,
  "unit_mode": "rad",
  "model_file": "model.onnx",
  "inputs": {
    "state": { "name": "observation.state", "shape": [1, 7], "dtype": "float32" }
  },
  "outputs": {
    "action": { "name": "action", "shape": [1, 7], "dtype": "float32" }
  },
  "normalization": {
    "state_mean": [0, 0, 0, 0, 0, 0, 0],
    "state_std": [1, 1, 1, 1, 1, 1, 1],
    "action_mean": [0, 0, 0, 0, 0, 0, 0],
    "action_std": [1, 1, 1, 1, 1, 1, 1]
  }
}
```

Optional keys:
- Visual input under `inputs["image"]`: `{"name": "observation.images.wrist", "shape": [1, 3, H, W], "dtype": "float32", "layout": "NCHW", "scale": 0.00392156862745098, "mean": [0, 0, 0], "std": [1, 1, 1]}`.
- Virtual tool I/O output under `outputs["io"]`: `{"name": "io", "channels": ["vacuum"], "threshold": 0.5}`.

Helpers in `rebot_companion.onnx_policy`:
- `scaffold_onnx_policy(target_dir, policy_id, action_dim=7, include_image=False)`: generates a valid `policy.json` + identity linear `model.onnx` using Python stdlib + `numpy` only (no `torch`/`onnx` required).
- `export_pytorch_to_onnx(model, dummy_inputs, output_path, input_names=..., output_names=...)`: exports trained PyTorch policies via lazy `import torch` (`opset_version=13`).
- `validate_policy_dir(path)`: validates manifest schema, file bounds, and `onnxruntime.InferenceSession` tensor names.

CLI commands:

```bash
rebot-scene init ./policies/pcb_demo --kind policy --id pcb_demo
rebot-scene validate ./policies/pcb_demo
```

---

## 5. Browser WASM Sandbox & Real-Time Sim Pre-Flight via `armd`

All sandbox REST APIs are hosted directly on `armd` (`http://127.0.0.1:8080` by default):

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/sandbox/catalog` | Active mode, active scene/tool/policy IDs, `wasm_runtime_ready`, lists uploaded and built-in packs |
| `POST /api/v1/sandbox/scenes` | Upload validated `scene.json` + `world.xml` (+ optional Base64 meshes) |
| `POST /api/v1/sandbox/tools` | Upload validated `tool.json` + `tool.xml` (+ optional Base64 meshes) |
| `POST /api/v1/sandbox/policies` | Upload validated `policy.json` + `model.onnx` (`rebot-onnx-policy/v1`) |
| `POST /api/v1/sandbox/activate` | Set `mode` (`"none" \| "wasm" \| "sim"`), `scene`, `tool`, `policy`, `seed`, `auto_start_policy`; composes MJCF XML and (in `"sim"`) forwards to `rebot-sim` |
| `GET /api/v1/sandbox/bundle` | Returns `composed_xml`, manifests, `policy_model_url`, `desired_io`, `reset_seq`, `io_seq` |
| `GET /api/v1/sandbox/state` | Returns latest browser WASM telemetry (`sim_time_s`, `positions`, `objects`, `task`, `io`, `policy_status`) or proxies `/sim/state` in `"sim"` mode |
| `POST /api/v1/sandbox/reset` | Resets scene objects/stages (`seed` optional); forwarded to `rebot-sim` in `"sim"` mode |
| `POST /api/v1/sandbox/io` | Writes virtual I/O (e.g. `{"updates": {"vacuum": true}}`) |
| `POST /api/v1/sandbox/telemetry` | Receives telemetry updates pushed from the browser `WASM_SIM` loop |

### Step-by-step Agent workflow

1. **Upload custom scene / tool / ONNX policy**:

```bash
rebot-scene upload ./my_screw_cell --armd-url http://127.0.0.1:8080
rebot-scene upload ./policies/pcb_demo --armd-url http://127.0.0.1:8080
```

2. **Preview in browser WASM (`armd --follower mock`, zero `rebot-sim` required)**:

```bash
rebot-scene activate --mode wasm --scene pcb_suction --tool suction --policy pcb_demo --armd-url http://127.0.0.1:8080
```

Open `http://127.0.0.1:8080/?sim=wasm` (or normal Cockpit URL once activated). `#v3-canvas` automatically opens, compiles `composed_xml` in `mujoco_wasm.js`, runs `onnxruntime-web` on `model.onnx`, renders `pcb_suction` + `suction` geoms, and streams stage telemetry back to `GET /api/v1/sandbox/state`. Inspect live progress from terminal:

```bash
rebot-scene status --armd-url http://127.0.0.1:8080
```

3. **Pre-flight in real-time backend (`armd --follower sim` + `rebot-sim --scene pcb_suction --tool suction`)**:

```bash
rebot-scene activate --mode sim --scene pcb_suction --tool suction --armd-url http://127.0.0.1:8080
rebot-policy --policy onnx:./policies/pcb_demo --max-seconds 15 --fps 30
```

4. **Run the same ONNX policy on RK + physical arm** (zero PyTorch/LeRobot needed; only pass `--i-am-at-the-arm` after confirming physical workspace safety and emergency power cut access):

```bash
REBOT_HOST=<rk-ip> rebot-policy --policy onnx:./policies/pcb_demo --fps 30 --max-seconds 15 --i-am-at-the-arm
```
