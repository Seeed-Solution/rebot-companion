# rebot-companion · Robot Arm Vision Companion Toolkit

[中文文档](./README.zh-CN.md)

> **reCamera provides high-performance atomic sensing; remote Agents provide flexible orchestration; hardened pipelines provide deterministic execution.**

An open-source embodied robotics companion toolkit bridging **Seeed reCamera (SG2002 RISC-V edge vision)** and **rebot-armd (ultra-lightweight Rust arm daemon)**.

---

## 💡 Architecture & Boundary Definition

Given the edge compute constraints (SG2002 single-core 1.0 GHz RISC-V with 256MB RAM) and real-world industrial requirements, this toolkit enforces a clear **three-tier boundary**:

```text
┌────────────────────────────────────────────────────────┐
│  Stage 1: Remote Cloud / Host Agent Flexible Planning  │
│  Stage 2: Hardened Local Deterministic Pipeline Loop   │
└───────────────────────────┬────────────────────────────┘
                            │ REST / WebSocket
                            ▼
┌───────────────────────────────────┬────────────────────┐
│ Arm Control Daemon: rebot-armd    │ Vision Sensor: reCamera
│ - Pure Rust motor servo (50~100Hz)│ - Native C++ (Zero Python on camera)
│ - Slew Rate limiter & G(q) gravity│ - H.264 video stream (Port 8001)
│ - Watchdog & E-Stop state machine │ - On-demand single trigger /api/trigger
│ - LeRobot Parquet recorder        │ - Dynamic standby throttling (idle_fps=1)
└───────────────────────────────────┴────────────────────┘
```

### 1. reCamera Responsibility: Pure Atomic Capabilities
- **Zero Python on Camera**: To avoid CPU bottlenecks, context-switching overhead, and thermal throttling.
- **Pure Standard Endpoints**:
  - `POST /api/trigger`: Single-shot trigger estimating 6-DoF pose (`x, y, z, roll, pitch, yaw`), immediately returning to standby.
  - `GET /snapshot.jpg`: High-res JPEG snapshot for multimodal VLM / LLM reasoning.
  - `WS :8001/`: Low-latency H.264 preview stream.

### 2. Two-Stage Evolution Workflow
- **Stage 1 (Exploration & Dynamic Orchestration)**:
  - Complex reasoning, multi-view inspection, and flexible task decomposition are driven by remote Agents / host Python scripts calling atomic APIs.
- **Stage 2 (Hardened Local Execution)**:
  - Once calibration matrices, trajectory points, and pick policies converge, the logic is serialized into a **deterministic local pipeline** running autonomously on the factory floor without external LLM dependencies.

---

## 🚀 Quickstart

```bash
cd rebot-companion
uv sync
uv run pytest
```
