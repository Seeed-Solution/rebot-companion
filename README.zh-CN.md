# rebot-companion · 机械臂视觉伴侣开源开发套件

> **reCamera 负责极致的高性能原子感知，远程 Agent 负责灵活编排，固化程序负责工业级闭环。**

面向 **Seeed reCamera（SG2002 边缘视觉）** 与 **rebot-armd（极简 Rust 机械臂底座）** 的开源具身协同框架。

---

## 💡 核心架构与边界划分

针对嵌入式芯片（SG2002 单核 1.0 GHz RISC-V，256MB 内存）的算力瓶颈与工业现场的灵活性需求，本项目确立了明确的**三层职责边界**：

```text
┌────────────────────────────────────────────────────────┐
│  阶段一：云端 / 上位机 Agent 柔性编排 (多模态推理与决策)     │
│  阶段二：本地固化确定性程序 (Hardened Pipeline 极速闭环)  │
└───────────────────────────┬────────────────────────────┘
                            │ REST / WebSocket
                            ▼
┌───────────────────────────────────┬────────────────────┐
│ 运动控制底座: rebot-armd          │ 视觉感知底座: reCamera
│ - 纯 Rust 电机伺服控制 (50~100Hz)  │ - 纯 C++ 原生运行 (0 Python 依赖)
│ - Slew Rate 平滑插值与重力补偿 G(q)│ - H.264 视频流 (Port 8001)
│ - 机械臂安全状态机与 E-Stop 看门狗  │ - 按需单次触发 /api/trigger
│ - LeRobot Parquet 格式示教录制    │ - 动态待机控频 (idle_fps=1, 38℃常温)
└───────────────────────────────────┴────────────────────┘
```

### 1. reCamera 边缘职责：只做原子能力，杜绝本地复杂编排
- ❌ **严禁在 reCamera 上运行重型 Python 脚本或大模型**：避免单核 RISC-V 发生卡顿与高发热。
- ✅ **只提供通用、标准的原子接口**：
  - `POST /api/trigger`：单次触发抓拍，返回工件 6-DoF 姿态（X, Y, Z, Roll, Pitch, Yaw），算完立即待机。
  - `GET /snapshot.jpg`：提供高清图象快照，供远程 VLM / Agent 理解。
  - `WS :8001/`：提供低时延 H.264 视频推流。

### 2. 两阶段业务演进流
- **阶段一（探索与动态编排 - Exploration Mode）**：
  - 复杂的视觉语义理解、机械臂抓取时序规划、多模态 VLM 推理，由上位机或云端 Agent（如 Claude / GPT-4o / 本地 Python 脚本）通过调用 reCamera 和 `rebot-armd` 的标准化 API 来驱动。
  - 开发者可以随意调整提示词、抓取逻辑与异常重试，不影响边缘设备。
- **阶段二（固化与本地化运行 - Hardened Mode）**：
  - 当手眼标定矩阵、抓取动作序列、工位坐标完全验证收敛后，将逻辑固化为**确定性流水线程序（`Deterministic Pipeline`）**。
  - 直接在本地工控机或上位机运行，脱离对远程大模型的依赖，实现毫秒级高可靠闭环。

---

## 📦 项目结构

```text
rebot-companion/
├── pyproject.toml              # 基于 uv 的现代 Python 项目依赖管理
├── rebot_companion/            # 核心 SDK
│   ├── __init__.py
│   ├── camera_client.py        # reCamera 原子能力客户端 (HTTP/WS)
│   ├── arm_client.py           # rebot-armd 运动控制客户端 (REST/WS)
│   ├── transforms.py           # 空间 SE(3) 刚体变换与坐标系映射
│   ├── calibration.py          # Eye-to-Hand 与 Eye-in-Hand 手眼标定求解器
│   └── pipeline.py             # 阶段二固化流水线执行引擎
├── examples/                   # 经典实操示例
│   ├── 01_atomic_api_demo.py   # 示例 1: 调用 reCamera 原子接口与待机控制
│   ├── 02_hand_eye_calib.py    # 示例 2: 交互式 9 点手眼标定向导
│   ├── 03_agent_orchestrator.py# 示例 3: 阶段一 远程 Agent 工具调用抓取编排
│   └── 04_hardened_pick_place.py# 示例 4: 阶段二 固化后的确定性现场抓取闭环
└── tests/                      # 单元测试与 Mock 验证
```

---

## 🚀 快速上手

### 1. 安装环境 (推荐使用 uv)
```bash
cd rebot-companion
uv sync
```

### 2. 验证 reCamera 原子接口
```bash
uv run python examples/01_atomic_api_demo.py --camera-url http://192.168.42.1:8090
```

### 3. 运行手眼标定
```bash
uv run python examples/02_hand_eye_calib.py
```

### 4. 运行阶段二固化闭环抓取
```bash
uv run python examples/04_hardened_pick_place.py
```
