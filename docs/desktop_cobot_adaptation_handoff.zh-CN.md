# 家用桌面型协作臂适配评估与开发交接文档（内部）

整理日期：2026-11-13

## 代码审计基线

- `rebot-armd`：以 `/home/harvest/project/rebot-armd-public`（commit `48e0252`，2026-11-13）为准。**同级目录 `/home/harvest/project/rebot-armd` 停留在 `9318635`（2026-11-09），缺少 URDF `chain`、碰撞残差、功率预算与力矩保护模块，接手开发前务必先核对最新主线分支。**
- `rebot-companion`：`/home/harvest/project/rebot-companion`（commit `2af6028`，2026-11-12）。

---

## 1. 已确认的产品形态与三层职责边界

- 产品定位：**自研家用桌面型协作臂**（无外部工业控制柜，无闭源高层 `moveJ` / `moveL` 控制器与 `armd` 争抢规划权）。
- 硬件拓扑：**一体化关节电机（内置 FOC 驱动）+ 底座小型外设 MCU + Linux 主控运行 `armd`**。
- 评估结论：**整体适配难度低；`armd` 核心逻辑约 90% 可复用，`rebot-companion` 仅需约 2–5 天小改，用户编程接口 `with Arm() as arm:` 与 24 个 MCP 工具保持零破坏性变更。**

| 层级 | 负责内容 | 不做的内容 |
|---|---|---|
| 底座小型外设 MCU（`< 1 ms` 硬实时保命与机身外设） | 母线电压/电流 ADC 采样与过流/反电动势硬件保护（解决 2026-11-12 J4 堵转拉崩 24V 电源导致全臂掉电问题）；失电抱闸开合时序；实体拍停急停回路与状态上报；底座状态灯环；腕部拖动键/记点键；末端快换工具在位识别 | **不做逆运动学、不做轨迹插补、不封装慢速 `moveL` 黑盒协议**（否则会阻塞 `rebot-policy` 的 **30–50 Hz** 连续策略流） |
| `armd`（`20 ms` 规划、安全围栏与协议网关） | 保持 `:8765` HTTP REST + WebSocket Lease（**2.5 s** 心跳互斥）不变；通用 URDF 正逆解（`src/chain.rs`，支持 **1..8 轴**）、轨迹规划、**16 个**三维动态禁飞盒扫掠检查（`src/keepout.rs`）、`servo_guard` 策略流限幅、视觉闭环 `grasp` / `place`、Web 驾驶舱、Parquet 录数与回放 | 不再靠纯软件数学估算替代真实母线采样；带抱闸机型不再为防掉臂强制全臂回窝 |
| `rebot-companion`（Python SDK / MCP / Policy） | 清理少量 B601-DM 专属硬编码常量，全部改为从 `armd` 的 `/api/v1/observation/schema` 与 `/api/v1/observation` 动态读取，实现一套代码同时兼容 B601-DM 与新款家用协作臂 | 不保留宿主机本地 B601 正逆解分支与硬编码型号分支 |

---

## 2. 硬件/固件立项联调必核清单

接手开发前需向硬件/固件团队核实以下 **5 项**输入：

1. 关节电机协议与力控能力：是否支持实时力矩反馈与力矩前馈（如 MIT 模式）（需核实；**无力矩反馈的舵机方案无法启用 `collision_guard` 防夹手与 `gravity_comp` 零力拖动**）。
2. 自由度与末端快换形态：沿用 **6 臂轴 + 1 CAN 夹爪（共 7 轴）**，还是 **6/7 轴臂 + 独立 IO / RS485 快换末端**（夹爪、吸盘、笔夹等）（需核实）。
3. 底座 MCU 物理挂载方式：是否作为独立节点挂在同一条 CAN 总线上（例如电机占 ID `1..7`，底座 MCU 占 ID `16`，共用 **1 根** CAN 线接入主控），还是走独立串口（需核实）。
4. 断电防坠与零点保持：关节采用失电电磁抱闸还是机械高阻尼防坠，是否配备掉电保持绝对值编码器（免开机回零）（需核实）。
5. 主控算力平台：`armd` 继续部署在 SG2002 reCamera（当前 **50 Hz** 占单核 **7% CPU / 3.7 MB RAM**）还是迁移至独立边缘计算盒（需核实；若电机总线升至 **500–1000 Hz** CAN-FD，需实测 SG2002 调度抖动）。

---

## 3. 底座外设 MCU ↔ `armd` 建议协议报文（4 类薄接口）

1. 急停与供电遥测（MCU → `armd`）：实体急停按下状态（映射 `hardware.physical_estop = true`）、母线电压、实测母线电流（替代 `src/power.rs` 数学估算）、硬件过流告警。
2. 抱闸与使能时序联动（`armd` ↔ MCU）：上使能“建力矩 → 松抱闸”，下使能/急停“合抱闸 → 断电流”；停机或看门狗超时原地锁抱闸，配置 `resting_pose: null`，不再触发自动全臂回窝（避免扫翻桌面物品）。
3. 机身按键事件（MCU → `armd`）：腕部拖动键按住/松开（触发 `src/gravity_comp.rs` 状态机进出零力拖动）、记点键单击（触发 `src/poses.rs` 保存当前位姿）。
4. 状态灯环与末端工具通道（`armd` ↔ MCU）：根据 `SafetyState` 切换底座灯环状态；上报在位末端工具 ID 与夹持/吸合反馈。

---

## 4. 跨仓库精确代码改造清单

### 4.1 协议契约收敛点（优先实施，向后兼容）

- `GET /api/v1/observation/schema`：显式提供 `gripper_joint: usize | null` 及末端开闭量程元数据。
- `GET /api/v1/observation`：直接输出实时 `tcp` 位姿与归一化 `gripper: {fraction, holding, tool_id?}`。

### 4.2 `rebot-armd` 改造清单（基准：`/home/harvest/project/rebot-armd-public`）

- 驱动与底座 MCU 接入：
  - `src/driver.rs:93-116`：接入底座 MCU 状态收发；若更换电机品牌则新增对应 `ArmDriver` 实现。
- 抱闸停靠与看门狗行为：
  - `src/model.rs`、`src/supervisor.rs`：支持带抱闸机型在 `resting_pose: null` 下原地锁止断力矩，看门狗超时原地停机锁抱闸而非回窝。
- 解除 `six_plus_gripper()` 硬门禁并抽离独立 `EndEffector` 接口（非“6 臂轴 + 第 7 轴 CAN 夹爪”时必须改）：
  - `src/model.rs:629-631`（`six_plus_gripper()` 定义）与 `src/model.rs:806-808`（`gravity_comp` 校验）
  - `src/chain.rs:448-452`（放开 `ArmModel::solve_dls` 仅允许 `c.dof() == 6` 的截断）
  - `src/motion.rs:117-147`（将固定 `[f32; 7]` 轨迹点与 `limits.len() < 7` 改为按 `dof` 动态分配）
  - `src/place.rs:110-112`、`src/grasp.rs:861-863`、`src/supervisor.rs:1968-1970`
- 去达妙专属参数与标定家用防夹保护：
  - `src/torque_guard.rs:66-69`（脱离 `damiao_ranges.t_max`，改为通用关节额定/峰值力矩配置）
  - `src/power.rs:248-254`（支持读取底座 MCU 实测电流替代 DM-J4340P/J4310 估算模型）
  - `src/collision.rs:41-54`、`src/supervisor.rs:2580-2582`（扩展动态轴数，实机运行 `rebot-sysid` 标定摩擦与残差阈值，出厂默认开启 `collision_guard`）
- Web 驾驶舱去 B601 硬编码：
  - `src/api/index.html:3878-3886`（浏览器端 `CHAIN` 连杆常数）与 `src/api/index.html:7433-7437`（3D URDF `joint1..6` + 双指映射），改为按 `schema` 与模型文件动态驱动。

### 4.3 `rebot-companion` 改造清单（本仓库，基准：`/home/harvest/project/rebot-companion`）

- MCP Server 清理本地正解与夹爪常数：
  - `rebot_companion/mcp_server.py:73-74`（`-5.7..0.0 rad`）、`rebot_companion/mcp_server.py:403-404`（本地 `tcp_dict` 与 `positions[GRIPPER_INDEX]`）、`rebot_companion/kinematics.py:50-59` → 统一改读 `armd` 返回的 `tcp` 与 `gripper`。
- SDK 夹爪索引动态化：
  - `rebot/client.py:72`（`GRIPPER_INDEX = 6`）、`rebot/client.py:1374-1376`（`servo` 步长豁免）、`rebot/client.py:1438-1442`（`_tolerances` 豁免）→ 改读 `schema.gripper_joint`。
- 抱闸停靠终态兼容：
  - `rebot/client.py:1041-1048`（`_wait_unload`）与 `rebot/arm.py:214-226`（`_park_for_exit`）→ 兼容抱闸锁止语义；按设备能力区分 `AGENTS.md:95-96` 的断电下垂说明。
- Policy Runner 与录制器动态自由度：
  - `rebot_companion/policy_runner.py:86`（`ROBOT_TYPE`）、`rebot_companion/policy_runner.py:704-706`（`dof != 7` 硬门禁）、`rebot_companion/policy_runner.py:763`（lead 裁剪豁免）、`rebot_companion/lerobot_recorder.py:74`（`JOINT_NAMES`）→ 改读 `schema` 的 `dof`、`joint_names`、`gripper_joint`，支持 `--robot-type`。
- 参数上限与本地拾取管线：
  - `rebot/client.py:1270-1273`、`rebot_companion/mcp_server.py:75-88` 上限对齐 `safety_describe.limits`；`rebot_companion/pipeline.py:220-228` 改调 `/api/v1/action/gripper` 而非直接覆写 `q[6]`。

---

## 5. 分阶段实施路线

1. **阶段 1（硬件到货前，纯软件解耦）**：完成 4.1 协议字段补充 + 4.3 `rebot-companion` 动态化 + 4.2 `six_plus_gripper()` 拆解；使用 `/home/harvest/project/rebot-armd-public/profiles/rebot_b601_rs.json`（6 轴无夹爪）在 `armd --follower mock` 下跑通笛卡尔移动、MCP 与 `rebot-policy`，同时保证 B601-DM 全部现有测试零回归。
2. **阶段 2（底座 MCU 与台架联调）**：验证 CAN/串口外设报文、`< 1 ms` 实体急停切断、抱闸使能时序、堵转防掉电保护。
3. **阶段 3（整机标定与家用安全验收）**：导入新臂 URDF/MJCF 跑 `sim` 闭环 → 真机跑 `rebot-sysid` 辨识摩擦与碰撞阈值并默认开启 `collision_guard` → 验证腕部按键拖动记点与视觉 `grasp` 连续抓取。

---

## 6. 接手参考文档与建议调用的 Skills

### 关联文档索引

- `/home/harvest/project/rebot-armd-public/docs/architecture.md`
- `/home/harvest/project/rebot-armd-public/docs/control_safety_contract.md`
- `/home/harvest/project/rebot-companion/AGENTS.md`
- `/home/harvest/project/rebot-companion/docs/plan_backlog.zh-CN.md`

### Suggested Skills

- `taste`：方案取舍、渐进替换与存量兼容检查
- `dev-flow`：跨仓库特性拆解、实现审查与实机/仿真端到端门禁
- `device-gotchas`：reCamera SG2002 / CAN 总线等边缘硬件排障经验库
- `codex-rescue` / `review` / `simplify`：控制与安全状态机独立代码审查及减熵瘦身
- `ux-review`：Web 驾驶舱 3D 视图、碰撞提示及机身交互体验验收

### 回归验证门禁命令

- `rebot-armd-public`：`cargo test && cargo clippy --all-targets -- -D warnings`
- `rebot-companion`：`uv run pytest -q`
