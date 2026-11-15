# reBot 机械臂虚拟仿真、前端轻量化与核心能力拓展需求规格说明书
(System Requirements Specification & Architecture RFC: Virtual Simulation, WebGL Frontend & 6-DOF Extensions)

---

## 1. 文档概述与背景 (Context & Background)

### 1.1 项目体系定位
当前项目群构成了面向 AI Agent 与边缘计算的轻量级机械臂控制栈：
- **`rebot-armd`**：采用 Rust 编写的高性能底层守护进程（常驻内存仅 ~15MB，CPU < 0.5%），通过独占线程执行 50Hz 运动控制循环与五次多项式规划，向下直接收发 CAN/串口总线驱动硬件电机，向上提供双通道网络协议（HTTP REST 状态探针 + WebSocket 排他性租约控制通道）。
- **`rebot-companion`**：上层 Python SDK 与 MCP (Model Context Protocol) Server，向上为大模型（Claude / Cursor / 自主 Agent）提供原子化控制工具（`move_to`, `gripper`, `home` 等），向下调用 `rebot-armd` 协议接口。
- **视觉系统**：基于 Seeed reCamera（微型 RISC-V SoC）或 UVC/RGB 相机，提供视频推流与 2D 目标检测 API。

### 1.2 现实痛点与核心诉求
1. **脱离真实硬件开发测试（虚拟环境诉求）**：开发者本地手头没有实体机械臂时，上层 Python SDK、MCP Server 与 AI Agent 链路需要一个高保真、零物理风险的模拟被控对象以完成功能闭环验证。
2. **跨平台客户端依赖复杂性**：终端用户多使用 Windows 等操作系统，本地配置 C++ 编译链、ROS 2、显卡驱动或大型仿真软件极其复杂。系统必须践行**“零安装（Zero-Install）”**原则，把繁重依赖从用户开发机上剥离。
3. **边缘算力与资源边界**：
   - 微型板卡（如 reCamera / SG2002，内存仅 64~512MB）严禁塞入重型脚本语言或复杂 3D 渲染库；
   - 算力盒子（如 RK3576 / RK3588 / Jetson）适合充当“免安装仿真服务器”，但仍需规避本地图形桌面（OpenGL/X11/Wayland）驱动冲突。

---

## 2. 总体设计原则与设备分层 (Design Principles & Hardware Tiers)

### 2.1 核心设计哲学
- **端-云/端-端算力解耦**：**后端只做数据与物理计算，前端显卡全权负责 3D 渲染**。
- **无头至上（Headless First）**：边缘服务端绝不在本地创建物理显示窗口，所有数据走 WebSocket 实时流。
- **最小必要依赖**：拒绝引入与业务重叠的大型冗余库（如 Pinocchio），优先复用成熟原生轻量规范。

### 2.2 设备画像与角色划分

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier 1：高算力物理控制主机 (各类 x86 工控机 / 高算力设备)              │
│  - 目标：生产环境物理真机直驱                                             │
│  - 部署：armd (物理 CAN 总线驱动) + 静态重力前馈补偿                     │
│  - 核心关注：物理安全锁、急停保护、通信确定性                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Tier 2：边缘算力盒子 (RK3576 / RK3588 / Jetson 盒子)                    │
│  - 目标：开箱即用的虚拟仿真站 / AI 算法基座 (免去用户本地配环境)          │
│  - 部署：armd mock / MuJoCo 无头物理仿真 / 静态 Web 托管服务               │
│  - 用户体验：提供纯网络服务，客户端 Windows 零依赖直连                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Tier 3：微型嵌入式宿主 (Seeed reCamera / SG2002 RISC-V 板)              │
│  - 目标：贴臂极致紧凑控制器                                              │
│  - 部署：armd 单一静态二进制 (无 Python、无渲染库、常驻内存 ~15MB)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心功能需求规格 (Functional Specifications)

---

### 需求一：纯前端 3D 渲染视窗（Zero-Install WebGL 前端算力方案）

#### 1. 业务目标
用户在 Windows/Mac 电脑上无需安装 Python、C++ 环境或专用 3D 软件，直接打开现代浏览器即可实时观看机械臂 3D 连杆姿态与末端动作。

#### 2. 技术选型与算力分配
- **弃用 MeshCat**：MeshCat 运动学算力 100% 在后端（Python 端每周期计算所有 Link 的 4x4 矩阵并通过网络广播），且无法作为标准组件嵌入现有网页。
- **选用 `<urdf-viewer>` (基于 `urdf-loader` 与 Three.js)**：
  - **后端（盒子/服务器）开销为 0**：仅需通过 WebSocket 广播 6 个关节弧度浮点数组（`[q1, ..., q6]`），单帧带宽仅数十字节。
  - **前端算力拉满**：用户电脑浏览器加载 URDF，利用自身显卡 WebGL 与场景树（Scene Graph）自动解算各级坐标系变换并进行 60FPS 渲染。

#### 3. 资产复用与集成
- **URDF 路径**：复用现有 [`assets/urdf_b601dm/urdf/reBot-DevArm_gripper.urdf`](file:///home/harvest/project/rebot-arm-workspace/assets/urdf_b601dm/urdf/reBot-DevArm_gripper.urdf)。
- **3D 网格优化**：使用本地已就绪的 **LOD 轻量化 STL 网格**（位于 [`workbench/assets_lod/meshes/`](file:///home/harvest/project/rebot-arm-workspace/workbench/assets_lod/meshes/)），整臂模型体积压缩至 **4.4MB**，确保浏览器秒级加载。
- **前端核心逻辑**：
  ```html
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/urdf-loader/javascript/bundle/urdf-loader.js"></script>
  <urdf-viewer id="arm-viewer" urdf="/assets/reBot-DevArm_gripper.urdf" package="reBot-DevArm_description_fixend:/assets/meshes/" up="Z+"></urdf-viewer>
  ```
  通过监听 `ws://<HOST>:8765/api/v1/ws`，在接收到 `positions` 时调用 `viewer.setJointValues(jointDict)` 触发重绘。

---

### 需求二：MuJoCo 无头物理仿真后端（Headless Physics Simulation）

#### 1. 业务目标
在脱离真实机械臂的情况下，对物块夹取、装配接触、重力扰动及掉落滑移进行严密的真实物理验证，同时向视觉算法提供虚拟相机图流。

#### 2. 硬件可行性（RK3576 / RK3588）
- **官方二进制支持**：MuJoCo 官方提供原生 `manylinux2014_aarch64` 预编译 wheel，支持通过 `pip install mujoco` / `uv add mujoco` 一键极速安装，无需本地编译。
- **算力开销**：7-DOF 机械臂刚体数量约 10 个，MuJoCo 在 RK3576 的 Cortex-A72 核心上跑单步 `mj_step` 耗时 `< 0.1ms`。维持 500Hz 物理求解仅占用单个 A72 核心的 10%~15%，内存占用 `< 50MB`。
- **运行模式**：**强制运行无头模式（Headless）**，严禁调用本地 `mujoco.viewer.launch()` 窗口，规避板端 Mali GPU 的 Linux 桌面 OpenGL 驱动冲突。

#### 3. 官方模型资产直接接入
- **核心场景资产**：直接使用本地已调优完成的权威官方资产 [`Rebot_Arm_description/RS/mjcf/rs_grasp_scene.xml`](file:///home/harvest/project/reBot-DevArm/Rebot_Arm_description/RS/mjcf/rs_grasp_scene.xml)。
- **物理特性就绪**：
  - **惯性与质量**：来自 SolidWorks CAD 材料密度真实导出值（臂总重约 4.7kg），质心与非对角惯量完整；
  - **夹爪凸分解（Convex Decomposition）**：手指内侧网格已切片拆分为 5 个独立凸包，彻底消除 MuJoCo 凸包化导致的夹爪穿模合不拢问题；
  - **传动联动**：已通过 `<equality>` 约束与 `joint7_motor` 建立单电机双指对称闭合动力学。
- **闭环数据流**：
  MuJoCo 纯物理循环计算位姿 $\rightarrow$ 将机械臂角度与物块三维位姿封装为 WebSocket 遥测 $\rightarrow$ 驱动前端 Three.js 画面同步更新。

---

### 需求三：6-DOF 全空间任意姿态抓取拓展

#### 1. 现状事实与升级成本
- **物理硬件**：reBot B601 本身即为标准的 6 自由度串联机械臂（J1 偏航、J2 肩俯仰、J3 肘俯仰、J4 腕俯仰、J5 腕侧摆、J6 轴滚转 + J7 夹爪）。
- **底层早已支持**：`rebot-armd` 内部已具备完整的 6 轴正解、6x6 几何雅可比矩阵、阻尼最小二乘 6-DOF 数值逆解（`solve_dls`）以及 6D 笛卡尔空间 SLERP 四元数旋转插补；MCP Server（`rebot-mcp`）对外也已声明了 `roll_deg, pitch_deg, yaw_deg` 接口。
- **历史瓶颈**：仅因 Python 端 [`rebot_companion/ik.py`](file:///home/harvest/project/rebot-companion/rebot_companion/ik.py) 内部采用了针对早期垂直下抓的 4-DOF 平面反解，强行将 J5 固死为 0.0 并压平了旋转矩阵。

#### 2. 升级改造方案
- **解除 Python 逆解压平**：在 `rebot_companion/ik.py` 中引入 6-DOF 数值逆解求解器（支持将目标 6D 位姿直接交由 `armd` 现成端点 `/api/v1/action/move_to` 执行，或在 Python 侧集成 6 轴 DLS 求解）。
- **完全向后兼容设计**：
  - 当调用方仅传入 `(x, y, z)` 时，默认使用 `pitch=-90°`（垂直俯视下抓），行为与既有业务 100% 一致；
  - 当调用方传入完整的 `(roll, pitch, yaw)` 或四元数 `quat` 时，启动 6-DOF 全姿态逆解，满足斜向 45° 插接、侧面推拉及复杂空间装配。

---

### 需求四：已知静态障碍物避障（安全走廊 / 禁飞区规划）

#### 1. 业务目标
避免机械臂在直线笛卡尔移动过程中碰撞到工作台料架、容器隔板或自身底座。

#### 2. 实现路径（纯算法层拓展）
- 复用 `rebot-armd` 内置的 `box_min_m`, `box_max_m`, `floor_z_m` 基础空间限位模型。
- 上位机规划层支持定义已知环境障碍物的 AABB（轴对齐包围盒）或 OBB（定向包围盒）禁飞区集合。
- 运动生成器在起点与终点之间如果检测到直线穿透障碍物，自动插入抬升中继航点（如“先提升安全高度 $\rightarrow$ 平移至上方 $\rightarrow$ 下降靠近目标”的门字形安全走廊），经由五次多项式平滑执行。

---

## 4. 架构边界与明确不做事项 (Non-Goals / Out of Scope)

为了确保工程可控性与系统稳定性，以下方向明确**不作为**当前阶段开发目标：

| 进阶方向 | 判定结论 | 限制因素与原因 |
|---|---|---|
| **纯力矩高频阻抗/柔顺力控** | **明确不做** | ① 硬件缺少末端六维力传感器和关节力矩传感器（舵机/电机粗略电流估算死区大）；<br>② 控制周期为 50Hz（20ms），刚性接触力控需要 500~1000Hz 才能防止撞击发散。 |
| **复杂 3D 动态实时点云避障** | **明确不做** | 当前标配单目 2D RGB 摄像头（reCamera），无 RGB-D 深度图、无 LiDAR 点云输入，无法感知空中障碍物的真实三维几何体。 |
| **超高速极限飞拍搬运 ($> 1.5\text{ m/s}$)** | **明确不做** | 机械臂定位于轻型桌面协作，连杆结构刚度有限，底层安全限速为 $0.15\text{ m/s}$，强行解开限速会引发剧烈机械震颤。 |
| **引入通用重型动力学库 (Pinocchio)** | **明确不做** | 自研解析逆解耗时仅 7.9 微秒（比 Pinocchio 快约 1000 倍且绝无局部死锁），正解与重力矩计算与 Pinocchio 误差仅 $10^{-16}\text{ m}$ 与 $10^{-7}\text{ N}\cdot\text{m}$，引入 Pinocchio 属于百兆级冗余依赖。 |

---

## 5. 阶段实施路线图 (Milestones & Roadmap)

```
阶段一：零依赖 WebGL 3D 监控视窗落地 (Week 1)
  ├── 静态资源映射：挂载 LOD STL (4.4MB) 与 reBot-DevArm_gripper.urdf
  ├── 集成 <urdf-viewer> 至现有的 armd 控制台或 companion 前端模板
  └── 联调 50Hz WebSocket positions 数据流驱动姿态重绘

阶段二：6-DOF 全姿态抓取接口解封 (Week 1-2)
  ├── 升级 rebot_companion/ik.py 与 pipeline.py，支持 6D Transform 透传
  ├── 完善默认向后兼容逻辑（缺省姿态自动回退至垂直顶抓）
  └── 编写 6-DOF 倾斜抓取与姿态插补自动化测试用例

阶段三：MuJoCo 无头物理仿真环境封装 (Week 2-3)
  ├── 验证 RK3576 / RK3588 上的 aarch64 mujoco pip 包与 headless 运行脚本
  ├── 接入现成的 Rebot_Arm_description/RS/mjcf/rs_grasp_scene.xml
  ├── 搭建虚拟环境 WebSocket 桥接器（将抓取位姿推流至前端与 Agent）
  └── 固化 Dockerfile 便于算力盒子一键分发

阶段四：静态禁飞区安全走廊规划 (Week 3-4)
  ├── 定义工作台 AABB 障碍物描述 Schema
  └── 实现离散安全中继航点插入与平滑执行逻辑
```
