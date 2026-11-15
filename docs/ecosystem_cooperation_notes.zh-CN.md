# reBot / reCamera 生态合作备忘（内部，不开源）

整理日期：2026-11-11。数据来自 GitHub API、pypistats、npm API、HF API、Discord invite API 和公开网页，时间为 2026-11-10/11。标"需核实"的条目未经确认。

## 1. 现状：已经进入的入口

| 项目 | 规模 | Seeed 已有的接入 |
|---|---|---|
| Hugging Face LeRobot | 28,797 star，6,049 fork，约 303 contributors；PyPI 月下载 237,322 | B601 原生集成：`docs/source/rebot_b601.mdx`，PR #3624（2026-05），PR #4535（2026-09，DM/RS） |
| MuJoCo Menagerie | 4,340 star，625 fork；mujoco PyPI 月下载约 402 万 | `seeed_rebot_devarm`（RobStride 版，PR #300，2026-08）；B601-DM 是否覆盖需核实 |

## 2. 候选动作（按"可触达受众 ÷ 工作量"排序）

| # | 动作 | 规模依据 | 工作量 | 备注 |
|---|---|---|---|---|
| 1 | LeRobot 插件 `lerobot_robot_rebot_armd`：LeRobot 在 PC 上录制/训练/跑策略，臂由 reCamera 上的 armd 控制；之后 PR 进 `third_party_robots` 页 | 见上 | 中 | LeRobot 有 `lerobot_robot_*` 插件自动发现；与学习型策略路线衔接；需说明与原生 B601 集成的区别 |
| 2 | rebot-mcp 提交 MCP 官方 Registry（server.json）；向 punkpeye/awesome-mcp-servers 提 PR | Registry 仓库 7,458 star；awesome-mcp-servers 97,579 star；`mcp` PyPI 月下载 2.35 亿 | 小 | Registry 搜 "robot" 前 30 条无机械臂类服务；需先确认 rebot-mcp 已发 PyPI（需核实） |
| 3 | 打包为 Claude Code 插件，提交官方市场 | anthropics/claude-plugins-official 46,768 star | 小 | 提交表单：https://clau.de/plugin-directory-submission |
| 4 | 申请 Anthropic MHS 研究预览 | 2026-08-27 发布，按申请准入 | 小（申请）/ 大（合作） | 合作方含 UR、Doosan、AWS、Hugging Face、Raspberry Pi |
| 5 | B601-DM MJCF 提交 Menagerie | 见上 | 小～中 | 先开 issue；仓库有多个 open PR，审核周期不确定 |
| 6 | `node-red-contrib-rebot` 上 flows.nodered.org | node-red 23,831 star；node-red-dashboard npm 月下载 196,818 | 中 | reCamera 自带 Node-RED |
| 7 | AWS strands-labs/robots 加 B601/armd 支持 | 214 star，2026-11-10 仍有提交 | 中 | 加机器人的路径需读代码核实 |
| 8 | robotmcp/ros-mcp-server 互相引用 / ros2_control 示例 | 1,519 star；`ros-mcp` PyPI 月下载约 1,464 | 中 | 受众较小 |
| — | Home Assistant、Foxglove、ros2_control | 92,129 / 321 / 1,054 star | — | 与机械臂需求匹配度未验证，暂不做 |

参照对象：Reachy Mini（reachy_mini 1,654 star，Pollen Discord 15,319 人，app store 2026-05-06 上线，200+ app），可作为设备端 app runner / 应用商店形态的参考。

## 3. 一周内可做 vs 需要商务

- 一周内可做（对外提交前需负责人确认）：确认 rebot-mcp 已发 PyPI 并写 server.json；准备 awesome-mcp-servers 条目；Claude Code 插件打包；MHS 申请表；核对 Menagerie 是否覆盖 DM；LeRobot `rebot_b601` 文档补 armd/reCamera 链接。
- 需要商务/合作：MHS 伙伴关系（联合发布、案例）；HF/LeRobot 文档或博客位置；AWS strands-labs 联合 PR；robotmcp 互相引用。

## 4. 风险

- 目录收录门槛低、同类条目多，收录本身不等于流量。
- LeRobot 插件 API 和 B601 类变化快（两次 PR 相隔约 4 个月），需要持续维护。
- 原生 B601 集成与 armd 插件功能重叠，需讲清差异：控制器在相机里、armd 侧安全兜底。
- Menagerie 审核排队，周期不确定。
- Registry / 插件市场的收录规则只读了文档，未实测提交。
- 未找到数据：Glama / Smithery / PulseMCP / mcp.so、Hackster / Hackaday、Seeed 社区的流量；Seeed 官网 SO-ARM101 在售情况（商品页 403，需核实）。

## 5. 同方向项目（2026-11-11 检索）

- 已有：机械臂 MCP（ROS、SO-101、UR、myCobot）；Claude 经 MCP + 相机控制真机（psamin/cortex PR#21、LeRobot 团队演示）；仿真/真机同一接口（AWS Strands Robots）；服务端安全包络（armguard-mcp，alpha，仅仿真）；教育臂 + LLM（Hiwonder NexArm，$259.99–279.99）；AI 生成设备 app（Reachy Mini）。
- 本次检索未找到同类公开实现（非绝对论断）：相机本体作为臂控制器；守护进程侧 lease + 心跳看门狗 + 机器可读安全自描述；MCP + 设备端 app runner + LeRobot v3 录制 + MuJoCo 跟随器 + 浏览器 cockpit 的组合。
- 方向信号：Anthropic MHS（2026-08-27）；Gemini Robotics-ER 2 公开预览（2026-07-30）；NVIDIA GR00T 2 / Isaac Lab 3.0（2026-11-09）；Physical Intelligence 2026-03 报道估值约 $11B。

## Sources

- https://github.com/huggingface/lerobot
- https://github.com/huggingface/lerobot/blob/main/docs/source/rebot_b601.mdx
- https://github.com/huggingface/lerobot/blob/main/docs/source/integrate_hardware.mdx
- https://github.com/huggingface/lerobot/blob/main/docs/source/third_party_robots.mdx
- https://github.com/google-deepmind/mujoco_menagerie
- https://github.com/modelcontextprotocol/registry
- https://github.com/punkpeye/awesome-mcp-servers
- https://github.com/anthropics/claude-plugins-official
- https://modelhardwarestandard.com
- https://www.anthropic.com/news/model-hardware-standard-research-preview
- https://github.com/strands-labs/robots
- https://github.com/robotmcp/ros-mcp-server
- https://github.com/pollen-robotics/reachy_mini
- https://github.com/node-red/node-red
- https://www.hiwonder.com/products/nexarm
- https://github.com/Lynn-hh/armguard-mcp
- https://github.com/psamin/cortex/pull/21
- https://pypistats.org/api/packages/lerobot/recent
- https://pypistats.org/api/packages/mcp/recent
- https://pypistats.org/api/packages/mujoco/recent
