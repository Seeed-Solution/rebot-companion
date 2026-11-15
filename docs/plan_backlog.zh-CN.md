# reBot / reCamera 开发计划与待办（内部）

更新：2026-11-11。状态：✅ 完成（mock / MuJoCo / 备用 reCamera 验证）· 🔶 代码完成待真机 · ⏳ 计划中 · ❓ 待决策

## 已合并并推送（2026-11-11）
- ✅ armd main `b3ff997`：开发平台契约 v6、app runner、驾驶舱、MuJoCo 仿真、碰撞保护、安全自描述、仿真相机与真值感知（`7173862`）、相机朝下抓取（`b3ff997`，v8 支架）。
- ✅ companion main `80bd04c`：stdlib SDK `rebot`、MCP 24 个工具、LeRobot 录制 `rebot-record`、策略运行桥 `rebot-policy`（LeRobot 后端 + 脚本策略）。

## 进行中
- 🔶 修复分支（armd，待审查后合并到本地 main）：
  - `fix/grasp-descend-clearance`：抓取每段移动检查桌面 +5 mm 间隙；codex 审查 1 个 medium（低于间隙豁免按单点比较）修复中。
  - `fix/app-user-gid-cgroup`：rebot-app 用户 gid 修复、pidfd 清理残留进程；codex 审查 1 medium（非 ESRCH 错误被吞）+ 1 low（exit 4 前可能已写 shadow）修复中。
  - `fix/sim-seg-readback`：仿真分割回读偶发越界，查根因中。
- 🔶 README 补充：armd 的 sim follower / 仿真相机 / 相机朝下抓取；companion 的仿真作为录制与策略的相机来源（`docs/readme-sim-grasp`、`docs/readme-sim` 分支）。

## 真机上线后（P0 测试包）
- 🔶 抓取 50 次成功率；手眼误差；内参标定（当前 FOV 为推导值）
- 🔶 摩擦辨识（rebot-sysid）→ 碰撞阈值 → 开启碰撞保护
- 🔶 电机真实参数回填仿真
- 🔶 v8 支架打印装配，camera_mount 外参与 rot/tilt 符号、ISP 镜像确认；桌面间隙 5 mm 与 recheck_tol_px 40 按 `agent_action_api.md`「Real-arm calibration (v8 mount)」标定
- 🔶 电机 CAN 通信超时（主控挂掉时电机行为）
- 🔶 驾驶舱 / 3D 视图 / app runner 在臂端 reCamera 与真手机上运行
- ⏳ 夹爪零位校验（需硬件实现）

## 计划中（排期待定）
- ⏳ 学习型策略 C：仿真中用抓取宏做脚本专家，rebot-record 录 N 条 → spark 训练 ACT → rebot-policy 仿真评估，与宏对比成功率。
- ⏳ 学习型策略 D：**运行时不依赖 torch/LeRobot**：ONNX 后端（onnxruntime + numpy），训练机导出 model.onnx + policy.json，仿真中对比 torch/ONNX 输出与成功率。
- ⏳ 宏动作与策略可互换接口（arm.grasp() 背后可选宏或策略）；宏 vs 策略评估。
- ⏳ 电脑端 `rebot deploy/logs`；驾驶舱用户程序面板；仿真模式进画廊或盒子打包；无 CAN 设备 mock 被 `requires:["can"]` 拦截的处理。
- ⏳ 最初两份规格文档的事实更正。
- ⏳ 生态：LeRobot armd 插件、MCP Registry / awesome-mcp-servers、Claude Code 插件、MHS 申请（见 ecosystem_cooperation_notes.zh-CN.md）。

## 待决策
- ❓ 认证范围（非本机 token、AP 默认密码、录制需租约）
- ❓ 硬件急停回路（B4 底板）
- ❓ 设备时钟 / NTP
