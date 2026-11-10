# Instructions for AI coding assistants

This file is for an AI assistant (Claude Code, Cursor, ...) that builds an arm program for a user
with this SDK. The user may have little programming experience: explain what a script will do
before running it on the arm, and keep scripts short.

Two layers:

- `rebot/`: Python standard library only. `rebot.Arm` is the API for user programs.
- `rebot_companion/mcp_server.py` (`rebot-mcp`): MCP tools for interactive control of the same arm.

## Workflow

1. **Teach poses.** Poses are saved in armd on the device. Either the user moves the arm in the web
   cockpit and saves each pose under a name, or, through the MCP server, the user puts the arm at a
   position and the assistant calls `save_pose(name)`. Names: letters, digits, `_`, `-`, up to 64
   characters. Typical names: `scan` (camera looks at the table), `bin_a`, `bin_b` (above each bin),
   `wave_1`..`wave_3`. `list_poses` shows what exists.
2. **Start from a template.** Copy `examples/templates/sort_by_label.py` (vision sorting into bins)
   or `examples/templates/demo_wave.py` (tour of three poses). Edit only the `CONFIG` dict at the
   top unless the user asks for different behaviour.
3. **Run on the PC** against a simulated arm or the real arm:
   - No arm: start armd with `armd --follower mock` and run `REBOT_HOST=127.0.0.1 python3 main.py`.
   - Real arm: `REBOT_HOST=192.168.42.1 python3 main.py` with `speed_scale` 0.3 and the user at the
     stop button. Motions are executed by the arm; stdout shows what the script is doing.
4. **Copy to the device**: `/userdata/rebot-apps/<name>/main.py`, together with the `rebot/` folder
   if the device does not have it installed. The script runs unchanged (without `REBOT_HOST` it uses
   `127.0.0.1`).
5. **Enable it** (needs rebot-teleop with the app runner). Either in the reCamera console, rebot-teleop
   settings, group "Your Program": `enabled` = `on`, `name` = `<name>`; or start it by hand with
   `/etc/init.d/K92rebot-teleop app-start <name>` (`app-status`, `app-logs 50`, `app-stop`). The
   setting is stored in `/userdata/local/apps/rebot-teleop.config.json` as
   `"app": {"enabled": "on", "name": "<name>"}`. The runner restarts a crashed program with backoff
   (gives up after 10 starts in 5 minutes) and writes `/userdata/rebot-teleop/logs/app-<name>.log`.
   It stops a program with SIGTERM and waits 10 s before SIGKILL; `Arm` turns SIGTERM into
   `SystemExit`, so the `with` block parks the arm and releases control.

## API cheat sheet

```python
from rebot import Arm, ActionFailed, ArmFaultError, RebotError

with Arm() as arm:                     # $REBOT_HOST or 127.0.0.1; takes control, torque on
    arm.home()
    arm.move_to_pose("scan", speed_scale=0.3)          # saved pose, joint-space; speed_scale 0.1..1.0
    for d in arm.detect(min_score=0.5):                # label, score, box_norm; best score first
        if arm.grasp(d).holding:                       # armd centres on the box, descends, closes, lifts
            arm.place(pose="bin_a", speed_scale=0.3)   # lift, approach, descend, open, retreat
        break
    arm.move_to((0.25, 0.0, 0.15))                     # metres, base frame: x forward, y left, z up
    arm.gripper_open(); arm.gripper_close()            # gripper_close().holding
# leaving the block parks the arm (torque off) and releases control, also after errors and Ctrl+C
```

| Call | Notes |
|---|---|
| `Arm(host, port, enable=True, park_on_exit=True, force=False, obs_hz=10, handle_sigterm=True)` | one `Arm` per program |
| `home()`, `park()` | `park` ends with torque off; the next motion turns it on |
| `move_to(xyz, rpy=None, speed=None)` | straight line; `speed` m/s (default 0.05) |
| `move_relative(dx, dy, dz, frame="base"\|"tool"\|"camera")` | relative move |
| `move_joints(positions)` | 7 values in radians |
| `poses()`, `save_pose(name)`, `delete_pose(name)`, `move_to_pose(name, speed_scale)` | capability `poses` |
| `place(pose=None, position=None, rpy=None, approach_height_m=0.08, release=True, retreat=True, speed_scale=1.0)` | exactly one of `pose` / `position`; capability `place` |
| `detect(label=None, pipelines=("yolo",), min_score=0.3)` | list of `Detection` |
| `grasp(detection)` | `result.holding` is True when an object is held |
| `gripper_open()`, `gripper_close(grasp=True)`, `gripper_move(fraction)` | 0 closed .. 1 open |
| `state()` | `positions`, `safety_state`, `is_torqued`, `fault`, `action`, `collision` |
| `stop()`, `e_stop()` | stop and hold; latched emergency stop |
| `recover()` | clears only faults armd marks recoverable, then torque on |
| `wait_until_clear(timeout_s=10)` | after `COLLISION`: blocks until the arm has resumed; raises `ArmFaultError` (code `COLLISION`) after repeated collisions; capability `collision_guard` |
| `set_collision_guard(enabled=None, sensitivity=None)`, `collision_guard()` | switch the guard / set `"low"`, `"medium"`, `"high"`; read the configuration; capability `collision_guard` |
| `guard(timeout_s)`, `feed()` | release control if `feed()` is not called for `timeout_s`; call `feed()` once per loop cycle |
| `capabilities` | dict of armd capabilities; version 6 has `poses`, `place`, `fault_codes`, `keepout` |

Every call blocks until armd reports the result and raises a `RebotError` subclass with `.code` and
`.message` on failure.

## Safety rules

- **Faults.** Never clear a fault or an E-stop in a script. `arm.recover()` is the only allowed
  call: it clears only faults with `recoverable: true` (`WATCHDOG_TIMEOUT`, `CONTROL_DEADLINE`,
  `HANDOVER_UNSETTLED`) and raises `ArmFaultError` for anything else. When it raises, the script
  stops and tells the user to check the arm and clear the fault in the web cockpit. Do not retry in
  a loop and do not work around it (for example with `--allow-clear-fault` in the MCP server).
- **Collision guard** (capability `collision_guard`). With the guard on, contact stops the arm and holds
  it; the arm resumes by itself once the contact is gone, and motion commands are refused with
  `COLLISION_HOLD` meanwhile. The action that was running fails with `COLLISION`. In a program: catch
  `ActionFailed` with `code == "COLLISION"`, call `arm.wait_until_clear()`, retry the step once, and stop
  if it fails again. After repeated trips armd latches a `COLLISION` fault: `arm.recover()` refuses it;
  stop and ask the user to check the workspace and clear the fault in the cockpit. The guard is **off by
  default** and its thresholds are **uncalibrated placeholders** (not measured on a real arm); do not
  rely on it as protection. Switching it on (`set_collision_guard`) is a configuration change: do it only
  when the user asks.
- **Safety description** (capability `safety_describe`). Before writing a program, call the MCP tool
  `describe_safety` (or `Arm.safety()` in a script). Keep speeds within `limits` (`action_speed_max`
  and the joint limits). Never automate recovery of a fault with `recoverable: false`. When
  `hardware.physical_estop` is false, tell the user that the arm has no physical e-stop: only software
  stops exist, so someone must be able to cut power.
- **After torque off.** `park()` or a thermal trip turns torque off and the wrist may droop. Call
  `home()` before the first `move_to` of a program; without it the simulator reported `IK_BRANCH_FLIP`.
- **Speed.** Use `speed_scale` <= 0.5 for the first real runs of any new script or pose, 0.3 when
  the arm moves near people, bins or the table. Raise it only after a successful run.
- **Always `with Arm() as arm:`.** It parks and releases control on errors, Ctrl+C and SIGTERM. Do not
  call `Arm()` without `with` unless `arm.close()` is guaranteed.
- **Hung loops.** In scripts that loop, call `arm.guard(seconds)` once and `arm.feed()` each cycle.
- **Bounded loops.** Give every loop a maximum number of cycles.
- **Configuration changes.** `save_pose`, `delete_pose` and `set_keepout` change what is stored on the
  device. Do them only when the user asks, and tell the user which names or boxes change.
- **Keep-out boxes** (`get_keepout`, `set_keepout`) mark space the arm must not enter (bins' walls,
  people's area). Motions into a box fail with `WORKSPACE_VIOLATION`.
- Do not use `e_stop()` as part of normal program flow. Do not run two programs on one arm: only one
  client holds control at a time.
- Do not use `force=True` in `Arm(...)`: it takes control from the web cockpit.

## Errors

| Code | Meaning | What to do |
|---|---|---|
| `NOT_HOLDING` | grasp closed on nothing | open the gripper, scan again, continue; count it |
| `TARGET_LOST` | the object disappeared during the grasp | same as `NOT_HOLDING` |
| `IK_UNREACHABLE`, `IK_BRANCH_FLIP` | target out of reach | move the object or the pose closer; do not loop on the same target |
| `WORKSPACE_VIOLATION` | target outside the workspace or inside a keep-out box | change the target or the box |
| `POSE_NOT_FOUND` | pose name not saved | `list_poses`; teach the pose |
| `TRACKING_STALL` | the arm did not follow, possibly blocked | stop, ask the user to check the scene |
| `PLACE_TIMEOUT`, `GRIPPER_TIMEOUT`, `MOTION_TIMEOUT` | action did not finish | stop; check the scene before retrying |
| `MOTION_BUSY` | another motion is running | `stop()`, then retry |
| `NOT_ARMED` | torque is off | `enable()` (the `Arm` methods do this themselves after a park) |
| `UNSUPPORTED` | armd lacks the capability | tell the user to update rebot-armd on the device |
| `WATCHDOG_TIMEOUT`, `CONTROL_DEADLINE`, `HANDOVER_UNSETTLED` | recoverable faults | `arm.recover()`, then continue |
| `COLLISION` (action failed) | the guard stopped the arm on contact; it resumes by itself | `arm.wait_until_clear()`, retry the step once; stop if it happens again |
| `COLLISION_HOLD` (command rejected) | the arm is holding after a collision | `arm.wait_until_clear()`, then send the command again |
| `COLLISION` (fault, `ArmFaultError`) | repeated collisions latched a fault; not recoverable | stop; the user checks the workspace and clears it in the cockpit |
| `FOLLOWING_ERROR` | arm could not follow the command; the only collision indicator when the guard is off | stop; the user checks the arm and clears it in the cockpit |
| `ESTOP`, `ESTOP_ACTIVE`, `THERMAL_TRIP`, `BUS_LOSS`, `JOINT_LIMIT`, other fault codes | not recoverable | stop; the user clears it in the cockpit |
| `LEASE_BUSY`, `LEASE_LOST`, `NO_LEASE` | another client (the cockpit) holds or took control | ask the user to release control in the cockpit |
| `GuardTimeout` | the script stopped calling `feed()` and control was released | fix the loop, restart the script |
| `CONNECTION` | armd not reachable | check `REBOT_HOST` and that armd is running |

## MCP tools for interactive work

`rebot-mcp` exposes `get_state`, `snapshot`, `snapshot_annotated`, `detect`, `move_to`,
`move_relative`, `gripper`, `home`, `park`, `stop`, `e_stop`, `enable`, `take_control`,
`release_control`, `list_poses`, `save_pose`, `move_to_pose`, `place`, `get_keepout`, `set_keepout`,
`recover`, `collision_guard_status`, `set_collision_guard` and `describe_safety`. `get_state.collision` shows a collision
stop in progress. Use them to find positions and test steps; put the final behaviour in a script so
it runs without the assistant. Tools needing a newer armd return `UNSUPPORTED`.

## Checks before finishing

- `uv run pytest -q` passes when the SDK itself was changed.
- The script ran against `armd --follower mock` or at `speed_scale` <= 0.5 on the real arm.
- The script has `CONFIG` at the top, comments a beginner can follow, no third-party imports, and
  a bounded loop.
