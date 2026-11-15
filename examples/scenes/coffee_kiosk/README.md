# `coffee_kiosk` Simulation Scene

A 4-stage coffee kiosk scenario loaded on demand into `rebot-sim`:

1. `cup_grasped` — pick the `cup` off the counter
2. `cup_on_tray` — place the `cup` on the espresso `drip_tray`, avoiding the `espresso_column`
3. `brew_pressed` — compress the physical spring `brew_btn_joint` (`brew_led` switches red to green)
4. `cup_served` — move the `cup` onto the `pickup_counter`

## Rules vs. Vision vs. Imitation Learning

| Sub-task | Recommended Method | Reason |
|---|---|---|
| Cup pick on table (`cup_grasped`) | 2D vision (`arm.detect` + `arm.grasp`) or IL policy | Customer/barista cup position varies across episodes (`jitter_xy`). Fixed replay misses when seed shifts. |
| Place under spout (`cup_on_tray`), press switch (`brew_pressed`), serve (`cup_served`) | Fixed `poses` + `keepout_boxes` (`coffee_kiosk/scene.json`) | The espresso machine column, drip tray, button pedestal, and pickup coaster are rigidly mounted fixtures. |
| Closed-loop recovery, compliant insertion, or lid handling | `LeRobotRecorder` + `rebot-policy` | Contact-rich adjustments or deformable props require closed-loop visual feedback. |

Compare `--no-vision` (fails when cup seed jitter moves the cup away from `cup_fallback`) against vision mode in `train_and_eval_demo.py`.

## Quickstart (3 terminals)

1. Start MuJoCo simulator with the built-in scene and custom cup gripper:

```bash
rebot-sim --camera --scene coffee_kiosk --tool wide_cup_gripper
```

2. Start `armd` connected to the simulated follower:

```bash
armd --follower sim
```

3. Run the pure-`rebot` FSM (`coffee_kiosk_fsm.py` has zero third-party dependencies):

```bash
REBOT_HOST=127.0.0.1 python examples/scenes/coffee_kiosk/coffee_kiosk_fsm.py
```

Inspect task stage progress at any time via `rebot-scene`:

```bash
rebot-scene info coffee_kiosk
rebot-scene load --scene coffee_kiosk --tool default          # switch back to stock gripper
rebot-scene load --scene coffee_kiosk --tool wide_cup_gripper # switch back to tall silicone cup jaws
```

## Multi-seed evaluation & LeRobot dataset recording

Preview seed jitter and stage definitions without `armd`:

```bash
python examples/scenes/coffee_kiosk/train_and_eval_demo.py --dry-run --episodes 5
```

Evaluate across seeds and save completed episodes into a LeRobot v3.0 dataset:

```bash
python examples/scenes/coffee_kiosk/train_and_eval_demo.py \
  --episodes 5 --camera-url ws://127.0.0.1:8001/ --record-dir /tmp/coffee_kiosk_ds
```

Replay a recorded episode back through `rebot-policy`:

```bash
rebot-policy --arm 127.0.0.1 --policy replay:/tmp/coffee_kiosk_ds:0 --fps 20
```

## Bring your own gripper or scene

Generate starter manifests and load them into `rebot-sim` without modifying `rebot` firmware:

```bash
rebot-scene init ./my_suction_scene --kind scene
rebot-scene init ./my_gripper --kind tool
rebot-scene validate ./my_gripper
rebot-scene load --scene ./my_suction_scene --tool ./my_gripper
```
