"""Multi-seed coffee_kiosk benchmark and optional LeRobot v3.0 dataset recorder.

Compares fixed-pose vs vision-guided execution across randomized cup placements on
`rebot-sim` (`coffee_kiosk` + optional custom gripper) and prints stage pass rates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from rebot import ActionFailed, Arm, ArmFaultError, RebotError
from rebot_companion.scenes import SimClientError, SimSceneClient

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from coffee_kiosk_fsm import run_episode  # noqa: E402


def _format_table(rows: list[dict[str, Any]], stages: list[str]) -> str:
    cols = ["seed", *stages, "completed"]
    widths = {c: max(len(c), 7) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        vals = [str(r["seed"]).ljust(widths["seed"])]
        st = r.get("stages", {})
        for s in stages:
            vals.append(("PASS" if st.get(s) else "FAIL").ljust(widths[s]))
        vals.append(("PASS" if r.get("completed") else "FAIL").ljust(widths["completed"]))
        lines.append(" | ".join(vals))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate coffee_kiosk across seeds and optionally record demos")
    parser.add_argument("--episodes", type=int, default=3, help="Number of seeded episodes to run")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting placement seed")
    parser.add_argument("--scene", default="coffee_kiosk", help="Scene pack id or local directory")
    parser.add_argument("--tool", default="wide_cup_gripper", help="Tool pack id or local directory")
    parser.add_argument("--sim-http", default="http://127.0.0.1:8090", help="rebot-sim perception/control URL")
    parser.add_argument("--arm-host", default=os.environ.get("REBOT_HOST", "127.0.0.1"), help="armd host")
    parser.add_argument("--camera-url", default=None, help="Optional ws://<host>:8001/ for LeRobot image recording")
    parser.add_argument("--record-dir", default=None, help="Save successful episodes as a LeRobot v3.0 dataset")
    parser.add_argument("--fps", type=int, default=20, help="Dataset recording FPS when --record-dir is set")
    parser.add_argument("--speed-scale", type=float, default=0.4, help="Motion speed scale 0.1..1.0")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision detection (fixed-rule baseline)")
    parser.add_argument("--dry-run", action="store_true", help="Load scene and reset seeds without connecting to armd")
    args = parser.parse_args(argv)

    sim = SimSceneClient(base_url=args.sim_http)
    try:
        loaded = sim.load_scene(scene=args.scene, tool=args.tool, seed=args.seed_start)
    except SimClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stage_names = list((loaded.get("task") or {}).get("stages", {}).keys()) or [
        "cup_grasped",
        "cup_on_tray",
        "brew_pressed",
        "cup_served",
    ]

    if args.dry_run:
        scene_id = loaded.get("scene") if isinstance(loaded.get("scene"), str) else args.scene
        tool_id = loaded.get("tool") if isinstance(loaded.get("tool"), str) else args.tool
        print(f"loaded scene={scene_id} tool={tool_id}")
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            st = sim.reset(seed=seed)
            cup = (st.get("objects") or {}).get("cup", {})
            stages = list((st.get("task") or {}).get("stages", {}).keys())
            print(f"  seed={seed}: cup pos={cup.get('pos')} stages={stages}")
        return 0

    recorder = None
    if args.record_dir:
        from rebot_companion.lerobot_recorder import LeRobotRecorder

        recorder = LeRobotRecorder(
            root=args.record_dir,
            repo_id="local/coffee_kiosk",
            fps=args.fps,
            camera_url=args.camera_url,
            armd_host=args.arm_host,
            task="place cup on drip tray, press brew button, serve cup",
        )

    results: list[dict[str, Any]] = []
    os.environ["REBOT_HOST"] = args.arm_host
    try:
        if recorder is not None:
            recorder.open()
        with Arm(host=args.arm_host) as arm:
            arm.guard(30.0)
            for ep in range(args.episodes):
                arm.feed()
                seed = args.seed_start + ep
                sim.reset(seed=seed)
                time.sleep(0.15)
                if recorder is not None:
                    recorder.start_episode()
                safety_failure = False
                try:
                    run_episode(arm, use_vision=not args.no_vision, speed_scale=args.speed_scale)
                except ActionFailed as exc:
                    if exc.code in {"NOT_HOLDING", "TARGET_LOST"}:
                        print(f"seed {seed} aborted ({exc.code}): {exc}")
                    else:
                        print(f"seed {seed} motion-safety failure ({exc.code}): {exc}")
                        safety_failure = True
                except (ArmFaultError, RebotError) as exc:
                    print(f"seed {seed} motion-safety failure: {exc}")
                    safety_failure = True
                task_info = sim.state().get("task") or {}
                completed = bool(task_info.get("completed"))
                if recorder is not None:
                    recorder.stop_episode()
                    if completed:
                        recorder.save()
                    else:
                        recorder.discard()
                results.append(
                    {
                        "seed": seed,
                        "stages": dict(task_info.get("stages") or {}),
                        "completed": completed,
                    }
                )
                if safety_failure:
                    print(f"terminating evaluation early due to motion-safety failure on seed {seed}")
                    break
    finally:
        if recorder is not None:
            recorder.close()

    print(_format_table(results, stage_names))
    passed = sum(1 for r in results if r["completed"])
    total = len(results)
    if total > 0:
        print(f"\ncompleted: {passed}/{total} ({(100.0 * passed / total):.1f}% pass rate)")
    else:
        print("\ncompleted: 0/0 (0.0% pass rate)")
    return 0 if (total > 0 and passed == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
