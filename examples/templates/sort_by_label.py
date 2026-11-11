"""Vision sorting: find objects with the camera, pick each one up, drop it into the bin for its label.

Before running, teach these poses (web cockpit, or the AI assistant's save_pose tool):
  - the scan pose: the arm looks at the table from here
  - one bin pose per label: the gripper is above the bin where objects of that label go

Run (PC):      REBOT_HOST=192.168.42.1 uv run python sort_by_label.py
Run (device):  python3 main.py            (no REBOT_HOST needed)
First real run: keep speed_scale at 0.3 or lower and stay near the stop button.
"""

import sys

from rebot import ActionFailed, Arm, ArmCommandRejected, ArmFaultError

# ---- edit this part ----------------------------------------------------------------
CONFIG = {
    "scan_pose": "scan",  # pose name the arm returns to before every detection
    "bins": {  # detected label -> pose name of the bin
        "cup": "bin_a",
        "bottle": "bin_b",
    },
    "max_cycles": 10,  # pick attempts before stopping
    "max_empty_scans": 3,  # stop after this many scans with nothing to sort
    "speed_scale": 0.3,  # 0.1 .. 1.0; keep <= 0.5 for the first real runs
    "min_score": 0.5,  # ignore detections below this confidence (0..1)
    "guard_s": 120,  # release the arm if this script hangs for this long
}
# ------------------------------------------------------------------------------------


def retry_after_collision(arm, step, retry=None):
    """Run step(); if the arm bumped into something (or still holds after a bump), wait for it to
    move on its own and run retry() (default: step) once more."""
    try:
        return step()
    except (ActionFailed, ArmCommandRejected) as exc:
        if exc.code not in ("COLLISION", "COLLISION_HOLD"):
            raise
        print("The arm touched something and stopped. Waiting for it to move on its own...")
        arm.wait_until_clear()  # raises ArmFaultError (code COLLISION) after repeated collisions
        print("The arm is free again, trying the step once more.")
        return (retry or step)()


def run(arm, cfg):
    """Sort until max_cycles or until the table is empty. Returns (successes, failures)."""
    missing = [n for n in [cfg["scan_pose"], *cfg["bins"].values()] if n not in {p["name"] for p in arm.poses()}]
    if missing:
        print("These poses are not saved yet:", ", ".join(sorted(set(missing))))
        return 0, 0

    arm.guard(cfg["guard_s"])  # if the loop below stops calling arm methods, control is released
    speed = cfg["speed_scale"]
    successes = failures = empty = 0

    for cycle in range(1, cfg["max_cycles"] + 1):
        arm.feed()  # tell the guard the script is alive
        try:
            if cycle == 1:  # known position first: after torque off the wrist may droop
                retry_after_collision(arm, arm.home)
            retry_after_collision(arm, lambda: arm.move_to_pose(cfg["scan_pose"], speed_scale=speed))
            retry_after_collision(arm, arm.gripper_open)  # start every cycle with an open gripper
            targets = [d for d in arm.detect(min_score=cfg["min_score"]) if d.label.lower() in cfg["bins"]]
            if not targets:
                empty += 1
                print(f"cycle {cycle}: nothing to sort ({empty}/{cfg['max_empty_scans']})")
                if empty >= cfg["max_empty_scans"]:
                    break
                continue
            empty = 0
            target = targets[0]  # best score first

            def grasp_again(label=target.label):  # the object may have moved: scan again, never reuse the old box
                arm.move_to_pose(cfg["scan_pose"], speed_scale=speed)
                fresh = arm.detect(label, min_score=cfg["min_score"])
                if not fresh:
                    raise ActionFailed("TARGET_LOST", f"{label} not found again after the collision")
                return arm.grasp(fresh[0])

            if not retry_after_collision(arm, lambda target=target: arm.grasp(target), grasp_again).holding:
                raise ActionFailed("NOT_HOLDING", "gripper closed on nothing")
            bin_pose = cfg["bins"][target.label.lower()]
            retry_after_collision(arm, lambda bin_pose=bin_pose: arm.place(pose=bin_pose, speed_scale=speed))
            successes += 1
            print(f"cycle {cycle}: {target.label} -> {cfg['bins'][target.label.lower()]}")
        except (ActionFailed, ArmFaultError) as exc:
            failures += 1
            print(f"cycle {cycle}: failed: {exc.code} {exc.message}")
            if exc.code == "COLLISION":  # hit again after the retry: let the arm free itself first
                try:
                    arm.wait_until_clear()
                except ArmFaultError:
                    print("stopping, repeated collisions. Check the workspace, then clear the fault in the cockpit.")
                    break
            if arm.state().is_stopped:  # fault or E-stop
                try:
                    arm.recover()  # clears only faults the arm reports as recoverable
                except ArmFaultError as fault:
                    print("stopping, the arm needs a person to check it:", fault.message)
                    break
            # NOT_HOLDING / TARGET_LOST (missed or lost the object) and other action errors:
            try:
                retry_after_collision(arm, arm.gripper_open)  # let go of anything still held, then next cycle
            except ActionFailed as stuck:
                print("stopping, the gripper could not open:", stuck.message)
                break

    print(f"done: {successes} sorted, {failures} failed")
    return successes, failures


if __name__ == "__main__":
    with Arm() as arm:  # connects, takes control; on exit parks and releases control
        ok, failed = run(arm, CONFIG)
    sys.exit(0 if failed == 0 else 1)
