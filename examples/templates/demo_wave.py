"""Demo: the arm goes home, visits three taught poses, opens and closes the gripper, then parks.

Before running, teach three poses named in CONFIG["poses"] (web cockpit, or the AI assistant's
save_pose tool). Pick positions that are clear of people and objects.

Run (PC):      REBOT_HOST=192.168.42.1 uv run python demo_wave.py
Run (device):  python3 main.py            (no REBOT_HOST needed)
"""

from rebot import ActionFailed, Arm, ArmFaultError

# ---- edit this part ----------------------------------------------------------------
CONFIG = {
    "poses": ["wave_1", "wave_2", "wave_3"],  # names of three taught poses, visited in order
    "rounds": 2,  # how many times to repeat the tour
    "speed_scale": 0.3,  # 0.1 .. 1.0; keep <= 0.5 for the first real runs
}
# ------------------------------------------------------------------------------------


def retry_after_collision(arm, step):
    """Run step(); if the arm bumped into something, wait for it to move on its own and try once more."""
    try:
        return step()
    except ActionFailed as exc:
        if exc.code != "COLLISION":
            raise
        print("The arm touched something and stopped. Waiting for it to move on its own...")
        arm.wait_until_clear()  # raises ArmFaultError (code COLLISION) after repeated collisions
        print("The arm is free again, trying the step once more.")
        return step()


def run(arm, cfg):
    """Returns the number of completed rounds (0 when a pose is missing)."""
    saved = {p["name"] for p in arm.poses()}
    missing = [name for name in cfg["poses"] if name not in saved]
    if missing:
        print("These poses are not saved yet:", ", ".join(missing))
        return 0

    speed = cfg["speed_scale"]
    rounds_done = 0
    try:
        retry_after_collision(arm, arm.home)  # start from a known position
        for round_no in range(1, cfg["rounds"] + 1):
            for name in cfg["poses"]:
                print(f"round {round_no}: moving to {name}")
                retry_after_collision(arm, lambda name=name: arm.move_to_pose(name, speed_scale=speed))
                retry_after_collision(arm, arm.gripper_open)
                retry_after_collision(arm, lambda: arm.gripper_close(grasp=False))  # close fully, no object search
            rounds_done += 1
        retry_after_collision(arm, arm.home)
        arm.park()  # resting pose, torque off
    except (ActionFailed, ArmFaultError) as exc:
        if exc.code != "COLLISION":
            raise
        print("The arm stopped after repeated collisions. Check the workspace, then clear it in the web cockpit.")
        return rounds_done
    print("done")
    return rounds_done


if __name__ == "__main__":
    with Arm() as arm:  # connects, takes control; on exit parks and releases control
        run(arm, CONFIG)
