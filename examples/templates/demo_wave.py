"""Demo: the arm goes home, visits three taught poses, opens and closes the gripper, then parks.

Before running, teach three poses named in CONFIG["poses"] (web cockpit, or the AI assistant's
save_pose tool). Pick positions that are clear of people and objects.

Run (PC):      REBOT_HOST=192.168.42.1 uv run python demo_wave.py
Run (device):  python3 main.py            (no REBOT_HOST needed)
"""

from rebot import Arm

# ---- edit this part ----------------------------------------------------------------
CONFIG = {
    "poses": ["wave_1", "wave_2", "wave_3"],  # names of three taught poses, visited in order
    "rounds": 2,  # how many times to repeat the tour
    "speed_scale": 0.3,  # 0.1 .. 1.0; keep <= 0.5 for the first real runs
}
# ------------------------------------------------------------------------------------


def run(arm, cfg):
    """Returns the number of completed rounds (0 when a pose is missing)."""
    saved = {p["name"] for p in arm.poses()}
    missing = [name for name in cfg["poses"] if name not in saved]
    if missing:
        print("These poses are not saved yet:", ", ".join(missing))
        return 0

    speed = cfg["speed_scale"]
    arm.home()  # start from a known position
    for round_no in range(1, cfg["rounds"] + 1):
        for name in cfg["poses"]:
            print(f"round {round_no}: moving to {name}")
            arm.move_to_pose(name, speed_scale=speed)
            arm.gripper_open()
            arm.gripper_close(grasp=False)  # grasp=False: close fully without looking for an object
    arm.home()
    arm.park()  # resting pose, torque off
    print("done")
    return cfg["rounds"]


if __name__ == "__main__":
    with Arm() as arm:  # connects, takes control; on exit parks and releases control
        run(arm, CONFIG)
