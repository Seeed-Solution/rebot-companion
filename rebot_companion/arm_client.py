"""Re-export of :mod:`rebot.client` (the protocol layer moved to the stdlib-only ``rebot`` package)."""

from rebot.client import *  # noqa: F401,F403
from rebot.client import (  # noqa: F401
    ACTION_MAX_TIMEOUT_S,
    GRIPPER_INDEX,
    SAFE_EXIT_POLICIES,
    SERVO_MAX_STEP,
    SERVO_MIN_RATE_HZ,
    __all__,
)
