"""reBot arm SDK core: standard library only, Python >= 3.8.

Runs unchanged on a PC (arm over the network) and on the reCamera itself (127.0.0.1).
"""

from .arm import Arm
from .client import (
    ActionResult,
    ActionStatus,
    ArmdClient,
    ArmSession,
    ArmStatus,
    JointLimit,
    ObservationSchema,
    SafeExitOutcome,
)
from .detection import Detection, DetectionClient, parse_detections
from .errors import (
    ActionFailed,
    ArmCommandRejected,
    ArmConnectionError,
    ArmdError,
    ArmFaultError,
    DetectorUnavailable,
    GuardTimeout,
    LeaseError,
    MotionTimeout,
    RebotError,
    ServoJumpError,
    ServoRejected,
    UnloadFailed,
    UnsupportedError,
)

__all__ = [
    "Arm",
    "ArmdClient",
    "ArmSession",
    "ArmStatus",
    "ActionStatus",
    "ActionResult",
    "ObservationSchema",
    "JointLimit",
    "SafeExitOutcome",
    "Detection",
    "DetectionClient",
    "parse_detections",
    "RebotError",
    "ArmdError",
    "ArmConnectionError",
    "UnsupportedError",
    "LeaseError",
    "ArmCommandRejected",
    "ServoRejected",
    "ArmFaultError",
    "MotionTimeout",
    "ActionFailed",
    "ServoJumpError",
    "UnloadFailed",
    "DetectorUnavailable",
    "GuardTimeout",
]
