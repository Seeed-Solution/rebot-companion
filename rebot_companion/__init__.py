"""rebot-companion: Robot Arm Vision Companion SDK and Orchestration Framework."""

from rebot.client import (
    ActionFailed,
    ActionResult,
    ActionStatus,
    ArmCommandRejected,
    ArmConnectionError,
    ArmdClient,
    ArmdError,
    ArmFaultError,
    ArmSession,
    ArmStatus,
    LeaseError,
    MotionTimeout,
    ObservationSchema,
    SafeExitOutcome,
    ServoJumpError,
    ServoRejected,
    UnloadFailed,
)

from . import ik, kinematics, snapshot
from .calibration import CalibrationError, CalibrationResult, CalibrationSample, HandEyeCalibration, HandEyeResult
from .camera_client import CameraError, CameraTriggerTimeout, MarkerDetection, ReCameraClient
from .ik import B601IkAdapter, IkError
from .pipeline import DeterministicPickPlacePipeline, PipelineConfig, PipelineResult
from .transforms import Transform

__all__ = [
    "ReCameraClient",
    "MarkerDetection",
    "CameraTriggerTimeout",
    "CameraError",
    "ArmdClient",
    "ArmSession",
    "ArmStatus",
    "ActionStatus",
    "ActionResult",
    "ActionFailed",
    "ObservationSchema",
    "ArmdError",
    "ArmConnectionError",
    "ArmCommandRejected",
    "ServoRejected",
    "LeaseError",
    "ArmFaultError",
    "MotionTimeout",
    "ServoJumpError",
    "UnloadFailed",
    "SafeExitOutcome",
    "Transform",
    "HandEyeCalibration",
    "CalibrationSample",
    "CalibrationResult",
    "CalibrationError",
    "HandEyeResult",
    "DeterministicPickPlacePipeline",
    "PipelineConfig",
    "PipelineResult",
    "B601IkAdapter",
    "IkError",
    "ik",
    "kinematics",
    "snapshot",
]
