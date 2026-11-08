"""rebot-companion: Robot Arm Vision Companion SDK and Orchestration Framework."""

from . import kinematics
from .arm_client import (
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
    ServoRejected,
)
from .calibration import CalibrationError, CalibrationResult, CalibrationSample, HandEyeCalibration
from .camera_client import MarkerDetection, ReCameraClient
from .pipeline import DeterministicPickPlacePipeline, PipelineConfig, PipelineResult
from .transforms import Transform

__all__ = [
    "ReCameraClient",
    "MarkerDetection",
    "ArmdClient",
    "ArmSession",
    "ArmStatus",
    "ObservationSchema",
    "ArmdError",
    "ArmConnectionError",
    "ArmCommandRejected",
    "ServoRejected",
    "LeaseError",
    "ArmFaultError",
    "MotionTimeout",
    "Transform",
    "HandEyeCalibration",
    "CalibrationSample",
    "CalibrationResult",
    "CalibrationError",
    "DeterministicPickPlacePipeline",
    "PipelineConfig",
    "PipelineResult",
    "kinematics",
]
