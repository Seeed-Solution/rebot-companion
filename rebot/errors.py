"""Exceptions. Every SDK error is a :class:`RebotError` with ``code`` and ``message``."""

from __future__ import annotations

from typing import Any, Optional


class RebotError(Exception):
    """Base class: ``code`` is a stable upper-case identifier, ``message`` says what to do."""

    code = "ERROR"

    def __init__(self, message: str = "", code: Optional[str] = None) -> None:
        if code:
            self.code = code
        self.message = message
        super().__init__(f"{self.code}: {message}" if message else self.code)


ArmdError = RebotError  # name used by rebot_companion before the stdlib core existed


class ArmConnectionError(RebotError):
    """armd could not be reached, or the WebSocket closed."""

    code = "CONNECTION"


class UnsupportedError(RebotError):
    """The connected armd does not offer this feature (check ``capabilities``)."""

    code = "UNSUPPORTED"


class LeaseError(RebotError):
    """Lease could not be acquired, or was lost while in use (``reason`` is armd's reason)."""

    code = "LEASE"

    def __init__(self, message: str, reason: str = "", raw: Optional[dict] = None) -> None:
        self.reason = reason
        self.raw = raw or {}
        code = {"controller_busy": "LEASE_BUSY", "preemption_cooldown": "PREEMPT_COOLDOWN"}.get(reason, "LEASE_LOST")
        if reason == "not_held":
            code = "NO_LEASE"
        super().__init__(message, code)


class ArmCommandRejected(RebotError):
    """armd rejected a command (``command_reject`` / ``servo_reject``)."""

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.status = raw.get("status")
        self.safety_state = raw.get("safety_state")
        self.epoch_id = raw.get("epoch_id")
        code = str(raw.get("code") or raw.get("error") or raw.get("reason") or "REJECTED")
        super().__init__(str(raw.get("message") or raw.get("reason") or code), code)


class ServoRejected(ArmCommandRejected):
    """armd rejected a streamed servo frame."""


class ArmFaultError(RebotError):
    """The arm is in EStop/Fault (``status`` is the observation; ``code`` the fault code when known)."""

    def __init__(self, status: Any, message: Optional[str] = None, code: Optional[str] = None) -> None:
        self.status = status
        fault = getattr(status, "fault", None) or {}
        if code is None:
            code = fault.get("code") or ("ESTOP" if getattr(status, "e_stopped", False) else "FAULT")
        if message is None:
            reason = fault.get("message") or getattr(status, "fault_reason", None)
            message = f"arm entered {getattr(status, 'safety_state', '?')}" + (f" ({reason})" if reason else "")
        super().__init__(message, code)


class MotionTimeout(RebotError):
    """Motion did not finish within the timeout. A stop command has been sent."""

    code = "MOTION_TIMEOUT"


class ActionFailed(RebotError):
    """A high-level action ended ``failed`` or ``cancelled`` (``code`` is armd's ``action.code``)."""

    def __init__(self, code: str, message: str = "", status: Any = None, obs: Any = None) -> None:
        self.status = status
        self.obs = obs
        super().__init__(message, code)


class ServoJumpError(RebotError, ValueError):
    """A servo frame steps more than 0.35 rad from the last commanded target; armd would drop it silently."""

    code = "SERVO_JUMP"


class UnloadFailed(RebotError):
    """armd finished the unload motion but did not cut torque (arm left Armed and torqued)."""

    code = "UNLOAD_FAILED"

    def __init__(self, status: Any) -> None:
        self.status = status
        super().__init__(
            "unload motion finished but armd did not cut torque (arm is Armed and torqued); "
            "the post-unload torque cut failed or the arm did not settle at the resting pose"
        )


class DetectorUnavailable(RebotError):
    """armd's detector proxy answered 502 or could not be reached."""

    code = "DETECTION_UNAVAILABLE"


class GuardTimeout(RebotError):
    """The program did not call ``Arm.feed()`` within the guard timeout; the lease was released."""

    code = "GUARD_TIMEOUT"


__all__ = [
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
