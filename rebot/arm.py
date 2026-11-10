"""``rebot.Arm``: the synchronous facade most programs use.

::

    from rebot import Arm

    with Arm() as arm:                 # REBOT_HOST or 127.0.0.1; takes control, enables torque
        arm.home()
        for obj in arm.detect("cup"):
            if arm.grasp(obj).holding:
                arm.move_to_pose("bin_a")
                arm.gripper_open()
    # leaving the block parks the arm (torque off where armd allows it) and releases control,
    # also after an exception or Ctrl+C

Every method blocks until armd reports the result and raises a :class:`rebot.RebotError`
(``.code``, ``.message``) when something goes wrong. Units: metres, radians, seconds.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Union

from ._log import _log

from .client import ActionResult, ArmdClient, ArmSession, ArmStatus, ObservationSchema
from .detection import PIPELINES, Detection, DetectionClient, parse_detections
from .errors import (
    ActionFailed,
    ArmCommandRejected,
    ArmConnectionError,
    ArmFaultError,
    GuardTimeout,
    RebotError,
    UnsupportedError,
)

DEFAULT_PORT = 8765
DEFAULT_OBS_HZ = 10

_HINTS = {
    "IK_UNREACHABLE": "is out of reach; choose a closer or higher target",
    "IK_BRANCH_FLIP": "needs the arm to flip its elbow; choose a closer or higher target",
    "WORKSPACE_VIOLATION": "is outside the allowed workspace",
    "NOT_ARMED": "cannot be reached because torque is off; call arm.enable()",
    "MOTION_BUSY": "cannot start while another motion runs; call arm.stop() first",
    "ESTOP_ACTIVE": "cannot be reached while the E-stop is latched",
}


def _base_url(host: Optional[str], port: Optional[int]) -> str:
    host = (host or os.environ.get("REBOT_HOST") or "127.0.0.1").strip()
    if "://" in host:
        return host.rstrip("/")
    if port is None and host.count(":") == 1:  # "host:port"
        return f"http://{host}"
    return f"http://{host}:{port or DEFAULT_PORT}"


def _fmt(xyz: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(v):g}" for v in xyz) + ")"


class Arm:
    """One program's control of the arm.

    ``host``: armd address (``"192.168.42.1"``, ``"host:port"`` or a URL); default ``$REBOT_HOST``
    or ``127.0.0.1`` (on the reCamera itself). ``enable``: turn torque on after connecting.
    ``park_on_exit``: on :meth:`close` (end of the ``with`` block) park the arm with torque off.
    ``force``: take control even if the web cockpit holds it (armd still applies its cooldown).
    ``obs_hz``: observations per second from armd (used only when armd supports ``obs_rate``).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        enable: bool = True,
        park_on_exit: bool = True,
        force: bool = False,
        obs_hz: Optional[int] = DEFAULT_OBS_HZ,
        client_id: Optional[str] = None,
        timeout: float = 5.0,
        transport: Any = None,
    ) -> None:
        self.url = _base_url(host, port)
        self.park_on_exit = park_on_exit
        self.client = ArmdClient(self.url, timeout_sec=timeout, transport=transport)
        self.detector = DetectionClient(self.url, transport=transport)
        self.exit_outcome: Optional[str] = None
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._guard_timeout: Optional[float] = None
        self._last_feed = time.monotonic()
        self._guard_tripped = False
        self._guard_stop = threading.Event()
        self._guard_thread: Optional[threading.Thread] = None
        try:
            self.schema: ObservationSchema = self.client.get_schema()
        except RebotError as exc:
            raise ArmConnectionError(
                f"cannot reach armd at {self.url} ({exc.message}); check the address and that rebot-armd is running"
            ) from exc
        hz = obs_hz if obs_hz and self.schema.has("obs_rate") else None
        self.session: ArmSession = self.client.session(
            client_id=client_id or f"rebot-{os.getpid()}", force=force, safe_exit="off", obs_hz=hz
        )
        self.session._schema = self.schema  # already fetched
        self.session.open()
        atexit.register(self.close)
        try:
            if enable:
                self.enable()
        except BaseException:
            self.close()
            raise

    # ---- lifecycle -----------------------------------------------------------

    def __enter__(self) -> Arm:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Park (when ``park_on_exit``) and release control. Safe to call more than once."""
        atexit.unregister(self.close)
        self._guard_stop.set()
        s = self.session
        if s._ws is None:
            return
        try:
            if self.park_on_exit and not self._guard_tripped and s.has_lease:
                self.exit_outcome = self._park_for_exit()
                (_log.info if self.exit_outcome.startswith("parked") else _log.warning)("exit: %s", self.exit_outcome)
        except BaseException as exc:  # never mask the program's own exception
            self.exit_outcome = f"failed: {exc}"
            _log.warning("exit: park failed: %s", exc)
        finally:
            s.close()

    def _park_for_exit(self) -> str:
        obs = self.session.wait_for_observation(timeout=2.0)
        if not obs.is_torqued:
            return "parked (torque already off)"
        if obs.is_stopped:
            if not (self.schema.has("fault_codes") and obs.recoverable):
                return f"left holding: {obs.fault_code or obs.safety_state} is not cleared automatically"
            self.recover()
        try:
            self.session.park_action(timeout=30.0)
        except ArmCommandRejected as exc:
            if exc.code == "UNSUPPORTED":
                return "left torqued: this armd has no resting pose to park at"
            raise
        return "parked (torque off)"

    # ---- loop guard ----------------------------------------------------------

    def guard(self, timeout_s: Optional[float]) -> None:
        """Release control if the program stops calling :meth:`feed` for ``timeout_s`` seconds.

        Protects against a hung main loop: the SDK's heartbeat thread would otherwise keep the
        lease alive. Time spent inside an ``Arm`` method counts as fed. After a trip every call
        raises :class:`GuardTimeout`; armd's lease watchdog handles the arm. ``None`` disables it.
        """
        self._guard_timeout = None if timeout_s is None else float(timeout_s)
        self.feed()
        if self._guard_timeout is not None and self._guard_thread is None:
            self._guard_thread = threading.Thread(target=self._guard_loop, name="rebot-guard", daemon=True)
            self._guard_thread.start()

    def feed(self) -> None:
        """Tell the guard the program is alive."""
        if self._guard_tripped:
            raise GuardTimeout("control was released because the program stopped calling arm.feed()")
        self._last_feed = time.monotonic()

    def _guard_loop(self) -> None:
        while not self._guard_stop.wait(0.05):
            limit = self._guard_timeout
            if limit is None or self._busy:
                continue
            if time.monotonic() - self._last_feed > limit:
                self._guard_tripped = True
                _log.error("guard: no arm.feed() for %.1f s; releasing control", limit)
                try:
                    self.session.close()
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning("guard: close failed: %s", exc)
                return

    def _run(self, fn: Callable[[], Any]) -> Any:
        if self._guard_tripped:
            raise GuardTimeout("control was released because the program stopped calling arm.feed()")
        with self._busy_lock:
            self._busy += 1
        try:
            return fn()
        except RebotError:
            if self._guard_tripped:
                raise GuardTimeout("control was released because the program stopped calling arm.feed()") from None
            raise
        finally:
            with self._busy_lock:
                self._busy -= 1
            self._last_feed = time.monotonic()

    # ---- state ---------------------------------------------------------------

    def state(self) -> ArmStatus:
        """Latest observation: ``positions``, ``safety_state``, ``is_torqued``, ``fault``, ``action``..."""
        return self._run(lambda: self.session.latest_observation() or self.session.wait_for_observation())

    @property
    def capabilities(self) -> dict:
        return dict(self.schema.capabilities)

    def _ready(self) -> None:
        """Before a motion: re-enable after a park; refuse in EStop/Fault with a hint."""
        obs = self.session.latest_observation() or self.session.wait_for_observation()
        if obs.is_stopped:
            hint = "call arm.recover()" if obs.recoverable else "check the arm, then clear it in the web cockpit"
            reason = f" ({obs.fault_reason})" if obs.fault_reason else ""
            raise ArmFaultError(obs, f"arm is in {obs.safety_state}{reason}; {hint}")
        if not obs.is_operational:
            self._enable()

    # ---- torque --------------------------------------------------------------

    def enable(self) -> None:
        """Torque on (holds the current position)."""
        self._run(self._enable)

    def _enable(self) -> None:
        obs = self.session.latest_observation() or self.session.wait_for_observation()
        if obs.is_operational:
            return
        if obs.is_stopped:
            raise ArmFaultError(
                obs, f"cannot enable in {obs.safety_state}; call arm.recover() or clear it in the web cockpit"
            )
        self.session.torque(True)
        self.session._wait_until(self.session._frames, lambda o: o.is_operational, 5.0, "torque on")

    # ---- motion --------------------------------------------------------------

    def _motion(self, fn: Callable[[], ActionResult], what: str) -> ActionResult:
        def go() -> ActionResult:
            self._ready()
            try:
                return fn()
            except ArmCommandRejected as exc:
                hint = _HINTS.get(exc.code)
                if hint:
                    raw = dict(exc.raw, message=f"{what} {hint} ({exc.message})")
                    raise type(exc)(raw) from None
                raise

        return self._run(go)

    def home(self, timeout: Optional[float] = None) -> ActionResult:
        """Move to armd's home position."""
        return self._motion(lambda: self.session.home_action(timeout=timeout), "home")

    def park(self, timeout: float = 30.0) -> ActionResult:
        """Move to the resting pose and turn torque off. The next motion turns torque on again."""
        return self._motion(lambda: self.session.park_action(timeout=timeout), "park")

    def move_to(
        self,
        xyz: Sequence[float],
        rpy: Optional[Sequence[float]] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> ActionResult:
        """Straight-line move of the gripper to ``xyz`` (metres, arm base frame: x forward, y left, z up).

        ``rpy``: orientation in radians (roll, pitch, yaw); None keeps the current one.
        ``speed``: m/s (armd default 0.05).
        """
        return self._motion(
            lambda: self.session.move_pose(xyz, rpy=rpy, linear_speed=speed, timeout=timeout),
            f"target {_fmt(xyz)}",
        )

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        frame: str = "base",
        droll: Optional[float] = None,
        dpitch: Optional[float] = None,
        dyaw: Optional[float] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> ActionResult:
        """Move by ``(dx, dy, dz)`` metres in ``frame`` (``base``, ``tool`` or ``camera``); rotations in radians."""
        return self._motion(
            lambda: self.session.move_relative(
                frame, (dx, dy, dz), dyaw=dyaw, dpitch=dpitch, droll=droll, linear_speed=speed, timeout=timeout
            ),
            f"offset {_fmt((dx, dy, dz))} ({frame} frame)",
        )

    def move_joints(self, positions: Sequence[float], duration: Optional[float] = None) -> ArmStatus:
        """Joint-space move to ``positions`` (radians, one per joint including the gripper)."""

        def go() -> ArmStatus:
            self._ready()
            return self.session.move_to(positions, duration=duration)  # type: ignore[return-value]

        return self._run(go)

    def gripper_open(self) -> ActionResult:
        return self._motion(lambda: self.session.gripper("open"), "gripper open")

    def gripper_close(self, grasp: bool = True) -> ActionResult:
        """Close the gripper; the result's ``holding`` tells whether something is held."""
        return self._motion(lambda: self.session.gripper("close", grasp=grasp), "gripper close")

    def gripper_move(self, fraction: float) -> ActionResult:
        """Open the gripper to ``fraction`` (0 closed .. 1 open)."""
        return self._motion(lambda: self.session.gripper("move", fraction=fraction), "gripper move")

    def grasp(
        self,
        target: Union[Detection, str],
        box: Optional[Sequence[float]] = None,
        timeout: Optional[float] = None,
    ) -> ActionResult:
        """Pick up a detected object: ``arm.grasp(detection)`` or ``arm.grasp("cup", box=(x0, y0, x1, y1))``.

        armd centres the camera on the box, descends, closes and lifts. ``result.holding`` is True
        when the object is in the gripper. A lost target raises :class:`ActionFailed`
        (``TARGET_LOST``, ``NOT_HOLDING``, ...).
        """
        if isinstance(target, Detection):
            label, box_norm, frame_id, ts_ms = target.label, target.box_norm, target.frame_id, target.ts_ms
        else:
            if box is None:
                raise ValueError("grasp(label) needs box=(x0, y0, x1, y1); or pass a detection from arm.detect()")
            label, box_norm, frame_id, ts_ms = str(target), tuple(box), None, None
        caps = self.schema.capabilities
        if caps and caps.get("grasp") is False:
            raise UnsupportedError(
                "this armd has no grasp action (needs a profile with kinematics, gripper and grasp setup)"
            )
        return self._motion(
            lambda: self.session.grasp(label, box_norm, frame_id=frame_id, ts_ms=ts_ms, timeout=timeout),
            f"grasp of '{label}'",
        )

    def stop(self) -> None:
        """Stop the current motion and hold position (torque stays on)."""
        self._run(self.session.stop)

    def e_stop(self) -> None:
        """Emergency stop (latched). Clear it in the web cockpit after checking the arm."""
        try:
            self.session.e_stop()
        except RebotError:
            self.client.e_stop()

    # ---- named poses (armd capability ``poses``) -----------------------------

    def poses(self) -> List[dict]:
        """Saved poses: ``[{"name", "positions", "tcp", "saved_at_ms"}]``."""
        return self._run(self.client.get_poses)

    def save_pose(self, name: str, positions: Optional[Sequence[float]] = None) -> None:
        """Save the current joint positions (or ``positions``) under ``name`` (letters, digits, _ and -)."""
        self._run(lambda: self.session.save_pose(name, positions))

    def delete_pose(self, name: str) -> None:
        self._run(lambda: self.session.delete_pose(name))

    def move_to_pose(self, name: str, speed_scale: float = 1.0, timeout: Optional[float] = None) -> ActionResult:
        """Joint-space move to a saved pose; ``speed_scale`` 0.1..1.0."""
        return self._motion(
            lambda: self.session.move_to_pose(name, speed_scale=speed_scale, timeout=timeout), f"pose '{name}'"
        )

    # ---- vision --------------------------------------------------------------

    def detect(
        self,
        label: Optional[str] = None,
        pipelines: Sequence[str] = ("yolo",),
        min_score: float = 0.3,
        wait_ms: int = 2500,
    ) -> List[Detection]:
        """Run the camera detector once; detections best first, filtered by ``label`` (case-insensitive).

        Each has ``label``, ``score`` and ``box_norm`` (x0, y0, x1, y1 in 0..1). Returns ``[]``
        when nothing matches or no fresh frame arrived within ``wait_ms``.
        """
        for p in pipelines:
            if p not in PIPELINES:
                raise ValueError(f"pipelines must be from {list(PIPELINES)}")

        def go() -> List[Detection]:
            res = self.detector.detect_once(wait_ms=wait_ms, pipelines=list(pipelines))
            out = []
            for d in parse_detections(res["raw"]):
                if d.score < min_score or (label is not None and d.label.lower() != label.lower()):
                    continue
                d.frame_id, d.ts_ms = res.get("frame_id"), res.get("ts_ms")
                out.append(d)
            return sorted(out, key=lambda d: -d.score)

        return self._run(go)

    # ---- faults --------------------------------------------------------------

    def recover(self, max_attempts: int = 1) -> ArmStatus:
        """Clear a fault that armd marks recoverable, then turn torque on again.

        Clears only when the observation's ``fault.recoverable`` is true (armd with fault codes).
        E-stops, faults armd marks not recoverable, and faults on an armd without fault codes
        raise :class:`ArmFaultError`: check the arm and clear them in the web cockpit.
        """
        return self._run(lambda: self._recover(max_attempts))

    def _recover(self, max_attempts: int) -> ArmStatus:
        obs = self.session.wait_for_observation(timeout=2.0)
        if not obs.is_stopped:
            return obs
        where = "; check the arm, then clear it in the web cockpit"
        if not self.schema.has("fault_codes"):
            raise ArmFaultError(
                obs,
                f"arm is in {obs.safety_state}; this armd reports no fault codes, "
                f"so faults are not cleared automatically{where}",
            )
        if not obs.recoverable:
            msg = (obs.fault or {}).get("message") or obs.safety_state
            raise ArmFaultError(obs, f"{msg} (not recoverable automatically){where}")
        last: Optional[RebotError] = None
        for _ in range(max(1, int(max_attempts))):
            try:
                self.session.clear_fault()
                self.session._wait_until(
                    self.session._frames,
                    lambda o: not o.is_stopped and o.feedback_valid,
                    3.0,
                    "fresh feedback after clear_fault",
                    check_stopped=False,
                )
                self.session.torque(True)
                return self.session._wait_until(self.session._frames, lambda o: o.is_operational, 5.0, "torque on")
            except (ArmCommandRejected, ActionFailed, ArmFaultError, ArmConnectionError) as exc:
                last = exc
                obs = self.session.latest_observation() or obs
                if obs.is_stopped and not obs.recoverable:
                    break
            except RebotError as exc:
                last = exc
        raise ArmFaultError(obs, f"recovery failed after {max_attempts} attempt(s): {last}{where}")


__all__ = ["Arm"]
