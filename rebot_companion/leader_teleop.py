"""PC-side USB Leader bridge (Feetech STS3215 bus) -> armd WebSocket servo streaming.

Connects to a 6-DOF Feetech leader arm (5 arm joints + 1 gripper) over USB serial
(/dev/ttyACM*, /dev/ttyUSB*, /dev/cu.usbmodem*) and streams servo frames to armd
at 50 Hz.

Zero new runtime dependencies: uses Python stdlib POSIX termios/tty/select, with
optional pyserial fast-path if installed.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import select
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rebot._http import host_url
from rebot.client import ArmdClient, ArmSession, ObservationSchema
from rebot.errors import ArmConnectionError, LeaseError, RebotError, ServoRejected

log = logging.getLogger("rebot_companion.leader_teleop")

DEFAULT_FPS = 50
DEFAULT_BAUD = 1000000
DEFAULT_MAX_GAP_RAD = 0.5
DEFAULT_RAMP_SECONDS = 2.0
ARMD_PORT = 8765
TICKS_PER_REV = 4096
DEFAULT_CENTER_TICK = 2048
DEFAULT_WRIST_ROLL_LIMITS = (-2.74, 2.74)

# Feetech STS3215 packet constants
HEADER = b"\xff\xff"
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x83

REG_TORQUE_ENABLE = 40
REG_PRESENT_POSITION = 56


class LeaderError(RuntimeError):
    """The leader teleop bridge encountered an error."""


# --------------------------------------------------------------------------- packet codec


def compute_checksum(data: Sequence[int] | bytes) -> int:
    return (~sum(data)) & 0xFF


def encode_packet(servo_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    length = len(params) + 2
    body = [servo_id, length, instruction] + list(params)
    checksum = compute_checksum(body)
    return bytes([0xFF, 0xFF] + body + [checksum])


def decode_packet(buffer: bytes) -> tuple[int | None, int | None, bytes | None, bytes]:
    """Decode the first Feetech STS response packet from buffer.

    Returns (servo_id, error, params, remaining_buffer).
    """
    idx = buffer.find(HEADER)
    if idx < 0:
        return None, None, None, b""
    if len(buffer) < idx + 6:
        return None, None, None, buffer[idx:]

    length = buffer[idx + 3]
    total_len = 4 + length
    if len(buffer) < idx + total_len:
        return None, None, None, buffer[idx:]

    packet = buffer[idx : idx + total_len]
    expected_checksum = compute_checksum(packet[2 : 3 + length])
    actual_checksum = packet[3 + length]
    if expected_checksum != actual_checksum:
        log.warning("checksum mismatch: expected 0x%02X, got 0x%02X", expected_checksum, actual_checksum)
        return None, None, None, buffer[idx + 2 :]

    servo_id = packet[2]
    error = packet[4]
    params = packet[5 : 3 + length]
    rem = buffer[idx + total_len :]
    return servo_id, error, bytes(params), rem


# --------------------------------------------------------------------------- serial transport


class LeaderTransport:
    """Abstract serial transport for leader bus communication."""

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class PosixSerialTransport(LeaderTransport):
    """Stdlib POSIX serial transport via termios and select."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD) -> None:
        import termios
        import tty

        self.port = port
        self.baud = baud
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            mode = termios.tcgetattr(self.fd)
            mode[tty.IFLAG] = 0
            mode[tty.OFLAG] = 0
            mode[tty.CFLAG] = termios.CS8 | termios.CREAD | termios.CLOCAL
            mode[tty.LFLAG] = 0
            mode[tty.CC][termios.VMIN] = 0
            mode[tty.CC][termios.VTIME] = 0

            baud_const = getattr(termios, f"B{baud}", None)
            if baud_const is not None:
                mode[tty.ISPEED] = baud_const
                mode[tty.OSPEED] = baud_const
                termios.tcsetattr(self.fd, termios.TCSANOW, mode)
            else:
                termios.tcsetattr(self.fd, termios.TCSANOW, mode)
                self._set_custom_baud_linux(baud)
        except Exception:
            os.close(self.fd)
            raise

    def _set_custom_baud_linux(self, baud: int) -> None:
        import fcntl
        import struct
        import sys

        if sys.platform != "linux":
            return
        TCGETS2 = 0x802C542A
        TCSETS2 = 0x402C542B
        BOTHER = 0x1000
        buf = bytearray(44)
        try:
            fcntl.ioctl(self.fd, TCGETS2, buf)
            c_cflag = struct.unpack_from("I", buf, 8)[0]
            c_cflag &= ~0o010017
            c_cflag |= BOTHER
            struct.pack_into("I", buf, 8, c_cflag)
            struct.pack_into("I", buf, 36, baud)
            struct.pack_into("I", buf, 40, baud)
            fcntl.ioctl(self.fd, TCSETS2, buf)
        except Exception as exc:
            log.warning("failed to set custom baud %d via termios2 (%s)", baud, exc)

    def write(self, data: bytes) -> None:
        os.write(self.fd, data)

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        out = bytearray()
        deadline = time.monotonic() + timeout
        while len(out) < n:
            rem = max(0.0, deadline - time.monotonic())
            r, _, _ = select.select([self.fd], [], [], rem)
            if not r:
                break
            chunk = os.read(self.fd, n - len(out))
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

    def flush(self) -> None:
        import termios

        try:
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except Exception:
            pass

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = -1


class PySerialTransport(LeaderTransport):
    """Pyserial transport wrapper when pyserial is installed."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = 0.05) -> None:
        import serial

        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        self.ser.timeout = timeout
        return self.ser.read(n)

    def flush(self) -> None:
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self) -> None:
        self.ser.close()


def open_serial_transport(port: str, baud: int = DEFAULT_BAUD) -> LeaderTransport:
    try:
        import serial  # noqa: F401

        return PySerialTransport(port, baud=baud)
    except ImportError:
        return PosixSerialTransport(port, baud=baud)


def auto_detect_leader_port() -> str:
    """Find a connected Feetech USB leader device on Linux or macOS."""
    candidates = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/cu.usbmodem*")
    if candidates:
        return sorted(candidates)[0]
    return "/dev/ttyACM0"


# --------------------------------------------------------------------------- bus & codec


class FeetechSerialBus:
    """High-level STS3215 packet codec and communication manager."""

    def __init__(self, transport: LeaderTransport) -> None:
        self.transport = transport
        self.dropouts = 0
        self.retries = 0

    def disable_torque(self, servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> None:
        """Disable torque on leader servos so the operator can backdrive them freely."""
        # Broadcast write reg 40 = 0
        self.transport.write(encode_packet(0xFE, INST_WRITE, [REG_TORQUE_ENABLE, 0]))
        for sid in servo_ids:
            self.transport.write(encode_packet(sid, INST_WRITE, [REG_TORQUE_ENABLE, 0]))

    def read_positions_sequential(
        self, servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6), max_retries: int = 1
    ) -> dict[int, int]:
        positions: dict[int, int] = {}
        for sid in servo_ids:
            val = None
            for attempt in range(max_retries + 1):
                self.transport.flush()
                pkt = encode_packet(sid, INST_READ, [REG_PRESENT_POSITION, 2])
                self.transport.write(pkt)
                resp = self.transport.read(8, timeout=0.03)
                s_id, err, params, _ = decode_packet(resp)
                if s_id == sid and err == 0 and params is not None and len(params) >= 2:
                    val = params[0] | (params[1] << 8)
                    break
                if attempt > 0:
                    self.retries += 1
            if val is not None:
                positions[sid] = val
            else:
                self.dropouts += 1
        return positions

    def read_positions_sync(
        self, servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6), timeout: float = 0.05
    ) -> dict[int, int]:
        self.transport.flush()
        pkt = encode_packet(0xFE, INST_SYNC_READ, [REG_PRESENT_POSITION, 2, *servo_ids])
        self.transport.write(pkt)
        expected_len = len(servo_ids) * 8
        buf = self.transport.read(expected_len, timeout=timeout)
        positions: dict[int, int] = {}
        rem = buf
        while rem:
            prev_rem = rem
            sid, err, params, rem = decode_packet(rem)
            if sid is not None and err == 0 and params is not None and len(params) >= 2:
                positions[sid] = params[0] | (params[1] << 8)
            elif sid is None and (not rem or rem == prev_rem):
                break
        if len(positions) == len(servo_ids):
            return positions
        # Fallback to sequential read if sync read missed any
        seq = self.read_positions_sequential(servo_ids)
        positions.update(seq)
        return positions


# --------------------------------------------------------------------------- calibration


@dataclass
class JointCalib:
    center: int = DEFAULT_CENTER_TICK
    sign: int = 1


@dataclass
class GripperCalib:
    closed_tick: int = 2048
    open_tick: int = 3072


class LeaderCalibration:
    """Converts 6-DOF leader raw ticks to follower radians."""

    def __init__(
        self,
        joints: Sequence[JointCalib] | None = None,
        gripper: GripperCalib | None = None,
        servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
    ) -> None:
        self.servo_ids = list(servo_ids)
        self.joints = list(joints or [JointCalib() for _ in range(5)])
        self.gripper = gripper or GripperCalib()

    @classmethod
    def load(cls, path_or_dict: str | Path | dict | None) -> LeaderCalibration:
        if path_or_dict is None:
            return cls()
        if isinstance(path_or_dict, dict):
            d = path_or_dict
        else:
            p = Path(path_or_dict).expanduser()
            if not p.is_file():
                raise LeaderError(f"calibration file {p} not found")
            content = p.read_text(encoding="utf-8")
            try:
                d = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml  # optional dependency

                    d = yaml.safe_load(content)
                except Exception as exc:
                    raise LeaderError(f"calibration file {p} is not valid JSON or YAML: {exc}") from exc

        joints: list[JointCalib] = []
        if "joints" in d and isinstance(d["joints"], list):
            for j in d["joints"]:
                c = int(j.get("center", j.get("homing_offset", DEFAULT_CENTER_TICK)))
                s = int(j.get("sign", j.get("drive_mode", 1)))
                joints.append(JointCalib(center=c, sign=s))
        else:
            for i in range(5):
                name = f"joint_{i + 1}"
                if name in d:
                    entry = d[name]
                    c = int(entry.get("center", entry.get("homing_offset", DEFAULT_CENTER_TICK)))
                    s = int(entry.get("sign", entry.get("drive_mode", 1)))
                    joints.append(JointCalib(center=c, sign=s))
                else:
                    joints.append(JointCalib())

        grip_cfg = d.get("gripper") or {}
        g_closed = int(grip_cfg.get("closed_tick", grip_cfg.get("closed", 2048)))
        g_open = int(grip_cfg.get("open_tick", grip_cfg.get("open", 3072)))
        gripper = GripperCalib(closed_tick=g_closed, open_tick=g_open)

        servo_ids = d.get("servo_ids") or (1, 2, 3, 4, 5, 6)
        return cls(joints=joints, gripper=gripper, servo_ids=servo_ids)

    def ticks_to_rad(
        self,
        ticks: dict[int, int],
        follower_limits: Sequence[tuple[float, float]] | None = None,
        follower_closed_rad: float = 0.0,
        follower_open_rad: float = 1.5,
    ) -> list[float]:
        """Map raw ticks to 6 follower URDF radians [j1, j2, j3, j4, j5, gripper]."""
        out: list[float] = []
        # Arm joints 1..5 (servos 1..5)
        for i in range(5):
            sid = self.servo_ids[i]
            tick = ticks.get(sid, self.joints[i].center)
            cal = self.joints[i]
            rad = cal.sign * (tick - cal.center) * (2.0 * math.pi / TICKS_PER_REV)
            if i == 4:  # wrist_roll
                rad = min(max(rad, DEFAULT_WRIST_ROLL_LIMITS[0]), DEFAULT_WRIST_ROLL_LIMITS[1])
            if follower_limits is not None and i < len(follower_limits):
                lo, hi = follower_limits[i]
                rad = min(max(rad, lo), hi)
            out.append(float(rad))

        # Gripper (servo 6)
        grip_sid = self.servo_ids[5]
        g_tick = ticks.get(grip_sid, self.gripper.closed_tick)
        c_tick = self.gripper.closed_tick
        o_tick = self.gripper.open_tick
        if o_tick == c_tick:
            u = 0.0
        else:
            lo_t, hi_t = (c_tick, o_tick) if c_tick <= o_tick else (o_tick, c_tick)
            t_clamped = min(max(float(g_tick), lo_t), hi_t)
            u = (t_clamped - c_tick) / (o_tick - c_tick)
        u = min(max(u, 0.0), 1.0)
        grip_rad = follower_closed_rad + u * (follower_open_rad - follower_closed_rad)
        if follower_limits is not None and len(follower_limits) > 5:
            lo, hi = follower_limits[5]
            lo_lim, hi_lim = (lo, hi) if lo <= hi else (hi, lo)
            grip_rad = min(max(grip_rad, lo_lim), hi_lim)
        out.append(float(grip_rad))

        return out


# --------------------------------------------------------------------------- leader bridge


class LeaderTeleopBridge:
    """Manages the lifecycle of reading leader ticks and streaming servo commands to armd."""

    def __init__(
        self,
        bus: FeetechSerialBus,
        session: ArmSession,
        schema: ObservationSchema,
        *,
        calib: LeaderCalibration | None = None,
        fps: int = DEFAULT_FPS,
        ramp_seconds: float = DEFAULT_RAMP_SECONDS,
        max_gap_rad: float = DEFAULT_MAX_GAP_RAD,
        operator_present: bool = False,
        park_on_exit: bool = True,
    ) -> None:
        self.bus = bus
        self.session = session
        self.schema = schema
        self.calib = calib or LeaderCalibration()
        self.fps = fps
        self.ramp_seconds = ramp_seconds
        self.max_gap_rad = max_gap_rad
        self.operator_present = operator_present
        self.park_on_exit = park_on_exit

        self.limits: list[tuple[float, float]] = [
            (float(lim.min_position), float(lim.max_position)) for lim in schema.limits
        ]
        self.closed_rad = 0.0
        self.open_rad = 1.5
        policy = schema.gripper_policy or {}
        if "closed_rad" in policy and "open_rad" in policy:
            self.closed_rad = float(policy["closed_rad"])
            self.open_rad = float(policy["open_rad"])

        self._stop = False
        self.frames_sent = 0

    def stop(self) -> None:
        self._stop = True

    def read_leader_target(self) -> list[float]:
        ticks = self.bus.read_positions_sync(self.calib.servo_ids)
        missing = [sid for sid in self.calib.servo_ids if sid not in ticks]
        if missing:
            raise LeaderError(f"missing leader readings for servo IDs {missing}")
        return self.calib.ticks_to_rad(
            ticks,
            follower_limits=self.limits,
            follower_closed_rad=self.closed_rad,
            follower_open_rad=self.open_rad,
        )

    def align_and_ramp(self) -> None:
        """Verify alignment against the follower arm and perform a smooth quintic ramp-in."""
        obs = self.session.wait_for_observation(timeout=5.0)
        if obs.is_stopped:
            raise LeaderError(
                f"follower is in {obs.safety_state} ({obs.fault_code or obs.fault_reason}); "
                "check it and clear the fault in the cockpit"
            )
        if obs.collision and obs.collision.get("active"):
            raise LeaderError("follower collision hold active; wait until it resumes")

        follower_start = list(obs.positions[:6])
        leader_target = self.read_leader_target()
        if len(leader_target) != 6 or len(follower_start) != 6:
            raise LeaderError(
                f"expected 6 DOF for SO-ARM101 (got follower={len(follower_start)}, leader={len(leader_target)})"
            )

        gap = max(abs(ldr - flw) for ldr, flw in zip(leader_target, follower_start))
        log.info("initial alignment gap: %.3f rad (threshold: %.3f rad)", gap, self.max_gap_rad)
        if gap > self.max_gap_rad and not self.operator_present:
            raise LeaderError(
                f"alignment gap {gap:.3f} rad exceeds --max-gap-rad {self.max_gap_rad:.3f}; "
                "align the leader arm with the follower or pass --i-am-at-the-arm"
            )

        if not obs.is_torqued or not obs.is_operational:
            log.info("enabling follower torque before ramp ...")
            self.session.torque(True)
            deadline = time.monotonic() + 5.0
            while not (obs.is_torqued and obs.is_operational):
                rem = max(0.1, deadline - time.monotonic())
                if time.monotonic() >= deadline:
                    raise LeaderError("timed out waiting for follower to enable torque and become operational")
                obs = self.session.wait_for_observation(timeout=min(1.0, rem))
                if obs.is_stopped:
                    raise LeaderError(
                        f"follower is in {obs.safety_state} ({obs.fault_code or obs.fault_reason}); "
                        "check it and clear the fault in the cockpit"
                    )
            follower_start = list(obs.positions[:6])

        if self.ramp_seconds <= 0.0:
            return

        ramp_steps = max(1, int(round(self.ramp_seconds * self.fps)))
        period = 1.0 / self.fps
        t0 = time.monotonic()
        for step in range(1, ramp_steps + 1):
            if self._stop:
                break
            due = t0 + step * period
            s = step / ramp_steps
            # Quintic polynomial: 10s^3 - 15s^4 + 6s^5 (smooth acceleration & deceleration)
            alpha = 10.0 * (s**3) - 15.0 * (s**4) + 6.0 * (s**5)
            cur_leader = self.read_leader_target()
            interp = [(1.0 - alpha) * flw + alpha * ldr for flw, ldr in zip(follower_start, cur_leader)]
            self.session.servo(interp)
            self.frames_sent += 1
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)

    def run(self) -> None:
        """Stream servo frames continuously at fps."""
        self.bus.disable_torque(self.calib.servo_ids)
        self.align_and_ramp()

        period = 1.0 / self.fps
        t0 = time.monotonic()
        k = 0
        while not self._stop:
            due = t0 + k * period
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)

            # Check telemetry / safety
            obs = self.session.latest_observation()
            if obs:
                if obs.safety_state == "EStop":
                    log.error("follower latched E-stop")
                    break
                if obs.safety_state == "Fault":
                    log.error("follower latched Fault: %s", obs.fault_code or obs.fault_reason)
                    break
                if obs.collision and obs.collision.get("active"):
                    log.warning("follower collision hold active")
                    break

            try:
                target = self.read_leader_target()
                self.session.servo(target)
                self.frames_sent += 1
            except (LeaderError, ServoRejected, LeaseError, ArmConnectionError, RebotError, OSError) as exc:
                log.error("servo streaming stopped: %s", exc)
                break
            k += 1

    def close(self) -> None:
        """Clean shutdown: park and release lease."""
        try:
            if self.park_on_exit:
                log.info("parking follower arm ...")
                self.session.unload(wait=True, timeout=15.0)
        except Exception as exc:
            log.warning("park on exit failed: %s", exc)


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rebot-teleop-leader",
        description="Stream USB Leader arm (STS3215 bus) positions to rebot-armd via servo WebSocket.",
    )
    p.add_argument("--arm-host", default="192.168.16.1", help="armd host or URL (default 192.168.16.1)")
    p.add_argument(
        "--leader-port",
        help="serial port for leader arm (/dev/ttyACM*, /dev/ttyUSB*, /dev/cu.usbmodem*; auto-detected if omitted)",
    )
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"serial baud rate (default {DEFAULT_BAUD})")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"streaming rate Hz (default {DEFAULT_FPS})")
    p.add_argument("--calib", help="path to leader calibration file (JSON or YAML)")
    p.add_argument(
        "--i-am-at-the-arm",
        action="store_true",
        help="required when initial gap exceeds threshold: confirms operator is watching follower arm",
    )
    p.add_argument(
        "--max-gap-rad",
        type=float,
        default=DEFAULT_MAX_GAP_RAD,
        help=f"maximum allowed initial gap in radians (default {DEFAULT_MAX_GAP_RAD})",
    )
    p.add_argument(
        "--ramp-seconds",
        type=float,
        default=DEFAULT_RAMP_SECONDS,
        help=f"smooth ramp-in duration in seconds (default {DEFAULT_RAMP_SECONDS})",
    )
    p.add_argument("--no-park", action="store_true", help="leave follower in place instead of parking on exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    port = args.leader_port or auto_detect_leader_port()
    log.info("opening leader serial port %s at %d baud", port, args.baud)
    try:
        transport = open_serial_transport(port, baud=args.baud)
    except Exception as exc:
        print(f"rebot-teleop-leader: failed to open {port}: {exc}", file=sys.stderr)
        return 1

    bus = FeetechSerialBus(transport)
    client = ArmdClient(host_url("http", args.arm_host, ARMD_PORT))
    try:
        schema = client.get_schema()
    except Exception as exc:
        print(f"rebot-teleop-leader: failed to fetch armd schema: {exc}", file=sys.stderr)
        transport.close()
        return 1

    calib = LeaderCalibration.load(args.calib)
    session = ArmSession(client, safe_exit="off").open()
    bridge = LeaderTeleopBridge(
        bus,
        session,
        schema,
        calib=calib,
        fps=args.fps,
        ramp_seconds=args.ramp_seconds,
        max_gap_rad=args.max_gap_rad,
        operator_present=args.i_am_at_the_arm,
        park_on_exit=not args.no_park,
    )

    def _sig_handler(*_: Any) -> None:
        bridge.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        bridge.run()
    finally:
        bridge.close()
        session.close()
        transport.close()
        print(f"leader teleop closed: {bridge.frames_sent} frames sent, {bus.dropouts} dropouts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
