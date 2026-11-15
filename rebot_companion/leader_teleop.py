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
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rebot._http import host_url
from rebot.client import ArmdClient, ArmSession, ObservationSchema, SERVO_MIN_RATE_HZ
from rebot.errors import ArmConnectionError, LeaseError, RebotError, ServoRejected
from rebot_companion import units

log = logging.getLogger("rebot_companion.leader_teleop")

DEFAULT_FPS = 50
DEFAULT_BAUD = 1000000
DEFAULT_MAX_GAP_RAD = 0.5
HARD_MAX_GAP_RAD = math.pi
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
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83

REG_TORQUE_ENABLE = 40
REG_PRESENT_POSITION = 56


class LeaderError(RuntimeError):
    """The leader teleop bridge encountered an error."""


LeaderTeleopError = LeaderError


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
    if length < 2:
        return None, None, None, buffer[idx + 2 :]
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

    def drain(self) -> None:
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
                self._set_custom_baud(baud)
        except Exception:
            os.close(self.fd)
            raise

    def _set_custom_baud(self, baud: int) -> None:
        import fcntl
        import struct
        import sys

        if sys.platform == "linux":
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
                raise LeaderError(
                    f"failed to set custom baud {baud} on Linux for {self.port}: {exc}; install pyserial"
                ) from exc
        elif sys.platform == "darwin":
            IOSSIOSPEED = 0x80085402
            buf = struct.pack("L", baud)
            try:
                fcntl.ioctl(self.fd, IOSSIOSPEED, buf)
            except Exception as exc:
                raise LeaderError(
                    f"failed to set custom baud {baud} on macOS for {self.port}: {exc}; install pyserial"
                ) from exc
        else:
            raise LeaderError(
                f"custom baud {baud} is unsupported on {sys.platform} without pyserial; install pyserial"
            )

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        total = len(view)
        written = 0
        deadline = time.monotonic() + 1.0
        while written < total:
            _, w, _ = select.select([], [self.fd], [], 0.1)
            if not w:
                if time.monotonic() > deadline:
                    raise LeaderError(f"write timeout on serial port {self.port} ({written}/{total} bytes written)")
                continue
            try:
                n = os.write(self.fd, view[written:])
                if n <= 0:
                    raise LeaderError(f"write EOF on serial port {self.port}")
                written += n
            except BlockingIOError:
                continue
            except OSError as exc:
                raise LeaderError(f"write failed on serial port {self.port}: {exc}") from exc

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        out = bytearray()
        deadline = time.monotonic() + timeout
        while len(out) < n:
            rem = max(0.0, deadline - time.monotonic())
            r, _, _ = select.select([self.fd], [], [], rem)
            if not r:
                break
            try:
                chunk = os.read(self.fd, n - len(out))
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                continue
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

    def flush(self) -> None:
        import termios

        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except Exception:
            pass

    def drain(self) -> None:
        import termios

        if getattr(self, "fd", -1) >= 0:
            try:
                termios.tcdrain(self.fd)
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

        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=0.1)

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def read(self, n: int, timeout: float = 0.05) -> bytes:
        self.ser.timeout = timeout
        return self.ser.read(n)

    def flush(self) -> None:
        self.ser.reset_input_buffer()

    def drain(self) -> None:
        try:
            self.ser.flush()
        except Exception:
            pass

    def close(self) -> None:
        self.ser.close()


def open_serial_transport(port: str, baud: int = DEFAULT_BAUD) -> LeaderTransport:
    try:
        import serial  # noqa: F401

        return PySerialTransport(port, baud=baud)
    except ImportError:
        return PosixSerialTransport(port, baud=baud)


def auto_detect_leader_port(
    baud: int = DEFAULT_BAUD,
    servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> str:
    """Find a connected Feetech USB leader device on Linux or macOS by probing expected servo IDs."""
    candidates = sorted(
        set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/cu.usbmodem*"))
    )
    if not candidates:
        raise LeaderError(
            "no USB serial candidate ports found (/dev/ttyACM*, /dev/ttyUSB*, /dev/cu.usbmodem*)"
        )

    expected = set(servo_ids)
    for port in candidates:
        transport: LeaderTransport | None = None
        try:
            transport = open_serial_transport(port, baud=baud)
            bus = FeetechSerialBus(transport)
            responded = set(bus.ping_ids(servo_ids))
            if expected.issubset(responded):
                return port
        except Exception as exc:
            log.debug("probing port %s failed: %s", port, exc)
        finally:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    raise LeaderError(
        f"no leader arm found responding on expected servo IDs {list(servo_ids)} across candidate ports {candidates}"
    )


# --------------------------------------------------------------------------- bus & codec


class FeetechSerialBus:
    """High-level STS3215 packet codec and communication manager."""

    def __init__(self, transport: LeaderTransport) -> None:
        self.transport = transport
        self.dropouts = 0
        self.retries = 0

    def ping(self, servo_id: int, timeout: float = 0.02) -> bool:
        """Ping a servo ID and return True if it responds with a matching ID."""
        self.transport.flush()
        pkt = encode_packet(servo_id, INST_PING, [])
        self.transport.write(pkt)
        resp = self.transport.read(6, timeout=timeout)
        s_id, _err, _params, _rem = decode_packet(resp)
        return s_id == servo_id

    def ping_ids(
        self, servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6), timeout: float = 0.02
    ) -> list[int]:
        """Ping expected servo IDs and return the list of IDs that responded."""
        return [sid for sid in servo_ids if self.ping(sid, timeout=timeout)]

    def disable_torque(self, servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> None:
        """Disable torque on leader servos so the operator can backdrive them freely."""
        # Broadcast write reg 40 = 0
        self.transport.write(encode_packet(0xFE, INST_WRITE, [REG_TORQUE_ENABLE, 0]))
        for sid in servo_ids:
            self.transport.write(encode_packet(sid, INST_WRITE, [REG_TORQUE_ENABLE, 0]))
        self.transport.drain()

    def read_positions_sequential(
        self,
        servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        max_retries: int = 1,
        read_timeout: float = 0.03,
        deadline_s: float | None = None,
    ) -> dict[int, int]:
        positions: dict[int, int] = {}
        t_start = time.monotonic()
        for sid in servo_ids:
            if deadline_s is not None and (time.monotonic() - t_start) >= deadline_s:
                self.dropouts += 1
                continue
            val = None
            for attempt in range(max_retries + 1):
                if deadline_s is not None and (time.monotonic() - t_start) >= deadline_s:
                    break
                self.transport.flush()
                pkt = encode_packet(sid, INST_READ, [REG_PRESENT_POSITION, 2])
                self.transport.write(pkt)
                resp = self.transport.read(8, timeout=read_timeout)
                s_id, err, params, _ = decode_packet(resp)
                if s_id == sid and err == 0 and params is not None and len(params) >= 2:
                    raw = params[0] | (params[1] << 8)
                    mag = raw & 0x7FFF
                    val = -mag if (raw & 0x8000) else mag
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
                raw = params[0] | (params[1] << 8)
                mag = raw & 0x7FFF
                positions[sid] = -mag if (raw & 0x8000) else mag
            elif sid is None and (not rem or rem == prev_rem):
                break
        if len(positions) == len(servo_ids):
            return positions
        # Fallback to sequential read for missing IDs only
        missing = [sid for sid in servo_ids if sid not in positions]
        seq = self.read_positions_sequential(missing, max_retries=0, read_timeout=0.015, deadline_s=0.08)
        positions.update(seq)
        return positions


# --------------------------------------------------------------------------- calibration


@dataclass
class JointCalib:
    center: int = DEFAULT_CENTER_TICK
    sign: int = 1
    explicit_sign: bool = False
    explicit_center: bool = False
    defaulted: bool = False


@dataclass
class GripperCalib:
    closed_tick: int = 2048
    open_tick: int = 3072


def _parse_joint_calib(entry: dict, name: str) -> JointCalib:
    if not isinstance(entry, dict):
        raise LeaderError(f"{name}: joint calibration entry must be a dict, got {type(entry).__name__}")

    explicit_center = False
    defaulted = False
    if "center" not in entry:
        if "homing_offset" in entry:
            ho = entry["homing_offset"]
            try:
                int(ho)
            except (ValueError, TypeError) as exc:
                raise LeaderError(f"{name}: invalid 'homing_offset' {ho!r}") from exc
            center = DEFAULT_CENTER_TICK
            explicit_center = True
            log.info(
                "%s: found 'homing_offset' (%s) without 'center'; assuming homing_offset was written to servo EEPROM and using center=%d",
                name,
                ho,
                DEFAULT_CENTER_TICK,
            )
        else:
            center = DEFAULT_CENTER_TICK
            defaulted = True
    else:
        try:
            center = int(entry["center"])
            explicit_center = True
        except (ValueError, TypeError) as exc:
            raise LeaderError(f"{name}: invalid 'center' tick {entry.get('center')!r}") from exc

    if not (0 <= center <= 4095):
        raise LeaderError(f"{name}: center tick must be in [0, 4095], got {center}")

    explicit_sign = False
    if "sign" in entry:
        try:
            sign = int(entry["sign"])
            explicit_sign = True
        except (ValueError, TypeError) as exc:
            raise LeaderError(f"{name}: invalid 'sign' {entry.get('sign')!r}") from exc
        if sign not in (-1, 1):
            raise LeaderError(f"{name}: 'sign' must be 1 or -1, got {sign}")
    elif "drive_mode" in entry:
        dm = entry["drive_mode"]
        if dm in (0, False):
            sign = 1
            explicit_sign = True
        elif dm in (1, True):
            sign = -1
            explicit_sign = True
        else:
            raise LeaderError(f"{name}: invalid 'drive_mode' {dm!r}; must be 0 (normal) or 1 (inverted)")
    else:
        sign = 1

    return JointCalib(
        center=center,
        sign=sign,
        explicit_sign=explicit_sign,
        explicit_center=explicit_center,
        defaulted=defaulted,
    )


class LeaderCalibration:
    """Converts 6-DOF leader raw ticks to follower radians."""

    def __init__(
        self,
        joints: Sequence[JointCalib] | None = None,
        gripper: GripperCalib | None = None,
        servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        has_defaulted_joints: bool = False,
    ) -> None:
        self.servo_ids = [int(s) for s in servo_ids]
        if len(self.servo_ids) != 6:
            raise LeaderError(f"servo_ids must contain exactly 6 servo IDs, got {len(self.servo_ids)}")
        if len(set(self.servo_ids)) != 6 or any(not (1 <= s <= 253) for s in self.servo_ids):
            raise LeaderError(f"servo_ids must be 6 unique integers between 1 and 253, got {self.servo_ids}")

        self.joints = list(joints or [JointCalib(defaulted=True) for _ in range(5)])
        self.has_defaulted_joints = bool(
            has_defaulted_joints or joints is None or any(getattr(j, "defaulted", False) for j in self.joints)
        )
        if len(self.joints) != 5:
            raise LeaderError(f"joints must contain exactly 5 joint calibrations, got {len(self.joints)}")
        for i, j in enumerate(self.joints):
            if not isinstance(j, JointCalib):
                raise LeaderError(f"joint {i + 1} must be a JointCalib instance, got {type(j)}")
            if j.sign not in (-1, 1):
                raise LeaderError(f"joint {i + 1} sign must be -1 or 1, got {j.sign}")
            if not (0 <= j.center <= 4095):
                raise LeaderError(f"joint {i + 1} center must be in [0, 4095], got {j.center}")

        self.gripper = gripper or GripperCalib()
        if not (0 <= self.gripper.closed_tick <= 4095):
            raise LeaderError(f"gripper closed_tick must be in [0, 4095], got {self.gripper.closed_tick}")
        if not (0 <= self.gripper.open_tick <= 4095):
            raise LeaderError(f"gripper open_tick must be in [0, 4095], got {self.gripper.open_tick}")
        if self.gripper.closed_tick == self.gripper.open_tick:
            raise LeaderError("gripper closed_tick and open_tick cannot be equal")

    @classmethod
    def load(cls, path_or_dict: str | Path | dict | None) -> LeaderCalibration:
        if path_or_dict is None:
            return cls(has_defaulted_joints=True)
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
        has_defaulted = False
        SO101_JOINT_ALIASES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        RECOGNIZED_TOP_KEYS = {
            "joints",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            *SO101_JOINT_ALIASES,
            "gripper",
            "servo_ids",
        }
        if not any(k in d for k in RECOGNIZED_TOP_KEYS):
            raise LeaderError(
                f"calibration data contains no recognized keys ({sorted(RECOGNIZED_TOP_KEYS)})"
            )

        if "joints" in d:
            if not isinstance(d["joints"], list):
                raise LeaderError("'joints' must be a list of 5 joint calibrations")
            if len(d["joints"]) != 5:
                raise LeaderError(f"'joints' list must contain exactly 5 joints, got {len(d['joints'])}")
            joints = [_parse_joint_calib(j, f"joint_{i + 1}") for i, j in enumerate(d["joints"])]
            if any(getattr(j, "defaulted", False) for j in joints):
                has_defaulted = True
        else:
            for i in range(5):
                name = f"joint_{i + 1}"
                alias = SO101_JOINT_ALIASES[i]
                if name in d:
                    entry = d[name]
                    j_cal = _parse_joint_calib(entry, name)
                    joints.append(j_cal)
                    if getattr(j_cal, "defaulted", False):
                        has_defaulted = True
                elif alias in d:
                    entry = d[alias]
                    j_cal = _parse_joint_calib(entry, alias)
                    joints.append(j_cal)
                    if getattr(j_cal, "defaulted", False):
                        has_defaulted = True
                else:
                    joints.append(JointCalib(defaulted=True))
                    has_defaulted = True

        grip_cfg = d.get("gripper") or {}
        g_closed = None
        g_open = None
        if "closed_tick" in grip_cfg or "closed" in grip_cfg:
            try:
                g_closed = int(grip_cfg.get("closed_tick", grip_cfg.get("closed")))
            except (ValueError, TypeError) as exc:
                raise LeaderError(f"invalid gripper closed_tick: {exc}") from exc
        if "open_tick" in grip_cfg or "open" in grip_cfg:
            try:
                g_open = int(grip_cfg.get("open_tick", grip_cfg.get("open")))
            except (ValueError, TypeError) as exc:
                raise LeaderError(f"invalid gripper open_tick: {exc}") from exc
        if g_closed is None or g_open is None:
            # Support LeRobot gripper keys range_min / range_max (with optional drive_mode)
            if "range_min" in grip_cfg and "range_max" in grip_cfg:
                try:
                    rmin = int(grip_cfg["range_min"])
                    rmax = int(grip_cfg["range_max"])
                except (ValueError, TypeError) as exc:
                    raise LeaderError(f"invalid gripper range_min/range_max: {exc}") from exc
                g_dm = grip_cfg.get("drive_mode", 0)
                g_drive_mode = bool(g_dm in (1, True, "1"))
                if g_closed is None:
                    g_closed = rmax if g_drive_mode else rmin
                if g_open is None:
                    g_open = rmin if g_drive_mode else rmax
            else:
                if g_closed is None:
                    g_closed = 2048
                    has_defaulted = True
                if g_open is None:
                    g_open = 3072
                    has_defaulted = True

        if not (0 <= g_closed <= 4095) or not (0 <= g_open <= 4095) or g_closed == g_open:
            raise LeaderError(
                f"gripper closed_tick ({g_closed}) and open_tick ({g_open}) must be in 0..4095 and closed_tick != open_tick"
            )
        gripper = GripperCalib(closed_tick=g_closed, open_tick=g_open)

        servo_ids = d.get("servo_ids") or (1, 2, 3, 4, 5, 6)
        inst = cls(joints=joints, gripper=gripper, servo_ids=servo_ids, has_defaulted_joints=has_defaulted)
        return inst

    from_file = load

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
            # STS3215 represents positions using 12-bit magnitude [0, 4095] with bit 15 indicating sign.
            # Negative ticks in [-4095, 0] occur when homing_offset pushes the zero crossing past 0,
            # and map continuously to negative angles without jumping to +47 rad.
            # Any magnitude outside [-4095, 4095] indicates corrupt bus data and must be rejected.
            if not (-4095 <= tick <= 4095):
                raise LeaderError(
                    f"joint {i + 1} (servo {sid}) position tick out of range [-4095, 4095]: got {tick}"
                )
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
        if not (-4095 <= g_tick <= 4095):
            raise LeaderError(
                f"gripper (servo {grip_sid}) position tick out of range [-4095, 4095]: got {g_tick}"
            )
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


def _clamp_lead_and_step(
    val: float,
    prev_val: float,
    ref_val: float,
    max_step: float,
    max_lead: float,
    j_min: float | None = None,
    j_max: float | None = None,
    joint_index: int | None = None,
) -> float:
    """Clamp target angle within both previous command step limit and follower reference lead limit.

    Intersects [prev_val - max_step, prev_val + max_step], [ref_val - max_lead, ref_val + max_lead],
    and [j_min, j_max] if provided. Raises LeaderTeleopError if lead and step intervals are disjoint.
    """
    converted: dict[str, float] = {}
    for name, v in (
        ("val", val),
        ("prev_val", prev_val),
        ("ref_val", ref_val),
        ("max_step", max_step),
        ("max_lead", max_lead),
    ):
        if isinstance(v, bool):
            raise LeaderTeleopError(f"{name} must be numeric, got {v!r}")
        try:
            vf = float(v)
        except (ValueError, TypeError) as exc:
            raise LeaderTeleopError(f"{name} must be numeric, got {v!r}") from exc
        if not math.isfinite(vf):
            raise LeaderTeleopError(f"{name} must be finite, got {v!r}")
        converted[name] = vf

    val = converted["val"]
    prev_val = converted["prev_val"]
    ref_val = converted["ref_val"]
    max_step = converted["max_step"]
    max_lead = converted["max_lead"]

    if j_min is not None:
        if isinstance(j_min, bool):
            raise LeaderTeleopError(f"j_min must be numeric, got {j_min!r}")
        try:
            jf_min = float(j_min)
        except (ValueError, TypeError) as exc:
            raise LeaderTeleopError(f"j_min must be numeric, got {j_min!r}") from exc
        if not math.isfinite(jf_min):
            raise LeaderTeleopError(f"j_min must be finite, got {j_min!r}")
        j_min = jf_min

    if j_max is not None:
        if isinstance(j_max, bool):
            raise LeaderTeleopError(f"j_max must be numeric, got {j_max!r}")
        try:
            jf_max = float(j_max)
        except (ValueError, TypeError) as exc:
            raise LeaderTeleopError(f"j_max must be numeric, got {j_max!r}") from exc
        if not math.isfinite(jf_max):
            raise LeaderTeleopError(f"j_max must be finite, got {j_max!r}")
        j_max = jf_max

    lo = max(prev_val - max_step, ref_val - max_lead)
    hi = min(prev_val + max_step, ref_val + max_lead)
    if j_min is not None:
        lo = max(lo, j_min)
    if j_max is not None:
        hi = min(hi, j_max)

    if lo > hi + 1e-9:
        j_str = f"joint {joint_index + 1}: " if joint_index is not None else ""
        raise LeaderTeleopError(
            f"{j_str}disjoint lead and step intervals (prev_val={prev_val:.4f}, "
            f"ref_val={ref_val:.4f}, max_step={max_step:.4f}, max_lead={max_lead:.4f}, "
            f"lo={lo:.4f}, hi={hi:.4f}); follower moved away from command (pushed/blocked or tracking lag)"
        )
    if lo > hi:
        return 0.5 * (lo + hi)
    return min(max(val, lo), hi)


def _measured_ref(obs: Any) -> list[float]:
    """Extract reference positions preferring measured positions over commanded positions."""
    positions = getattr(obs, "positions", None)
    if positions is not None and len(positions) > 0:
        return [float(x) for x in positions]
    commanded = getattr(obs, "commanded_positions", None)
    if commanded is not None and len(commanded) > 0:
        return [float(x) for x in commanded]
    return []


def _validate_telemetry_ref(ref: Any) -> list[float]:
    """Validate follower reference positions: must be length 6 and all elements finite floats."""
    if not isinstance(ref, (list, tuple)) or len(ref) != 6:
        raise LeaderError(f"follower reference positions must have length 6, got {ref!r}")
    validated: list[float] = []
    for i, x in enumerate(ref):
        try:
            val = float(x)
        except (ValueError, TypeError) as exc:
            raise LeaderError(f"follower reference position[{i}] must be numeric, got {x!r}") from exc
        if not math.isfinite(val):
            raise LeaderError(f"follower reference position[{i}] must be finite, got {x!r}")
        validated.append(val)
    return validated


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
        follower: str | None = None,
        fps: int = DEFAULT_FPS,
        ramp_seconds: float = DEFAULT_RAMP_SECONDS,
        max_gap_rad: float = DEFAULT_MAX_GAP_RAD,
        operator_present: bool = False,
        allow_large_gap: bool = False,
        park_on_exit: bool = True,
    ) -> None:
        if not (isinstance(fps, (int, float)) and math.isfinite(fps) and SERVO_MIN_RATE_HZ <= fps <= 100):
            raise LeaderError(f"fps must be between {SERVO_MIN_RATE_HZ:g} and 100, got {fps!r}")
        if not (isinstance(max_gap_rad, (int, float)) and math.isfinite(max_gap_rad) and max_gap_rad > 0):
            raise LeaderError(f"max_gap_rad must be positive and finite, got {max_gap_rad!r}")
        if not (isinstance(ramp_seconds, (int, float)) and math.isfinite(ramp_seconds) and ramp_seconds >= 0.2):
            raise LeaderError(f"ramp_seconds must be >= 0.2, got {ramp_seconds!r}")

        if schema.dof != 6 or schema.arm_dof not in (5, None) or schema.gripper_joint != 5:
            raise LeaderError(
                f"expected 6 DOF for SO-ARM101 (schema.dof={schema.dof}, "
                f"schema.arm_dof={schema.arm_dof}, schema.gripper_joint={schema.gripper_joint})"
            )

        self.bus = bus
        self.session = session
        self.schema = schema
        self.calib = calib or LeaderCalibration()
        self.fps = fps
        self.ramp_seconds = ramp_seconds
        self.max_gap_rad = max_gap_rad
        self.operator_present = operator_present
        self.allow_large_gap = allow_large_gap
        self.park_on_exit = park_on_exit

        if follower is None:
            client = getattr(session, "client", None)
            if client is not None and hasattr(client, "get_safety_describe"):
                try:
                    describe = client.get_safety_describe()
                    if isinstance(describe, dict):
                        dev = describe.get("device")
                        if isinstance(dev, dict):
                            follower = dev.get("follower")
                except Exception:
                    pass

        if follower not in ("mock", "sim") and getattr(self.calib, "has_defaulted_joints", False):
            raise LeaderError(
                f"Leader teleop cannot use defaulted/uncalibrated joint settings on non-mock/sim follower {follower or 'hardware'!r}; "
                "calibrate all joints or run against mock/sim"
            )

        # Clone joints cleanly so we do not mutate caller's LeaderCalibration unexpectedly
        calib_joints = [
            JointCalib(
                center=j.center,
                sign=j.sign,
                explicit_sign=getattr(j, "explicit_sign", False),
                explicit_center=getattr(j, "explicit_center", False),
                defaulted=getattr(j, "defaulted", False),
            )
            for j in self.calib.joints
        ]
        self.calib = LeaderCalibration(
            joints=calib_joints,
            gripper=self.calib.gripper,
            servo_ids=self.calib.servo_ids,
            has_defaulted_joints=self.calib.has_defaulted_joints,
        )

        feetech_joints = getattr(schema, "feetech_joints", None)
        if isinstance(feetech_joints, Sequence) and len(feetech_joints) >= 5:
            for i in range(5):
                fj = feetech_joints[i]
                if not isinstance(fj, dict):
                    continue
                armd_sign = -1 if fj.get("drive_mode_invert") else 1
                cal = self.calib.joints[i]
                if getattr(cal, "explicit_sign", False):
                    if cal.sign != armd_sign:
                        raise LeaderError(
                            f"joint_{i + 1} direction mismatch: leader calibration specifies explicit sign={cal.sign}, "
                            f"but follower schema.feetech_joints[{i}] has drive_mode_invert={fj.get('drive_mode_invert')} (expected sign={armd_sign})"
                        )
                else:
                    cal.sign = armd_sign

                if getattr(cal, "defaulted", False):
                    zs = fj.get("zero_step")
                    if isinstance(zs, int) and 0 <= zs <= 4095:
                        cal.center = zs

        lead = getattr(schema, "servo_max_lead_rad", None)
        self.arm_lead = min(0.30, float(lead)) if isinstance(lead, (int, float)) and lead > 0 else 0.30
        self._lead_clamp_consecutive = 0

        self.limits: list[tuple[float, float]] = [
            (float(lim.min_position), float(lim.max_position)) for lim in schema.limits
        ]
        self.follower_limits = self.limits
        self.gripper_joint = schema.gripper_joint
        self.closed_rad, self.open_rad = units.resolve_schema_gripper_bounds(
            schema.gripper_policy, self.limits, self.gripper_joint
        )
        self.follower_gripper_open = self.open_rad
        self.follower_gripper_closed = self.closed_rad
        if getattr(session, "_schema", None) is None:
            if hasattr(session, "set_schema"):
                session.set_schema(schema)
            else:
                session._schema = schema

        self._stop = False
        self._activated = False
        self._parked = False
        self._error: BaseException | None = None
        self._last_servo_cmd: list[float] | None = None
        self.frames_sent = 0

        self._last_obs_timestamp: float | None = None
        self._last_obs_frames: int | None = None
        self._last_obs_time: float = time.monotonic()
        self._transient_disjoint_counts = [0] * 6

    def _check_follower_healthy(self, obs: Any, *, require_torqued: bool = True) -> None:
        if obs is None:
            raise LeaderError("no observation received from follower")
        if getattr(obs, "is_stopped", False) or getattr(obs, "safety_state", None) in ("EStop", "Fault"):
            state = getattr(obs, "safety_state", "Fault")
            reason = (obs.fault or {}).get("code") if isinstance(getattr(obs, "fault", None), dict) else getattr(obs, "fault_reason", None)
            raise LeaderError(
                f"follower is in {state} ({reason}); check it and clear the fault in the cockpit"
            )
        collision = getattr(obs, "collision", None)
        if collision and collision.get("active"):
            raise LeaderError("follower collision hold active; wait until it resumes")
        ml = getattr(obs, "motion_limit", None)
        if ml and ml.get("reason") in ("blocked", "torque"):
            raise LeaderError(f"follower motion limit active: {ml.get('reason')} on joint {ml.get('joint')}")
        if require_torqued:
            is_torqued = getattr(obs, "is_torqued", False)
            is_operational = getattr(obs, "is_operational", False)
            if not is_torqued or not is_operational:
                raise LeaderError(
                    f"follower is not torqued/operational (state {getattr(obs, 'safety_state', 'Unknown')}, "
                    f"torque {'on' if is_torqued else 'off'})"
                )
        lease_lost = getattr(self.session, "lease_lost", None)
        if isinstance(lease_lost, dict):
            raise LeaderError(f"follower lease lost: {lease_lost}")

        now = time.monotonic()
        ts = getattr(obs, "timestamp", None)
        if not isinstance(ts, (int, float)):
            ts = None
        frames = getattr(self.session, "_frames", None)
        if not isinstance(frames, int):
            frames = None
        if ts is not None or frames is not None:
            if self._last_obs_timestamp is None and self._last_obs_frames is None:
                self._last_obs_timestamp = ts
                self._last_obs_frames = frames
                self._last_obs_time = now
            elif (ts is not None and ts != self._last_obs_timestamp) or (
                frames is not None and frames != self._last_obs_frames
            ):
                self._last_obs_timestamp = ts
                self._last_obs_frames = frames
                self._last_obs_time = now
            else:
                stale_threshold = max(0.5, 5.0 / self.fps)
                if (now - self._last_obs_time) > stale_threshold:
                    raise LeaderError(
                        f"follower telemetry is stale (unchanged for {now - self._last_obs_time:.2f}s > {stale_threshold:.2f}s)"
                    )

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

    def _check_alignment_gap(
        self,
        leader_target: Sequence[float],
        follower_positions: Sequence[float],
        *,
        stage: str,
    ) -> float:
        if len(leader_target) != 6 or len(follower_positions) != 6:
            raise LeaderError(
                f"expected 6 DOF for SO-ARM101 ({stage}: got follower={len(follower_positions)}, leader={len(leader_target)})"
            )
        arm_gap = max(
            (abs(l - f) for i, (l, f) in enumerate(zip(leader_target, follower_positions)) if i != self.gripper_joint),
            default=0.0,
        )
        overall_gap = max(abs(ldr - flw) for ldr, flw in zip(leader_target, follower_positions))
        log.info(
            "%s alignment arm_gap: %.3f rad, overall_gap: %.3f rad (threshold: %.3f rad)",
            stage,
            arm_gap,
            overall_gap,
            self.max_gap_rad,
        )
        if arm_gap > HARD_MAX_GAP_RAD:
            raise LeaderError(
                f"{stage} alignment gap {arm_gap:.3f} rad exceeds hard limit {HARD_MAX_GAP_RAD:.3f} rad (pi); "
                "re-align leader arm before proceeding"
            )
        if arm_gap > self.max_gap_rad and not self.allow_large_gap:
            raise LeaderError(
                f"{stage} alignment gap {arm_gap:.3f} rad exceeds --max-gap-rad {self.max_gap_rad:.3f}; "
                "align the leader arm with the follower or pass --allow-large-gap"
            )
        return overall_gap

    def _arm_velocity_caps(self) -> list[float]:
        caps: list[float] = []
        limits = getattr(self.schema, "limits", None) or []
        for i, lim in enumerate(limits[:5]):
            val = getattr(lim, "max_velocity", None)
            if val is None and isinstance(lim, dict):
                val = lim.get("max_velocity")
            if val is not None:
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    raise LeaderTeleopError(
                        f"follower joint {i + 1} ({getattr(lim, 'name', f'j{i+1}')}) has non-numeric max_velocity: {val}"
                    )
                if not math.isfinite(v) or v <= 0:
                    raise LeaderTeleopError(
                        f"follower joint {i + 1} ({getattr(lim, 'name', f'j{i+1}')}) has non-finite or non-positive max_velocity: {val}"
                    )
                caps.append(v)
            else:
                caps.append(0.6)
        while len(caps) < 5:
            caps.append(0.6)
        return caps

    def _clamp_gripper_step_and_range(self, target_val: float, prev_val: float) -> float:
        """Clamp gripper target against per-frame max step and follower limits/bounds."""
        for name, v in (("target_val", target_val), ("prev_val", prev_val)):
            if isinstance(v, bool):
                raise LeaderError(f"gripper {name} must be numeric, got {v!r}")
            try:
                vf = float(v)
            except (ValueError, TypeError) as exc:
                raise LeaderError(f"gripper {name} must be numeric, got {v!r}") from exc
            if not math.isfinite(vf):
                raise LeaderError(f"gripper {name} must be finite, got {v!r}")

        max_gripper_step = max(
            0.30, 0.25 * abs(self.follower_gripper_open - self.follower_gripper_closed)
        )
        if isinstance(max_gripper_step, bool) or not math.isfinite(max_gripper_step) or max_gripper_step < 0:
            raise LeaderError(
                f"max_gripper_step must be non-negative finite float, got {max_gripper_step!r}"
            )

        follower_limits = getattr(self, "follower_limits", None) or self.limits
        gj = self.gripper_joint if self.gripper_joint is not None else 5
        if follower_limits and len(follower_limits) > gj:
            fl_lo, fl_hi = follower_limits[gj]
            g_lo, g_hi = (fl_lo, fl_hi) if fl_lo <= fl_hi else (fl_hi, fl_lo)
        else:
            g_lo = min(self.follower_gripper_open, self.follower_gripper_closed)
            g_hi = max(self.follower_gripper_open, self.follower_gripper_closed)
        g_lo = float(g_lo)
        g_hi = float(g_hi)

        step_lo = float(prev_val) - max_gripper_step
        step_hi = float(prev_val) + max_gripper_step
        eff_lo = max(step_lo, g_lo)
        eff_hi = min(step_hi, g_hi)
        if eff_lo > eff_hi + 1e-9:
            raise LeaderError(
                f"gripper position {float(prev_val):.4f} rad is too far from permitted range [{g_lo:.4f}, {g_hi:.4f}] "
                f"within step limit {max_gripper_step:.4f} rad"
            )
        if eff_lo > eff_hi:
            eff_lo = eff_hi = 0.5 * (eff_lo + eff_hi)
        clamped = min(max(float(target_val), eff_lo), eff_hi)
        return float(clamped)

    def _clamp_with_transient_droop_recovery(
        self,
        val: float,
        prev_val: float,
        ref_val: float,
        max_step: float,
        max_lead: float,
        j_min: float | None = None,
        j_max: float | None = None,
        joint_index: int = 0,
    ) -> float:
        """Clamp joint target using _clamp_lead_and_step with bounded transient droop recovery."""
        try:
            clamped = _clamp_lead_and_step(
                val,
                prev_val,
                ref_val,
                max_step,
                max_lead,
                j_min=j_min,
                j_max=j_max,
                joint_index=joint_index,
            )
            if 0 <= joint_index < len(self._transient_disjoint_counts):
                self._transient_disjoint_counts[joint_index] = 0
            return clamped
        except LeaderTeleopError:
            lead_lo = float(ref_val) - float(max_lead)
            lead_hi = float(ref_val) + float(max_lead)
            if j_min is not None:
                lead_lo = max(lead_lo, float(j_min))
            if j_max is not None:
                lead_hi = min(lead_hi, float(j_max))
            if lead_lo > lead_hi + 1e-9:
                raise

            lead_nearest = min(max(float(prev_val), lead_lo), lead_hi)
            gap_step = abs(lead_nearest - float(prev_val))
            safe_jump_cap = min(0.25, max(3.0 * float(max_step), 0.06))

            cur_count = (
                self._transient_disjoint_counts[joint_index]
                if 0 <= joint_index < len(self._transient_disjoint_counts)
                else 0
            )
            if gap_step <= safe_jump_cap and cur_count < 5:
                if 0 <= joint_index < len(self._transient_disjoint_counts):
                    self._transient_disjoint_counts[joint_index] += 1
                cur_count_display = (
                    self._transient_disjoint_counts[joint_index]
                    if 0 <= joint_index < len(self._transient_disjoint_counts)
                    else 1
                )
                log.warning(
                    "leader teleop: joint %d transient disjoint interval (gap=%.4f <= %.4f, count=%d/5); clamping to lead window",
                    joint_index + 1,
                    gap_step,
                    safe_jump_cap,
                    cur_count_display,
                )
                eff_lo = max(lead_lo, float(prev_val) - safe_jump_cap)
                eff_hi = min(lead_hi, float(prev_val) + safe_jump_cap)
                if eff_lo > eff_hi:
                    return 0.5 * (eff_lo + eff_hi)
                return min(max(float(val), eff_lo), eff_hi)

            raise

    def align_and_ramp(self) -> None:
        """Verify alignment against the follower arm and perform a smooth quintic ramp-in."""
        if self._stop:
            return
        self._transient_disjoint_counts = [0] * 6

        if self.schema.dof != 6 or self.schema.arm_dof not in (5, None) or self.schema.gripper_joint != 5:
            raise LeaderError(
                f"expected 6 DOF for SO-ARM101 (schema.dof={self.schema.dof}, "
                f"schema.arm_dof={self.schema.arm_dof}, schema.gripper_joint={self.schema.gripper_joint})"
            )
        if len(self.limits) != 6:
            raise LeaderError(f"expected 6 joint limits for SO-ARM101, got {len(self.limits)}")

        obs = self.session.wait_for_observation(timeout=5.0)
        self._check_follower_healthy(obs, require_torqued=False)

        if len(obs.positions) != 6:
            raise LeaderError(
                f"expected 6 DOF for SO-ARM101 (schema.dof={self.schema.dof}, "
                f"schema.arm_dof={self.schema.arm_dof}, follower_positions={len(obs.positions)})"
            )

        follower_start = list(obs.positions)
        leader_target = self.read_leader_target()
        gap = self._check_alignment_gap(leader_target, follower_start, stage="initial")

        v_caps = self._arm_velocity_caps()
        min_v_max = min(v_caps) if v_caps else 0.6
        max_ramp_vel = min(0.6, 0.8 * min_v_max)

        base_steps = max(1, int(round(self.ramp_seconds * self.fps)))
        v_steps = math.ceil(1.875 * gap * self.fps / max_ramp_vel) if gap > 0 else 1
        s_steps = math.ceil(1.875 * gap / self.arm_lead) if gap > 0 else 1
        ramp_steps = max(base_steps, v_steps, s_steps, 1)
        log.info(
            "Alignment ramp planned: %d steps (%.2fs) for initial gap %.3f rad (max_ramp_vel=%.2f rad/s, fps=%g)",
            ramp_steps,
            ramp_steps / self.fps,
            gap,
            max_ramp_vel,
            self.fps,
        )

        if self._stop:
            return

        if not obs.is_torqued or not obs.is_operational:
            if self._stop:
                return
            log.info("enabling follower torque before ramp ...")
            self._activated = True
            self.session.torque(True)
            deadline = time.monotonic() + 5.0
            while not (obs.is_torqued and obs.is_operational):
                if self._stop:
                    return
                rem = max(0.1, deadline - time.monotonic())
                if time.monotonic() >= deadline:
                    raise LeaderError("timed out waiting for follower to enable torque and become operational")
                obs = self.session.wait_for_observation(timeout=min(1.0, rem))
                self._check_follower_healthy(obs, require_torqued=False)
            if len(obs.positions) != 6:
                raise LeaderError(f"expected 6 follower positions after enabling torque, got {len(obs.positions)}")
            follower_start = list(obs.positions)
            leader_target = self.read_leader_target()
            post_gap = self._check_alignment_gap(leader_target, follower_start, stage="post-torque")
            v_steps_post = math.ceil(1.875 * post_gap * self.fps / max_ramp_vel) if post_gap > 0 else 1
            s_steps_post = math.ceil(1.875 * post_gap / self.arm_lead) if post_gap > 0 else 1
            safe_steps = max(ramp_steps, v_steps_post, s_steps_post, 1)
            if safe_steps != ramp_steps:
                log.info(
                    "Re-sizing ramp steps for post-torque gap %.3f rad from %d to %d steps (%.2fs)",
                    post_gap,
                    ramp_steps,
                    safe_steps,
                    safe_steps / self.fps,
                )
                ramp_steps = safe_steps

        if self._stop:
            return

        self._activated = True

        period = 1.0 / self.fps
        t0 = time.monotonic()
        prev_cmd = list(follower_start)
        prev_t = t0 - period
        v_cap = max_ramp_vel
        for step in range(1, ramp_steps + 1):
            if self._stop:
                break
            due = t0 + step * period
            cur_obs = self.session.latest_observation() or obs
            self._check_follower_healthy(cur_obs, require_torqued=True)
            ref = _validate_telemetry_ref(_measured_ref(cur_obs))
            s = step / ramp_steps
            # Quintic polynomial: 10s^3 - 15s^4 + 6s^5 (smooth acceleration & deceleration)
            alpha = 10.0 * (s**3) - 15.0 * (s**4) + 6.0 * (s**5)
            cur_leader = self.read_leader_target()
            if self._stop:
                break
            interp = [(1.0 - alpha) * flw + alpha * ldr for flw, ldr in zip(follower_start, cur_leader)]
            now = time.monotonic()
            dt = min(max(now - prev_t, 1e-3), 3.0 * period)
            prev_t = now
            if len(ref) < len(interp):
                raise LeaderError(f"follower reference length ({len(ref)}) < command length ({len(interp)})")
            for j in range(min(len(interp), len(ref))):
                if j != self.gripper_joint and j < 5:
                    max_step = v_cap * dt
                    j_min, j_max = self.limits[j] if j < len(self.limits) else (None, None)
                    interp[j] = self._clamp_with_transient_droop_recovery(
                        interp[j],
                        prev_cmd[j],
                        ref[j],
                        max_step,
                        self.arm_lead,
                        j_min=j_min,
                        j_max=j_max,
                        joint_index=j,
                    )
            gj = self.gripper_joint if self.gripper_joint is not None else 5
            if gj < len(interp) and gj < len(prev_cmd):
                interp[gj] = self._clamp_gripper_step_and_range(interp[gj], prev_cmd[gj])
            prev_cmd = list(interp)
            self.session.servo(interp)
            self._last_servo_cmd = list(interp)
            self.frames_sent += 1
            if due > now:
                time.sleep(due - now)
            elif now > due:
                t0 += now - due
            if self._stop:
                break

        if not self._stop:
            cur_obs = self.session.latest_observation() or obs
            self._check_follower_healthy(cur_obs, require_torqued=True)
            cur_follower = cur_obs.positions
            cur_leader = self.read_leader_target()
            arm_gap = max(
                (abs(l - f) for i, (l, f) in enumerate(zip(cur_leader, cur_follower)) if i != self.gripper_joint),
                default=0.0,
            )
            if arm_gap > 0.08:
                settle_timeout = max(2.0, ramp_steps / self.fps)
                settle_steps = int(math.ceil(settle_timeout * self.fps))
                log.info(
                    "Entering settle phase: arm_gap=%.3f rad > 0.08 rad (timeout=%.2fs, max %d steps)",
                    arm_gap,
                    settle_timeout,
                    settle_steps,
                )
                t_settle_start = time.monotonic()
                prev_t = t_settle_start - period
                for step in range(1, settle_steps + 1):
                    if self._stop:
                        break
                    due = t_settle_start + step * period
                    cur_obs = self.session.latest_observation() or obs
                    self._check_follower_healthy(cur_obs, require_torqued=True)
                    cur_follower = cur_obs.positions
                    ref = _validate_telemetry_ref(_measured_ref(cur_obs))
                    cur_leader = self.read_leader_target()
                    if self._stop:
                        break
                    arm_gap = max(
                        (abs(l - f) for i, (l, f) in enumerate(zip(cur_leader, cur_follower)) if i != self.gripper_joint),
                        default=0.0,
                    )
                    if arm_gap <= 0.08:
                        log.info("Settle phase converged: arm_gap=%.3f rad <= 0.08 rad", arm_gap)
                        break
                    settle_cmd = list(cur_leader)
                    now = time.monotonic()
                    dt = min(max(now - prev_t, 1e-3), 3.0 * period)
                    prev_t = now
                    if len(ref) < len(settle_cmd):
                        raise LeaderError(f"follower reference length ({len(ref)}) < command length ({len(settle_cmd)})")
                    for j in range(min(len(settle_cmd), len(ref))):
                        if j != self.gripper_joint and j < 5:
                            max_step = v_cap * dt
                            j_min, j_max = self.limits[j] if j < len(self.limits) else (None, None)
                            settle_cmd[j] = self._clamp_with_transient_droop_recovery(
                                settle_cmd[j],
                                prev_cmd[j],
                                ref[j],
                                max_step,
                                self.arm_lead,
                                j_min=j_min,
                                j_max=j_max,
                                joint_index=j,
                            )
                    gj = self.gripper_joint if self.gripper_joint is not None else 5
                    if gj < len(settle_cmd) and gj < len(prev_cmd):
                        settle_cmd[gj] = self._clamp_gripper_step_and_range(settle_cmd[gj], prev_cmd[gj])
                    prev_cmd = list(settle_cmd)
                    self.session.servo(settle_cmd)
                    self._last_servo_cmd = list(settle_cmd)
                    self.frames_sent += 1
                    if due > now:
                        time.sleep(due - now)
                    elif now > due:
                        t_settle_start += now - due
                if not self._stop and arm_gap > 0.08:
                    raise LeaderError(
                        f"settle phase failed to converge within timeout: arm_gap={arm_gap:.3f} rad > 0.08 rad"
                    )

    def run(self) -> None:
        """Stream servo frames continuously at fps."""
        self._error = None
        self._parked = False
        self._transient_disjoint_counts = [0] * 6
        self.bus.disable_torque(self.calib.servo_ids)
        try:
            try:
                self.align_and_ramp()
            except Exception as exc:
                self._error = exc
                log.error("leader alignment/ramp failed: %s", exc)
                return

            period = 1.0 / self.fps
            t0 = time.monotonic()
            prev_t = t0 - period
            k = 0
            v_caps = self._arm_velocity_caps()
            while not self._stop:
                due = t0 + k * period
                now = time.monotonic()
                if now - due > period:
                    t0 = now
                    k = 0
                    due = t0
                elif due > now:
                    time.sleep(due - now)
                if self._stop:
                    break

                now = time.monotonic()
                dt = min(max(now - prev_t, 1e-3), 3.0 * period)
                prev_t = now

                try:
                    obs = self.session.latest_observation()
                    self._check_follower_healthy(obs, require_torqued=True)
                    ref = _validate_telemetry_ref(_measured_ref(obs))
                    target = self.read_leader_target()
                    if self._stop:
                        break
                    clamped_target = list(target)
                    prev = self._last_servo_cmd or list(ref)
                    saturated = False
                    if len(ref) < len(clamped_target):
                        raise LeaderError(f"follower reference length ({len(ref)}) < target length ({len(clamped_target)})")
                    for j in range(min(len(clamped_target), len(ref))):
                        if j != self.gripper_joint and j < 5:
                            orig_val = clamped_target[j]
                            j_min, j_max = self.limits[j] if j < len(self.limits) else (None, None)
                            v_cap_j = v_caps[j] if j < len(v_caps) else 0.6
                            v_run_j = 0.8 * v_cap_j
                            max_frame_step_j = min(0.30, v_run_j * dt)
                            clamped_val = self._clamp_with_transient_droop_recovery(
                                orig_val,
                                prev[j],
                                ref[j],
                                max_frame_step_j,
                                self.arm_lead,
                                j_min=j_min,
                                j_max=j_max,
                                joint_index=j,
                            )
                            if clamped_val != orig_val:
                                saturated = True
                            clamped_target[j] = clamped_val

                    gj = self.gripper_joint if self.gripper_joint is not None else 5
                    if gj < len(clamped_target) and gj < len(prev):
                        clamped_target[gj] = self._clamp_gripper_step_and_range(clamped_target[gj], prev[gj])

                    if saturated:
                        self._lead_clamp_consecutive += 1
                        if self._lead_clamp_consecutive in (1, 10, 50, 100) or (self._lead_clamp_consecutive % 100 == 0):
                            log.warning(
                                "leader teleop: lead clamping active for %d consecutive frames (arm target clamped to ±%.2f rad from follower commanded/measured)",
                                self._lead_clamp_consecutive,
                                self.arm_lead,
                            )
                    else:
                        if self._lead_clamp_consecutive > 0:
                            log.info(
                                "leader teleop: lead clamping clear after %d consecutive frames",
                                self._lead_clamp_consecutive,
                            )
                        self._lead_clamp_consecutive = 0

                    self.session.servo(clamped_target)
                    self._last_servo_cmd = list(clamped_target)
                    self.frames_sent += 1
                except (LeaderError, ServoRejected, LeaseError, ArmConnectionError, RebotError, OSError) as exc:
                    self._error = exc
                    log.error("servo streaming stopped: %s", exc)
                    break
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    self._error = exc
                    raise
                k += 1
        except Exception as exc:
            if self._error is None:
                self._error = exc
            raise
        finally:
            if self._error is not None and self.park_on_exit and self._activated and not self._parked:
                try:
                    self.session.unload(wait=True, timeout=15.0)
                    self._parked = True
                except Exception as exc:
                    log.warning("fail-closed park on error failed: %s", exc)

    def close(self) -> None:
        """Clean shutdown: park and release lease."""
        try:
            if self.park_on_exit and self._activated and not self._parked:
                log.info("parking follower arm ...")
                self.session.unload(wait=True, timeout=15.0)
                self._parked = True
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
        help="required on real hardware (non-mock/sim follower): confirms operator is standing at the follower arm with a way to cut power",
    )
    p.add_argument(
        "--max-gap-rad",
        type=float,
        default=DEFAULT_MAX_GAP_RAD,
        help=f"maximum allowed initial gap in radians (default {DEFAULT_MAX_GAP_RAD})",
    )
    p.add_argument(
        "--allow-large-gap",
        action="store_true",
        help="allow initial gap exceeding --max-gap-rad up to pi radians ceiling",
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

    try:
        calib = LeaderCalibration.load(args.calib)
    except Exception as exc:
        print(f"rebot-teleop-leader: failed to load calibration: {exc}", file=sys.stderr)
        return 1

    try:
        port = args.leader_port or auto_detect_leader_port(baud=args.baud, servo_ids=calib.servo_ids)
    except Exception as exc:
        print(f"rebot-teleop-leader: failed to find leader port: {exc}", file=sys.stderr)
        return 1

    log.info("opening leader serial port %s at %d baud", port, args.baud)
    try:
        transport = open_serial_transport(port, baud=args.baud)
    except Exception as exc:
        print(f"rebot-teleop-leader: failed to open {port}: {exc}", file=sys.stderr)
        return 1

    session: ArmSession | None = None
    bridge: LeaderTeleopBridge | None = None
    bus: FeetechSerialBus | None = None
    saved_handlers: list[tuple[int, Any]] = []
    try:
        bus = FeetechSerialBus(transport)
        client = ArmdClient(host_url("http", args.arm_host, ARMD_PORT))
        try:
            schema = client.get_schema()
        except Exception as exc:
            print(f"rebot-teleop-leader: failed to fetch armd schema: {exc}", file=sys.stderr)
            return 1

        if schema.dof != 6 or schema.arm_dof not in (5, None) or schema.gripper_joint != 5:
            print(
                f"rebot-teleop-leader: unsupported arm schema (dof={schema.dof}, arm_dof={schema.arm_dof}); "
                "expected 6 DOF (5 arm + 1 gripper) for SO-ARM101",
                file=sys.stderr,
            )
            return 1

        follower = None
        try:
            describe = client.get_safety_describe()
            if isinstance(describe, dict):
                device_info = describe.get("device")
                if isinstance(device_info, dict):
                    follower = device_info.get("follower")
        except Exception as exc:
            log.info("safety describe unavailable: %s", exc)

        if follower not in ("mock", "sim") and not args.i_am_at_the_arm:
            print(
                f"rebot-teleop-leader: follower is {follower or 'unknown'} (not mock/sim): "
                "leader teleop will move a real arm. Stand at the arm with a way to cut power and pass --i-am-at-the-arm.",
                file=sys.stderr,
            )
            return 1

        stop_requested = threading.Event()

        def _sig_handler(*_: Any) -> None:
            stop_requested.set()
            if bridge is not None:
                bridge.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.signal(sig, _sig_handler)
                saved_handlers.append((sig, prev))
            except ValueError:
                pass

        responded = bus.ping_ids(calib.servo_ids)
        if set(responded) != set(calib.servo_ids):
            missing = sorted(set(calib.servo_ids) - set(responded))
            print(
                f"rebot-teleop-leader: leader arm on {port} did not respond on servo IDs {missing} "
                f"(responded: {responded}); check wiring and bus before disabling torque",
                file=sys.stderr,
            )
            return 1

        if stop_requested.is_set():
            return 0

        session = ArmSession(client, safe_exit="off")
        session.open()
        if stop_requested.is_set():
            return 0
        if hasattr(session, "set_schema"):
            session.set_schema(schema)
        else:
            session._schema = schema
        bridge = LeaderTeleopBridge(
            bus,
            session,
            schema,
            calib=calib,
            follower=follower,
            fps=args.fps,
            ramp_seconds=args.ramp_seconds,
            max_gap_rad=args.max_gap_rad,
            operator_present=args.i_am_at_the_arm,
            allow_large_gap=args.allow_large_gap,
            park_on_exit=not args.no_park,
        )
        if stop_requested.is_set():
            bridge.stop()

        bridge.run()
        if bridge._error is not None:
            return 1
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"rebot-teleop-leader: error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if bridge is not None:
                bridge.close()
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            transport.close()
            if bridge is not None and bus is not None:
                print(f"leader teleop closed: {bridge.frames_sent} frames sent, {bus.dropouts} dropouts")
        finally:
            for sig, prev in saved_handlers:
                try:
                    signal.signal(sig, prev)
                except (ValueError, OSError):
                    pass


if __name__ == "__main__":
    sys.exit(main())
