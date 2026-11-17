#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pymavlink import mavutil


@dataclass
class VehicleState:
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"

    range_m: Optional[float] = None
    flow_quality: Optional[int] = None

    # Tọa độ optical flow tích luỹ theo mét
    opt_m_x: float = 0.0
    opt_m_y: float = 0.0

    last_heartbeat_t: float = 0.0
    last_range_t: float = 0.0
    last_flow_t: float = 0.0

    last_statustext: str = ""
    last_statustext_t: float = 0.0


class MavlinkInterface:
    def __init__(self, conn_str: str, baud: int = 115200, heartbeat_timeout_s: float = 3.0):
        self.conn_str = conn_str
        self.baud = baud
        self.heartbeat_timeout_s = heartbeat_timeout_s

        self.master: Optional[mavutil.mavfile] = None
        self.state = VehicleState()

    # =================================================
    # Connection
    # =================================================
    def connect(self, wait_heartbeat_timeout: float = 10.0) -> None:
        self.master = mavutil.mavlink_connection(
            self.conn_str,
            baud=self.baud,
            autoreconnect=True,
        )

        hb = self.master.wait_heartbeat(timeout=wait_heartbeat_timeout)
        if hb is None:
            raise RuntimeError("Did not receive heartbeat from Pixhawk")

        self.state.connected = True
        self.state.last_heartbeat_t = time.time()

        self.master.target_system = hb.get_srcSystem()
        self.master.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        time.sleep(0.3)
        self.poll(max_batch=100)

        try:
            self.request_default_streams()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.master is not None:
                self.master.close()
        except Exception:
            pass

    def is_connected_recent(self) -> bool:
        return (time.time() - self.state.last_heartbeat_t) <= self.heartbeat_timeout_s

    # =================================================
    # Mode helpers
    # =================================================
    def _normalize_mode_name(self, name: Optional[str]) -> str:
        if not name:
            return "UNKNOWN"

        text = str(name).strip().upper().replace(" ", "_").replace("-", "_")
        aliases = {
            "ALTHOLD": "ALT_HOLD",
            "POS_HOLD": "POSHOLD",
            "AUTO_RTL": "AUTO_RTL",
            "GUIDEDNOGPS": "GUIDED_NOGPS",
            "SMARTRTL": "SMART_RTL",
            "OFLOITER": "OF_LOITER",
        }
        return aliases.get(text, text)

    def _tx_mode_name(self, mode_name: str) -> str:
        """
        Tên mode gửi xuống FC.
        Mặc định giữ nguyên dạng normalize; một số alias được chuyển về tên
        mà ArduPilot/pymavlink thường hiểu chắc chắn hơn.
        """
        normalized = self._normalize_mode_name(mode_name)
        tx_aliases = {
            "ALT_HOLD": "ALT_HOLD",
            "GUIDED_NOGPS": "GUIDED_NOGPS",
            "SMART_RTL": "SMART_RTL",
            "OF_LOITER": "OF_LOITER",
        }
        return tx_aliases.get(normalized, normalized)

    # =================================================
    # Generic helpers
    # =================================================
    def _update_statustext(self, text: str) -> None:
        self.state.last_statustext = str(text)
        self.state.last_statustext_t = time.time()

    def last_statustext(self, max_age_s: float = 3.0) -> Optional[str]:
        if (time.time() - self.state.last_statustext_t) <= max_age_s and self.state.last_statustext:
            return self.state.last_statustext
        return None

    def ack_result_to_str(self, result: int) -> str:
        accepted = getattr(mavutil.mavlink, "MAV_RESULT_ACCEPTED", 0)
        temp_rejected = getattr(mavutil.mavlink, "MAV_RESULT_TEMPORARILY_REJECTED", 1)
        denied = getattr(mavutil.mavlink, "MAV_RESULT_DENIED", 2)
        unsupported = getattr(mavutil.mavlink, "MAV_RESULT_UNSUPPORTED", 3)
        failed = getattr(mavutil.mavlink, "MAV_RESULT_FAILED", 4)
        in_progress = getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5)
        cancelled = getattr(mavutil.mavlink, "MAV_RESULT_CANCELLED", 6)

        table = {
            accepted: "ACCEPTED",
            temp_rejected: "TEMP_REJECTED",
            denied: "DENIED",
            unsupported: "UNSUPPORTED",
            failed: "FAILED",
            in_progress: "IN_PROGRESS",
            cancelled: "CANCELLED",
        }
        return table.get(result, f"UNKNOWN({result})")

    def _decode_mode_name(self, msg) -> str:
        try:
            name = mavutil.mode_string_v10(msg)
            if name and name != "UNKNOWN" and not str(name).startswith("Mode("):
                return self._normalize_mode_name(str(name))
        except Exception:
            pass

        try:
            if self.master is not None and hasattr(self.master, "mode_mapping"):
                mapping = self.master.mode_mapping()
                if mapping:
                    inv = {v: k for k, v in mapping.items()}
                    custom_mode = int(getattr(msg, "custom_mode", 0))
                    if custom_mode in inv:
                        return self._normalize_mode_name(str(inv[custom_mode]))
        except Exception:
            pass

        return "UNKNOWN"

    def reset_optical_xy(self) -> None:
        self.state.opt_m_x = 0.0
        self.state.opt_m_y = 0.0

    # =================================================
    # Stream request
    # =================================================
    def request_message_interval(self, message_id: int, rate_hz: float) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        try:
            interval_us = int(1e6 / rate_hz) if rate_hz > 0 else -1
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(message_id),
                float(interval_us),
                0, 0, 0, 0, 0,
            )
            return True
        except Exception:
            return False

    def request_default_streams(self) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )
        except Exception:
            pass

        msg_rates = {
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT: 2,
            mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR: 15,
            mavutil.mavlink.MAVLINK_MSG_ID_OPTICAL_FLOW: 15,
            mavutil.mavlink.MAVLINK_MSG_ID_OPTICAL_FLOW_RAD: 15,
        }

        # RANGEFINDER không phải firmware nào cũng hỗ trợ SET_MESSAGE_INTERVAL riêng
        rangefinder_id = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_RANGEFINDER", None)
        if rangefinder_id is not None:
            msg_rates[rangefinder_id] = 15

        for msg_id, rate_hz in msg_rates.items():
            self.request_message_interval(msg_id, rate_hz)

    # =================================================
    # Message consume
    # =================================================
    def _consume_message(self, msg) -> None:
        if self.master is None:
            return

        now = time.time()
        mtype = msg.get_type()

        if mtype == "BAD_DATA":
            return

        if mtype == "STATUSTEXT":
            txt = getattr(msg, "text", "")
            self._update_statustext(str(txt))
            return

        if mtype == "HEARTBEAT":
            src_sys = msg.get_srcSystem()
            if src_sys != self.master.target_system:
                return

            autopilot = getattr(msg, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID)
            if autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                return

            self.state.connected = True
            self.state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.state.mode = self._decode_mode_name(msg)
            self.state.last_heartbeat_t = now
            return

        if mtype == "DISTANCE_SENSOR":
            current_cm = getattr(msg, "current_distance", None)
            if current_cm is not None:
                self.state.range_m = float(current_cm) / 100.0
                self.state.last_range_t = now
            return

        if mtype == "RANGEFINDER":
            dist = getattr(msg, "distance", None)
            if dist is not None:
                self.state.range_m = float(dist)
                self.state.last_range_t = now
            return

        if mtype == "OPTICAL_FLOW_RAD":
            q = getattr(msg, "quality", None)
            if q is not None:
                self.state.flow_quality = int(q)
                self.state.last_flow_t = now

            rng = self.state.range_m
            ix = getattr(msg, "integrated_x", None)
            iy = getattr(msg, "integrated_y", None)

            if rng is not None and ix is not None and iy is not None:
                # Small-angle approximation: displacement = angle * height
                dx_m = float(ix) * float(rng)
                dy_m = float(iy) * float(rng)
                self.state.opt_m_x += dx_m
                self.state.opt_m_y += dy_m
            return

        if mtype == "OPTICAL_FLOW":
            q = getattr(msg, "quality", None)
            if q is not None:
                self.state.flow_quality = int(q)
                self.state.last_flow_t = now
            return

    def poll(self, max_batch: int = 50) -> VehicleState:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        try:
            for _ in range(max_batch):
                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    break
                self._consume_message(msg)
        except Exception as exc:
            self.state.connected = False
            raise RuntimeError(f"MAVLink poll failed: {exc}") from exc

        return self.state

    # =================================================
    # State helpers
    # =================================================
    def mode_is(self, mode_name: str) -> bool:
        return self.state.mode == self._normalize_mode_name(mode_name)

    def heartbeat_recent(self) -> bool:
        return self.is_connected_recent()

    def range_recent(self, timeout_s: float = 0.7) -> bool:
        return (time.time() - self.state.last_range_t) <= timeout_s

    def flow_recent(self, timeout_s: float = 0.7) -> bool:
        return (time.time() - self.state.last_flow_t) <= timeout_s

    def range_valid(self, min_m: float = 0.15, max_m: float = 8.0, timeout_s: float = 0.7) -> bool:
        r = self.state.range_m
        if r is None or not self.range_recent(timeout_s):
            return False
        return min_m <= r <= max_m

    def flow_valid(self, quality_min: int = 30, timeout_s: float = 0.7) -> bool:
        q = self.state.flow_quality
        if q is None or not self.flow_recent(timeout_s):
            return False
        return q >= quality_min

    # =================================================
    # Command ACK helpers
    # =================================================
    def wait_command_ack(self, command_id: int, timeout_s: float = 2.5) -> Dict[str, Any]:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        deadline = time.time() + timeout_s
        last_statustext = self.last_statustext()

        while time.time() < deadline:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.2)
            except Exception as exc:
                return {
                    "ok": False,
                    "result": None,
                    "result_str": f"READ_ERROR({exc})",
                    "statustext": last_statustext,
                }

            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "STATUSTEXT":
                txt = getattr(msg, "text", "")
                last_statustext = str(txt)
                self._consume_message(msg)
                continue

            if mtype == "COMMAND_ACK":
                cmd = getattr(msg, "command", None)
                result = int(getattr(msg, "result", -1))
                if cmd == command_id:
                    return {
                        "ok": result in (
                            mavutil.mavlink.MAV_RESULT_ACCEPTED,
                            mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
                        ),
                        "result": result,
                        "result_str": self.ack_result_to_str(result),
                        "statustext": last_statustext,
                    }
                continue

            self._consume_message(msg)

        return {
            "ok": False,
            "result": None,
            "result_str": "NO_ACK",
            "statustext": last_statustext,
        }

    def wait_mode(self, mode_name: str, timeout_s: float = 3.0) -> bool:
        target_mode = self._normalize_mode_name(mode_name)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.poll()
            if self.state.mode == target_mode:
                return True
            time.sleep(0.1)
        return False

    # =================================================
    # Commands
    # =================================================
    def set_mode(self, mode_name: str) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        try:
            tx_name = self._tx_mode_name(mode_name)
            self.master.set_mode_apm(tx_name)
            return True
        except Exception:
            return False

    def set_mode_and_wait(self, mode_name: str, timeout_s: float = 3.0) -> Dict[str, Any]:
        ok = self.set_mode(mode_name)
        if not ok:
            return {
                "ok": False,
                "result_str": "SEND_FAILED",
                "statustext": self.last_statustext(),
            }

        changed = self.wait_mode(mode_name, timeout_s=timeout_s)
        return {
            "ok": changed,
            "result_str": "MODE_CONFIRMED" if changed else "MODE_NOT_CONFIRMED",
            "statustext": self.last_statustext(),
        }

    def send_nav_takeoff(self, alt_m: float) -> Dict[str, Any]:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        # Bay optical: không dựa vào GPS/lat/lon
        self.master.mav.command_long_send(
            self.master.target_system,
            mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0.0,         # pitch
            0.0,
            0.0,
            float("nan"),
            0.0,
            0.0,
            float(alt_m),
        )
        return self.wait_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout_s=2.5)

    def send_body_velocity(self, vx: float, vy: float, vz: float) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0.0, 0.0, 0.0,
            float(vx), float(vy), float(vz),
            0.0, 0.0, 0.0,
            0.0, 0.0,
        )

    def set_param(self, name: str, value: float, retries: int = 3, timeout_s: float = 1.0) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        for _ in range(retries):
            try:
                self.master.mav.param_set_send(
                    self.master.target_system,
                    self.master.target_component,
                    name.encode("utf-8"),
                    float(value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                )

                ack_deadline = time.time() + timeout_s
                while time.time() < ack_deadline:
                    msg = self.master.recv_match(blocking=True, timeout=0.2)
                    if msg is None:
                        continue

                    if msg.get_type() == "PARAM_VALUE":
                        ack_name = getattr(msg, "param_id", b"")
                        if isinstance(ack_name, bytes):
                            ack_name = ack_name.decode(errors="ignore")
                        ack_name = str(ack_name).rstrip("\x00")

                        if ack_name == name:
                            return True
                        continue

                    self._consume_message(msg)

            except Exception:
                pass

            time.sleep(0.05)

        return False

    # =================================================
    # Debug
    # =================================================
    def summary(self) -> str:
        s = self.state
        return (
            f"mode={s.mode} armed={s.armed} connected={self.is_connected_recent()} "
            f"range={s.range_m} flow_q={s.flow_quality} "
            f"opt_m_x={s.opt_m_x:.3f} opt_m_y={s.opt_m_y:.3f} "
            f"statustext={self.last_statustext()}"
        )
