#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String
from pymavlink import mavutil

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

MODE_GUIDED = 4
MODE_LAND = 9
TYPEMASK_VEL_YAWRATE = 1479

MISSION_RED1 = 'RED1'
MISSION_YELLOW_LEFT = 'YELLOW_LEFT'
MISSION_RED2 = 'RED2'
MISSION_YELLOW_RIGHT = 'YELLOW_RIGHT'
MISSION_BLUE = 'BLUE'
MISSION_COMPLETE = 'MISSION_COMPLETE'

SEARCH_FORWARD = 'FORWARD'
SEARCH_LATERAL_LEFT = 'LATERAL_LEFT'
SEARCH_LATERAL_RIGHT = 'LATERAL_RIGHT'

ORIGINAL_PARAMS = {
    b'WPNAV_SPEED': 1000.0,
    b'WPNAV_SPEED_UP': 250.0,
    b'WPNAV_SPEED_DN': 150.0,
    b'WPNAV_ACCEL': 100.0,
    b'ANGLE_MAX': 3000.0,
    b'PILOT_SPEED_UP': 250.0,
    b'LOIT_SPEED': 1250.0,
    b'LAND_SPEED': 50.0,
    b'LAND_SPEED_HIGH': 0.0,
}

AUTO_PARAMS = {
    b'WPNAV_SPEED': 500.0,
    b'WPNAV_SPEED_UP': 220.0,
    b'WPNAV_SPEED_DN': 90.0,
    b'WPNAV_ACCEL': 120.0,
    b'ANGLE_MAX': 3000.0,
    b'PILOT_SPEED_UP': 220.0,
    b'LOIT_SPEED': 500.0,
    b'LAND_SPEED': 50.0,
    b'LAND_SPEED_HIGH': 50.0,
}


class AutoControlNode(Node):
    """
    Auto control node - NO SERVO VERSION.

    Mission order compatible with vision_node V1.0:
      RED1 -> YELLOW_LEFT -> RED2 -> YELLOW_RIGHT -> BLUE -> LAND

    Completion rule:
      - When /vision/aligned=True for 30 consecutive control loops, this step is considered done.
      - No /dropper/fire, no /dropper/done, no servo waiting.
      - After 30 stable frames, auto commits current target to vision memory and switches to next mission step.
    """

    def __init__(self):
        super().__init__('auto_control_node')
        self.get_logger().info(
            'Auto control NO-SERVO started: RED1 -> YELLOW_LEFT -> RED2 -> YELLOW_RIGHT -> BLUE -> LAND'
        )

        # MAVLink / vehicle state
        self.master = None
        self.ready = False
        self.armed = False
        self.airborne = False
        self.auto_mode = False
        self.mission_running = False
        self.landing = False
        self.current_mode = ''
        self.phase = 'IDLE'
        self.auto_start_lock = threading.Lock()
        self.guided_lost_since = None

        # Mission state
        self.mission_step = MISSION_RED1
        self.search_mode = SEARCH_FORWARD
        self.total_targets = 5
        self.completed_targets = 0
        self.vision_profile_dirty = True
        self.step_seen_target_once = False
        self.recover_disabled_until = 0.0
        self.line_disabled_until = 0.0

        # Optical flow / range
        self.flow_quality = None
        self.flow_min_quality = 10
        self.last_flow_time = 0.0
        self.lidar_height = None
        self.lidar_ground_level = 0.18
        self.land_threshold = 0.25
        self.land_noise_margin = 0.05

        # Blind / pending handling
        self.blind_resume_block_s = 1.2
        self.blind_disabled_until = 0.0
        self.pending_hold_s = 0.8
        self.pending_detect_until = 0.0

        # Mission config
        self.takeoff_alt = 4.5
        self.hover_before_search = 0.8
        self.search_forward_speed_mps = 0.50
        self.left_scan_speed_mps = -0.22
        self.right_scan_speed_mps = 0.22
        self.search_yaw_rate = 0.0
        self.takeoff_complete_alt_m = 3.2
        self.takeoff_fallback_alt_m = 2.8
        self.takeoff_cmd_timeout_s = 14.0
        self.step_switch_recover_block_s = 0.75
        self.step_switch_line_block_s = 0.35
        self.vision_switch_guard_s = 0.35

        # Vision freshness / gating
        self.vision_fresh_timeout_s = 0.45
        self.line_fresh_timeout_s = 0.45

        # Circle recover
        self.circle_recover_hold_s = 0.55
        self.circle_recover_max_dist_px = 120.0
        self.circle_recover_min_radius_px = 45.0
        self.circle_recover_track_scale = 0.38
        self.last_good_circle_t = 0.0
        self.last_good_circle_dist_px = 1e9
        self.last_good_circle_radius = 0.0
        self.last_good_circle_err_x = None
        self.last_good_circle_err_y = None

        # Coarse circle tracking
        self.track_kp_x = 0.0040  # err_x -> vy
        self.track_kp_y = 0.0040  # err_y -> vx
        self.track_max_speed_mps = 0.35

        # Fine settle PID
        self.settle_enter_dist_px = 30.0
        self.settle_exit_dist_px = 42.0
        self.settle_max_speed_mps = 0.08
        self.pid_kp_x = 0.0012764
        self.pid_ki_x = 0.000241785
        self.pid_kd_x = 0.000431
        self.pid_kp_y = 0.0011752
        self.pid_ki_y = 0.00024812
        self.pid_kd_y = 0.000425
        self.pid_integral_limit = 80.0
        self.in_settle_mode = False

        # Stable frame completion
        self.stable_frame_required = 30
        self.stable_frame_count = 0

        # Line guidance PID, optional if vision publishes line data
        self.line_forward_speed_mps = 0.26
        self.line_err_to_vy_sign = 1.0   # nếu kéo ngược line thì đổi thành -1.0
        self.line_lateral_kp = 0.0048
        self.line_lateral_ki = 0.00035
        self.line_lateral_kd = 0.00065
        self.line_integral_limit = 120.0
        self.line_deadband_px = 10.0
        self.line_lateral_max_speed_mps = 0.24
        self.line_ix = 0.0
        self.line_prev_ex = None
        self.line_prev_t = None

        self.vision_x_to_vy_sign = 1.0
        self.vision_y_to_vx_sign = 1.0

        # Camera / vision state
        self.camera_alive = False
        self.camera_info = 'camera_unknown'
        self.vision_target_found = False
        self.vision_aligned = False
        self.vision_status = 'vision_unknown'
        self.vision_current_color = 'unknown'
        self.vision_err_x = None
        self.vision_err_y = None
        self.vision_center_x = None
        self.vision_center_y = None
        self.vision_radius = None
        self.vision_last_seen_t = 0.0

        # White line state
        self.line_found = False
        self.line_err_x = 0.0
        self.line_angle_deg = 0.0
        self.line_half_width = 0.0
        self.line_status = 'no_line'
        self.line_last_seen_t = 0.0

        # PID / mission progress
        self.pid_ix = 0.0
        self.pid_iy = 0.0
        self.pid_prev_ex = None
        self.pid_prev_ey = None
        self.pid_prev_t = None

        # ROS pubs/subs
        self.mode_pub = self.create_publisher(String, '/flight_mode', 10)
        self.auto_pub = self.create_publisher(Bool, '/auto_mode', 10)

        self.vision_mode_pub = self.create_publisher(String, '/vision/desired_target_mode', 10)
        self.vision_reset_pub = self.create_publisher(Bool, '/vision/reset_target', 10)
        self.vision_commit_pub = self.create_publisher(String, '/vision/commit_target', 10)
        self.vision_clear_memory_pub = self.create_publisher(Bool, '/vision/clear_mission_memory', 10)
        self.vision_ban_enable_pub = self.create_publisher(Bool, '/vision/ban_zone_enabled', 10)
        self.vision_ban_center_pub = self.create_publisher(PointStamped, '/vision/ban_zone_center', 10)

        self.create_subscription(Bool, '/drone_takeoff', self._cb_takeoff, 10)
        self.create_subscription(Bool, '/start_trajectory', self._cb_start, 10)
        self.create_subscription(Bool, '/drone_land', self._cb_land, 10)
        self.create_subscription(Bool, '/camera/raw_alive', self._cb_camera_alive, 10)
        self.create_subscription(String, '/camera/raw_info', self._cb_camera_info, 10)

        self.create_subscription(Bool, '/vision/target_found', self._cb_vision_found, 10)
        self.create_subscription(Bool, '/vision/aligned', self._cb_vision_aligned, 10)
        self.create_subscription(String, '/vision/status', self._cb_vision_status, 10)
        self.create_subscription(String, '/vision/current_color', self._cb_vision_current_color, 10)
        self.create_subscription(PointStamped, '/vision/target_error', self._cb_vision_error, 10)
        self.create_subscription(PointStamped, '/vision/target_center', self._cb_vision_center, 10)

        self.create_subscription(Bool, '/vision/line_found', self._cb_line_found, 10)
        self.create_subscription(PointStamped, '/vision/line_error', self._cb_line_error, 10)
        self.create_subscription(String, '/vision/line_status', self._cb_line_status, 10)

        self.create_timer(0.5, self._log_status)

        t = threading.Thread(target=self._connect_and_read)
        t.daemon = False
        t.start()

    # ---------- Mission / vision helpers ----------
    def _current_target_mode(self) -> str:
        if self.mission_step == MISSION_RED1:
            return 'red_1'
        if self.mission_step == MISSION_YELLOW_LEFT:
            return 'yellow_left'
        if self.mission_step == MISSION_RED2:
            return 'red_2'
        if self.mission_step == MISSION_YELLOW_RIGHT:
            return 'yellow_right'
        if self.mission_step == MISSION_BLUE:
            return 'blue'
        return 'idle'

    def _is_red_phase(self) -> bool:
        return self.mission_step in (MISSION_RED1, MISSION_RED2)

    def _is_forward_phase(self) -> bool:
        return self.search_mode == SEARCH_FORWARD

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _reset_step_circle_memory(self):
        self.last_good_circle_t = 0.0
        self.last_good_circle_dist_px = 1e9
        self.last_good_circle_radius = 0.0
        self.last_good_circle_err_x = None
        self.last_good_circle_err_y = None
        self.step_seen_target_once = False

    def _publish_vision_profile(self, reset_lock: bool = False):
        mode_msg = String()
        mode_msg.data = self._current_target_mode()
        self.vision_mode_pub.publish(mode_msg)

        ben = Bool()
        ben.data = False
        self.vision_ban_enable_pub.publish(ben)

        center_msg = PointStamped()
        center_msg.header.stamp = self.get_clock().now().to_msg()
        center_msg.header.frame_id = 'camera'
        center_msg.point.x = 0.0
        center_msg.point.y = 0.0
        center_msg.point.z = 0.0
        self.vision_ban_center_pub.publish(center_msg)

        if reset_lock:
            rmsg = Bool()
            rmsg.data = True
            self.vision_reset_pub.publish(rmsg)

        self.vision_profile_dirty = False

    def _clear_vision_memory(self):
        msg = Bool()
        msg.data = True
        self.vision_clear_memory_pub.publish(msg)

    def _commit_vision_target(self, mode_str: str):
        msg = String()
        msg.data = mode_str
        self.vision_commit_pub.publish(msg)

    def _set_mission_step(self, step: str, *, reset_lock: bool = True):
        self.mission_step = step
        if step in (MISSION_RED1, MISSION_RED2, MISSION_BLUE):
            self.search_mode = SEARCH_FORWARD
        elif step == MISSION_YELLOW_LEFT:
            self.search_mode = SEARCH_LATERAL_LEFT
        elif step == MISSION_YELLOW_RIGHT:
            self.search_mode = SEARCH_LATERAL_RIGHT
        else:
            self.search_mode = SEARCH_FORWARD

        self._reset_target_progress()
        self._reset_step_circle_memory()
        self.recover_disabled_until = time.time() + self.step_switch_recover_block_s
        self.line_disabled_until = time.time() + self.step_switch_line_block_s
        self.vision_profile_dirty = True
        self._publish_vision_profile(reset_lock=reset_lock)
        self.get_logger().info(f'MISSION STEP -> {self.mission_step} | search_mode={self.search_mode}')

    # ---------- MAVLink helpers ----------
    def _request_message_interval(self, msg_id: int, hz: float):
        if self.master is None:
            return
        interval_us = int(1e6 / hz) if hz > 0 else 0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(msg_id),
            float(interval_us),
            0, 0, 0, 0, 0,
        )
        time.sleep(0.05)

    def _set_params(self, params: dict, label: str):
        if self.master is None:
            return
        self.get_logger().info(f'Setting {label} params...')
        for name, value in params.items():
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                name,
                value,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            time.sleep(0.1)
        self.get_logger().info(f'{label} params set ✅')

    def _set_auto_params(self):
        self._set_params(AUTO_PARAMS, 'AUTO')

    def _restore_original_params(self):
        self._set_params(ORIGINAL_PARAMS, 'ORIGINAL')

    def _connect_and_read(self):
        self.get_logger().info(f'Connecting: {SERIAL_PORT}')
        try:
            self.master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
            self.master.wait_heartbeat()
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 2)
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 10)
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_OPTICAL_FLOW, 10)
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_OPTICAL_FLOW_RAD, 10)
            self.get_logger().info(f'Connected! System: {self.master.target_system}')

            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )

            self.ready = True
            lidar_calibrated = False
            samples = []
            lidar_wait_t0 = time.time()
            calib_t0 = None

            while True:
                try:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    if msg is not None:
                        self._handle_message(msg)

                    if not lidar_calibrated:
                        if self.lidar_height is None:
                            if time.time() - lidar_wait_t0 > 5.0:
                                self.get_logger().warn(
                                    f'No lidar, use default ground: {self.lidar_ground_level:.2f}m'
                                )
                                lidar_calibrated = True
                        else:
                            if calib_t0 is None:
                                self.get_logger().info('Calibrating lidar ground level (3s)...')
                                calib_t0 = time.time()
                            samples.append(self.lidar_height)
                            if time.time() - calib_t0 >= 3.0:
                                self.lidar_ground_level = sum(samples) / len(samples)
                                self.get_logger().info(f'Lidar ground: {self.lidar_ground_level:.2f}m ✅')
                                lidar_calibrated = True
                except Exception as e:
                    self.get_logger().warn(f'Message error: {e}')
        except Exception as e:
            self.get_logger().error(f'Connection failed: {e}')
            time.sleep(3.0)
            self._connect_and_read()

    def _handle_mode_change(self, mode: str):
        if mode == 'GUIDED' and not self.auto_mode:
            if not self.armed:
                self.get_logger().warn('AUTO REJECTED: chưa ARM!')
            elif not self._flow_is_healthy():
                self.get_logger().warn('AUTO REJECTED: Optical Flow chưa tốt!')
            elif not self.camera_alive:
                self.get_logger().warn('AUTO REJECTED: camera chưa sẵn sàng!')
            else:
                self._set_auto_params()
                self.auto_mode = True
                self.get_logger().info('AUTO MODE ON ✅')
                threading.Thread(target=self._auto_start, daemon=True).start()

    def _handle_guided_state(self, mode: str):
        if mode == 'GUIDED':
            self.guided_lost_since = None
            return
        if not self.auto_mode:
            return
        if self.guided_lost_since is None:
            self.guided_lost_since = time.time()
            return
        if time.time() - self.guided_lost_since < 0.4:
            return

        self.auto_mode = False
        self.mission_running = False
        self.landing = False
        self.phase = 'PILOT_TAKEOVER'
        self._reset_target_progress()
        threading.Thread(target=self._restore_original_params, daemon=True).start()
        self.get_logger().info(f'AUTO OFF -> {mode}')

    def _handle_message(self, msg):
        t = msg.get_type()

        if t == 'HEARTBEAT':
            if self.master is None:
                return
            src_sys = msg.get_srcSystem()
            src_comp = msg.get_srcComponent()
            if src_sys != self.master.target_system:
                return
            if src_comp not in (1, self.master.target_component):
                return

        if t in ('OPTICAL_FLOW', 'OPTICAL_FLOW_RAD'):
            if hasattr(msg, 'quality'):
                self.flow_quality = int(msg.quality)
                self.last_flow_time = time.time()
            return

        if t == 'DISTANCE_SENSOR':
            self.lidar_height = msg.current_distance / 100.0
            return

        if t == 'RANGEFINDER':
            self.lidar_height = msg.distance
            return

        if t != 'HEARTBEAT':
            return

        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        if armed != self.armed:
            self.armed = armed
            self.get_logger().info('ARMED ✅' if armed else 'DISARMED')
            if not armed and self.landing:
                self._on_landed()

        mode_map = {
            0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO',
            4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 9: 'LAND',
            16: 'POSHOLD', 20: 'GUIDED_NOGPS', 22: 'FLOWHOLD',
        }
        mode = mode_map.get(msg.custom_mode, f'UNKNOWN({msg.custom_mode})')

        if mode != self.current_mode:
            self.current_mode = mode
            mode_msg = String(); mode_msg.data = mode; self.mode_pub.publish(mode_msg)
            auto_msg = Bool(); auto_msg.data = (mode == 'GUIDED') and self.auto_mode; self.auto_pub.publish(auto_msg)
            self._handle_mode_change(mode)

        self._handle_guided_state(mode)

    # ---------- Low-level control ----------
    def _flow_is_healthy(self):
        age_ok = (time.time() - self.last_flow_time) < 1.0
        quality_ok = self.flow_quality is not None and self.flow_quality >= self.flow_min_quality
        return age_ok and quality_ok

    def _lidar_info(self):
        return f'{self.lidar_height:.2f}m' if self.lidar_height is not None else 'N/A'

    def _is_landed(self):
        if self.lidar_height is None:
            return False
        near_ground = abs(self.lidar_height - self.lidar_ground_level) <= self.land_noise_margin
        below_threshold = self.lidar_height < self.land_threshold
        return below_threshold or near_ground

    def _set_mode(self, mode_num: int) -> bool:
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num,
        )
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if mode_num == MODE_GUIDED and self.current_mode == 'GUIDED':
                return True
            if mode_num == MODE_LAND and self.current_mode == 'LAND':
                return True
            time.sleep(0.1)
        return False

    def _arm(self) -> bool:
        self.get_logger().info('Arming...')
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self.armed:
                self.get_logger().info('Armed!')
                return True
            time.sleep(0.1)
        self.get_logger().warn('Arm timeout!')
        return False

    def _send_body_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            TYPEMASK_VEL_YAWRATE,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0,
            yaw_rate,
        )

    def _hold_body(self, seconds: float):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.current_mode != 'GUIDED' or not self.armed:
                return False
            self._send_body_velocity(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.2)
        return True

    def _reset_pid(self):
        self.pid_ix = 0.0
        self.pid_iy = 0.0
        self.pid_prev_ex = None
        self.pid_prev_ey = None
        self.pid_prev_t = None

    def _reset_line_pid(self):
        self.line_ix = 0.0
        self.line_prev_ex = None
        self.line_prev_t = None

    def _reset_target_progress(self):
        self.stable_frame_count = 0
        self.in_settle_mode = False
        self._reset_pid()
        self._reset_line_pid()

    def _on_landed(self):
        self.airborne = False
        self.landing = False
        self.auto_mode = False
        self.mission_running = False
        self.phase = 'LANDED'
        self._reset_target_progress()
        threading.Thread(target=self._restore_original_params, daemon=True).start()
        self.get_logger().info('Landed ✅')

    # ---------- Vision freshness / strategy ----------
    def _vision_is_fresh(self) -> bool:
        return (time.time() - self.vision_last_seen_t) <= self.vision_fresh_timeout_s

    def _have_confirmed_target(self) -> bool:
        have_err = (self.vision_err_x is not None and self.vision_err_y is not None)
        return self.vision_target_found and have_err and self._vision_is_fresh()

    def _blind_is_blocked(self) -> bool:
        return time.time() < self.blind_disabled_until

    def _refresh_blind_block(self):
        self.blind_disabled_until = time.time() + self.blind_resume_block_s

    def _line_is_fresh(self) -> bool:
        return (time.time() - self.line_last_seen_t) <= self.line_fresh_timeout_s

    def _have_confirmed_line(self) -> bool:
        return self.line_found and self._line_is_fresh()

    def _circle_close_recently(self) -> bool:
        if not self.step_seen_target_once:
            return False
        if time.time() < self.recover_disabled_until:
            return False
        if self.last_good_circle_t <= 0.0:
            return False
        recent_ok = (time.time() - self.last_good_circle_t) <= self.circle_recover_hold_s
        close_ok = (
            self.last_good_circle_dist_px <= self.circle_recover_max_dist_px or
            self.last_good_circle_radius >= self.circle_recover_min_radius_px
        )
        have_err = self.last_good_circle_err_x is not None and self.last_good_circle_err_y is not None
        return recent_ok and close_ok and have_err

    def _compute_circle_recover_velocity(self):
        ex = float(self.last_good_circle_err_x)
        ey = float(self.last_good_circle_err_y)
        vx_cmd = self.vision_y_to_vx_sign * self.track_kp_y * ey * self.circle_recover_track_scale
        vy_cmd = self.vision_x_to_vy_sign * self.track_kp_x * ex * self.circle_recover_track_scale
        vx_cmd = self._clamp(vx_cmd, -0.18, 0.18)
        vy_cmd = self._clamp(vy_cmd, -0.18, 0.18)
        return vx_cmd, vy_cmd

    def _circle_takeover_ready(self) -> bool:
        return self._have_confirmed_target()

    def _should_use_line_guidance(self) -> bool:
        if time.time() < self.line_disabled_until:
            return False
        if not self._have_confirmed_line():
            return False
        if not self._is_red_phase():
            return False
        if not self._is_forward_phase():
            return False
        if self._have_confirmed_target():
            return False
        return True

    def _compute_line_guidance_velocity(self):
        ex = float(self.line_err_x)
        now = time.time()
        if self.line_prev_t is None:
            dt = 0.10
        else:
            dt = max(0.04, min(0.30, now - self.line_prev_t))

        if abs(ex) < self.line_deadband_px:
            ex_eff = 0.0
        else:
            ex_eff = ex

        self.line_ix += ex_eff * dt
        self.line_ix = self._clamp(self.line_ix, -self.line_integral_limit, self.line_integral_limit)
        dex = 0.0 if self.line_prev_ex is None else (ex_eff - self.line_prev_ex) / dt

        vy_cmd = self.line_err_to_vy_sign * (
            self.line_lateral_kp * ex_eff +
            self.line_lateral_ki * self.line_ix +
            self.line_lateral_kd * dex
        )
        vy_cmd = self._clamp(vy_cmd, -self.line_lateral_max_speed_mps, self.line_lateral_max_speed_mps)

        abs_ex = abs(ex)
        if abs_ex > 90.0:
            vx_cmd = 0.14
        elif abs_ex > 55.0:
            vx_cmd = 0.20
        else:
            vx_cmd = self.line_forward_speed_mps

        self.line_prev_ex = ex_eff
        self.line_prev_t = now
        return vx_cmd, vy_cmd

    def _compute_search_velocity(self) -> Tuple[float, float]:
        if self.search_mode == SEARCH_LATERAL_LEFT:
            return 0.0, self.left_scan_speed_mps
        if self.search_mode == SEARCH_LATERAL_RIGHT:
            return 0.0, self.right_scan_speed_mps
        return self.search_forward_speed_mps, 0.0

    def _compute_settle_pid_velocity(self, ex: float, ey: float):
        now = time.time()
        if self.pid_prev_t is None:
            dt = 0.05
        else:
            dt = max(0.02, min(0.30, now - self.pid_prev_t))

        self.pid_ix += ex * dt
        self.pid_iy += ey * dt
        self.pid_ix = self._clamp(self.pid_ix, -self.pid_integral_limit, self.pid_integral_limit)
        self.pid_iy = self._clamp(self.pid_iy, -self.pid_integral_limit, self.pid_integral_limit)

        dex = 0.0 if self.pid_prev_ex is None else (ex - self.pid_prev_ex) / dt
        dey = 0.0 if self.pid_prev_ey is None else (ey - self.pid_prev_ey) / dt

        vy_cmd = self.vision_x_to_vy_sign * (
            self.pid_kp_x * ex + self.pid_ki_x * self.pid_ix + self.pid_kd_x * dex
        )
        vx_cmd = self.vision_y_to_vx_sign * (
            self.pid_kp_y * ey + self.pid_ki_y * self.pid_iy + self.pid_kd_y * dey
        )

        vx_cmd = self._clamp(vx_cmd, -self.settle_max_speed_mps, self.settle_max_speed_mps)
        vy_cmd = self._clamp(vy_cmd, -self.settle_max_speed_mps, self.settle_max_speed_mps)

        self.pid_prev_ex = ex
        self.pid_prev_ey = ey
        self.pid_prev_t = now
        return vx_cmd, vy_cmd

    def _vision_mode_to_commit_string(self) -> str:
        return self._current_target_mode()

    def _advance_mission_after_done(self):
        self.completed_targets += 1
        self._hold_body(0.25)

        if self.mission_step == MISSION_RED1:
            self._set_mission_step(MISSION_YELLOW_LEFT, reset_lock=True)
            return 'continue'

        if self.mission_step == MISSION_YELLOW_LEFT:
            self._set_mission_step(MISSION_RED2, reset_lock=True)
            return 'continue'

        if self.mission_step == MISSION_RED2:
            self._set_mission_step(MISSION_YELLOW_RIGHT, reset_lock=True)
            return 'continue'

        if self.mission_step == MISSION_YELLOW_RIGHT:
            self._set_mission_step(MISSION_BLUE, reset_lock=True)
            return 'continue'

        if self.mission_step == MISSION_BLUE:
            self._set_mission_step(MISSION_COMPLETE, reset_lock=True)
            return 'complete'

        return 'continue'

    # ---------- ROS callbacks ----------
    def _cb_takeoff(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._auto_start, daemon=True).start()

    def _cb_start(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._auto_start, daemon=True).start()

    def _cb_land(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._land, daemon=True).start()

    def _cb_camera_alive(self, msg: Bool):
        self.camera_alive = msg.data

    def _cb_camera_info(self, msg: String):
        self.camera_info = msg.data

    def _cb_vision_found(self, msg: Bool):
        self.vision_target_found = msg.data
        if msg.data:
            self.vision_last_seen_t = time.time()
            self._refresh_blind_block()
        else:
            self.vision_err_x = None
            self.vision_err_y = None
            self.vision_radius = None
            self.stable_frame_count = 0

    def _cb_vision_aligned(self, msg: Bool):
        self.vision_aligned = msg.data

    def _cb_vision_status(self, msg: String):
        self.vision_status = msg.data
        if msg.data.startswith('candidate_pending_'):
            self.pending_detect_until = time.time() + self.pending_hold_s

    def _candidate_pending_recent(self) -> bool:
        return time.time() < self.pending_detect_until

    def _cb_vision_current_color(self, msg: String):
        self.vision_current_color = msg.data

    def _cb_vision_error(self, msg: PointStamped):
        self.vision_err_x = msg.point.x
        self.vision_err_y = msg.point.y
        self.vision_radius = msg.point.z
        self.vision_last_seen_t = time.time()
        self._refresh_blind_block()
        self.step_seen_target_once = True

        d = math.hypot(float(self.vision_err_x), float(self.vision_err_y))
        r = 0.0 if self.vision_radius is None else float(self.vision_radius)
        self.last_good_circle_t = time.time()
        self.last_good_circle_dist_px = d
        self.last_good_circle_radius = r
        self.last_good_circle_err_x = float(self.vision_err_x)
        self.last_good_circle_err_y = float(self.vision_err_y)

    def _cb_vision_center(self, msg: PointStamped):
        self.vision_center_x = msg.point.x
        self.vision_center_y = msg.point.y

    def _cb_line_found(self, msg: Bool):
        self.line_found = msg.data
        if msg.data:
            self.line_last_seen_t = time.time()
        else:
            self._reset_line_pid()

    def _cb_line_error(self, msg: PointStamped):
        self.line_err_x = msg.point.x
        self.line_angle_deg = msg.point.y
        self.line_half_width = msg.point.z
        self.line_last_seen_t = time.time()

    def _cb_line_status(self, msg: String):
        self.line_status = msg.data

    # ---------- Mission control ----------
    def _auto_start(self):
        with self.auto_start_lock:
            if self.mission_running or self.landing:
                return

            if not self.airborne:
                for _ in range(5):
                    self._takeoff()
                    if self.airborne:
                        break
                    if self.current_mode != 'GUIDED' or self.landing:
                        return
                    time.sleep(0.5)

            if self.airborne and not self.mission_running and not self.landing:
                self._run_full_vision_mission()

    def _takeoff(self):
        if not self.ready:
            self.get_logger().warn('Not ready!')
            return
        if self.airborne:
            self.get_logger().warn('Already airborne!')
            return
        if not self._flow_is_healthy():
            self.get_logger().warn('Optical Flow not ready -> reject takeoff')
            return
        if not self.camera_alive:
            self.get_logger().warn('Camera not ready -> reject takeoff')
            return
        if self.current_mode != 'GUIDED':
            if not self._set_mode(MODE_GUIDED):
                self.get_logger().warn('Failed to enter GUIDED!')
                return
        time.sleep(0.5)
        if not self.armed:
            if not self._arm():
                self.get_logger().warn('Arm failed!')
                return
            time.sleep(1.0)

        self.phase = 'TAKEOFF'
        self.get_logger().info(f'Taking off to {self.takeoff_alt:.1f}m ...')
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, float('nan'), 0, 0, self.takeoff_alt,
        )

        t0 = time.time()
        max_lidar = 0.0
        had_valid_lidar = False

        while time.time() - t0 < self.takeoff_cmd_timeout_s:
            if self.current_mode != 'GUIDED' or not self.armed:
                return
            lh = self.lidar_height
            if lh is not None and lh > 0.25:
                had_valid_lidar = True
                max_lidar = max(max_lidar, float(lh))
                if max_lidar >= self.takeoff_complete_alt_m:
                    break
            time.sleep(0.25)

        if had_valid_lidar and max_lidar < self.takeoff_complete_alt_m:
            if max_lidar >= self.takeoff_fallback_alt_m:
                self.get_logger().warn(
                    f'Takeoff fallback: lidar max {max_lidar:.2f}m < complete threshold {self.takeoff_complete_alt_m:.2f}m -> continue mission'
                )
            else:
                self.get_logger().warn(
                    f'Takeoff incomplete: lidar max {max_lidar:.2f}m -> keep continuing mission carefully'
                )
        elif not had_valid_lidar:
            self.get_logger().warn('Takeoff completed without reliable lidar confirmation')

        self.phase = 'PRE_SEARCH_HOVER'
        self._hold_body(self.hover_before_search)
        self.airborne = True
        self.get_logger().info('Takeoff complete ✅')

    def _run_full_vision_mission(self):
        try:
            if not self.airborne or self.landing or self.current_mode != 'GUIDED':
                self.get_logger().warn('Cannot start mission in current state')
                return

            self.mission_running = True
            self.completed_targets = 0
            self._clear_vision_memory()
            self._set_mission_step(MISSION_RED1, reset_lock=True)

            while self.mission_step != MISSION_COMPLETE:
                result = self._run_one_mission_step()

                if result == 'done_one':
                    self.get_logger().info(
                        f'MISSION DONE step={self.mission_step} completed={self.completed_targets + 1}/{self.total_targets}'
                    )
                    adv = self._advance_mission_after_done()
                    if adv == 'complete':
                        break
                    continue

                if result == 'takeover':
                    self.mission_running = False
                    self.phase = 'PILOT_TAKEOVER'
                    return

                if result == 'abort':
                    self.mission_running = False
                    self.phase = 'ABORTED'
                    return

            self.get_logger().info('Hoàn thành 5 target ✅ -> auto land')
            self.mission_running = False
            self.phase = 'MISSION_COMPLETE'
            self._land()

        except Exception as e:
            import traceback
            self.get_logger().error(f'_run_full_vision_mission crashed: {e}')
            self.get_logger().error(traceback.format_exc())
            self.mission_running = False
            self.phase = 'MISSION_CRASH'

    def _run_one_mission_step(self):
        self._reset_target_progress()

        while True:
            if self.vision_profile_dirty:
                self._publish_vision_profile(reset_lock=False)

            self.phase = f'{self.mission_step}_GUIDE'

            if self.current_mode != 'GUIDED':
                return 'takeover'
            if not self.armed or self.landing:
                return 'abort'

            of_wait_t0 = time.time()
            while time.time() - of_wait_t0 < 2.0:
                if self._flow_is_healthy():
                    break
                time.sleep(0.1)
            if not self._flow_is_healthy():
                self.get_logger().warn('Optical Flow not ready after 2s -> abort mission step')
                return 'abort'

            # confirmed target: track / stable count
            if self._circle_takeover_ready():
                ex = float(self.vision_err_x)
                ey = float(self.vision_err_y)
                d = math.hypot(ex, ey)

                if self.vision_aligned:
                    self.phase = f'{self.mission_step}_STABLE_COUNT'
                    self._send_body_velocity(0.0, 0.0, 0.0, 0.0)
                    self._reset_pid()
                    self.stable_frame_count += 1

                    self.get_logger().info(
                        f'STABLE FRAME {self.stable_frame_count}/{self.stable_frame_required} '
                        f'step={self.mission_step} color={self.vision_current_color} d={d:.1f} '
                        f'ex={ex:.1f} ey={ey:.1f}'
                    )

                    if self.stable_frame_count >= self.stable_frame_required:
                        mode = self._vision_mode_to_commit_string()
                        self._commit_vision_target(mode)
                        self._publish_vision_profile(reset_lock=True)
                        self.get_logger().info(f'STEP COMPLETE BY 30 FRAMES -> {mode}')
                        return 'done_one'

                    time.sleep(0.10)
                    continue

                # Not aligned: reset stable count and keep tracking/settling.
                self.stable_frame_count = 0

                if not self.in_settle_mode:
                    if d <= self.settle_enter_dist_px:
                        self.in_settle_mode = True
                else:
                    if d > self.settle_exit_dist_px:
                        self.in_settle_mode = False
                        self._reset_pid()

                if self.in_settle_mode:
                    self.phase = f'{self.mission_step}_SETTLE_PID'
                    vx_cmd, vy_cmd = self._compute_settle_pid_velocity(ex, ey)
                    self._send_body_velocity(vx_cmd, vy_cmd, 0.0, 0.0)
                    time.sleep(0.10)
                    continue

                self._reset_pid()
                self.phase = f'{self.mission_step}_TRACK_CIRCLE'
                vx_cmd = self.vision_y_to_vx_sign * self.track_kp_y * ey
                vy_cmd = self.vision_x_to_vy_sign * self.track_kp_x * ex
                vx_cmd = self._clamp(vx_cmd, -self.track_max_speed_mps, self.track_max_speed_mps)
                vy_cmd = self._clamp(vy_cmd, -self.track_max_speed_mps, self.track_max_speed_mps)
                self._send_body_velocity(vx_cmd, vy_cmd, 0.0, 0.0)
                time.sleep(0.10)
                continue

            # Line guidance first, only if no circle is confirmed.
            if self._should_use_line_guidance():
                self.phase = f'{self.mission_step}_LINE_GUIDE'
                self._reset_target_progress()
                vx_cmd, vy_cmd = self._compute_line_guidance_velocity()
                self._send_body_velocity(vx_cmd, vy_cmd, 0.0, 0.0)
                time.sleep(0.12)
                continue

            # Candidate / recover / hold / blind search
            if not self._have_confirmed_target():
                self.stable_frame_count = 0

                if self._candidate_pending_recent():
                    self.phase = f'{self.mission_step}_CANDIDATE_HOLD'
                    self._send_body_velocity(0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.10)
                    continue

                if self._circle_close_recently():
                    self.phase = f'{self.mission_step}_CIRCLE_RECOVER'
                    vx_cmd, vy_cmd = self._compute_circle_recover_velocity()
                    self._send_body_velocity(vx_cmd, vy_cmd, 0.0, 0.0)
                    time.sleep(0.10)
                    continue

                if self._blind_is_blocked():
                    self.phase = f'{self.mission_step}_TARGET_HOLD'
                    self._send_body_velocity(0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.10)
                    continue

                vx_cmd, vy_cmd = self._compute_search_velocity()
                self.phase = f'{self.mission_step}_SEARCH_{self.search_mode}'
                self._reset_pid()
                self._send_body_velocity(vx_cmd, vy_cmd, 0.0, self.search_yaw_rate)
                time.sleep(0.12)
                continue

            time.sleep(0.12)

    def _land(self):
        if not self.airborne or self.landing:
            return
        self.landing = True
        self.mission_running = False
        self.phase = 'LAND'
        self.get_logger().info('Landing...')
        self._set_mode(MODE_LAND)
        t0 = time.time()
        while time.time() - t0 < 30.0:
            if self.lidar_height is not None and self._is_landed():
                time.sleep(1.0)
                break
            if not self.armed:
                break
            time.sleep(0.3)
        self._on_landed()

    def _log_status(self):
        err_txt = 'N/A' if self.vision_err_x is None else f'({self.vision_err_x:.1f},{self.vision_err_y:.1f})'
        flow_txt = self.flow_quality if self.flow_quality is not None else 'N/A'
        self.get_logger().info(
            f'phase={self.phase} mode={self.current_mode} armed={self.armed} airborne={self.airborne} '
            f'mission_step={self.mission_step} search_mode={self.search_mode} '
            f'progress={self.completed_targets}/{self.total_targets} '
            f'stable={self.stable_frame_count}/{self.stable_frame_required} '
            f'flow_q={flow_txt} flow_ok={self._flow_is_healthy()} lidar={self._lidar_info()} '
            f'cam_alive={self.camera_alive} vision_found={self.vision_target_found} '
            f'vision_aligned={self.vision_aligned} color={self.vision_current_color} err={err_txt} '
            f'line_found={self.line_found} line_ex={self.line_err_x:.1f} line_angle={self.line_angle_deg:.1f} '
            f'step_seen={self.step_seen_target_once}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = AutoControlNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    exec_thread = threading.Thread(target=executor.spin)
    exec_thread.daemon = True
    exec_thread.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if node.master:
            node._restore_original_params()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
