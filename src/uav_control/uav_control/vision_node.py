#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class VisionNode(Node):
    """
    vision_node.py - bản ưu tiên ổn định/nhanh theo V1.0.py

    Giữ ROS2 I/O cho auto_control_node:
      /vision/target_found
      /vision/target_error
      /vision/target_center
      /vision/aligned
      /vision/current_color
      /vision/status
      /vision/target_bbox

    Logic xử lý ảnh chính lấy theo V1.0.py:
      SEQUENCE: red -> yellow -> red -> yellow -> blue
      HSV: red/yellow/blue như V1.0
      mask -> morphology open/close -> contour -> minEnclosingCircle
      lock target -> ROI tracking -> lost timeout
      select theo vị trí: red1 thấp, yellow trái, red2 cao hơn red1, yellow phải, blue giữa dưới

    Ghi chú:
      - Vision chỉ publish target/error/aligned. Drop/chuyển bước vẫn do auto_control_node quyết định.
      - Khi auto_control_node commit_target, vision sẽ lưu target đã hoàn thành vào memory để chọn target kế tiếp.
      - Line detection được tắt để nhẹ và ổn định hơn; vẫn publish /vision/line_found=False để không lỗi topic.
    """

    def detect_white_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Bắt vùng trắng/xám sáng: saturation thấp, value cao
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.WHITE_V_MIN]),
            np.array([180, self.WHITE_S_MAX, 255])
        )

        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        edges = cv2.Canny(white_mask, 30, 100)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.LINE_THRESHOLD,
            minLineLength=self.LINE_MIN_LENGTH,
            maxLineGap=self.LINE_MAX_GAP,
        )

        result: List[Tuple[int, int, int, int]] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length < self.LINE_MIN_LENGTH:
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                # Giữ cả line ngang/dọc/chéo nhẹ, vì camera nhìn xiên nên line không nhất thiết thẳng ngang
                if abs(angle) < 45 or abs(abs(angle) - 90) < 45:
                    result.append((int(x1), int(y1), int(x2), int(y2)))

        return result

    def publish_line_result(self, line: Optional[Tuple[int, int, int, int]], frame: np.ndarray):
        if line is None:
            self.publish_no_line()
            return

        x1, y1, x2, y2 = line
        mid_x = 0.5 * (x1 + x2)
        mid_y = 0.5 * (y1 + y2)
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        length = math.hypot(x2 - x1, y2 - y1)

        lateral_px = float(mid_x - self.cx)

        fmsg = Bool()
        fmsg.data = True
        self.line_found_pub.publish(fmsg)

        emsg = PointStamped()
        emsg.header.stamp = self.get_clock().now().to_msg()
        emsg.header.frame_id = 'camera'
        emsg.point.x = lateral_px
        emsg.point.y = angle_deg
        emsg.point.z = length
        self.line_error_pub.publish(emsg)

        smsg = String()
        smsg.data = f'line lateral={lateral_px:.1f} angle={angle_deg:.1f} len={length:.1f}'
        self.line_status_pub.publish(smsg)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.circle(frame, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
        cv2.putText(frame, f'LINE ex={lateral_px:.1f}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def __init__(self):
        super().__init__('vision_node')

        # ==================== Parameters ====================
        self.declare_parameter('save_debug_video', True)
        self.declare_parameter('debug_video_path', '')
        self.declare_parameter('debug_video_fps', 20.0)
        self.declare_parameter('log_dir', '/ws/logs/vision_runs')

        self.declare_parameter('image_topic', '/cam/front/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/raw_info')
        self.declare_parameter('desired_target_mode_topic', '/vision/desired_target_mode')
        self.declare_parameter('reset_target_topic', '/vision/reset_target')
        self.declare_parameter('commit_target_topic', '/vision/commit_target')
        self.declare_parameter('clear_mission_memory_topic', '/vision/clear_mission_memory')

        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('process_hz', 20.0)
        self.declare_parameter('debug_view', False)
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('debug_image_topic', '/vision/debug_image')

        self.save_debug_video = bool(self.get_parameter('save_debug_video').value)
        self.debug_video_path = str(self.get_parameter('debug_video_path').value)
        self.debug_video_fps = float(self.get_parameter('debug_video_fps').value)
        self.log_dir = str(self.get_parameter('log_dir').value)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.desired_target_mode_topic = str(self.get_parameter('desired_target_mode_topic').value)
        self.reset_target_topic = str(self.get_parameter('reset_target_topic').value)
        self.commit_target_topic = str(self.get_parameter('commit_target_topic').value)
        self.clear_mission_memory_topic = str(self.get_parameter('clear_mission_memory_topic').value)

        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.process_hz = float(self.get_parameter('process_hz').value)
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)

        # ==================== V1.0 CONFIG ====================
        self.SEQUENCE = ['red', 'yellow', 'red', 'yellow', 'blue']
        self.MODE_TO_SEQ_INDEX = {
            'red_1': 0,
            'yellow_left': 1,
            'red_2': 2,
            'yellow_right': 3,
            'blue': 4,
        }
        self.SEQ_LABELS = ['red1', 'yellow1', 'red2', 'yellow2', 'blue']

        self.CENTER_THRESHOLDS = {
            'red': 80,
            'yellow': 50,
            'blue': 10,
        }

        self.STABLE_TIME = 0.5
        self.MAX_LOST_TIME = 1.0
        self.MIN_AREA = 500
        self.ROI_SIZE = 150
        self.OFFSET_Y = -50

        self.LOWER_BLUE = np.array([110, 180, 120])
        self.UPPER_BLUE = np.array([125, 255, 255])

        self.LOWER_YELLOW = np.array([20, 100, 100])
        self.UPPER_YELLOW = np.array([35, 255, 255])

        self.LOWER_RED1 = np.array([0, 100, 100])
        self.UPPER_RED1 = np.array([10, 255, 255])
        self.LOWER_RED2 = np.array([170, 100, 100])
        self.UPPER_RED2 = np.array([180, 255, 255])

        self.LOCK_RADIUS = 50
        self.IGNORE_RADIUS = 80

        # ROS2 publish smoothing, nhẹ thôi để tránh giật dữ liệu cho auto_control
        self.CENTER_MEDIAN_LEN = 3
        self.CENTER_RATE_LIMIT_PX = 6.0
        self.RADIUS_RATE_LIMIT_PX = 5.0
        self.ERR_DEADBAND_PX = 4.0
        self.CAMERA_INSIDE_MARGIN_PX = 6
        self.SWITCH_GUARD_S = 0.25

        self.DRAW_COLORS = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0),
        }

        # ==================== White line detection ====================
        self.LINE_ENABLE = True

        # Ngưỡng bắt line trắng/xám sáng
        self.WHITE_V_MIN = 170
        self.WHITE_S_MAX = 70
        self.LINE_THRESHOLD = 30
        self.LINE_MIN_LENGTH = 50
        self.LINE_MAX_GAP = 15

        # ==================== Runtime state ====================
        self.cx = self.frame_width // 2
        self.cy = self.frame_height // 2 + self.OFFSET_Y

        self.current_target_mode = 'idle'
        self.current_seq_index: Optional[int] = None
        self.target_color = 'unknown'
        self.target_side = 'any'

        self.latest_frame: Optional[np.ndarray] = None
        self.last_camera_info = 'camera_info_unknown'
        self.last_status_text = 'init'
        self.last_no_frame_warn_t = 0.0
        self.switch_block_until = 0.0

        self.locked_target: Optional[Tuple[int, int, int]] = None
        self.lost_time: Optional[float] = None

        self.prev_detections: Dict[str, Tuple[int, int, int]] = {}
        self.ignored_targets: List[Tuple[str, int, int]] = []

        self.center_hist: Deque[Tuple[float, float, float]] = deque(maxlen=self.CENTER_MEDIAN_LEN)
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        self.target_r: Optional[float] = None
        self.raw_measure_x: Optional[float] = None
        self.raw_measure_y: Optional[float] = None
        self.raw_measure_r: Optional[float] = None
        self.camera_inside_circle_state = False

        self.frame_counter = 0
        self.detect_counter = 0
        self.mode_change_counter = 0
        self.reset_counter = 0

        # ==================== Debug video ====================
        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.save_debug_video:
            if not self.debug_video_path:
                os.makedirs(self.log_dir, exist_ok=True)
                self.debug_video_path = os.path.join(self.log_dir, f'vision_debug_{run_stamp}.avi')
            else:
                parent = os.path.dirname(self.debug_video_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

        self.video_writer = None
        self.video_writer_ready = False

        # ==================== ROS pubs ====================
        self.target_found_pub = self.create_publisher(Bool, '/vision/target_found', 10)
        self.target_error_pub = self.create_publisher(PointStamped, '/vision/target_error', 10)
        self.target_center_pub = self.create_publisher(PointStamped, '/vision/target_center', 10)
        self.aligned_pub = self.create_publisher(Bool, '/vision/aligned', 10)
        self.target_bbox_pub = self.create_publisher(String, '/vision/target_bbox', 10)
        self.current_color_pub = self.create_publisher(String, '/vision/current_color', 10)
        self.status_pub = self.create_publisher(String, '/vision/status', 10)
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # Giữ topic line để auto_control không bị thiếu topic, nhưng bản V1.0 không dùng line.
        self.line_found_pub = self.create_publisher(Bool, '/vision/line_found', 10)
        self.line_error_pub = self.create_publisher(PointStamped, '/vision/line_error', 10)
        self.line_status_pub = self.create_publisher(String, '/vision/line_status', 10)

        # ==================== ROS subs ====================
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(String, self.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(String, self.desired_target_mode_topic, self.desired_target_mode_cb, 10)
        self.create_subscription(Bool, self.reset_target_topic, self.reset_target_cb, 10)
        self.create_subscription(String, self.commit_target_topic, self.commit_target_cb, 10)
        self.create_subscription(Bool, self.clear_mission_memory_topic, self.clear_mission_memory_cb, 10)

        self.create_timer(1.0 / max(self.process_hz, 1.0), self.process_frame)

        self.get_logger().info(
            f'vision_node V1.0-logic started | image_topic={self.image_topic} process_hz={self.process_hz:.1f} '
            f'video={self.save_debug_video} path={self.debug_video_path}'
        )

    # ==================== Mission memory ====================
    def _label_from_mode(self, mode: str) -> Optional[str]:
        mapping = {
            'red_1': 'red1',
            'yellow_left': 'yellow1',
            'red_2': 'red2',
            'yellow_right': 'yellow2',
            'blue': 'blue',
        }
        return mapping.get(mode)

    def _color_from_mode(self, mode: str) -> Optional[str]:
        idx = self.MODE_TO_SEQ_INDEX.get(mode)
        if idx is None:
            return None
        return self.SEQUENCE[idx]

    def commit_target_cb(self, msg: String):
        mode = (msg.data or '').strip().lower()
        label = self._label_from_mode(mode)
        color = self._color_from_mode(mode)
        if label is None or color is None:
            self.get_logger().warn(f'commit_target ignored: unknown mode {mode}')
            return
        if self.target_x is None or self.target_y is None or self.target_r is None:
            self.get_logger().warn(f'commit_target {mode} ignored: no current target center')
            return

        x = int(round(self.target_x))
        y = int(round(self.target_y))
        r = int(round(self.target_r))
        self.prev_detections[label] = (x, y, r)
        self.ignored_targets.append((color, x, y))
        self.get_logger().info(f'commit_target -> {label} @ ({x},{y}) r={r}')

        # giống V1.0: sau khi drop/commit thì bỏ lock để qua target mới
        self._reset_transient_target_state()

    def clear_mission_memory_cb(self, msg: Bool):
        if msg.data:
            self.prev_detections.clear()
            self.ignored_targets.clear()
            self.get_logger().info('mission memory cleared')

    # ==================== Basic helpers ====================
    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _get_camera_center(self) -> Tuple[int, int]:
        return self.cx, self.cy

    def _compute_circle_error(self, circle_x: int, circle_y: int, circle_r: int) -> Tuple[float, float, float, bool]:
        cam_x, cam_y = self._get_camera_center()
        raw_err_x = float(circle_x - cam_x)
        raw_err_y = float(cam_y - circle_y)
        err_x = 0.0 if abs(raw_err_x) <= self.ERR_DEADBAND_PX else raw_err_x
        err_y = 0.0 if abs(raw_err_y) <= self.ERR_DEADBAND_PX else raw_err_y
        dist_center = math.hypot(raw_err_x, raw_err_y)
        inside_thresh = max(0.0, float(circle_r) - self.CAMERA_INSIDE_MARGIN_PX)
        aligned = dist_center <= inside_thresh
        return err_x, err_y, dist_center, aligned

    def _reset_transient_target_state(self):
        self.locked_target = None
        self.lost_time = None
        self.center_hist.clear()
        self.target_x = None
        self.target_y = None
        self.target_r = None
        self.raw_measure_x = None
        self.raw_measure_y = None
        self.raw_measure_r = None
        self.camera_inside_circle_state = False

    def _publish_no_target(self, status: str, reset: bool = True):
        if reset:
            self._reset_transient_target_state()
        self.publish_found(False)
        self.publish_aligned(False)
        self.publish_status(status)

    def _update_control_center(self, x: float, y: float, r: float):
        self.raw_measure_x = float(x)
        self.raw_measure_y = float(y)
        self.raw_measure_r = float(r)
        self.center_hist.append((float(x), float(y), float(r)))

        xs = np.array([p[0] for p in self.center_hist], dtype=np.float32)
        ys = np.array([p[1] for p in self.center_hist], dtype=np.float32)
        rs = np.array([p[2] for p in self.center_hist], dtype=np.float32)
        mx = float(np.median(xs))
        my = float(np.median(ys))
        mr = float(np.median(rs))

        if self.target_x is None:
            self.target_x = mx
            self.target_y = my
            self.target_r = mr
            return

        self.target_x += self._clamp(mx - self.target_x, -self.CENTER_RATE_LIMIT_PX, self.CENTER_RATE_LIMIT_PX)
        self.target_y += self._clamp(my - self.target_y, -self.CENTER_RATE_LIMIT_PX, self.CENTER_RATE_LIMIT_PX)
        self.target_r += self._clamp(mr - self.target_r, -self.RADIUS_RATE_LIMIT_PX, self.RADIUS_RATE_LIMIT_PX)

    # ==================== ROS callbacks ====================
    def image_cb(self, msg: Image):
        try:
            if msg.encoding not in ('bgr8', 'rgb8'):
                return
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.latest_frame = frame.copy()
            self.frame_width = int(msg.width)
            self.frame_height = int(msg.height)
            self.cx = self.frame_width // 2
            self.cy = self.frame_height // 2 + self.OFFSET_Y
        except Exception as exc:
            self.get_logger().warn(f'image_cb error: {exc}')

    def camera_info_cb(self, msg: String):
        self.last_camera_info = msg.data

    def desired_target_mode_cb(self, msg: String):
        mode = (msg.data or 'idle').strip().lower()
        idx = self.MODE_TO_SEQ_INDEX.get(mode)
        if idx is None:
            mode = 'idle'
            color = 'unknown'
            side = 'any'
            seq_index = None
        else:
            color = self.SEQUENCE[idx]
            seq_index = idx
            if mode == 'yellow_left':
                side = 'left'
            elif mode == 'yellow_right':
                side = 'right'
            else:
                side = 'any'

        changed = (
            mode != self.current_target_mode or
            color != self.target_color or
            side != self.target_side
        )

        self.current_target_mode = mode
        self.current_seq_index = seq_index
        self.target_color = color
        self.target_side = side

        if changed:
            self.mode_change_counter += 1
            self._reset_transient_target_state()
            self.switch_block_until = time.time() + self.SWITCH_GUARD_S
            self.publish_status(f'target_mode_{self.current_target_mode}')
            self.get_logger().info(
                f'Target mode -> {self.current_target_mode} | seq={self.current_seq_index} color={self.target_color} side={self.target_side}'
            )

    def reset_target_cb(self, msg: Bool):
        if msg.data:
            self.reset_counter += 1
            self._reset_transient_target_state()
            self.switch_block_until = time.time() + self.SWITCH_GUARD_S
            self.publish_status(f'reset_target_{self.current_target_mode}')
            self.get_logger().info(f'Reset target state for mode {self.current_target_mode}')

    # ==================== V1.0 image processing ====================
    def get_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        if color == 'blue':
            mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
            _, s, _ = cv2.split(hsv)
            mask_s = cv2.inRange(s, 120, 255)
            mask = mask & mask_s
            return mask

        if color == 'yellow':
            return cv2.inRange(hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)

        if color == 'red':
            return cv2.inRange(hsv, self.LOWER_RED1, self.UPPER_RED1) + \
                   cv2.inRange(hsv, self.LOWER_RED2, self.UPPER_RED2)

        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_targets(self, mask: np.ndarray) -> List[Tuple[int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: List[Tuple[int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA:
                continue

            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r <= 1e-6:
                continue

            circ = area / (np.pi * r * r + 1e-6)
            if 0.7 < circ < 1.2 and 10 < r < 150:
                result.append((int(x), int(y), int(r)))

        return result

    def select_nearest(self, targets: List[Tuple[int, int, int]], ref: Tuple[int, int],
                       color: Optional[str] = None, ignored: Optional[List[Tuple[str, int, int]]] = None) -> Optional[Tuple[int, int, int]]:
        if not targets:
            return None

        rx, ry = ref
        best = None
        min_d = 1e18

        for x, y, r in targets:
            if ignored and color:
                skip = False
                for ic, ix, iy in ignored:
                    if ic == color and (x - ix) ** 2 + (y - iy) ** 2 <= self.IGNORE_RADIUS ** 2:
                        skip = True
                        break
                if skip:
                    continue

            d = (x - rx) ** 2 + (y - ry) ** 2
            if d < min_d:
                min_d = d
                best = (x, y, r)

        return best

    def select_locked(self, targets: List[Tuple[int, int, int]], locked: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        if not targets or locked is None:
            return None

        lx, ly, lr = locked
        best = None
        best_score = 1e18

        for x, y, r in targets:
            d = (x - lx) ** 2 + (y - ly) ** 2
            if d > self.LOCK_RADIUS ** 2:
                continue

            dr = abs(r - lr)
            score = d + 5 * dr ** 2

            if score < best_score:
                best_score = score
                best = (x, y, r)

        return best

    def has_white_stripe(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        h, w = frame.shape[:2]
        x1 = max(0, int(x - r * 0.35))
        x2 = min(w, int(x + r * 0.35))
        y1 = min(h, int(y + r * 0.35))
        y2 = min(h, int(y + r * 1.05))

        if x1 >= x2 or y1 >= y2:
            return False

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        white_pixels = cv2.countNonZero(white_mask)
        area = roi.shape[0] * roi.shape[1]
        return white_pixels > 0.15 * area

    def select_by_sequence(self, frame: np.ndarray, targets: List[Tuple[int, int, int]], current_color: str,
                           seq_index: int, prev_detections: Dict[str, Tuple[int, int, int]],
                           frame_w: int, frame_h: int, ignored: List[Tuple[str, int, int]]) -> Optional[Tuple[int, int, int]]:
        if not targets:
            return None

        def choose(candidates: List[Tuple[int, int, int]], ref: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
            if not candidates:
                return None
            return self.select_nearest(candidates, ref, color=current_color, ignored=ignored)

        if current_color == 'red' and seq_index == 0:
            candidates = [t for t in targets if self.has_white_stripe(frame, t[0], t[1], t[2])]
            if candidates:
                return choose(candidates, (frame_w // 2, frame_h * 4 // 5))
            return choose(targets, (frame_w // 2, frame_h * 4 // 5))

        if current_color == 'yellow' and seq_index == 1:
            candidates = [t for t in targets if t[0] < frame_w * 0.33]
            return choose(candidates, (frame_w // 4, frame_h // 2)) or choose(targets, (frame_w // 4, frame_h // 2))

        if current_color == 'red' and seq_index == 2:
            red1 = prev_detections.get('red1')
            candidates = list(targets)
            if red1:
                candidates = [t for t in candidates if t[1] < red1[1]]
            return choose(candidates, (frame_w // 2, frame_h // 5)) or choose(targets, (frame_w // 2, frame_h * 2 // 5))

        if current_color == 'yellow' and seq_index == 3:
            candidates = [t for t in targets if t[0] > frame_w * 0.66]
            return choose(candidates, (frame_w * 3 // 4, frame_h // 2)) or choose(targets, (frame_w * 3 // 4, frame_h // 2))

        if current_color == 'blue' and seq_index == 4:
            return choose(targets, (frame_w // 2, frame_h * 3 // 4))

        return choose(targets, (frame_w // 2, frame_h // 2))

    def _get_targets_like_v1(self, frame: np.ndarray, hsv: np.ndarray, current_color: str) -> List[Tuple[int, int, int]]:
        frame_h, frame_w = frame.shape[:2]

        if self.locked_target:
            x0, y0, _ = self.locked_target
            x1 = max(0, x0 - self.ROI_SIZE)
            y1 = max(0, y0 - self.ROI_SIZE)
            x2 = min(frame_w, x0 + self.ROI_SIZE)
            y2 = min(frame_h, y0 + self.ROI_SIZE)

            if x2 <= x1 or y2 <= y1:
                self.locked_target = None
                mask = self.clean_mask(self.get_mask(hsv, current_color))
                return self.find_targets(mask)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                self.locked_target = None
                mask = self.clean_mask(self.get_mask(hsv, current_color))
                return self.find_targets(mask)

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = self.clean_mask(self.get_mask(hsv_roi, current_color))
            targets = self.find_targets(mask)
            return [(tx + x1, ty + y1, r) for (tx, ty, r) in targets]

        mask = self.clean_mask(self.get_mask(hsv, current_color))
        return self.find_targets(mask)

    # ==================== Publishers ====================
    def publish_found(self, found: bool):
        msg = Bool(); msg.data = found; self.target_found_pub.publish(msg)

    def publish_aligned(self, aligned: bool):
        msg = Bool(); msg.data = aligned; self.aligned_pub.publish(msg)

    def publish_current_color(self, color: str):
        msg = String(); msg.data = color; self.current_color_pub.publish(msg)

    def publish_status(self, text: str):
        self.last_status_text = text
        msg = String(); msg.data = text; self.status_pub.publish(msg)

    def publish_error(self, err_x: float, err_y: float, radius: float):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.point.x = float(err_x)
        msg.point.y = float(err_y)
        msg.point.z = float(radius)
        self.target_error_pub.publish(msg)

    def publish_center(self, x: int, y: int, radius: float):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(radius)
        self.target_center_pub.publish(msg)

    def publish_bbox(self, x: int, y: int, r: int):
        bbox = f'x={x-r},y={y-r},w={2*r},h={2*r},r={r}'
        msg = String(); msg.data = bbox; self.target_bbox_pub.publish(msg)

    def publish_no_line(self):
        fmsg = Bool(); fmsg.data = False; self.line_found_pub.publish(fmsg)
        emsg = PointStamped()
        emsg.header.stamp = self.get_clock().now().to_msg()
        emsg.header.frame_id = 'camera'
        emsg.point.x = 0.0
        emsg.point.y = 0.0
        emsg.point.z = 0.0
        self.line_error_pub.publish(emsg)
        smsg = String(); smsg.data = 'line_disabled_v1'; self.line_status_pub.publish(smsg)

    # ==================== Debug drawing ====================
    def _draw_target_output(self, frame: np.ndarray, target: Tuple[int, int, int], current_color: str,
                            seq_index: int, err_x: float, err_y: float):
        frame_h, _ = frame.shape[:2]
        tx, ty, tr = target
        cv2.circle(frame, (tx, ty), tr, (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (255, 255, 0), -1)

        target_label = self.SEQ_LABELS[seq_index]
        cv2.putText(frame, target_label, (tx + tr + 10, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'dx={err_x:.1f} dy={err_y:.1f}', (10, frame_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'R={tr}', (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.drawMarker(frame, (self.cx, self.cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.line(frame, (self.cx - 20, self.cy), (self.cx + 20, self.cy), (255, 0, 255), 1)
        cv2.line(frame, (self.cx, self.cy - 20), (self.cx, self.cy + 20), (255, 0, 255), 1)

        cv2.putText(frame, current_color, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def publish_debug_frame(self, frame: np.ndarray):
        if self.save_debug_video:
            self._ensure_video_writer(frame)
            if self.video_writer is not None:
                self.video_writer.write(frame)

        if self.publish_debug_image:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.height = int(frame.shape[0])
            msg.width = int(frame.shape[1])
            msg.encoding = 'bgr8'
            msg.is_bigendian = False
            msg.step = int(frame.shape[1] * frame.shape[2])
            msg.data = frame.tobytes()
            self.debug_image_pub.publish(msg)

        if self.debug_view:
            try:
                cv2.imshow('VISION DEBUG V1', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.debug_view = False
                    cv2.destroyAllWindows()
            except cv2.error:
                self.debug_view = False

    def _ensure_video_writer(self, frame: np.ndarray):
        if not self.save_debug_video:
            return
        if self.video_writer_ready and self.video_writer is not None:
            return

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.debug_video_path,
            fourcc,
            float(self.debug_video_fps),
            (w, h),
        )

        if self.video_writer is None or not self.video_writer.isOpened():
            self.get_logger().error(f'Cannot open debug video writer: {self.debug_video_path}')
            self.video_writer = None
            self.video_writer_ready = False
            return

        self.video_writer_ready = True
        self.get_logger().info(f'Debug video recording -> {self.debug_video_path}')

    # ==================== Main loop ====================
    def process_frame(self):
        self.publish_current_color(self.target_color)
        # Line detection bật lại để auto có thể đi theo line trắng trước khi thấy vòng tròn
        line_frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if self.latest_frame is None:
            now = time.time()
            if now - self.last_no_frame_warn_t >= 1.0:
                self.last_no_frame_warn_t = now
                self.get_logger().warn('vision waiting for image...')
            self.publish_found(False)
            self.publish_aligned(False)
            self.publish_status('no_frame_yet')
            return

        self.frame_counter += 1
        frame = self.latest_frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # giống V1.0: blur trước xử lý
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.LINE_ENABLE:
            lines = self.detect_white_lines(frame)
            best_line = None
            best_len = -1.0
            for ln in lines:
                x1, y1, x2, y2 = ln
                length = math.hypot(x2 - x1, y2 - y1)
                if length > best_len:
                    best_len = length
                    best_line = ln
            self.publish_line_result(best_line, frame)
        else:
            self.publish_no_line() 

        if self.current_seq_index is None or self.target_color == 'unknown':
            self._publish_no_target(f'idle_waiting_target_mode_{self.current_target_mode}')
            cv2.drawMarker(frame, (self.cx, self.cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, 'IDLE - V1 vision ready', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.publish_debug_frame(frame)
            return

        if time.time() < self.switch_block_until:
            self._publish_no_target(f'switch_guard_{self.current_target_mode}')
            cv2.putText(frame, f'SWITCH GUARD {self.current_target_mode}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.publish_debug_frame(frame)
            return

        current_color = self.SEQUENCE[self.current_seq_index]
        th = self.CENTER_THRESHOLDS[current_color]

        targets = self._get_targets_like_v1(frame, hsv, current_color)

        x: Optional[int] = None
        y: Optional[int] = None

        # LOCK + TRACK giống V1.0
        if self.locked_target is None:
            self.locked_target = self.select_by_sequence(
                frame, targets, current_color, self.current_seq_index,
                self.prev_detections, frame_w, frame_h, self.ignored_targets,
            )
            if self.locked_target:
                x, y = self.locked_target[0], self.locked_target[1]
            else:
                x, y = None, None
        else:
            new_target = self.select_locked(targets, self.locked_target)

            if new_target:
                self.locked_target = new_target
                self.lost_time = None
                x, y = new_target[0], new_target[1]
            else:
                x, y = None, None

                if self.lost_time is None:
                    self.lost_time = time.time()
                elif time.time() - self.lost_time > self.MAX_LOST_TIME:
                    self.locked_target = None
                    self.lost_time = None

        if self.locked_target and x is not None and y is not None:
            tx, ty, tr = self.locked_target
            self._update_control_center(tx, ty, tr)

            sx = int(round(self.target_x))
            sy = int(round(self.target_y))
            sr = int(round(self.target_r))

            err_x, err_y, dist_center, inside_circle = self._compute_circle_error(sx, sy, sr)
            self.camera_inside_circle_state = inside_circle

            # Aligned dùng threshold V1.0 theo màu để hợp auto_control hơn.
            aligned = abs(err_x) < th and abs(err_y) < th

            self.publish_found(True)
            self.detect_counter += 1
            self.publish_center(sx, sy, sr)
            self.publish_error(err_x, err_y, sr)
            self.publish_bbox(sx, sy, sr)
            self.publish_aligned(aligned)

            status = (
                f'{self.current_target_mode} color={current_color} '
                f'err_x={err_x:.1f} err_y={err_y:.1f} r={sr} '
                f'd={dist_center:.1f} aligned={aligned}'
            )
            self.publish_status(status)

            self._draw_target_output(frame, (sx, sy, sr), current_color, self.current_seq_index, err_x, err_y)
            cv2.putText(frame, f'aligned={aligned} th={th}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f'mode={self.current_target_mode}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f'prev={list(self.prev_detections.keys())}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            self.publish_debug_frame(frame)
            return

        # No target
        self.publish_found(False)
        self.publish_aligned(False)
        self.publish_status(f'no_target_{self.current_target_mode}_{current_color}')
        cv2.drawMarker(frame, (self.cx, self.cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f'NO TARGET {current_color}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'mode={self.current_target_mode}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self.publish_debug_frame(frame)

    # ==================== Shutdown ====================
    def destroy_node(self):
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info('Debug video writer released')
        except Exception as exc:
            try:
                self.get_logger().warn(f'video_writer release error: {exc}')
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
