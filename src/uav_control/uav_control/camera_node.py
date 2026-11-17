#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('image_topic', '/cam/front/image_raw')
        self.declare_parameter('grab_count', 0)

        self.camera_index = int(self.get_parameter('camera_index').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.grab_count = int(self.get_parameter('grab_count').value)

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.alive_pub = self.create_publisher(Bool, '/camera/raw_alive', 10)
        self.info_pub = self.create_publisher(String, '/camera/raw_info', 10)

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {self.camera_index}')

        self.frame_counter = 0
        self.last_log_t = time.time()
        self.last_fps_t = time.time()
        self.last_fps_counter = 0
        self.current_fps = 0.0

        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period, self.read_and_publish)

        self.get_logger().info(
            f'camera_node started | cam={self.camera_index} size={self.frame_width}x{self.frame_height} '
            f'fps={self.fps:.1f} topic={self.image_topic}'
        )

    def publish_alive(self, ok: bool):
        msg = Bool()
        msg.data = ok
        self.alive_pub.publish(msg)

    def publish_info(self, text: str):
        msg = String()
        msg.data = text
        self.info_pub.publish(msg)

    def publish_image(self, frame):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'front_camera'
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = int(frame.shape[1] * frame.shape[2])
        msg.data = frame.tobytes()
        self.image_pub.publish(msg)

    def read_and_publish(self):
        for _ in range(max(self.grab_count, 0)):
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.publish_alive(False)
            self.publish_info('camera_read_failed')
            self.get_logger().warn('camera read failed')
            return

        self.frame_counter += 1
        now = time.time()
        dt = now - self.last_fps_t
        if dt >= 1.0:
            self.current_fps = (self.frame_counter - self.last_fps_counter) / dt
            self.last_fps_counter = self.frame_counter
            self.last_fps_t = now

        self.publish_image(frame)
        self.publish_alive(True)

        info = (
            f'camera_ok frames={self.frame_counter} fps={self.current_fps:.1f} '
            f'size={frame.shape[1]}x{frame.shape[0]}'
        )
        self.publish_info(info)

        if now - self.last_log_t >= 1.0:
            self.last_log_t = now
            self.get_logger().info(info)

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
