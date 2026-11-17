#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from smbus2 import SMBus

# =========================
# PCA9685 low-level config
# =========================
I2C_ADDRESS = 0x40
PWM_FREQUENCY = 50
CANDIDATE_BUS_IDS = [3, 5, 2, 0, 1]

PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
LED0_ON_L = 0x06

# =========================
# Mission mapping
# 5 target drops from V1.0 mission sequence:
# red_1 -> yellow_left -> red_2 -> yellow_right -> blue
#
# PCA9685 channel mapping:
# red_1        -> CH4
# yellow_left  -> CH5
# red_2        -> CH6
# yellow_right -> CH7
# blue         -> CH8
#
# Nếu phần cứng của bạn CHỈ có CH4-CH7 thì không đủ 5 ngăn riêng.
# Khi đó có thể tạm cho blue dùng lại CH7, nhưng mặc định dưới đây dùng CH8 cho đúng 5 lần thả riêng.
# =========================
SERVO_CHANNEL_MAP: Dict[str, int] = {
    'red_1': 4,
    'yellow_left': 5,
    'red_2': 6,
    'yellow_right': 7,
    'blue': 8,
}

# Pulse/angle tuning
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
SERVO_CLOSE_ANGLE = 0.0
SERVO_OPEN_ANGLE = 38.0
SERVO_OPEN_HOLD_S = 1.3


def detect_pca9685_bus(address: int = I2C_ADDRESS, candidates=CANDIDATE_BUS_IDS) -> Optional[int]:
    for bus_id in candidates:
        dev = f'/dev/i2c-{bus_id}'
        if not os.path.exists(dev):
            continue

        bus = None
        try:
            bus = SMBus(bus_id)
            _ = bus.read_byte_data(address, PCA9685_MODE1)
            bus.close()
            return bus_id
        except Exception:
            try:
                if bus is not None:
                    bus.close()
            except Exception:
                pass

    return None


class PCA9685:
    def __init__(self, bus_id: int, address: int = I2C_ADDRESS, frequency: int = PWM_FREQUENCY):
        self.bus_id = bus_id
        self.address = address
        self.frequency = frequency
        self.bus = SMBus(bus_id)

        self.write8(PCA9685_MODE1, 0x00)
        time.sleep(0.01)
        self.set_pwm_freq(frequency)

    def write8(self, reg: int, value: int):
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def read8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq_hz: int):
        prescaleval = 25000000.0 / 4096.0 / float(freq_hz) - 1.0
        prescale = int(prescaleval + 0.5)

        oldmode = self.read8(PCA9685_MODE1)
        sleep_mode = (oldmode & 0x7F) | 0x10
        self.write8(PCA9685_MODE1, sleep_mode)
        self.write8(PCA9685_PRESCALE, prescale)
        self.write8(PCA9685_MODE1, oldmode)
        time.sleep(0.005)
        self.write8(PCA9685_MODE1, oldmode | 0xA1)

    def set_pwm(self, channel: int, on_tick: int, off_tick: int):
        if channel < 0 or channel > 15:
            raise ValueError(f'Invalid PCA9685 channel: {channel}')

        reg = LED0_ON_L + 4 * channel
        self.write8(reg + 0, on_tick & 0xFF)
        self.write8(reg + 1, (on_tick >> 8) & 0xFF)
        self.write8(reg + 2, off_tick & 0xFF)
        self.write8(reg + 3, (off_tick >> 8) & 0xFF)

    def release_channel(self, channel: int):
        self.set_pwm(channel, 0, 0)

    def close(self):
        self.bus.close()


class ServoDropper:
    def __init__(self, bus_id: int):
        self.pca = PCA9685(bus_id=bus_id, address=I2C_ADDRESS, frequency=PWM_FREQUENCY)

    def angle_to_ticks(self, angle_deg: float) -> int:
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        pulse_us = SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * (angle_deg / 180.0)
        period_us = 1_000_000.0 / PWM_FREQUENCY
        ticks = int((pulse_us / period_us) * 4096)
        return max(0, min(4095, ticks))

    def set_angle(self, channel: int, angle_deg: float):
        ticks = self.angle_to_ticks(angle_deg)
        self.pca.set_pwm(channel, 0, ticks)

    def close_servo(self, channel: int):
        self.set_angle(channel, SERVO_CLOSE_ANGLE)

    def open_servo(self, channel: int):
        self.set_angle(channel, SERVO_OPEN_ANGLE)

    def drop(self, channel: int, hold_s: float = SERVO_OPEN_HOLD_S):
        self.open_servo(channel)
        time.sleep(hold_s)
        self.close_servo(channel)

    def cleanup(self, release_pwm: bool = False):
        try:
            if release_pwm:
                for ch in sorted(set(SERVO_CHANNEL_MAP.values())):
                    self.pca.release_channel(ch)
        finally:
            self.pca.close()


class ServoDropperNode(Node):
    def __init__(self):
        super().__init__('servo_dropper_node')

        self.declare_parameter('fire_topic', '/dropper/fire')
        self.declare_parameter('done_topic', '/dropper/done')
        self.declare_parameter('busy_topic', '/dropper/busy')
        self.declare_parameter('force_bus_id', -1)
        self.declare_parameter('open_angle', SERVO_OPEN_ANGLE)
        self.declare_parameter('close_angle', SERVO_CLOSE_ANGLE)
        self.declare_parameter('open_hold_s', SERVO_OPEN_HOLD_S)
        self.declare_parameter('release_pwm_on_exit', False)

        self.fire_topic = str(self.get_parameter('fire_topic').value)
        self.done_topic = str(self.get_parameter('done_topic').value)
        self.busy_topic = str(self.get_parameter('busy_topic').value)
        self.force_bus_id = int(self.get_parameter('force_bus_id').value)
        self.open_angle = float(self.get_parameter('open_angle').value)
        self.close_angle = float(self.get_parameter('close_angle').value)
        self.open_hold_s = float(self.get_parameter('open_hold_s').value)
        self.release_pwm_on_exit = bool(self.get_parameter('release_pwm_on_exit').value)

        if self.force_bus_id >= 0:
            bus_id = self.force_bus_id
            if not os.path.exists(f'/dev/i2c-{bus_id}'):
                raise RuntimeError(f'force_bus_id={bus_id}, but /dev/i2c-{bus_id} does not exist')
        else:
            bus_id = detect_pca9685_bus()
            if bus_id is None:
                raise RuntimeError(f'Cannot find PCA9685 at 0x{I2C_ADDRESS:02X} on buses {CANDIDATE_BUS_IDS}')

        self.bus_id = bus_id
        self.dropper = ServoDropper(bus_id=bus_id)
        self.busy = False
        self.lock = threading.Lock()

        self.done_pub = self.create_publisher(String, self.done_topic, 10)
        self.busy_pub = self.create_publisher(Bool, self.busy_topic, 10)
        self.create_subscription(String, self.fire_topic, self.fire_cb, 10)

        self._init_servos_closed()
        self._publish_busy(False)

        self.get_logger().info(
            f'servo_dropper_node started | bus={self.bus_id} addr=0x{I2C_ADDRESS:02X} '
            f'map={SERVO_CHANNEL_MAP} open={self.open_angle:.1f} close={self.close_angle:.1f} hold={self.open_hold_s:.2f}s'
        )

    def _init_servos_closed(self):
        for mode, ch in SERVO_CHANNEL_MAP.items():
            try:
                self.dropper.set_angle(ch, self.close_angle)
                self.get_logger().info(f'Init close -> mode={mode} ch={ch}')
                time.sleep(0.12)
            except Exception as exc:
                self.get_logger().warn(f'Init servo mode={mode} CH{ch} failed: {exc}')

    def _publish_busy(self, value: bool):
        self.busy = value
        msg = Bool()
        msg.data = value
        self.busy_pub.publish(msg)

    def _normalize_mode(self, text: str) -> str:
        mode = (text or '').strip().lower()
        aliases = {
            'red1': 'red_1',
            'yellow1': 'yellow_left',
            'red2': 'red_2',
            'yellow2': 'yellow_right',
        }
        return aliases.get(mode, mode)

    def fire_cb(self, msg: String):
        mode = self._normalize_mode(msg.data)

        if mode not in SERVO_CHANNEL_MAP:
            self.get_logger().warn(f'Unknown fire command: {mode}')
            return

        with self.lock:
            if self.busy:
                self.get_logger().warn(f'Dropper busy, ignore command: {mode}')
                return
            self._publish_busy(True)

        thread = threading.Thread(target=self._fire_worker, args=(mode,), daemon=True)
        thread.start()

    def _fire_worker(self, mode: str):
        channel = SERVO_CHANNEL_MAP[mode]
        try:
            self.get_logger().info(f'FIRE -> mode={mode} ch={channel}')
            self.dropper.set_angle(channel, self.open_angle)
            time.sleep(self.open_hold_s)
            self.dropper.set_angle(channel, self.close_angle)

            done = String()
            done.data = mode
            self.done_pub.publish(done)
            self.get_logger().info(f'DONE -> mode={mode} ch={channel}')

        except Exception as exc:
            self.get_logger().error(f'Drop failed for mode={mode} ch={channel}: {exc}')

        finally:
            self._publish_busy(False)

    def destroy_node(self):
        try:
            self.dropper.cleanup(release_pwm=self.release_pwm_on_exit)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ServoDropperNode()
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
