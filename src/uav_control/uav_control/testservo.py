#!/usr/bin/env python3
# dieu_khien_xung_3ms.py

import time
import smbus2

# Địa chỉ I2C của PCA9685 (mặc định 0x40)
PCA9685_ADDR = 0x40

# Các thanh ghi quan trọng
MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06   # Thanh ghi ON low cho kênh 0
LED0_ON_H = 0x07   # Thanh ghi ON high cho kênh 0
LED0_OFF_L = 0x08  # Thanh ghi OFF low cho kênh 0
LED0_OFF_H = 0x09  # Thanh ghi OFF high cho kênh 0

class PCA9685:
    def __init__(self, bus=3, address=PCA9685_ADDR):
        """Khởi tạo PCA9685 trên bus I2C-3"""
        self.bus = smbus2.SMBus(bus)
        self.address = address
        self.reset()
        self.set_pwm_freq(50)  # 50Hz mặc định
    
    def reset(self):
        """Reset PCA9685 về trạng thái ban đầu"""
        self.write_byte(MODE1, 0x00)
        self.write_byte(MODE2, 0x04)  # OUTDRV = 1 (ngõ ra totem-pole)
        time.sleep(0.01)
    
    def write_byte(self, reg, value):
        """Ghi 1 byte vào thanh ghi"""
        self.bus.write_byte_data(self.address, reg, value)
    
    def set_pwm_freq(self, freq_hz):
        """Đặt tần số PWM (Hz)"""
        # Tính giá trị prescale
        prescale = int(25000000.0 / (4096 * freq_hz) - 0.5)
        prescale = max(3, min(255, prescale))
        
        # Chuyển sang chế độ sleep để set prescale
        old_mode = self.bus.read_byte_data(self.address, MODE1)
        self.write_byte(MODE1, (old_mode & 0x7F) | 0x10)  # Sleep
        self.write_byte(PRESCALE, prescale)
        self.write_byte(MODE1, old_mode)
        time.sleep(0.005)
        self.write_byte(MODE1, old_mode | 0x80)  # Restart
    
    def set_pulse_width_us(self, channel, pulse_width_us):
        """
        Đặt độ rộng xung cho kênh (đơn vị microseconds)
        Chu kỳ cố định 20,000 us (tần số 50Hz)
        """
        # Giới hạn pulse_width_us trong khoảng 0-20000us
        pulse_width_us = max(0, min(pulse_width_us, 20000))
        
        # Chuyển đổi sang giá trị off_time (0-4095)
        off_time = int((pulse_width_us / 20000.0) * 4096)
        
        # Ghi giá trị vào thanh ghi (on_time = 0)
        self.bus.write_i2c_block_data(
            self.address,
            LED0_ON_L + 4 * channel,
            [0x00, 0x00, off_time & 0xFF, (off_time >> 8) & 0xFF]
        )
    
    def set_pwm_high(self, channel, duration_ms):
        """
        Bật mức HIGH trên kênh trong khoảng thời gian duration_ms (ms)
        Sau đó tự động tắt (về LOW) khi hết thời gian
        """
        # Chuyển đổi ms sang us
        pulse_width_us = duration_ms * 1000
        
        # Giới hạn trong 20ms (20000us)
        if pulse_width_us > 20000:
            print(f"Cảnh báo: Độ rộng xung {duration_ms}ms vượt quá chu kỳ 20ms")
            pulse_width_us = 20000
        
        # Đặt độ rộng xung
        self.set_pulse_width_us(channel, pulse_width_us)
        print(f"Kênh {channel}: BẬT HIGH trong {duration_ms}ms")
    
    def set_pwm_low(self, channel):
        """Tắt kênh (mức LOW)"""
        self.set_pulse_width_us(channel, 0)
        print(f"Kênh {channel}: TẮT (LOW)")
    
    def release(self):
        """Giải phóng bus I2C"""
        self.bus.close()


# ============ PHẦN CHÍNH ============
if __name__ == "__main__":
    print("Khởi tạo PCA9685 trên bus I2C-3...")
    pca = PCA9685(bus=3)
    
    channel = 0  # Điều khiển chân 0
    pulse_time_ms = 100      # 3ms HIGH
    low_time_sec = 0.05       # 3 giây LOW (3000ms)
    
    print(f"\n=== BẮT ĐẦU TẠO XUNG ===")
    print(f"Kênh: {channel}")
    print(f"Thời gian HIGH: {pulse_time_ms}ms")
    print(f"Thời gian LOW: {low_time_sec}s")
    print("Nhấn Ctrl+C để dừng\n")
    
    try:
        while True:
            # Tạo xung HIGH trong 3ms
            pca.set_pwm_high(channel, pulse_time_ms)
            
            # Đợi trong thời gian LOW (3 giây)
            # Lưu ý: Trong thời gian này, chân vẫn giữ LOW từ lệnh trên
            print(f"Đợi {low_time_sec} giây (chân ở mức LOW)...")
            time.sleep(low_time_sec)
            
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n\nNgười dùng dừng chương trình")
    finally:
        # Tắt hết các kênh trước khi thoát
        print("Tắt tất cả các kênh...")
        pca.set_pwm_low(channel)
        pca.release()
        print("Đã giải phóng tài nguyên. Kết thúc.")






# import time
# import smbus2

# ADDR = 0x40
# bus = smbus2.SMBus(3)

# # Reset và cấu hình PCA9685 (tóm gọn)
# bus.write_byte_data(ADDR, 0x00, 0x00)
# prescale = int(25000000.0 / (4096 * 50) - 1)
# bus.write_byte_data(ADDR, 0xFE, prescale)
# time.sleep(0.01)
# bus.write_byte_data(ADDR, 0x00, 0x80)

# # Gửi xung 1500µs (góc 90°) cho kênh 0
# pulse_90 = int((1500 / 20000.0) * 4096)
# data = [0x00, 0x00, pulse_90 & 0xFF, (pulse_90 >> 8) & 0xFF]
# bus.write_i2c_block_data(ADDR, 0x06, data)

# print("Đã gửi lệnh về vị trí 90°. Servo sẽ quay đến giữa.")
# print("Nếu servo vẫn đứng yên, hãy kiểm tra lại nguồn và dây nối.")
# input("Nhấn Enter để kết thúc...")