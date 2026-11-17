# UAV Autonomous Payload Drop

Autonomous quadcopter mission built for the **Intelligent UAV Contest 2025** (Consolation Prize). The drone takes off, finds five colored circular targets in a fixed order, centers itself over each one using onboard vision, drops a payload, and lands.

Mission order: **red 1 → yellow (left) → red 2 → yellow (right) → blue → land**

## System overview

| Part | Detail |
|---|---|
| Companion computer | Orange Pi running ROS 2 Humble |
| Flight controller | Pixhawk with ArduPilot, connected over USB serial (MAVLink, pymavlink) |
| Positioning | Optical flow sensor and rangefinder for position and height hold |
| Camera | USB camera, 640 x 480 at 20 fps |
| Payload release | PCA9685 servo driver over I2C, one channel per payload |

## ROS 2 nodes

All nodes live in the `uav_control` package (`src/uav_control`).

| Node | Role |
|---|---|
| `camera_node` | Reads the camera with OpenCV and publishes frames on `/cam/front/image_raw` |
| `vision_node` | Finds the current target color with HSV masks, morphology and `minEnclosingCircle`, locks the target and tracks it in a region of interest, then publishes the pixel error and an `aligned` flag |
| `auto_control_node` | Talks to ArduPilot over MAVLink, runs the mission state machine and sends body-frame velocity commands in GUIDED mode |
| `servo` | Drives the PCA9685 channels that release each payload |
| `testservo` | Bench test for the servo driver |

How a target is reached:

1. `vision_node` reports the offset between the image center and the target circle.
2. `auto_control_node` moves toward the target with a proportional controller.
3. Within 30 px of the center it switches to a slower PID settle loop, capped at 0.08 m/s.
4. When the camera center stays inside the circle for 30 consecutive control loops, the step is complete and the mission moves to the next color.

Safety checks before the mission starts: the vehicle must be armed, the optical flow quality must be good, and the camera must be streaming. Switching out of GUIDED mode hands control back to the pilot.

## Run

```bash
# on the Orange Pi
cd ~/ROS2_WS_UAV
colcon build --packages-select uav_control
source install/setup.bash
ros2 launch uav_control auto_control.launch.py
```

`scripts/start_uav_stack.sh` does the same and can be used to start the stack at boot.

## Demo

[Flight video](VIDEO_LINK)

## Author

Le Duc Duy, Control and Automation Engineering student at HCMUTE.
