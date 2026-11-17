#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /home/orangepi/ROS2_WS_UAV/install/setup.bash
exec ros2 launch uav_control auto_control.launch.py
