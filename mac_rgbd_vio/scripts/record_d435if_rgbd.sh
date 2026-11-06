#!/bin/bash
# Script to launch and record synchronized RGB-D + IMU data from Intel RealSense D435if
# Tailored for MAC-RGBD-VIO (Target: IEEE RA-L)
#
# Usage: ./record_d435if_rgbd.sh <sequence_name>
# Example: ./record_d435if_rgbd.sh exp101_office_loop

SEQ_NAME=${1:-"rgbd_test_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR="data/${SEQ_NAME}"

mkdir -p data

echo "=========================================================="
echo " Starting RealSense D435if RGB-D + IMU Recording"
echo " Sequence Name: ${SEQ_NAME}"
echo " Saving to: ${OUTPUT_DIR}"
echo "=========================================================="
echo " Key settings:"
echo " - RGB Stream: 640x480 @ 30 FPS"
echo " - Aligned Depth: Hardware active IR depth aligned to RGB frame"
echo " - Active Emitter: ON (Enhances textureless wall depth)"
echo " - IMU: 200 Hz linear interpolation fused"
echo " - Hardware Frame Sync: ENABLED"
echo "=========================================================="

# Check if ros2 command is available; if not, use native pyrealsense2 + rosbags recorder
if command -v ros2 &> /dev/null; then
    echo "Using ROS 2 bag record..."
    ros2 bag record -o ${OUTPUT_DIR} \
      /camera/camera/color/image_raw \
      /camera/camera/color/camera_info \
      /camera/camera/aligned_depth_to_color/image_raw \
      /camera/camera/imu
else
    echo "ROS 2 command line not detected on host. Using high-speed native Python/RealSense recorder..."
    python3 "$(dirname "$0")/record_d435if_rgbd.py" "${SEQ_NAME}" --duration 60 --emitter 1
fi
