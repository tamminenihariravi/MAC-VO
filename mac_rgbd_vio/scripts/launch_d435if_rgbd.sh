#!/bin/bash
# Launch ROS 2 realsense2_camera with RGB, Aligned Depth, Active Emitter, and Unified IMU

echo "Launching RealSense D435if with hardware aligned depth and IMU..."

ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true \
    enable_depth:=true \
    align_depth.enable:=true \
    enable_infra1:=false \
    enable_infra2:=false \
    enable_sync:=true \
    unite_imu_method:=2 \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    depth_module.emitter_enabled:=1
