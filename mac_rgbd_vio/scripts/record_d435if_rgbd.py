"""
RealSense D435if Synchronized RGB-D + Unified IMU Recorder for MAC-RGBD-VIO.
Records hardware-aligned depth to color, RGB, and 200Hz unified IMU.

Outputs:
1. Native RealSense ROS Bag (.bag file)
2. Standard ROS 2 Bag (sqlite3 format) with topics:
   - /camera/camera/color/image_raw
   - /camera/camera/color/camera_info
   - /camera/camera/aligned_depth_to_color/image_raw
   - /camera/camera/imu
3. Extracted raw dataset: color/, aligned_depth/, imu.csv, camera_info.yaml
"""

import os
import sys
import time
import yaml
import shutil
import queue
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


def parse_args():
    parser = argparse.ArgumentParser(description="Record synchronized RGB-D + IMU from RealSense D435if")
    parser.add_argument("seq_name", nargs="?", default="exp102_laptop_office_walk", help="Sequence name / directory")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60s)")
    parser.add_argument("--emitter", type=int, default=1, choices=[0, 1], help="Active IR Emitter (1: ON, 0: OFF)")
    parser.add_argument("--preview", action="store_true", default=True, help="Show live preview HUD")
    parser.add_argument("--no_preview", dest="preview", action="store_false", help="Disable live preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = repo_root / "data" / args.seq_name

    os.makedirs(output_dir, exist_ok=True)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "aligned_depth"
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rosbag2_dir = output_dir / "rosbag2_recording"
    if rosbag2_dir.exists():
        shutil.rmtree(rosbag2_dir)

    realsense_bag_path = str(output_dir / f"{args.seq_name}.db3")
    imu_csv_path = str(output_dir / "imu.csv")

    print("=" * 65)
    print("  REALSENSE D435if SYNCHRONIZED RGB-D + IMU RECORDER")
    print(f"  Target Project:  MAC-RGBD-VIO (IEEE RA-L)")
    print(f"  Sequence Name:   {args.seq_name}")
    print(f"  Saving to:       {output_dir}")
    print(f"  Duration:        {args.duration:.1f} s")
    print(f"  Active Emitter:  {'ON (Speckle Projected)' if args.emitter == 1 else 'OFF'}")
    print("=" * 65)

    # 1. Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Save native RealSense rosbag
    config.enable_record_to_file(realsense_bag_path)

    # Enable Streams: Color 640x480 @ 30fps, Depth 640x480 @ 30fps, Gyro 200Hz, Accel 200Hz
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)

    # State variables for 200Hz Unified IMU
    latest_gyro = [0.0, 0.0, 0.0]
    latest_accel = [0.0, 0.0, 0.0]
    has_gyro = False
    has_accel = False
    imu_lock = threading.Lock()

    color_frame_count = 0
    depth_frame_count = 0
    imu_sample_count = 0

    fps_color = 0.0
    fps_depth = 0.0
    hz_imu = 0.0

    colorizer = rs.colorizer()

    # Asynchronous queues
    disk_queue = queue.Queue(maxsize=200)
    video_queue = queue.Queue(maxsize=30)
    running = True

    t_prev_imu = time.time()
    t_prev_frame = time.time()

    def on_frame(frame):
        nonlocal imu_sample_count, has_gyro, has_accel, latest_gyro, latest_accel, hz_imu, t_prev_imu
        if not running:
            return

        if frame.is_motion_frame():
            mf = frame.as_motion_frame()
            stream_name = mf.get_profile().stream_name()
            val = mf.get_motion_data()
            ts_ns = int(mf.get_timestamp() * 1e6)

            with imu_lock:
                if stream_name == "Gyro":
                    latest_gyro = [val.x, val.y, val.z]
                    has_gyro = True
                elif stream_name == "Accel":
                    latest_accel = [val.x, val.y, val.z]
                    has_accel = True

                if has_gyro and has_accel:
                    imu_file.write(f"{ts_ns},{latest_accel[0]:.6f},{latest_accel[1]:.6f},{latest_accel[2]:.6f},"
                                   f"{latest_gyro[0]:.6f},{latest_gyro[1]:.6f},{latest_gyro[2]:.6f}\n")
                    disk_queue.put(("imu", (ts_ns, list(latest_accel), list(latest_gyro))))
                    imu_sample_count += 1
                    t_now = time.time()
                    hz_imu = 0.95 * hz_imu + 0.05 * (1.0 / max(t_now - t_prev_imu, 1e-4))
                    t_prev_imu = t_now
                    has_gyro = False
                    has_accel = False

        elif frame.is_frameset():
            try:
                video_queue.put_nowait(frame.as_frameset())
            except queue.Full:
                pass

    # Prepare IMU CSV before starting pipeline callback
    imu_file = open(imu_csv_path, "w")
    imu_file.write("timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")

    # Start Pipeline with Callback
    profile = pipeline.start(config, on_frame)

    # Configure Hardware Emitter & Auto-Exposure
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
        print(f"[HARDWARE] Active IR Emitter set to: {args.emitter}")

    # Create Hardware Alignment Engine (Align Depth -> Color)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Extract Color Intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    camera_info_dict = {
        "camera_name": "realsense_d435if_color",
        "image_width": 640,
        "image_height": 480,
        "distortion_model": "plumb_bob",
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "distortion_coefficients": list(intrinsics.coeffs),
        "K": [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ],
        "depth_scale": depth_sensor.get_depth_scale() # typically 0.001 (1mm/unit)
    }

    with open(output_dir / "camera_info.yaml", "w") as f:
        yaml.dump(camera_info_dict, f, default_flow_style=False)

    print(f"[CALIBRATION] Aligned Camera: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")



    def disk_worker():
        typestore = get_typestore(Stores.LATEST)
        writer = Writer(rosbag2_dir, version=9)
        writer.open()

        conn_color = writer.add_connection('/camera/camera/color/image_raw', 'sensor_msgs/msg/Image', typestore=typestore)
        conn_info  = writer.add_connection('/camera/camera/color/camera_info', 'sensor_msgs/msg/CameraInfo', typestore=typestore)
        conn_depth = writer.add_connection('/camera/camera/aligned_depth_to_color/image_raw', 'sensor_msgs/msg/Image', typestore=typestore)
        conn_imu   = writer.add_connection('/camera/camera/imu', 'sensor_msgs/msg/Imu', typestore=typestore)

        while running or not disk_queue.empty():
            try:
                item = disk_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            item_type, data = item
            if item_type == "frame":
                ts_ns, color_np, depth_np, idx = data
                # 1. Save PNG files
                cv2.imwrite(str(color_dir / f"{idx:06d}.png"), color_np)
                cv2.imwrite(str(depth_dir / f"{idx:06d}.png"), depth_np)

                # 2. Write to ROS 2 Bag
                sec = int(ts_ns // 1e9)
                nanosec = int(ts_ns % 1e9)

                # Color msg
                msg_c = typestore.types['sensor_msgs/msg/Image'](
                    header=typestore.types['std_msgs/msg/Header'](
                        stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                        frame_id='camera_color_optical_frame'
                    ),
                    height=480, width=640, encoding='bgr8',
                    is_bigendian=0, step=640 * 3,
                    data=color_np.flatten().view(np.uint8)
                )
                writer.write(conn_color, ts_ns, typestore.serialize_cdr(msg_c, 'sensor_msgs/msg/Image'))

                # CameraInfo msg
                msg_info = typestore.types['sensor_msgs/msg/CameraInfo'](
                    header=typestore.types['std_msgs/msg/Header'](
                        stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                        frame_id='camera_color_optical_frame'
                    ),
                    height=480, width=640, distortion_model='plumb_bob',
                    d=np.zeros(5, dtype=np.float64),
                    k=np.array([intrinsics.fx, 0.0, intrinsics.ppx, 0.0, intrinsics.fy, intrinsics.ppy, 0.0, 0.0, 1.0], dtype=np.float64),
                    r=np.eye(3, dtype=np.float64).flatten(),
                    p=np.array([
                        intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
                        0.0, intrinsics.fy, intrinsics.ppy, 0.0,
                        0.0, 0.0, 1.0, 0.0
                    ], dtype=np.float64),
                    binning_x=0, binning_y=0,
                    roi=typestore.types['sensor_msgs/msg/RegionOfInterest'](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
                )
                writer.write(conn_info, ts_ns, typestore.serialize_cdr(msg_info, 'sensor_msgs/msg/CameraInfo'))

                # Aligned Depth msg (16UC1: uint16 in mm)
                msg_d = typestore.types['sensor_msgs/msg/Image'](
                    header=typestore.types['std_msgs/msg/Header'](
                        stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                        frame_id='camera_color_optical_frame'
                    ),
                    height=480, width=640, encoding='16UC1',
                    is_bigendian=0, step=640 * 2,
                    data=depth_np.flatten().view(np.uint8)
                )
                writer.write(conn_depth, ts_ns, typestore.serialize_cdr(msg_d, 'sensor_msgs/msg/Image'))

            elif item_type == "imu":
                ts_ns, a, g = data
                sec = int(ts_ns // 1e9)
                nanosec = int(ts_ns % 1e9)

                msg_imu = typestore.types['sensor_msgs/msg/Imu'](
                    header=typestore.types['std_msgs/msg/Header'](
                        stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                        frame_id='camera_imu_optical_frame'
                    ),
                    orientation=typestore.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9, dtype=np.float64),
                    angular_velocity=typestore.types['geometry_msgs/msg/Vector3'](x=g[0], y=g[1], z=g[2]),
                    angular_velocity_covariance=np.eye(3, dtype=np.float64).flatten() * 0.01,
                    linear_acceleration=typestore.types['geometry_msgs/msg/Vector3'](x=a[0], y=a[1], z=a[2]),
                    linear_acceleration_covariance=np.eye(3, dtype=np.float64).flatten() * 0.1
                )
                writer.write(conn_imu, ts_ns, typestore.serialize_cdr(msg_imu, 'sensor_msgs/msg/Imu'))

            disk_queue.task_done()
        writer.close()

    disk_thread = threading.Thread(target=disk_worker, daemon=True)
    disk_thread.start()

    print("\n[READY] Recording started. Follow the trajectory instructions...")
    print("  -> Press [Q] in the HUD window to finish early.\n")

    start_time = time.time()
    last_print = time.time()
    t_prev_frame = time.time()
    t_prev_imu = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n[COMPLETE] Target duration of {args.duration:.1f}s reached.")
                break

            try:
                frames = video_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Align Depth to Color (only when both color and depth frames are present)
            if not (frames.first_or_default(rs.stream.color) and frames.first_or_default(rs.stream.depth)):
                video_queue.task_done()
                continue

            try:
                aligned_frames = align.process(frames)
            except Exception:
                video_queue.task_done()
                continue

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                video_queue.task_done()
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data()) # uint16, depth in mm
            ts_ns = int(color_frame.get_timestamp() * 1e6)

            disk_queue.put(("frame", (ts_ns, color_np, depth_np, color_frame_count)))
            color_frame_count += 1
            depth_frame_count += 1
            video_queue.task_done()

            t_now = time.time()
            fps_color = 0.9 * fps_color + 0.1 * (1.0 / max(t_now - t_prev_frame, 1e-4))
            t_prev_frame = t_now

            # Console health telemetry
            if time.time() - last_print > 1.0:
                print(f"[REC] Time: {elapsed:4.1f}/{args.duration:.0f}s | "
                      f"RGB: {fps_color:4.1f} FPS | "
                      f"Aligned Depth: {fps_color:4.1f} FPS | "
                      f"IMU: {hz_imu:5.1f} Hz | "
                      f"Frames: {color_frame_count:4d} | "
                      f"IMU Samples: {imu_sample_count:5d}")
                last_print = time.time()

            # Live Verification HUD
            if args.preview:
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                # Add text overlay
                cv2.putText(color_np, f"RGB: {fps_color:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_np, f"IMU: {hz_imu:.1f} Hz", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(colorized_depth, "Hardware Aligned Depth", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(colorized_depth, f"Time: {elapsed:.1f}s / {args.duration:.0f}s", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                hud_combined = np.hstack((color_np, colorized_depth))
                cv2.imshow("MAC-RGBD-VIO Hardware Stream Verification", hud_combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[USER] Stop requested early.")
                    break

    finally:
        print("\nStopping RealSense pipeline and finalizing files...")
        pipeline.stop()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        imu_file.close()

        running = False
        print("Flushing remaining frames to disk and ROS 2 bag...")
        disk_queue.join()
        disk_thread.join()

        total_time = time.time() - start_time
        print("=" * 65)
        print("  DATASET RECORDING SUCCESSFULLY COMPLETED")
        print(f"  Duration:              {total_time:.1f} s")
        print(f"  Color Frames:          {color_frame_count} (~{color_frame_count/max(total_time,1e-3):.1f} FPS)")
        print(f"  Aligned Depth Frames:  {depth_frame_count} (~{depth_frame_count/max(total_time,1e-3):.1f} FPS)")
        print(f"  IMU Samples:           {imu_sample_count} (~{imu_sample_count/max(total_time,1e-3):.1f} Hz)")
        print(f"  ROS 2 Bag:             {rosbag2_dir}")
        print(f"  RealSense Bag:         {realsense_bag_path}")
        print("=" * 65)


if __name__ == "__main__":
    main()
