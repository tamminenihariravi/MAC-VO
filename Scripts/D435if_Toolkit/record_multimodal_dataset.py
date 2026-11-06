"""
Multi-Modal Data Recorder for Intel RealSense D435if.
Records:
1. Left IR (infra1, 640x480, 30fps)
2. Right IR (infra2, 640x480, 30fps)
3. Color RGB (color, 640x480, 30fps)
4. United IMU (accel + gyro at 200Hz) - exactly matches ROS 2 unite_imu_method:=2

Outputs:
1. MAC-VO ready dataset directory (left/, right/, color/, imu.csv, camera_config.yaml)
2. Standard ROS 2 Bag (sqlite3 format) containing ROS 2 sensor_msgs
"""

import os
import sys
import time
import argparse
import shutil
import queue
import threading
import cv2
import numpy as np
import pyrealsense2 as rs

from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


def parse_args():
    parser = argparse.ArgumentParser(description="Record multi-modal data from Intel RealSense D435if")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save dataset and ROS 2 bag")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds (default: 60s)")
    parser.add_argument("--emitter", type=str, choices=["on", "off"], default="off", help="IR Emitter mode (on: active projector, off: passive)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    parser.add_argument("--no_bag", action="store_true", help="Skip creating ROS 2 bag")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    left_dir = os.path.join(args.output_dir, "left")
    right_dir = os.path.join(args.output_dir, "right")
    color_dir = os.path.join(args.output_dir, "color")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    
    imu_csv_path = os.path.join(args.output_dir, "imu.csv")
    bag_path = os.path.join(args.output_dir, "rosbag2_recording")
    if os.path.exists(bag_path):
        shutil.rmtree(bag_path)
    
    print("=" * 65)
    print(f"  RECORDING REALSENSE D435if DATASET")
    print(f"  Output Dir:    {args.output_dir}")
    print(f"  Duration:      {args.duration:.1f} seconds")
    print(f"  Emitter Mode:  {args.emitter.upper()}")
    print("=" * 65)
    
    # Initialize pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    
    cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    
    # First start pipeline to configure emitter and get intrinsics
    profile = pipe.start(cfg)
    
    # Configure emitter
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        target_emitter = 1.0 if args.emitter == "on" else 0.0
        depth_sensor.set_option(rs.option.emitter_enabled, target_emitter)
        print(f"-> Emitter set to: {depth_sensor.get_option(rs.option.emitter_enabled)} ({args.emitter.upper()})")
        
    # Get Intrinsics & Extrinsics
    ir1_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    ir1_intr = ir1_stream.get_intrinsics()
    ir2_intr = ir2_stream.get_intrinsics()
    ir1_to_ir2 = ir1_stream.get_extrinsics_to(ir2_stream)
    baseline = abs(ir1_to_ir2.translation[0])
    
    # Stop initial profile to restart with callback
    pipe.stop()
    
    clean_root_path = os.path.abspath(args.output_dir).replace('\\', '/')
    seq_name = os.path.basename(os.path.abspath(args.output_dir))
    config_yaml_content = f"""type: GeneralStereo
name: {seq_name}
args:
  root: {clean_root_path}
  camera:
    fx: {ir1_intr.fx:.4f}
    fy: {ir1_intr.fy:.4f}
    cx: {ir1_intr.ppx:.4f}
    cy: {ir1_intr.ppy:.4f}
  bl: {baseline:.6f}
  format: png
"""
    yaml_config_path = os.path.join(args.output_dir, "camera_config.yaml")
    with open(yaml_config_path, "w") as f:
        f.write(config_yaml_content)
    print(f"-> MAC-VO configuration generated: {yaml_config_path}")
    
    # Pre-construct CameraInfo static arrays
    k_arr = np.array([ir1_intr.fx, 0.0, ir1_intr.ppx, 0.0, ir1_intr.fy, ir1_intr.ppy, 0.0, 0.0, 1.0], dtype=np.float64)
    p_l_arr = np.array([ir1_intr.fx, 0.0, ir1_intr.ppx, 0.0, 0.0, ir1_intr.fy, ir1_intr.ppy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    p_r_arr = np.array([ir2_intr.fx, 0.0, ir2_intr.ppx, -ir2_intr.fx * baseline, 0.0, ir2_intr.fy, ir2_intr.ppy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    r_arr = np.eye(3, dtype=np.float64).flatten()
    d_arr = np.zeros(5, dtype=np.float64)
    zero_cov = np.zeros(9, dtype=np.float64)
    
    # State variables
    latest_gyro = [0.0, 0.0, 0.0]
    latest_accel = [0.0, 0.0, 0.0]
    has_gyro = False
    has_accel = False
    
    imu_file = open(imu_csv_path, "w")
    imu_file.write("timestamp_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n")
    
    # Setup ROS 2 Bag Writer inside worker thread
    frame_queue = queue.Queue(maxsize=500)
    stop_event = threading.Event()
    
    frame_idx = 0
    imu_sample_count = 0
    
    # Worker thread to write frames and bag messages
    def worker():
        nonlocal frame_idx, imu_sample_count
        writer = None
        typestore = None
        connections = {}
        if not args.no_bag:
            typestore = get_typestore(Stores.LATEST)
            writer = Writer(bag_path, version=9)
            writer.open()
            connections["ir1_img"] = writer.add_connection("/camera/camera/infra1/image_rect_raw", "sensor_msgs/msg/Image", typestore=typestore)
            connections["ir2_img"] = writer.add_connection("/camera/camera/infra2/image_rect_raw", "sensor_msgs/msg/Image", typestore=typestore)
            connections["rgb_img"] = writer.add_connection("/camera/camera/color/image_raw", "sensor_msgs/msg/Image", typestore=typestore)
            connections["ir1_info"] = writer.add_connection("/camera/camera/infra1/camera_info", "sensor_msgs/msg/CameraInfo", typestore=typestore)
            connections["ir2_info"] = writer.add_connection("/camera/camera/infra2/camera_info", "sensor_msgs/msg/CameraInfo", typestore=typestore)
            connections["imu"] = writer.add_connection("/camera/camera/imu", "sensor_msgs/msg/Imu", typestore=typestore)
            print(f"-> ROS 2 Bag Writer initialized in worker thread at: {bag_path}")
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
                
            kind, payload = item
            if kind == "imu":
                ts_ns, g, a = payload
                imu_file.write(f"{ts_ns},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f}\n")
                imu_sample_count += 1
                
                if writer and typestore:
                    sec = int(ts_ns // 1e9)
                    nanosec = int(ts_ns % 1e9)
                    imu_msg = typestore.types['sensor_msgs/msg/Imu'](
                        header=typestore.types['std_msgs/msg/Header'](
                            stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                            frame_id='camera_imu_optical_frame'
                        ),
                        orientation=typestore.types['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0),
                        orientation_covariance=zero_cov,
                        angular_velocity=typestore.types['geometry_msgs/msg/Vector3'](x=g[0], y=g[1], z=g[2]),
                        angular_velocity_covariance=zero_cov,
                        linear_acceleration=typestore.types['geometry_msgs/msg/Vector3'](x=a[0], y=a[1], z=a[2]),
                        linear_acceleration_covariance=zero_cov
                    )
                    writer.write(connections["imu"], ts_ns, typestore.serialize_cdr(imu_msg, 'sensor_msgs/msg/Imu'))
                    
            elif kind == "video":
                ts_ns, ir1_img, ir2_img, col_img = payload
                cv2.imwrite(os.path.join(left_dir, f"{frame_idx:06d}.png"), ir1_img)
                cv2.imwrite(os.path.join(right_dir, f"{frame_idx:06d}.png"), ir2_img)
                if col_img is not None:
                    cv2.imwrite(os.path.join(color_dir, f"{frame_idx:06d}.png"), col_img)
                    
                if writer and typestore:
                    sec = int(ts_ns // 1e9)
                    nanosec = int(ts_ns % 1e9)
                    
                    msg_l = typestore.types['sensor_msgs/msg/Image'](
                        header=typestore.types['std_msgs/msg/Header'](
                            stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                            frame_id='camera_infra1_optical_frame'
                        ),
                        height=args.height, width=args.width, encoding='mono8',
                        is_bigendian=0, step=args.width,
                        data=ir1_img.flatten().view(np.uint8)
                    )
                    writer.write(connections["ir1_img"], ts_ns, typestore.serialize_cdr(msg_l, 'sensor_msgs/msg/Image'))
                    
                    msg_r = typestore.types['sensor_msgs/msg/Image'](
                        header=typestore.types['std_msgs/msg/Header'](
                            stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                            frame_id='camera_infra2_optical_frame'
                        ),
                        height=args.height, width=args.width, encoding='mono8',
                        is_bigendian=0, step=args.width,
                        data=ir2_img.flatten().view(np.uint8)
                    )
                    writer.write(connections["ir2_img"], ts_ns, typestore.serialize_cdr(msg_r, 'sensor_msgs/msg/Image'))
                    
                    if col_img is not None:
                        msg_c = typestore.types['sensor_msgs/msg/Image'](
                            header=typestore.types['std_msgs/msg/Header'](
                                stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                                frame_id='camera_color_optical_frame'
                            ),
                            height=args.height, width=args.width, encoding='bgr8',
                            is_bigendian=0, step=args.width * 3,
                            data=col_img.flatten().view(np.uint8)
                        )
                        writer.write(connections["rgb_img"], ts_ns, typestore.serialize_cdr(msg_c, 'sensor_msgs/msg/Image'))
                        
                    info_l = typestore.types['sensor_msgs/msg/CameraInfo'](
                        header=typestore.types['std_msgs/msg/Header'](
                            stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                            frame_id='camera_infra1_optical_frame'
                        ),
                        height=args.height, width=args.width, distortion_model='plumb_bob',
                        d=d_arr, k=k_arr, r=r_arr, p=p_l_arr,
                        binning_x=0, binning_y=0,
                        roi=typestore.types['sensor_msgs/msg/RegionOfInterest'](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
                    )
                    writer.write(connections["ir1_info"], ts_ns, typestore.serialize_cdr(info_l, 'sensor_msgs/msg/CameraInfo'))
                    
                    info_r = typestore.types['sensor_msgs/msg/CameraInfo'](
                        header=typestore.types['std_msgs/msg/Header'](
                            stamp=typestore.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=nanosec),
                            frame_id='camera_infra2_optical_frame'
                        ),
                        height=args.height, width=args.width, distortion_model='plumb_bob',
                        d=d_arr, k=k_arr, r=r_arr, p=p_r_arr,
                        binning_x=0, binning_y=0,
                        roi=typestore.types['sensor_msgs/msg/RegionOfInterest'](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
                    )
                    writer.write(connections["ir2_info"], ts_ns, typestore.serialize_cdr(info_r, 'sensor_msgs/msg/CameraInfo'))
                    
                frame_idx += 1
                
            frame_queue.task_done()
            
        if writer:
            writer.close()
            
    writer_thread = threading.Thread(target=worker, daemon=True)
    writer_thread.start()
    
    # Callback for incoming frames
    def frame_callback(frame):
        nonlocal latest_gyro, latest_accel, has_gyro, has_accel
        if frame.is_motion_frame():
            mf = frame.as_motion_frame()
            name = mf.get_profile().stream_name()
            d = mf.get_motion_data()
            ts_ns = int(mf.get_timestamp() * 1e6)
            
            if name == "Gyro":
                latest_gyro = [d.x, d.y, d.z]
                has_gyro = True
            elif name == "Accel":
                latest_accel = [d.x, d.y, d.z]
                has_accel = True
                
            if has_gyro and has_accel:
                try:
                    frame_queue.put_nowait(("imu", (ts_ns, latest_gyro[:], latest_accel[:])))
                except queue.Full:
                    pass
                    
        elif frame.is_frameset():
            fs = frame.as_frameset()
            ir1_f = fs.get_infrared_frame(1)
            ir2_f = fs.get_infrared_frame(2)
            col_f = fs.get_color_frame()
            
            if ir1_f and ir2_f:
                ts_ns = int(ir1_f.get_timestamp() * 1e6)
                ir1_img = np.asanyarray(ir1_f.get_data()).copy()
                ir2_img = np.asanyarray(ir2_f.get_data()).copy()
                col_img = np.asanyarray(col_f.get_data()).copy() if col_f else None
                try:
                    frame_queue.put_nowait(("video", (ts_ns, ir1_img, ir2_img, col_img)))
                except queue.Full:
                    pass
                    
    # Restart pipeline with callback
    profile = pipe.start(cfg, frame_callback)
    
    print("\nStarting recording... Move camera or walk the sequence.")
    print("Press Ctrl+C to stop early.\n")
    
    start_time = time.time()
    last_print = start_time
    
    try:
        while time.time() - start_time < args.duration:
            time.sleep(0.5)
            now = time.time()
            elapsed = now - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            imu_rate = imu_sample_count / elapsed if elapsed > 0 else 0
            print(f"\rProgress: [{elapsed:4.1f}/{args.duration:4.1f}s] | Stereo Frames: {frame_idx:4d} ({fps:4.1f} fps) | IMU Samples: {imu_sample_count:5d} ({imu_rate:5.1f} Hz)", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
        
    finally:
        pipe.stop()
        stop_event.set()
        writer_thread.join()
        imu_file.close()
            
    total_time = time.time() - start_time
    print(f"\n\nRecording Finished!")
    print(f"  Total Time:       {total_time:.2f} s")
    print(f"  Stereo Frames:    {frame_idx} (~{frame_idx/total_time:.1f} FPS)")
    print(f"  IMU Samples:      {imu_sample_count} (~{imu_sample_count/total_time:.1f} Hz)")
    print(f"  Saved MAC-VO dir: {os.path.abspath(args.output_dir)}")
    if not args.no_bag:
        print(f"  Saved ROS 2 Bag:  {os.path.abspath(bag_path)}")


if __name__ == "__main__":
    main()
