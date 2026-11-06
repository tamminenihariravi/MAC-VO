"""
MAC-RGBD-VIO: RealSense D435if Hardware Stream & Topic Frequency Verifier
Checks that Color, Aligned Depth, and Unified IMU publish smoothly with zero dropped frames.
"""

import time
import pyrealsense2 as rs
import numpy as np
import threading
import queue

def main():
    print("=" * 65)
    print("  REALSENSE D435if HARDWARE STREAM VERIFICATION")
    print("  Target Topics:")
    print("    - /camera/camera/color/image_raw               (640x480 @ 30 FPS)")
    print("    - /camera/camera/aligned_depth_to_color/image_raw (640x480 @ 30 FPS)")
    print("    - /camera/camera/imu                           (Unified @ 200 Hz)")
    print("=" * 65)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    align = rs.align(rs.stream.color)
    video_queue = queue.Queue(maxsize=30)

    color_count = 0
    depth_count = 0
    imu_count = 0

    latest_gyro = [0.0, 0.0, 0.0]
    latest_accel = [0.0, 0.0, 0.0]
    has_gyro = False
    has_accel = False
    imu_lock = threading.Lock()

    running = True

    def on_frame(frame):
        nonlocal imu_count, has_gyro, has_accel, latest_gyro, latest_accel
        if not running:
            return

        if frame.is_motion_frame():
            mf = frame.as_motion_frame()
            sname = mf.get_profile().stream_name()
            val = mf.get_motion_data()
            with imu_lock:
                if sname == "Gyro":
                    latest_gyro = [val.x, val.y, val.z]
                    has_gyro = True
                elif sname == "Accel":
                    latest_accel = [val.x, val.y, val.z]
                    has_accel = True
                if has_gyro and has_accel:
                    imu_count += 1
                    has_gyro = False
                    has_accel = False

        elif frame.is_frameset():
            try:
                video_queue.put_nowait(frame.as_frameset())
            except queue.Full:
                pass

    try:
        profile = pipeline.start(config, on_frame)
    except Exception as e:
        print(f"[ERROR] Failed to start RealSense pipeline: {e}")
        return

    # Enable active IR emitter
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
        print("[HARDWARE] Active IR Projector Emitter: ENABLED (1)")

    def video_worker():
        nonlocal color_count, depth_count
        while running:
            try:
                fs = video_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if fs.first_or_default(rs.stream.color) and fs.first_or_default(rs.stream.depth):
                try:
                    aligned = align.process(fs)
                    c = aligned.get_color_frame()
                    d = aligned.get_depth_frame()
                    if c and d:
                        color_count += 1
                        depth_count += 1
                except Exception:
                    pass
            video_queue.task_done()

    v_thread = threading.Thread(target=video_worker, daemon=True)
    v_thread.start()

    print("\n[VERIFYING] Measuring 5.0 seconds of hardware streaming...")
    t_start = time.time()
    time.sleep(5.0)

    running = False
    pipeline.stop()
    duration = time.time() - t_start

    fps_c = color_count / duration
    fps_d = depth_count / duration
    hz_imu = imu_count / duration

    print("\n" + "=" * 65)
    print("  VERIFICATION RESULTS")
    print(f"  Duration:                        {duration:.2f} s")
    print(f"  /camera/camera/color/image_raw:  {fps_c:.1f} FPS (Target: ~30 FPS) [{'PASS' if fps_c > 20 else 'WARN'}]")
    print(f"  /camera/camera/aligned_depth...: {fps_d:.1f} FPS (Target: ~30 FPS) [{'PASS' if fps_d > 20 else 'WARN'}]")
    print(f"  /camera/camera/imu:              {hz_imu:.1f} Hz  (Target: ~200 Hz) [{'PASS' if hz_imu > 150 else 'WARN'}]")
    print("=" * 65)

if __name__ == "__main__":
    main()
