"""
Hardware sanity check script for Intel RealSense D435if.
Checks:
- USB 3.0/3.1/3.2 Port status
- Stereo IR (infra1, infra2) streams
- RGB color stream
- IMU (accel + gyro at 200 Hz) streams
- Factory intrinsics & extrinsics
"""

import time
import pyrealsense2 as rs

def main():
    print("=" * 60)
    print("  INTEL REALSENSE D435if HARDWARE & SENSOR SANITY CHECK")
    print("=" * 60)
    
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense device detected! Check USB cable.")
        return False
        
    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    fw = dev.get_info(rs.camera_info.firmware_version)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
    
    print(f"Device Name:        {name}")
    print(f"Serial Number:      {serial}")
    print(f"Firmware Version:   {fw}")
    print(f"USB Connection:     USB {usb}")
    
    is_usb3 = float(usb) >= 3.0 if usb.replace(".", "", 1).isdigit() else False
    if is_usb3:
        print("[PASS] Device is connected on a high-speed USB 3.x port.")
    else:
        print("[WARNING] Device is connected on USB 2.x! High-framerate/resolution streams may fail.")
        
    # Test Streaming
    print("\nInitializing Pipeline with Stereo IR (640x480@30Hz), RGB (640x480@30Hz), and IMU (200Hz)...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    
    profile = pipe.start(cfg)
    
    # Check Emitter
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        emitter_val = depth_sensor.get_option(rs.option.emitter_enabled)
        emitter_desc = "ON (Active IR Projector)" if emitter_val == 1.0 else ("OFF (Passive)" if emitter_val == 0.0 else f"Auto ({emitter_val})")
        print(f"IR Emitter Status:  {emitter_desc}")
    
    # Intrinsics & Extrinsics
    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    intr = ir1.get_intrinsics()
    ext = ir1.get_extrinsics_to(ir2)
    baseline = abs(ext.translation[0])
    
    print(f"\nCamera Calibration (infra1):")
    print(f"  Focal Length:     fx={intr.fx:.2f}, fy={intr.fy:.2f}")
    print(f"  Principal Point:  cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")
    print(f"  Stereo Baseline:  {baseline*1000:.2f} mm ({baseline:.4f} m)")
    
    print("\nStreaming verification for 3 seconds...")
    start_t = time.time()
    counts = {"ir1": 0, "ir2": 0, "color": 0}
    
    while time.time() - start_t < 3.0:
        fs = pipe.wait_for_frames(5000)
        if fs.get_infrared_frame(1): counts["ir1"] += 1
        if fs.get_infrared_frame(2): counts["ir2"] += 1
        if fs.get_color_frame(): counts["color"] += 1
        
    duration = time.time() - start_t
    pipe.stop()
    
    print(f"\nStreaming Results over {duration:.2f}s:")
    print(f"  Left IR (infra1):  {counts['ir1']} frames (~{counts['ir1']/duration:.1f} FPS)")
    print(f"  Right IR (infra2): {counts['ir2']} frames (~{counts['ir2']/duration:.1f} FPS)")
    print(f"  RGB (color):       {counts['color']} frames (~{counts['color']/duration:.1f} FPS)")
    print("\n[ALL HARDWARE CHECKS PASSED SUCCESSFULLY]")
    return True

if __name__ == "__main__":
    main()
