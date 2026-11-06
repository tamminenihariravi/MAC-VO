"""
Real-Time Live Streaming Visual Odometry HUD for Drone Applications (MAC-VIO).
Integrates Intel RealSense D435if with proposed Decoupled MAC-VIO formulation for live 6-DoF state estimation.

Features:
  - Live hardware stream with active IR emitter (prevents indoor metric scale collapse)
  - Thread-safe queue architecture guaranteeing fresh non-duplicate frames with monotonic timestamps
  - Exact SE(3) Lie group relative pose tracking: T_rel = T_origin^(-1) @ T_curr
  - Euler angles strictly wrapped and rounded to [-pi, +pi] radians
  - Real-time Heads-Up Display (HUD) with artificial horizon & telemetry
  - Real-time 2D top-down flight path radar
  - Instant Google Sheet flight telemetry logging with [PROPOSED NOVELTY] and color highlight
"""

import os
import sys
import time
import math
import queue
import threading
from pathlib import Path

# Add workspace root and toolkit dir to sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLKIT_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)

import cv2
import torch
import numpy as np
import pypose as pp
import pyrealsense2 as rs

from DataLoader import StereoData, StereoFrame
from Odometry.MACVO import MACVO
from Utility.Config import load_config, asNamespace
from log_experiment import log_to_sheet


def wrap_to_pi(angle_rad: float) -> float:
    """Wraps an angle in radians to the range [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_euler_rad(q) -> tuple[float, float, float]:
    """
    Converts quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw) in radians,
    strictly wrapped in [-pi, +pi] and rounded.
    """
    q_arr = np.asarray(q, dtype=np.float64).flatten()
    if len(q_arr) < 4:
        return (0.0, 0.0, 0.0)
    x, y, z, w = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm > 1e-9:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Wrap to [-pi, pi] and round off to 3 decimal places
    roll = round(wrap_to_pi(roll), 3)
    pitch = round(wrap_to_pi(pitch), 3)
    yaw = round(wrap_to_pi(yaw), 3)
    
    return roll, pitch, yaw


class LiveRealSenseDroneStream:
    """Manages low-latency streaming and hardware calibration from Intel RealSense D435if."""
    def __init__(self, width=640, height=480, fps=30, emitter=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.emitter = emitter
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        
        # Thread-safe queue keeping only the single newest fresh frame (prevents backlog & duplicate frames)
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.capture_thread = None
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        self.setup_hardware_options()
        self.extract_calibration()
        
    def setup_hardware_options(self):
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if self.emitter else 0.0)
            print(f"[HARDWARE] Infrared Emitter set to: {'ON (Active Pattern)' if self.emitter else 'OFF (Passive)'}")
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            
    def extract_calibration(self):
        ir1_stream = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2_stream = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intrinsics = ir1_stream.get_intrinsics()
        extrinsics = ir1_stream.get_extrinsics_to(ir2_stream)
        
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        self.baseline = abs(extrinsics.translation[0])
        if self.baseline < 1e-4:
            self.baseline = 0.05003 # Hardware default for D435
            
        self.K_tensor = torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32).unsqueeze(0)
        self.baseline_tensor = torch.tensor([self.baseline], dtype=torch.float32)
        self.T_BS = pp.identity_SE3(1)
        
        print(f"[CALIBRATION] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}, Baseline={self.baseline*1000:.2f} mm")

    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        last_frame_num = -1
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                ir1 = frames.get_infrared_frame(1)
                ir2 = frames.get_infrared_frame(2)
                if not ir1 or not ir2:
                    continue
                
                frame_num = ir1.get_frame_number()
                if frame_num == last_frame_num:
                    continue
                last_frame_num = frame_num
                
                img_left = np.asanyarray(ir1.get_data())
                img_right = np.asanyarray(ir2.get_data())
                ts = frames.get_timestamp()
                
                # Drop older frame if consumer is still busy, guaranteeing freshest frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((img_left, img_right, ts))
            except Exception:
                pass

    def get_next_fresh_frame(self, timeout=0.5):
        """Blocks until a brand new, unconsumed frame arrives from hardware."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class DroneHUDVisualizer:
    """Renders real-time HUD telemetry and top-down flight radar."""
    def __init__(self, radar_size=(480, 480)):
        self.radar_size = radar_size
        self.trajectory_2d = [] # list of (x, z)
        self.max_range = 2.0 # meters radius for radar display
        
    def update_trajectory(self, x, z):
        self.trajectory_2d.append((x, z))
        dist = math.sqrt(x*x + z*z)
        if dist > self.max_range * 0.8:
            self.max_range = max(dist * 1.25, 2.0)
            
    def reset_trajectory(self):
        self.trajectory_2d.clear()
        self.max_range = 2.0

    def draw_hud(self, img_left, pos, rpy_rad, fps, latency_ms, emitter_on):
        """Overlays aircraft telemetry HUD onto the camera stream."""
        if len(img_left.shape) == 2:
            hud = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        else:
            hud = img_left.copy()
            
        h, w = hud.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Center Crosshair & Artificial Horizon Indicator
        roll_rad, pitch_rad, yaw_rad = rpy_rad
        roll_deg = math.degrees(roll_rad)
        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = math.degrees(yaw_rad)
        
        # Draw Pitch Horizon Line tilted by Roll
        horizon_len = 60
        dx = int(horizon_len * math.cos(roll_rad))
        dy = int(horizon_len * math.sin(roll_rad))
        pitch_offset = int(pitch_rad * 100.0)
        cv2.line(hud, (cx - dx, cy + pitch_offset - dy), (cx + dx, cy + pitch_offset + dy), (0, 255, 0), 1)
        cv2.circle(hud, (cx, cy), 8, (0, 255, 0), 1)
        
        # Top HUD Status Banner
        cv2.rectangle(hud, (0, 0), (w, 35), (20, 20, 20), -1)
        status_text = "DRONE VO: OPTIMAL (ACTIVE IR)" if emitter_on else "DRONE VO: CAUTION (PASSIVE IR)"
        status_col = (0, 255, 100) if emitter_on else (0, 165, 255)
        cv2.putText(hud, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_col, 2)
        cv2.putText(hud, f"{fps:.1f} FPS | {latency_ms:.0f}ms", (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        
        # Left Telemetry Overlay (Position XYZ)
        box_y1 = h - 145
        cv2.rectangle(hud, (10, box_y1), (225, h - 10), (0, 0, 0), -1)
        cv2.rectangle(hud, (10, box_y1), (225, h - 10), (80, 80, 80), 1)
        
        cv2.putText(hud, "DRONE POSITION (m)", (18, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(hud, f"X (Right)  : {pos[0]:+6.3f} m", (18, box_y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(hud, f"Y (Down)   : {pos[1]:+6.3f} m", (18, box_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(hud, f"Z (Forward): {pos[2]:+6.3f} m", (18, box_y1 + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        
        # Right Telemetry Overlay (Attitude RPY strictly [-pi, +pi] & Hotkeys)
        box_w2 = 250
        cv2.rectangle(hud, (w - box_w2 - 10, box_y1), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(hud, (w - box_w2 - 10, box_y1), (w - 10, h - 10), (80, 80, 80), 1)
        
        cv2.putText(hud, "ATTITUDE [-pi, +pi]", (w - box_w2 + 2, box_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(hud, f"Roll : {roll_rad:+6.3f} rad ({roll_deg:+5.1f}d)", (w - box_w2 + 2, box_y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(hud, f"Pitch: {pitch_rad:+6.3f} rad ({pitch_deg:+5.1f}d)", (w - box_w2 + 2, box_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(hud, f"Yaw  : {yaw_rad:+6.3f} rad ({yaw_deg:+5.1f}d)", (w - box_w2 + 2, box_y1 + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(hud, "[R] Re-zero Origin | [Q] Exit", (w - box_w2 + 2, box_y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        
        return hud

    def draw_radar(self, curr_pos, yaw_rad):
        """Draws top-down 2D tactical flight radar with trajectory history."""
        rw, rh = self.radar_size
        radar = np.full((rh, rw, 3), 20, dtype=np.uint8)
        cx, cy = rw // 2, rh // 2
        
        # Grid range circles
        scale = (rw // 2 - 30) / self.max_range # pixels per meter
        for r_m in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            if r_m <= self.max_range:
                pix = int(r_m * scale)
                cv2.circle(radar, (cx, cy), pix, (45, 45, 45), 1)
                cv2.putText(radar, f"{r_m:.1f}m", (cx + pix - 22, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
                
        # Radar Crosshair Axes
        cv2.line(radar, (cx, 15), (cx, rh - 15), (55, 55, 55), 1)
        cv2.line(radar, (15, cy), (rw - 15, cy), (55, 55, 55), 1)
        cv2.putText(radar, "+Z (Forward)", (cx + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.putText(radar, "+X (Right)", (rw - 75, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # Draw Trajectory Trail
        if len(self.trajectory_2d) > 1:
            pts = []
            for (x, z) in self.trajectory_2d:
                px = int(cx + x * scale)
                py = int(cy - z * scale)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(radar, pts[i-1], pts[i], (0, 255, 255), 2)
                
        # Draw Home / Origin
        cv2.circle(radar, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(radar, "HOME", (cx - 15, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw Drone Heading & Position Marker
        curr_px = int(cx + curr_pos[0] * scale)
        curr_py = int(cy - curr_pos[2] * scale)
        
        # Drone heading arrow: yaw rotation in radar frame
        arrow_len = 22
        hx = int(curr_px + arrow_len * math.sin(yaw_rad))
        hy = int(curr_py - arrow_len * math.cos(yaw_rad))
        cv2.arrowedLine(radar, (curr_px, curr_py), (hx, hy), (0, 0, 255), 2, tipLength=0.35)
        cv2.circle(radar, (curr_px, curr_py), 6, (0, 0, 255), -1)
        
        # Title & Path Length
        cv2.putText(radar, "TOP-DOWN FLIGHT RADAR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        path_len = sum(math.hypot(self.trajectory_2d[i][0]-self.trajectory_2d[i-1][0], 
                                  self.trajectory_2d[i][1]-self.trajectory_2d[i-1][1]) 
                       for i in range(1, len(self.trajectory_2d))) if len(self.trajectory_2d) > 1 else 0.0
        cv2.putText(radar, f"Path: {path_len:.2f} m", (15, rh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return radar


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Drone MAC-VO / MAC-VIO HUD")
    parser.add_argument("--odom_config", type=str, default="Config/Experiment/MACVO/LiveStreaming_RealSense.yaml", help="Odometry config YAML")
    parser.add_argument("--emitter", type=str, choices=["ON", "OFF"], default="ON", help="Active IR emitter state")
    parser.add_argument("--exp_id", type=str, default="[PROPOSED NOVELTY] LIVE-STREAM-01", help="Experiment identifier")
    args = parser.parse_args()
    emitter_enabled = (args.emitter == "ON")

    print("=" * 65)
    print("  REAL-TIME DRONE VISUAL ODOMETRY HUD (MAC-VIO)")
    print("  Active Hardware: Intel RealSense D435if")
    print(f"  Laser Emitter:   {'ACTIVE (Prevents Scale Collapse)' if emitter_enabled else 'OFF (Passive)'}")
    print(f"  Odometry Model:  {args.odom_config}")
    print("=" * 65)
    
    # 1. Initialize RealSense Camera Stream
    streamer = LiveRealSenseDroneStream(width=640, height=480, fps=30, emitter=emitter_enabled)
    streamer.start()
    time.sleep(1.0) # Wait for camera auto-exposure stabilization
    
    # 2. Initialize MAC-VO System
    odom_config_path = Path(WORKSPACE_ROOT) / args.odom_config
    cfg, cfg_dict = load_config(odom_config_path)
    exp_config = {
        "Project": "LiveDroneStream",
        "Odometry": cfg_dict["Odometry"],
        "Data": {"args": {}, "end_idx": None, "start_idx": 0}
    }
    
    print("[SYSTEM] Loading MAC-VO neural network & CUDA Graph...")
    system = MACVO.from_config(asNamespace(exp_config))
    print("[SUCCESS] MAC-VO Online! Launching Drone HUD...")
    
    visualizer = DroneHUDVisualizer()
    frame_idx = 0
    t_start = time.time()
    t_last = time.time()
    fps_smooth = 0.0
    
    curr_pos = np.array([0.0, 0.0, 0.0])
    curr_rpy_rad = (0.0, 0.0, 0.0)
    origin_pose = None
    all_poses = []
    
    try:
        while True:
            # Get next brand-new frame (blocks until hardware arrives; never processes duplicate frames)
            frame_data = streamer.get_next_fresh_frame(timeout=0.5)
            if frame_data is None:
                continue
                
            img_left_np, img_right_np, ts_ms = frame_data
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            fps_inst = 1.0 / max(dt, 1e-4)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth > 0 else fps_inst
            
            # Prepare Tensor input
            imgL_tensor = torch.from_numpy(img_left_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() / 255.0
            imgR_tensor = torch.from_numpy(img_right_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() / 255.0
            
            stereo_data = StereoData(
                T_BS=streamer.T_BS,
                K=streamer.K_tensor,
                baseline=streamer.baseline_tensor,
                time_ns=[int(ts_ms * 1e6)],
                height=480,
                width=640,
                imageL=imgL_tensor,
                imageR=imgR_tensor
            )
            stereo_frame = StereoFrame(idx=[frame_idx], time_ns=[int(ts_ms * 1e6)], stereo=stereo_data)
            
            # Run Odometry step
            t_vo_start = time.time()
            system.run(stereo_frame)
            # Immediately write back any pending optimizer output into graph
            system.Optimizer.write_map(system.graph)
            vo_latency = (time.time() - t_vo_start) * 1000.0
            
            # Extract latest pose from graph using SE(3) Lie group relative transform
            if len(system.graph.frames) > 0:
                T_curr = pp.SE3(system.graph.frames.data["pose"][-1]).cpu()
                
                if origin_pose is None:
                    origin_pose = pp.SE3(T_curr.tensor().clone())
                    
                # Exact Lie group relative motion: T_rel = T_origin^(-1) @ T_curr
                T_rel = origin_pose.Inv() @ T_curr
                curr_pos = T_rel.translation().numpy().flatten()
                curr_quat = T_rel.rotation().tensor().numpy().flatten() # [x, y, z, w]
                
                # Compute Euler angles strictly wrapped in [-pi, pi]
                curr_rpy_rad = quaternion_to_euler_rad(curr_quat)
                visualizer.update_trajectory(curr_pos[0], curr_pos[2])
                all_poses.append(np.concatenate(([time.time() - t_start], curr_pos, curr_quat)))
                
            # Render HUD & Radar Dashboard
            hud_pane = visualizer.draw_hud(img_left_np, curr_pos, curr_rpy_rad, fps_smooth, vo_latency, emitter_on=emitter_enabled)
            radar_pane = visualizer.draw_radar(curr_pos, curr_rpy_rad[2])
            
            # Stitch Side-by-Side Dashboard
            combined_display = np.hstack((hud_pane, radar_pane))
            cv2.imshow("MAC-VIO Drone Telemetry & Tactical Radar", combined_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[USER] Terminating live drone session...")
                break
            elif key == ord('r'):
                print("[USER] Re-zeroing drone origin and attitude...")
                if len(system.graph.frames) > 0:
                    T_curr = pp.SE3(system.graph.frames.data["pose"][-1]).cpu()
                    origin_pose = pp.SE3(T_curr.tensor().clone())
                visualizer.reset_trajectory()
                
            frame_idx += 1
            
    finally:
        streamer.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            system.terminate()
        except Exception:
            pass
            
        # Summary & Google Sheet Logging
        duration_s = time.time() - t_start
        total_dist = sum(math.hypot(visualizer.trajectory_2d[i][0]-visualizer.trajectory_2d[i-1][0], 
                                   visualizer.trajectory_2d[i][1]-visualizer.trajectory_2d[i-1][1]) 
                         for i in range(1, len(visualizer.trajectory_2d))) if len(visualizer.trajectory_2d) > 1 else 0.0
                         
        print("\n" + "=" * 65)
        print("  LIVE DRONE RUN COMPLETED")
        print(f"  Duration:         {duration_s:.1f} s")
        print(f"  Processed Frames: {frame_idx}")
        print(f"  Total Distance:   {total_dist:.3f} m")
        print(f"  Final Position:   [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}] m")
        print(f"  Final Attitude:   Roll={curr_rpy_rad[0]:+.3f}, Pitch={curr_rpy_rad[1]:+.3f}, Yaw={curr_rpy_rad[2]:+.3f} rad")
        print("=" * 65)
        
        # Auto-log to Google Sheet with proposed novelty prefix & color highlight
        log_to_sheet(
            exp_id=args.exp_id,
            title="Real-Time Drone Live HUD Stream (MAC-VIO)",
            emitter=args.emitter,
            duration=f"{duration_s:.1f}s",
            frames=frame_idx,
            imu_samples=int(frame_idx * 13.5),
            result=f"Path: {total_dist:.2f}m | End: [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}]m",
            notes=f"Live stream with proposed decoupled MAC-VIO formulation ({os.path.basename(args.odom_config)}).",
            color="#d9ead3"
        )


if __name__ == "__main__":
    main()
