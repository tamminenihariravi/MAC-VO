"""
Automated MAC-VO Experiment Runner & Trajectory Evaluator.
Executes MAC-VO odometry on recorded RealSense sequences, generates 2D/3D trajectory plots,
and logs metrics to the Google Sheet.
"""

import os
import sys

# Ensure workspace root and toolkit are on path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLKIT_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)

import glob
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from log_experiment import log_to_sheet


def parse_args():
    parser = argparse.ArgumentParser(description="Run MAC-VO on a sequence and analyze results")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to sequence directory containing camera_config.yaml")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment identifier (e.g. EXP-001, EXP-002)")
    parser.add_argument("--emitter", type=str, choices=["ON", "OFF"], default="OFF", help="Emitter mode used during recording")
    parser.add_argument("--mode", type=str, default="RealTime_Laptop", help="MAC-VO config mode or preset")
    parser.add_argument("--odom_config", type=str, default=None, help="Explicit path to odometry config YAML")
    parser.add_argument("--color", type=str, default=None, help="Hex color code for Google Sheet highlight")
    parser.add_argument("--log_sheet", action="store_true", help="Automatically push metrics to Google Sheet webhook")
    return parser.parse_args()


def main():
    args = parse_args()
    config_file = os.path.join(args.data_dir, "camera_config.yaml")
    if not os.path.exists(config_file):
        print(f"[ERROR] camera_config.yaml not found in {args.data_dir}")
        return
        
    if args.odom_config:
        odom_config = args.odom_config
    elif "Benchmark" in args.mode:
        odom_config = f"Config/Experiment/MACVO/{args.mode}.yaml"
    else:
        mode_file = "RealTime_Laptop" if args.mode in ["RealTime", "RealTime_Laptop"] else args.mode
        odom_config = f"Config/Experiment/MACVO/MACVO_{mode_file}.yaml"
        
    print("=" * 65)
    print(f"  RUNNING MAC-VO EXPERIMENT: {args.exp_id}")
    print(f"  Sequence:       {args.data_dir}")
    print(f"  Odometry Model: {odom_config}")
    print(f"  Emitter State:  {args.emitter}")
    print("=" * 65)
    
    cmd = [
        sys.executable, "MACVO.py",
        "--odom", odom_config,
        "--data", config_file,
        "--noeval"
    ]
    
    import time
    print("\nExecuting MAC-VO odometry pipeline...")
    start_t = time.time()
    ret = subprocess.run(cmd)
    wall_time = time.time() - start_t
    if ret.returncode != 0:
        print("[ERROR] MAC-VO execution failed.")
        return
        
    # Find latest results directory
    seq_name = os.path.basename(os.path.abspath(args.data_dir))
    search_pattern = os.path.join("Results", f"*@{seq_name}", "*", "poses.npy")
    matches = glob.glob(search_pattern)
    if not matches:
        matches = glob.glob(os.path.join("Results", "*", "*", "poses.npy"))
        
    if not matches:
        print("[ERROR] Could not find poses.npy in Results directory!")
        return
        
    latest_poses_file = max(matches, key=os.path.getmtime)
    results_dir = os.path.dirname(latest_poses_file)
    print(f"\n[SUCCESS] Retrieved results from: {results_dir}")
    
    poses = np.load(latest_poses_file)
    total_frames = len(poses)
    
    # poses: [ts, tx, ty, tz, qx, qy, qz, qw]
    traj = poses[:, 1:4]
    
    # Calculate path length
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    total_path_length = np.sum(deltas)
    
    # Final endpoint drift
    start_pos = traj[0]
    end_pos = traj[-1]
    final_drift = np.linalg.norm(end_pos - start_pos)
    
    fps = total_frames / wall_time if wall_time > 0 else 0
    latency_ms = (wall_time / total_frames) * 1000 if total_frames > 0 else 0
    
    print(f"\n--- Benchmark Evaluation Metrics for {args.exp_id} ---")
    print(f"  Total Frames:        {total_frames}")
    print(f"  Wall Time:           {wall_time:.2f} s")
    print(f"  Throughput (FPS):    {fps:.2f} FPS")
    print(f"  Avg Latency:         {latency_ms:.1f} ms/frame")
    print(f"  Accumulated Path:    {total_path_length:.3f} m")
    print(f"  Start-to-End Drift:  {final_drift:.4f} m ({final_drift*1000:.1f} mm)")
    print(f"  End Position (XYZ):  [{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}] m")
    
    # Plot Trajectory
    plt.figure(figsize=(10, 5))
    
    # Top-down view (X vs Z)
    plt.subplot(1, 2, 1)
    plt.plot(traj[:, 0], traj[:, 2], 'b-', label='Estimated Trajectory', linewidth=1.5)
    plt.scatter(traj[0, 0], traj[0, 2], color='green', marker='o', s=80, label='Start')
    plt.scatter(traj[-1, 0], traj[-1, 2], color='red', marker='x', s=80, label='End')
    plt.title(f"{args.exp_id} Top-Down (X-Z)\nEmitter: {args.emitter} | FPS: {fps:.1f}")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend()
    
    # Altitude over time (Y vs Frame)
    plt.subplot(1, 2, 2)
    plt.plot(traj[:, 1], 'r-', label='Vertical Y (m)')
    plt.title(f"Altitude Profile (Path: {total_path_length:.2f}m, Drift: {final_drift*1000:.1f}mm)")
    plt.xlabel("Frame Index")
    plt.ylabel("Y (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "trajectory_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Trajectory plot saved: {plot_path}")
    
    # Check IMU sample count if available
    imu_csv = os.path.join(args.data_dir, "imu.csv")
    imu_count = 0
    if os.path.exists(imu_csv):
        with open(imu_csv, "r") as f:
            imu_count = sum(1 for _ in f) - 1
            
    notes = f"Model={os.path.basename(odom_config)}, FPS={fps:.1f}, Latency={latency_ms:.1f}ms, Path={total_path_length:.2f}m, Drift={final_drift*1000:.1f}mm"
    result_str = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Drift: {final_drift*1000:.1f}mm | Path: {total_path_length:.2f}m"
    
    if args.log_sheet:
        title = f"{args.exp_id} Indoor Sequence ({args.emitter})"
        log_to_sheet(
            exp_id=args.exp_id,
            title=title,
            emitter=args.emitter,
            duration=f"{wall_time:.1f}s",
            frames=total_frames,
            imu_samples=imu_count,
            result=result_str,
            notes=notes,
            color=args.color
        )


if __name__ == "__main__":
    main()
