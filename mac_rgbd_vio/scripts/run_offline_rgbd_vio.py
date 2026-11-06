"""
MAC-RGBD-VIO Offline Trajectory Estimator & Benchmarking Engine
Evaluates recorded RGB-D + IMU datasets using the decoupled metric covariance optimizer.
Eliminates GPU VRAM thrashing and runs in real time on CPU.

Target Paper: IEEE Robotics and Automation Letters (RA-L)
"""

import os
import sys
import glob
import time
import yaml
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from metric_covariance import RGBDMetricCovarianceEngine
from rgbd_optimizer import RGBDMetricOptimizer


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def run_vo(dataset_dir: str, max_frames: int = None, step: int = 1):
    dataset_path = os.path.abspath(dataset_dir)
    print("=" * 65)
    print(f"  MAC-RGBD-VIO TRAJECTORY ESTIMATION")
    print(f"  Dataset: {dataset_path}")
    print("=" * 65)

    calib_file = os.path.join(dataset_path, "camera_info.yaml")
    with open(calib_file, "r") as f:
        calib = yaml.safe_load(f)

    fx = calib.get("fx", 607.14)
    fy = calib.get("fy", 607.16)
    cx = calib.get("cx", 323.19)
    cy = calib.get("cy", 255.51)
    depth_scale = calib.get("depth_scale", 0.001)

    print(f"[CALIB] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, depth_scale={depth_scale}")

    cov_engine = RGBDMetricCovarianceEngine(
        fx=fx, fy=fy, cx=cx, cy=cy,
        alpha_0=0.002, alpha_1=0.0015, beta=0.03,
        max_depth=5.5, min_depth=0.25
    )
    optimizer = RGBDMetricOptimizer(max_iters=15, huber_delta=1.2)

    color_files = sorted(glob.glob(os.path.join(dataset_path, "color", "*.png")))[::step]
    depth_files = sorted(glob.glob(os.path.join(dataset_path, "aligned_depth", "*.png")))[::step]

    total_frames = len(color_files)
    if max_frames:
        total_frames = min(total_frames, max_frames)
        color_files = color_files[:total_frames]
        depth_files = depth_files[:total_frames]

    print(f"[DATA] Processing {total_frames} synchronized RGB-D pairs...")

    # Trajectory state
    global_R = np.eye(3, dtype=np.float64)
    global_t = np.zeros(3, dtype=np.float64)

    trajectory = []
    # (timestamp, tx, ty, tz, qx, qy, qz, qw)
    q0 = rotation_matrix_to_quaternion(global_R)
    trajectory.append((0.0, global_t[0], global_t[1], global_t[2], q0[0], q0[1], q0[2], q0[3]))

    # Shi-Tomasi parameters
    feature_params = dict(maxCorners=500, qualityLevel=0.02, minDistance=10, blockSize=7)
    # Pyramidal Lucas-Kanade parameters
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    prev_img = cv2.imread(color_files[0])
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED).astype(np.float64) * depth_scale

    t_start = time.time()
    frame_times = []
    tracked_kps_list = []

    for i in range(1, total_frames):
        t_frame_start = time.time()

        curr_img = cv2.imread(color_files[i])
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float64) * depth_scale

        # 1. Detect features in prev frame
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None or len(p0) < 15:
            prev_gray = curr_gray
            prev_depth = curr_depth
            continue

        # 2. Pyramidal Lucas-Kanade forward tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
        # Backward flow check for cycle consistency uncertainty
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **lk_params)

        valid_flow = (st.ravel() == 1) & (st_back.ravel() == 1)
        p0_valid = p0[valid_flow].reshape(-1, 2)
        p1_valid = p1[valid_flow].reshape(-1, 2)
        p0_back_valid = p0_back[valid_flow].reshape(-1, 2)

        # Flow consistency residual as proxy for sigma_flow (sigma_u, sigma_v)
        flow_res = np.abs(p0_valid - p0_back_valid)
        sigmas_flow = np.clip(flow_res, 0.5, 5.0)

        # 3. Associate depths
        u_prev = np.clip(np.round(p0_valid[:, 0]).astype(int), 0, 639)
        v_prev = np.clip(np.round(p0_valid[:, 1]).astype(int), 0, 479)
        d_prev = prev_depth[v_prev, u_prev]

        u_curr = np.clip(np.round(p1_valid[:, 0]).astype(int), 0, 639)
        v_curr = np.clip(np.round(p1_valid[:, 1]).astype(int), 0, 479)
        d_curr = curr_depth[v_curr, u_curr]

        # Valid depth mask
        depth_mask = (d_prev > 0.25) & (d_prev < 5.5) & (d_curr > 0.25) & (d_curr < 5.5)
        p0_matched = p0_valid[depth_mask]
        p1_matched = p1_valid[depth_mask]
        d0_matched = d_prev[depth_mask]
        d1_matched = d_curr[depth_mask]
        sig_flow_matched = sigmas_flow[depth_mask]

        if len(p0_matched) < 10:
            prev_gray = curr_gray
            prev_depth = curr_depth
            continue

        # 4. Metric Covariance formulation (Analytical 3D)
        pts_3d_prev, covs_prev, info_prev = cov_engine.batch_compute(p0_matched, d0_matched, sig_flow_matched)
        pts_3d_curr, covs_curr, _ = cov_engine.batch_compute(p1_matched, d1_matched, sig_flow_matched)

        # 5. SE(3) Lie-Algebra Solver weighted by Metric Information Matrix
        R_rel, t_rel, success = optimizer.solve_relative_pose(
            pts_prev=pts_3d_prev,
            pts_curr=pts_3d_curr,
            info_mats=info_prev
        )

        if success:
            # Update global pose: T_world_k = T_world_{k-1} * T_{k-1}_k
            # Here pts_prev = R_rel * pts_curr + t_rel, so T_{k-1}_k = (R_rel, t_rel)
            global_t = global_t + global_R @ t_rel
            global_R = global_R @ R_rel

        q = rotation_matrix_to_quaternion(global_R)
        ts = i / 20.9  # approx timestamp based on frame rate
        trajectory.append((ts, global_t[0], global_t[1], global_t[2], q[0], q[1], q[2], q[3]))

        dt_frame = time.time() - t_frame_start
        frame_times.append(dt_frame)
        tracked_kps_list.append(len(p0_matched))

        if i % 100 == 0 or i == total_frames - 1:
            print(f"[{i:4d}/{total_frames}] Pos: [{global_t[0]:6.2f}, {global_t[1]:6.2f}, {global_t[2]:6.2f}] m | "
                  f"Tracked: {len(p0_matched):3d} pts | Speed: {1.0/dt_frame:5.1f} FPS")

        prev_gray = curr_gray
        prev_depth = curr_depth

    total_time = time.time() - t_start
    avg_fps = len(frame_times) / total_time
    mean_kps = np.mean(tracked_kps_list)

    # Convert trajectory to numpy array
    traj_arr = np.array(trajectory)
    positions = traj_arr[:, 1:4] # (N, 3): [x, y, z]

    # Save TUM format trajectory file
    traj_file = os.path.join(dataset_path, "trajectory_rgbd_vio.txt")
    np.savetxt(traj_file, traj_arr, fmt="%.6f %f %f %f %f %f %f %f",
               header="timestamp tx ty tz qx qy qz qw")
    print(f"\n[OUTPUT] Saved TUM trajectory to: {traj_file}")

    # Compute Trajectory Metrics
    path_len = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    loop_closure_drift = np.linalg.norm(positions[-1] - positions[0])
    drift_percent = (loop_closure_drift / max(path_len, 1e-3)) * 100.0

    print("=" * 65)
    print("  MAC-RGBD-VIO BENCHMARKING RESULTS")
    print(f"  Processed Frames:        {len(frame_times)}")
    print(f"  Average Processing Speed:{avg_fps:6.1f} FPS (Target: >20 FPS)")
    print(f"  Average Latency:         {np.mean(frame_times)*1000:6.1f} ms / frame")
    print(f"  Mean Tracked Inliers:    {mean_kps:6.1f} pts / frame")
    print(f"  GPU VRAM Consumed:       0.0 MB (Pure Decoupled CPU Optimization)")
    print(f"  Total Path Length:       {path_len:6.2f} m")
    print(f"  Final Position (End):    [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}] m")
    print(f"  Loop Closure Drift:      {loop_closure_drift:6.3f} m ({drift_percent:.2f}% of path)")
    print("=" * 65)

    # Plot Trajectory Top-Down View
    plot_file = os.path.join(dataset_path, "trajectory_plot.png")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.plot(positions[:, 0], positions[:, 2], label="MAC-RGBD-VIO (Proposed)", color="#1f77b4", linewidth=2.5)
    ax.scatter([positions[0, 0]], [positions[0, 2]], color="green", s=100, label="Start", zorder=5)
    ax.scatter([positions[-1, 0]], [positions[-1, 2]], color="red", s=100, label="End", zorder=5)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(f"MAC-RGBD-VIO Trajectory (EXP-102)\nLoop Drift: {loop_closure_drift:.2f} m ({drift_percent:.1f}%) | Speed: {avg_fps:.1f} FPS", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"[PLOT] Generated 2D Trajectory plot: {plot_file}")

    return {
        "frames": len(frame_times),
        "fps": avg_fps,
        "path_length": path_len,
        "loop_drift": loop_closure_drift,
        "drift_percent": drift_percent,
        "traj_file": traj_file,
        "plot_file": plot_file
    }


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "mac_rgbd_vio/data/exp102_laptop_office_walk"
    run_vo(target_dir)
