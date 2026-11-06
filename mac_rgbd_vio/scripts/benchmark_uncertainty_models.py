"""
Comparative Benchmark Script: Evaluating Uncertainty Models in VIO Trajectory Estimation
Compares:
- Baseline A: Fixed empirical tracking uncertainty
- Baseline B: Existing clipped forward-backward LK error (Slide 5)
- Proposed C: OptiTrack-calibrated structure-tensor + FB uncertainty engine

Metrics:
- ATE RMSE, Mean Error, Max Drift
- CPU Latency per frame (ms) & FPS throughput
- Memory usage (VRAM: 0 MB)

Target Venue: IEEE Robotics and Automation Letters (RA-L)
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from metric_covariance import RGBDMetricCovarianceEngine
from calibrated_covariance import OptiTrackCalibratedCovarianceEngine
from rgbd_optimizer import RGBDMetricOptimizer


def run_benchmark_on_sequence(dataset_dir: str, mode: str = "proposed", max_frames: int = 500):
    """
    Runs VIO relative pose tracking on dataset_dir using specified uncertainty mode.
    Modes:
        'baseline_a': Constant empirical covariance
        'baseline_b': Clipped Forward-Backward error (Slide 5)
        'proposed': OptiTrack-calibrated engine
    """
    import glob

    fx, fy, cx, cy = 607.1433, 607.1600, 323.1906, 255.5139
    depth_scale = 0.001

    if mode == "proposed":
        cov_engine = OptiTrackCalibratedCovarianceEngine(fx=fx, fy=fy, cx=cx, cy=cy)
    else:
        cov_engine = RGBDMetricCovarianceEngine(fx=fx, fy=fy, cx=cx, cy=cy)

    optimizer = RGBDMetricOptimizer(max_iters=12, huber_delta=1.2)

    color_files = sorted(glob.glob(os.path.join(dataset_dir, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(dataset_dir, "aligned_depth", "*.png")))
    total_frames = min(len(color_files), len(depth_files), max_frames)

    feature_params = dict(maxCorners=250, qualityLevel=0.015, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Trajectory state: T_W_C
    T_w_c = np.eye(4, dtype=np.float64)
    trajectory = []

    prev_gray = None
    prev_depth = None

    latencies = []

    for i in range(total_frames):
        t_start = time.perf_counter()

        curr_bgr = cv2.imread(color_files[i])
        curr_depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED) * depth_scale
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None and len(p0) >= 15:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
                p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **lk_params)

                valid = (st.ravel() == 1) & (st_back.ravel() == 1)
                p0_v = p0[valid].reshape(-1, 2)
                p1_v = p1[valid].reshape(-1, 2)
                p0_bv = p0_back[valid].reshape(-1, 2)

                u_p = np.clip(np.round(p0_v[:, 0]).astype(int), 0, 639)
                v_p = np.clip(np.round(p0_v[:, 1]).astype(int), 0, 479)
                d_p = prev_depth[v_p, u_p]

                u_c = np.clip(np.round(p1_v[:, 0]).astype(int), 0, 639)
                v_c = np.clip(np.round(p1_v[:, 1]).astype(int), 0, 479)
                d_c = curr_depth[v_c, u_c]

                mask = (d_p > 0.25) & (d_p < 3.5) & (d_c > 0.25) & (d_c < 3.5)
                if np.sum(mask) >= 10:
                    fb_err = np.abs(p0_v[mask] - p0_bv[mask])

                    if mode == "baseline_a":
                        # Constant empirical sigmas
                        sigmas = np.ones((np.sum(mask), 2)) * np.array([4.18, 2.75])
                        pts_p, _, info_p = cov_engine.batch_compute(p0_v[mask], d_p[mask], sigmas)
                        pts_c, _, _ = cov_engine.batch_compute(p1_v[mask], d_c[mask], sigmas)
                    elif mode == "baseline_b":
                        # Existing clipped forward-backward LK error (Slide 5)
                        sigmas = np.clip(fb_err, 0.5, 5.0)
                        pts_p, _, info_p = cov_engine.batch_compute(p0_v[mask], d_p[mask], sigmas)
                        pts_c, _, _ = cov_engine.batch_compute(p1_v[mask], d_c[mask], sigmas)
                    elif mode == "proposed":
                        # OptiTrack-calibrated engine
                        pts_p, _, info_p = cov_engine.batch_compute(p0_v[mask], d_p[mask], fb_err)
                        pts_c, _, _ = cov_engine.batch_compute(p1_v[mask], d_c[mask], fb_err)

                    R_rel, t_rel, ok = optimizer.solve_relative_pose(pts_prev=pts_p, pts_curr=pts_c, info_mats=info_p)
                    if ok:
                        # Kinematic velocity clamp (1.16 m/s)
                        dt = 0.0333
                        step = np.linalg.norm(t_rel)
                        if step > 1.16 * dt:
                            t_rel = t_rel * (1.16 * dt / step)

                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R_rel
                        T_rel[:3, 3] = t_rel
                        T_w_c = T_w_c @ np.linalg.inv(T_rel)

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_elapsed)

        # Store pose: tx, ty, tz, qx, qy, qz, qw
        from scipy.spatial.transform import Rotation as R_scipy
        q = R_scipy.from_matrix(T_w_c[:3, :3]).as_quat()
        t = i * 0.033333
        trajectory.append([t, T_w_c[0, 3], T_w_c[1, 3], T_w_c[2, 3], q[0], q[1], q[2], q[3]])

        prev_gray = curr_gray
        prev_depth = curr_depth

    traj_arr = np.array(trajectory)
    mean_lat = np.mean(latencies[1:])
    return traj_arr, mean_lat


def main():
    parser = argparse.ArgumentParser(description="Benchmark uncertainty models in MAC-RGBD-VIO")
    parser.add_argument("--data", type=str, default="d:/Post_doc/mac/MAC-VO/mac_rgbd_vio/data/exp102_laptop_office_walk")
    parser.add_argument("--opti", type=str, default="d:/Post_doc/mac/MAC-VO/optitrack_data/Take_new.csv")
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    print("=" * 70)
    print("  COMPARATIVE BENCHMARK: BASELINE A vs BASELINE B vs PROPOSED C")
    print("=" * 70)

    results = {}
    for mode in ["baseline_a", "baseline_b", "proposed"]:
        print(f"\n[BENCHMARK] Running VIO with mode: {mode.upper()}...")
        traj, lat = run_benchmark_on_sequence(args.data, mode=mode, max_frames=args.max_frames)
        save_file = f"d:/Post_doc/mac/MAC-VO/optitrack_data/trajectory_{mode}.txt"
        np.savetxt(save_file, traj, fmt="%.6f")
        results[mode] = {"traj_file": save_file, "latency_ms": lat, "fps": 1000.0 / lat}
        print(f"[{mode.upper()}] Mean Latency: {lat:.2f} ms ({1000.0/lat:.1f} FPS)")

    # Evaluate each against OptiTrack
    import subprocess
    print("\n" + "=" * 70)
    print("  EVALUATING AGAINST OPTITRACK GROUND TRUTH")
    print("=" * 70)
    for mode in ["baseline_a", "baseline_b", "proposed"]:
        traj_file = results[mode]["traj_file"]
        cmd = [
            sys.executable,
            "d:/Post_doc/mac/MAC-VO/mac_rgbd_vio/scripts/align_optitrack.py",
            "--est", traj_file,
            "--gt", args.opti,
            "--calibrate_hand_eye"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Parse ATE RMSE from output
        ate_rmse = "N/A"
        for line in proc.stdout.split("\n"):
            if "Absolute Trajectory Error (ATE RMSE)" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    ate_rmse = parts[1].strip()
                    break
        results[mode]["ate_rmse"] = ate_rmse
        print(f"  {mode.upper():12s} | ATE RMSE: {ate_rmse} | Latency: {results[mode]['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()

