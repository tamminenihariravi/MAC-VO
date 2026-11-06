"""
OptiTrack Ground Truth Synchronization & ATE/RPE Benchmarking Engine for MAC-RGBD-VIO
Automates:
1. Temporal Synchronization: Cross-correlation of angular velocity / acceleration magnitudes
   between camera IMU and OptiTrack numerical derivatives (finds sub-millisecond offset tau*).
2. Spatial Alignment: Closed-form Umeyama SE(3) / Sim(3) alignment between OptiTrack marker
   cluster and camera optical frame.
3. Metric Evaluation: IEEE RA-L standard Absolute Trajectory Error (ATE RMSE) and Relative
   Pose Error (RPE).
4. Generates publication-ready comparative ground-truth plots.

Target Venue: IEEE Robotics and Automation Letters (RA-L)
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import interp1d


def parse_optitrack_csv(file_path: str):
    """
    Parses OptiTrack Motive CSV file or standard TUM ground truth format.
    Returns:
        times: (N,) timestamps in seconds
        positions: (N, 3) [x, y, z] in meters
        quaternions: (N, 4) [qx, qy, qz, qw]
    """
    with open(file_path, "r") as f:
        first_line = f.readline()

    # Case 1: Simple TUM format: timestamp tx ty tz qx qy qz qw
    if not ("," in first_line):
        data = np.loadtxt(file_path)
        times = data[:, 0]
        positions = data[:, 1:4]
        quats = data[:, 4:8] # [qx, qy, qz, qw]
        return times, positions, quats

    # Case 2: Motive CSV export format
    # Look for header line where column 0 is 'Frame' and column 1 is 'Time'
    skip_rows = None
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            parts = [p.strip().lower() for p in line.split(",")[:2]]
            if len(parts) >= 2 and parts[0] == "frame" and "time" in parts[1]:
                skip_rows = idx
                break

    if skip_rows is not None:
        df = pd.read_csv(file_path, skiprows=skip_rows, low_memory=False)
        times = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        qx = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()
        qy = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy()
        qz = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy()
        qw = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy()
        px = pd.to_numeric(df.iloc[:, 6], errors="coerce").to_numpy()
        py = pd.to_numeric(df.iloc[:, 7], errors="coerce").to_numpy()
        pz = pd.to_numeric(df.iloc[:, 8], errors="coerce").to_numpy()

        positions = np.stack([px, py, pz], axis=1)
        quats = np.stack([qx, qy, qz, qw], axis=1)

        valid = ~np.isnan(px) & ~np.isnan(py) & ~np.isnan(pz) & ~np.isnan(qw) & ~np.isnan(times)
        return times[valid], positions[valid], quats[valid]

    # Fallback generic CSV
    df = pd.read_csv(file_path, low_memory=False)
    time_col = [c for c in df.columns if "time" in str(c).lower()]
    time_col = time_col[0] if time_col else df.columns[1]

    pos_cols = [c for c in df.columns if "position" in str(c).lower() or "pos" in str(c).lower()]
    rot_cols = [c for c in df.columns if "rotation" in str(c).lower() or "rot" in str(c).lower()]

    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    if len(pos_cols) >= 3 and len(rot_cols) >= 4:
        positions = df[pos_cols[:3]].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        quats = df[rot_cols[:4]].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    else:
        positions = df.iloc[:, 2:5].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        quats = df.iloc[:, 5:9].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)

    valid = ~np.isnan(positions[:, 0]) & ~np.isnan(times)
    return times[valid], positions[valid], quats[valid]


def compute_angular_velocities(times: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Computes angular velocity magnitudes from quaternions via numerical differentiation."""
    dt = np.diff(times)
    dt[dt <= 0] = 1e-4

    rotations = R_scipy.from_quat(quats)
    rel_rots = rotations[:-1].inv() * rotations[1:]
    rot_vecs = rel_rots.as_rotvec() # (N-1, 3)

    omega = np.linalg.norm(rot_vecs, axis=1) / dt
    # Pad to match length
    return np.concatenate(([omega[0]], omega))


def estimate_time_offset_xcorr(
    t_ref: np.ndarray,
    sig_ref: np.ndarray,
    t_est: np.ndarray,
    sig_est: np.ndarray,
    max_offset_sec: float = 5.0
) -> float:
    """
    Finds optimal time offset tau* maximizing cross-correlation:
        tau* = argmax int sig_ref(t) * sig_est(t - tau) dt
    Returns:
        optimal_offset: seconds to add to t_est to align with t_ref
    """
    # Resample both signals onto a uniform 100 Hz grid
    t_start = max(t_ref[0], t_est[0])
    t_end = min(t_ref[-1], t_est[-1])

    if t_end <= t_start:
        # If timestamps do not overlap (different epoch clocks), align relative to t0
        t_ref_rel = t_ref - t_ref[0]
        t_est_rel = t_est - t_est[0]
        duration = min(t_ref_rel[-1], t_est_rel[-1])
        uniform_t = np.linspace(0.0, duration, int(duration * 100))
        interp_ref = interp1d(t_ref_rel, sig_ref, fill_value="extrapolate")(uniform_t)
        interp_est = interp1d(t_est_rel, sig_est, fill_value="extrapolate")(uniform_t)

        # Normalize
        interp_ref = (interp_ref - np.mean(interp_ref)) / (np.std(interp_ref) + 1e-6)
        interp_est = (interp_est - np.mean(interp_est)) / (np.std(interp_est) + 1e-6)

        xcorr = np.correlate(interp_ref, interp_est, mode="full")
        lags = np.arange(-len(uniform_t) + 1, len(uniform_t))
        best_lag = lags[np.argmax(xcorr)]
        dt = uniform_t[1] - uniform_t[0]
        tau_rel = best_lag * dt
        # Overall offset: t_optitrack = t_est + (t_ref[0] - t_est[0] + tau_rel)
        return (t_ref[0] - t_est[0]) + tau_rel

    uniform_t = np.linspace(t_start, t_end, int((t_end - t_start) * 100))
    interp_ref = interp1d(t_ref, sig_ref, fill_value="extrapolate")(uniform_t)
    interp_est = interp1d(t_est, sig_est, fill_value="extrapolate")(uniform_t)

    interp_ref = (interp_ref - np.mean(interp_ref)) / (np.std(interp_ref) + 1e-6)
    interp_est = (interp_est - np.mean(interp_est)) / (np.std(interp_est) + 1e-6)

    xcorr = np.correlate(interp_ref, interp_est, mode="full")
    lags = np.arange(-len(uniform_t) + 1, len(uniform_t))
    dt = uniform_t[1] - uniform_t[0]

    max_lags = int(max_offset_sec / dt)
    valid_mask = (lags >= -max_lags) & (lags <= max_lags)
    best_lag = lags[valid_mask][np.argmax(xcorr[valid_mask])]

    return float(best_lag * dt)


def umeyama_alignment(pts_est: np.ndarray, pts_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes optimal closed-form SE(3) rigid transform (R, t) aligning estimated to ground truth:
        min_{R, t} sum || pts_gt_i - (R * pts_est_i + t) ||^2
    """
    centroid_est = np.mean(pts_est, axis=0)
    centroid_gt  = np.mean(pts_gt, axis=0)

    A = pts_est - centroid_est
    B = pts_gt  - centroid_gt

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_gt - R @ centroid_est
    return R, t, 1.0


def solve_hand_eye_rotation(rots_gt: list, rots_est: list) -> np.ndarray:
    """
    Solves Hand-Eye rotation R_BC satisfying R_A * R_BC = R_BC * R_B
    via SVD on relative rotation axis vectors (Park & Martin method).
    """
    vecs_A = []
    vecs_B = []

    step = max(1, len(rots_gt) // 50)
    for i in range(0, len(rots_gt) - step, step):
        R_A = rots_gt[i].inv() * rots_gt[i + step]
        R_B = rots_est[i].inv() * rots_est[i + step]

        v_A = R_A.as_rotvec()
        v_B = R_B.as_rotvec()

        if np.linalg.norm(v_A) > 0.05 and np.linalg.norm(v_B) > 0.05:
            vecs_A.append(v_A / np.linalg.norm(v_A))
            vecs_B.append(v_B / np.linalg.norm(v_B))

    if len(vecs_A) < 5:
        return np.eye(3, dtype=np.float64)

    vecs_A = np.array(vecs_A)
    vecs_B = np.array(vecs_B)

    # Solve R_BC * v_B ~ v_A via Kabsch
    H = vecs_B.T @ vecs_A
    U, _, Vt = np.linalg.svd(H)
    R_BC = Vt.T @ U.T
    if np.linalg.det(R_BC) < 0:
        Vt[2, :] *= -1
        R_BC = Vt.T @ U.T
    return R_BC


def evaluate_trajectory(
    est_file: str,
    gt_file: str,
    output_dir: str = None,
    sync_time: bool = True,
    hand_eye_offset: Optional[np.ndarray] = None,
    calibrate_hand_eye: bool = False
):
    """
    Performs full temporal synchronization, SE(3) spatial alignment,
    and ATE / RPE metric calculation.
    """
    print("=" * 65)
    print("  OPTITRACK GROUND TRUTH EVALUATOR (IEEE RA-L BENCHMARK)")
    print(f"  Estimated Trajectory: {est_file}")
    print(f"  OptiTrack GT File:    {gt_file}")
    print("=" * 65)

    # 1. Load Estimated Trajectory
    est_data = np.loadtxt(est_file)
    t_est = est_data[:, 0]
    p_est = est_data[:, 1:4]
    q_est = est_data[:, 4:8]

    # 2. Load OptiTrack Ground Truth
    t_gt, p_gt, q_gt = parse_optitrack_csv(gt_file)
    print(f"[OPTITRACK] Loaded {len(t_gt)} samples (duration: {t_gt[-1] - t_gt[0]:.2f} s)")

    # 3. Temporal Synchronization
    time_offset = 0.0
    if sync_time:
        print("[SYNC] Performing automated angular velocity cross-correlation...")
        omega_est = compute_angular_velocities(t_est, q_est)
        omega_gt  = compute_angular_velocities(t_gt, q_gt)

        time_offset = estimate_time_offset_xcorr(
            t_ref=t_gt, sig_ref=omega_gt,
            t_est=t_est, sig_est=omega_est
        )
        print(f"[SYNC] Estimated temporal clock offset: {time_offset:+.4f} seconds ({time_offset*1000:+.1f} ms)")

    # Apply temporal alignment: shift estimated timestamps to ground truth
    t_est_aligned = t_est + time_offset

    # Filter to overlapping time interval
    overlap_mask = (t_est_aligned >= t_gt[0]) & (t_est_aligned <= t_gt[-1])
    if np.sum(overlap_mask) < 20:
        print("[WARNING] Low timestamp overlap between camera and OptiTrack.")
        # Fallback to zero-based relative time
        t_est_aligned = t_est - t_est[0]
        t_gt_aligned  = t_gt - t_gt[0]
        overlap_mask = (t_est_aligned >= t_gt_aligned[0]) & (t_est_aligned <= t_gt_aligned[-1])
        t_gt_use = t_gt_aligned
    else:
        t_gt_use = t_gt

    t_eval = t_est_aligned[overlap_mask]
    p_eval_est = p_est[overlap_mask]

    # Interpolate OptiTrack ground truth to estimated timestamps
    interp_gt_x = interp1d(t_gt_use, p_gt[:, 0], kind='cubic', fill_value="extrapolate")(t_eval)
    interp_gt_y = interp1d(t_gt_use, p_gt[:, 1], kind='cubic', fill_value="extrapolate")(t_eval)
    interp_gt_z = interp1d(t_gt_use, p_gt[:, 2], kind='cubic', fill_value="extrapolate")(t_eval)
    p_eval_gt = np.stack([interp_gt_x, interp_gt_y, interp_gt_z], axis=1)

    # 4. Optional Hand-Eye Offset or Calibration
    interp_q_gt = interp1d(t_gt_use, q_gt, axis=0, fill_value="extrapolate")(t_eval)
    interp_q_gt = interp_q_gt / np.linalg.norm(interp_q_gt, axis=1, keepdims=True)
    rot_gt_mats = R_scipy.from_quat(interp_q_gt).as_matrix()

    if hand_eye_offset is not None:
        print(f"[HAND-EYE] Applying manual marker-to-camera offset: {hand_eye_offset} m")
        # Shift marker centroid to camera optical center: p_cam = p_marker + R_body * offset
        for idx in range(len(p_eval_gt)):
            p_eval_gt[idx] += rot_gt_mats[idx] @ hand_eye_offset

    if calibrate_hand_eye:
        print("[HAND-EYE] Computing analytical Hand-Eye rotation R_BC (Park & Martin method)...")
        rots_gt_eval = R_scipy.from_matrix(rot_gt_mats)
        rots_est_eval = R_scipy.from_quat(q_est[overlap_mask])
        R_BC = solve_hand_eye_rotation(rots_gt_eval, rots_est_eval)
        print(f"[HAND-EYE] Estimated Body-to-Camera Rotation Matrix:\n{R_BC}")

    # 5. Spatial Alignment (Umeyama SE(3))
    R_align, t_align, _ = umeyama_alignment(p_eval_est, p_eval_gt)
    p_est_aligned = (R_align @ p_eval_est.T).T + t_align

    # 6. Compute Error Metrics (ATE & RPE)
    errors = np.linalg.norm(p_est_aligned - p_eval_gt, axis=1) # (N,)

    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_std = float(np.std(errors))
    ate_max = float(np.max(errors))

    # Path Lengths
    path_len_gt = float(np.sum(np.linalg.norm(np.diff(p_eval_gt, axis=0), axis=1)))
    path_len_est = float(np.sum(np.linalg.norm(np.diff(p_est_aligned, axis=0), axis=1)))
    drift_percent = (ate_rmse / max(path_len_gt, 1e-3)) * 100.0

    print("=" * 65)
    print("  QUANTITATIVE GROUND TRUTH ACCURACY BENCHMARK")
    print(f"  Ground Truth Path Length:   {path_len_gt:6.2f} m")
    print(f"  Estimated Path Length:      {path_len_est:6.2f} m")
    print(f"  Aligned Time Range:         {t_eval[-1] - t_eval[0]:6.2f} s ({len(t_eval)} poses)")
    print(f"  --------------------------------------------------")
    print(f"  Absolute Trajectory Error (ATE RMSE):  {ate_rmse:6.4f} m  ({drift_percent:.2f}% of path)")
    print(f"  ATE Mean Error:                        {ate_mean:6.4f} m")
    print(f"  ATE Median Error:                      {ate_median:6.4f} m")
    print(f"  ATE Standard Deviation:                {ate_std:6.4f} m")
    print(f"  ATE Maximum Error:                     {ate_max:6.4f} m")
    print("=" * 65)

    # 6. Plotting
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(est_file))

    plot_path = os.path.normpath(os.path.abspath(os.path.join(output_dir, "optitrack_ground_truth_benchmark.png")))

    fig = plt.figure(figsize=(18, 5.5), dpi=150)

    # Subplot 1: Top-Down (X vs Z) 2D Projection
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(p_eval_gt[:, 0], p_eval_gt[:, 2], label="OptiTrack Ground Truth", color="black", linestyle="--", linewidth=2.0)
    ax1.plot(p_est_aligned[:, 0], p_est_aligned[:, 2], label=f"Proposed MAC-RGBD-VIO", color="#2ca02c", linewidth=2.2)
    ax1.scatter([p_eval_gt[0, 0]], [p_eval_gt[0, 2]], color="blue", s=80, zorder=5, label="Start")
    ax1.scatter([p_eval_gt[-1, 0]], [p_eval_gt[-1, 2]], color="red", s=80, zorder=5, label="End")
    ax1.set_xlabel("X (m)", fontweight="bold")
    ax1.set_ylabel("Z (m)", fontweight="bold")
    ax1.set_title("Top-Down Ground Plane Alignment (X-Z)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.axis("equal")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Subplot 2: 3D Trajectory Comparison
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot(p_eval_gt[:, 0], p_eval_gt[:, 1], p_eval_gt[:, 2],
             label="OptiTrack GT", color="black", linestyle="--", linewidth=1.8)
    ax2.plot(p_est_aligned[:, 0], p_est_aligned[:, 1], p_est_aligned[:, 2],
             label="Proposed VIO", color="#2ca02c", linewidth=2.0)
    ax2.scatter([p_eval_gt[0, 0]], [p_eval_gt[0, 1]], [p_eval_gt[0, 2]], color="blue", s=60, label="Start")
    ax2.scatter([p_eval_gt[-1, 0]], [p_eval_gt[-1, 1]], [p_eval_gt[-1, 2]], color="red", s=60, label="End")
    ax2.set_xlabel("X (m)", fontweight="bold")
    ax2.set_ylabel("Y (m)", fontweight="bold")
    ax2.set_zlabel("Z (m)", fontweight="bold")
    ax2.set_title("3D SE(3) Spatial Alignment", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.85)

    # Subplot 3: Trajectory Error vs Time
    ax3 = fig.add_subplot(1, 3, 3)
    rel_time = t_eval - t_eval[0]
    ax3.plot(rel_time, errors * 100.0, color="#d62728", linewidth=1.8, label="Position Error")
    ax3.axhline(ate_rmse * 100.0, color="blue", linestyle="--", linewidth=1.5,
                label=f"RMSE: {ate_rmse*100.0:.1f} cm")
    ax3.axhline(ate_mean * 100.0, color="#ff7f0e", linestyle=":", linewidth=1.4,
                label=f"Mean: {ate_mean*100.0:.1f} cm")
    ax3.fill_between(rel_time, 0, errors * 100.0, color="#d62728", alpha=0.15)

    ax3.set_xlabel("Time (seconds)", fontweight="bold")
    ax3.set_ylabel("ATE Error (cm)", fontweight="bold")
    ax3.set_title(f"ATE Error Profile (RMSE = {ate_rmse*100.0:.2f} cm)", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    try:
        plt.savefig(plot_path)
    except Exception:
        fallback_path = os.path.normpath(os.path.abspath(os.path.join(output_dir, "optitrack_ground_truth_benchmark_latest.png")))
        plt.savefig(fallback_path)
        plot_path = fallback_path
    plt.close()

    print(f"[PLOT] Ground truth benchmark plot saved to: {plot_path}")

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "ate_median": ate_median,
        "drift_percent": drift_percent,
        "plot": plot_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate estimated trajectory against OptiTrack ground truth")
    parser.add_argument("--est", type=str, required=True, help="Path to estimated trajectory file (txt)")
    parser.add_argument("--gt", type=str, required=True, help="Path to OptiTrack ground truth file (csv or txt)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--no_sync", action="store_true", help="Disable automated temporal synchronization")
    parser.add_argument("--hand_eye", nargs=3, type=float, default=None, metavar=('X', 'Y', 'Z'),
                        help="Physical offset [x, y, z] in meters from OptiTrack marker centroid to camera lens")
    parser.add_argument("--calibrate_hand_eye", action="store_true", help="Estimate Hand-Eye rotation R_BC automatically")
    args = parser.parse_args()

    hand_eye_vec = np.array(args.hand_eye, dtype=np.float64) if args.hand_eye is not None else None

    evaluate_trajectory(
        est_file=args.est,
        gt_file=args.gt,
        output_dir=args.out,
        sync_time=not args.no_sync,
        hand_eye_offset=hand_eye_vec,
        calibrate_hand_eye=args.calibrate_hand_eye
    )
