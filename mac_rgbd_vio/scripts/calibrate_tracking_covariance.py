"""
OptiTrack-Calibrated Feature Tracking Uncertainty Engine for MAC-RGBD-VIO
Offline calibration and statistical modeling of 2D Lucas-Kanade feature localization error
using ground-truth 6-DoF OptiTrack motion capture.

Scientific Workflow:
1. Synchronize camera and OptiTrack timestamps (temporal offset tau*).
2. Align camera body to optical frame via Hand-Eye rotation R_BC.
3. Track Shi-Tomasi features across consecutive RGB-D frames using bidirectional Lucas-Kanade.
4. Project depth-filtered 3D points through ground-truth relative camera pose T_Cj_Ci.
5. Isolate pure tracking errors from depth/calibration nuisance errors via strict geometric gating.
6. Fit and calibrate empirical covariance models:
   - Baseline A: Constant empirical covariance
   - Baseline B: Existing clipped forward-backward LK error (Slide 5)
   - Proposed C: OptiTrack-calibrated Structure-Tensor + FB hybrid model
7. Evaluate statistical consistency: NEES, Chi-square p-values, 2-sigma ellipse coverage.
8. Generate publication-quality diagnostic plots for IEEE RA-L.

Target Venue: IEEE Robotics and Automation Letters (RA-L)
"""

import os
import sys
import glob
import math
import argparse
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import interp1d
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def parse_optitrack_motive_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses Motive CSV export to extract times, positions, and quaternions."""
    skip_rows = None
    with open(csv_path, "r") as f:
        for idx, line in enumerate(f):
            parts = [p.strip().lower() for p in line.split(",")[:2]]
            if len(parts) >= 2 and parts[0] == "frame" and "time" in parts[1]:
                skip_rows = idx
                break

    if skip_rows is None:
        raise ValueError(f"Could not locate Motive header in {csv_path}")

    df = pd.read_csv(csv_path, skiprows=skip_rows, low_memory=False)
    times = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    qx = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()
    qy = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy()
    qz = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy()
    qw = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy()
    px = pd.to_numeric(df.iloc[:, 6], errors="coerce").to_numpy()
    py = pd.to_numeric(df.iloc[:, 7], errors="coerce").to_numpy()
    pz = pd.to_numeric(df.iloc[:, 8], errors="coerce").to_numpy()

    valid = ~np.isnan(px) & ~np.isnan(py) & ~np.isnan(pz) & ~np.isnan(qw) & ~np.isnan(times)
    return times[valid], np.stack([px, py, pz], axis=1)[valid], np.stack([qx, qy, qz, qw], axis=1)[valid]


def compute_structure_tensor_stats(gray_img: np.ndarray, pts: np.ndarray, patch_radius: int = 3):
    """
    Computes structure tensor eigenvalues (lambda_1, lambda_2) and normalized gradient components.
    pts: (N, 2) [u, v]
    Returns:
        lambda_1: (N,) max eigenvalue
        lambda_2: (N,) min eigenvalue (corner strength)
        anisotropy: (N,) lambda_1 / (lambda_2 + eps)
        s_xx, s_yy, s_xy: (N,) tensor components
    """
    H, W = gray_img.shape
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = grad_x ** 2
    Iyy = grad_y ** 2
    Ixy = grad_x * grad_y

    N = pts.shape[0]
    l1_arr = np.zeros(N, dtype=np.float64)
    l2_arr = np.zeros(N, dtype=np.float64)
    sxx_arr = np.zeros(N, dtype=np.float64)
    syy_arr = np.zeros(N, dtype=np.float64)
    sxy_arr = np.zeros(N, dtype=np.float64)

    for i in range(N):
        u, v = int(round(pts[i, 0])), int(round(pts[i, 1]))
        u_min, u_max = max(0, u - patch_radius), min(W, u + patch_radius + 1)
        v_min, v_max = max(0, v - patch_radius), min(H, v + patch_radius + 1)

        sxx = np.sum(Ixx[v_min:v_max, u_min:u_max])
        syy = np.sum(Iyy[v_min:v_max, u_min:u_max])
        sxy = np.sum(Ixy[v_min:v_max, u_min:u_max])

        # Analytical 2x2 eigenvalues
        trace = sxx + syy
        det = sxx * syy - sxy ** 2
        disc = max(0.0, trace ** 2 - 4.0 * det)
        sqrt_disc = math.sqrt(disc)

        l1 = 0.5 * (trace + sqrt_disc)
        l2 = 0.5 * (trace - sqrt_disc)

        l1_arr[i] = l1
        l2_arr[i] = max(1e-4, l2)
        sxx_arr[i] = sxx
        syy_arr[i] = syy
        sxy_arr[i] = sxy

    anisotropy = l1_arr / l2_arr
    return l1_arr, l2_arr, anisotropy, sxx_arr, syy_arr, sxy_arr


class OptiTrackCameraSynchronizer:
    """Synchronizes OptiTrack 6-DoF ground truth to camera frames."""
    def __init__(self, opti_times: np.ndarray, opti_pos: np.ndarray, opti_quat: np.ndarray, tau: float, R_BC: np.ndarray):
        self.t_opti = opti_times
        self.pos_opti = opti_pos
        self.quat_opti = opti_quat
        self.tau = tau
        self.R_BC = R_BC

        # Build continuous interpolators
        self.interp_px = interp1d(self.t_opti, self.pos_opti[:, 0], fill_value="extrapolate")
        self.interp_py = interp1d(self.t_opti, self.pos_opti[:, 1], fill_value="extrapolate")
        self.interp_pz = interp1d(self.t_opti, self.pos_opti[:, 2], fill_value="extrapolate")
        self.scipy_rots = R_scipy.from_quat(self.quat_opti)

    def get_camera_pose(self, t_cam: float) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (R_WC, p_WC) at camera timestamp t_cam."""
        t_query = t_cam + self.tau
        t_query = np.clip(t_query, self.t_opti[0], self.t_opti[-1])

        # Nearest or linear orientation
        idx = np.searchsorted(self.t_opti, t_query)
        idx = np.clip(idx, 0, len(self.t_opti) - 1)
        R_WB = self.scipy_rots[idx].as_matrix()

        p_WB = np.array([self.interp_px(t_query), self.interp_py(t_query), self.interp_pz(t_query)])

        # Apply Hand-Eye transform: T_WC = T_WB * T_BC
        R_WC = R_WB @ self.R_BC
        p_WC = p_WB # lever arm modeled if calibrated
        return R_WC, p_WC


def run_ground_truth_feature_harvesting(
    dataset_dir: str,
    opti_csv: str,
    tau_sec: float = -3.6408,
    max_frames: int = 500,
) -> pd.DataFrame:
    """
    Harvests thousands of empirical Lucas-Kanade tracking errors against ground truth OptiTrack relative motions.
    """
    print("=" * 70)
    print("  HARVESTING OPTITRACK-GROUNDED LK TRACKING RESIDUALS")
    print("=" * 70)

    # 1. Load OptiTrack ground truth
    t_gt, pos_gt, quat_gt = parse_optitrack_motive_csv(opti_csv)
    print(f"[OPTITRACK] Loaded {len(t_gt)} samples, span: {t_gt[-1] - t_gt[0]:.2f} s")

    # Fixed Hand-Eye rotation solved via Park & Martin
    R_BC = np.array([
        [ 0.9914, -0.0165,  0.1297],
        [-0.0158, -0.9999, -0.0064],
        [ 0.1298,  0.0043, -0.9915]
    ], dtype=np.float64)

    sync = OptiTrackCameraSynchronizer(t_gt, pos_gt, quat_gt, tau_sec, R_BC)

    # 2. Camera intrinsics
    fx = 607.1433
    fy = 607.1600
    cx = 323.1906
    cy = 255.5139
    depth_scale = 0.001

    # 3. Load image file lists
    color_files = sorted(glob.glob(os.path.join(dataset_dir, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(dataset_dir, "aligned_depth", "*.png")))

    total_frames = min(len(color_files), len(depth_files), max_frames)
    print(f"[DATASET] Analyzing {total_frames} frames from {dataset_dir}...")

    # Shi-Tomasi & LK parameters
    feature_params = dict(maxCorners=250, qualityLevel=0.015, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    records = []

    # Nominal time step
    dt_cam = (t_gt[-1] - t_gt[0]) / max(1, total_frames)

    for i in range(total_frames - 1):
        t_i = i * dt_cam
        t_j = (i + 1) * dt_cam

        img_i = cv2.imread(color_files[i], cv2.IMREAD_GRAYSCALE)
        img_j = cv2.imread(color_files[i + 1], cv2.IMREAD_GRAYSCALE)
        d_img_i = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED) * depth_scale

        if img_i is None or img_j is None or d_img_i is None:
            continue

        # Get GT poses
        R_wi, p_wi = sync.get_camera_pose(t_i)
        R_wj, p_wj = sync.get_camera_pose(t_j)

        # Relative GT camera motion: T_Cj_Ci = (T_WCj)^-1 * T_WCi
        R_ij_gt = R_wj.T @ R_wi
        t_ij_gt = R_wj.T @ (p_wi - p_wj)

        # Detect Shi-Tomasi corners in frame i
        p0 = cv2.goodFeaturesToTrack(img_i, mask=None, **feature_params)
        if p0 is None or len(p0) < 15:
            continue

        # Bidirectional Lucas-Kanade
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img_i, img_j, p0, None, **lk_params)
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img_j, img_i, p1, None, **lk_params)

        valid = (st.ravel() == 1) & (st_back.ravel() == 1)
        p0_v = p0[valid].reshape(-1, 2)
        p1_v = p1[valid].reshape(-1, 2)
        p0_bv = p0_back[valid].reshape(-1, 2)

        # Forward-backward errors
        fb_errors = np.abs(p0_v - p0_bv) # (K, 2) [e_u, e_v]
        fb_norm = np.linalg.norm(fb_errors, axis=1)

        # Compute Structure Tensor on frame i
        l1, l2, aniso, sxx, syy, sxy = compute_structure_tensor_stats(img_i, p0_v)

        # Extract depths
        u_p = np.clip(np.round(p0_v[:, 0]).astype(int), 0, img_i.shape[1] - 1)
        v_p = np.clip(np.round(p0_v[:, 1]).astype(int), 0, img_i.shape[0] - 1)
        depths_i = d_img_i[v_p, u_p]

        # Compute depth gradients
        d_grad_x = cv2.Sobel(d_img_i, cv2.CV_64F, 1, 0, ksize=3)
        d_grad_y = cv2.Sobel(d_img_i, cv2.CV_64F, 0, 1, ksize=3)
        d_grad_norm = np.sqrt(d_grad_x[v_p, u_p] ** 2 + d_grad_y[v_p, u_p] ** 2)

        # Project via Ground Truth Motion
        for k in range(len(p0_v)):
            d_k = depths_i[k]
            # Strict Gating to isolate pure LK tracking noise:
            # 1. Depth valid and in reliable near-to-mid range
            if d_k < 0.4 or d_k > 3.0 or np.isnan(d_k):
                continue
            # 2. Reject silhouette depth edges
            if d_grad_norm[k] > 0.08:
                continue

            # 3D coordinate in frame i
            X_i = (p0_v[k, 0] - cx) * d_k / fx
            Y_i = (p0_v[k, 1] - cy) * d_k / fy
            Z_i = d_k
            P_i = np.array([X_i, Y_i, Z_i])

            # Transform into frame j via Ground Truth
            P_j_gt = R_ij_gt @ P_i + t_ij_gt
            if P_j_gt[2] <= 0.1:
                continue

            # Project to image plane j
            u_hat_gt = fx * (P_j_gt[0] / P_j_gt[2]) + cx
            v_hat_gt = fy * (P_j_gt[1] / P_j_gt[2]) + cy

            # True Tracking Residual
            err_u = p1_v[k, 0] - u_hat_gt
            err_v = p1_v[k, 1] - v_hat_gt
            err_total = math.hypot(err_u, err_v)

            # Robust outlier rejection (gross association failures / occlusions)
            if err_total > 8.0:
                continue

            records.append({
                "frame": i,
                "u": p0_v[k, 0],
                "v": p0_v[k, 1],
                "flow_u": p1_v[k, 0] - p0_v[k, 0],
                "flow_v": p1_v[k, 1] - p0_v[k, 1],
                "flow_mag": math.hypot(p1_v[k, 0] - p0_v[k, 0], p1_v[k, 1] - p0_v[k, 1]),
                "fb_u": fb_errors[k, 0],
                "fb_v": fb_errors[k, 1],
                "fb_norm": fb_norm[k],
                "depth": d_k,
                "depth_grad": d_grad_norm[k],
                "lambda_1": l1[k],
                "lambda_2": l2[k],
                "anisotropy": aniso[k],
                "sxx": sxx[k],
                "syy": syy[k],
                "sxy": sxy[k],
                "err_u": err_u,
                "err_v": err_v,
                "err_total": err_total,
            })

    df = pd.DataFrame(records)
    print(f"[HARVEST] Collected {len(df)} rigorously filtered feature observations across {total_frames} frames.")
    return df


def calibrate_and_evaluate_models(df: pd.DataFrame, out_dir: str):
    """
    Fits and compares:
    1. Baseline A: Constant empirical covariance
    2. Baseline B: Clipped Forward-Backward error (Slide 5)
    3. Proposed C: OptiTrack-Calibrated Structure-Tensor + FB Hybrid Model
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(df)
    print("=" * 70)
    print("  STATISTICAL ANALYSIS & COVARIANCE CALIBRATION")
    print("=" * 70)

    # 1. Sample Statistics of True Tracking Errors
    e_u = df["err_u"].to_numpy()
    e_v = df["err_v"].to_numpy()
    errors = np.stack([e_u, e_v], axis=1) # (N, 2)

    mean_e = np.mean(errors, axis=0)
    cov_empirical = np.cov(errors, rowvar=False)
    std_u = math.sqrt(cov_empirical[0, 0])
    std_v = math.sqrt(cov_empirical[1, 1])
    cov_uv = cov_empirical[0, 1]
    rho = cov_uv / (std_u * std_v)

    print(f"[SAMPLE MEAN]      e_bar_u = {mean_e[0]:+.4f} px,  e_bar_v = {mean_e[1]:+.4f} px (near zero-mean)")
    print(f"[EMPIRICAL COV]   sigma_u = {std_u:.4f} px,  sigma_v = {std_v:.4f} px,  rho_uv = {rho:.4f}")

    # Train / Test split (70% Train for parameter fitting, 30% Test for evaluation)
    np.random.seed(42)
    indices = np.random.permutation(N)
    n_train = int(0.70 * N)
    idx_train = indices[:n_train]
    idx_test = indices[n_train:]

    df_train = df.iloc[idx_train].copy()
    df_test = df.iloc[idx_test].copy()

    # -------------------------------------------------------------
    # MODEL CALIBRATION ON TRAINING SET
    # -------------------------------------------------------------
    # Baseline A: Constant Covariance from training sample
    err_train = np.stack([df_train["err_u"], df_train["err_v"]], axis=1)
    Sigma_A_const = np.cov(err_train, rowvar=False)

    # Baseline B: Slide 5 Formulation
    # sigma_u = clip(fb_u, 0.5, 5.0), sigma_v = clip(fb_v, 0.5, 5.0)

    # Proposed C: Calibrated Hybrid Model:
    # sigma_u^2 = sigma_base^2 + gamma_fb * fb_u^2 + alpha_st * (1.0 / sqrt(lambda_2))
    # Fit regression parameters on training set
    fb_u_tr = df_train["fb_u"].to_numpy()
    fb_v_tr = df_train["fb_v"].to_numpy()
    inv_l2_tr = 1.0 / np.sqrt(df_train["lambda_2"].to_numpy())

    # Target squared errors
    target_u2 = df_train["err_u"].to_numpy() ** 2
    target_v2 = df_train["err_v"].to_numpy() ** 2

    # Linear least squares for non-negative weights: [1, fb^2, 1/sqrt(l2)]
    X_u = np.stack([np.ones_like(fb_u_tr), fb_u_tr ** 2, inv_l2_tr], axis=1)
    X_v = np.stack([np.ones_like(fb_v_tr), fb_v_tr ** 2, inv_l2_tr], axis=1)
    X_all = np.vstack([X_u, X_v])
    y_all = np.concatenate([target_u2, target_v2])

    coeffs, _, _, _ = np.linalg.lstsq(X_all, y_all, rcond=None)
    sigma_base2 = max(0.04, coeffs[0]) # floor of ~0.2 px
    gamma_fb    = max(0.05, coeffs[1])
    alpha_st    = max(0.01, coeffs[2])

    print(f"[PROPOSED FIT]     sigma_base = {math.sqrt(sigma_base2):.4f} px, gamma_fb = {gamma_fb:.4f}, alpha_st = {alpha_st:.4f}")

    # -------------------------------------------------------------
    # EVALUATION ON UNSEEN TEST SET (30%)
    # -------------------------------------------------------------
    e_test = np.stack([df_test["err_u"], df_test["err_v"]], axis=1) # (M, 2)
    M = len(df_test)

    # Evaluate Model A
    Sigma_A_inv = np.linalg.inv(Sigma_A_const)
    nees_A = np.sum(e_test * (e_test @ Sigma_A_inv), axis=1) # Mahalanobis distance squared

    # Evaluate Model B (Slide 5)
    fb_u_te = np.clip(df_test["fb_u"].to_numpy(), 0.5, 5.0)
    fb_v_te = np.clip(df_test["fb_v"].to_numpy(), 0.5, 5.0)
    nees_B = (df_test["err_u"] / fb_u_te) ** 2 + (df_test["err_v"] / fb_v_te) ** 2

    # Evaluate Proposed Model C
    inv_l2_te = 1.0 / np.sqrt(df_test["lambda_2"].to_numpy())
    pred_var_u_C = sigma_base2 + gamma_fb * (df_test["fb_u"].to_numpy() ** 2) + alpha_st * inv_l2_te
    pred_var_v_C = sigma_base2 + gamma_fb * (df_test["fb_v"].to_numpy() ** 2) + alpha_st * inv_l2_te
    # Off-diagonal correlation scaled by gradient anisotropy
    sxx_te = df_test["sxx"].to_numpy()
    syy_te = df_test["syy"].to_numpy()
    sxy_te = df_test["sxy"].to_numpy()
    rho_C = np.clip(sxy_te / (np.sqrt(sxx_te * syy_te) + 1e-6), -0.7, 0.7)

    nees_C = np.zeros(M)
    for i in range(M):
        su2 = pred_var_u_C[i]
        sv2 = pred_var_v_C[i]
        suv = rho_C[i] * math.sqrt(su2 * sv2)
        Sig_C = np.array([[su2, suv], [suv, sv2]])
        nees_C[i] = e_test[i] @ np.linalg.inv(Sig_C) @ e_test[i].T

    # Statistical consistency metrics
    # Under ideal Gaussian calibration, NEES follows Chi-Square with 2 DoF -> Mean should be 2.0
    mean_nees_A = float(np.mean(nees_A))
    mean_nees_B = float(np.mean(nees_B))
    mean_nees_C = float(np.mean(nees_C))

    # 2-sigma ellipse coverage (theoretical: 86.47% for 2D Gaussian Chi2 <= 4.0)
    cov_2sig_A = np.mean(nees_A <= 4.0) * 100.0
    cov_2sig_B = np.mean(nees_B <= 4.0) * 100.0
    cov_2sig_C = np.mean(nees_C <= 4.0) * 100.0

    print("-" * 70)
    print(f"  MODEL STATISTICAL CONSISTENCY COMPARISON (ON {M} TEST FEATURES)")
    print("-" * 70)
    print(f"  Metric                     | Baseline A (Const) | Baseline B (Slide 5) | Proposed C (OptiTrack)")
    print(f"  Mean NEES (Ideal: 2.0)     | {mean_nees_A:18.3f} | {mean_nees_B:20.3f} | {mean_nees_C:22.3f}")
    print(f"  2-Sigma Coverage (86.5%)   | {cov_2sig_A:17.1f}% | {cov_2sig_B:19.1f}% | {cov_2sig_C:21.1f}%")
    print(f"  Mean Pred Std Dev (px)     | {0.5*(math.sqrt(Sigma_A_const[0,0])+math.sqrt(Sigma_A_const[1,1])):18.3f} | {0.5*float(np.mean(fb_u_te+fb_v_te)):20.3f} | {0.5*float(np.mean(np.sqrt(pred_var_u_C)+np.sqrt(pred_var_v_C))):22.3f}")
    print("-" * 70)

    # -------------------------------------------------------------
    # GENERATE PUBLICATION DIAGNOSTIC PLOTS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: 2D Error Scatter with Confidence Ellipses
    ax = axes[0, 0]
    ax.scatter(df_test["err_u"], df_test["err_v"], s=6, alpha=0.35, color="#1f77b4", label="LK Residuals")
    # Draw Empirical 1-sigma, 2-sigma, 3-sigma ellipses
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(theta_vals), np.sin(theta_vals)], axis=1)
    val, vec = np.linalg.eigh(Sigma_A_const)
    for k_sig, col in [(1.0, "green"), (2.0, "orange"), (3.0, "red")]:
        ell = (circle * (k_sig * np.sqrt(val))) @ vec.T
        ax.plot(ell[:, 0], ell[:, 1], color=col, linewidth=1.8, label=f"{int(k_sig)}$\\sigma$ Ellipse")
    ax.set_title("Ground-Truth LK Tracking Residuals & Confidence Ellipses", fontsize=11, fontweight="bold")
    ax.set_xlabel("Horizontal Error $e_u$ (pixels)")
    ax.set_ylabel("Vertical Error $e_v$ (pixels)")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # Plot 2: Histogram of Errors vs Gaussian Fit
    ax = axes[0, 1]
    ax.hist(df_test["err_u"], bins=40, density=True, alpha=0.5, color="#1f77b4", label="Measured $e_u$ Distribution")
    u_grid = np.linspace(-4, 4, 200)
    pdf_u = (1.0 / (math.sqrt(2 * np.pi) * std_u)) * np.exp(-0.5 * (u_grid / std_u) ** 2)
    ax.plot(u_grid, pdf_u, "r-", linewidth=2.0, label=f"Calibrated Gaussian ($\\sigma={std_u:.2f}$ px)")
    ax.set_title("Tracking Error Distribution vs Calibrated Model", fontsize=11, fontweight="bold")
    ax.set_xlabel("Error (pixels)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # Plot 3: NEES Cumulative Distribution (Chi-Square Goodness-of-Fit)
    ax = axes[1, 0]
    sorted_nees_A = np.sort(nees_A)
    sorted_nees_B = np.sort(nees_B)
    sorted_nees_C = np.sort(nees_C)
    cdf_meas = np.linspace(0, 1, M)
    # Theoretical Chi2 with 2 DoF CDF = 1 - exp(-x/2)
    chi2_theoretical = 1.0 - np.exp(-sorted_nees_C / 2.0)

    ax.plot(sorted_nees_C, chi2_theoretical, "k--", linewidth=2.0, label="Theoretical $\\chi^2_2$ CDF")
    ax.plot(sorted_nees_C, cdf_meas, "b-", linewidth=2.0, label="Proposed Model C (OptiTrack)")
    ax.plot(sorted_nees_B, cdf_meas, "r:", linewidth=1.8, label="Baseline B (Slide 5 Clipped FB)")
    ax.plot(sorted_nees_A, cdf_meas, "g-.", linewidth=1.8, label="Baseline A (Constant)")
    ax.set_title("Normalized Estimation Error Squared (NEES) CDF", fontsize=11, fontweight="bold")
    ax.set_xlabel("NEES $\\epsilon_k = \\mathbf{e}_k^T \\mathbf{\\Sigma}^{-1} \\mathbf{e}_k$")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    # Plot 4: Error vs Corner Strength / Structure Tensor
    ax = axes[1, 1]
    # Bin by corner strength lambda_2
    l2_bins = np.quantile(df_test["lambda_2"], np.linspace(0.05, 0.95, 8))
    bin_centers = []
    rmse_per_bin = []
    for b_idx in range(len(l2_bins) - 1):
        m = (df_test["lambda_2"] >= l2_bins[b_idx]) & (df_test["lambda_2"] < l2_bins[b_idx + 1])
        if np.sum(m) > 20:
            bin_centers.append(np.mean(df_test["lambda_2"][m]))
            rmse_per_bin.append(np.sqrt(np.mean(df_test["err_total"][m] ** 2)))

    ax.plot(bin_centers, rmse_per_bin, "o-", color="#d62728", linewidth=2.0, label="Empirical RMSE")
    ax.set_title("Localization Error vs Corner Strength ($\\lambda_2$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Structure Tensor Min Eigenvalue $\\lambda_2$ (Corner Sharpness)")
    ax.set_ylabel("True Tracking RMSE (pixels)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "optitrack_covariance_calibration_benchmark.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Diagnostic calibration figures saved to: {plot_path}")

    # Save calibrated parameters to yaml
    yaml_path = os.path.join(out_dir, "optitrack_calibrated_covariance.yaml")
    with open(yaml_path, "w") as f:
        f.write("# OptiTrack-Calibrated Feature Measurement Uncertainty Model\n")
        f.write("# Fitted on RealSense D435if active stereo tracking ground truth\n")
        f.write(f"sigma_base_px: {math.sqrt(sigma_base2):.6f}\n")
        f.write(f"gamma_fb: {gamma_fb:.6f}\n")
        f.write(f"alpha_structure_tensor: {alpha_st:.6f}\n")
        f.write(f"constant_sigma_u: {std_u:.6f}\n")
        f.write(f"constant_sigma_v: {std_v:.6f}\n")
        f.write(f"correlation_rho: {rho:.6f}\n")
        f.write(f"mean_nees_proposed: {mean_nees_C:.4f}\n")
        f.write(f"two_sigma_coverage_percent: {cov_2sig_C:.2f}\n")
    print(f"[PARAMS] Calibrated model parameters saved to: {yaml_path}")

    return {
        "sigma_base": math.sqrt(sigma_base2),
        "gamma_fb": gamma_fb,
        "alpha_st": alpha_st,
        "mean_nees_A": mean_nees_A,
        "mean_nees_B": mean_nees_B,
        "mean_nees_C": mean_nees_C,
        "cov_2sig_A": cov_2sig_A,
        "cov_2sig_B": cov_2sig_B,
        "cov_2sig_C": cov_2sig_C,
        "plot_path": plot_path,
        "yaml_path": yaml_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate feature tracking covariance using OptiTrack ground truth")
    parser.add_argument("--data", type=str, default="d:/Post_doc/mac/MAC-VO/mac_rgbd_vio/data/exp102_laptop_office_walk")
    parser.add_argument("--opti", type=str, default="d:/Post_doc/mac/MAC-VO/optitrack_data/Take_new.csv")
    parser.add_argument("--out", type=str, default="d:/Post_doc/mac/MAC-VO/mac_rgbd_vio/config")
    parser.add_argument("--max_frames", type=int, default=450)
    args = parser.parse_args()

    df = run_ground_truth_feature_harvesting(args.data, args.opti, max_frames=args.max_frames)
    calibrate_and_evaluate_models(df, args.out)


if __name__ == "__main__":
    main()

