"""
Sliding-Window Local Keyframe Factor Optimizer for MAC-RGBD-VIO
Formulates joint visual, IMU, and marginalization prior factor residuals
over a fixed-lag sliding window (N = 6 to 8 keyframes) on the SE(3) x R^3 manifold.
Executes on CPU via sparse Gauss-Newton / Levenberg-Marquardt with Schur-complement marginalization.

Target Venue: IEEE Robotics and Automation Letters (RA-L)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


def skew(v: np.ndarray) -> np.ndarray:
    """Skew symmetric matrix of 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) tangent space to SO(3) rotation matrix."""
    theta = np.linalg.norm(omega)
    if theta < 1e-7:
        k = skew(omega)
        return np.eye(3, dtype=np.float64) + k
    axis = omega / theta
    k = skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to so(3) vector [omega_x, omega_y, omega_z]."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-7:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) * 0.5
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) * scale


def condition_matrix(mat: np.ndarray, min_eig: float = 1.0, max_eig: float = 5e4) -> np.ndarray:
    """Conditions symmetric matrix eigenvalues to prevent numerical instability."""
    mat_sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat_sym)
    vals = np.clip(vals, min_eig, max_eig)
    return vecs @ np.diag(vals) @ vecs.T


@dataclass
class Keyframe:
    id: int
    frame_idx: int
    timestamp: float
    R: np.ndarray              # (3, 3) world-to-body/camera rotation
    p: np.ndarray              # (3,) world position
    v: np.ndarray              # (3,) world velocity
    pts_3d: np.ndarray         # (M, 3) 3D camera keypoints
    info_mats: np.ndarray      # (M, 3, 3) metric information matrices
    kp_2d: np.ndarray          # (M, 2) 2D pixel coordinates
    descriptors: Optional[np.ndarray] = None
    orig_R: Optional[np.ndarray] = None
    orig_p: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.orig_R is None:
            self.orig_R = self.R.copy()
        if self.orig_p is None:
            self.orig_p = self.p.copy()


@dataclass
class KeyframeIMUFactor:
    i: int                     # from keyframe index in window
    j: int                     # to keyframe index in window
    delta_R: np.ndarray        # (3, 3) preintegrated rotation
    delta_v: np.ndarray        # (3,) preintegrated velocity
    delta_p: np.ndarray        # (3,) preintegrated position
    dt: float                  # integration duration
    cov: np.ndarray            # (9, 9) preintegration covariance


@dataclass
class KeyframeVisualFactor:
    i: int                     # from keyframe index in window
    j: int                     # to keyframe index in window
    R_rel: np.ndarray          # (3, 3) relative rotation
    t_rel: np.ndarray          # (3,) relative translation
    info: np.ndarray           # (6, 6) metric Fisher information matrix


class KeyframeSelector:
    """
    Keyframe selector spawning new keyframes when:
    optical flow translation exceeds delta_t > 0.15 m OR
    angular rotation exceeds delta_theta > 12 degrees.
    """
    def __init__(self, trans_thresh: float = 0.15, rot_thresh_deg: float = 12.0):
        self.trans_thresh = trans_thresh
        self.rot_thresh_rad = np.radians(rot_thresh_deg)
        self.last_kf_R: Optional[np.ndarray] = None
        self.last_kf_p: Optional[np.ndarray] = None

    def is_keyframe(self, cur_R: np.ndarray, cur_p: np.ndarray) -> bool:
        if self.last_kf_R is None or self.last_kf_p is None:
            return True
        delta_p = np.linalg.norm(cur_p - self.last_kf_p)
        delta_R = self.last_kf_R.T @ cur_R
        delta_theta = np.linalg.norm(so3_log(delta_R))
        return (delta_p >= self.trans_thresh) or (delta_theta >= self.rot_thresh_rad)

    def register_keyframe(self, cur_R: np.ndarray, cur_p: np.ndarray):
        self.last_kf_R = cur_R.copy()
        self.last_kf_p = cur_p.copy()


class SlidingWindowSmoother:
    """
    Fixed-Lag Sliding Window Local Keyframe Optimizer.
    Maintains a sliding window of N = 6 to 8 keyframes.
    Jointly minimizes visual, IMU, and marginalization prior residuals:
        min_X sum_{vis} ||r_vis||^2_{Omega_3D} + sum_{imu} ||r_IMU||^2_{Sigma_IMU^-1} + ||r_prior||^2_{Sigma_prior^-1}
    Performs Schur-complement marginalization on the oldest keyframe.
    """
    def __init__(
        self,
        window_size: int = 7,
        gravity: np.ndarray = np.array([0.0, 9.80665, 0.0]),
        max_iters: int = 4,
        damping: float = 1e-3
    ):
        self.window_size = window_size
        self.gravity = gravity.astype(np.float64)
        self.max_iters = max_iters
        self.damping = damping

        self.window: List[Keyframe] = []
        self.all_keyframes: List[Keyframe] = []
        self.imu_factors: List[KeyframeIMUFactor] = []
        self.vis_factors: List[KeyframeVisualFactor] = []

        # 9x9 Marginalization prior storage (Schur complement on oldest state)
        # H_prior * delta_x_0 = b_prior
        self.prior_H: Optional[np.ndarray] = None
        self.prior_b: Optional[np.ndarray] = None

    def add_keyframe(
        self,
        kf: Keyframe,
        imu_factor: Optional[KeyframeIMUFactor] = None,
        vis_factor: Optional[KeyframeVisualFactor] = None
    ) -> List[Keyframe]:
        """
        Adds a new keyframe to the sliding window.
        Marginalizes the oldest keyframe if window exceeds window_size.
        Optimizes window states and returns the current window.
        """
        # Marginalize oldest keyframe if window is already full before adding new one
        if len(self.window) >= self.window_size:
            self._marginalize_oldest()

        # Update factor indices to point to the new keyframe
        new_j = len(self.window)
        new_i = max(0, new_j - 1)

        if imu_factor is not None:
            imu_factor.i = new_i
            imu_factor.j = new_j
            self.imu_factors.append(imu_factor)

        if vis_factor is not None:
            vis_factor.i = new_i
            vis_factor.j = new_j
            # Condition visual information matrix
            vis_factor.info = condition_matrix(vis_factor.info, min_eig=1.0, max_eig=5e4)
            self.vis_factors.append(vis_factor)

        self.window.append(kf)
        self.all_keyframes.append(kf)

        # Joint window optimization on CPU
        if len(self.window) >= 2:
            self.optimize()

        return self.window

    def optimize(self):
        """
        Solves joint visual-inertial-prior normal equations:
            (H + lambda * I) * delta_x = b
        over the sliding window state vector:
            X = [delta_theta_0, delta_p_0, delta_v_0, ..., delta_theta_{W-1}, delta_p_{W-1}, delta_v_{W-1}]
        where each keyframe state has 9 DoF (3 rot, 3 pos, 3 vel).
        """
        W = len(self.window)
        if W < 2:
            return

        dim = 9 * W

        for _ in range(self.max_iters):
            H = np.zeros((dim, dim), dtype=np.float64)
            b = np.zeros(dim, dtype=np.float64)

            # 1. Add Prior Factor on oldest keyframe in window
            if self.prior_H is not None and self.prior_b is not None:
                H[0:9, 0:9] += self.prior_H
                b[0:9] += self.prior_b
            else:
                # Gauge anchor for Keyframe 0
                H[0:9, 0:9] += 1e5 * np.eye(9, dtype=np.float64)

            # 2. Add Visual Factors
            for vf in self.vis_factors:
                i, j = vf.i, vf.j
                if i >= W or j >= W:
                    continue

                kf_i = self.window[i]
                kf_j = self.window[j]

                R_ij_meas = vf.R_rel
                t_ij_meas = vf.t_rel

                R_ij_est = kf_i.R.T @ kf_j.R
                r_R = so3_log(R_ij_meas.T @ R_ij_est)
                r_t = kf_i.R.T @ (kf_j.p - kf_i.p) - t_ij_meas

                r_vis = np.concatenate([r_R, r_t])
                W_vis = vf.info

                J_i = np.zeros((6, 9), dtype=np.float64)
                J_j = np.zeros((6, 9), dtype=np.float64)

                J_i[0:3, 0:3] = -kf_j.R.T @ kf_i.R
                J_j[0:3, 0:3] = np.eye(3, dtype=np.float64)

                J_i[3:6, 0:3] = skew(kf_i.R.T @ (kf_j.p - kf_i.p))
                J_i[3:6, 3:6] = -kf_i.R.T
                J_j[3:6, 3:6] = kf_i.R.T

                idx_i = 9 * i
                idx_j = 9 * j

                H[idx_i:idx_i+9, idx_i:idx_i+9] += J_i.T @ W_vis @ J_i
                H[idx_i:idx_i+9, idx_j:idx_j+9] += J_i.T @ W_vis @ J_j
                H[idx_j:idx_j+9, idx_i:idx_i+9] += J_j.T @ W_vis @ J_i
                H[idx_j:idx_j+9, idx_j:idx_j+9] += J_j.T @ W_vis @ J_j

                b[idx_i:idx_i+9] -= J_i.T @ W_vis @ r_vis
                b[idx_j:idx_j+9] -= J_j.T @ W_vis @ r_vis

            # 3. Add IMU Preintegration Factors
            for imuf in self.imu_factors:
                i, j = imuf.i, imuf.j
                if i >= W or j >= W:
                    continue

                kf_i = self.window[i]
                kf_j = self.window[j]
                dt = max(imuf.dt, 1e-4)

                R_ij_est = kf_i.R.T @ kf_j.R
                r_R = so3_log(imuf.delta_R.T @ R_ij_est)
                r_v = kf_i.R.T @ (kf_j.v - kf_i.v - self.gravity * dt) - imuf.delta_v
                r_p = kf_i.R.T @ (kf_j.p - kf_i.p - kf_i.v * dt - 0.5 * self.gravity * (dt ** 2)) - imuf.delta_p

                r_imu = np.concatenate([r_R, r_v, r_p])

                cov_reg = imuf.cov + 1e-3 * np.eye(9, dtype=np.float64)
                W_imu = condition_matrix(np.linalg.inv(cov_reg), min_eig=1.0, max_eig=1e4)

                J_i = np.zeros((9, 9), dtype=np.float64)
                J_j = np.zeros((9, 9), dtype=np.float64)

                J_i[0:3, 0:3] = -kf_j.R.T @ kf_i.R
                J_j[0:3, 0:3] = np.eye(3, dtype=np.float64)

                J_i[3:6, 0:3] = skew(kf_i.R.T @ (kf_j.v - kf_i.v - self.gravity * dt))
                J_i[3:6, 6:9] = -kf_i.R.T
                J_j[3:6, 6:9] = kf_i.R.T

                J_i[6:9, 0:3] = skew(kf_i.R.T @ (kf_j.p - kf_i.p - kf_i.v * dt - 0.5 * self.gravity * (dt ** 2)))
                J_i[6:9, 3:6] = -kf_i.R.T
                J_i[6:9, 6:9] = -kf_i.R.T * dt
                J_j[6:9, 3:6] = kf_i.R.T

                idx_i = 9 * i
                idx_j = 9 * j

                H[idx_i:idx_i+9, idx_i:idx_i+9] += J_i.T @ W_imu @ J_i
                H[idx_i:idx_i+9, idx_j:idx_j+9] += J_i.T @ W_imu @ J_j
                H[idx_j:idx_j+9, idx_i:idx_i+9] += J_j.T @ W_imu @ J_i
                H[idx_j:idx_j+9, idx_j:idx_j+9] += J_j.T @ W_imu @ J_j

                b[idx_i:idx_i+9] -= J_i.T @ W_imu @ r_imu
                b[idx_j:idx_j+9] -= J_j.T @ W_imu @ r_imu

            # Levenberg-Marquardt damping
            H_damped = H + self.damping * np.eye(dim, dtype=np.float64)

            try:
                delta_x = np.linalg.solve(H_damped, b)
            except np.linalg.LinAlgError:
                break

            if np.any(np.isnan(delta_x)) or np.linalg.norm(delta_x) < 1e-6:
                break

            # Clip delta_x step to prevent large deviations
            delta_x = np.clip(delta_x, -0.2, 0.2)

            # Update states on manifold
            for k in range(W):
                d_theta = delta_x[9*k : 9*k+3]
                d_p     = delta_x[9*k+3 : 9*k+6]
                d_v     = delta_x[9*k+6 : 9*k+9]

                self.window[k].R = self.window[k].R @ so3_exp(d_theta)
                self.window[k].p = self.window[k].p + d_p
                self.window[k].v = self.window[k].v + d_v

    def _marginalize_oldest(self):
        """
        Schur-complement marginalization of the oldest keyframe (state 0).
        Evaluates the sub-system connecting state 0 and state 1:
            [ H_00  H_01 ] [ delta_x_0 ] = [ b_0 ]
            [ H_10  H_11 ] [ delta_x_1 ]   [ b_1 ]
        Schur Complement:
            H_prior_new = H_11 - H_10 * (H_00 + eps * I)^-1 * H_01
            b_prior_new = b_1  - H_10 * (H_00 + eps * I)^-1 * b_0
        Yields a clean, non-compounding 9x9 prior on state 1 (which becomes state 0).
        """
        if len(self.window) < 2:
            return

        kf0, kf1 = self.window[0], self.window[1]

        # 18x18 linear system for state 0 and state 1
        H18 = np.zeros((18, 18), dtype=np.float64)
        b18 = np.zeros(18, dtype=np.float64)

        # Prior on state 0
        if self.prior_H is not None and self.prior_b is not None:
            H18[0:9, 0:9] += self.prior_H
            b18[0:9] += self.prior_b
        else:
            H18[0:9, 0:9] += 1e4 * np.eye(9, dtype=np.float64)

        # Factors between 0 and 1
        for vf in self.vis_factors:
            if vf.i == 0 and vf.j == 1:
                R_ij_meas, t_ij_meas = vf.R_rel, vf.t_rel
                R_ij_est = kf0.R.T @ kf1.R
                r_R = so3_log(R_ij_meas.T @ R_ij_est)
                r_t = kf0.R.T @ (kf1.p - kf0.p) - t_ij_meas
                r_vis = np.concatenate([r_R, r_t])
                W_vis = vf.info

                J_0 = np.zeros((6, 9), dtype=np.float64)
                J_1 = np.zeros((6, 9), dtype=np.float64)
                J_0[0:3, 0:3] = -kf1.R.T @ kf0.R
                J_1[0:3, 0:3] = np.eye(3, dtype=np.float64)
                J_0[3:6, 0:3] = skew(kf0.R.T @ (kf1.p - kf0.p))
                J_0[3:6, 3:6] = -kf0.R.T
                J_1[3:6, 3:6] = kf0.R.T

                H18[0:9, 0:9] += J_0.T @ W_vis @ J_0
                H18[0:9, 9:18] += J_0.T @ W_vis @ J_1
                H18[9:18, 0:9] += J_1.T @ W_vis @ J_0
                H18[9:18, 9:18] += J_1.T @ W_vis @ J_1

                b18[0:9] -= J_0.T @ W_vis @ r_vis
                b18[9:18] -= J_1.T @ W_vis @ r_vis

        for imuf in self.imu_factors:
            if imuf.i == 0 and imuf.j == 1:
                dt = max(imuf.dt, 1e-4)
                R_ij_est = kf0.R.T @ kf1.R
                r_R = so3_log(imuf.delta_R.T @ R_ij_est)
                r_v = kf0.R.T @ (kf1.v - kf0.v - self.gravity * dt) - imuf.delta_v
                r_p = kf0.R.T @ (kf1.p - kf0.p - kf0.v * dt - 0.5 * self.gravity * (dt ** 2)) - imuf.delta_p
                r_imu = np.concatenate([r_R, r_v, r_p])

                cov_reg = imuf.cov + 1e-3 * np.eye(9, dtype=np.float64)
                W_imu = condition_matrix(np.linalg.inv(cov_reg), min_eig=1.0, max_eig=1e4)

                J_0 = np.zeros((9, 9), dtype=np.float64)
                J_1 = np.zeros((9, 9), dtype=np.float64)
                J_0[0:3, 0:3] = -kf1.R.T @ kf0.R
                J_1[0:3, 0:3] = np.eye(3, dtype=np.float64)
                J_0[3:6, 0:3] = skew(kf0.R.T @ (kf1.v - kf0.v - self.gravity * dt))
                J_0[3:6, 6:9] = -kf0.R.T
                J_1[3:6, 6:9] = kf0.R.T
                J_0[6:9, 0:3] = skew(kf0.R.T @ (kf1.p - kf0.p - kf0.v * dt - 0.5 * self.gravity * (dt ** 2)))
                J_0[6:9, 3:6] = -kf0.R.T
                J_0[6:9, 6:9] = -kf0.R.T * dt
                J_1[6:9, 3:6] = kf0.R.T

                H18[0:9, 0:9] += J_0.T @ W_imu @ J_0
                H18[0:9, 9:18] += J_0.T @ W_imu @ J_1
                H18[9:18, 0:9] += J_1.T @ W_imu @ J_0
                H18[9:18, 9:18] += J_1.T @ W_imu @ J_1

                b18[0:9] -= J_0.T @ W_imu @ r_imu
                b18[9:18] -= J_1.T @ W_imu @ r_imu

        # Schur Complement elimination:
        H_00 = H18[0:9, 0:9]
        H_01 = H18[0:9, 9:18]
        H_10 = H18[9:18, 0:9]
        H_11 = H18[9:18, 9:18]
        b_0  = b18[0:9]
        b_1  = b18[9:18]

        H_00_inv = np.linalg.inv(H_00 + 1e-4 * np.eye(9, dtype=np.float64))
        new_prior_H = H_11 - H_10 @ H_00_inv @ H_01
        new_prior_b = b_1 - H_10 @ H_00_inv @ b_0

        self.prior_H = condition_matrix(new_prior_H, min_eig=1.0, max_eig=5e4)
        self.prior_b = np.clip(new_prior_b, -100.0, 100.0)

        # Pop keyframe 0 from window
        self.window.pop(0)

        # Shift indices of remaining factors by -1
        new_imu_factors = []
        for imuf in self.imu_factors:
            if imuf.i > 0 and imuf.j > 0:
                imuf.i -= 1
                imuf.j -= 1
                new_imu_factors.append(imuf)
        self.imu_factors = new_imu_factors

        new_vis_factors = []
        for vf in self.vis_factors:
            if vf.i > 0 and vf.j > 0:
                vf.i -= 1
                vf.j -= 1
                new_vis_factors.append(vf)
        self.vis_factors = new_vis_factors
