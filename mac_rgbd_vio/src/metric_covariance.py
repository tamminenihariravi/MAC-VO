"""
Metric Covariance Engine for RGB-D Visual Odometry (MAC-RGBD)
Formulates the full 3x3 spatial metric covariance matrix from 2D optical flow
uncertainty and active depth sensor physical noise characteristics.

Target Journal: IEEE Robotics and Automation Letters (RA-L)
"""

import numpy as np


class RGBDMetricCovarianceEngine:
    def __init__(
        self,
        fx: float = 384.0,
        fy: float = 384.0,
        cx: float = 320.0,
        cy: float = 240.0,
        alpha_0: float = 0.002,     # Baseline depth noise (meters)
        alpha_1: float = 0.0018,    # Quadratic axial depth noise coefficient
        beta: float = 0.04,         # Depth edge penalty coefficient
        max_depth: float = 5.0,     # Maximum valid depth (meters)
        min_depth: float = 0.2,     # Minimum valid depth (meters)
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.beta = beta
        self.max_depth = max_depth
        self.min_depth = min_depth

    def compute_depth_uncertainty(
        self, 
        d: float, 
        depth_grad_norm: float = 0.0
    ) -> float:
        """
        Physical noise model for active stereo / structured-light RGB-D sensors.
        sigma_d = alpha_0 + alpha_1 * d^2 + beta * ||grad(d)||
        """
        if d < self.min_depth or d > self.max_depth or np.isnan(d):
            return 1e6  # Infinite uncertainty for invalid depth
        
        sigma_d = self.alpha_0 + self.alpha_1 * (d ** 2) + self.beta * depth_grad_norm
        return float(sigma_d)

    def backproject_point(self, u: float, v: float, d: float) -> np.ndarray:
        """
        Backprojects 2D pixel (u, v) with depth d to 3D camera coordinate [X, Y, Z]^T.
        """
        x = (u - self.cx) * d / self.fx
        y = (v - self.cy) * d / self.fy
        z = d
        return np.array([x, y, z], dtype=np.float64)

    def compute_covariance_3d(
        self,
        u: float,
        v: float,
        d: float,
        sigma_u: float = 1.0,
        sigma_v: float = 1.0,
        depth_grad_norm: float = 0.0,
    ) -> np.ndarray:
        """
        Analytically computes the full 3x3 spatial metric covariance matrix:
        Sigma_3D = J * diag(sigma_u^2, sigma_v^2, sigma_d^2) * J^T
        """
        sigma_d = self.compute_depth_uncertainty(d, depth_grad_norm)
        
        # Intermediate normalized coordinates
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        
        var_u = sigma_u ** 2
        var_v = sigma_v ** 2
        var_d = sigma_d ** 2
        
        # Analytical 3x3 Metric Covariance formulation:
        cov_3d = np.zeros((3, 3), dtype=np.float64)
        
        # Diagonal terms:
        cov_3d[0, 0] = ((d / self.fx) ** 2) * var_u + (xn ** 2) * var_d
        cov_3d[1, 1] = ((d / self.fy) ** 2) * var_v + (yn ** 2) * var_d
        cov_3d[2, 2] = var_d
        
        # Off-diagonal cross-axis terms:
        cov_3d[0, 1] = xn * yn * var_d
        cov_3d[1, 0] = cov_3d[0, 1]
        
        cov_3d[0, 2] = xn * var_d
        cov_3d[2, 0] = cov_3d[0, 2]
        
        cov_3d[1, 2] = yn * var_d
        cov_3d[2, 1] = cov_3d[1, 2]
        
        return cov_3d

    def batch_compute(
        self,
        pts_2d: np.ndarray,      # (N, 2) [u, v]
        depths: np.ndarray,      # (N,) in meters
        sigmas_flow: np.ndarray, # (N, 2) [sigma_u, sigma_v]
        grad_norms: np.ndarray = None,
    ):
        """
        Batch processing of N points for high-frequency real-time execution.
        Returns:
            pts_3d: (N, 3)
            covs_3d: (N, 3, 3)
            info_mats_3d: (N, 3, 3) [Inverted metric covariance matrices]
        """
        N = pts_2d.shape[0]
        if grad_norms is None:
            grad_norms = np.zeros(N, dtype=np.float64)

        u = pts_2d[:, 0]
        v = pts_2d[:, 1]
        d = depths

        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy

        pts_3d = np.stack([xn * d, yn * d, d], axis=1)

        # Vectorized depth uncertainty
        sigma_d = self.alpha_0 + self.alpha_1 * (d ** 2) + self.beta * grad_norms
        invalid_mask = (d < self.min_depth) | (d > self.max_depth) | np.isnan(d)
        sigma_d[invalid_mask] = 1e6

        var_u = sigmas_flow[:, 0] ** 2
        var_v = sigmas_flow[:, 1] ** 2
        var_d = sigma_d ** 2

        covs_3d = np.zeros((N, 3, 3), dtype=np.float64)
        covs_3d[:, 0, 0] = ((d / self.fx) ** 2) * var_u + (xn ** 2) * var_d
        covs_3d[:, 1, 1] = ((d / self.fy) ** 2) * var_v + (yn ** 2) * var_d
        covs_3d[:, 2, 2] = var_d

        covs_3d[:, 0, 1] = xn * yn * var_d
        covs_3d[:, 1, 0] = covs_3d[:, 0, 1]

        covs_3d[:, 0, 2] = xn * var_d
        covs_3d[:, 2, 0] = covs_3d[:, 0, 2]

        covs_3d[:, 1, 2] = yn * var_d
        covs_3d[:, 2, 1] = covs_3d[:, 1, 2]

        # Regularized inversion for information matrices (numerically robust)
        info_mats_3d = np.zeros((N, 3, 3), dtype=np.float64)
        reg = 1e-7 * np.eye(3)
        for i in range(N):
            if invalid_mask[i]:
                info_mats_3d[i] = np.zeros((3, 3), dtype=np.float64)
            else:
                try:
                    info_mats_3d[i] = np.linalg.inv(covs_3d[i] + reg)
                except np.linalg.LinAlgError:
                    info_mats_3d[i] = np.linalg.pinv(covs_3d[i])

        return pts_3d, covs_3d, info_mats_3d
