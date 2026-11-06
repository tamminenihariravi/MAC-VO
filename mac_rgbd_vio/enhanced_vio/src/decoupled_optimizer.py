"""Robust visual registration, tangent-space rotation fusion, conditional translation.

Convention: P_reference = R_reference_current @ P_current + t_reference_current.
For fixed R and IRLS weights w_i, the exact conditional minimizer is
  t = (sum w_i Omega_i)^-1 sum w_i Omega_i (P_reference_i - R P_current_i).
Huber weights depend on t, so the robust problem requires repeated conditional
solves; it has no single closed-form solution for arbitrary rho.
"""
from dataclasses import dataclass
import numpy as np
from .lie import so3_exp, so3_log, skew_batch


@dataclass
class PoseEstimate:
    rotation: np.ndarray
    translation: np.ndarray
    success: bool
    inliers: np.ndarray
    reason: str = ''


class DecoupledOptimizer:
    def __init__(self, visual_weight=0.75, max_iters=10, huber_delta=2.5,
                 inlier_threshold_m=0.06, min_inliers=12):
        if not 0 <= visual_weight <= 1:
            raise ValueError("Visual weight must be in [0,1]")
        self.visual_weight, self.max_iters = visual_weight, max_iters
        self.huber_delta, self.threshold, self.min_inliers = huber_delta, inlier_threshold_m, min_inliers

    @staticmethod
    def information(cov_reference, cov_current, rotation):
        # Residuals live in reference coordinates. Rotate CURRENT covariance.
        C = cov_reference + rotation[None] @ cov_current @ rotation.T[None]
        L = np.linalg.cholesky(C + 1e-9 * np.eye(3))
        invL = np.linalg.solve(L, np.broadcast_to(np.eye(3), L.shape))
        return np.swapaxes(invL, 1, 2) @ invL

    def weights(self, residual, information):
        s = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', residual, information, residual), 0))
        return np.minimum(1, self.huber_delta / np.maximum(s, 1e-12))

    @staticmethod
    def rigid_fit(a, b):
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        U, s, Vt = np.linalg.svd((b - bc).T @ (a - ac))
        if s[1] < 1e-10:
            raise ValueError("Collinear or collapsed point geometry")
        R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        return R, ac - R @ bc

    def solve_translation(self, reference, current, rotation, information):
        d = reference - current @ rotation.T
        t = np.median(d, axis=0)
        for _ in range(20):
            w = self.weights(d - t, information)
            W = w[:, None, None] * information
            candidate = np.linalg.solve(W.sum(axis=0), np.einsum('nij,nj->i', W, d))
            change = np.linalg.norm(candidate - t)
            t = candidate
            if change < 1e-9:
                break
        return t

    def consensus(self, a, b):
        rng = np.random.default_rng(17)
        R, t = self.rigid_fit(a, b)
        best = np.sum((a - b @ R.T - t) ** 2, axis=1) < self.threshold ** 2
        if best.all():
            return best
        required, draws = 64, 0
        while draws < min(64, required):
            count = min(8, 64 - draws)
            ids = np.array([rng.choice(len(a), 3, replace=False) for _ in range(count)])
            aa, bb = a[ids], b[ids]
            ac, bc = aa.mean(axis=1), bb.mean(axis=1)
            H = np.swapaxes(bb - bc[:, None], 1, 2) @ (aa - ac[:, None])
            U, s, Vt = np.linalg.svd(H)
            V = np.swapaxes(Vt, 1, 2).copy()
            V[:, :, 2] *= np.linalg.det(V @ np.swapaxes(U, 1, 2))[:, None]
            R = V @ np.swapaxes(U, 1, 2)
            t = ac - (R @ bc[:, :, None])[:, :, 0]
            residual = a[None] - b[None] @ np.swapaxes(R, 1, 2) - t[:, None]
            masks = np.sum(residual ** 2, axis=2) < self.threshold ** 2
            masks[s[:, 1] < 1e-10] = False
            index = int(np.argmax(masks.sum(axis=1)))
            if masks[index].sum() > best.sum():
                best = masks[index]
            draws += count
            if best.all():
                break
            ratio = float(best.mean())
            if ratio > 0:
                required = min(required, max(8, int(np.ceil(np.log(.001) / np.log1p(-ratio ** 3)))))
        return best

    def solve(self, reference, current, cov_reference, cov_current, imu_rotation=None):
        a, b = np.asarray(reference, float), np.asarray(current, float)
        ca, cb = np.asarray(cov_reference, float), np.asarray(cov_current, float)
        n = len(a)
        fail = lambda why: PoseEstimate(np.eye(3), np.zeros(3), False, np.zeros(n, bool), why)
        if a.shape != (n, 3) or b.shape != a.shape or ca.shape != (n, 3, 3) or cb.shape != ca.shape:
            raise ValueError("Invalid correspondence/covariance shapes")
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(ca).all(axis=(1, 2)) & np.isfinite(cb).all(axis=(1, 2))
        ids = np.flatnonzero(valid)
        if len(ids) < self.min_inliers:
            return fail('insufficient correspondences')
        av, bv = a[ids], b[ids]
        try:
            # Deterministic RANSAC keeps mismatches from corrupting the visual rotation.
            best = self.consensus(av, bv)
            ids = ids[best]
            if len(ids) < self.min_inliers:
                return fail('insufficient geometric inliers')
            av, bv, ca, cb = a[ids], b[ids], ca[ids], cb[ids]
            R, t = self.rigid_fit(av, bv)
            for _ in range(self.max_iters):
                rotated = bv @ R.T
                residual = rotated + t - av
                info = self.information(ca, cb, R)
                W = self.weights(residual, info)[:, None, None] * info
                J = np.concatenate((-skew_batch(rotated), np.broadcast_to(np.eye(3), (len(ids), 3, 3))), axis=2)
                JTW = np.swapaxes(J, 1, 2) @ W
                H = (JTW @ J).sum(axis=0)
                if np.linalg.cond(H) > 1e12:
                    return fail('ill-conditioned visual registration')
                step = np.linalg.solve(H + 1e-9 * np.eye(6), -np.einsum('nij,nj->i', JTW, residual))
                R = so3_exp(step[:3]) @ R
                t += step[3:]
                if np.linalg.norm(step) < 1e-7:
                    break
            if imu_rotation is not None:
                # Stage 1 (Rotation): Tangent-space SO(3) fusion
                # Fused R* = Exp(0.75 * omega_vis + 0.25 * omega_imu_debiased)
                omega_vis = so3_log(R)
                omega_imu = so3_log(imu_rotation)
                fused_omega = self.visual_weight * omega_vis + (1 - self.visual_weight) * omega_imu
                R = so3_exp(fused_omega)
            # Stage 2 (Translation conditioned on R*):
            # With R* fixed, solve t* = argmin_t sum_i rho(|| P_prev_i - (R* P_curr_i + t) ||^2_{Omega_i})
            info = self.information(ca, cb, R)
            t = self.solve_translation(av, bv, R, info)
            inliers = np.zeros(n, bool)
            inliers[ids] = np.linalg.norm(av - bv @ R.T - t, axis=1) < self.threshold
            if inliers.sum() < self.min_inliers or not np.isfinite(t).all():
                return fail('fused pose inconsistent with correspondences')
            return PoseEstimate(R, t, True, inliers)
        except (np.linalg.LinAlgError, ValueError) as exc:
            return fail(str(exc))

    def solve_relative_pose(self, pts_prev, pts_curr, cov_prev=None, cov_curr=None,
                            info_mats=None, init_R=None, imu_rotation=None):
        """Convenience alias conforming to relative pose solver signature."""
        target_imu = imu_rotation if imu_rotation is not None else init_R
        n = len(pts_prev)
        if cov_prev is None:
            if info_mats is not None:
                cov_prev = np.linalg.pinv(info_mats)
            else:
                cov_prev = np.broadcast_to(np.eye(3) * 1e-4, (n, 3, 3)).copy()
        if cov_curr is None:
            cov_curr = cov_prev.copy()
        est = self.solve(pts_prev, pts_curr, cov_prev, cov_curr, target_imu)
        return est.rotation, est.translation, est.success
