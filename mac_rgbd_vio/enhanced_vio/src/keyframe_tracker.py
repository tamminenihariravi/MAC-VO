"""Keyframe-referenced RGB-D tracking with frame-local flow for ZUPT."""
from dataclasses import dataclass
import cv2
import numpy as np
from .calibrated_covariance import OptiTrackCalibratedCovarianceEngine


@dataclass
class Tracks:
    reference_pixels: np.ndarray
    previous_pixels: np.ndarray
    current_pixels: np.ndarray
    reference_points: np.ndarray
    current_points: np.ndarray
    reference_covariance: np.ndarray
    current_covariance: np.ndarray
    disparity: float


class KeyframeTracker:
    def __init__(self, intrinsics, covariance_config=None, max_features=300):
        self.engine = OptiTrackCalibratedCovarianceEngine(**intrinsics, **(covariance_config or {}))
        self.max_features = max_features
        self.lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.reference_gray = None
        self.reference_count = 0
        self._eigen_gray = None
        self._gradients = None
        self.detection_count = 0
        self.detected_keyframes = 0
        self.reused_keyframes = 0

    def point_eigenvalues(self, gray, x, y):
        if gray is not self._eigen_gray:
            # Same 3x3 Sobel and 7x7 tensor as cornerMinEigenVal, evaluated
            # only at tracked points instead of every one of 307,200 pixels.
            scale = 1. / (4 * 255)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=scale)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=scale)
            self._gradients = [cv2.copyMakeBorder(g, 3, 3, 3, 3, cv2.BORDER_REFLECT_101) for g in (gx, gy)]
            self._eigen_gray = gray
        oy, ox = np.mgrid[-3:4, -3:4]
        yy, xx = y[:, None] + oy.ravel() + 3, x[:, None] + ox.ravel() + 3
        gx, gy = (g[yy, xx] for g in self._gradients)
        a, c, b = np.mean(gx * gx, axis=1), np.mean(gy * gy, axis=1), np.mean(gx * gy, axis=1)
        return .5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b ** 2))

    def unproject(self, pixels, depth, fb=None, gray=None):
        h, w = depth.shape
        xy = np.rint(np.nan_to_num(pixels, nan=-1, posinf=-1, neginf=-1)).astype(int)
        inside = np.isfinite(pixels).all(axis=1) & (xy[:, 0] >= 1) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 1) & (xy[:, 1] < h - 1)
        x, y = np.clip(xy[:, 0], 1, w - 2), np.clip(xy[:, 1], 1, h - 2)
        d = depth[y, x]
        grad = np.hypot(depth[y, x + 1] - depth[y, x - 1], depth[y + 1, x] - depth[y - 1, x]) * 0.5
        valid = inside & np.isfinite(d) & (d > self.engine.min_depth) & (d < self.engine.max_depth) & np.isfinite(grad) & (grad < 0.10)
        lam = np.ones(len(pixels))
        if gray is not None:
            lam = self.point_eigenvalues(gray, x, y)
        P, C, _ = self.engine.batch_compute(pixels, d, np.zeros_like(pixels) if fb is None else fb, lambda_2=lam, grad_norms=np.nan_to_num(grad), return_information=False)
        return P, C, valid

    def set_reference(self, gray, depth, timestamp):
        self.reference_gray = gray.copy()
        self.timestamp = timestamp
        points = cv2.goodFeaturesToTrack(gray, self.max_features, 0.01, 9, blockSize=7)
        points = np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)
        P, C, good = self.unproject(points, depth, gray=gray)
        self.points = points[good]
        self.P, self.C = P[good], C[good]
        self.ids = np.arange(len(self.points))
        self.previous_pixels = self.points.copy()
        self.reference_count = len(self.points)
        self.detection_count = self.reference_count
        self.detected_keyframes += 1

    def advance_reference(self, gray, depth, timestamp, tracks, inliers):
        mask = np.asarray(inliers, dtype=bool)
        if mask.sum() < max(20, .65 * self.detection_count):
            self.set_reference(gray, depth, timestamp)
            return
        # A keyframe can retain well-supported features already measured in this
        # image. Redetect only when their support decays, instead of every step.
        self.reference_gray = gray.copy()
        self.timestamp = timestamp
        self.points = tracks.current_pixels[mask].copy()
        self.P = tracks.current_points[mask].copy()
        self.C = tracks.current_covariance[mask].copy()
        self.ids = np.arange(len(self.points))
        self.previous_pixels = self.points.copy()
        self.reference_count = len(self.points)
        self.reused_keyframes += 1

    def track(self, gray, depth):
        ids = self.ids
        if len(ids) == 0:
            empty = np.empty((0, 2))
            return Tracks(empty, empty, empty, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 3, 3)), float('inf'))
        p0 = self.points[ids].reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.reference_gray, gray, p0, self.previous_pixels.reshape(-1, 1, 2).copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        back, sb, _ = cv2.calcOpticalFlowPyrLK(gray, self.reference_gray, p1, None, **self.lk)
        p1, back = p1.reshape(-1, 2), back.reshape(-1, 2)
        fb = back - p0.reshape(-1, 2)
        P, C, vd = self.unproject(p1, depth, fb, gray)
        valid = st.ravel().astype(bool) & sb.ravel().astype(bool) & vd & np.isfinite(p1).all(axis=1) & (np.linalg.norm(fb, axis=1) < 1.5)
        selected = ids[valid]
        p_ref, p_curr = self.points[selected], p1[valid]
        disparity = float(np.median(np.linalg.norm(p_curr - p_ref, axis=1))) if len(selected) else float('inf')
        result = Tracks(p_ref, self.previous_pixels[valid].copy(), p_curr, self.P[selected], P[valid], self.C[selected], C[valid], disparity)
        self.ids, self.previous_pixels = selected, p_curr.copy()
        return result

    def should_insert(self, disparity, rotation_angle, inliers):
        if isinstance(inliers, float) and inliers <= 1.0:
            inlier_cond = inliers < 0.65
        else:
            inlier_cond = inliers < 0.65 * self.reference_count
        return disparity > 3.5 or rotation_angle > np.deg2rad(2.5) or inlier_cond
