"""Conservative visual/gyro stationary gating; missing tracks imply unknown motion."""
from dataclasses import dataclass
import numpy as np


@dataclass
class StationaryDecision:
    is_stationary: bool
    median_flow_px: float
    gyro_norm_rad_s: float

    @property
    def IS_STATIONARY(self) -> bool:
        return self.is_stationary

    @property
    def delta_p_flow(self) -> float:
        return self.median_flow_px


class ZUPTDetector:
    def __init__(self, flow_threshold=0.35, gyro_threshold=0.025,
                 min_tracks=20, confirmation_frames=3):
        self.flow_threshold, self.gyro_threshold = flow_threshold, gyro_threshold
        self.min_tracks, self.confirmation_frames = min_tracks, confirmation_frames
        self.streak = 0

    def update(self, previous_pixels, current_pixels, corrected_gyro):
        a, b = np.asarray(previous_pixels), np.asarray(current_pixels)
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("Flow arrays must have matching (N,2) shape")
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        flow = float(np.median(np.linalg.norm(b[valid] - a[valid], axis=1))) if valid.sum() >= self.min_tracks else float('inf')
        w = np.asarray(corrected_gyro, dtype=float).reshape(-1, 3)
        # Maximum, not norm of the mean: opposite rotations must not cancel.
        gyro = float(np.max(np.linalg.norm(w, axis=1))) if len(w) and np.isfinite(w).all() else float('inf')
        candidate = flow < self.flow_threshold and gyro < self.gyro_threshold
        self.streak = self.streak + 1 if candidate else 0
        return StationaryDecision(self.streak >= self.confirmation_frames, flow, gyro)
