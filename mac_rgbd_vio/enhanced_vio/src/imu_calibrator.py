"""Startup gyro calibration and time-constant-based stationary bias tracking."""
from collections import deque
import numpy as np


class IMUCalibrator:
    def __init__(self, duration=2.0, min_samples=400, max_std=0.01,
                 online_time_constant=60.0):
        self.duration, self.min_samples, self.max_std = duration, min_samples, max_std
        self.online_time_constant = online_time_constant
        self.samples = deque()
        self.bias = np.zeros(3)
        self.std = np.full(3, np.inf)
        self.ready = False
        self.last_timestamp = None
        self.rejected_windows = 0

    @property
    def b_g(self):
        """Static or currently tracked gyro bias vector (rad/s)."""
        return self.bias

    def debias(self, raw_gyro):
        """Debias raw gyro: omega_corr = omega_raw - b_g."""
        return self.correct(raw_gyro)

    def add_startup(self, timestamp, raw_gyro, visual_stationary=True):
        w = np.asarray(raw_gyro, dtype=float)
        if w.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(timestamp):
            raise ValueError("Invalid gyro sample")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Gyro timestamps must be strictly increasing")
        self.last_timestamp = timestamp
        if self.ready:
            return True
        if not visual_stationary:
            self.samples.clear()
            return False
        self.samples.append((timestamp, w.copy()))
        if len(self.samples) < self.min_samples or timestamp - self.samples[0][0] < self.duration:
            return False
        a = np.array([s[1] for s in self.samples])
        self.std = a.std(axis=0, ddof=1)
        mean = a.mean(axis=0)
        # A constant rapid rotation also has low variance; never calibrate it away.
        if np.max(self.std) >= self.max_std or np.linalg.norm(mean) >= 0.025:
            self.rejected_windows += 1
            self.samples.clear()
            return False
        self.bias = mean
        self.ready = True
        return True

    def correct(self, raw_gyro):
        if not self.ready:
            raise RuntimeError("Gyro startup calibration is not complete")
        return np.asarray(raw_gyro, dtype=float) - self.bias

    def update_stationary(self, raw_samples, duration, stationary):
        a = np.asarray(raw_samples, dtype=float).reshape(-1, 3)
        if not self.ready or not stationary or len(a) < 2 or duration <= 0:
            return False
        if not np.isfinite(a).all() or np.max(a.std(axis=0)) >= self.max_std:
            return False
        alpha = -np.expm1(-duration / self.online_time_constant)
        self.bias += alpha * (a.mean(axis=0) - self.bias)
        return True
