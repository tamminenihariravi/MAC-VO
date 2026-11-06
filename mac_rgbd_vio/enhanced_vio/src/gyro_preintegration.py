"""Timestamp-bounded gyro integration in the RGB optical coordinate system."""
import numpy as np
from .lie import so3_exp


class GyroBuffer:
    def __init__(self, rotation_color_from_gyro=None, max_gap=0.03):
        self.rotation = np.eye(3) if rotation_color_from_gyro is None else np.asarray(rotation_color_from_gyro)
        if self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5) or np.linalg.det(self.rotation) < 0:
            raise ValueError("IMU-to-color rotation must be in SO(3)")
        self.samples = []
        self.max_gap = max_gap

    def append(self, timestamp, gyro):
        w = np.asarray(gyro, dtype=float)
        if w.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(timestamp):
            raise ValueError("Invalid gyro packet")
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("Duplicate or out-of-order gyro timestamp")
        self.samples.append((float(timestamp), w.copy()))

    def interval(self, start, end, bias):
        if end <= start or len(self.samples) < 2:
            raise ValueError("Insufficient IMU coverage")
        times = np.array([x[0] for x in self.samples])
        if start < times[0] or end > times[-1]:
            raise ValueError("IMU samples must bracket both image timestamps")
        lo = max(0, np.searchsorted(times, start, side='right') - 1)
        hi = min(len(times), np.searchsorted(times, end, side='left') + 1)
        ts = times[lo:hi]
        if np.max(np.diff(ts), initial=0) > self.max_gap:
            raise ValueError("IMU packet gap exceeds limit")
        raw = np.array([x[1] for x in self.samples[lo:hi]])
        knots = np.r_[start, ts[(ts > start) & (ts < end)], end]
        wk = np.column_stack([np.interp(knots, ts, raw[:, j]) for j in range(3)])
        corrected = (wk - bias) @ self.rotation.T
        R = np.eye(3)
        for dt, a, b in zip(np.diff(knots), corrected[:-1], corrected[1:]):
            R = R @ so3_exp((a + b) * (0.5 * dt))
        # R maps the current camera's coordinates into the reference camera.
        return R, corrected, wk

    def discard_before(self, timestamp):
        while len(self.samples) > 2 and self.samples[1][0] <= timestamp:
            self.samples.pop(0)
