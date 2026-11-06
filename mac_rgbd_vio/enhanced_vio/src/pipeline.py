"""Keyframe-anchored camera poses. No accelerometer position dead reckoning."""
from dataclasses import dataclass
import numpy as np
from .keyframe_tracker import KeyframeTracker
from .decoupled_optimizer import DecoupledOptimizer
from .zupt_detector import ZUPTDetector
from .lie import so3_log


@dataclass
class FrameResult:
    rotation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    stationary: bool
    status: str
    inliers: int
    disparity_px: float
    inserted_keyframe: bool
    pixels: np.ndarray


class EnhancedVIO:
    def __init__(self, intrinsics, calibrator, gyro_buffer, covariance_config=None, max_features=300):
        self.tracker = KeyframeTracker(intrinsics, covariance_config, max_features=max_features)
        self.optimizer = DecoupledOptimizer()
        self.zupt = ZUPTDetector()
        self.calibrator, self.gyro = calibrator, gyro_buffer
        self.R, self.p, self.v = np.eye(3), np.zeros(3), np.zeros(3)
        self.key_R, self.key_p = self.R.copy(), self.p.copy()
        self.last_timestamp = None
        self.keyframes = 0
        self.tracking_failures = 0
        self.key_delta_R = np.eye(3)
        self.imu_complete = True

    def process(self, gray, depth_m, timestamp):
        if not self.calibrator.ready:
            raise RuntimeError("Calibrate IMU before starting VIO")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Image timestamps must increase")
        if self.tracker.reference_gray is None:
            self.tracker.set_reference(gray, depth_m, timestamp)
            self.last_timestamp = timestamp
            self.keyframes += 1
            return FrameResult(self.R.copy(), self.p.copy(), self.v.copy(), False, 'initialized', 0, 0, True, self.tracker.points)
        dt = timestamp - self.last_timestamp
        tracks = self.tracker.track(gray, depth_m)
        imu_R = None
        raw, corrected = np.empty((0, 3)), np.empty((0, 3))
        try:
            frame_R, corrected, raw = self.gyro.interval(self.last_timestamp, timestamp, self.calibrator.bias)
            self.key_delta_R = self.key_delta_R @ frame_R
            if self.imu_complete:
                imu_R = self.key_delta_R
        except ValueError:
            self.imu_complete = False
        decision = self.zupt.update(tracks.previous_pixels, tracks.current_pixels, corrected)
        pose = self.optimizer.solve(tracks.reference_points, tracks.current_points,
                                    tracks.reference_covariance, tracks.current_covariance, imu_R)
        inserted = False
        inliers = int(pose.inliers.sum())
        if pose.success:
            old_p = self.p.copy()
            self.R = self.key_R @ pose.rotation
            if decision.is_stationary:
                # Zero FRAME step. A zero KEYFRAME translation would snap backwards.
                self.v[:] = 0
                self.calibrator.update_stationary(raw, dt, True)
            else:
                self.p = self.key_p + self.key_R @ pose.translation
                self.v = (self.p - old_p) / dt
            status = 'stationary' if decision.is_stationary else ('tracking' if imu_R is not None else 'visual_only_imu_gap')
            if self.tracker.should_insert(tracks.disparity, np.linalg.norm(so3_log(pose.rotation)), inliers):
                self.key_R, self.key_p = self.R.copy(), self.p.copy()
                self.tracker.advance_reference(gray, depth_m, timestamp, tracks, pose.inliers)
                self.gyro.discard_before(timestamp)
                self.keyframes += 1
                inserted = True
                self.key_delta_R = np.eye(3)
                self.imu_complete = True
        else:
            # Reacquire at the last reliable pose. Unknown motion is explicitly lost.
            self.tracking_failures += 1
            self.v[:] = 0
            status = 'lost:' + pose.reason
            self.key_R, self.key_p = self.R.copy(), self.p.copy()
            self.tracker.set_reference(gray, depth_m, timestamp)
            self.gyro.discard_before(timestamp)
            self.keyframes += 1
            inserted = True
            self.key_delta_R = np.eye(3)
            self.imu_complete = True
        self.last_timestamp = timestamp
        self.gyro.discard_before(timestamp)
        return FrameResult(self.R.copy(), self.p.copy(), self.v.copy(), decision.is_stationary and pose.success,
                           status, inliers, tracks.disparity, inserted, tracks.current_pixels)
