"""Numerical regression tests for conventions, fusion, gating, and evaluation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.imu_calibrator import IMUCalibrator
from src.gyro_preintegration import GyroBuffer
from src.zupt_detector import ZUPTDetector
from src.decoupled_optimizer import DecoupledOptimizer
from src.lie import so3_exp, so3_log, quaternion, quat_matrix
from src.keyframe_tracker import KeyframeTracker
from src.pipeline import EnhancedVIO
from src.evaluation import Trajectory, metrics, interpolate, estimate_offset, path_length


class CalibrationTests(unittest.TestCase):
    def test_startup_400_samples_and_two_seconds_then_debias(self):
        c = IMUCalibrator()
        bias = np.array([.001, np.deg2rad(-.25), -.002])
        rng = np.random.default_rng(1)
        for i in range(400):
            c.add_startup(i / 200, bias + rng.normal(0, .0002, 3))
        self.assertFalse(c.ready)  # 400 samples at 200Hz span 1.995 seconds.
        self.assertTrue(c.add_startup(2., bias))
        np.testing.assert_allclose(c.bias, bias, atol=5e-5)
        np.testing.assert_allclose(c.correct(bias), 0, atol=5e-5)
        before = c.bias.copy()
        self.assertFalse(c.update_stationary(np.tile(bias + .001, (10, 1)), .1, False))
        np.testing.assert_array_equal(c.bias, before)
        self.assertTrue(c.update_stationary(np.tile(bias + .001, (10, 1)), .1, True))
        self.assertLess(np.linalg.norm(c.bias - before), .00001)

    def test_motion_and_duplicate_timestamps_rejected(self):
        c = IMUCalibrator()
        for i in range(401):
            c.add_startup(i / 200, [0, 0, .1])
        self.assertFalse(c.ready)
        self.assertEqual(c.rejected_windows, 1)
        with self.assertRaises(ValueError):
            c.add_startup(2., [0, 0, 0])
        c = IMUCalibrator()
        for i in range(401):
            c.add_startup(i / 200, [.02 * (-1) ** i, 0, 0])
        self.assertFalse(c.ready)


class InertialTests(unittest.TestCase):
    def test_exact_endpoint_coverage_bias_and_extrinsic(self):
        C = so3_exp([0, 0, np.pi / 2])
        b = GyroBuffer(C)
        bias = np.array([0, -.004, .001])
        for t in np.arange(0, .101, .005):
            b.append(t, np.array([.1, 0, 0]) + bias)
        R, _, _ = b.interval(.007, .083, bias)
        np.testing.assert_allclose(so3_log(R), C @ np.array([.0076, 0, 0]), atol=1e-10)
        with self.assertRaises(ValueError):
            b.interval(.01, .12, bias)
        with self.assertRaises(ValueError):
            b.append(.1, [0, 0, 0])

    def test_zupt_requires_tracks_and_all_gyro_samples(self):
        d = ZUPTDetector()
        p = np.zeros((30, 2))
        for _ in range(2):
            self.assertFalse(d.update(p, p + .1, [[0, 0, 0]]).is_stationary)
        self.assertTrue(d.update(p, p + .1, [[0, 0, 0]]).is_stationary)
        self.assertFalse(d.update(p, p, [[0, 0, .1], [0, 0, -.1]]).is_stationary)
        self.assertFalse(d.update(p[:0], p[:0], [[0, 0, 0]]).is_stationary)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.b = rng.uniform([-1, -.6, 1.], [1., .6, 3.], (240, 3))
        self.R = so3_exp([.025, -.035, .01])
        self.t = np.array([.045, -.013, .025])
        self.a = self.b @ self.R.T + self.t
        self.C = np.broadcast_to(np.diag([.0001, .0002, .00004]), (240, 3, 3)).copy()
        self.opt = DecoupledOptimizer()

    def test_exact_pose_and_rotation_direction(self):
        result = self.opt.solve(self.a, self.b, self.C, self.C, self.R)
        self.assertTrue(result.success, result.reason)
        np.testing.assert_allclose(result.rotation, self.R, atol=1e-8)
        np.testing.assert_allclose(result.translation, self.t, atol=1e-8)

    def test_translation_refit_after_fusion_and_covariance_frame(self):
        imu = so3_exp([.038, -.021, .002])
        pose = self.opt.solve(self.a, self.b, self.C, self.C, imu)
        self.assertTrue(pose.success, pose.reason)
        expected_R = so3_exp(.75 * so3_log(self.R) + .25 * so3_log(imu))
        np.testing.assert_allclose(pose.rotation, expected_R, atol=1e-7)
        W = self.opt.information(self.C, self.C, expected_R)
        d = self.a - self.b @ expected_R.T
        expected_t = self.opt.solve_translation(self.a, self.b, expected_R, W)
        np.testing.assert_allclose(pose.translation, expected_t, atol=1e-8)
        def cost(t):
            r = d - t
            return np.einsum('ni,nij,nj->', r, W, r)
        self.assertLess(cost(pose.translation), cost(self.t))
        cov = self.C + expected_R[None] @ self.C @ expected_R.T[None]
        np.testing.assert_allclose(W @ (cov + 1e-9 * np.eye(3)), np.broadcast_to(np.eye(3), W.shape), atol=1e-10)

    def test_outliers_and_degenerate_geometry(self):
        a = self.a.copy()
        a[:50] += [1, -.5, .4]
        pose = self.opt.solve(a, self.b, self.C, self.C, self.R)
        self.assertTrue(pose.success, pose.reason)
        np.testing.assert_allclose(pose.translation, self.t, atol=1e-7)
        self.assertEqual(pose.inliers.sum(), 190)
        b = np.zeros_like(self.b)
        self.assertFalse(self.opt.solve(b, b, self.C, self.C).success)


class TrackingTests(unittest.TestCase):
    def test_sparse_structure_tensor_matches_opencv(self):
        rng = np.random.default_rng(77)
        gray = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        x, y = rng.integers(1, 319, 100), rng.integers(1, 239, 100)
        tracker = KeyframeTracker(dict(fx=400., fy=400., cx=160., cy=120.))
        expected = cv2.cornerMinEigenVal(gray, blockSize=7)[y, x]
        np.testing.assert_allclose(tracker.point_eigenvalues(gray, x, y), expected, atol=3e-8)

    def test_keyframe_policy_and_stationary_pipeline(self):
        cv2.setNumThreads(1)
        intr = dict(fx=400., fy=400., cx=160., cy=120.)
        tracker = KeyframeTracker(intr)
        tracker.reference_count = 100
        self.assertFalse(tracker.should_insert(3.4, np.deg2rad(2.4), 65))
        self.assertTrue(tracker.should_insert(3.6, 0, 100))
        self.assertTrue(tracker.should_insert(0, np.deg2rad(2.6), 100))
        self.assertTrue(tracker.should_insert(0, 0, 64))
        rng = np.random.default_rng(3)
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        depth = np.full(gray.shape, 2., np.float32)
        c = IMUCalibrator()
        for t in np.arange(0, 2.006, .005):
            c.add_startup(t, [0, -.004, 0])
        g = GyroBuffer()
        for t in np.arange(2, 3.01, .005):
            g.append(t, [0, -.004, 0])
        vio = EnhancedVIO(intr, c, g)
        results = [vio.process(gray, depth, 2.01 + i / 30) for i in range(15)]
        self.assertTrue(results[-1].stationary)
        np.testing.assert_allclose(results[-1].position, 0, atol=1e-9)
        np.testing.assert_array_equal(results[-1].velocity, 0)
        self.assertEqual(vio.keyframes, 1)
        self.assertEqual(vio.tracking_failures, 0)

    def test_keyframe_pose_is_not_reaccumulated_each_frame(self):
        intr = dict(fx=400., fy=400., cx=160., cy=120.)
        rng = np.random.default_rng(22)
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), .5)
        depth = np.full(gray.shape, 2., np.float32)
        c = IMUCalibrator()
        for i in range(402):
            c.add_startup(i / 200, [0, 0, 0])
        g = GyroBuffer()
        for t in np.arange(2, 3.01, .005):
            g.append(t, [0, 0, 0])
        vio = EnhancedVIO(intr, c, g)
        for i in range(10):
            # Camera moves left: scene shifts right by 1 pixel per frame.
            image = cv2.warpAffine(gray, np.float32([[1, 0, i], [0, 1, 0]]), (320, 240))
            r = vio.process(image, depth, 2.01 + i / 30)
        self.assertEqual(vio.tracking_failures, 0)
        self.assertAlmostEqual(r.position[0], -.045, delta=.003)
        self.assertLess(np.linalg.norm(r.position[1:]), .003)
        self.assertGreater(vio.keyframes, 1)
        self.assertGreater(vio.tracker.reused_keyframes, 0)

    def test_stop_within_keyframe_freezes_current_position(self):
        intr = dict(fx=400., fy=400., cx=160., cy=120.)
        rng = np.random.default_rng(11)
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        depth = np.full(gray.shape, 2., np.float32)
        c = IMUCalibrator()
        for i in range(402):
            c.add_startup(i / 200, [0, 0, 0])
        g = GyroBuffer()
        for t in np.arange(2, 3.01, .005):
            g.append(t, [0, 0, 0])
        vio = EnhancedVIO(intr, c, g)
        held_position = None
        for i, shift in enumerate([0, 1, 2, 3, 3, 3, 3, 3, 3]):
            image = cv2.warpAffine(gray, np.float32([[1, 0, shift], [0, 1, 0]]), (320, 240))
            result = vio.process(image, depth, 2.01 + i / 30)
            if result.stationary and held_position is None:
                held_position = result.position.copy()
        self.assertTrue(result.stationary)
        self.assertEqual(vio.keyframes, 1)
        self.assertAlmostEqual(result.position[0], -.015, delta=.002)
        np.testing.assert_array_equal(result.position, held_position)
        np.testing.assert_array_equal(result.velocity, np.zeros(3))


class EvaluationTests(unittest.TestCase):
    def test_rigid_alignment_does_not_rescale(self):
        rng = np.random.default_rng(30)
        a = rng.normal(size=(100, 3))
        b = a @ so3_exp([.2, -.3, .1]).T + [1, 2, 3]
        m, _, _ = metrics(a, b)
        self.assertLess(m['ate_rmse_m'], 1e-12)
        m, _, _ = metrics(a * 2, b)
        self.assertGreater(m['ate_rmse_m'], 1)
        self.assertEqual(m['alignment_scale'], 1.)

    def test_no_interpolation_over_missing_gt(self):
        t = np.array([0, .01, .5, .51])
        tr = Trajectory(t, np.zeros((4, 3)), np.tile([0, 0, 0, 1], (4, 1)))
        _, valid = interpolate(tr, np.array([.005, .2, .505, .52]))
        np.testing.assert_array_equal(valid, [True, False, True, False])

    def test_time_offset_sign(self):
        t = np.arange(0, 20, .02)
        angle = .2 * np.sin(t * 1.3) + .07 * np.sin(t * 3.7) + .002 * t ** 2
        q = np.array([quaternion(so3_exp([0, a, 0])) for a in angle])
        p = np.column_stack((np.sin(t), np.zeros_like(t), np.cos(t)))
        gt = Trajectory(t + .73, p, q)
        est = Trajectory(t, p, q)
        offset, _ = estimate_offset(est, gt, 2)
        self.assertAlmostEqual(offset, .73, delta=.02)

    def test_near_pi_rotation_roundtrip(self):
        R = so3_exp([np.pi - 1e-7, 0, 0])
        np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-6)
        np.testing.assert_allclose(quat_matrix(quaternion(R)), R, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
