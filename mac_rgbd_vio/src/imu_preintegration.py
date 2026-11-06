"""
IMU Preintegration Engine on SO(3) Manifold (Forster et al. formulation)
Integrates high-rate (200 Hz) Gyroscope and Accelerometer measurements between visual keyframes.

Target Paper: IEEE Robotics and Automation Letters (RA-L)
"""

import numpy as np


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) tangent space to SO(3) rotation matrix."""
    theta = np.linalg.norm(omega)
    if theta < 1e-7:
        k = np.array([
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0]
        ], dtype=np.float64)
        return np.eye(3) + k
    axis = omega / theta
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to so(3) vector [omega_x, omega_y, omega_z]."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-7:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * scale


def skew(v: np.ndarray) -> np.ndarray:
    """Skew symmetric matrix of 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


class IMUPreintegrator:
    def __init__(
        self,
        gyr_noise: float = 0.002,   # rad / s / sqrt(Hz)
        acc_noise: float = 0.04,    # m / s^2 / sqrt(Hz)
        gravity: np.ndarray = np.array([0.0, 9.80665, 0.0]) # RealSense Y-down default
    ):
        self.gyr_noise = gyr_noise
        self.acc_noise = acc_noise
        self.g = gravity

        self.reset()

    def reset(self):
        """Resets the preintegration accumulator."""
        self.delta_R = np.eye(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.delta_t = 0.0

        # 9x9 Covariance (delta_phi, delta_v, delta_p)
        self.cov = np.zeros((9, 9), dtype=np.float64)

    def integrate_measurement(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Integrates a single IMU sample (accel, gyro, dt).
        accel: (3,) in body frame
        gyro:  (3,) in body frame
        dt:    sample time delta (seconds)
        """
        if dt <= 0.0 or dt > 0.1:
            return

        acc = accel.copy()
        gyr = gyro.copy()

        # Incremental rotation
        dR = so3_exp(gyr * dt)

        # Position and Velocity increments
        acc_world_i = self.delta_R @ acc
        self.delta_p += self.delta_v * dt + 0.5 * acc_world_i * (dt ** 2)
        self.delta_v += acc_world_i * dt
        self.delta_R = self.delta_R @ dR
        self.delta_t += dt

        # First-order error state propagation covariance
        # F = I + [ -skew(gyr*dt)   0   0 ]
        #     [ -delta_R*skew(acc)*dt  0   0 ]
        #     [ -0.5*delta_R*skew(acc)*dt^2 dt*I 0 ]
        F = np.eye(9, dtype=np.float64)
        F[0:3, 0:3] = dR.T
        F[3:6, 0:3] = -self.delta_R @ skew(acc) * dt
        F[6:9, 0:3] = -0.5 * self.delta_R @ skew(acc) * (dt ** 2)
        F[6:9, 3:6] = np.eye(3) * dt

        # Noise covariance
        G = np.zeros((9, 6), dtype=np.float64)
        G[0:3, 0:3] = np.eye(3) * dt
        G[3:6, 3:6] = self.delta_R * dt
        G[6:9, 3:6] = 0.5 * self.delta_R * (dt ** 2)

        Q = np.diag([
            (self.gyr_noise ** 2), (self.gyr_noise ** 2), (self.gyr_noise ** 2),
            (self.acc_noise ** 2), (self.acc_noise ** 2), (self.acc_noise ** 2)
        ])

        self.cov = F @ self.cov @ F.T + G @ Q @ G.T

    def predict_state(self, R_i: np.ndarray, v_i: np.ndarray, p_i: np.ndarray):
        """
        Predicts state at step j from state at step i:
        R_j = R_i * delta_R
        v_j = v_i + g * delta_t + R_i * delta_v
        p_j = p_i + v_i * delta_t + 0.5 * g * delta_t^2 + R_i * delta_p
        """
        R_j = R_i @ self.delta_R
        v_j = v_i + self.g * self.delta_t + R_i @ self.delta_v
        p_j = p_i + v_i * self.delta_t + 0.5 * self.g * (self.delta_t ** 2) + R_i @ self.delta_p
        return R_j, v_j, p_j
