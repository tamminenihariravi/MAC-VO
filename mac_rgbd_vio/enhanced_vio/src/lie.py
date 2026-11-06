"""SO(3) maps, including the near-pi logarithm, without SciPy."""
import cv2
import numpy as np


def so3_exp(w):
    return cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3))[0]


def so3_log(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def skew_batch(p):
    p = np.asarray(p)
    s = np.zeros((len(p), 3, 3))
    s[:, 0, 1], s[:, 0, 2] = -p[:, 2], p[:, 1]
    s[:, 1, 0], s[:, 1, 2] = p[:, 2], -p[:, 0]
    s[:, 2, 0], s[:, 2, 1] = -p[:, 1], p[:, 0]
    return s


def quaternion(R):
    w = so3_log(R)
    a = np.linalg.norm(w)
    return np.r_[w * (np.sin(a / 2) / a if a > 1e-12 else 0.5), np.cos(a / 2)]


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    v, w = q[:3], q[3]
    S = skew_batch(v[None])[0]
    return np.eye(3) + 2 * w * S + 2 * S @ S
