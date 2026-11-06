"""Reproducible rigid (scale=1) trajectory evaluation without GT trajectory tuning."""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .lie import quat_matrix, so3_log


@dataclass
class Trajectory:
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


def clean(t, p, q):
    t, p, q = np.asarray(t, float), np.asarray(p, float), np.asarray(q, float)
    valid = np.isfinite(t) & np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > .1)
    t, p, q = t[valid], p[valid], q[valid]
    order = np.argsort(t, kind='stable')
    t, p, q = t[order], p[order], q[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, p, q = t[unique], p[unique], q[unique]
    if len(t) < 3:
        raise ValueError('Fewer than three valid unique trajectory poses')
    return Trajectory(t, p, q / np.linalg.norm(q, axis=1)[:, None])


def load_tum(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 8:
        raise ValueError('TUM input must have eight columns: timestamp tx ty tz qx qy qz qw')
    return clean(data[:, 0], data[:, 1:4], data[:, 4:8])


def load_motive(path, rigid_body=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = []
        for row in reader:
            if len(row) > 1 and row[0].strip().lower() == 'frame' and 'time' in row[1].lower():
                axes = row
                break
            headers.append(row)
        else:
            raise ValueError('Motive Frame/Time header not found')
        kind, names, quantity = headers[-4], headers[-3], headers[-1]
        bodies = list(dict.fromkeys(names[i] for i in range(2, len(kind)) if kind[i] == 'Rigid Body'))
        if rigid_body is None and len(bodies) != 1:
            raise ValueError(f'Select --rigid-body from {bodies}')
        body = rigid_body or bodies[0]
        def columns(qty, wanted):
            cols = [next((i for i in range(2, len(axes)) if kind[i] == 'Rigid Body' and names[i] == body and quantity[i] == qty and axes[i] == a), None) for a in wanted]
            if None in cols:
                raise ValueError(f'No {qty} {wanted} columns for rigid body {body}')
            return cols
        pc, qc = columns('Position', 'XYZ'), columns('Rotation', 'XYZW')
        metadata = headers[0]
        units = metadata[metadata.index('Length Units') + 1].lower() if 'Length Units' in metadata else 'meters'
        scale = {'meters': 1., 'millimeters': .001, 'centimeters': .01}.get(units)
        if scale is None:
            raise ValueError(f'Unsupported Motive length units: {units}')
        values = []
        for row in reader:
            try:
                values.append([float(row[1]), *[float(row[i]) * scale for i in pc], *[float(row[i]) for i in qc]])
            except (ValueError, IndexError):
                continue
    a = np.asarray(values)
    if not len(a):
        raise ValueError('No valid Motive poses')
    return clean(a[:, 0], a[:, 1:4], a[:, 4:8])


def angular_speed(trajectory):
    matrices = np.array([quat_matrix(q) for q in trajectory.quaternions])
    dR = np.swapaxes(matrices[:-1], 1, 2) @ matrices[1:]
    speed = np.array([np.linalg.norm(so3_log(r)) for r in dR]) / np.diff(trajectory.times)
    return (trajectory.times[:-1] + trajectory.times[1:]) * .5, speed


def estimate_offset(est, gt, max_offset=5.0):
    te, we = angular_speed(est)
    tg, wg = angular_speed(gt)
    # Smooth angular speed at a common 50 Hz cadence, then evaluate bounded shifts.
    base = gt.times[0] - est.times[0] if abs(gt.times[0] - est.times[0]) > 100 else 0.0
    grid = np.arange(te[0], te[-1], .02)
    a = np.interp(grid, te, we)
    kernel = np.ones(11) / 11
    a = np.convolve(a, kernel, mode='same')
    ggrid = np.arange(tg[0], tg[-1], .02)
    b = np.convolve(np.interp(ggrid, tg, wg), kernel, mode='same')
    scores = []
    offsets = base + np.arange(-max_offset, max_offset + .005, .01)
    for offset in offsets:
        valid = (grid + offset >= ggrid[0]) & (grid + offset <= ggrid[-1])
        if valid.sum() < max(30, .80 * len(grid)):
            scores.append(-np.inf)
            continue
        x, y = a[valid], np.interp(grid[valid] + offset, ggrid, b)
        x, y = x - x.mean(), y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        scores.append(float(x @ y / denom) if denom > 1e-8 else -np.inf)
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        raise ValueError('Angular motion cannot identify clock offset; supply --time-offset')
    return float(offsets[best]), dict(correlation=float(scores[best]), search_boundary=best in (0, len(offsets) - 1), resolution_seconds=.01,
                                     method='bounded angular-speed correlation; no position-error minimization')


def interpolate(trajectory, times, offset=0, max_gap=.1):
    ts = trajectory.times + offset
    ix = np.searchsorted(ts, times, side='right')
    left, right = np.clip(ix - 1, 0, len(ts) - 1), np.clip(ix, 0, len(ts) - 1)
    exact = np.abs(ts[left] - times) < 1e-8
    valid = (times >= ts[0]) & (times <= ts[-1]) & ((ts[right] - ts[left] <= max_gap) | exact)
    p = np.column_stack([np.interp(times, ts, trajectory.positions[:, j]) for j in range(3)])
    return p, valid


def umeyama_se3(estimate, reference):
    a, b = estimate - estimate.mean(axis=0), reference - reference.mean(axis=0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = Vt.T @ np.diag([1., 1., np.linalg.det(Vt.T @ U.T)]) @ U.T
    t = reference.mean(axis=0) - R @ estimate.mean(axis=0)
    return R, t


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def compute_rpe(estimate, ground_truth, times, delta_t=1.0):
    """Computes Relative Pose Error (RPE) in translation over delta_t second windows."""
    if len(times) < 5:
        return dict(rpe_trans_rmse_m=0.0, rpe_trans_mean_m=0.0, rpe_trans_median_m=0.0)
    trans_errs = []
    for i in range(len(times)):
        target_t = times[i] + delta_t
        j = np.searchsorted(times, target_t)
        if j < len(times) and abs(times[j] - target_t) <= 0.15:
            d_est = estimate[j] - estimate[i]
            d_gt = ground_truth[j] - ground_truth[i]
            trans_errs.append(np.linalg.norm(d_est - d_gt))
    if not trans_errs:
        d_est = np.diff(estimate, axis=0)
        d_gt = np.diff(ground_truth, axis=0)
        trans_errs = np.linalg.norm(d_est - d_gt, axis=1)
    trans_errs = np.asarray(trans_errs)
    return dict(
        rpe_trans_rmse_m=float(np.sqrt(np.mean(trans_errs ** 2))),
        rpe_trans_mean_m=float(np.mean(trans_errs)),
        rpe_trans_median_m=float(np.median(trans_errs)),
    )


def metrics(estimate, ground_truth, times=None):
    R, t = umeyama_se3(estimate, ground_truth)
    aligned = estimate @ R.T + t
    error = aligned - ground_truth
    norm = np.linalg.norm(error, axis=1)
    gt_path = path_length(ground_truth)
    result = dict(ate_rmse_m=float(np.sqrt(np.mean(norm ** 2))), ate_median_m=float(np.median(norm)),
                  ate_max_m=float(norm.max()), floor_projected_ate_rmse_m=float(np.sqrt(np.mean(np.sum(error[:, [0, 2]] ** 2, axis=1)))),
                  vertical_error_change_m=float(error[-1, 1] - error[0, 1]),
                  vertical_error_peak_to_peak_m=float(np.ptp(error[:, 1])),
                  raw_y_displacement_m=float(estimate[-1, 1] - estimate[0, 1]),
                  estimated_path_m=path_length(estimate), gt_path_m=gt_path,
                  ate_percent_of_gt_path=float(100 * np.sqrt(np.mean(norm ** 2)) / max(gt_path, 1e-12)),
                  alignment_rotation=R.tolist(), alignment_translation=t.tolist(), alignment_scale=1.0)
    if times is not None:
        rpe_dict = compute_rpe(aligned, ground_truth, times, delta_t=1.0)
        result.update(rpe_dict)
    # Independent SE(2) floor alignment is a different metric from the 3D projection.
    aa, bb = estimate[:, [0, 2]], ground_truth[:, [0, 2]]
    U, _, Vt = np.linalg.svd((aa - aa.mean(axis=0)).T @ (bb - bb.mean(axis=0)))
    Q = Vt.T @ np.diag([1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    floor_error = (aa - aa.mean(axis=0)) @ Q.T - (bb - bb.mean(axis=0))
    result['floor_independent_se2_ate_rmse_m'] = float(np.sqrt(np.mean(np.sum(floor_error ** 2, axis=1))))
    return result, aligned, norm
