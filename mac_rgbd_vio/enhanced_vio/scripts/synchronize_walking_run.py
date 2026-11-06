"""Align raw gyro with finite-window Motive orientation changes, without fitting positions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import load_motive, interpolate
from src.paths import output_path


def quaternion_at(gt, times):
    ix = np.clip(np.searchsorted(gt.times, times), 1, len(gt.times) - 1)
    lo, hi = ix - 1, ix
    alpha = np.clip((times - gt.times[lo]) / (gt.times[hi] - gt.times[lo]), 0, 1)
    qa, qb = gt.quaternions[lo], gt.quaternions[hi].copy()
    qb[np.sum(qa * qb, axis=1) < 0] *= -1
    q = (1 - alpha[:, None]) * qa + alpha[:, None] * qb
    return q / np.linalg.norm(q, axis=1)[:, None]


def synchronize(capture, gt_file, output, window=.3, max_offset=10):
    if window <= 0 or max_offset <= 0:
        raise ValueError('Window and offset range must be positive')
    out = output_path(output)
    gt = load_motive(gt_file)
    raw = np.loadtxt(capture / 'gyro.csv', delimiter=',', skiprows=1)
    metadata = json.loads((capture / 'metadata.json').read_text())
    raw[:, 1:4] -= metadata['startup_bias_rad_s']
    origin = raw[0, 0]
    raw[:, 0] -= origin
    grid = np.arange(window / 2, raw[-1, 0] - window / 2, .02)
    # Compare vector-averaged rates rather than mean noisy rate magnitudes.
    samples = np.arange(-window / 2, window / 2 + .001, .005)
    w = np.stack([np.column_stack([np.interp(grid + s, raw[:, 0], raw[:, j]) for j in range(1, 4)]) for s in samples])
    signal = np.linalg.norm(w.mean(axis=0), axis=1)
    ggrid = np.arange(gt.times[0] + window / 2, gt.times[-1] - window / 2, .02)
    qa = quaternion_at(gt, ggrid - window / 2)
    qb = quaternion_at(gt, ggrid + window / 2)
    dot = np.abs(np.sum(qa * qb, axis=1))
    gsignal = 2 * np.arccos(np.clip(dot, -1, 1)) / window
    # Every position within the derivative window must have nearby GT support.
    gvalid = np.ones(len(ggrid), bool)
    for shift in np.arange(-window / 2, window / 2 + .001, .02):
        _, valid = interpolate(gt, ggrid + shift, max_gap=.1)
        gvalid &= valid
    # Reject orientation flips which are inconsistent with the recorded physical rate.
    # Report exclusion explicitly; this filtering does not change any position samples.
    physical_limit = max(2., float(np.max(np.linalg.norm(raw[:, 1:4], axis=1))) * 2)
    gvalid &= gsignal < physical_limit
    offsets = np.arange(-max_offset, max_offset + .005, .01)
    scores, support = [], []
    for lag in offsets:
        shifted = grid + lag
        valid = (shifted >= ggrid[0]) & (shifted <= ggrid[-1]) & (np.interp(shifted, ggrid, gvalid.astype(float)) > .999)
        support.append(float(valid.sum() * .02))
        if valid.sum() < 250:
            scores.append(-1.)
            continue
        a, b = signal[valid], np.interp(shifted[valid], ggrid, gsignal)
        a, b = a - a.mean(), b - b.mean()
        scores.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    best = int(np.argmax(scores))
    if scores[best] <= -1:
        raise ValueError('Insufficient valid GT orientation support for independent synchronization')
    lag = float(offsets[best])
    result = dict(capture=str(capture.resolve()), gt=str(gt_file.resolve()),
                  time_offset_seconds=lag - origin, relative_offset_from_first_gyro_seconds=lag,
                  correlation=scores[best], reliable_support_seconds=support[best],
                  orientation_window_seconds=window, maximum_gt_gap_seconds=.1,
                  orientation_rate_rejection_limit_rad_s=physical_limit,
                  gt_signal_valid_fraction=float(gvalid.mean()),
                  search_boundary=best in (0, len(offsets) - 1),
                  method='raw debiased gyro vector window mean vs finite-window GT quaternion displacement; no position fitting')
    (out / 'synchronization.json').write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    ax[0].plot(grid + lag, signal, label='Debiased gyro', lw=1)
    visible_gt = gsignal.copy()
    visible_gt[~gvalid] = np.nan
    ax[0].plot(ggrid, visible_gt, label='OptiTrack finite-window rotation', lw=1)
    ax[0].set(xlabel='Motive time (s)', ylabel='Angular speed (rad/s)', title=f'Independent IMU sync, r={scores[best]:.3f}')
    ax[0].legend(fontsize=8)
    ax[1].plot(offsets, scores)
    ax[1].axvline(lag, color='red', ls=':')
    ax[1].set(xlabel='Offset from first gyro packet (s)', ylabel='Correlation', title='Bounded time-offset search')
    fig.tight_layout()
    fig.savefig(out / 'synchronization.png')
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=float, default=.3)
    parser.add_argument('--max-offset', type=float, default=10)
    args = parser.parse_args()
    synchronize(args.capture, args.gt, args.output, args.window, args.max_offset)
