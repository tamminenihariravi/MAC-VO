"""Evaluate native baseline and optional measured enhanced trajectory on shared times."""
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
from src.evaluation import load_tum, load_motive, estimate_offset, interpolate, metrics, path_length
from src.paths import output_path
from src.lie import quat_matrix


def benchmark(args):
    out = output_path(args.output)
    baseline, gt = load_tum(args.baseline), load_motive(args.gt, args.rigid_body)
    if args.marker_offset is not None:
        gt.positions += np.array([quat_matrix(q) @ np.asarray(args.marker_offset) for q in gt.quaternions])
    sync_file = getattr(args, 'sync_report', None)
    if sync_file is not None:
        sync = json.loads(Path(sync_file).read_text())
        offset = sync['time_offset_seconds']
    else:
        offset, sync = estimate_offset(baseline, gt, args.max_offset) if args.time_offset is None else (args.time_offset, {'method': 'user supplied'})
    confidence = 'manual' if args.time_offset is not None and sync_file is None else ('low; verify with a hardware synchronization event' if sync.get('correlation', 0) < .4 else f'angular correlation r={sync["correlation"]:.3f}')
    baseline_label = getattr(args, 'baseline_label', 'Baseline')
    enhanced_label = getattr(args, 'enhanced_label', 'Enhanced')
    historical_path_target = not getattr(args, 'different_take', False)
    enhanced = load_tum(args.enhanced) if args.enhanced else None
    enhanced_offset = args.enhanced_time_offset
    if enhanced is not None and enhanced_offset is None:
        enhanced_offset = offset  # Same raw stream and sensor clock required.
    times = baseline.times + offset
    pg, valid = interpolate(gt, times, max_gap=args.max_gap)
    pe = None
    if enhanced is not None:
        pe, ev = interpolate(enhanced, times, enhanced_offset, args.max_gap)
        valid &= ev
    if valid.sum() < 20:
        raise ValueError('Insufficient common valid timestamps; check recording and clock offsets')
    times, pg, pb = times[valid], pg[valid], baseline.positions[valid]
    mb, ab, eb = metrics(pb, pg, times)
    me, ae, ee = (None, None, None) if pe is None else metrics(pe[valid], pg, times)
    report = dict(baseline_file=str(args.baseline.resolve()), ground_truth_file=str(args.gt.resolve()),
                  enhanced_file=str(args.enhanced.resolve()) if args.enhanced else None,
                  status='measured comparison' if enhanced is not None else 'single measured trajectory',
                  baseline_label=baseline_label, enhanced_label=enhanced_label if enhanced else None,
                  time_offset_seconds=offset, enhanced_time_offset_seconds=enhanced_offset,
                  marker_to_camera_offset_body_m=args.marker_offset,
                  synchronization=sync, evaluated_poses=len(times), duration_seconds=float(times[-1] - times[0]),
                  common_timestamp_fraction=float(valid.mean()), max_interpolation_gap_seconds=args.max_gap,
                  baseline_input_poses=len(baseline.times), ground_truth_valid_poses=len(gt.times),
                  ground_truth_large_gaps=int(np.sum(np.diff(gt.times) > args.max_gap)),
                  ground_truth_largest_gap_seconds=float(np.max(np.diff(gt.times))),
                  full_native_baseline_raw_y_displacement_m=float(baseline.positions[-1, 1] - baseline.positions[0, 1]),
                  synchronization_confidence=confidence,
                  baseline=mb, enhanced=me,
                  full_native_baseline_path_m=path_length(baseline.positions), full_native_gt_path_m=path_length(gt.positions),
                  full_native_enhanced_path_m=path_length(enhanced.positions) if enhanced else None,
                  targets=dict(ate_rmse_m=.45, path_min_m=32.5 if historical_path_target else None, path_max_m=36.5 if historical_path_target else None, vertical_error_change_m=.25),
                  target_results=None if me is None else dict(ate_pass=me['ate_rmse_m'] < .45,
                     path_pass=32.5 <= path_length(enhanced.positions) <= 36.5 if historical_path_target else None,
                     vertical_pass=abs(me['vertical_error_change_m']) < .25),
                  limitations=['Marker origin compared with camera origin unless --marker-offset is supplied.',
                               'ATE divided by path length is a normalized ATE, not end-point drift.',
                               'Native path length depends on sampling rate and noise; common-rate path also reported.',
                               'Filtered common-time paths bridge excluded intervals; full-native paths also reported.',
                               'Automatic clock offset is exploratory when angular correlation is low.',
                               'Existing covariance parameters may overlap this take; held-out trials required.'])
    (out / 'benchmark_metrics.json').write_text(json.dumps(report, indent=2))
    np.savez(out / 'aligned_comparison.npz', times=times, ground_truth=pg, baseline=ab,
             enhanced=ae if ae is not None else np.empty((0, 3)))
    # Break lines across missing GT instead of drawing invented connecting paths.
    gaps = np.r_[False, np.diff(times) > args.max_gap]
    def broken(a):
        a = a.copy()
        a[gaps] = np.nan
        return a
    pg, ab, eb = broken(pg), broken(ab), broken(eb)
    if ae is not None:
        ae, ee = broken(ae), broken(ee)
    fig = plt.figure(figsize=(17, 5), dpi=160)
    ax = fig.add_subplot(131)
    ax.plot(pg[:, 0], pg[:, 2], 'k--', lw=1.6, label='OptiTrack')
    ax.plot(ab[:, 0], ab[:, 2], color='#d62728', lw=1.3, label=f'{baseline_label} ATE {mb["ate_rmse_m"]:.2f} m')
    if ae is not None:
        ax.plot(ae[:, 0], ae[:, 2], color='#2ca02c', lw=1.3, label=f'{enhanced_label} ATE {me["ate_rmse_m"]:.2f} m')
    ax.set(xlabel='X (m)', ylabel='Z (m)', title='Floor projection after rigid SE(3) alignment')
    ax.axis('equal')
    ax.legend(fontsize=8)
    ax = fig.add_subplot(132, projection='3d')
    ax.plot(*pg.T, 'k--', lw=1.4, label='OptiTrack')
    ax.plot(*ab.T, color='#d62728', lw=1.1, label=baseline_label)
    if ae is not None:
        ax.plot(*ae.T, color='#2ca02c', lw=1.1, label=enhanced_label)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Metric trajectories (scale fixed at 1)')
    ax.legend(fontsize=8)
    ax = fig.add_subplot(133)
    ax.plot(times - times[0], eb, color='#d62728', lw=1.1, label=baseline_label)
    if ee is not None:
        ax.plot(times - times[0], ee, color='#2ca02c', lw=1.1, label=enhanced_label)
    ax.axhline(.45, ls=':', color='gray', label='Target RMSE 0.45 m')
    ax.set(xlabel='Time (s)', ylabel='Aligned position error (m)', title='Absolute position error')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    if enhanced is None:
        fig.suptitle(f'{baseline_label} against OptiTrack; no second trajectory supplied')
    display_offset = sync.get('relative_offset_from_first_gyro_seconds', offset)
    fig.text(.5, .01, f'Clock offset {display_offset:+.2f} s (relative clocks) | synchronization: {confidence} | common timestamp coverage {valid.mean():.1%}', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / 'enhanced_vs_baseline_optitrack.png')
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in ('limitations',)}, indent=2))


def main():
    data = ROOT.parents[1] / 'optitrack_data'
    default_enhanced = data / 'enhanced_135s_trajectory.txt'
    if not default_enhanced.exists():
        default_enhanced = data / 'enhanced_live_trajectory.txt'
    if not default_enhanced.exists():
        default_enhanced = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=data / 'live_camera_trajectory_20261029_182649.txt')
    parser.add_argument('--gt', type=Path, default=data / 'Take_live_135s.csv')
    parser.add_argument('--enhanced', type=Path, default=default_enhanced)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/baseline_135s')
    parser.add_argument('--rigid-body')
    parser.add_argument('--marker-offset', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Marker-body to camera optical-center lever arm in meters')
    parser.add_argument('--time-offset', type=float, help='Seconds to add to baseline timestamps')
    parser.add_argument('--enhanced-time-offset', type=float)
    parser.add_argument('--max-offset', type=float, default=5)
    parser.add_argument('--max-gap', type=float, default=.1)
    parser.add_argument('--sync-report', type=Path, help='Independent synchronization JSON from synchronize_walking_run.py')
    parser.add_argument('--baseline-label', default='Baseline')
    parser.add_argument('--enhanced-label', default='Enhanced')
    parser.add_argument('--different-take', action='store_true', help='Disable the old 135-second take-specific absolute path target')
    args = parser.parse_args()
    benchmark(args)


if __name__ == '__main__':
    main()
