"""Export measured values for editable PGFPlots figures in the Beamer source."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
VIO = ROOT.parent
sys.path.insert(0, str(VIO))
from src.evaluation import load_motive, interpolate
from scripts.synchronize_walking_run import quaternion_at


def write(name, array, header):
    np.savetxt(ROOT / 'data' / name, array, delimiter=',', header=header, comments='', fmt='%.8f')


run = VIO / 'results/walking_201116'
report = json.loads((run / 'comparison/benchmark_metrics.json').read_text())
comparison = np.load(run / 'comparison/aligned_comparison.npz')
t = comparison['times']
gaps = np.r_[False, np.diff(t) > .1]
gt, recorded, revised = [comparison[key].copy() for key in ['ground_truth', 'baseline', 'enhanced']]
for name, points in [('gt.csv', gt), ('recorded.csv', recorded), ('revised.csv', revised)]:
    points[gaps] = np.nan
    write(name, np.column_stack((t - t[0], points)), 'time,x,y,z')
er = np.linalg.norm(recorded - gt, axis=1)
ev = np.linalg.norm(revised - gt, axis=1)
er[gaps], ev[gaps] = np.nan, np.nan
write('position_error.csv', np.column_stack((t - t[0], er, ev)), 'time,recorded,revised')

capture = VIO / 'results/capture_20261029_201116'
meta = json.loads((capture / 'metadata.json').read_text())
raw = np.loadtxt(capture / 'gyro.csv', delimiter=',', skiprows=1)
startup = raw[(raw[:, 0] >= meta['calibration_start_timestamp']) & (raw[:, 0] <= meta['calibration_end_timestamp'])]
write('gyro_startup.csv', np.column_stack((startup[:, 0] - startup[0, 0], startup[:, 1], startup[:, 1] - meta['startup_bias_rad_s'][0])), 'time,raw,corrected')

# Exactly the same finite-window synchronization signal as the evaluated report.
sync = report['synchronization']
gt_raw = load_motive(Path(report['ground_truth_file']))
origin = raw[0, 0]
raw[:, 1:4] -= meta['startup_bias_rad_s']
raw[:, 0] -= origin
window = sync['orientation_window_seconds']
grid = np.arange(window / 2, raw[-1, 0] - window / 2, .02)
samples = np.arange(-window / 2, window / 2 + .001, .005)
w = np.stack([np.column_stack([np.interp(grid + s, raw[:, 0], raw[:, j]) for j in range(1, 4)]) for s in samples])
signal = np.linalg.norm(w.mean(axis=0), axis=1)
write('gyro_sync.csv', np.column_stack((grid + sync['relative_offset_from_first_gyro_seconds'], signal)), 'time,speed')
ggrid = np.arange(gt_raw.times[0] + window / 2, gt_raw.times[-1] - window / 2, .02)
qa, qb = quaternion_at(gt_raw, ggrid - window / 2), quaternion_at(gt_raw, ggrid + window / 2)
gsignal = 2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, axis=1)), -1, 1)) / window
valid = np.ones(len(ggrid), bool)
for shift in np.arange(-window / 2, window / 2 + .001, .02):
    valid &= interpolate(gt_raw, ggrid + shift, max_gap=.1)[1]
valid &= gsignal < sync['orientation_rate_rejection_limit_rad_s']
gsignal[~valid] = np.nan
write('gt_sync.csv', np.column_stack((ggrid, gsignal)), 'time,speed')

write('timing.csv', np.array([[1, 39.77640198, 53.92987001], [2, 18.16821703, 24.53596001]]), 'run,mean,p95')
write('path.csv', np.array([[1, report['baseline']['gt_path_m']], [2, report['baseline']['estimated_path_m']], [3, report['enhanced']['estimated_path_m']]]), 'run,path')
write('vertical.csv', np.array([[1, abs(report['baseline']['vertical_error_change_m'])], [2, abs(report['enhanced']['vertical_error_change_m'])]]), 'run,error')
(ROOT / 'data/source_metrics.json').write_text(json.dumps(report, indent=2))
print('Measured plot data exported:', len(list((ROOT / 'data').glob('*.csv'))), 'CSV files')
