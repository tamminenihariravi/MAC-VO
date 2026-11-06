"""Independent raw-IMU synchronization should recover an epoch clock shift."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.synchronize_walking_run import synchronize


class SynchronizationTests(unittest.TestCase):
    def test_noisy_quaternions_and_missing_gt_preserve_offset_sign(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as temporary:
            folder = Path(temporary)
            t = np.arange(0, 16, .005)
            origin, lag = 1.e9, .73
            bias = np.array([.001, -.004, .002])
            angle = lambda x: .2 * np.sin(1.1 * x) + .05 * np.sin(3.3 * x)
            rate = .22 * np.cos(1.1 * t) + .165 * np.cos(3.3 * t)
            gyro = np.tile(bias, (len(t), 1))
            gyro[:, 2] += rate
            np.savetxt(folder / 'gyro.csv', np.column_stack((origin + t, gyro)), delimiter=',', header='timestamp,gx,gy,gz', comments='')
            (folder / 'metadata.json').write_text(json.dumps(dict(startup_bias_rad_s=bias.tolist())))
            tg = np.arange(0, 16, 1 / 120)
            tg = tg[(tg < 6) | (tg > 7)]
            rng = np.random.default_rng(9)
            noisy_angle = angle(tg) + rng.normal(0, .0007, len(tg))
            with (folder / 'gt.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([['Format Version', '1.22', 'Length Units', 'Meters'], [],
                                 ['', '', *(['Rigid Body'] * 7)], ['', '', *(['Camera'] * 7)],
                                 ['', '', *(['id'] * 7)], ['', '', *(['Rotation'] * 4), *(['Position'] * 3)],
                                 ['Frame', 'Time (Seconds)', 'X', 'Y', 'Z', 'W', 'X', 'Y', 'Z']])
                for i, (stamp, a) in enumerate(zip(tg, noisy_angle)):
                    writer.writerow([i, stamp + lag, 0, 0, np.sin(a / 2), np.cos(a / 2), 0, 0, 0])
            with contextlib.redirect_stdout(io.StringIO()):
                report = synchronize(folder, folder / 'gt.csv', folder / 'sync', max_offset=2)
            self.assertAlmostEqual(report['time_offset_seconds'] + origin, lag, delta=.03)
            self.assertGreater(report['correlation'], .99)
            self.assertFalse(report['search_boundary'])
            self.assertLess(report['gt_signal_valid_fraction'], 1.)


if __name__ == '__main__':
    unittest.main()
