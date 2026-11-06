"""Exercise the real three-trajectory evaluation and Motive header parsing."""
import contextlib
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_135s_run import benchmark
from src.evaluation import load_motive
from src.lie import so3_exp


class EvaluationIntegrationTests(unittest.TestCase):
    def test_motive_units_and_comparison_share_timestamps(self):
        with tempfile.TemporaryDirectory(prefix='evaluator_test_', dir=ROOT / 'results') as temporary:
            folder = Path(temporary)
            times = np.arange(60) * .05
            p = np.column_stack((np.sin(times), .1 * np.sin(times * 2), np.cos(times)))
            q = np.tile([0., 0., 0., 1.], (len(times), 1))
            est = p @ so3_exp([.2, -.1, .05]).T + [1, 2, 3]
            np.savetxt(folder / 'baseline.txt', np.column_stack((times, est, q)))
            # Enhanced has one second less coverage; both must be cropped identically.
            np.savetxt(folder / 'enhanced.txt', np.column_stack((times[10:-10], est[10:-10], q[10:-10])))
            with (folder / 'gt.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Format Version', '1.22', 'Length Units', 'Millimeters'])
                writer.writerow([])
                writer.writerow(['', '', *(['Rigid Body'] * 7)])
                writer.writerow(['', '', *(['Camera'] * 7)])
                writer.writerow(['', '', *(['id'] * 7)])
                writer.writerow(['', '', *(['Rotation'] * 4), *(['Position'] * 3)])
                writer.writerow(['Frame', 'Time (Seconds)', 'X', 'Y', 'Z', 'W', 'X', 'Y', 'Z'])
                for i, t in enumerate(times):
                    writer.writerow([i, t, *q[i], *(p[i] * 1000)])
            gt = load_motive(folder / 'gt.csv', 'Camera')
            np.testing.assert_allclose(gt.positions, p)
            args = SimpleNamespace(output=folder / 'out', baseline=folder / 'baseline.txt', gt=folder / 'gt.csv',
                                   rigid_body='Camera', marker_offset=None, time_offset=0., max_offset=5.,
                                   enhanced=folder / 'enhanced.txt', enhanced_time_offset=0., max_gap=.1)
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark(args)
            result = json.loads((folder / 'out/benchmark_metrics.json').read_text())
            self.assertEqual(result['evaluated_poses'], 40)
            self.assertLess(result['baseline']['ate_rmse_m'], 1e-12)
            self.assertLess(result['enhanced']['ate_rmse_m'], 1e-12)
            self.assertEqual(result['baseline']['alignment_scale'], 1.)
            self.assertTrue((folder / 'out/enhanced_vs_baseline_optitrack.png').exists())


if __name__ == '__main__':
    unittest.main()
