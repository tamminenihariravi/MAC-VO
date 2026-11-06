"""Measure estimator CPU cost on a cached block of real recorded walking frames."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imu_calibrator import IMUCalibrator
from src.gyro_preintegration import GyroBuffer
from src.pipeline import EnhancedVIO
from src.paths import output_path
from scripts.enhanced_live_vio import covariance_config


def run(args):
    out = output_path(args.output)
    capture = args.capture
    meta = json.loads((capture / 'metadata.json').read_text())
    with (capture / 'frames.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))[args.start:args.start + args.frames]
    frames = [(float(r['timestamp']), cv2.cvtColor(cv2.imread(str(capture / r['color'])), cv2.COLOR_BGR2GRAY), np.load(capture / r['depth'])) for r in rows]
    raw = np.loadtxt(capture / 'gyro.csv', delimiter=',', skiprows=1)
    raw = raw[(raw[:, 0] >= frames[0][0] - .03) & (raw[:, 0] <= frames[-1][0] + .03)]
    reports = []
    for threads in args.threads:
        cv2.setNumThreads(threads)
        calibrator = IMUCalibrator()
        calibrator.ready = True
        calibrator.bias = np.array(meta['startup_bias_rad_s'])
        gyro = GyroBuffer(meta['rotation_color_from_gyro'])
        for row in raw:
            gyro.append(row[0], row[1:4])
        vio = EnhancedVIO(meta['intrinsics'], calibrator, gyro, covariance_config())
        stages = {name: [] for name in ['tracking', 'reference', 'optimization', 'gyro']}
        def measure(obj, name, label):
            original = getattr(obj, name)
            def measured(*a, **kw):
                tick = time.perf_counter()
                result = original(*a, **kw)
                stages[label].append((time.perf_counter() - tick) * 1000)
                return result
            setattr(obj, name, measured)
        measure(vio.tracker, 'track', 'tracking')
        measure(vio.tracker, 'set_reference', 'reference')
        measure(vio.optimizer, 'solve', 'optimization')
        measure(vio.gyro, 'interval', 'gyro')
        timings = []
        for timestamp, gray, depth in frames:
            tick = time.perf_counter()
            vio.process(gray, depth, timestamp)
            timings.append((time.perf_counter() - tick) * 1000)
        use = timings[5:]
        reports.append(dict(opencv_threads=threads, frames=len(frames), start_frame=args.start,
                            estimator_mean_ms=float(np.mean(use)), estimator_p95_ms=float(np.percentile(use, 95)),
                            stage_mean_ms={k: float(np.mean(v)) if v else None for k, v in stages.items()},
                            keyframes=vio.keyframes, tracking_failures=vio.tracking_failures,
                            detected_keyframes=vio.tracker.detected_keyframes, reused_keyframes=vio.tracker.reused_keyframes,
                            limitation='Cached 60-frame walking excerpt; excludes raw file IO, alignment, display and recording.'))
    (out / 'compute_benchmark.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=700)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    run(parser.parse_args())
