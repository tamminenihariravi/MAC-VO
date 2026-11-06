"""Recording must preserve raw buffers and expose overload/failures without stalling VIO."""
import csv
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.raw_recorder import RawRecorder


class RecorderTests(unittest.TestCase):
    def test_raw_buffers_are_owned_and_manifest_is_ordered(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as temporary:
            recorder = RawRecorder(temporary)
            color = np.full((24, 32, 3), 73, np.uint8)
            depth = np.full((24, 32), 2.123, np.float32)
            try:
                self.assertTrue(recorder.submit(2, .2, color, depth))
                self.assertTrue(recorder.submit(1, .1, color, depth))
                color[:] = 255  # Simulate the live HUD modifying its own buffers.
                depth[:] = 9
            finally:
                result = recorder.close()
            self.assertTrue(result['complete'])
            self.assertEqual(result['saved_frames'], 2)
            np.testing.assert_array_equal(cv2.imread(str(Path(temporary) / 'color/000001.png')), 73)
            np.testing.assert_array_equal(np.load(Path(temporary) / 'depth/000001.npy'), np.float32(2.123))
            with (Path(temporary) / 'frames.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row['color'] for row in rows], ['color/000001.png', 'color/000002.png'])
            self.assertEqual(result, recorder.close())
            with self.assertRaises(RuntimeError):
                recorder.submit(3, .3, color, depth)

    def test_full_recording_queue_returns_without_waiting_for_disk(self):
        started, release = threading.Event(), threading.Event()
        class SlowRecorder(RawRecorder):
            def _save(self, *item):
                started.set()
                if not release.wait(5):
                    raise TimeoutError('Test writer not released')
                return super()._save(*item)
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as temporary:
            recorder = SlowRecorder(temporary, capacity=1, workers=1)
            color, depth = np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32)
            try:
                self.assertTrue(recorder.submit(0, 0., color, depth))
                self.assertTrue(started.wait(2))
                self.assertTrue(recorder.submit(1, .1, color, depth))
                self.assertFalse(recorder.submit(2, .2, color, depth))
                self.assertFalse(release.is_set())
            finally:
                release.set()
                result = recorder.close()
            self.assertEqual(result['saved_frames'], 2)
            self.assertEqual(result['dropped_frames'], 1)
            self.assertFalse(result['complete'])

    def test_failed_write_is_reported_and_excluded_from_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as temporary:
            recorder = RawRecorder(temporary, workers=1)
            with patch('src.raw_recorder.cv2.imwrite', return_value=False):
                recorder.submit(0, 0., np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32))
                result = recorder.close()
            self.assertFalse(result['complete'])
            self.assertEqual(result['failed_frames'], 1)
            self.assertEqual(result['saved_frames'], 0)
            self.assertIn('Could not save', result['errors'][0]['error'])


if __name__ == '__main__':
    unittest.main()
