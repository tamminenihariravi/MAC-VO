"""Bounded background RGB-D recording; disk pressure never blocks pose estimation."""
import csv
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np


class RawRecorder:
    def __init__(self, output, capacity=64, workers=2):
        if capacity < 1 or workers < 1:
            raise ValueError('Recorder capacity and worker count must be positive')
        self.output = Path(output)
        (self.output / 'color').mkdir(exist_ok=True)
        (self.output / 'depth').mkdir(exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.rows, self.errors, self.write_ms = [], [], []
        self.accepted = self.dropped = self.peak_queue = 0
        self.closed = False
        self.threads = [threading.Thread(target=self._worker, name=f'rgbd-writer-{i}') for i in range(workers)]
        for thread in self.threads:
            thread.start()

    def submit(self, index, timestamp, color, depth):
        if self.closed:
            raise RuntimeError('Recorder already closed')
        # Own the buffers: HUD overlays and SDK frame reuse must not alter raw files.
        item = (index, timestamp, color.copy(), depth.copy())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            return False
        self.accepted += 1
        self.peak_queue = max(self.peak_queue, self.queue.qsize())
        return True

    def _save(self, index, timestamp, color, depth):
        name = f'{index:06d}'
        color_name, depth_name = f'color/{name}.png', f'depth/{name}.npy'
        if not cv2.imwrite(str(self.output / color_name), color, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f'Could not save {color_name}')
        np.save(self.output / depth_name, depth)
        return (index, timestamp, color_name, depth_name)

    def _worker(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                tick = time.perf_counter()
                try:
                    row = self._save(*item)
                    with self.lock:
                        self.rows.append(row)
                        self.write_ms.append((time.perf_counter() - tick) * 1000)
                except Exception as exc:
                    with self.lock:
                        self.errors.append(dict(frame=int(item[0]), error=str(exc)))
            finally:
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            # Drain every accepted frame; sentinel insertion may wait only on shutdown.
            for _ in self.threads:
                self.queue.put(None)
            self.queue.join()
            for thread in self.threads:
                thread.join()
            with (self.output / 'frames.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'color', 'depth'])
                writer.writerows(row[1:] for row in sorted(self.rows))
        return dict(accepted_frames=self.accepted, saved_frames=len(self.rows),
                    dropped_frames=self.dropped, failed_frames=len(self.errors), errors=self.errors,
                    complete=self.dropped == 0 and not self.errors,
                    peak_queue_frames=self.peak_queue, capacity_frames=self.queue.maxsize,
                    workers=len(self.threads), png_compression=1,
                    frame_write_mean_ms=float(np.mean(self.write_ms)) if self.write_ms else None,
                    frame_write_p95_ms=float(np.percentile(self.write_ms, 95)) if self.write_ms else None)
