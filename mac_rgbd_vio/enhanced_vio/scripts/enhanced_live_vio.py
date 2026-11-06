"""Standalone RealSense runner, raw recording and deterministic raw-stream replay."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imu_calibrator import IMUCalibrator
from src.gyro_preintegration import GyroBuffer
from src.pipeline import EnhancedVIO
from src.lie import quaternion
from src.paths import output_path
from src.raw_recorder import RawRecorder


def covariance_config():
    cfg = yaml.safe_load((ROOT / 'config/optitrack_calibrated_covariance.yaml').read_text())
    return dict(sigma_base_px=cfg['sigma_base_px'], gamma_fb=cfg['gamma_fb'],
                alpha_st=cfg['alpha_structure_tensor'], correlation_rho=cfg['correlation_rho'])


def radar(history, size=480):
    image = np.full((size, size, 3), 25, np.uint8)
    if len(history) > 1:
        points = np.asarray(history)[:, [0, 2]]
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        scale = (size - 60) / max(np.ptp(points, axis=0).max(), 2)
        xy = ((points - center) * scale + size / 2).astype(np.int32)
        xy[:, 1] = size - xy[:, 1]
        cv2.polylines(image, [xy], False, (70, 210, 70), 2)
        cv2.circle(image, tuple(xy[-1]), 5, (0, 200, 255), -1)
    cv2.putText(image, 'Camera world X-Z (m)', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
    return image


def run(args):
    cv2.setNumThreads(args.opencv_threads)
    out = output_path(args.output or ROOT / 'results' / ('capture_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    calibrator = IMUCalibrator()
    image_queue = queue.Queue(maxsize=args.frame_queue)
    lock = threading.Lock()
    gyro_packets, accel_packets = [], []
    dropped = 0
    pipeline = None
    if args.replay:
        replay = Path(args.replay)
        meta = json.loads((replay / 'metadata.json').read_text())
        gyro_data = np.loadtxt(replay / 'gyro.csv', delimiter=',', skiprows=1, ndmin=2)
        frames = list(csv.DictReader((replay / 'frames.csv').open(newline='')))
        intrinsics = meta['intrinsics']
        gyro = GyroBuffer(meta['rotation_color_from_gyro'])
        for row in gyro_data:
            gyro.append(row[0], row[1:4])
        # Re-run startup calibration on original packet timestamps, never use GT.
        calibration_start = meta.get('calibration_start_timestamp', gyro_data[0, 0])
        calibration_end = meta.get('calibration_end_timestamp', float('inf'))
        for row in gyro_data:
            if row[0] < calibration_start or row[0] > calibration_end:
                continue
            if calibrator.add_startup(row[0], row[1:4]):
                break
        if not calibrator.ready:
            raise RuntimeError('Raw recording lacks a stationary startup interval')
        replay_start_timestamp = meta.get('first_pose_timestamp')
        if replay_start_timestamp is None:
            original_trajectory = replay / 'trajectory_enhanced.txt'
            replay_start_timestamp = float(np.loadtxt(original_trajectory, ndmin=2)[0, 0]) if original_trajectory.exists() else calibrator.last_timestamp
        source = iter(frames)
        def next_image():
            row = next(source)
            available_wall = time.perf_counter()
            return float(row['timestamp']), cv2.imread(str(replay / row['color'])), np.load(replay / row['depth']), available_wall, 0.
    else:
        import pyrealsense2 as rs
        pipeline, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        if args.record_sdk:
            cfg.enable_record_to_file(str(out / ('raw_stream.' + args.sdk_extension)))
        def callback(frame):
            nonlocal dropped
            if frame.is_motion_frame():
                m = frame.as_motion_frame().get_motion_data()
                packet = (frame.get_timestamp() * 0.001, np.array([m.x, m.y, m.z]))
                with lock:
                    (gyro_packets if frame.get_profile().stream_type() == rs.stream.gyro else accel_packets).append(packet)
            elif frame.is_frameset():
                if image_queue.full():
                    try:
                        image_queue.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        pass
                image_queue.put_nowait((frame.as_frameset(), time.perf_counter()))
        profile = pipeline.start(cfg, callback)
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        intrinsics = dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy)
        ex = profile.get_stream(rs.stream.gyro).get_extrinsics_to(color_profile)
        # rs2_extrinsics.rotation is column-major.
        gyro = GyroBuffer(np.array(ex.rotation).reshape(3, 3, order='F'))
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        def next_image():
            f, arrival_wall = image_queue.get(timeout=5)
            available_wall = time.perf_counter()
            aligned = align.process(f)
            c, d = aligned.get_color_frame(), aligned.get_depth_frame()
            if not c or not d:
                raise RuntimeError('RGB-D frame missing after alignment')
            return c.get_timestamp() * .001, np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).astype(np.float32) * depth_scale, available_wall, (available_wall - arrival_wall) * 1000
    metadata = dict(intrinsics=intrinsics, rotation_color_from_gyro=gyro.rotation.tolist(),
                    clock='RealSense sensor timestamp seconds', world='initial color optical frame: X right, Y down, Z forward',
                    method='keyframe RGB-D + debiased gyro rotation; no accelerometer position integration',
                    covariance_provenance='copied existing fitted parameters; independent held-out validation required')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    raw_recorder = RawRecorder(out, args.record_queue, args.record_workers) if args.record_raw else None
    vio, history, latency, total_latency, diagnostics = None, [], [], [], []
    pipeline_latency = []
    trajectory, raw_gyro, raw_accel = [], [], []
    display_latency, enqueue_latency, queue_latency = [], [], []
    start = time.perf_counter()
    first_pose_wall = None
    startup_previous = None
    visual_still = False
    frame_idx = 0
    next_display_wall = 0.
    error = None
    print('Keep camera stationary for 2 seconds (>=400 gyro samples).', flush=True)
    try:
        while True:
            if args.duration and time.perf_counter() - start >= args.duration:
                break
            if args.max_frames and len(trajectory) >= args.max_frames:
                break
            tick = time.perf_counter()
            try:
                timestamp, color, depth, available_wall, queue_residence_ms = next_image()
            except StopIteration:
                break
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            if args.replay and timestamp < replay_start_timestamp - 1e-7:
                continue
            if not args.replay:
                # Allow a gyro packet beyond this image timestamp to arrive.
                deadline = time.perf_counter() + 0.025
                while time.perf_counter() < deadline:
                    with lock:
                        covered = bool(gyro_packets and gyro_packets[-1][0] >= timestamp)
                    if covered:
                        break
                    time.sleep(.001)
                if not calibrator.ready:
                    visual_still = False
                    if startup_previous is not None:
                        pp = cv2.goodFeaturesToTrack(startup_previous, 100, .02, 10)
                        if pp is not None and len(pp) >= 20:
                            pn, st, _ = cv2.calcOpticalFlowPyrLK(startup_previous, gray, pp, None)
                            good = st.ravel().astype(bool)
                            visual_still = good.sum() >= 20 and np.median(np.linalg.norm((pn - pp).reshape(-1, 2)[good], axis=1)) < .35
                    startup_previous = gray.copy()
                with lock:
                    packets, accels = gyro_packets[:], accel_packets[:]
                    gyro_packets.clear()
                    accel_packets.clear()
                raw_gyro.extend((t, *w) for t, w in packets)
                raw_accel.extend((t, *a) for t, a in accels)
                for t, w in packets:
                    gyro.append(t, w)
                    if not calibrator.ready:
                        calibrator.add_startup(t, w, visual_still)
            if args.record_raw:
                enqueue_start = time.perf_counter()
                raw_recorder.submit(frame_idx, timestamp, color, depth)
                enqueue_latency.append((time.perf_counter() - enqueue_start) * 1000)
            frame_idx += 1
            if not calibrator.ready:
                if not args.headless:
                    cv2.putText(color, f'Hold still: {len(calibrator.samples)}/400 samples', (15, 30), 0, .7, (0, 220, 255), 2)
                    cv2.imshow('Enhanced RGB-D VIO', color)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
                continue
            if vio is None:
                vio = EnhancedVIO(intrinsics, calibrator, gyro, covariance_config(), max_features=args.max_features)
                first_pose_wall = time.perf_counter()
                metadata['startup_bias_rad_s'] = calibrator.bias.tolist()
                metadata['startup_std_rad_s'] = calibrator.std.tolist()
                metadata['calibration_start_timestamp'] = calibrator.samples[0][0]
                metadata['calibration_end_timestamp'] = calibrator.last_timestamp
                metadata['first_pose_timestamp'] = timestamp
                print('Calibrated gyro bias [rad/s]:', calibrator.bias, flush=True)
            processing_start = time.perf_counter()
            result = vio.process(gray, depth, timestamp)
            elapsed = (time.perf_counter() - processing_start) * 1000
            pipeline_elapsed = (time.perf_counter() - available_wall) * 1000
            pipeline_latency.append(pipeline_elapsed)
            queue_latency.append(queue_residence_ms)
            latency.append(elapsed)
            trajectory.append((timestamp, *result.position, *quaternion(result.rotation)))
            history.append(result.position)
            diagnostics.append(dict(timestamp=timestamp, status=result.status, inliers=result.inliers,
                                    stationary=result.stationary, disparity_px=result.disparity_px,
                                    keyframe=result.inserted_keyframe, processing_ms=elapsed,
                                    frame_available_to_pose_ms=pipeline_elapsed,
                                    callback_to_pose_ms=queue_residence_ms + pipeline_elapsed))
            display_start = time.perf_counter()
            if not args.headless and display_start >= next_display_wall:
                next_display_wall = display_start + 1. / args.hud_fps
                for p in result.pixels:
                    cv2.circle(color, tuple(np.rint(p).astype(int)), 2, (0, 220, 0), -1)
                cv2.putText(color, f'{result.status} | {result.inliers} inliers | {elapsed:.1f} ms', (10, 30), 0, .6, (0, 255, 255), 2)
                cv2.imshow('Enhanced RGB-D VIO', np.hstack((color, radar(history))))
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
            display_latency.append((time.perf_counter() - display_start) * 1000)
            total_latency.append((time.perf_counter() - tick) * 1000)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        acquisition_end_wall = time.perf_counter()
        if pipeline is not None:
            pipeline.stop()
            with lock:
                raw_gyro.extend((t, *w) for t, w in gyro_packets)
                raw_accel.extend((t, *a) for t, a in accel_packets)
        cv2.destroyAllWindows()
        flush_start = time.perf_counter()
        recording = raw_recorder.close() if raw_recorder is not None else None
        flush_seconds = time.perf_counter() - flush_start
        if recording is not None:
            recording['saved_all_submitted_frames'] = recording['complete']
            recording['acquisition_dropped_frames'] = dropped
            recording['complete'] = recording['complete'] and dropped == 0
            recording['shutdown_flush_seconds'] = flush_seconds
            (out / 'recording.json').write_text(json.dumps(recording, indent=2))
        if trajectory:
            np.savetxt(out / 'trajectory_enhanced.txt', trajectory, fmt='%.9f', header='timestamp tx ty tz qx qy qz qw')
            optitrack_dir = ROOT.parents[1] / 'optitrack_data'
            optitrack_dir.mkdir(parents=True, exist_ok=True)
            live_out = optitrack_dir / 'enhanced_live_trajectory.txt'
            np.savetxt(live_out, trajectory, fmt='%.9f', header='timestamp tx ty tz qx qy qz qw')
            print(f'[ENHANCED VIO] Live trajectory saved directly to: {live_out}', flush=True)
        if raw_gyro:
            np.savetxt(out / 'gyro.csv', raw_gyro, delimiter=',', header='timestamp,gx,gy,gz', comments='')
        if raw_accel:
            np.savetxt(out / 'accel.csv', raw_accel, delimiter=',', header='timestamp,ax,ay,az', comments='')
        (out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        metadata['final_bias_rad_s'] = calibrator.bias.tolist()
        metadata['calibration_rejected_windows'] = calibrator.rejected_windows
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        timing = dict(poses=len(trajectory), calibrated=calibrator.ready, error=error,
                      processing_mean_ms=float(np.mean(latency)) if latency else None,
                      processing_p95_ms=float(np.percentile(latency, 95)) if latency else None,
                      frame_available_to_pose_mean_ms=float(np.mean(pipeline_latency)) if pipeline_latency else None,
                      frame_available_to_pose_p95_ms=float(np.percentile(pipeline_latency, 95)) if pipeline_latency else None,
                      callback_to_pose_p95_ms=float(np.percentile(np.array(queue_latency) + pipeline_latency, 95)) if pipeline_latency and not args.replay else None,
                      queue_residence_p95_ms=float(np.percentile(queue_latency, 95)) if queue_latency and not args.replay else None,
                      recording_enqueue_mean_ms=float(np.mean(enqueue_latency)) if enqueue_latency else None,
                      display_mean_ms=float(np.mean(display_latency)) if display_latency else None,
                      latency_definition='Replay: estimator and file-load-to-pose latency; no live callback timing.' if args.replay else 'Core estimator, frame-dequeue-to-pose, and SDK-callback-to-pose reported separately; sensor exposure/transport and display excluded.',
                      source_mode='replay' if args.replay else 'live',
                      opencv_threads=args.opencv_threads, max_features=args.max_features,
                      hud_fps=args.hud_fps, frame_queue_capacity=args.frame_queue,
                      loop_mean_ms=float(np.mean(total_latency)) if total_latency else None,
                      achieved_fps=len(trajectory) / max(acquisition_end_wall - first_pose_wall, 1e-6) if first_pose_wall else 0,
                      pose_sensor_cadence_hz=(len(trajectory) - 1) / max(trajectory[-1][0] - trajectory[0][0], 1e-6) if len(trajectory) > 1 else 0,
                      dropped_frames=dropped, keyframes=vio.keyframes if vio else 0,
                      tracking_failures=vio.tracking_failures if vio else 0, gpu_used=False,
                      detected_keyframes=vio.tracker.detected_keyframes if vio else 0,
                      reused_keyframes=vio.tracker.reused_keyframes if vio else 0,
                      raw_recording=recording,
                      accuracy='unmeasured without matching ground truth')
        (out / 'timing.json').write_text(json.dumps(timing, indent=2))
        print(json.dumps(timing, indent=2), flush=True)
        if recording is not None and not recording['complete']:
            print('Raw recording incomplete: acquisition drops, recorder overload or write failure; see recording.json.', flush=True)
        print('Outputs:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--duration', type=float, default=0, help='Wall duration including startup; 0 runs until q')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record-raw', action='store_true', help='Save aligned RGB-D/IMU replay files')
    parser.add_argument('--record-queue', type=int, default=64, help='Bounded background recording queue in frames')
    parser.add_argument('--record-workers', type=int, default=2, help='Background PNG/depth writers')
    parser.add_argument('--record-sdk', action='store_true', help='Also record native SDK stream (runtime dependent)')
    parser.add_argument('--sdk-extension', choices=['db3', 'bag'], default='db3', help='Installed SDK requires db3; older SDKs may require bag')
    parser.add_argument('--replay', type=Path, help='Replay an enhanced_vio raw capture directory')
    parser.add_argument('--opencv-threads', type=int, default=4)
    parser.add_argument('--frame-queue', type=int, default=1, help='Live SDK frames retained before processing; 1 keeps the newest frame')
    parser.add_argument('--hud-fps', type=float, default=15, help='Display refresh cap; pose estimation still processes at sensor cadence')
    parser.add_argument('--max-features', type=int, default=300, help='Feature budget for CPU tracking; increase for richer but more expensive registration')
    args = parser.parse_args()
    if args.record_queue < 1 or args.record_workers < 1:
        parser.error('--record-queue and --record-workers must be positive')
    if args.frame_queue < 1 or args.hud_fps <= 0 or args.opencv_threads < 1:
        parser.error('--frame-queue, --hud-fps and --opencv-threads must be positive')
    if args.max_features < 20:
        parser.error('--max-features must be at least 20')
    if args.replay and (args.record_raw or args.record_sdk):
        parser.error('--replay and recording options are mutually exclusive')
    run(args)


if __name__ == '__main__':
    main()
