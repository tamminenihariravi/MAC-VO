# Validation record — 29 October 2026

## Concurrent walking / OptiTrack validation

The later capture `capture_20261029_201116` was evaluated against `Take 2026-10-29 08.11.21 PM.csv`. Independent raw-gyro synchronization gives correlation 0.912. The recorded ATE is 0.420 m and the CPU revision replay ATE is 0.308 m on the same 902 valid matched poses. The core replay estimator averages 18.17 ms with p95 24.54 ms. However, the common-time estimated path increases from 25.67 m to 27.11 m against 14.32 m GT, and tracking failures increase from 58 to 66. Accuracy targets are not collectively met. GT dropouts limit coverage to 63.75% of estimated poses.

Twenty tests pass. A short updated stationary live recording/HUD check reaches 29.87 FPS, with p95 frame-dequeue-to-pose 28.94 ms and five acquisition drops. This is not a 30 Hz walking validation. See [the complete walking report](results/walking_201116/REPORT.md) and its plots for protocols, outputs, code changes and limits. Earlier measurements below are retained as history.

## Recording throughput correction

The user's longer `capture_20261029_194137` exposed synchronous recording overhead: 965 poses, 24.41 FPS, 245 camera-processing frame drops, and 55.18 ms frame-dequeue-to-pose p95, despite core estimation averaging 12.16 ms. The earlier short stationary check did not establish sustained recording performance.

PNG/depth writes now use a bounded background writer pool with lossless low-compression PNGs, owned buffers, chronological manifests and explicit recording overload/error accounting. A fresh **15-second hardware run with the HUD and raw recording enabled** produced:

| Measurement | Updated result |
|---|---|
| Tracking throughput | 30.04 FPS; 29.98 Hz sensor cadence |
| Camera-processing drops / tracking failures | 0 / 0 |
| Core estimator mean / p95 | 9.32 / 10.88 ms |
| Frame-dequeue-to-pose p95 | 16.35 ms |
| SDK-callback-to-pose p95 | 16.48 ms |
| Recording enqueue mean | 0.46 ms |
| HUD display mean | 8.83 ms |
| Raw frames accepted / successfully saved | 438 / 438, including startup |
| Recording drops / failed writes | 0 / 0 |
| Maximum recording queue backlog | 1 frame |
| Test suite | 18 passed |

Evidence is in `results/async_recording_hud_check/timing.json`, `recording.json` and `results/tests_recording_fix.txt`. Replay of the new background recording reproduced all 363 live poses exactly, with zero serialized pose difference (`results/async_recording_verification.json`). The 21 original source files remain unchanged. This was a stationary hardware run; moving-scene and sustained disk-load performance remain workload dependent. The three added tests exercise independent raw buffers, sorted manifest order, queue saturation without blocking tracking, final draining, and explicit failed-write reporting. Older validation results below are retained as historical measurements.

## Results established in this workspace

| Check | Result | Evidence |
|---|---|---|
| Numerical / integration tests | 15 passed | `results/tests.txt` |
| Original source isolation | 21 original files verified by SHA-256; zero changed | `results/verification.json`, `results/original_source_hashes.json` |
| Live stationary RealSense D435IF run | 286 estimated poses; startup calibration succeeded | `results/hardware_final/metadata.json` |
| Measured throughput | 30.07 frames/s wall throughput; sensor pose cadence 29.98 Hz | `results/hardware_final/timing.json` |
| Core CPU estimator latency | Mean 10.81 ms, p95 12.30 ms | Same timing report |
| Frame dequeue to pose, including SDK alignment | Mean 14.99 ms, p95 16.51 ms | Same timing report |
| Frame drops / tracking failures | 0 / 0 | Same timing report |
| GPU use | No GPU operations in estimator; GPU memory was not externally profiled | Implementation and dependencies |
| Stationary behavior | 283 stationary decisions after initialization and two confirmation frames | `results/hardware_final/diagnostics.json` |
| Stationary accumulated path | 0.000935 m; Y displacement 0.000562 m | `results/hardware_final/trajectory_enhanced.txt` |
| Raw live/replay consistency | 164 matching poses; maximum serialized pose difference exactly 0 | `results/verification.json` |
| Existing 135-second baseline ATE | 1.6147 m on accepted GT intervals; exploratory synchronization | `results/baseline_135s/benchmark_metrics.json` |

The final live check ran for 12 seconds including stationary startup. It was a stationary hardware check, not the requested 135-second walking trial. Timing excludes sensor exposure, USB transport, queue residence and display; separate estimator and frame-dequeue-to-pose timing prevents confusing frame cadence with processing latency. The wall throughput is slightly above the sensor cadence because it counts the first pose; both are reported. Walking scenes with outliers or failures may cost more. None of these checks establish the 135-second accuracy targets or worst-case latency.

The raw recording check also reached approximately 30 frames/s, with mean estimator latency 10.89 ms and frame-dequeue-to-pose p95 28.78 ms including synchronous raw file writes. It retains aligned images, metric depth, original gyro packets, accelerometer packets and calibration metadata in `results/raw_smoke`. `results/replay_verified` is its verified replay. The earlier `hardware_smoke` and `replay_smoke` folders are development diagnostics; use `hardware_final` and `replay_verified` for current results.

## Numerical coverage

The tests check the minimum calibration duration and sample count; rejection of moving startup windows and duplicate gyro timestamps; stationary-only online bias updates; endpoint-bounded integration and gyro-to-color rotation; ZUPT confirmation, insufficient tracks and opposing gyro samples; exact pose recovery and transform direction; fixed-rotation translation refitting under anisotropic covariance; gross outlier rejection and degenerate geometry; all three keyframe triggers; stationary zero-step behavior; a moving synthetic RGB-D sequence with multiple reference insertions; stopping between reference insertions without snapping backwards; scale-preserving rigid alignment; missing-GT gap rejection; clock-offset sign; near-pi SO(3) maps; Motive unit conversion; and the real benchmark/plotting code with both trajectories restricted to identical timestamps.

## What remains unmeasured

| Requested result | Current status |
|---|---|
| Enhanced ATE below 0.45 m | Requires matching raw 135-second data or a new paired capture |
| Enhanced path 34.5 ± 2 m | Requires complete walking replay/capture |
| Enhanced vertical error below 0.25 m over 135 s | Requires paired walking evaluation and a measured marker-to-camera lever arm |
| Three actual GT/baseline/enhanced curves | Plot generator implemented; actual enhanced old-take curve unavailable |
| Sustained performance during walking | Stationary hardware and synthetic moving checks only |
| RA-L publication claims | Requires independent synchronization, calibration split, held-out trials and ablations |

The requested PNG exists at `results/baseline_135s/enhanced_vs_baseline_optitrack.png`. It explicitly marks enhanced ATE as unmeasured. Supplying `--enhanced` with a trajectory estimated from the matching original raw data produces the green curve and measured target checks automatically. A new stationary hardware trajectory is not compared with the old walking ground truth.

## Ground-truth and protocol findings

Only 9,725 of 16,495 Motive rows have valid rigid-body poses. There are 103 gaps longer than the 0.1-second interpolation limit, with a largest gap of 2.225 seconds. Automatic angular-speed alignment selects +4.53 seconds, but its correlation is only 0.218. Treat the default score as exploratory until synchronization is independently verified. The common-time subset includes 2,084/3,046 estimated poses, spanning 132.356 seconds with gaps.

The full supplied baseline has native path 64.7826 m and raw camera-Y displacement 3.276881 m. The valid native GT polyline is 34.9701 m, with missing segments bridged. These differ from the historical 62.79 m / 34.48 m values in the request. The evaluator exposes its exact crop, interpolation policy, native paths and common-time paths, instead of silently modifying trajectories to reproduce old figures. Its baseline ATE of 1.6147 m is close to the stated 1.616 m, but the protocol and coverage limitations still matter.
