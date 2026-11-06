# Speaker notes and sources

## Slide 1 — Title

Local enhanced_vio implementation and results/walking_201116/REPORT.md. RGB-D odometry aided by gyro rotation and a stationary gate. No claim of a tightly coupled full inertial navigation filter.

## Slide 2 — Our measurement setup

Hardware photo: mac_rgbd_vio/presentation/figures/rgbd_cam_markers.jpg. Stream configuration: enhanced_live_vio.py. Motive export header: Take 2026-10-29 08.11.21 PM.csv. Accelerometer packets are recorded but do not drive inertial position integration.

## Slide 3 — Why errors build up

Gyro CSV and startup bias from capture_20261029_201116. The 0.25 degrees/s illustration is a theoretical constant-bias calculation, not a measured 135-second tilt from this walking take.

## Slide 4 — Camera motion in one equation

Convention: decoupled_optimizer.py and pipeline.py. RealSense coordinates: https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md . Initial world Y follows camera Y and is not gravity-calibrated vertical.

## Slide 5 — Processing steps

Conceptual illustration generated with imagegen for this presentation. Method: enhanced_vio/src/pipeline.py. Illustration is not an experimental image or numerical evidence.

## Slide 6 — Gyro bias calibration

imu_calibrator.py. At exactly 200 Hz, 400 samples span 1.995 s, so the duration check normally needs at least 401 samples. Startup also rejects visible motion and a large constant gyro mean. Online updates occur only under the stationary gate.

## Slide 7 — Gyro rotation between images

gyro_preintegration.py and pipeline.py. Midpoint rates, image-interval bracketing and actual gyro timestamps. Accelerometer timestamps do not add artificial zero-rate gyro steps.

## Slide 8 — Stationary gating and zero steps

zupt_detector.py and pipeline.py. Flow is previous-frame-to-current-frame displacement, not keyframe disparity. Maximum corrected gyro norm prevents opposing turns from canceling. This is a gate, not a Kalman velocity-state update.

## Slide 9 — Keyframes give a larger visual baseline

keyframe_tracker.py. Generated schematic, not a measured 4x to 8x SNR claim. Thresholds can still insert keyframes frequently. Supported tracks can be reused, with detection when their count falls.

## Slide 10 — Depth turns image pixels into 3D points

calibrated_covariance.py and keyframe_tracker.py. Pinhole unprojection uses aligned RGB-depth and color intrinsics. The pixel example holds depth fixed and is illustrative.

## Slide 11 — Uncertainty sets the weight of each match

Inherited parameters: config/optitrack_calibrated_covariance.yaml. calibrated_covariance.py and DecoupledOptimizer.information. Measurement covariance has mixed pixel-squared and meter-squared units; J converts them. No new held-out covariance calibration was established.

## Slide 12 — Stage 1: rotation fusion

decoupled_optimizer.py. Visual fit uses RANSAC and robust metric refinement. Both rotations map current coordinates into the same keyframe coordinates. This tangent-space blend is a local small-rotation rule.

## Slide 13 — Stage 2: translation with rotation fixed

DecoupledOptimizer.solve_translation. Huber weight decreases with large Mahalanobis residual. Conditional weighted least-squares, not a one-shot robust closed form. Translation is always refitted after rotation fusion.

## Slide 14 — A reference pose anchors the world position

pipeline.py. Camera poses map to initial-camera world. A stationary gate holds world position and zeros finite-difference velocity. Loss holds the last reliable pose, so motion through loss remains unknown.

## Slide 15 — CPU work and recording

Walking live timing: mean 39.7764 ms, p95 53.9299 ms. Final replay: mean 18.1682 ms, p95 24.5360 ms. Revision uses 300 features, four OpenCV threads and a 15 Hz HUD. Original capture lost 1,065 frames, which saved images cannot reconstruct.

## Slide 16 — Clock synchronization with independent gyro data

synchronize_walking_run.py and imu_sync/synchronization.json. Shift from first gyro packet: -4.71 s. Absolute shift: -1793284882.0330672 s because clocks have different origins. Support excludes long GT gaps and implausible orientation flips. GT positions remain unchanged.

## Slide 17 — Accuracy measurement

comparison/benchmark_metrics.json. Separate Umeyama rigid alignment per estimate, same GT times. Only 5,126 of 11,527 rows contain valid rigid-body poses. Marker-to-camera lever arm was not supplied. ATE/path is normalized ATE, not endpoint drift.

## Slide 18 — Walking trajectory on valid reference intervals

comparison/aligned_comparison.npz and benchmark_metrics.json. Same 902 poses, scale one. This is a new take, not the historical 135-second take with 1.616 m ATE. Do not present these as a direct old-take before/after.

## Slide 19 — Path inflation and vertical error remain

comparison/benchmark_metrics.json. Common paths: 14.3232, 25.6703, 27.1066 m. Absolute vertical-error changes: 0.445763, 0.284475 m. Peak-to-peak errors: 1.0574 and 0.9628 m. The previous take's 34.5 m path target does not apply here.

## Slide 20 — What the evidence supports

walking_201116/REPORT.md, tests.txt, verification.json and live_revision_check/timing.json. Twenty-one original source files remain unchanged. Future work needs held-out trials, independent synchronization and ablations. Hardware: https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md . Point covariance context: https://isprs-annals.copernicus.org/articles/II-5-W2/127/2013/ .
