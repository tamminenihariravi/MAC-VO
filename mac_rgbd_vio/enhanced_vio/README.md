# Enhanced MAC RGB-D / gyro odometry

All implementation, configuration, tests, captures and reports are contained in this directory. The original `mac_rgbd_vio/src` and `scripts` are untouched. This is an independent CPU implementation using NumPy and OpenCV; it does not import Torch, CUDA, the original pipeline, or SciPy.

Implemented components: stationary startup calibration and online gyro bias tracking; visual/gyro ZUPT gating; persistent reference keyframes; robust visual rotation registration followed by 75% visual / 25% gyro tangent-space fusion; translation refitting with calibrated full metric covariance; live HUD and X-Z radar; raw recording and replay; and a rigid SE(3) OptiTrack evaluator.

The original 135-second take's requested accuracy targets remain **unverified** because its matching raw RGB-D/IMU recording was not supplied. The later 20:11 walking capture now has a concurrent OptiTrack evaluation: recorded ATE 0.420 m and revised replay ATE 0.308 m on valid common timestamps, with path inflation and tracking loss still unresolved. See [the walking report](results/walking_201116/REPORT.md). No ground-truth-assisted trajectory correction, endpoint closure, assumed constant height or smoothing of the baseline is presented as enhanced odometry.

## Run

From `D:\Post_doc\mac\MAC-VO\mac_rgbd_vio` in PowerShell, using the existing Python environment:

```powershell
python enhanced_vio/scripts/enhanced_live_vio.py --record-raw
```

Hold the camera still until startup calibration completes. The HUD opens after stream startup; press **q** to stop. Output defaults to a new capture directory under `enhanced_vio/results`. A 2-second window needs at least 400 gyro samples and all per-axis sample standard deviations below 0.01 rad/s. Exactly 400 samples at exactly 200 Hz span 1.995 seconds, so 401 samples are normally required. Visible motion or a failed gyro stationarity check restarts the window. A constant rotation with mean gyro norm above 0.025 rad/s is also rejected. Gyro statistics alone cannot distinguish a very slow constant turn from bias; the live startup additionally checks optical flow.

```powershell
# Short automated check, including startup; no windows.
python enhanced_vio/scripts/enhanced_live_vio.py --headless --duration 12

# Reprocess recorded aligned RGB-D and original gyro packets.
python enhanced_vio/scripts/enhanced_live_vio.py --headless --replay enhanced_vio/results/<capture> --output enhanced_vio/results/<replay>

# Re-evaluate supplied baseline. Produces JSON, NPZ and three-panel PNG.
python enhanced_vio/scripts/benchmark_135s_run.py

# A genuine comparison after replaying the MATCHING original raw recording.
python enhanced_vio/scripts/benchmark_135s_run.py --enhanced enhanced_vio/results/<replay>/trajectory_enhanced.txt

python -m unittest discover -s enhanced_vio/tests -v
```

The optional `--record-sdk` also saves the native SDK stream. This machine's installed SDK requires `.db3`; older SDKs can use `--sdk-extension bag`. The portable `--record-raw` recording is independent of that SDK container and contains `color/*.png`, metric `depth/*.npy`, `frames.csv`, `gyro.csv`, `accel.csv` and `metadata.json`. Raw recording adds disk I/O; it is disabled by default. Replay requires the recorded stationary startup interval. All CLI output directories must resolve inside `enhanced_vio`.

Raw image/depth writes run on two background workers with lossless PNG compression level 1. The tracking thread only copies the raw arrays into a bounded queue (64 frames by default, approximately 140 MB at 640×480). HUD drawing cannot change the recorded buffers. Accepted writes are drained before exit, and the manifest is sorted into capture order. Use `--record-workers` and `--record-queue` to adjust the writer pool and memory limit. If the disk stays overloaded, raw recording drops are counted separately from camera processing drops; `recording.json` and `timing.json` set `raw_recording.complete` to false and list write errors. A larger queue absorbs spikes but cannot fix sustained insufficient disk throughput. The normal `--record-raw` command uses this background writer automatically.

Timing now also reports display time, recording enqueue time, acquisition queue residence and SDK-callback-to-pose latency. The SDK callback occurs after sensor transport; this is not exposure-to-pose latency. Shutdown disk flushing is excluded from tracking FPS and reported separately.

Walking defaults use 300 features, four OpenCV threads, a 15 Hz HUD and one queued SDK frame. Pose estimation still targets the 30 Hz sensor stream. Adjust `--max-features`, `--opencv-threads`, `--hud-fps` and `--frame-queue` explicitly when needed. Raw recording completeness now includes acquisition drops as well as writer overload/failures. Use the separate `saved_all_submitted_frames` field to check only the writer.

## Estimation conventions

`R_kc` maps current camera coordinates to reference keyframe coordinates:

`P_k = R_kc P_c + t_kc`.

Camera poses map camera coordinates to the initial camera world. Thus `R_wc = R_wk R_kc` and `p_wc = p_wk + R_wk t_kc`. Every current pose is anchored to the reference pose; keyframe-relative translations are never accumulated frame after frame. Features are detected only at reference insertion/reacquisition, tracked directly against the reference with forward/backward LK checks, and paired with valid depth in both images. Depth discontinuities and large forward/backward residuals are rejected. Registration uses deterministic RANSAC and Huber-weighted refinement; collinear geometry and ill-conditioned normal equations fail explicitly.

Keyframes are inserted only when disparity exceeds 3.5 px, relative rotation exceeds 2.5 degrees, or geometric inliers fall below 65% of original reference features. Tracking failure is logged as `lost:<reason>` and reacquires at the last reliable pose. Motion during a loss is unknown; it is not reconstructed. Constant-height and loop-closure assumptions are absent.

Each gyro packet has its own sensor timestamp. Only gyro packets drive orientation integration; accelerometer timestamps never create zero-gyro steps. SDK gyro-to-color extrinsics are applied, interpreting the SDK rotation array as column-major. The integrator interpolates image interval boundaries, checks coverage and 30 ms packet gaps, and composes midpoint SO(3) increments. The incremental result is accumulated from the keyframe in constant memory. An IMU gap disables gyro fusion until the next reference; it never supplies a fabricated stationary measurement. Bias changes apply to subsequent intervals.

Stationarity requires median **previous-frame to current-frame** valid flow below 0.35 px and the maximum corrected gyro norm in that frame interval below 0.025 rad/s, at least 20 tracks, and three consecutive qualifying frames. Stationary frames have an exactly zero translation increment and zero velocity. The existing keyframe-to-current translation is retained, avoiding a jump back to the reference. Stationary gyro samples update bias with `alpha = 1 - exp(-dt/60 s)`; samples failing the variance check are excluded.

This is RGB-D metric odometry aided by gyro rotation and a zero-displacement/velocity gate, not a fully observable tightly coupled inertial navigation filter. Accelerometer packets are recorded for future validation, but are not integrated into unobservable position/velocity states. Initial world Y is camera Y, not gravity-aligned vertical; physical vertical accuracy is evaluated along OptiTrack Y after rigid alignment. Velocity is a visual finite difference, zeroed by the stationary gate; no inertial velocity covariance is claimed.

## Two-stage optimization

Stage 1 estimates visual `R_vis, t_vis` using robust 3D correspondence registration. Both visual and gyro rotations refer to the same reference-to-current interval, in the same camera coordinates. Fuse

`R* = Exp(0.75 Log(R_vis) + 0.25 Log(R_imu))`.

This specified local tangent-space rule is appropriate for small relative rotations. It is a fixed heuristic weight, not a statistically optimal covariance fusion or an observability guarantee.

Stage 2 fixes `R*`, forms `d_i = P_ki - R* P_ci`, and uses

`Sigma_ri = Sigma_ki + R* Sigma_ci R*^T`, `Omega_i = Sigma_ri^-1`.

For fixed Huber IRLS weights `w_i`, solve the exact 3-by-3 conditional normal equation

`t = (sum_i w_i Omega_i)^-1 sum_i w_i Omega_i d_i`.

For the Huber objective, recompute weights from `sqrt((d_i-t)^T Omega_i (d_i-t))` and repeat until convergence. There is no one-shot closed form for arbitrary robust `rho`. Translation is always refit after rotation fusion. The covariance engine and YAML parameters were copied from the original implementation; its batch factorization is vectorized locally. These fitted parameters are inherited, not independently established as calibrated on held-out data in this task.

## Evaluation protocol and data caveats

The evaluator parses the named Motive rigid body and XYZ/XYZW fields from its multirow header, honors length units, removes invalid samples, and refuses ambiguous multiple bodies. It estimates one bounded clock shift using angular-speed correlation, interpolates positions only over valid GT intervals up to 0.1 s, then applies Umeyama rigid alignment with **scale fixed to one**. Use `--time-offset` for an independently known offset; `--enhanced-time-offset` is separate when required. Same-run replay normally shares baseline timestamps. A new live capture must be compared with its own concurrent OptiTrack take, not the old office walk.

For an enhanced comparison, both trajectories are evaluated on the same baseline timestamp subset. Each receives its own rigid alignment to the same GT points. The default 135-second path-length target is only meaningful for the complete matching take; inspect reported overlap before interpreting it. Native path lengths and common-time sampled lengths are reported separately. Common-time paths bridge removed intervals and can undercount noisy native paths. The PNG breaks lines across large missing intervals. No interpolation extrapolation, scale fitting or GT position-error search is used to improve scores.

The Motive file has 16,495 rows but only **9,725 valid rigid-body poses**, 103 gaps over 0.1 s and a largest gap of **2.225 s**. Default automatic clock shift is **+4.53 s**, with angular-speed correlation only **0.218**. This is low confidence and needs a hardware synchronization event or independently measured offset before publication. The exploratory default comparison uses 2,084 poses (68.4% of baseline) and gives baseline ATE **1.6147 m**. Its floor-projected ATE is **1.5271 m**; independent SE(2) floor alignment gives **1.8209 m**. These are different metrics and cannot be compared interchangeably.

The supplied full baseline has native path length **64.7826 m** and raw Y displacement **3.276881 m**. Native GT path length over valid samples is **34.9701 m**. These values differ from the request's historical path lengths because sampling, cropping, synchronization and gap handling differ. They are reported directly instead of forced to match. Normalized ATE divided by GT path length is not endpoint drift.

`--marker-offset X Y Z` applies a measured marker-body-to-optical-center lever arm in meters using the GT rigid-body quaternion. Without it, marker-origin versus optical-center error remains. No hand-eye parameters are fitted against this evaluation trajectory. Accuracy targets and existing covariance calibration need held-out walking trials and ablations before RA-L claims.

See [VALIDATION.md](VALIDATION.md) for measured checks and [results/baseline_135s/benchmark_metrics.json](results/baseline_135s/benchmark_metrics.json) for the complete benchmark.

## Sources

The RealSense [D435i sensor documentation](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md) describes sensor timestamping, coordinates, IMU calibration behavior and extrinsics. The SDK [motion example](https://github.com/realsenseai/librealsense/blob/master/examples/motion/readme.md) illustrates timestamp-based angular-rate integration. The RGB-D covariance approach is consistent with propagation of measurement uncertainty for point registration; see [Generation and weighting of 3D point correspondences for improved registration of RGB-D data](https://isprs-annals.copernicus.org/articles/II-5-W2/127/2013/). These sources support implementation choices, not the unmeasured accuracy targets.
