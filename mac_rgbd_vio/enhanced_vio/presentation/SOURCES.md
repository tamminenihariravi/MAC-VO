# Data and illustration sources

## Primary project evidence

- Original walking data: `enhanced_vio/results/capture_20261029_201116/`.
- OptiTrack export: `C:/Users/Hari/Downloads/Take 2026-10-29 08.11.21 PM.csv`.
- CPU revision replay: `enhanced_vio/results/walking_201116/final_replay/`.
- Paired metrics and aligned arrays: `enhanced_vio/results/walking_201116/comparison/`.
- Independent raw gyro synchronization: `enhanced_vio/results/walking_201116/imu_sync/`.
- Scope and limitations: `enhanced_vio/results/walking_201116/REPORT.md`.
- Hardware photo: existing `mac_rgbd_vio/presentation/figures/rgbd_cam_markers.jpg`, copied unchanged.

`data/source_metrics.json` preserves exact metric values, source paths, alignment transforms, synchronization details and evaluation coverage. Per-slide `\src` notes in `main.tex` identify the applicable source.

## External technical references

- RealSense sensor coordinates, timing and calibration: https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md
- Timestamp-based gyro integration example: https://github.com/realsenseai/librealsense/blob/master/examples/motion/readme.md
- RGB-D correspondence uncertainty context: https://isprs-annals.copernicus.org/articles/II-5-W2/127/2013/

## Generated explanatory assets

`figures/pipeline.png` and `figures/keyframes.png` were generated with the built-in imagegen tool. They are concept diagrams, not photographs or measurements. The exact prompts are in `figures/illustration_prompts.md`. No plot data or reported numerical result comes from image generation.
