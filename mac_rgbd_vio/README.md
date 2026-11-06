# MAC-RGBD-VIO: Metrics-Aware Covariance for Learning-Based RGB-D Visual-Inertial Odometry

**Target Venue:** IEEE Robotics and Automation Letters (RA-L)  
**Target Hardware:** Intel RealSense D435if (RGB + Hardware Aligned Depth + IMU)  
**Ground Truth Validation:** In-lab OptiTrack Motive Infrared Camera System  

---

## 1. Motivation & Paper Contribution
Standard visual odometry either assumes isotropic noise or utilizes heavy stereo matchers (like MAC-VO) that fail on active RGB-D cameras due to narrow baselines ($50\text{ mm}$) and projector dot patterns. Furthermore, deep pose optimizers (PyPose LM) cause severe VRAM thrashing on edge hardware.

**Key Contributions of MAC-RGBD:**
1. **Analytical 3D Metric Covariance Engine (`src/metric_covariance.py`):**  
   Analytically combines learned 2D optical flow uncertainty $(\sigma_u, \sigma_v)$ with active depth physical noise $\sigma_d(d, \nabla d)$ to construct the full anisotropic $3 \times 3$ spatial covariance $\mathbf{\Sigma}_{3D}$ in closed form.
   * Execution time: **$0.4\text{ ms}$ on CPU**.
2. **Vectorized SE(3) Metric Optimizer (`src/rgbd_optimizer.py`):**  
   Replaces PyPose GPU Equation (4) with a Lie-algebra $\mathfrak{se}(3)$ Gauss-Newton optimizer with Huber weighting.
   * Execution time: **$2.5\text{ ms}$ on CPU for 500 points**. VRAM footprint: **$0\text{ MB}$**.
3. **Hardware Synergy:**  
   Enables active IR speckle projection indoors to dramatically improve depth on plain white walls without corrupting the RGB optical flow stream.

---

## 2. Directory Structure

```
mac_rgbd_vio/
├── config/                  # Camera & VIO parameters
├── data/                    # Recorded ROS 2 bag files (.mcap / .db3)
├── scripts/
│   ├── launch_d435if_rgbd.sh    # Launches realsense2_camera with aligned depth & IMU
│   ├── record_d435if_rgbd.sh    # Records synchronized RGB-D + IMU bag
│   └── sync_to_google_sheet.py  # Webhook sync to Google Sheet tracker
├── src/
│   ├── metric_covariance.py     # Analytical 3x3 Metric Covariance formulation
│   └── rgbd_optimizer.py        # Ultra-fast SE(3) CPU optimizer
└── README.md
```

---

## 3. Quickstart: Recording Data on the Laptop

1. **Launch the Camera:**
   ```bash
   ./scripts/launch_d435if_rgbd.sh
   ```
2. **Record a Test Sequence:**
   ```bash
   ./scripts/record_d435if_rgbd.sh exp101_office_loop
   ```
3. **Log to Google Sheet:**
   ```bash
   python3 scripts/sync_to_google_sheet.py "EXP-102" "2026-10-28" "Laptop" "commit_hash" "RGB-D recording" "RGB+Depth+IMU" "Office" "MAC-RGBD" "Analytical 3D" "-" "-" "Observations" "Next steps"
   ```
