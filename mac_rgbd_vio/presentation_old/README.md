# MAC-RGBD-VIO Presentation (LaTeX Beamer)

This directory contains the professional LaTeX Beamer presentation for the **MAC-RGBD-VIO** project, formatted for conference presentations, lab seminars, and IEEE RA-L manuscript defense.

---

## Files Included

* **`mac_rgbd_vio_presentation.tex`**: Complete, standalone 16:9 Beamer presentation containing all mathematical derivations displayed at full width with ample breathing room, architectural diagrams, and experimental benchmarks.
* **`figures/optitrack_covariance_calibration_benchmark.png`**: 4-panel OptiTrack covariance calibration diagnostic benchmark (Error scatter, Gaussian fit, NEES Chi-square CDF, and error vs corner strength).
* **`figures/rgbd_cam_markers.jpg`**: Hardware rig photograph showing the Intel RealSense D435if with rigidly mounted OptiTrack retro-reflective IR marker spheres.
* **`figures/optitrack_benchmark.png`**: 3-panel OptiTrack ground truth benchmark (Top-down X-Z, 3D SE(3), and ATE error profile).
* **`figures/final_ral_benchmark.png`**: 3-way trajectory comparison (Pure VO vs IMU Fusion vs Keyframe-Smoothed VIO).
* **`figures/live_trajectory.png`**: Real-time live camera trajectory plot.
* **`figures/ground_truth_alone.png`**: Standalone OptiTrack ground truth trajectory.

---

## How to Compile to PDF

### Method 1: Local Terminal (TeX Live / MiKTeX)
Run from within this directory (`mac_rgbd_vio/presentation/`):
```bash
pdflatex mac_rgbd_vio_presentation.tex
pdflatex mac_rgbd_vio_presentation.tex
```

### Method 2: Overleaf (Zero Setup)
1. Zip this `presentation/` directory:
   * `mac_rgbd_vio_presentation.tex`
   * `figures/` folder
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select the zip file.
3. Click **Recompile** to get the high-resolution PDF presentation.

---

## Complete Slide Deck Outline (24 Spacious Slides)

1. **Title Slide**: Project Title, Subtitle, IEEE RA-L Post-Doc Initiative.
2. **Motivation**: Why deep learned stereo VO fails on edge robots (3.8 GB VRAM thrashing, active IR speckle conflict).
3. **System Architecture**: High-frequency pipeline flow from D435if to 6-DoF Pose Graph.
4. **Architectural Comparison**: Deep Learning MAC-VO (FlowFormer) vs MAC-RGBD-VIO side-by-side comparison.
5. **Limitations of Heuristic Uncertainty**: Why clipped forward-backward LK error is mathematically flawed (aperture ambiguity, overconfidence, NEES explosion to 120.3).
6. **OptiTrack-Calibrated Covariance Formulation**: Ground-truth relative projection $\hat{\mathbf{p}}_j^{GT}$, structure tensor eigenvalues $\lambda_1, \lambda_2$, and hybrid parametric model.
7. **OptiTrack Covariance Calibration & Validation Benchmark**: Diagnostic 4-panel plot (`optitrack_covariance_calibration_benchmark.png`), NEES reduction from $120.3 \rightarrow 1.87$, and 46.4% VIO ATE error drop.
8. **Active Stereo Depth Uncertainty**: Physics of stereo triangulation ($d = \frac{f B}{D} \implies \sigma_d \propto \frac{\sigma_D}{f B} d^2$), quadratic distance degradation, and depth edge gradient penalty $\beta \|\nabla d\|$.
9. **Analytical Metric Covariance Engine (Part 1)**: Pinhole unprojection $\mathbf{P}(u,v,d)$, Point Jacobian $J_{\mathbf{P}}$, full measurement matrix $\mathbf{\Sigma}_{\text{meas}}$ with correlation $\sigma_{uv}$, and Cholesky solve.
10. **Analytical Metric Covariance Engine (Part 2)**: Full expanded $3\times 3$ matrix display with $\sigma_{uv}$, and two-point correspondence uncertainty propagation $\mathbf{\Sigma}_{\mathbf{r}, k} = \mathbf{\Sigma}_{3D, c, k} + \mathbf{R} \mathbf{\Sigma}_{3D, p, k} \mathbf{R}^T$.
11. **Decoupled Lie-Algebra Optimizer (Part 1)**: $SE(3)$ parameterization, $\mathfrak{se}(3)$ twist vector $\boldsymbol{\xi}$, 3D geometric residuals $\mathbf{r}_k$, and robust Huber cost.
12. **Decoupled Lie-Algebra Optimizer (Part 2)**: Analytical manifold Jacobian $J_k$, closed-form Levenberg-Marquardt normal equations ($<2.5\text{ ms}$ on CPU), and Lie exponential update.
13. **Unified IMU Preintegration on $SO(3)$**: Continuous-time kinematics, Forster preintegration $\Delta \mathbf{R}_{ij}$, first-order gyro bias updates, and high-rate gyro heading guidance $\boldsymbol{\omega}_{\text{fused}}$.
14. **Sliding-Window Keyframe Smoother**: Keyframe selection criteria and joint multi-factor sliding window objective ($N=7$).
15. **Exact Schur-Complement Marginalization**: Block partition of normal equations and exact prior update ($H_{\text{prior}} = H_{RR} - H_{R0} H_{00}^{-1} H_{0R}$).
16. **6-DoF Pose Graph Optimization**: $SE(3)$ edge optimization and continuous Lie manifold geodesic frame interpolation $\mathbf{T}_{\text{corr}}(f)$.
17. **Experimental Hardware Rig & OptiTrack**: Embedded photograph `rgbd_cam_markers.jpg`, RealSense D435if specifications, 200 Hz IMU, and 120 Hz OptiTrack MoCap.
18. **Ground Truth Calibration**: Temporal clock sync cross-correlation ($\tau^*$) and closed-form Umeyama $SE(3)$ SVD spatial rigid alignment.
19. **Hand-Eye Calibration & Error Metrics**: Rigid hand-eye offset modeling $\mathbf{t}_{\text{hand-eye}}$ and standard IEEE RA-L error formulations (ATE RMSE \& RPE).
20. **Closed-Loop Parameter Autotuner**: Discovery of tracking velocity spikes ($14.91\text{ m/s}$), kinematic velocity gating ($v_{\max} = 1.16\text{ m/s}$), and Nelder-Mead optimization against OptiTrack.
21. **Real-World OptiTrack Experimental Benchmark**: Full table and 3-panel figure showing $51.57\text{ cm}$ ATE RMSE and $0.0\text{ MB}$ GPU VRAM.
22. **Scientific Positioning for IEEE RA-L**: Novelty vs. classical systems (ORB-SLAM3, VINS-Fusion) and the 4 core pillars of contribution.
23. **Literature Benchmark Comparison & Path to Sub-5 cm Accuracy**: Two-tier comparison tables (TUM RGB-D public dataset vs In-Lab RealSense), root cause analysis of ~35 cm ATE, and Park-Martin hand-eye calibration fix.
24. **Milestone Progression & 10-Minute Stress Test**: Drift reduction progression ($140\times$ improvement) and 600-second multi-lap stress test plan.
25. **Closing / Q&A Slide**: Questions & Discussion.
