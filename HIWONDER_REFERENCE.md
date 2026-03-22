# Hiwonder JetAuto Documentation Reference
> Distilled from https://docs.hiwonder.com/projects/JetAuto/en/jetauto-orin-nano/
> Eithan's version: JetAuto (non-Pro, no robot arm), Orin Nano, ROS2 Humble

## Hardware Specs
- **Chassis:** Mecanum wheel (ABAB pattern) — omnidirectional movement
- **Lidar:** SLAMTEC A1 (360° rotating, triangulation-based ranging)
  - Can also use G4 — switchable via Hiwonder config tool (Lidar TAB → select → Save → Apply)
- **Depth Camera:** DaBai (Orbbec-based), connected to Jetson port 4
  - ROS2 SDK: `OrbbecSDK_ROS2` package, launch with `ros2 launch orbbec_camera dabai.launch.py`
  - Topics: `/camera/color/image_raw`, `/camera/depth/image_raw`, point cloud available
  - Has LDP (Lens Data Protection) — auto-shutoff when objects too close; can disable via service call
- **IMU:** Built-in, 6-DOF (3-axis gyro + 3-axis accelerometer)
- **Motors:** Magnetic encoder motors, 4x Mecanum wheels
- **Screen:** 7-inch LCD (if Advanced kit)
- **Battery:** 12.6V, charged via 2A charger

## Key ROS2 Packages (Pre-installed)
### Robot Description / URDF
- **Package:** `Jetauto_description` (note capital J — access via `colcon_cd Jetauto_description`)
- **URDF location:** `urdf/jetauto.xacro` (master xacro file)
- **Sub-models included:**
  - `jetauto_car.urdf.xacro` — main chassis body
  - `lidar_a1` / `lidar_g4` — lidar models
  - `depth_camera` — depth camera frame
  - `imu` — IMU frame
  - `usb_camera` — USB camera frame
  - `materials` — colors
  - `inertial_matrix` — inertia calculations
  - `common` — shared components
  - `connect` — physical connectors
  - `jetauto_arm` / `gripper` / `arm.transmission` / `gripper.transmission` — **Pro only** (not on Eithan's)
- **TF tree root:** `base_footprint` → `base_link` (fixed joint, offset xyz 0.0 0.0 0.005)
- **Wheel links:** `wheel_left_front_link`, `wheel_left_back_link`, `wheel_right_front_link`, `wheel_right_back_link` (all fixed joints to `base_link`)
- **Mesh files:** `package://jetauto_description/meshes/*.stl`

### Motor Control / Kinematics
- **Mecanum kinematics class:** `ros2_ws/src/driver/controller/controller/mecanum.py`
  - Wheelbase: 0.216m, Track width: 0.195m, Wheel diameter: 0.097m
  - `set_velocity(linear_x, linear_y, angular_z)` → computes per-wheel RPS
  - Publishes `MotorsState` messages with per-motor `MotorState(id, rps)`
  - Motor IDs: 1-4 (left-front, left-back, right-front, right-back)
  - Speed conversion: m/s → rotations/sec via `speed / (π × wheel_diameter)`
- **Controller node launch:** `ros2 launch ros_robot_controller ros_robot_controller.launch.py`
  - Publishes: odometry, IMU data
  - Subscribes: `/cmd_vel` (presumably — standard ROS2 convention)

### Calibration
- **Calibration config:** `~/ros2_ws/src/driver/controller/config/calibrate_params.yaml`
  - `angular_correction_factor` — odometer angle scale
  - `linear_correction_factor` — odometer linear scale
- **IMU calibration:** `ros2 run imu_calib do_calib` (6-direction calibration)
  - Output: `/home/ubuntu/ros2_ws/src/calibration/config/imu_calib.yaml`
- **Angular velocity calibration:** `ros2 launch calibration angular_calib.launch.py` (interactive GUI)
- **Linear velocity calibration:** `ros2 launch calibration linear_calib.launch.py` (interactive GUI)
- **Factory-calibrated** — only recalibrate if noticeable drift

### SLAM (Pre-installed)
- **Package:** `slam`
- **Config:** `ros2_ws/src/slam/config/slam.yaml`
- **Algorithm:** `slam_toolbox` (Karto-based, graph optimization, Ceres solver)
  - Async/sync modes, map serialization, elastic pose-graph localization
- **Launch mapping:** `ros2 launch slam slam.launch.py`
- **Visualize:** `ros2 launch slam rviz_slam.launch.py`
- **Save map:** `cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true`
- **RTAB-VSLAM (3D):** `ros2 launch slam rtabmap_slam.launch.py` (auto-saves on Ctrl+C)
  - Visualize: `ros2 launch slam rviz_rtabmap.launch.py`
- **Launch internals:**
  - `slam.launch.py` → includes `robot.launch.py` (hardware bringup) + `slam_base.launch.py` (slam_toolbox)
  - 5-second delay between hardware bringup and SLAM start
  - Topics: `{robot_name}/scan` for lidar data
  - Frames: map → odom → base_link

### Navigation (Pre-installed)
- **Stack:** Nav2 (ROS2 Navigation Stack) — already installed
- **Architecture:** BT Navigator Server → Planner Server + Controller Server + Recovery Server
- **Costmaps:**
  - Global: Static Map Layer + Obstacle Map Layer + Inflation Layer
  - Local: Obstacle Map Layer + Inflation Layer
- **Localization:** AMCL (Adaptive Monte Carlo Localization) — particle filter-based
  - TF chain: `map` → `odom` → `base_footprint` → sensor frames

### Peripherals
- **Keyboard teleop:** `ros2 launch peripherals teleop_key_control.launch.py` (W/S/A/D)
- **IMU viewer:** `ros2 launch peripherals imu_view.launch.py`
- **Depth camera:** `ros2 launch peripherals depth_camera.launch.py`
- **Lidar apps:** `ros2 launch app lidar_node.launch.py debug:=true`
  - Obstacle avoidance: service call `set_running {data: 1}`
  - Following: `set_running {data: 2}`
  - Guarding: `set_running {data: 3}`

## Auto-start Service
- **Service:** `start_app_node.service`
- **Stop before manual work:** `sudo systemctl stop start_app_node.service`
- **Alternative stop:** `~/.stop_ros.sh`

## Important Notes for Our Project
1. **URDF is already provided** — we can reuse `Jetauto_description` package directly for TF tree. No need to measure offsets manually.
2. **slam_toolbox already installed** — but their launch files assume specific namespace patterns. We should write our own launch files that include their `robot.launch.py` for hardware bringup.
3. **Nav2 already installed** — we need to write our own Nav2 params.yaml tuned for house environments (not their demo arena defaults).
4. **Mecanum kinematics hardcoded** — wheelbase 0.216m, track 0.195m, wheel diameter 0.097m. Important for odometry accuracy.
5. **The `robot.launch.py` inside slam package** handles hardware bringup (motor controller, lidar, TF publishers). We should investigate what it launches and either reuse it or create our own.
6. **IMU + wheel odometry fusion** is supported — important for Mecanum wheels which can slip.
7. **Depth camera uses Orbbec SDK** — not Intel RealSense. Topic names differ.
8. **Hiwonder's SLAM demo expects a "closed setup"** — their config params are likely conservative/small-area defaults. We'll need to tune `slam_toolbox` params for larger house spaces (increase scan range, adjust loop closure thresholds, etc.).
