"""Base hardware bringup for JetAuto.

Starts the shared hardware layer that all other launch files depend on:
- Motor controller + odometry + IMU (ros_robot_controller)
- URDF robot description + TF publishers (jetauto_description)
- Depth camera (Orbbec DaBai) — always on
- Lidar (SLAMTEC A1) — conditionally launched via 'use_lidar' arg

Usage:
  ros2 launch jetauto_autonomy base_bringup.launch.py
  ros2 launch jetauto_autonomy base_bringup.launch.py use_lidar:=true
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='false',
        description='Start the lidar driver (only needed for SLAM/Nav)',
    )

    use_lidar = LaunchConfiguration('use_lidar')

    # ── Hiwonder's robot controller (motors, odometry, IMU, TF) ──
    # This is what start_app_node.service runs on boot.
    # If the service is running, we skip this. If not, we start it.
    #
    # IMPORTANT: Before launching this, ensure start_app_node.service is stopped:
    #   sudo systemctl stop start_app_node.service
    #
    # The ros_robot_controller publishes:
    #   - /odom (nav_msgs/Odometry)
    #   - /imu/data (sensor_msgs/Imu)
    #   - /cmd_vel subscriber (geometry_msgs/Twist)
    #   - TF: odom -> base_footprint
    robot_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_robot_controller'),
                'launch',
                'ros_robot_controller.launch.py',
            ])
        ),
    )

    # ── URDF Robot Description ──
    # Publishes static TF transforms for all links (lidar, camera, wheels, etc.)
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jetauto_description'),
                'launch',
                'jetauto_description.launch.py',
            ])
        ),
    )

    # ── Depth Camera (Orbbec DaBai) ──
    # Always-on for vision pipeline
    # Publishes: /camera/color/image_raw, /camera/depth/image_raw
    depth_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('peripherals'),
                'launch',
                'depth_camera.launch.py',
            ])
        ),
    )

    # ── Lidar (SLAMTEC A1) — conditional ──
    # Only started when SLAM or Navigation is needed (saves battery)
    # Publishes: /scan (sensor_msgs/LaserScan)
    lidar = GroupAction(
        condition=IfCondition(use_lidar),
        actions=[
            LogInfo(msg='Starting lidar driver (SLAMTEC A1)...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('peripherals'),
                        'launch',
                        'lidar.launch.py',
                    ])
                ),
            ),
        ],
    )

    return LaunchDescription([
        use_lidar_arg,
        LogInfo(msg='╔══════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto Base Hardware Bringup      ║'),
        LogInfo(msg='╚══════════════════════════════════════╝'),
        robot_controller,
        robot_description,
        depth_camera,
        lidar,
    ])
