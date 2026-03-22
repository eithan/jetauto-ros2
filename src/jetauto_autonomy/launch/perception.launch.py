"""Perception-only launch — YOLO + TTS + Florence-2 (no lidar, no SLAM).

This is the "stationary observer" mode. Camera-based only:
- YOLOv8 object detection → dashboard
- Florence-2 scene captioning → TTS
- No lidar, no navigation, no battery drain from spinning motor

Usage:
  ros2 launch jetauto_autonomy perception.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Base hardware bringup (NO lidar) ──
    base_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jetauto_autonomy'),
                'launch',
                'base_bringup.launch.py',
            ])
        ),
        launch_arguments={'use_lidar': 'false'}.items(),
    )

    # ── Vision (YOLO detector) ──
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jetauto_vision'),
                'launch',
                'vision_launch.py',
            ])
        ),
    )

    # ── TTS ──
    tts = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jetauto_tts'),
                'launch',
                'tts_only.launch.py',
            ])
        ),
    )

    return LaunchDescription([
        LogInfo(msg='╔══════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto Perception Only            ║'),
        LogInfo(msg='║   YOLO + TTS (no lidar)              ║'),
        LogInfo(msg='╚══════════════════════════════════════╝'),
        base_bringup,
        vision,
        tts,
    ])
