"""Full stack launch — Autonomy + Perception combined.

Runs everything: SLAM, Nav2, frontier exploration, YOLO, TTS, Florence-2.
The robot explores the house autonomously while describing what it sees.

This is the "final form" — all systems active, shared hardware layer.

Usage:
  ros2 launch jetauto_autonomy full_stack.launch.py

Note: GPU memory sharing is important here:
  - YOLO runs always (~500MB VRAM)
  - Florence-2 loads on-demand (caption_node lifecycle)
  - SLAM + Nav2 are CPU-only (no GPU)
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    autonomy_share = get_package_share_directory('jetauto_autonomy')

    slam_config_arg = DeclareLaunchArgument(
        'slam_config',
        default_value=os.path.join(autonomy_share, 'config', 'slam_params.yaml'),
        description='Path to slam_toolbox params YAML',
    )

    nav2_config_arg = DeclareLaunchArgument(
        'nav2_config',
        default_value=os.path.join(autonomy_share, 'config', 'nav2_params.yaml'),
        description='Path to Nav2 params YAML',
    )

    explore_timeout_arg = DeclareLaunchArgument(
        'explore_timeout',
        default_value='0.0',
        description='Max seconds to explore (0 = unlimited)',
    )

    # ── Base hardware bringup (with lidar ON) ──
    base_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jetauto_autonomy'),
                'launch',
                'base_bringup.launch.py',
            ])
        ),
        launch_arguments={'use_lidar': 'true'}.items(),
    )

    # ── SLAM ──
    slam_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    LaunchConfiguration('slam_config'),
                    {'use_sim_time': False},
                ],
            ),
        ],
    )

    # ── Nav2 ──
    nav2_stack = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('nav2_bringup'),
                        'launch',
                        'navigation_launch.py',
                    ])
                ),
                launch_arguments={
                    'params_file': LaunchConfiguration('nav2_config'),
                    'autostart': 'true',
                    'use_sim_time': 'false',
                }.items(),
            ),
        ],
    )

    # ── Frontier Explorer ──
    frontier_explorer = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='jetauto_autonomy',
                executable='frontier_explorer',
                name='frontier_explorer',
                output='screen',
                parameters=[{
                    'explore_timeout': LaunchConfiguration('explore_timeout'),
                }],
            ),
        ],
    )

    # ── Safety Monitor ──
    safety_monitor = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='jetauto_autonomy',
                executable='safety_monitor',
                name='safety_monitor',
                output='screen',
            ),
        ],
    )

    # ── Vision (YOLO) — delay to let SLAM claim GPU first ──
    vision = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('jetauto_vision'),
                        'launch',
                        'vision_launch.py',
                    ])
                ),
            ),
        ],
    )

    # ── TTS ──
    tts = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('jetauto_tts'),
                        'launch',
                        'tts_only.launch.py',
                    ])
                ),
            ),
        ],
    )

    return LaunchDescription([
        slam_config_arg,
        nav2_config_arg,
        explore_timeout_arg,
        LogInfo(msg='╔══════════════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto Full Stack                         ║'),
        LogInfo(msg='║   SLAM + Nav2 + Explorer + YOLO + TTS        ║'),
        LogInfo(msg='╚══════════════════════════════════════════════╝'),
        base_bringup,
        slam_node,
        nav2_stack,
        frontier_explorer,
        safety_monitor,
        vision,
        tts,
    ])
