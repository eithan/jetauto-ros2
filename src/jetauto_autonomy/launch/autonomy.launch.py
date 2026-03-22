"""Full autonomy launch — SLAM + Nav2 + Frontier Exploration.

This is the "explore an unknown house" mode. It combines:
1. slam_toolbox (async mapping — builds map in real-time)
2. Nav2 (navigation — plans paths, avoids obstacles)
3. frontier_explorer (picks unexplored frontiers, sends Nav2 goals)
4. safety_monitor (emergency stop, battery check, cliff detection)

The robot will autonomously explore room by room, building a map as it goes.

Usage:
  ros2 launch jetauto_autonomy autonomy.launch.py

  # Save map when exploration is complete:
  ros2 run nav2_map_server map_saver_cli -f "house_map" --ros-args -p map_subscribe_transient_local:=true
"""

import os

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
        default_value='300.0',
        description='Max seconds to explore before stopping (0 = unlimited)',
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

    # ── slam_toolbox (async mapping) ──
    # Delay 5s for hardware + lidar
    slam_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='Starting slam_toolbox (async mapping)...'),
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

    # ── Nav2 (navigation without a static map — uses SLAM's live map) ──
    # SLAM publishes /map, Nav2 subscribes to it for global costmap
    # We launch Nav2 bringup but skip map_server and amcl (SLAM handles localization)
    nav2_stack = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='Starting Nav2 (using SLAM live map)...'),
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

    # ── Frontier Explorer (our custom node) ──
    # Waits for SLAM + Nav2 to be ready, then starts picking frontiers
    frontier_explorer = TimerAction(
        period=15.0,
        actions=[
            LogInfo(msg='Starting frontier explorer...'),
            Node(
                package='jetauto_autonomy',
                executable='frontier_explorer',
                name='frontier_explorer',
                output='screen',
                parameters=[{
                    'explore_timeout': LaunchConfiguration('explore_timeout'),
                    'min_frontier_size': 5,
                    'robot_radius': 0.20,
                    'potential_scale': 3.0,
                    'gain_scale': 1.0,
                    'transform_tolerance': 0.5,
                }],
            ),
        ],
    )

    # ── Safety Monitor ──
    # Runs alongside everything, can emergency-stop the robot
    safety_monitor = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='jetauto_autonomy',
                executable='safety_monitor',
                name='safety_monitor',
                output='screen',
                parameters=[{
                    'min_obstacle_distance': 0.15,   # 15cm emergency stop
                    'check_frequency': 10.0,          # 10Hz safety checks
                }],
            ),
        ],
    )

    return LaunchDescription([
        slam_config_arg,
        nav2_config_arg,
        explore_timeout_arg,
        LogInfo(msg='╔══════════════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto Autonomous Exploration             ║'),
        LogInfo(msg='║   SLAM + Nav2 + Frontier Explorer            ║'),
        LogInfo(msg='╚══════════════════════════════════════════════╝'),
        base_bringup,
        slam_node,
        nav2_stack,
        frontier_explorer,
        safety_monitor,
    ])
