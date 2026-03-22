"""Navigation launch file for JetAuto.

Starts the full Nav2 stack with a pre-built map for autonomous point-to-point
navigation. Use this AFTER you've built and saved a map with slam_mapping.launch.py.

Includes base_bringup with lidar enabled.

Usage:
  ros2 launch jetauto_autonomy navigation.launch.py map:=/path/to/house_map.yaml

  # Then send a navigation goal:
  ros2 topic pub /goal_pose geometry_msgs/PoseStamped '{...}'
  # Or use the frontier_explorer for autonomous coverage
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
    GroupAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    autonomy_share = get_package_share_directory('jetauto_autonomy')

    nav2_config_arg = DeclareLaunchArgument(
        'nav2_config',
        default_value=os.path.join(autonomy_share, 'config', 'nav2_params.yaml'),
        description='Path to Nav2 params YAML',
    )

    map_arg = DeclareLaunchArgument(
        'map',
        description='Path to map YAML file (from map_saver_cli)',
    )

    autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Auto-start Nav2 lifecycle nodes',
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

    # ── Nav2 Stack ──
    # Delay 5s for hardware + lidar spin-up
    nav2_stack = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='Starting Nav2 navigation stack...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('nav2_bringup'),
                        'launch',
                        'bringup_launch.py',
                    ])
                ),
                launch_arguments={
                    'map': LaunchConfiguration('map'),
                    'params_file': LaunchConfiguration('nav2_config'),
                    'autostart': LaunchConfiguration('autostart'),
                    'use_sim_time': 'false',
                }.items(),
            ),
        ],
    )

    return LaunchDescription([
        nav2_config_arg,
        map_arg,
        autostart_arg,
        LogInfo(msg='╔══════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto Navigation (Nav2)          ║'),
        LogInfo(msg='╚══════════════════════════════════════╝'),
        base_bringup,
        nav2_stack,
    ])
