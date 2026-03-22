"""SLAM Mapping launch file for JetAuto.

Starts slam_toolbox in async mapping mode for building a map of unknown spaces.
Includes base_bringup with lidar enabled.

The robot explores (manually via teleop or autonomously via frontier_explorer)
while slam_toolbox builds the occupancy grid map.

Usage:
  # Mapping with manual teleop control:
  ros2 launch jetauto_autonomy slam_mapping.launch.py

  # After mapping, save the map:
  cd ~/ros2_ws/src/jetauto-ros2/src/jetauto_autonomy/maps
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

    # ── slam_toolbox (async mapping mode) ──
    # Delay 5s to let hardware stabilize (lidar needs spin-up time)
    slam_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='Starting slam_toolbox in async mapping mode...'),
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    LaunchConfiguration('slam_config'),
                    {'use_sim_time': False},
                ],
                remappings=[
                    ('/scan', '/scan'),
                ],
            ),
        ],
    )

    # ── Keyboard teleop for manual mapping ──
    # Optional: can be killed once frontier_explorer is started
    teleop_note = LogInfo(
        msg='\n'
            '╔══════════════════════════════════════════════════╗\n'
            '║  SLAM Mapping Active                            ║\n'
            '║                                                 ║\n'
            '║  Drive with keyboard teleop:                    ║\n'
            '║    ros2 launch peripherals                      ║\n'
            '║      teleop_key_control.launch.py               ║\n'
            '║                                                 ║\n'
            '║  Or start autonomous exploration:               ║\n'
            '║    ros2 run jetauto_autonomy frontier_explorer  ║\n'
            '║                                                 ║\n'
            '║  Save map when done:                            ║\n'
            '║    ros2 run nav2_map_server map_saver_cli       ║\n'
            '║      -f "house_map" --ros-args                  ║\n'
            '║      -p map_subscribe_transient_local:=true     ║\n'
            '╚══════════════════════════════════════════════════╝'
    )

    return LaunchDescription([
        slam_config_arg,
        LogInfo(msg='╔══════════════════════════════════════╗'),
        LogInfo(msg='║   JetAuto SLAM Mapping               ║'),
        LogInfo(msg='╚══════════════════════════════════════╝'),
        base_bringup,
        slam_node,
        teleop_note,
    ])
