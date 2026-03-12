"""Launch file for the JetAuto Dashboard."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('jetauto_dashboard'),
        'config',
        'dashboard_params.yaml',
    )

    return LaunchDescription([
        Node(
            package='jetauto_dashboard',
            executable='dashboard_node',
            name='dashboard_node',
            output='screen',
            parameters=[config],
        ),
    ])
