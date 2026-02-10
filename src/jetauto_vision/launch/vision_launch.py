"""Launch file for the object detection node."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('jetauto_vision'),
        'config',
        'vision_params.yaml',
    )

    return LaunchDescription([
        Node(
            package='jetauto_vision',
            executable='detector_node',
            name='detector_node',
            parameters=[config],
            output='screen',
            emulate_tty=True,
        ),
    ])
