"""Launch file for the TTS node + vision node (full pipeline)."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    vision_config = os.path.join(
        get_package_share_directory('jetauto_vision'),
        'config',
        'vision_params.yaml',
    )
    tts_config = os.path.join(
        get_package_share_directory('jetauto_tts'),
        'config',
        'tts_params.yaml',
    )

    return LaunchDescription([
        # Vision (object detection)
        Node(
            package='jetauto_vision',
            executable='detector_node',
            name='detector_node',
            parameters=[vision_config],
            output='screen',
            emulate_tty=True,
        ),
        # TTS (spoken announcements)
        Node(
            package='jetauto_tts',
            executable='tts_node',
            name='tts_node',
            parameters=[tts_config],
            output='screen',
            emulate_tty=True,
        ),
    ])
