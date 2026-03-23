"""Launch file for the interactive face enrollment node.

Starts enrollment_node with parameters from face_params.yaml so
it shares the same camera topic, model, and DB path as the
face_recognition_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('jetauto_faces'),
        'config',
        'face_params.yaml',
    )

    enrollment_node = Node(
        package='jetauto_faces',
        executable='enrollment_node',
        name='face_enrollment_node',
        namespace='',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([enrollment_node])
