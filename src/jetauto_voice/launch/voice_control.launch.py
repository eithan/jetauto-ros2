"""
Launch file for JetAuto custom voice control.

Starts ONLY our voice_control_node.
Hiwonder's asr_node must already be running (it's the hardware ASR bridge).
Do NOT launch any of Hiwonder's voice_control_*.py nodes — ours replaces them.

Usage:
  ros2 launch jetauto_voice voice_control.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetauto_voice',
            executable='voice_control_node',
            name='voice_control_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
