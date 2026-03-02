"""
JetAuto Simulation Launch File
───────────────────────────────
Launches detector (lifecycle) + lifecycle manager + sim node.

Compatible with ROS2 Humble and Jazzy.

Usage:
  ros2 launch jetauto_sim sim.launch.py

Optional args:
  model_name:=yolov8n.pt
  confidence_threshold:=0.5
  device:=cpu
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    # ── Launch args ───────────────────────────────────────────────────── #
    model_arg = DeclareLaunchArgument(
        'model_name', default_value='yolov8n.pt',
        description='YOLO model weights file'
    )
    conf_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.4',
        description='Detection confidence threshold (lower = more detections)'
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='Inference device: cpu, 0 (GPU), etc.'
    )

    # ── Detector node (lifecycle) ─────────────────────────────────────── #
    detector = LifecycleNode(
        package='jetauto_vision',
        executable='detector_node',
        name='detector_node',
        namespace='',
        output='screen',
        parameters=[{
            'model_name':           LaunchConfiguration('model_name'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device':               LaunchConfiguration('device'),
            'image_topic':          '/camera/color/image_raw',
            'publish_annotated_image': False,   # skip annotated for sim
        }],
    )

    # ── Lifecycle transitions ─────────────────────────────────────────── #
    # Wait 3s for node to start, then configure; wait another 4s, then activate.
    configure_cmd = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'lifecycle', 'set', '/detector_node', 'configure'],
            output='screen',
        )],
    )

    activate_cmd = TimerAction(
        period=7.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'lifecycle', 'set', '/detector_node', 'activate'],
            output='screen',
        )],
    )

    # ── Sim node — starts after detector is active ────────────────────── #
    sim = TimerAction(
        period=10.0,
        actions=[Node(
            package='jetauto_sim',
            executable='sim_node',
            name='jetauto_sim',
            output='screen',
        )],
    )

    return LaunchDescription([
        model_arg,
        conf_arg,
        device_arg,
        detector,
        configure_cmd,
        activate_cmd,
        sim,
    ])
