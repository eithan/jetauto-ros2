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
            'model_name':              LaunchConfiguration('model_name'),
            'confidence_threshold':    LaunchConfiguration('confidence_threshold'),
            'device':                  LaunchConfiguration('device'),
            'image_topic':             '/camera/color/image_raw',
            'publish_annotated_image': False,
        }],
    )

    # ── Lifecycle transitions ─────────────────────────────────────────── #
    # Use a shell script that polls until the node is actually in the
    # expected state, so we're not racing against startup time.
    configure_cmd = TimerAction(
        period=4.0,
        actions=[ExecuteProcess(
            cmd=[
                'bash', '-c',
                'for i in $(seq 1 10); do '
                '  STATE=$(ros2 lifecycle get /detector_node 2>/dev/null); '
                '  echo "[lifecycle] state: $STATE"; '
                '  if echo "$STATE" | grep -q "unconfigured"; then '
                '    ros2 lifecycle set /detector_node configure && break; '
                '  elif echo "$STATE" | grep -q "inactive\\|active"; then '
                '    echo "[lifecycle] already past unconfigured, skipping configure"; break; '
                '  fi; '
                '  sleep 1; '
                'done'
            ],
            output='screen',
        )],
    )

    activate_cmd = TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=[
                'bash', '-c',
                'for i in $(seq 1 10); do '
                '  STATE=$(ros2 lifecycle get /detector_node 2>/dev/null); '
                '  echo "[lifecycle] state: $STATE"; '
                '  if echo "$STATE" | grep -q "inactive"; then '
                '    ros2 lifecycle set /detector_node activate && break; '
                '  elif echo "$STATE" | grep -q "active"; then '
                '    echo "[lifecycle] already active"; break; '
                '  fi; '
                '  sleep 1; '
                'done'
            ],
            output='screen',
        )],
    )

    # ── Sim node — starts after detector activation window ────────────── #
    sim = TimerAction(
        period=28.0,
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
