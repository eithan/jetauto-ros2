"""Launch file for the full vision + TTS pipeline.

Starts both lifecycle nodes and auto-transitions them:
  detector_node: configure -> activate
  tts_node:      configure -> activate
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

import lifecycle_msgs.msg


def _auto_activate(lc_node):
    """Return event handlers that configure then activate a lifecycle node."""
    configure = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lc_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    activate = launch.actions.RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=lc_node,
            goal_state='inactive',
            entities=[
                launch.actions.EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(lc_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )
    return [configure, activate]


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

    detector = LifecycleNode(
        package='jetauto_vision',
        executable='detector_node',
        name='detector_node',
        namespace='',
        parameters=[vision_config],
        output='screen',
        emulate_tty=True,
    )

    tts = LifecycleNode(
        package='jetauto_tts',
        executable='tts_node',
        name='tts_node',
        namespace='',
        parameters=[tts_config],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        detector,
        tts,
        *_auto_activate(detector),
        *_auto_activate(tts),
    ])
