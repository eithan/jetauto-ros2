"""Launch file for TTS node only (lifecycle managed).

Used by the dashboard to run tts_node as a shared resource
independent of detector_node and voice_commander_node.
"""

import os

import launch
import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState


def generate_launch_description():
    tts_config = os.path.join(
        get_package_share_directory('jetauto_tts'),
        'config',
        'tts_params.yaml',
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

    configure = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(tts),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    activate = launch.actions.RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=tts,
            goal_state='inactive',
            entities=[
                launch.actions.EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(tts),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    return LaunchDescription([tts, configure, activate])
