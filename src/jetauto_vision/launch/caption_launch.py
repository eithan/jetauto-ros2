"""Launch file for the Florence-2 scene captioning lifecycle node.

Starts caption_node and auto-transitions through configure -> activate.
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
    config = os.path.join(
        get_package_share_directory('jetauto_vision'),
        'config',
        'caption_params.yaml',
    )

    caption = LifecycleNode(
        package='jetauto_vision',
        executable='caption_node',
        name='caption_node',
        namespace='',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )

    configure = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(caption),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    activate = launch.actions.RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=caption,
            goal_state='inactive',
            entities=[
                launch.actions.EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(caption),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    return LaunchDescription([caption, configure, activate])
