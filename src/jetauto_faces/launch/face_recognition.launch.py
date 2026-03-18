"""Launch file for the face recognition lifecycle node.

Starts the face_recognition_node and automatically transitions it through
configure -> activate so it begins processing immediately.
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

import lifecycle_msgs.msg


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('jetauto_faces'),
        'config',
        'face_params.yaml',
    )

    face_node = LifecycleNode(
        package='jetauto_faces',
        executable='face_recognition_node',
        name='face_recognition_node',
        namespace='',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )

    # Auto-configure after startup
    configure_event = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(face_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    # Auto-activate after configure succeeds
    activate_event = launch.actions.RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=face_node,
            goal_state='inactive',
            entities=[
                launch.actions.EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(face_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    return LaunchDescription([
        face_node,
        configure_event,
        activate_event,
    ])
