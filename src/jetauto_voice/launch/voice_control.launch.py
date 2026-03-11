"""
Launch file for JetAuto voice control + vision detection + TTS.

Usage::

    ros2 launch jetauto_voice voice_control.launch.py
"""

import os

import launch
import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState


def generate_launch_description():
    use_iflytek_arg = DeclareLaunchArgument(
        'use_iflytek', default_value='false',
        description='Legacy iFlyTek ASR bridge (true) vs offline commander (false)',
    )
    wake_word_model_arg = DeclareLaunchArgument(
        'wake_word_model', default_value='hey_jarvis',
    )
    wake_word_threshold_arg = DeclareLaunchArgument(
        'wake_word_threshold', default_value='0.5',
    )
    stt_model_size_arg = DeclareLaunchArgument(
        'stt_model_size', default_value='base.en',
        description='tiny.en (fast/poor) | base.en (balanced) | small.en (slow/good)',
    )
    stt_device_arg = DeclareLaunchArgument(
        'stt_device', default_value='cpu',
    )
    stt_compute_type_arg = DeclareLaunchArgument(
        'stt_compute_type', default_value='int8',
    )
    mic_device_index_arg = DeclareLaunchArgument(
        'mic_device_index', default_value='1',
        description='1=XFM-DP (JetAuto mic), 2=USB Audio, -1=system default',
    )
    vad_aggressiveness_arg = DeclareLaunchArgument(
        'vad_aggressiveness', default_value='2',
        description='0=permissive … 3=strict',
    )
    vad_listen_timeout_sec_arg = DeclareLaunchArgument(
        'vad_listen_timeout_sec', default_value='30.0',
    )
    vad_speech_end_frames_arg = DeclareLaunchArgument(
        'vad_speech_end_frames', default_value='30',
    )
    wake_cooldown_sec_arg = DeclareLaunchArgument(
        'wake_cooldown_sec', default_value='5.0',
    )

    use_iflytek = LaunchConfiguration('use_iflytek')

    voice_commander = Node(
        package='jetauto_voice',
        executable='voice_commander_node',
        name='voice_commander_node',
        output='screen',
        emulate_tty=True,
        condition=UnlessCondition(use_iflytek),
        parameters=[{
            'wake_word_model': LaunchConfiguration('wake_word_model'),
            'wake_word_threshold': LaunchConfiguration('wake_word_threshold'),
            'stt_model_size': LaunchConfiguration('stt_model_size'),
            'stt_device': LaunchConfiguration('stt_device'),
            'stt_compute_type': LaunchConfiguration('stt_compute_type'),
            'mic_device_index': LaunchConfiguration('mic_device_index'),
            'vad_aggressiveness': LaunchConfiguration('vad_aggressiveness'),
            'vad_listen_timeout_sec': LaunchConfiguration('vad_listen_timeout_sec'),
            'vad_speech_end_frames': LaunchConfiguration('vad_speech_end_frames'),
            'wake_cooldown_sec': LaunchConfiguration('wake_cooldown_sec'),
        }],
    )

    voice_control = Node(
        package='jetauto_voice',
        executable='voice_control_node',
        name='voice_control_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(use_iflytek),
    )

    # -- Detector (lifecycle) --
    vision_config = os.path.join(
        get_package_share_directory('jetauto_vision'), 'config', 'vision_params.yaml',
    )
    detector = LifecycleNode(
        package='jetauto_vision', executable='detector_node',
        name='detector_node', namespace='',
        parameters=[vision_config], output='screen', emulate_tty=True,
    )
    det_cfg = launch.actions.EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matches_action(detector),
        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
    ))
    det_act = launch.actions.RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=detector, goal_state='inactive',
        entities=[launch.actions.EmitEvent(event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(detector),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
        ))],
    ))

    # -- TTS (lifecycle) --
    tts_config = os.path.join(
        get_package_share_directory('jetauto_tts'), 'config', 'tts_params.yaml',
    )
    tts = LifecycleNode(
        package='jetauto_tts', executable='tts_node',
        name='tts_node', namespace='',
        parameters=[tts_config], output='screen', emulate_tty=True,
    )
    tts_cfg = launch.actions.EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matches_action(tts),
        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
    ))
    tts_act = launch.actions.RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=tts, goal_state='inactive',
        entities=[launch.actions.EmitEvent(event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(tts),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
        ))],
    ))

    return LaunchDescription([
        use_iflytek_arg,
        wake_word_model_arg, wake_word_threshold_arg,
        stt_model_size_arg, stt_device_arg, stt_compute_type_arg,
        mic_device_index_arg, vad_aggressiveness_arg,
        vad_listen_timeout_sec_arg, vad_speech_end_frames_arg,
        wake_cooldown_sec_arg,
        detector, det_cfg, det_act,
        voice_commander, voice_control,
        tts, tts_cfg, tts_act,
    ])
