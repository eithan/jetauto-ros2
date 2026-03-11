"""
Launch file for JetAuto custom voice control.

Launches the legacy iFlyTek bridge node (voice_control_node) OR the new
fully-offline voice commander (voice_commander_node).  By default only the
offline commander is started.  Set the ``use_iflytek`` launch argument to
``true`` to start the legacy bridge instead (requires asr_node running).

Also launches detector_node (jetauto_vision) so that "start vision" /
"stop vision" voice commands and find-object intents actually work — the
voice commander publishes to /jetauto/detection/enable and
/jetauto/detection/target, which detector_node subscribes to.

Usage::

    # Offline commander + vision (default — no iFlyTek needed):
    ros2 launch jetauto_voice voice_control.launch.py

    # Legacy iFlyTek bridge:
    ros2 launch jetauto_voice voice_control.launch.py use_iflytek:=true

    # Offline commander with custom parameters:
    ros2 launch jetauto_voice voice_control.launch.py \\
        wake_word_model:=alexa \\
        wake_word_threshold:=0.4 \\
        stt_model_size:=small
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
        'use_iflytek',
        default_value='false',
        description='If true, launch the legacy iFlyTek ASR bridge instead of the offline commander',
    )

    # -- Offline voice commander parameters --
    wake_word_model_arg = DeclareLaunchArgument(
        'wake_word_model',
        default_value='hey_jarvis',
        description='openWakeWord model name (hey_jarvis, alexa, hey_mycroft, ...)',
    )
    wake_word_threshold_arg = DeclareLaunchArgument(
        'wake_word_threshold',
        default_value='0.5',
        description='Wake word detection threshold [0.0, 1.0]',
    )
    stt_model_size_arg = DeclareLaunchArgument(
        'stt_model_size',
        default_value='tiny.en',
        description='faster-whisper model size: tiny.en, base.en, small.en, medium, large-v2 (.en = English-only, more accurate)',
    )
    stt_device_arg = DeclareLaunchArgument(
        'stt_device',
        default_value='cpu',
        description='STT inference device: cuda or cpu (default: cpu — Jetson CTranslate2 is not CUDA-compiled)',
    )
    stt_compute_type_arg = DeclareLaunchArgument(
        'stt_compute_type',
        default_value='int8',
        description='faster-whisper compute type: float16, int8, float32 (default: int8 for CPU)',
    )
    mic_device_index_arg = DeclareLaunchArgument(
        'mic_device_index',
        default_value='1',
        description='ALSA mic device index. 1=XFM-DP (JetAuto built-in mic), 2=USB Audio, -1=system default',
    )
    vad_aggressiveness_arg = DeclareLaunchArgument(
        'vad_aggressiveness',
        default_value='1',
        description='WebRTC VAD aggressiveness 0-3. Higher = more strict (requires louder speech).',
    )
    vad_listen_timeout_sec_arg = DeclareLaunchArgument(
        'vad_listen_timeout_sec',
        default_value='10.0',
        description='Seconds to wait for speech to start before giving up.',
    )
    vad_speech_end_frames_arg = DeclareLaunchArgument(
        'vad_speech_end_frames',
        default_value='30',
        description='Consecutive 20ms silence frames to trigger end-of-speech (30=600ms).',
    )
    wake_cooldown_sec_arg = DeclareLaunchArgument(
        'wake_cooldown_sec',
        default_value='5.0',
        description='Seconds to suppress wake word re-triggering after a detection.',
    )

    use_iflytek = LaunchConfiguration('use_iflytek')

    # -- Offline voice commander (default) --
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

    # -- Legacy iFlyTek bridge (opt-in) --
    voice_control = Node(
        package='jetauto_voice',
        executable='voice_control_node',
        name='voice_control_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(use_iflytek),
    )

    # -- Vision detector node (lifecycle) —  handles /jetauto/detection/enable + /target --
    # Starts alongside voice so "start vision" / "find X" commands actually reach it.
    vision_config = os.path.join(
        get_package_share_directory('jetauto_vision'),
        'config',
        'vision_params.yaml',
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
    detector_configure = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(detector),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    detector_activate = launch.actions.RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=detector,
            goal_state='inactive',
            entities=[
                launch.actions.EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(detector),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    # -- TTS node (lifecycle) — speaks responses aloud --
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
    tts_configure = launch.actions.EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(tts),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    tts_activate = launch.actions.RegisterEventHandler(
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

    return LaunchDescription([
        use_iflytek_arg,
        wake_word_model_arg,
        wake_word_threshold_arg,
        stt_model_size_arg,
        stt_device_arg,
        stt_compute_type_arg,
        mic_device_index_arg,
        vad_aggressiveness_arg,
        vad_listen_timeout_sec_arg,
        vad_speech_end_frames_arg,
        wake_cooldown_sec_arg,
        detector,
        detector_configure,
        detector_activate,
        voice_commander,
        voice_control,
        tts,
        tts_configure,
        tts_activate,
    ])
