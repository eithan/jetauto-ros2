"""
Launch file for JetAuto custom voice control.

Launches the legacy iFlyTek bridge node (voice_control_node) OR the new
fully-offline voice commander (voice_commander_node).  By default only the
offline commander is started.  Set the ``use_iflytek`` launch argument to
``true`` to start the legacy bridge instead (requires asr_node running).

Usage::

    # Offline commander (default — no iFlyTek needed):
    ros2 launch jetauto_voice voice_control.launch.py

    # Legacy iFlyTek bridge:
    ros2 launch jetauto_voice voice_control.launch.py use_iflytek:=true

    # Offline commander with custom parameters:
    ros2 launch jetauto_voice voice_control.launch.py \\
        wake_word_model:=alexa \\
        wake_word_threshold:=0.4 \\
        stt_model_size:=small \\
        vad_energy_threshold:=400 \\
        vad_silence_ms:=800
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
        default_value='base.en',
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
        default_value='-1',
        description='ALSA mic device index (-1 = system default)',
    )
    vad_aggressiveness_arg = DeclareLaunchArgument(
        'vad_aggressiveness',
        default_value='2',
        description='WebRTC VAD aggressiveness 0-3 (0=permissive, 3=aggressive). Higher filters more non-speech.',
    )
    vad_speech_end_frames_arg = DeclareLaunchArgument(
        'vad_speech_end_frames',
        default_value='25',
        description='Consecutive silence frames (20ms each) to trigger end-of-speech. Default 25 = 500ms.',
    )
    vad_listen_timeout_sec_arg = DeclareLaunchArgument(
        'vad_listen_timeout_sec',
        default_value='4.0',
        description='Seconds to wait for speech to start before giving up.',
    )
    wake_cooldown_sec_arg = DeclareLaunchArgument(
        'wake_cooldown_sec',
        default_value='3.0',
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
            'vad_speech_end_frames': LaunchConfiguration('vad_speech_end_frames'),
            'vad_listen_timeout_sec': LaunchConfiguration('vad_listen_timeout_sec'),
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

    return LaunchDescription([
        use_iflytek_arg,
        wake_word_model_arg,
        wake_word_threshold_arg,
        stt_model_size_arg,
        stt_device_arg,
        stt_compute_type_arg,
        mic_device_index_arg,
        vad_aggressiveness_arg,
        vad_speech_end_frames_arg,
        vad_listen_timeout_sec_arg,
        wake_cooldown_sec_arg,
        voice_commander,
        voice_control,
    ])
