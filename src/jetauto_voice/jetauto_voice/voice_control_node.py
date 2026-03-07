#!/usr/bin/env python3
"""
Custom voice control node for JetAuto.

Subscribes to /asr_node/voice_words (output of Hiwonder's iFlytek ASR node)
and maps recognized phrases to robot actions — currently enable/disable of
the vision detection pipeline.

# ---------------------------------------------------------------------------
# VOCABULARY NOTE
# ---------------------------------------------------------------------------
# The iFlytek offline ASR has a fixed grammar. Known English phrases (from
# Hiwonder's voice_control_move.py) are listed in KNOWN_PHRASES below.
#
# If you need custom phrases (e.g. "start detection"), you have two options:
#   A) Remap an existing phrase to a new action (see COMMAND_MAP below).
#   B) Modify the iFlytek grammar file (.bnf/.grm) on the robot — search
#      /home/ubuntu/ros2_ws/src/xf_mic_asr_offline/ for .bnf/.grm files.
#   C) Replace iFlytek with Vosk or Whisper for fully custom vocab.
# ---------------------------------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


# System messages published by asr_node that are NOT user commands.
_SYSTEM_MESSAGES = {
    '唤醒成功(wake-up-success)',
    '休眠(Sleep)',
    '失败5次(Fail-5-times)',
    '失败10次(Fail-10-times)',
}

# ---------------------------------------------------------------------------
# COMMAND MAP  —  edit this to change voice command behaviour
# ---------------------------------------------------------------------------
# Format:  'recognized phrase' -> callable(node) or string action key
#
# Built-in action keys:
#   'enable_detection'   → publishes True  to /jetauto/detection/enable
#   'disable_detection'  → publishes False to /jetauto/detection/enable
#
# The iFlytek ASR (English mode) recognises these phrases out of the box:
#   'go forward', 'go backward', 'turn left', 'turn right', 'come here'
#
# Remap them however you like.  When you have a custom grammar, add new
# phrases here and they'll just work.
# ---------------------------------------------------------------------------
COMMAND_MAP: dict[str, str] = {
    # --- Detection control (custom phrases added to call.bnf) ---
    'start detection':  'enable_detection',
    'enable vision':    'enable_detection',
    'stop detection':   'disable_detection',
    'disable vision':   'disable_detection',
    'what do you see':  'enable_detection',   # convenience alias
}


class VoiceControlNode(Node):
    """Translates ASR word events into robot control commands."""

    def __init__(self):
        super().__init__('voice_control_node')

        # -- Publishers --
        self._detection_pub = self.create_publisher(
            Bool, '/jetauto/detection/enable', 1
        )

        # -- Subscribe to ASR output --
        self.create_subscription(
            String, '/asr_node/voice_words', self._words_callback, 1
        )

        self.get_logger().info(
            'Voice control node started — listening on /asr_node/voice_words'
        )
        self.get_logger().info(
            f'Active commands: {list(COMMAND_MAP.keys())}'
        )

    # ------------------------------------------------------------------
    # ASR callback
    # ------------------------------------------------------------------

    def _words_callback(self, msg: String):
        phrase = msg.data.strip()
        self.get_logger().debug(f'ASR received: "{phrase}"')

        # Ignore system state messages
        if phrase in _SYSTEM_MESSAGES:
            return

        action = COMMAND_MAP.get(phrase)
        if action is None:
            self.get_logger().debug(f'No command mapped for: "{phrase}"')
            return

        self.get_logger().info(f'Voice command: "{phrase}" → {action}')
        self._dispatch(action)

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: str):
        if action == 'enable_detection':
            self._set_detection(True)
        elif action == 'disable_detection':
            self._set_detection(False)
        else:
            self.get_logger().warn(f'Unknown action: "{action}"')

    def _set_detection(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self._detection_pub.publish(msg)
        state = 'ENABLED' if enabled else 'DISABLED'
        self.get_logger().info(f'Detection {state}')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
