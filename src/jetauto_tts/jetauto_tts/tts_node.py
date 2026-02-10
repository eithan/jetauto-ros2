"""
TTS node — listens for detected objects and speaks them aloud.

Subscribes to /detected_objects (JSON string from detector_node),
builds natural-language sentences, and speaks via pyttsx3.

Also accepts manual text on /tts/speak for arbitrary announcements.
"""

import json
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TTSNode(Node):
    """ROS2 node for text-to-speech announcements of detected objects."""

    def __init__(self):
        super().__init__('tts_node')

        # -- Declare parameters --
        self.declare_parameter('detection_topic', '/detected_objects')
        self.declare_parameter('engine', 'pyttsx3')
        self.declare_parameter('rate', 150)
        self.declare_parameter('volume', 0.9)
        self.declare_parameter('voice_id', '')
        self.declare_parameter('cooldown', 5.0)
        self.declare_parameter('min_confidence', 0.6)
        self.declare_parameter('max_objects_per_announcement', 5)
        self.declare_parameter('announce_new_only', True)
        self.declare_parameter('greeting', 'I can see')

        # -- Read parameters --
        self.detection_topic = self.get_parameter('detection_topic').value
        self.engine_name = self.get_parameter('engine').value
        self.rate = self.get_parameter('rate').value
        self.volume = self.get_parameter('volume').value
        self.voice_id = self.get_parameter('voice_id').value
        self.cooldown = self.get_parameter('cooldown').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.max_objects = self.get_parameter('max_objects_per_announcement').value
        self.announce_new_only = self.get_parameter('announce_new_only').value
        self.greeting = self.get_parameter('greeting').value

        # -- State --
        self.last_announcement_time = 0.0
        self.last_announced_labels = set()
        self._tts_lock = threading.Lock()

        # -- Initialize TTS engine --
        self.tts_engine = None
        self._init_tts()

        # -- Subscriptions --
        self.sub_detections = self.create_subscription(
            String, self.detection_topic, self._detection_callback, 10
        )
        self.sub_manual = self.create_subscription(
            String, '/tts/speak', self._manual_speak_callback, 10
        )

        self.get_logger().info(
            f'TTSNode ready — engine={self.engine_name}, '
            f'cooldown={self.cooldown}s, topic={self.detection_topic}'
        )

    # ------------------------------------------------------------------ #
    # TTS engine
    # ------------------------------------------------------------------ #

    def _init_tts(self):
        """Initialize the TTS engine."""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.rate)
            self.tts_engine.setProperty('volume', self.volume)
            if self.voice_id:
                self.tts_engine.setProperty('voice', self.voice_id)
            self.get_logger().info('pyttsx3 TTS engine initialized')
        except ImportError:
            self.get_logger().error(
                'pyttsx3 not installed! Run: pip3 install pyttsx3'
            )
        except Exception as e:
            self.get_logger().error(f'TTS init failed: {e}')

    def _speak(self, text: str):
        """Speak text in a thread-safe way (pyttsx3 is not thread-safe)."""
        if self.tts_engine is None:
            self.get_logger().warn(f'TTS unavailable, would say: "{text}"')
            return

        with self._tts_lock:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.get_logger().error(f'TTS speak error: {e}')

    # ------------------------------------------------------------------ #
    # Detection callback
    # ------------------------------------------------------------------ #

    def _detection_callback(self, msg: String):
        """Process detected objects and announce them."""
        now = time.time()
        if now - self.last_announcement_time < self.cooldown:
            return

        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON in detection message')
            return

        if not detections:
            return

        # Filter by confidence
        filtered = [
            d for d in detections
            if d.get('confidence', 0) >= self.min_confidence
        ]

        if not filtered:
            return

        # Get unique labels (preserve order, cap count)
        labels = []
        seen = set()
        for d in filtered:
            label = d.get('label', 'unknown')
            if label not in seen:
                labels.append(label)
                seen.add(label)
            if len(labels) >= self.max_objects:
                break

        # Check if anything new
        current_set = set(labels)
        if self.announce_new_only and current_set == self.last_announced_labels:
            return

        self.last_announced_labels = current_set
        self.last_announcement_time = now

        # Build sentence
        sentence = self._build_sentence(labels)
        self.get_logger().info(f'Announcing: "{sentence}"')

        # Speak in background thread to avoid blocking the callback
        threading.Thread(target=self._speak, args=(sentence,), daemon=True).start()

    # ------------------------------------------------------------------ #
    # Manual speak
    # ------------------------------------------------------------------ #

    def _manual_speak_callback(self, msg: String):
        """Speak arbitrary text published to /tts/speak."""
        text = msg.data.strip()
        if text:
            self.get_logger().info(f'Manual TTS: "{text}"')
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    # ------------------------------------------------------------------ #
    # Sentence building
    # ------------------------------------------------------------------ #

    def _build_sentence(self, labels: list) -> str:
        """Build a natural-language sentence from object labels.

        Examples:
            ["cup"] → "I can see a cup"
            ["cup", "laptop"] → "I can see a cup and a laptop"
            ["cup", "laptop", "phone"] → "I can see a cup, a laptop, and a phone"
        """
        if not labels:
            return ''

        # Add articles
        items = [self._with_article(label) for label in labels]

        if len(items) == 1:
            obj_str = items[0]
        elif len(items) == 2:
            obj_str = f'{items[0]} and {items[1]}'
        else:
            obj_str = ', '.join(items[:-1]) + f', and {items[-1]}'

        return f'{self.greeting} {obj_str}'

    @staticmethod
    def _with_article(label: str) -> str:
        """Add 'a' or 'an' before a label."""
        vowels = 'aeiou'
        article = 'an' if label[0].lower() in vowels else 'a'
        return f'{article} {label}'


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
