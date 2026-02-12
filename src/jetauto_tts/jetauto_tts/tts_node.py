"""
TTS node — listens for detected objects and speaks them aloud.

Subscribes to /detected_objects (DetectedObjectArray from detector_node),
builds natural-language sentences, and speaks via pyttsx3.

Uses a dedicated speaker thread with a queue to avoid blocking ROS
callbacks and prevent pyttsx3 threading deadlocks.

Also accepts manual text on /tts/speak for arbitrary announcements.
"""

import queue
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from std_msgs.msg import String

from jetauto_msgs.msg import DetectedObjectArray


class TTSNode(LifecycleNode):
    """ROS2 lifecycle node for text-to-speech announcements of detected objects."""

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

        # -- State --
        self.last_announcement_time = 0.0
        self.last_announced_labels = set()
        self.tts_engine = None
        self._speech_queue = queue.Queue(maxsize=10)
        self._speaker_thread = None
        self._shutdown_event = threading.Event()

        self.get_logger().info('TTSNode created (inactive — waiting for configure)')

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks
    # ------------------------------------------------------------------ #

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Read parameters and initialize the TTS engine."""
        self._read_parameters()

        if not self._init_tts():
            return TransitionCallbackReturn.FAILURE

        # Start the dedicated speaker thread
        self._shutdown_event.clear()
        self._speaker_thread = threading.Thread(
            target=self._speaker_loop, daemon=True, name='tts_speaker'
        )
        self._speaker_thread.start()

        self.get_logger().info(
            f'Configured — engine={self.engine_name}, cooldown={self.cooldown}s'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Create subscriptions and start listening."""
        self.sub_detections = self.create_subscription(
            DetectedObjectArray, self.detection_topic,
            self._detection_callback, 10
        )
        self.sub_manual = self.create_subscription(
            String, '/tts/speak', self._manual_speak_callback, 10
        )
        self.get_logger().info('Activated — listening for detections')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop listening."""
        if hasattr(self, 'sub_detections'):
            self.destroy_subscription(self.sub_detections)
        if hasattr(self, 'sub_manual'):
            self.destroy_subscription(self.sub_manual)
        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop speaker thread and release TTS engine."""
        self._stop_speaker()
        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        self._stop_speaker()
        self.get_logger().info('Shut down')
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def _read_parameters(self):
        """Read all parameters into instance variables."""
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

    # ------------------------------------------------------------------ #
    # TTS engine
    # ------------------------------------------------------------------ #

    def _init_tts(self) -> bool:
        """Initialize the pyttsx3 engine. Returns True on success."""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.rate)
            self.tts_engine.setProperty('volume', self.volume)
            if self.voice_id:
                self.tts_engine.setProperty('voice', self.voice_id)
            self.get_logger().info('pyttsx3 TTS engine initialized')
            return True
        except ImportError:
            self.get_logger().error(
                'pyttsx3 not installed! Run: pip3 install pyttsx3'
            )
        except Exception as e:
            self.get_logger().error(f'TTS init failed: {e}')
        return False

    def _speaker_loop(self):
        """Dedicated thread that pulls text from the queue and speaks it.

        This avoids pyttsx3 threading issues — the engine is only ever
        accessed from this single thread.
        """
        while not self._shutdown_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.tts_engine is None:
                continue

            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.get_logger().error(f'TTS speak error: {e}')

    def _stop_speaker(self):
        """Signal the speaker thread to stop and wait for it."""
        self._shutdown_event.set()
        if self._speaker_thread is not None and self._speaker_thread.is_alive():
            self._speaker_thread.join(timeout=3.0)
        self.tts_engine = None

    def _enqueue_speech(self, text: str):
        """Add text to the speech queue (non-blocking, drops if full)."""
        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            self.get_logger().warn('Speech queue full — dropping utterance')

    # ------------------------------------------------------------------ #
    # Detection callback
    # ------------------------------------------------------------------ #

    def _detection_callback(self, msg: DetectedObjectArray):
        """Process detected objects and announce them."""
        now = time.time()
        if now - self.last_announcement_time < self.cooldown:
            return

        if not msg.objects:
            return

        # Filter by confidence
        filtered = [
            obj for obj in msg.objects
            if obj.confidence >= self.min_confidence
        ]

        if not filtered:
            return

        # Get unique labels (preserve order, cap count)
        labels = []
        seen = set()
        for obj in filtered:
            if obj.label not in seen:
                labels.append(obj.label)
                seen.add(obj.label)
            if len(labels) >= self.max_objects:
                break

        # Check if anything new
        current_set = set(labels)
        if self.announce_new_only and current_set == self.last_announced_labels:
            return

        self.last_announced_labels = current_set
        self.last_announcement_time = now

        # Build and enqueue sentence
        sentence = self._build_sentence(labels)
        self.get_logger().info(f'Announcing: "{sentence}"')
        self._enqueue_speech(sentence)

    # ------------------------------------------------------------------ #
    # Manual speak
    # ------------------------------------------------------------------ #

    def _manual_speak_callback(self, msg: String):
        """Speak arbitrary text published to /tts/speak."""
        text = msg.data.strip()
        if text:
            self.get_logger().info(f'Manual TTS: "{text}"')
            self._enqueue_speech(text)

    # ------------------------------------------------------------------ #
    # Sentence building
    # ------------------------------------------------------------------ #

    def _build_sentence(self, labels: list) -> str:
        """Build a natural-language sentence from object labels.

        Examples:
            ["cup"] -> "I can see a cup"
            ["cup", "laptop"] -> "I can see a cup and a laptop"
            ["cup", "laptop", "phone"] -> "I can see a cup, a laptop, and a phone"
        """
        if not labels:
            return ''

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
        if not label:
            return 'an unknown object'
        article = 'an' if label[0].lower() in 'aeiou' else 'a'
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
