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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String

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
        self.declare_parameter('scene_forget_timeout', 30.0)  # seconds before resetting known labels
        self.declare_parameter('greeting', 'I can see')
        self.declare_parameter('announce_detections', True)
        self.declare_parameter('caption_greeting', 'I can see')

        # -- State --
        self.last_announcement_time = 0.0
        self._caption_active = False  # set True when first caption arrives
        # Per-label state: {label: {'last_seen': float, 'count': int, 'announced_count': int}}
        # Replaces the old flat set — tracks instance counts and per-label expiry.
        self._label_state: dict = {}
        self.tts_engine = None
        self._speech_queue = queue.Queue(maxsize=10)
        self._speaker_thread = None
        self._shutdown_event = threading.Event()
        self._speaking_pub = None        # set in on_activate
        self._voice_state_pub = None     # set in on_activate

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
        self.sub_caption = self.create_subscription(
            String, '/scene_caption', self._caption_callback, 1
        )
        # Publish speaking state so the voice commander can mute the mic
        self._speaking_pub = self.create_publisher(Bool, '/tts/speaking', 1)
        # Publish voice state so the dashboard face animates during TTS.
        # TRANSIENT_LOCAL: late-joining subscribers (dashboard) get the last
        # value, which prevents losing the very first 'speaking' publish.
        _qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._voice_state_pub = self.create_publisher(String, '/jetauto/voice/state', _qos_latched)
        self.get_logger().info('Activated — listening for detections')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop listening."""
        if hasattr(self, 'sub_detections'):
            self.destroy_subscription(self.sub_detections)
        if hasattr(self, 'sub_manual'):
            self.destroy_subscription(self.sub_manual)
        if hasattr(self, 'sub_caption'):
            self.destroy_subscription(self.sub_caption)
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
        self.scene_forget_timeout = self.get_parameter('scene_forget_timeout').value
        self.greeting = self.get_parameter('greeting').value
        self.announce_detections = self.get_parameter('announce_detections').value
        self.caption_greeting = self.get_parameter('caption_greeting').value

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
                self._publish_speaking(True)
                self._publish_voice_state('speaking')
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                time.sleep(0.8)   # room reverb buffer — mic drains while this runs
            except Exception as e:
                self.get_logger().error(f'TTS speak error: {e}')
            finally:
                self._publish_speaking(False)
                self._publish_voice_state('idle')

    def _stop_speaker(self):
        """Signal the speaker thread to stop and wait for it."""
        self._shutdown_event.set()
        if self._speaker_thread is not None and self._speaker_thread.is_alive():
            self._speaker_thread.join(timeout=3.0)
        self.tts_engine = None

    def _publish_speaking(self, speaking: bool) -> None:
        """Publish /tts/speaking so voice commander can mute the mic."""
        if self._speaking_pub is not None:
            msg = Bool()
            msg.data = speaking
            self._speaking_pub.publish(msg)

    def _publish_voice_state(self, state: str) -> None:
        """Publish /jetauto/voice/state so the dashboard face animates."""
        if self._voice_state_pub is not None:
            msg = String()
            msg.data = state
            self._voice_state_pub.publish(msg)

    def _enqueue_speech(self, text: str):
        """Add text to the speech queue (non-blocking, drops if full)."""
        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            self.get_logger().warn('Speech queue full — dropping utterance')

    # ------------------------------------------------------------------ #
    # Detection callback
    # ------------------------------------------------------------------ #

    def _caption_callback(self, msg: String):
        """Speak a scene caption from Florence-2.

        Replaces YOLO detection announcements with natural-language
        descriptions when the caption pipeline is active.
        """
        text = msg.data.strip()
        if not text:
            return

        # Mark caption pipeline as active — suppresses YOLO announcements
        self._caption_active = True

        # Strip common third-person prefixes and convert to first-person
        for prefix in (
            'The image shows ', 'The image depicts ', 'The image features ',
            'In this image, ', 'In the image, ', 'This image shows ',
            'The picture shows ', 'The photo shows ',
        ):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
            lower = text.lower()
            if lower.startswith(prefix.lower()):
                text = text[len(prefix):]
                break

        # Capitalize first letter and prepend greeting
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        announcement = f'{self.caption_greeting} {text}' if self.caption_greeting else text

        self.get_logger().info(f'Scene caption: "{announcement}"')
        self._enqueue_speech(announcement)

    def _detection_callback(self, msg: DetectedObjectArray):
        """Process detected objects and announce them.

        Announcement logic:
        - Suppressed when Florence-2 caption pipeline is active.
        - A label is announced when first seen (announced_count == 0).
        - It is announced again when the instance count increases
          (e.g. 1 person → 2 people in frame simultaneously).
        - Per-label expiry: if a label hasn't been seen for
          scene_forget_timeout seconds it is evicted. When it reappears it
          is treated as new and announced again (handles "person leaves,
          different person enters" across frames).
        - The global cooldown still prevents spam — all announcement
          decisions still respect self.cooldown.
        """
        now = time.time()

        # Skip YOLO announcements when Florence-2 captions are active
        if self._caption_active or not self.announce_detections:
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

        # Evict labels not seen within scene_forget_timeout
        for label in list(self._label_state.keys()):
            if now - self._label_state[label]['last_seen'] > self.scene_forget_timeout:
                self.get_logger().debug(f'Label "{label}" expired from scene memory')
                del self._label_state[label]

        # Count instances per label in this frame
        frame_counts: dict = {}
        for obj in filtered:
            frame_counts[obj.label] = frame_counts.get(obj.label, 0) + 1

        # Update per-label state with current frame
        for label, count in frame_counts.items():
            if label not in self._label_state:
                self._label_state[label] = {
                    'last_seen': now,
                    'count': count,
                    'announced_count': 0,
                }
            else:
                self._label_state[label]['last_seen'] = now
                self._label_state[label]['count'] = count

        # Decide what to announce: new labels OR count increased
        to_announce = []
        for label, count in frame_counts.items():
            state = self._label_state[label]
            if state['announced_count'] == 0 or count > state['announced_count']:
                to_announce.append(label)

        if not to_announce:
            return

        # Respect global cooldown
        if now - self.last_announcement_time < self.cooldown:
            return

        # Mark announced counts
        for label in to_announce:
            self._label_state[label]['announced_count'] = frame_counts[label]

        self.last_announcement_time = now

        # Build and enqueue sentence (cap at max_objects)
        sentence = self._build_sentence(to_announce[:self.max_objects])
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
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down by signal handler


if __name__ == '__main__':
    main()
