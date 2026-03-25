"""
Face enrollment ROS2 node for JetAuto dashboard-guided face enrollment.

Guides the user through a 5-pose photo sequence using live camera frames
and InsightFace. Reports real-time status to the dashboard over a ROS topic.

Commands (subscribe /faces/enroll/command, String):
    start:<name>  — begin enrollment for person named <name>
    cancel        — abort current enrollment

Status (publish /faces/enroll/status, String — JSON payload):
    {"state":"idle"}
    {"state":"loading","name":"Alice","message":"Loading model..."}
    {"state":"ready","name":"Alice","progress":0,"total":5,"face_visible":false,"message":"..."}
    {"state":"capturing","name":"Alice","progress":2,"total":5,"face_visible":true,"message":"..."}
    {"state":"done","name":"Alice","message":"Enrolled successfully!"}
    {"state":"error","message":"..."}
    {"state":"cancelled"}
"""

import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


# Spoken pose guidance — index N is the prompt after capture N (to set up capture N+1).
_POSE_PROMPTS = [
    "Now turn slightly to your left.",
    "Now face a little to your right.",
    "Now tilt your head up a little.",
    "One more — look straight at me.",
]


class EnrollmentNode(Node):
    """Dashboard-guided interactive face enrollment via InsightFace."""

    def __init__(self):
        super().__init__('face_enrollment_node')

        # -- Parameters --
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('image_encoding', 'bgr8')
        self.declare_parameter(
            'faces_db_path',
            '/home/ubuntu/ros2_ws/src/jetauto-ros2/data/faces',
        )
        self.declare_parameter('model_name', 'buffalo_l')
        self.declare_parameter('det_size', 640)
        self.declare_parameter('gpu_id', 0)
        self.declare_parameter('num_captures', 5)
        self.declare_parameter('tts_topic', '/tts/speak')
        self.declare_parameter('face_stable_seconds', 2.0)

        # -- Internal state --
        self.bridge = CvBridge()
        self.app = None            # InsightFace FaceAnalysis (loaded on demand)

        self._state: str = 'idle'
        self._name: str = ''
        self._captures: list = []  # accumulated face embeddings
        self._face_visible: bool = False
        self._face_stable_since: float | None = None
        self._sub_image = None
        self._enc: str = 'bgr8'

        self._frame_lock = threading.Lock()
        self._latest_frame = None

        # -- Publishers --
        self.pub_status = self.create_publisher(String, '/faces/enroll/status', 10)
        self.pub_tts = self.create_publisher(
            String, self.get_parameter('tts_topic').value, 1)
        self.pub_reload = self.create_publisher(String, '/faces/reload', 1)

        # -- Subscribers --
        self.create_subscription(
            String, '/faces/enroll/command', self._on_command, 10)

        # 10 Hz processing loop
        self.create_timer(0.1, self._tick)

        self.get_logger().info('EnrollmentNode ready')
        self._publish_status()

    # ── Helpers ─────────────────────────────────────────────────────

    def _publish_status(self, message: str = ''):
        payload: dict = {'state': self._state}
        if self._name:
            payload['name'] = self._name
        if self._state in ('ready', 'capturing'):
            payload['progress'] = len(self._captures)
            payload['total'] = self.get_parameter('num_captures').value
            payload['face_visible'] = self._face_visible
        if message:
            payload['message'] = message
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_status.publish(msg)

    def _speak(self, text: str):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)

    # ── Command handler ──────────────────────────────────────────────

    def _on_command(self, msg: String):
        cmd = msg.data.strip()
        if cmd.startswith('start:'):
            name = cmd[6:].strip()
            if name:
                self._begin_enrollment(name)
            else:
                self._publish_status('Name cannot be empty')
        elif cmd == 'cancel':
            self._cancel()

    # ── Enrollment state machine ─────────────────────────────────────

    def _begin_enrollment(self, name: str):
        """Reset state and kick off model loading in a background thread."""
        self._name = name
        self._captures = []
        self._face_visible = False
        self._face_stable_since = None
        self._state = 'loading'
        self._publish_status('Loading face model...')
        self.get_logger().info(f'Starting enrollment for "{name}"')
        threading.Thread(target=self._load_and_start, daemon=True).start()

    def _load_and_start(self):
        """Load InsightFace model (runs in background thread)."""
        try:
            self.get_logger().info('Importing insightface...')
            from insightface.app import FaceAnalysis
            self.get_logger().info('insightface imported OK')

            model = self.get_parameter('model_name').value
            gpu_id = self.get_parameter('gpu_id').value
            det_size = self.get_parameter('det_size').value

            self.get_logger().info(f'Loading model "{model}" (gpu_id={gpu_id}) — may take 1-3 min on first run')
            providers = (
                [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider']
                if gpu_id >= 0 else ['CPUExecutionProvider']
            )
            self.get_logger().info('Creating FaceAnalysis instance...')
            self.app = FaceAnalysis(name=model, providers=providers)
            self.get_logger().info('Running app.prepare() — loading ONNX models...')
            self.app.prepare(ctx_id=gpu_id, det_size=(det_size, det_size))
            self.get_logger().info('Model ready!')

            # Subscribe to camera now that model is ready
            topic = self.get_parameter('image_topic').value
            self._enc = self.get_parameter('image_encoding').value
            self._sub_image = self.create_subscription(
                Image, topic, self._on_image, 10)

            n = self.get_parameter('num_captures').value
            self._state = 'ready'
            self._publish_status(
                f"Hi {self._name}! I'll take {n} photos. "
                "Face the camera and hold still."
            )
            self._speak(
                f"Hi {self._name}! I'll take {n} photos of your face. "
                "Please face the camera and hold still."
            )

        except ImportError:
            self._state = 'error'
            self._publish_status('insightface not installed on this machine')
            self.get_logger().error('insightface not installed')
        except Exception as e:
            self._state = 'error'
            self._publish_status(f'Model load failed: {e}')
            self.get_logger().error(f'Model load error: {e}')

    def _on_image(self, msg: Image):
        """Buffer the latest camera frame (non-blocking)."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self._enc)
            with self._frame_lock:
                self._latest_frame = frame
        except Exception:
            pass

    def _tick(self):
        """10 Hz loop — detect face presence and auto-capture when stable."""
        if self._state not in ('ready', 'capturing') or self.app is None:
            return

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            return

        try:
            faces = self.app.get(frame)
        except Exception as e:
            self.get_logger().debug(f'InsightFace error: {e}')
            return

        valid = [f for f in faces if f.embedding is not None]
        face_now = bool(valid)
        was = self._face_visible
        self._face_visible = face_now

        if face_now and not was:
            # Face just appeared
            self._face_stable_since = time.time()
            self._publish_status('Face detected! Hold still...')
        elif not face_now and was:
            # Face disappeared
            self._face_stable_since = None
            self._publish_status("Move closer — I can't see your face.")
        else:
            # Periodic status push so browser stays current
            self._publish_status()

        # Auto-capture once face has been stable long enough
        stable_secs = self.get_parameter('face_stable_seconds').value
        if (
            face_now
            and self._face_stable_since is not None
            and (time.time() - self._face_stable_since) >= stable_secs
        ):
            self._capture(valid[0])

    def _capture(self, face):
        """Record one face embedding and prompt for the next pose."""
        self._face_stable_since = None  # reset timer before next capture
        self._captures.append(face.embedding.copy())
        n = len(self._captures)
        total = self.get_parameter('num_captures').value
        self._state = 'capturing'

        self.get_logger().info(f'Enrollment capture {n}/{total} for "{self._name}"')

        if n < total:
            prompt_idx = n - 1  # index 0 = prompt after 1st capture
            prompt = (
                _POSE_PROMPTS[prompt_idx]
                if prompt_idx < len(_POSE_PROMPTS)
                else "Hold another pose."
            )
            self._speak(f"Got it! {prompt}")
            self._publish_status(f'{n} of {total} captured. {prompt}')
        else:
            self._finish()

    def _finish(self):
        """Average embeddings, save .npz, and notify the recognition node."""
        db_path = self.get_parameter('faces_db_path').value
        os.makedirs(db_path, exist_ok=True)

        all_embs = np.array(self._captures, dtype=np.float32)
        avg = np.mean(all_embs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm

        safe = self._name.lower().replace(' ', '_')
        filepath = os.path.join(db_path, f'{safe}.npz')
        np.savez(
            filepath,
            name=self._name,
            embedding=avg,
            num_samples=len(self._captures),
        )
        self.get_logger().info(f'Enrolled "{self._name}" → {filepath}')

        # Signal recognition node to hot-reload
        reload_msg = String()
        reload_msg.data = 'reload'
        self.pub_reload.publish(reload_msg)

        # Final state + TTS
        self._state = 'done'
        self._speak(
            f"Done! I've learned your face, {self._name}. "
            "I'll recognize you from now on."
        )
        self._publish_status(
            f'Enrolled "{self._name}" with {len(self._captures)} samples!'
        )

        # Stop consuming camera frames
        self._unsubscribe_image()

    def _cancel(self):
        """Abort enrollment cleanly."""
        self._unsubscribe_image()
        self._state = 'cancelled'
        self._name = ''
        self._captures = []
        self._face_visible = False
        self._face_stable_since = None
        self._publish_status()
        self.get_logger().info('Enrollment cancelled')

    def _unsubscribe_image(self):
        if self._sub_image is not None:
            try:
                self.destroy_subscription(self._sub_image)
            except Exception:
                pass
            self._sub_image = None


def main(args=None):
    rclpy.init(args=args)
    node = EnrollmentNode()
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
            pass


if __name__ == '__main__':
    main()
