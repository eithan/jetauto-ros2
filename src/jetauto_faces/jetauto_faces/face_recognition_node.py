"""
Face recognition node using InsightFace.

Subscribes to:
  - /detected_objects (DetectedObjectArray) — person bounding boxes from YOLO
  - camera image topic — raw frames for face cropping

When a "person" detection arrives, crops the region from the latest camera
frame, runs InsightFace detection + embedding extraction, compares against
enrolled face embeddings, and publishes recognized faces.

Optionally greets recognized people via the TTS node.

Supports ROS2 lifecycle management for clean startup/shutdown.
"""

import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

from jetauto_msgs.msg import DetectedObjectArray, RecognizedFace, RecognizedFaceArray


class FaceRecognitionNode(LifecycleNode):
    """ROS2 lifecycle node for face recognition via InsightFace."""

    def __init__(self):
        super().__init__('face_recognition_node')

        # -- Declare parameters --
        self.declare_parameter('image_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('image_encoding', 'bgr8')
        self.declare_parameter('detection_topic', '/detected_objects')
        self.declare_parameter('model_name', 'buffalo_l')
        self.declare_parameter('det_size', 640)
        self.declare_parameter('gpu_id', 0)
        self.declare_parameter('faces_db_path', '')
        self.declare_parameter('recognition_threshold', 0.4)
        self.declare_parameter('min_face_size', 40)
        self.declare_parameter('recognition_interval', 0.5)
        self.declare_parameter('max_persons_per_frame', 5)
        self.declare_parameter('greeting_cooldown', 60.0)
        self.declare_parameter('greeting_topic', '/tts/speak')
        self.declare_parameter('auto_greet', True)
        self.declare_parameter('greeting_template', 'Hello {name}!')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('annotated_image_topic', '/recognized_faces/image')

        # -- Internal state --
        self.app = None  # InsightFace FaceAnalysis app
        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_header = None
        self.last_recognition_time = 0.0
        self._greeting_times: dict = {}  # {name: last_greeting_timestamp}

        # Enrolled face database: {name: np.ndarray (embedding)}
        self._face_db: dict = {}

        self.get_logger().info('FaceRecognitionNode created (inactive — waiting for configure)')

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks
    # ------------------------------------------------------------------ #

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load parameters, InsightFace model, and face database."""
        self._read_parameters()

        if not self._load_model():
            return TransitionCallbackReturn.FAILURE

        self._load_face_db()

        # Publishers
        self.pub_faces = self.create_publisher(
            RecognizedFaceArray, '/recognized_faces', 10
        )
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, self.annotated_topic, 10
            )
        if self.auto_greet:
            self.pub_greeting = self.create_publisher(
                String, self.greeting_topic, 10
            )

        self.get_logger().info(
            f'Configured — model={self.model_name}, '
            f'faces_db={self.faces_db_path} ({len(self._face_db)} enrolled), '
            f'threshold={self.recognition_threshold}'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Create subscriptions and start processing."""
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._image_callback, 10
        )
        self.sub_detections = self.create_subscription(
            DetectedObjectArray, self.detection_topic,
            self._detection_callback, 10
        )
        # Subscription to reload face database on-the-fly
        self.sub_reload = self.create_subscription(
            String, '/faces/reload', self._reload_callback, 1
        )
        self.get_logger().info('Activated — listening for person detections')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop processing."""
        for attr in ('sub_image', 'sub_detections', 'sub_reload'):
            if hasattr(self, attr):
                self.destroy_subscription(getattr(self, attr))
        self.get_logger().info('Deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release model."""
        self.app = None
        self._face_db.clear()
        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        self.app = None
        self.get_logger().info('Shut down')
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def _read_parameters(self):
        """Read all parameters into instance variables."""
        self.image_topic = self.get_parameter('image_topic').value
        self.image_encoding = self.get_parameter('image_encoding').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.det_size = self.get_parameter('det_size').value
        self.gpu_id = self.get_parameter('gpu_id').value
        self.faces_db_path = self.get_parameter('faces_db_path').value
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        self.min_face_size = self.get_parameter('min_face_size').value
        self.recognition_interval = self.get_parameter('recognition_interval').value
        self.max_persons = self.get_parameter('max_persons_per_frame').value
        self.greeting_cooldown = self.get_parameter('greeting_cooldown').value
        self.greeting_topic = self.get_parameter('greeting_topic').value
        self.auto_greet = self.get_parameter('auto_greet').value
        self.greeting_template = self.get_parameter('greeting_template').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.annotated_topic = self.get_parameter('annotated_image_topic').value

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> bool:
        """Load the InsightFace model. Returns True on success."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=self.model_name,
                providers=self._get_providers(),
            )
            self.app.prepare(
                ctx_id=self.gpu_id,
                det_size=(self.det_size, self.det_size),
            )

            self.get_logger().info(
                f'InsightFace model "{self.model_name}" loaded '
                f'(det_size={self.det_size}, gpu_id={self.gpu_id})'
            )
            return True

        except ImportError:
            self.get_logger().error(
                'insightface not installed! Run: '
                'pip3 install insightface onnxruntime-gpu'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load InsightFace: {e}')
        return False

    def _get_providers(self) -> list:
        """Return ONNX Runtime execution providers based on GPU config."""
        if self.gpu_id >= 0:
            return [
                ('CUDAExecutionProvider', {'device_id': self.gpu_id}),
                'CPUExecutionProvider',
            ]
        return ['CPUExecutionProvider']

    # ------------------------------------------------------------------ #
    # Face database
    # ------------------------------------------------------------------ #

    def _load_face_db(self):
        """Load enrolled face embeddings from the database directory.

        Each person is stored as a .npz file with:
          - 'embedding': averaged face embedding (512-d float32)
          - 'name': person's name
          - 'num_samples': number of images used for enrollment
        """
        self._face_db.clear()

        if not self.faces_db_path or not os.path.isdir(self.faces_db_path):
            self.get_logger().warn(
                f'Face database path not found: "{self.faces_db_path}". '
                'No faces enrolled — recognition will detect faces but not identify them.'
            )
            return

        for filename in os.listdir(self.faces_db_path):
            if not filename.endswith('.npz'):
                continue

            filepath = os.path.join(self.faces_db_path, filename)
            try:
                data = np.load(filepath, allow_pickle=True)
                name = str(data['name'])
                embedding = data['embedding'].astype(np.float32)
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                self._face_db[name] = embedding
                self.get_logger().info(
                    f'Loaded face: "{name}" '
                    f'({int(data.get("num_samples", 0))} enrollment samples)'
                )
            except Exception as e:
                self.get_logger().warn(f'Failed to load {filename}: {e}')

        self.get_logger().info(f'Face database loaded: {len(self._face_db)} people enrolled')

    def _reload_callback(self, msg: String):
        """Reload the face database on-the-fly."""
        self.get_logger().info('Reloading face database...')
        self._load_face_db()

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image):
        """Store the latest camera frame (non-blocking)."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.image_encoding
            )
            with self._frame_lock:
                self._latest_frame = cv_image
                self._frame_header = msg.header
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    # ------------------------------------------------------------------ #
    # Detection callback — main processing pipeline
    # ------------------------------------------------------------------ #

    def _detection_callback(self, msg: DetectedObjectArray):
        """Process YOLO detections: find person BBs, run face recognition."""
        now = time.time()

        # Rate-limit recognition
        if now - self.last_recognition_time < self.recognition_interval:
            return
        self.last_recognition_time = now

        if self.app is None:
            return

        # Filter for "person" detections only
        persons = [
            obj for obj in msg.objects
            if obj.label == 'person'
        ]
        if not persons:
            return

        # Get the latest frame
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            frame_header = self._frame_header

        # Sort by bounding box area (largest first — likely closest)
        persons.sort(
            key=lambda p: (p.x2 - p.x1) * (p.y2 - p.y1),
            reverse=True,
        )
        persons = persons[:self.max_persons]

        h, w = frame.shape[:2]
        recognized_faces = []

        for person in persons:
            # Expand person bounding box slightly for better face capture
            px1 = max(0, int(person.x1) - 10)
            py1 = max(0, int(person.y1) - 10)
            px2 = min(w, int(person.x2) + 10)
            py2 = min(h, int(person.y2) + 10)

            # Crop person region
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            # Run InsightFace on the cropped region
            try:
                faces = self.app.get(person_crop)
            except Exception as e:
                self.get_logger().debug(f'InsightFace error: {e}')
                continue

            for face in faces:
                bbox = face.bbox.astype(int)
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]

                # Skip tiny faces
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    continue

                # Map face bbox back to full image coordinates
                abs_x1 = float(px1 + bbox[0])
                abs_y1 = float(py1 + bbox[1])
                abs_x2 = float(px1 + bbox[2])
                abs_y2 = float(py1 + bbox[3])

                # Identify face
                name = 'unknown'
                confidence = 0.0

                if face.embedding is not None and len(self._face_db) > 0:
                    name, confidence = self._identify_face(face.embedding)

                face_msg = RecognizedFace()
                face_msg.name = name
                face_msg.confidence = round(confidence, 3)
                face_msg.x1 = round(abs_x1, 1)
                face_msg.y1 = round(abs_y1, 1)
                face_msg.x2 = round(abs_x2, 1)
                face_msg.y2 = round(abs_y2, 1)
                face_msg.person_x1 = round(person.x1, 1)
                face_msg.person_y1 = round(person.y1, 1)
                face_msg.person_x2 = round(person.x2, 1)
                face_msg.person_y2 = round(person.y2, 1)
                recognized_faces.append(face_msg)

                if name != 'unknown':
                    self.get_logger().info(
                        f'Recognized: {name} ({confidence:.2f})'
                    )
                    self._maybe_greet(name, now)

        # Publish results
        result_msg = RecognizedFaceArray()
        result_msg.header = Header()
        result_msg.header.stamp = frame_header.stamp if frame_header else self.get_clock().now().to_msg()
        result_msg.header.frame_id = frame_header.frame_id if frame_header else 'camera'
        result_msg.faces = recognized_faces
        self.pub_faces.publish(result_msg)

        # Publish annotated image
        if self.publish_annotated and hasattr(self, 'pub_annotated') and recognized_faces:
            self._publish_annotated(frame, recognized_faces, frame_header)

        if recognized_faces:
            names = [f.name for f in recognized_faces]
            self.get_logger().debug(f'Faces: {names}')

    # ------------------------------------------------------------------ #
    # Face identification
    # ------------------------------------------------------------------ #

    def _identify_face(self, embedding: np.ndarray) -> tuple:
        """Compare a face embedding against the enrolled database.

        Returns (name, confidence). If no match above threshold,
        returns ('unknown', 0.0).
        """
        # Normalize query embedding
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_name = 'unknown'
        best_score = 0.0

        for name, db_embedding in self._face_db.items():
            # Cosine similarity (both are already normalized)
            score = float(np.dot(embedding, db_embedding))

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.recognition_threshold:
            return best_name, best_score

        return 'unknown', 0.0

    # ------------------------------------------------------------------ #
    # Greeting
    # ------------------------------------------------------------------ #

    def _maybe_greet(self, name: str, now: float):
        """Send a greeting via TTS if cooldown has elapsed."""
        if not self.auto_greet or not hasattr(self, 'pub_greeting'):
            return

        last_greeted = self._greeting_times.get(name, 0.0)
        if now - last_greeted < self.greeting_cooldown:
            return

        self._greeting_times[name] = now
        greeting = self.greeting_template.replace('{name}', name)

        msg = String()
        msg.data = greeting
        self.pub_greeting.publish(msg)
        self.get_logger().info(f'Greeting: "{greeting}"')

    # ------------------------------------------------------------------ #
    # Annotated image
    # ------------------------------------------------------------------ #

    def _publish_annotated(self, frame: np.ndarray, faces: list, header):
        """Draw face bounding boxes and names on the frame."""
        annotated = frame.copy()

        for face in faces:
            x1, y1 = int(face.x1), int(face.y1)
            x2, y2 = int(face.x2), int(face.y2)

            # Color: green for known, red for unknown
            if face.name != 'unknown':
                color = (0, 255, 0)
                label = f'{face.name} ({face.confidence:.2f})'
            else:
                color = (0, 0, 255)
                label = 'unknown'

            # Draw face bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = header if header else Header()
            self.pub_annotated.publish(ann_msg)
        except Exception as e:
            self.get_logger().warn(f'Annotated image error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
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
