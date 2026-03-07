"""
Object detection node using YOLOv8 (ultralytics).

Subscribes to a camera image topic, runs inference, and publishes
detected objects as DetectedObjectArray on /detected_objects.

Optionally publishes an annotated image with bounding boxes.

Supports ROS2 lifecycle management for clean startup/shutdown.
"""

import time

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from std_msgs.msg import Bool
from jetauto_msgs.msg import DetectedObject, DetectedObjectArray


class DetectorNode(LifecycleNode):
    """ROS2 lifecycle node for real-time object detection via YOLOv8."""

    def __init__(self):
        super().__init__('detector_node')

        # -- Declare parameters (available before configure) --
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', '0')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('image_encoding', 'bgr8')
        self.declare_parameter('inference_interval', 0.2)
        self.declare_parameter('max_detections', 10)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('annotated_image_topic', '/detected_objects/image')
        self.declare_parameter('start_enabled', True)

        # -- Internal state --
        self.model = None
        self.bridge = CvBridge()
        self.last_inference_time = 0.0
        self.sub_image = None
        self.pub_detections = None
        self.pub_annotated = None

        # -- Voice control enable/disable --
        # Subscribes to /jetauto/detection/enable (std_msgs/Bool)
        # Can be toggled at any time, even before activation.
        self.enabled = self.get_parameter('start_enabled').value
        self.create_subscription(Bool, '/jetauto/detection/enable', self._enable_callback, 1)

        self.get_logger().info('DetectorNode created (inactive — waiting for configure)')

    def _enable_callback(self, msg: Bool):
        """Toggle detection on/off via voice or external command."""
        self.enabled = msg.data
        state = 'ENABLED' if self.enabled else 'DISABLED'
        self.get_logger().info(f'Detection {state} via /jetauto/detection/enable')

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks
    # ------------------------------------------------------------------ #

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Load parameters and YOLO model."""
        self._read_parameters()

        if not self._load_model():
            return TransitionCallbackReturn.FAILURE

        self.pub_detections = self.create_publisher(
            DetectedObjectArray, '/detected_objects', 10
        )
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, self.annotated_topic, 10
            )

        self.get_logger().info(
            f'Configured — model={self.model_name}, conf={self.conf_threshold}, '
            f'device={self.device}, topic={self.image_topic}'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Create the image subscription and start processing."""
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._image_callback, 10
        )
        self.get_logger().info('Activated — listening for images')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop processing images."""
        if self.sub_image is not None:
            self.destroy_subscription(self.sub_image)
            self.sub_image = None
        self.get_logger().info('Deactivated — stopped listening')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release model and publishers."""
        self.model = None
        if self.pub_detections is not None:
            self.destroy_publisher(self.pub_detections)
            self.pub_detections = None
        if self.pub_annotated is not None:
            self.destroy_publisher(self.pub_annotated)
            self.pub_annotated = None
        self.get_logger().info('Cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        self.model = None
        self.get_logger().info('Shut down')
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def _read_parameters(self):
        """Read all parameters into instance variables."""
        self.model_name = self.get_parameter('model_name').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.device = self.get_parameter('device').value
        self.image_topic = self.get_parameter('image_topic').value
        self.image_encoding = self.get_parameter('image_encoding').value
        self.inference_interval = self.get_parameter('inference_interval').value
        self.max_detections = self.get_parameter('max_detections').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.annotated_topic = self.get_parameter('annotated_image_topic').value

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> bool:
        """Load the YOLO model. Returns True on success."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            self.get_logger().info(
                f'YOLO model "{self.model_name}" loaded on device={self.device}'
            )
            return True
        except ImportError:
            self.get_logger().error(
                'ultralytics not installed! Run: pip3 install ultralytics'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
        return False

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image):
        """Process incoming camera frames at the configured interval."""
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        if self.model is None:
            return

        # Convert ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.image_encoding
            )
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        # Run inference
        results = self.model.predict(
            cv_image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = self._parse_results(results)

        # Publish typed message
        det_msg = DetectedObjectArray()
        det_msg.header = Header()
        det_msg.header.stamp = msg.header.stamp
        det_msg.header.frame_id = msg.header.frame_id
        det_msg.objects = detections
        self.pub_detections.publish(det_msg)

        # Publish annotated image
        if self.publish_annotated and self.pub_annotated and results:
            try:
                annotated = results[0].plot()
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = msg.header
                self.pub_annotated.publish(ann_msg)
            except Exception as e:
                self.get_logger().warn(f'Annotated image error: {e}')

        if detections:
            labels = [d.label for d in detections]
            self.get_logger().debug(f'Detected: {labels}')

    # ------------------------------------------------------------------ #
    # Result parsing
    # ------------------------------------------------------------------ #

    def _parse_results(self, results) -> list:
        """Convert YOLO results to a list of DetectedObject messages."""
        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for i, box in enumerate(boxes):
            if i >= self.max_detections:
                break

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names.get(cls_id, f'class_{cls_id}')
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            obj = DetectedObject()
            obj.label = label
            obj.confidence = round(conf, 3)
            obj.x1 = round(x1, 1)
            obj.y1 = round(y1, 1)
            obj.x2 = round(x2, 1)
            obj.y2 = round(y2, 1)
            detections.append(obj)

        return detections


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
