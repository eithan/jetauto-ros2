"""
Object detection node using YOLOv8 (ultralytics).

Subscribes to a camera image topic, runs inference, and publishes
detected objects as JSON on /detected_objects.

Optionally publishes an annotated image with bounding boxes.
"""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class DetectorNode(Node):
    """ROS2 node for real-time object detection via YOLOv8."""

    def __init__(self):
        super().__init__('detector_node')

        # -- Declare parameters --
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', '0')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('image_encoding', 'bgr8')
        self.declare_parameter('inference_interval', 0.2)
        self.declare_parameter('max_detections', 10)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('annotated_image_topic', '/detected_objects/image')

        # -- Read parameters --
        self.model_name = self.get_parameter('model_name').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.device = self.get_parameter('device').value
        self.image_topic = self.get_parameter('image_topic').value
        self.image_encoding = self.get_parameter('image_encoding').value
        self.inference_interval = self.get_parameter('inference_interval').value
        self.max_detections = self.get_parameter('max_detections').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.annotated_topic = self.get_parameter('annotated_image_topic').value

        # -- Load YOLO model (deferred import so node starts even if ultralytics missing) --
        self.model = None
        self._load_model()

        # -- ROS2 plumbing --
        self.bridge = CvBridge()
        self.last_inference_time = 0.0

        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._image_callback, 10
        )
        self.pub_detections = self.create_publisher(String, '/detected_objects', 10)

        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, self.annotated_topic, 10
            )

        self.get_logger().info(
            f'DetectorNode ready — model={self.model_name}, '
            f'conf={self.conf_threshold}, device={self.device}, '
            f'topic={self.image_topic}'
        )

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load the YOLO model. Logs error but doesn't crash if unavailable."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            # Warm up with a dummy image
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            self.get_logger().info(f'YOLO model "{self.model_name}" loaded on device={self.device}')
        except ImportError:
            self.get_logger().error(
                'ultralytics not installed! Run: pip3 install ultralytics'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image):
        """Process incoming camera frames at the configured interval."""
        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        if self.model is None:
            return

        # Convert ROS Image → OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
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

        # Publish detections as JSON
        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.pub_detections.publish(det_msg)

        # Publish annotated image
        if self.publish_annotated and results:
            try:
                annotated = results[0].plot()  # ultralytics built-in visualization
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = msg.header
                self.pub_annotated.publish(ann_msg)
            except Exception as e:
                self.get_logger().warn(f'Annotated image error: {e}')

        if detections:
            labels = [d['label'] for d in detections]
            self.get_logger().debug(f'Detected: {labels}')

    # ------------------------------------------------------------------ #
    # Result parsing
    # ------------------------------------------------------------------ #

    def _parse_results(self, results) -> list:
        """Convert YOLO results to a list of detection dicts."""
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

            detections.append({
                'label': label,
                'confidence': round(conf, 3),
                'bbox': {
                    'x1': round(x1, 1),
                    'y1': round(y1, 1),
                    'x2': round(x2, 1),
                    'y2': round(y2, 1),
                },
            })

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
