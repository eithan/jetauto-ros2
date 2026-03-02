"""
JetAuto Simulation Node — Static Image Publisher
─────────────────────────────────────────────────
Cycles through a manifest of test images, publishing each on the camera
topic for 5 seconds, then aggregates detections and logs results.

Compatible with ROS2 Humble and Jazzy.

Log format:
  [sim] displaying <description>
  [jetauto] I see <objects>.
  [sim] completed cycling through N images. Rate of success: X%

Tail the log:
  tail -f /tmp/jetauto_sim.log
"""

import json
import os
import sys
import time

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from jetauto_msgs.msg import DetectedObjectArray

# ── Human-readable label map (YOLO → display name) ──────────────────────── #
LABEL_DISPLAY = {
    'person':       ('person',    'people'),
    'dog':          ('dog',       'dogs'),
    'cat':          ('cat',       'cats'),
    'bird':         ('bird',      'birds'),
    'car':          ('car',       'cars'),
    'dining table': ('table',     'tables'),
    'chair':        ('chair',     'chairs'),
    'truck':        ('truck',     'trucks'),
    'bicycle':      ('bicycle',   'bicycles'),
    'motorcycle':   ('motorcycle','motorcycles'),
    'bus':          ('bus',       'buses'),
    'bench':        ('bench',     'benches'),
    'horse':        ('horse',     'horses'),
    'cow':          ('cow',       'cows'),
    'sheep':        ('sheep',     'sheep'),
    'elephant':     ('elephant',  'elephants'),
    'bear':         ('bear',      'bears'),
    'laptop':       ('laptop',    'laptops'),
    'cell phone':   ('phone',     'phones'),
    'backpack':     ('backpack',  'backpacks'),
    'umbrella':     ('umbrella',  'umbrellas'),
    'bottle':       ('bottle',    'bottles'),
    'cup':          ('cup',       'cups'),
    'tv':           ('TV',        'TVs'),
    'sofa':         ('couch',     'couches'),
    'couch':        ('couch',     'couches'),
    'bed':          ('bed',       'beds'),
    'toilet':       ('toilet',    'toilets'),
    'potted plant': ('plant',     'plants'),
    'sports ball':  ('ball',      'balls'),
    'kite':         ('kite',      'kites'),
    'airplane':     ('airplane',  'airplanes'),
    'train':        ('train',     'trains'),
    'boat':         ('boat',      'boats'),
    'traffic light':('traffic light', 'traffic lights'),
    'stop sign':    ('stop sign', 'stop signs'),
    'fire hydrant': ('fire hydrant', 'fire hydrants'),
    'pizza':        ('pizza',     'pizzas'),
    'hot dog':      ('hot dog',   'hot dogs'),
    'sandwich':     ('sandwich',  'sandwiches'),
    'cake':         ('cake',      'cakes'),
    'donut':        ('donut',     'donuts'),
    'banana':       ('banana',    'bananas'),
    'apple':        ('apple',     'apples'),
    'orange':       ('orange',    'oranges'),
    'fork':         ('fork',      'forks'),
    'knife':        ('knife',     'knives'),
    'spoon':        ('spoon',     'spoons'),
    'bowl':         ('bowl',      'bowls'),
    'book':         ('book',      'books'),
    'clock':        ('clock',     'clocks'),
    'vase':         ('vase',      'vases'),
    'scissors':     ('scissors',  'pairs of scissors'),
    'teddy bear':   ('teddy bear','teddy bears'),
}

LOG_FILE = '/tmp/jetauto_sim.log'


def labels_to_sentence(label_set: set) -> str:
    """Convert a set of YOLO label strings to a readable sentence fragment."""
    if not label_set:
        return 'nothing'

    parts = []
    for label in sorted(label_set):
        singular, _ = LABEL_DISPLAY.get(label, (label, label + 's'))
        parts.append(f'a {singular}')

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f'{parts[0]} and {parts[1]}'
    return ', '.join(parts[:-1]) + f', and {parts[-1]}'


class SimNode(Node):
    """Publishes test images and evaluates detector output."""

    IMAGE_INTERVAL = 5.0   # seconds per image
    PUB_RATE       = 0.1   # seconds between image re-publishes (~10 Hz)

    def __init__(self):
        super().__init__('jetauto_sim')

        # ── Load manifest ──────────────────────────────────────────────── #
        pkg_share = get_package_share_directory('jetauto_sim')
        manifest_path = os.path.join(pkg_share, 'manifest.json')
        images_dir    = os.path.join(pkg_share, 'images')

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.images_dir = images_dir
        self.bridge     = CvBridge()

        # ── ROS comms ─────────────────────────────────────────────────── #
        self.pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.sub = self.create_subscription(
            DetectedObjectArray, '/detected_objects',
            self._detection_callback, 10
        )

        # ── State ─────────────────────────────────────────────────────── #
        self.current_index      = 0
        self.current_image_msg  = None
        self.detected_labels    = set()   # union of labels seen this window
        self.successes          = 0
        self.total              = 0
        self.skipped            = 0

        # ── Timers ────────────────────────────────────────────────────── #
        self.pub_timer     = self.create_timer(self.PUB_RATE, self._publish_frame)
        self.advance_timer = self.create_timer(self.IMAGE_INTERVAL, self._advance)

        # ── Start ─────────────────────────────────────────────────────── #
        self._log_sim(f'Starting simulation — {len(self.manifest)} images, '
                      f'{int(self.IMAGE_INTERVAL)}s each')
        self._log_sim(f'Tail output: tail -f {LOG_FILE}')
        self._load_image(0)

    # ──────────────────────────────────────────────────────────────────── #
    # Image loading
    # ──────────────────────────────────────────────────────────────────── #

    def _load_image(self, index: int):
        entry    = self.manifest[index]
        img_path = os.path.join(self.images_dir, entry['file'])

        if not os.path.exists(img_path):
            self._log_sim(f'⚠️  missing {entry["file"]} — skipping')
            self.current_image_msg = None
            self.skipped += 1
            return

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            self._log_sim(f'⚠️  failed to read {entry["file"]} — skipping')
            self.current_image_msg = None
            self.skipped += 1
            return

        self.current_image_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        self.detected_labels   = set()
        self._log_sim(f'displaying {entry["description"]}')

    # ──────────────────────────────────────────────────────────────────── #
    # Timers
    # ──────────────────────────────────────────────────────────────────── #

    def _publish_frame(self):
        if self.current_image_msg is None:
            return
        self.current_image_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.current_image_msg)

    def _advance(self):
        entry = self.manifest[self.current_index]

        if self.current_image_msg is None:
            # Image was skipped — don't evaluate
            pass
        else:
            # Report detections
            self.total += 1
            sentence = labels_to_sentence(self.detected_labels)
            self._log_jetauto(f'I see {sentence}.')

            # Evaluate success
            expected = set(entry.get('expected_yolo', []))
            if expected and expected.issubset(self.detected_labels):
                self.successes += 1

        # Advance or finish
        self.current_index += 1
        if self.current_index >= len(self.manifest):
            self._finish()
        else:
            self._load_image(self.current_index)

    def _finish(self):
        self.pub_timer.cancel()
        self.advance_timer.cancel()

        rate = int(100 * self.successes / self.total) if self.total else 0
        skip_note = f' ({self.skipped} skipped)' if self.skipped else ''
        self._log_sim(
            f'completed cycling through {len(self.manifest)} images{skip_note}. '
            f'Rate of success: {rate}%'
        )
        self.get_logger().info('Simulation complete — shutting down.')
        rclpy.shutdown()

    # ──────────────────────────────────────────────────────────────────── #
    # Detection callback
    # ──────────────────────────────────────────────────────────────────── #

    def _detection_callback(self, msg: DetectedObjectArray):
        for obj in msg.objects:
            self.detected_labels.add(obj.label)

    # ──────────────────────────────────────────────────────────────────── #
    # Logging
    # ──────────────────────────────────────────────────────────────────── #

    def _log_sim(self, text: str):
        line = f'[sim] {text}'
        print(line, flush=True)
        self.get_logger().info(line)
        self._write_log(line)

    def _log_jetauto(self, text: str):
        line = f'[jetauto] {text}'
        print(line, flush=True)
        self.get_logger().info(line)
        self._write_log(line)

    @staticmethod
    def _write_log(line: str):
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(f'[{time.strftime("%H:%M:%S")}] {line}\n')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
