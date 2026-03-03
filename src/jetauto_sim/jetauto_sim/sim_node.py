"""
JetAuto Simulation Node — Scenario-Based Test Runner
─────────────────────────────────────────────────────
Publishes test images to the camera topic, evaluates detector output,
and produces a structured test report.

Each manifest entry supports:
  file            – image filename (from images/)
  description     – human-readable scene description
  expected_yolo   – labels that MUST be detected (YOLO class names)
  expected_counts – minimum count per label  e.g. {"person": 2}
  forbidden_yolo  – labels that must NOT appear (false-positive test)
  motion          – if true, applies random blur/jitter per published frame
  tags            – list of category strings for grouped reporting

Three independent pass/fail dimensions per scenario:
  detection_pass  – all expected_yolo labels detected at least once
  count_pass      – max detected count >= expected_counts (per label)
  fp_pass         – no forbidden_yolo labels detected

Tail the log:
  tail -f /tmp/jetauto_sim.log
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from jetauto_msgs.msg import DetectedObjectArray


LOG_FILE = '/tmp/jetauto_sim.log'


# ── Result dataclass ──────────────────────────────────────────────────────── #

@dataclass
class ScenarioResult:
    index: int
    description: str
    tags: List[str]
    expected: Set[str]
    expected_counts: Dict[str, int]
    forbidden: Set[str]
    detected_counts: Dict[str, int]   # label -> max count in any single frame
    motion: bool
    skipped: bool = False

    @property
    def detected_set(self) -> Set[str]:
        return set(self.detected_counts.keys())

    @property
    def detection_pass(self) -> bool:
        return self.expected.issubset(self.detected_set) if self.expected else True

    @property
    def count_pass(self) -> bool:
        for label, needed in self.expected_counts.items():
            if self.detected_counts.get(label, 0) < needed:
                return False
        return True

    @property
    def false_positives(self) -> Set[str]:
        return self.forbidden.intersection(self.detected_set)

    @property
    def fp_pass(self) -> bool:
        return len(self.false_positives) == 0

    @property
    def passed(self) -> bool:
        return (not self.skipped
                and self.detection_pass
                and self.count_pass
                and self.fp_pass)

    def summary_line(self) -> str:
        if self.skipped:
            return f'  #{self.index + 1:02d} SKIP  {self.description}'

        status  = 'PASS' if self.passed else 'FAIL'
        det_sym = '✓' if self.detection_pass else '✗'
        cnt_sym = '✓' if self.count_pass     else '✗'
        fp_sym  = '✓' if self.fp_pass        else '✗'

        details = []
        if not self.detection_pass:
            missing = sorted(self.expected - self.detected_set)
            details.append(f'missing={missing}')
        if not self.count_pass:
            for lbl, needed in self.expected_counts.items():
                actual = self.detected_counts.get(lbl, 0)
                if actual < needed:
                    details.append(f'{lbl}: got {actual}, need {needed}')
        if not self.fp_pass:
            details.append(f'false_pos={sorted(self.false_positives)}')

        detected_str = dict(sorted(self.detected_counts.items()))
        detail_str   = '  — ' + '; '.join(details) if details else ''
        motion_str   = ' [motion]' if self.motion else ''
        return (f'  #{self.index + 1:02d} {status}  '
                f'det={det_sym}  cnt={cnt_sym}  fp={fp_sym}  '
                f'{self.description}{motion_str}{detail_str}  '
                f'detected={detected_str}')


# ── Node ─────────────────────────────────────────────────────────────────── #

class SimNode(Node):
    """Publishes test images and evaluates detector output."""

    IMAGE_INTERVAL = 5.0   # seconds per image
    PUB_RATE       = 0.1   # seconds between re-publishes (~10 Hz)

    def __init__(self):
        super().__init__('jetauto_sim')

        # ── Load manifest ──────────────────────────────────────────────── #
        pkg_share       = get_package_share_directory('jetauto_sim')
        manifest_path   = os.path.join(pkg_share, 'manifest.json')
        self.images_dir = os.path.join(pkg_share, 'images')

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.bridge = CvBridge()

        # ── ROS comms ─────────────────────────────────────────────────── #
        self.pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.sub = self.create_subscription(
            DetectedObjectArray, '/detected_objects',
            self._detection_callback, 10
        )

        # ── State ─────────────────────────────────────────────────────── #
        self.current_index    = 0
        self.current_entry    = None
        self.current_cv_image = None           # numpy array, base (unblurred)
        self.frame_counts: Dict[str, int] = {} # label -> max count across frames
        self.results: List[ScenarioResult] = []

        # ── Timers ────────────────────────────────────────────────────── #
        self.pub_timer     = self.create_timer(self.PUB_RATE,       self._publish_frame)
        self.advance_timer = self.create_timer(self.IMAGE_INTERVAL, self._advance)

        self._log_sim(f'Starting simulation — {len(self.manifest)} scenarios, '
                      f'{int(self.IMAGE_INTERVAL)}s each')
        self._load_image(0)

    # ── Image loading ─────────────────────────────────────────────────── #

    def _load_image(self, index: int):
        entry    = self.manifest[index]
        img_path = os.path.join(self.images_dir, entry['file'])

        self.current_entry  = entry
        self.frame_counts   = {}

        if not os.path.exists(img_path):
            self._log_sim(f'⚠️  missing {entry["file"]} — skipping')
            self.current_cv_image = None
            return

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            self._log_sim(f'⚠️  failed to read {entry["file"]} — skipping')
            self.current_cv_image = None
            return

        self.current_cv_image = cv_img
        motion_tag = ' [motion]' if entry.get('motion', False) else ''
        self._log_sim(f'displaying: {entry["description"]}{motion_tag}')

    # ── Motion simulation ─────────────────────────────────────────────── #

    def _apply_motion(self, cv_img: np.ndarray) -> np.ndarray:
        """Simulate robot-in-motion: random blur, brightness jitter, translation."""
        # Gaussian blur with random odd kernel (mimics motion blur)
        k = random.choice([3, 5, 7, 9, 11])
        img = cv2.GaussianBlur(cv_img, (k, k), 0)
        # Brightness jitter
        delta = random.randint(-35, 35)
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        # Small random translation (camera shake)
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        h, w = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h))
        return img

    # ── Timers ────────────────────────────────────────────────────────── #

    def _publish_frame(self):
        if self.current_cv_image is None:
            return
        if self.current_entry and self.current_entry.get('motion', False):
            frame = self._apply_motion(self.current_cv_image)
        else:
            frame = self.current_cv_image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def _advance(self):
        entry   = self.current_entry or self.manifest[self.current_index]
        skipped = self.current_cv_image is None

        result = ScenarioResult(
            index           = self.current_index,
            description     = entry['description'],
            tags            = entry.get('tags', []),
            expected        = set(entry.get('expected_yolo', [])),
            expected_counts = entry.get('expected_counts', {}),
            forbidden       = set(entry.get('forbidden_yolo', [])),
            detected_counts = dict(self.frame_counts),
            motion          = entry.get('motion', False),
            skipped         = skipped,
        )
        self.results.append(result)
        self._log_sim(result.summary_line())

        self.current_index += 1
        if self.current_index >= len(self.manifest):
            self._finish()
        else:
            self._load_image(self.current_index)

    def _finish(self):
        self.pub_timer.cancel()
        self.advance_timer.cancel()
        self._print_summary()
        self.get_logger().info('Simulation complete — shutting down.')
        rclpy.shutdown()

    # ── Detection callback ────────────────────────────────────────────── #

    def _detection_callback(self, msg: DetectedObjectArray):
        """Track max per-class count seen in any single frame."""
        per_frame: Dict[str, int] = {}
        for obj in msg.objects:
            per_frame[obj.label] = per_frame.get(obj.label, 0) + 1
        for label, count in per_frame.items():
            if count > self.frame_counts.get(label, 0):
                self.frame_counts[label] = count

    # ── Summary ───────────────────────────────────────────────────────── #

    def _print_summary(self):
        valid   = [r for r in self.results if not r.skipped]
        skipped = [r for r in self.results if r.skipped]

        if not valid:
            self._log_sim('No valid scenarios to summarize.')
            return

        total      = len(valid)
        n_pass     = sum(1 for r in valid if r.passed)
        det_pass   = sum(1 for r in valid if r.detection_pass)
        count_pass = sum(1 for r in valid if r.count_pass)
        fp_pass    = sum(1 for r in valid if r.fp_pass)

        sep = '─' * 62
        self._log_sim(sep)
        self._log_sim('SIMULATION SUMMARY')
        self._log_sim(sep)
        self._log_sim(f'  Overall:       {n_pass}/{total} passed  ({_pct(n_pass, total)}%)')
        self._log_sim(f'  Detection:     {det_pass}/{total}  ({_pct(det_pass, total)}%)  — expected labels found')
        self._log_sim(f'  Count:         {count_pass}/{total}  ({_pct(count_pass, total)}%)  — correct object counts')
        self._log_sim(f'  False-pos:     {fp_pass}/{total}  ({_pct(fp_pass, total)}%)  — no forbidden labels')
        if skipped:
            self._log_sim(f'  Skipped:       {len(skipped)} (missing images)')

        # By tag
        all_tags = sorted({tag for r in valid for tag in r.tags})
        if all_tags:
            self._log_sim('')
            self._log_sim('  By category:')
            col = max(len(t) for t in all_tags) + 2
            for tag in all_tags:
                tagged = [r for r in valid if tag in r.tags]
                tp     = sum(1 for r in tagged if r.passed)
                self._log_sim(f'    {tag:<{col}} {tp}/{len(tagged)}  ({_pct(tp, len(tagged))}%)')

        self._log_sim(sep)

    # ── Logging ───────────────────────────────────────────────────────── #

    def _log_sim(self, text: str):
        line = f'[sim] {text}'
        print(line, flush=True)
        self.get_logger().info(line)
        _write_log(line)


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _pct(n: int, d: int) -> int:
    return int(100 * n / d) if d else 0


def _write_log(line: str):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {line}\n')
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────── #

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
