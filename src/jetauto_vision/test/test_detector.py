"""Unit tests for the detector node.

Tests the parsing and sentence-building logic without requiring ROS2 or
YOLO to be installed.  A lightweight DetectedObject stand-in mirrors the
real message so the helper functions exercise the same field names,
rounding, and bbox extraction as the shipped code.
"""

from dataclasses import dataclass, field

import pytest


# -- Stand-in for jetauto_msgs.msg.DetectedObject (no ROS2 needed) --

@dataclass
class DetectedObject:
    label: str = ''
    confidence: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


# ------------------------------------------------------------------ #
# Parse-results tests
# ------------------------------------------------------------------ #

class TestParseResults:
    """Test _parse_results logic (standalone mirror of DetectorNode)."""

    def test_none_input(self):
        assert _parse_results(None) == []

    def test_empty_list(self):
        assert _parse_results([]) == []

    def test_no_boxes(self):
        """Result object with boxes=None."""
        result = _make_mock_result(names={}, boxes=None)
        assert _parse_results([result]) == []

    def test_single_detection(self):
        boxes = [_make_mock_box(cls_id=0, conf=0.87, xyxy=[10.123, 20.456, 300.789, 400.012])]
        result = _make_mock_result(names={0: 'cup'}, boxes=boxes)
        dets = _parse_results([result])
        assert len(dets) == 1
        assert dets[0].label == 'cup'
        assert dets[0].confidence == 0.87
        assert dets[0].x1 == 10.1
        assert dets[0].y1 == 20.5
        assert dets[0].x2 == 300.8
        assert dets[0].y2 == 400.0

    def test_unknown_class_id(self):
        boxes = [_make_mock_box(cls_id=99, conf=0.5, xyxy=[0, 0, 1, 1])]
        result = _make_mock_result(names={0: 'cup'}, boxes=boxes)
        dets = _parse_results([result])
        assert dets[0].label == 'class_99'

    def test_caps_at_max(self):
        boxes = [_make_mock_box(cls_id=0, conf=0.9, xyxy=[0, 0, 1, 1]) for _ in range(20)]
        result = _make_mock_result(names={0: 'obj'}, boxes=boxes)
        dets = _parse_results([result], max_detections=5)
        assert len(dets) == 5

    def test_multiple_classes(self):
        boxes = [
            _make_mock_box(cls_id=0, conf=0.95, xyxy=[0, 0, 50, 50]),
            _make_mock_box(cls_id=1, conf=0.72, xyxy=[60, 60, 120, 120]),
        ]
        result = _make_mock_result(names={0: 'person', 1: 'laptop'}, boxes=boxes)
        dets = _parse_results([result])
        assert len(dets) == 2
        assert dets[0].label == 'person'
        assert dets[1].label == 'laptop'
        assert dets[1].confidence == 0.72


# ------------------------------------------------------------------ #
# Sentence-building tests
# ------------------------------------------------------------------ #

class TestSentenceBuilding:
    """Test natural-language sentence generation."""

    def test_single_object(self):
        assert _build_sentence(['cup'], 'I can see') == 'I can see a cup'

    def test_two_objects(self):
        assert _build_sentence(['cup', 'laptop'], 'I can see') == 'I can see a cup and a laptop'

    def test_three_objects(self):
        assert _build_sentence(['cup', 'laptop', 'phone'], 'I can see') == (
            'I can see a cup, a laptop, and a phone'
        )

    def test_vowel_article(self):
        assert _build_sentence(['apple'], 'I can see') == 'I can see an apple'

    def test_empty_labels(self):
        assert _build_sentence([], 'I can see') == ''

    def test_custom_greeting(self):
        assert _build_sentence(['cat'], 'I detect') == 'I detect a cat'


# ------------------------------------------------------------------ #
# Mock helpers
# ------------------------------------------------------------------ #

class _MockScalar:
    """Mimics a single-element tensor: tensor[0] returns the scalar."""
    def __init__(self, val):
        self._val = val
    def __getitem__(self, idx):
        return self._val
    def __int__(self):
        return int(self._val)
    def __float__(self):
        return float(self._val)


class _MockCoords:
    """Mimics xyxy tensor: tensor[0] returns a sub-tensor with .tolist()."""
    def __init__(self, coords):
        self._coords = coords  # [x1, y1, x2, y2]
    def __getitem__(self, idx):
        return self  # xyxy[0] returns self (the row)
    def tolist(self):
        return list(self._coords)


class _MockBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _MockScalar(cls_id)
        self.conf = _MockScalar(conf)
        self.xyxy = _MockCoords(xyxy)


class _MockResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def _make_mock_box(cls_id, conf, xyxy):
    return _MockBox(cls_id, conf, xyxy)


def _make_mock_result(names, boxes):
    return _MockResult(names, boxes)


# ------------------------------------------------------------------ #
# Standalone mirrors of node methods (must stay in sync with source)
# ------------------------------------------------------------------ #

def _parse_results(results, max_detections=10):
    """Mirror of DetectorNode._parse_results — returns DetectedObject list."""
    detections = []
    if not results or len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    for i, box in enumerate(boxes):
        if i >= max_detections:
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


def _build_sentence(labels: list, greeting: str) -> str:
    """Mirror of TTSNode._build_sentence."""
    if not labels:
        return ''
    items = []
    for label in labels:
        if not label:
            items.append('an unknown object')
        else:
            article = 'an' if label[0].lower() in 'aeiou' else 'a'
            items.append(f'{article} {label}')

    if len(items) == 1:
        obj_str = items[0]
    elif len(items) == 2:
        obj_str = f'{items[0]} and {items[1]}'
    else:
        obj_str = ', '.join(items[:-1]) + f', and {items[-1]}'

    return f'{greeting} {obj_str}'
