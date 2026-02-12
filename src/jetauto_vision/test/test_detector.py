"""Unit tests for the detector node."""

import pytest


class TestDetectorNodeHelpers:
    """Test result parsing logic without requiring ROS or YOLO."""

    def test_parse_empty_results(self):
        """Empty YOLO results should return an empty list."""
        assert _parse_results_standalone(None) == []
        assert _parse_results_standalone([]) == []

    def test_parse_results_caps_at_max(self):
        """Detections should be capped at max_detections."""
        mock_boxes = [_make_mock_box(f'obj_{i}', 0.9) for i in range(20)]
        results = _parse_results_standalone(mock_boxes, max_detections=5)
        assert len(results) == 5


class TestSentenceBuilding:
    """Test natural-language sentence generation."""

    def test_single_object(self):
        result = _build_sentence(['cup'], 'I can see')
        assert result == 'I can see a cup'

    def test_two_objects(self):
        result = _build_sentence(['cup', 'laptop'], 'I can see')
        assert result == 'I can see a cup and a laptop'

    def test_three_objects(self):
        result = _build_sentence(['cup', 'laptop', 'phone'], 'I can see')
        assert result == 'I can see a cup, a laptop, and a phone'

    def test_vowel_article(self):
        result = _build_sentence(['apple'], 'I can see')
        assert result == 'I can see an apple'

    def test_empty_labels(self):
        result = _build_sentence([], 'I can see')
        assert result == ''

    def test_custom_greeting(self):
        result = _build_sentence(['cat'], 'I detect')
        assert result == 'I detect a cat'


# -- Helpers (standalone versions of node methods for testing) --

def _build_sentence(labels: list, greeting: str) -> str:
    """Standalone version of TTSNode._build_sentence."""
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


def _make_mock_box(label, conf):
    """Create a minimal mock for a YOLO detection box."""
    class MockTensor:
        def __init__(self, val):
            self._val = val
        def __getitem__(self, idx):
            return self._val
        def tolist(self):
            return [0.0, 0.0, 100.0, 100.0]

    class MockBox:
        def __init__(self, cls_id, confidence):
            self.cls = MockTensor(cls_id)
            self.conf = MockTensor(confidence)
            self.xyxy = MockTensor(None)

    return MockBox(0, conf)


def _parse_results_standalone(results_or_boxes, max_detections=10):
    """Simplified parse logic matching DetectorNode._parse_results."""
    if not results_or_boxes:
        return []
    detections = []
    for i, box in enumerate(results_or_boxes):
        if i >= max_detections:
            break
        detections.append({
            'label': f'object_{i}',
            'confidence': float(box.conf[0]),
        })
    return detections
