"""
Intent mapper — extracts target objects from voice commands and maps them to
YOLO COCO class labels.

This module has zero ROS2 dependencies so it can be imported and tested
in any Python environment.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# COCO 80-class label map
# ---------------------------------------------------------------------------
# Maps spoken synonyms / common variations → canonical YOLO class name.
# The canonical names match ultralytics YOLOv8 default COCO class labels.
# ---------------------------------------------------------------------------
COCO_CLASSES: dict[str, str] = {
    # People
    "person": "person",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    # Vehicles
    "bicycle": "bicycle",
    "bike": "bicycle",
    "car": "car",
    "automobile": "car",
    "vehicle": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "airplane": "airplane",
    "plane": "airplane",
    "aeroplane": "airplane",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "lorry": "truck",
    "boat": "boat",
    "ship": "boat",
    "traffic light": "traffic light",
    "stoplight": "traffic light",
    "fire hydrant": "fire hydrant",
    "hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    # Animals
    "bird": "bird",
    "cat": "cat",
    "kitten": "cat",
    "dog": "dog",
    "puppy": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    # Accessories
    "backpack": "backpack",
    "rucksack": "backpack",
    "bag": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "purse": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "luggage": "suitcase",
    # Sports
    "frisbee": "frisbee",
    "skis": "skis",
    "ski": "skis",
    "snowboard": "snowboard",
    "sports ball": "sports ball",
    "ball": "sports ball",
    "kite": "kite",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "racket": "tennis racket",
    # Kitchen / Food
    "bottle": "bottle",
    "wine glass": "wine glass",
    "glass": "wine glass",
    "cup": "cup",
    "mug": "cup",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "hotdog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "doughnut": "donut",
    "cake": "cake",
    # Furniture
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "potted plant": "potted plant",
    "plant": "potted plant",
    "bed": "bed",
    "dining table": "dining table",
    "table": "dining table",
    "desk": "dining table",
    "toilet": "toilet",
    # Electronics
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "laptop": "laptop",
    "computer": "laptop",
    "mouse": "mouse",
    "remote": "remote",
    "remote control": "remote",
    "keyboard": "keyboard",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    # Appliances
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    # Misc
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "teddy": "teddy bear",
    "hair drier": "hair drier",
    "hairdryer": "hair drier",
    "toothbrush": "toothbrush",
}

# ---------------------------------------------------------------------------
# Polite/filler prefixes to strip before intent matching
# ---------------------------------------------------------------------------
# Handles natural speech like "please find the person" or "Jarvis find the bottle"
# ---------------------------------------------------------------------------
_PREFIX_RE = re.compile(
    r"^(?:please|jarvis|hey\s+jarvis|ok\s+jarvis|okay\s+jarvis|robot|hey|ok|okay|now|just|can\s+you|could\s+you|would\s+you)\s+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Known Whisper hallucination phrases to reject outright
# ---------------------------------------------------------------------------
# Whisper commonly outputs these on near-silent audio; they produce false intents.
# ---------------------------------------------------------------------------
_HALLUCINATIONS: set[str] = {
    "thank you",
    "thank you.",
    "thanks",
    "you",
    ".",
    "",
    "goodbye",
    "bye",
    "bye bye",
    "yes",
    "no",
    "okay",
    "ok",
    "um",
    "uh",
    "hmm",
    "hm",
    "ah",
    "oh",
    "subscribe",
    "like and subscribe",
    "the",
}

# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------
# Each pattern captures the raw object name in group 1.
# Ordered from most specific to most general.
# ---------------------------------------------------------------------------
_INTENT_PATTERNS: list[str] = [
    r"(?:find|locate|search\s+for|look\s+for|detect|spot|show\s+me)\s+(?:a|an|the|my|some)?\s*(.+)",
    r"where\s+is\s+(?:the|a|an|my)?\s*(.+)",
    r"what\s+is\s+(?:that|the|a)?\s*(.+)",
    r"can\s+you\s+(?:find|see|locate|spot)\s+(?:a|an|the|my)?\s*(.+)",
    r"(?:i'm|i am)\s+looking\s+for\s+(?:a|an|the|my)?\s*(.+)",
    r"start\s+(?:looking|searching)\s+for\s+(?:a|an|the|my)?\s*(.+)",
    r"(?:get|grab|bring)\s+(?:me\s+)?(?:a|an|the|my)?\s*(.+)",
]

_COMPILED_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _INTENT_PATTERNS
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_target(text: str) -> Optional[tuple[str, str]]:
    """Parse a voice command and return (yolo_label, raw_object_name) or None.

    Applies intent pattern matching to extract the target object name, then
    maps it to a canonical YOLO COCO class label.

    Args:
        text: Transcribed speech, e.g. "find the bottle".

    Returns:
        A tuple ``(yolo_label, raw_object_name)`` where ``yolo_label`` is the
        canonical COCO class name (e.g. ``"cell phone"``) and
        ``raw_object_name`` is the spoken name (e.g. ``"phone"``).
        Returns ``None`` if no intent is matched or the object is unknown.

    Examples::

        >>> extract_target("find the bottle")
        ('bottle', 'bottle')
        >>> extract_target("where is the phone")
        ('cell phone', 'phone')
        >>> extract_target("hello robot")
        None
    """
    text = text.strip()
    if not text:
        return None

    # Reject known Whisper hallucination phrases
    if text.lower().rstrip("?.!,") in _HALLUCINATIONS:
        return None

    # Strip polite/filler prefixes ("please find..." → "find...")
    text = _PREFIX_RE.sub("", text).strip()

    # Try the full text first, then fall back to each sentence.
    # Handles Whisper fragments like "person. Find the person." where the
    # first clause is junk but a later sentence contains the real command.
    raw_object = _match_intent(text)
    if raw_object is None:
        for sentence in re.split(r'[.!?]\s+', text):
            sentence = _PREFIX_RE.sub("", sentence.strip()).strip()
            raw_object = _match_intent(sentence)
            if raw_object is not None:
                break

    # Last resort: bare noun ("the person", "phone", "chair").
    # Exact COCO class match only — no prefix/substring matching here,
    # otherwise "stop" → prefix-matches "stop sign" and hijacks stop commands.
    if raw_object is None:
        bare = re.sub(r'^(?:a|an|the|my|some)\s+', '', text.lower().strip().rstrip('?.!,'))
        if bare and bare in COCO_CLASSES:
            raw_object = bare

    if raw_object is None:
        return None

    # Normalise to lowercase for consistent TTS output ("Looking for bottle.")
    raw_object = raw_object.lower()

    yolo_label = map_to_yolo_class(raw_object)
    if yolo_label is None:
        return None

    return (yolo_label, raw_object)


def map_to_yolo_class(raw_object: str) -> Optional[str]:
    """Map a spoken object name to its canonical YOLO COCO class label.

    Matching order:
    1. Exact match (case-insensitive)
    2. Prefix match — spoken word is a prefix of a known key or vice-versa
       (handles plurals like "bottles" → "bottle")
    3. Substring containment — known key is contained in spoken word

    Args:
        raw_object: Spoken object name, e.g. ``"phone"``, ``"bottles"``.

    Returns:
        Canonical COCO label string, or ``None`` if no match is found.
    """
    normalized = raw_object.lower().strip()

    if not normalized:
        return None

    # 1. Exact match
    if normalized in COCO_CLASSES:
        return COCO_CLASSES[normalized]

    # 2. Prefix match — handles plurals ("bottles" → "bottle") and abbreviations
    for key, label in COCO_CLASSES.items():
        if normalized.startswith(key) or key.startswith(normalized):
            return label

    # 3. Substring match — "big cup" → "cup"
    for key, label in COCO_CLASSES.items():
        if key in normalized:
            return label

    return None


# ---------------------------------------------------------------------------
# Enable / Disable helpers (shared with voice_commander_node)
# ---------------------------------------------------------------------------


def is_enable_command(text: str) -> bool:
    """Return True if the text matches a detection-enable voice command.

    Args:
        text: Transcribed speech.

    Returns:
        ``True`` if the text is an enable command.
    """
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "start vision",
            "start detection",
            "enable detection",
            "enable vision",
            "turn on detection",
            "turn on vision",
            "start looking",
            "start scanning",
        ]
    )


def is_what_do_you_see_command(text: str) -> bool:
    """Return True if the user is asking Jarvis to describe the current scene."""
    t = text.lower().strip()
    return any(
        phrase in t
        for phrase in [
            "what do you see",
            "what can you see",
            "describe what you see",
            "what's in front of you",
            "what is in front of you",
            "look around",
            "describe the scene",
            "what do you observe",
        ]
    )


def is_disable_command(text: str) -> bool:
    """Return True if the text matches a detection-disable voice command.

    Args:
        text: Transcribed speech.

    Returns:
        ``True`` if the text is a disable command.
    """
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "stop vision",
            "stop detection",
            "stop finding",
            "stop it",
            "disable detection",
            "disable vision",
            "turn off detection",
            "turn off vision",
            "stop looking",
            "stop scanning",
            "cancel detection",
            "cancel vision",
        ]
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _match_intent(text: str) -> Optional[str]:
    """Apply intent patterns and return the raw object name, or None."""
    for pattern in _COMPILED_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            raw = match.group(1).strip().rstrip("?!.,")
            return raw
    return None
