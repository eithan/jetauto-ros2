"""
Unit tests for jetauto_voice.intent_mapper.

All tests run without ROS2, sounddevice, openWakeWord, or faster-whisper.
Run with:
    python -m pytest src/jetauto_voice/test/
"""

import sys
import os

# Allow import from src tree without a ROS2 install.
# __file__ is src/jetauto_voice/test/test_intent_mapper.py
# inserting ".." gives src/jetauto_voice/ which contains the jetauto_voice/ package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from jetauto_voice.intent_mapper import (
    extract_target,
    map_to_yolo_class,
    is_enable_command,
    is_disable_command,
    COCO_CLASSES,
    _match_intent,
)


# ===========================================================================
# map_to_yolo_class
# ===========================================================================


class TestMapToYoloClass:
    """Test YOLO class label mapping."""

    def test_exact_match(self):
        assert map_to_yolo_class("bottle") == "bottle"

    def test_synonym_phone(self):
        assert map_to_yolo_class("phone") == "cell phone"

    def test_synonym_fridge(self):
        assert map_to_yolo_class("fridge") == "refrigerator"

    def test_synonym_tv(self):
        assert map_to_yolo_class("television") == "tv"

    def test_synonym_sofa(self):
        assert map_to_yolo_class("sofa") == "couch"

    def test_synonym_laptop(self):
        assert map_to_yolo_class("laptop") == "laptop"

    def test_synonym_bag(self):
        assert map_to_yolo_class("bag") == "backpack"

    def test_synonym_teddy(self):
        assert map_to_yolo_class("teddy") == "teddy bear"

    def test_synonym_bike(self):
        assert map_to_yolo_class("bike") == "bicycle"

    def test_prefix_plural_bottle(self):
        """Plural 'bottles' should match 'bottle' via prefix."""
        assert map_to_yolo_class("bottles") == "bottle"

    def test_prefix_plural_cups(self):
        assert map_to_yolo_class("cups") == "cup"

    def test_case_insensitive(self):
        assert map_to_yolo_class("BOTTLE") == "bottle"
        assert map_to_yolo_class("Phone") == "cell phone"

    def test_unknown_object(self):
        assert map_to_yolo_class("lightsaber") is None

    def test_empty_string(self):
        assert map_to_yolo_class("") is None

    def test_all_coco_keys_round_trip(self):
        """Every key in COCO_CLASSES must map to its own value."""
        for key, expected_label in COCO_CLASSES.items():
            result = map_to_yolo_class(key)
            assert result == expected_label, (
                f"map_to_yolo_class({key!r}) = {result!r}, expected {expected_label!r}"
            )


# ===========================================================================
# extract_target
# ===========================================================================


class TestExtractTarget:
    """Test full intent extraction pipeline."""

    def test_find_bottle(self):
        assert extract_target("find the bottle") == ("bottle", "bottle")

    def test_look_for_person(self):
        assert extract_target("look for a person") == ("person", "person")

    def test_where_is_cup(self):
        assert extract_target("where is the cup") == ("cup", "cup")

    def test_locate_phone(self):
        result = extract_target("locate the phone")
        assert result == ("cell phone", "phone")

    def test_search_for_laptop(self):
        assert extract_target("search for a laptop") == ("laptop", "laptop")

    def test_can_you_find_cat(self):
        assert extract_target("can you find a cat") == ("cat", "cat")

    def test_show_me_dog(self):
        assert extract_target("show me a dog") == ("dog", "dog")

    def test_i_am_looking_for_fridge(self):
        result = extract_target("I am looking for the fridge")
        assert result == ("refrigerator", "fridge")

    def test_detect_pizza(self):
        assert extract_target("detect pizza") == ("pizza", "pizza")

    def test_find_sofa(self):
        assert extract_target("find the sofa") == ("couch", "sofa")

    def test_find_bike(self):
        assert extract_target("find my bike") == ("bicycle", "bike")

    def test_find_teddy(self):
        assert extract_target("find the teddy") == ("teddy bear", "teddy")

    def test_no_intent_hello(self):
        assert extract_target("hello robot") is None

    def test_no_intent_empty(self):
        assert extract_target("") is None

    def test_no_intent_unknown_object(self):
        assert extract_target("find the lightsaber") is None

    def test_strips_trailing_punctuation(self):
        """Trailing '?' or '.' should not break matching."""
        assert extract_target("where is the cup?") == ("cup", "cup")
        assert extract_target("find the bottle.") == ("bottle", "bottle")

    def test_case_insensitive(self):
        assert extract_target("FIND THE BOTTLE") == ("bottle", "bottle")
        assert extract_target("Where Is The Cup") == ("cup", "cup")

    def test_start_looking_for(self):
        assert extract_target("start looking for a person") == ("person", "person")

    def test_grab_me(self):
        result = extract_target("grab me the bottle")
        assert result == ("bottle", "bottle")


# ===========================================================================
# _match_intent (internal, but important to test directly)
# ===========================================================================


class TestMatchIntent:
    """Test the raw intent-pattern matching helper."""

    def test_find_extracts_object(self):
        assert _match_intent("find the bottle") == "bottle"

    def test_where_is_extracts_object(self):
        assert _match_intent("where is the cat") == "cat"

    def test_no_match_returns_none(self):
        assert _match_intent("hello") is None

    def test_strips_trailing_question_mark(self):
        assert _match_intent("where is the cup?") == "cup"

    def test_strips_trailing_period(self):
        assert _match_intent("find the bottle.") == "bottle"


# ===========================================================================
# is_enable_command / is_disable_command
# ===========================================================================


class TestEnableDisableCommands:
    """Test detection enable/disable command detection."""

    def test_start_detection(self):
        assert is_enable_command("start detection") is True

    def test_enable_detection(self):
        assert is_enable_command("enable detection") is True

    def test_enable_vision(self):
        assert is_enable_command("enable vision") is True

    def test_turn_on_detection(self):
        assert is_enable_command("turn on detection") is True

    def test_start_looking(self):
        assert is_enable_command("start looking") is True

    def test_what_do_you_see(self):
        assert is_enable_command("what do you see") is True

    def test_stop_detection(self):
        assert is_disable_command("stop detection") is True

    def test_disable_detection(self):
        assert is_disable_command("disable detection") is True

    def test_disable_vision(self):
        assert is_disable_command("disable vision") is True

    def test_turn_off_detection(self):
        assert is_disable_command("turn off detection") is True

    def test_stop_looking(self):
        assert is_disable_command("stop looking") is True

    def test_enable_not_disable(self):
        assert is_disable_command("enable detection") is False

    def test_disable_not_enable(self):
        assert is_enable_command("stop detection") is False

    def test_random_phrase_not_enable(self):
        assert is_enable_command("hello robot") is False

    def test_random_phrase_not_disable(self):
        assert is_disable_command("find the bottle") is False

    def test_case_insensitive_enable(self):
        assert is_enable_command("START DETECTION") is True

    def test_case_insensitive_disable(self):
        assert is_disable_command("STOP DETECTION") is True
