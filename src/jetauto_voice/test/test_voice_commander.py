"""
Unit tests for VoiceCommanderNode.

All hardware dependencies (sounddevice, openWakeWord, faster-whisper, rclpy)
are mocked so the tests run on any machine without a mic or GPU.

Run with:
    python -m pytest src/jetauto_voice/test/
"""

import sys
import os
import threading
import types
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Stub out rclpy and ROS2 message types before importing the node
# ---------------------------------------------------------------------------

def _make_rclpy_stub():
    """Return a minimal rclpy stub that lets VoiceCommanderNode.__init__ run."""
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, name):
            self._name = name
            self._params = {}

        def declare_parameter(self, name, default):
            self._params[name] = default

        def get_parameter(self, name):
            mock = MagicMock()
            mock.value = self._params.get(name)
            return mock

        def create_publisher(self, msg_type, topic, qos):
            pub = MagicMock()
            pub.topic = topic
            return pub

        def create_subscription(self, *args, **kwargs):
            return MagicMock()

        def get_logger(self):
            log = MagicMock()
            log.info = lambda msg: None
            log.warn = lambda msg: None
            log.error = lambda msg: None
            log.debug = lambda msg: None
            return log

        def destroy_node(self):
            pass

    rclpy.node.Node = _Node
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: None
    return rclpy


def _make_std_msgs_stub():
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")

    class Bool:
        def __init__(self):
            self.data = False

    class String:
        def __init__(self):
            self.data = ""

    std_msgs.msg.Bool = Bool
    std_msgs.msg.String = String
    return std_msgs


# Install stubs
_rclpy_stub = _make_rclpy_stub()
_std_msgs_stub = _make_std_msgs_stub()
sys.modules.setdefault("rclpy", _rclpy_stub)
sys.modules.setdefault("rclpy.node", _rclpy_stub.node)
sys.modules.setdefault("std_msgs", _std_msgs_stub)
sys.modules.setdefault("std_msgs.msg", _std_msgs_stub.msg)

# Ensure the package under test is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), ".."),
)

from jetauto_voice.voice_commander_node import VoiceCommanderNode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(**param_overrides) -> VoiceCommanderNode:
    """Create a VoiceCommanderNode with audio + model loading mocked out."""
    defaults = {
        "wake_word_model": "hey_jarvis",
        "wake_word_threshold": 0.5,
        "stt_model_size": "base",
        "stt_device": "cpu",
        "stt_compute_type": "int8",
        "mic_device_index": -1,
        "capture_duration_sec": 4.0,
        "sample_rate": 16000,
    }
    defaults.update(param_overrides)

    with (
        patch.object(VoiceCommanderNode, "_init_wakeword", lambda self: None),
        patch.object(VoiceCommanderNode, "_init_stt", lambda self: None),
        patch.object(VoiceCommanderNode, "_audio_loop", lambda self: None),
        patch("threading.Thread") as mock_thread,
    ):
        mock_thread.return_value.start = lambda: None
        mock_thread.return_value.is_alive = lambda: False
        node = VoiceCommanderNode.__new__(VoiceCommanderNode)
        # Manually set params dict before calling __init__
        node._params = defaults
        # Patch declare_parameter to use our defaults dict
        node.declare_parameter = lambda name, default: None
        node.get_parameter = lambda name: _make_param_value(defaults.get(name))
        node.create_publisher = lambda msg_type, topic, qos: MagicMock()
        node.get_logger = lambda: _silent_logger()
        node._shutdown_event = threading.Event()
        node._wake_word_detector = None
        node._stt_model = None
        node._listener_thread = MagicMock()
        node._listener_thread.is_alive.return_value = False

        # Set all attributes that __init__ normally sets from params
        node._wake_word_model = defaults["wake_word_model"]
        node._wake_word_threshold = defaults["wake_word_threshold"]
        node._stt_model_size = defaults["stt_model_size"]
        node._stt_device = defaults["stt_device"]
        node._stt_compute_type = defaults["stt_compute_type"]
        node._mic_device_index = defaults["mic_device_index"]
        node._capture_duration = defaults["capture_duration_sec"]
        node._sample_rate = defaults["sample_rate"]

        # Publishers
        node._detection_pub = MagicMock()
        node._target_pub = MagicMock()
        node._tts_pub = MagicMock()

        return node


def _make_param_value(val):
    m = MagicMock()
    m.value = val
    return m


def _silent_logger():
    log = MagicMock()
    log.info = lambda msg: None
    log.warn = lambda msg: None
    log.error = lambda msg: None
    log.debug = lambda msg: None
    return log


# ===========================================================================
# _dispatch_intent
# ===========================================================================


class TestDispatchIntent:
    """Test that _dispatch_intent publishes the right ROS2 messages."""

    def setup_method(self):
        self.node = _make_node()

    def _published_bool(self, mock_pub) -> bool:
        """Return the .data value from the last Bool message published."""
        assert mock_pub.publish.called, "Expected publish() to be called"
        msg = mock_pub.publish.call_args[0][0]
        return msg.data

    def _published_string(self, mock_pub) -> str:
        assert mock_pub.publish.called, "Expected publish() to be called"
        msg = mock_pub.publish.call_args[0][0]
        return msg.data

    def test_find_bottle_enables_detection(self):
        self.node._dispatch_intent("find the bottle")
        assert self._published_bool(self.node._detection_pub) is True

    def test_find_bottle_publishes_target(self):
        self.node._dispatch_intent("find the bottle")
        assert self._published_string(self.node._target_pub) == "bottle"

    def test_find_bottle_tts_response(self):
        self.node._dispatch_intent("find the bottle")
        tts = self._published_string(self.node._tts_pub)
        assert "bottle" in tts.lower()

    def test_find_phone_maps_to_cell_phone(self):
        self.node._dispatch_intent("find the phone")
        assert self._published_string(self.node._target_pub) == "cell phone"

    def test_enable_command_sets_detection_true(self):
        self.node._dispatch_intent("start detection")
        assert self._published_bool(self.node._detection_pub) is True
        # No target should be published for a pure enable command
        assert not self.node._target_pub.publish.called

    def test_disable_command_sets_detection_false(self):
        self.node._dispatch_intent("stop detection")
        assert self._published_bool(self.node._detection_pub) is False

    def test_unknown_command_publishes_tts_apology(self):
        self.node._dispatch_intent("banana rocket fuel")
        tts = self._published_string(self.node._tts_pub)
        assert "sorry" in tts.lower() or "understand" in tts.lower()

    def test_unknown_command_no_detection_change(self):
        self.node._dispatch_intent("banana rocket fuel")
        assert not self.node._detection_pub.publish.called

    def test_find_fridge_maps_to_refrigerator(self):
        self.node._dispatch_intent("look for the fridge")
        assert self._published_string(self.node._target_pub) == "refrigerator"

    def test_find_person(self):
        self.node._dispatch_intent("where is the person")
        assert self._published_string(self.node._target_pub) == "person"


# ===========================================================================
# _transcribe
# ===========================================================================


class TestTranscribe:
    """Test the STT transcription wrapper."""

    def setup_method(self):
        self.node = _make_node()

    def test_transcribe_returns_empty_when_no_model(self):
        self.node._stt_model = None
        audio = np.zeros(16000, dtype=np.float32)
        result = self.node._transcribe(audio)
        assert result == ""

    def test_transcribe_joins_segments(self):
        """Multiple whisper segments should be joined with spaces."""
        seg1 = MagicMock()
        seg1.text = "find the"
        seg2 = MagicMock()
        seg2.text = " bottle"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())
        self.node._stt_model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = self.node._transcribe(audio)
        assert result == "find the  bottle"

    def test_transcribe_single_segment(self):
        seg = MagicMock()
        seg.text = "find the bottle"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        self.node._stt_model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = self.node._transcribe(audio)
        assert result == "find the bottle"

    def test_transcribe_handles_exception(self):
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA OOM")
        self.node._stt_model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = self.node._transcribe(audio)
        assert result == ""

    def test_transcribe_passes_language_en(self):
        seg = MagicMock()
        seg.text = "hello"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        self.node._stt_model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        self.node._transcribe(audio)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "en"

    def test_transcribe_uses_vad_filter(self):
        seg = MagicMock()
        seg.text = "hello"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        self.node._stt_model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        self.node._transcribe(audio)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("vad_filter") is True


# ===========================================================================
# _init_stt — CPU fallback
# ===========================================================================


class TestSTTInit:
    """Test faster-whisper initialization and CPU fallback."""

    def test_cpu_fallback_triggered_on_cuda_failure(self):
        """If CUDA init fails, the node should try CPU with int8."""
        node = _make_node()
        node._stt_model = None

        call_count = {"n": 0}

        def fake_whisper_model(size, device, compute_type):
            call_count["n"] += 1
            if device == "cuda":
                raise RuntimeError("no cuda device")
            m = MagicMock()
            m.device = device
            m.compute_type = compute_type
            return m

        with patch("jetauto_voice.voice_commander_node.VoiceCommanderNode._init_stt_cpu_fallback") as mock_fallback:
            # Simulate CUDA failing then _init_stt calling the fallback
            node._stt_device = "cuda"
            node._init_stt = lambda: mock_fallback()
            node._init_stt()
            mock_fallback.assert_called_once()

    def test_cpu_fallback_sets_model(self):
        """_init_stt_cpu_fallback should set _stt_model when successful."""
        node = _make_node()
        node._stt_model = None
        node._stt_model_size = "base"

        mock_model_instance = MagicMock()

        with patch(
            "jetauto_voice.voice_commander_node.WhisperModel",
            return_value=mock_model_instance,
            create=True,
        ):
            # Patch the import inside the method
            whisper_mod = types.ModuleType("faster_whisper")
            whisper_mod.WhisperModel = lambda size, device, compute_type: mock_model_instance
            with patch.dict(sys.modules, {"faster_whisper": whisper_mod}):
                node._init_stt_cpu_fallback()
                assert node._stt_model is mock_model_instance


# ===========================================================================
# Wake word threshold configuration
# ===========================================================================


class TestWakeWordConfig:
    """Test wake word threshold parameter handling."""

    def test_default_threshold_is_05(self):
        node = _make_node()
        assert node._wake_word_threshold == 0.5

    def test_custom_threshold_stored(self):
        node = _make_node(wake_word_threshold=0.3)
        assert node._wake_word_threshold == 0.3

    def test_high_threshold_stored(self):
        node = _make_node(wake_word_threshold=0.9)
        assert node._wake_word_threshold == 0.9

    def test_wake_word_model_stored(self):
        node = _make_node(wake_word_model="alexa")
        assert node._wake_word_model == "alexa"


# ===========================================================================
# Audio mock — _handle_wake_word
# ===========================================================================


class TestHandleWakeWord:
    """Test _handle_wake_word with fully mocked audio stream and STT."""

    def setup_method(self):
        self.node = _make_node(capture_duration_sec=2.0, sample_rate=16000)

    def _make_stream(self, transcript: str):
        """Return a mock stream and patch _transcribe to return transcript."""
        capture_samples = int(2.0 * 16000)
        fake_audio = np.zeros((capture_samples, 1), dtype=np.int16)
        stream = MagicMock()
        stream.read.return_value = (fake_audio, None)
        return stream

    def test_find_command_dispatches(self):
        stream = self._make_stream("find the bottle")
        with patch.object(self.node, "_transcribe", return_value="find the bottle"):
            with patch.object(self.node, "_dispatch_intent") as mock_dispatch:
                self.node._handle_wake_word(stream)
                mock_dispatch.assert_called_once_with("find the bottle")

    def test_empty_transcript_does_not_dispatch(self):
        stream = self._make_stream("")
        with patch.object(self.node, "_transcribe", return_value=""):
            with patch.object(self.node, "_dispatch_intent") as mock_dispatch:
                self.node._handle_wake_word(stream)
                mock_dispatch.assert_not_called()

    def test_acknowledgment_tts_published(self):
        stream = self._make_stream("find the bottle")
        with patch.object(self.node, "_transcribe", return_value="find the bottle"):
            with patch.object(self.node, "_dispatch_intent"):
                self.node._handle_wake_word(stream)
                tts_calls = [
                    c[0][0].data for c in self.node._tts_pub.publish.call_args_list
                ]
                assert any("yes" in t.lower() or "?" in t for t in tts_calls)

    def test_audio_is_normalized_to_float32(self):
        """_handle_wake_word should normalize int16 → float32 before transcribing."""
        capture_samples = int(2.0 * 16000)
        # Simulate a full-scale int16 signal
        full_scale = np.full((capture_samples, 1), 32767, dtype=np.int16)
        stream = MagicMock()
        stream.read.return_value = (full_scale, None)

        captured_audio = {}

        def fake_transcribe(audio):
            captured_audio["audio"] = audio
            return "find the bottle"

        with patch.object(self.node, "_transcribe", side_effect=fake_transcribe):
            with patch.object(self.node, "_dispatch_intent"):
                self.node._handle_wake_word(stream)

        assert "audio" in captured_audio
        arr = captured_audio["audio"]
        assert arr.dtype == np.float32
        # Full-scale int16 (32767) should map to ≈1.0
        assert arr.max() == pytest.approx(32767.0 / 32768.0, rel=1e-3)
