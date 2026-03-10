#!/usr/bin/env python3
"""
Voice Commander Node — fully offline, open-source voice pipeline.

Replaces the iFlyTek/asr_node dependency with:
  1. openWakeWord  — always-on wake word detection (low CPU, background thread)
  2. faster-whisper — CUDA-accelerated STT (falls back to CPU on Orin Nano)
  3. intent_mapper  — regex-based intent parsing + YOLO class mapping

Flow::

    mic audio  →  openWakeWord (continuous, 80ms chunks)
                      │ wake word detected
                      ▼
               VAD endpoint detection
               (capture until silence)
                      │
                      ▼
               faster-whisper → transcript text
                      │
                      ▼
               intent_mapper.extract_target()
                      │
              ┌───────┴────────────────────────┐
              │ find X intent                  │ enable/disable
              ▼                                ▼
    publish /jetauto/detection/enable (True)  publish /jetauto/detection/enable
    publish /jetauto/detection/target (label)
    publish /tts/speak ("Looking for X...")

ROS2 Parameters
---------------
wake_word_model : str
    openWakeWord model name. Built-in options include ``'hey_jarvis'``,
    ``'alexa'``, ``'hey_mycroft'``. Default: ``'hey_jarvis'``.
wake_word_threshold : float
    Detection confidence threshold in [0.0, 1.0]. Lower values are more
    sensitive (more false positives). Default: ``0.5``.
stt_model_size : str
    faster-whisper model size: ``'tiny'``, ``'base'``, ``'small'``,
    ``'medium'``, ``'large-v2'``. Larger = more accurate, slower.
    Default: ``'base'``.
stt_device : str
    Inference device: ``'cuda'`` or ``'cpu'``. Falls back to CPU
    automatically if CUDA is unavailable. Default: ``'cuda'``.
stt_compute_type : str
    faster-whisper compute type: ``'float16'``, ``'int8'``, ``'float32'``.
    ``'float16'`` requires CUDA; CPU falls back to ``'int8'``.
    Default: ``'float16'``.
mic_device_index : int
    ALSA device index for sounddevice. ``-1`` uses the system default.
    Default: ``-1``.
vad_energy_threshold : int
    RMS energy level (0–32768) above which a chunk is considered speech.
    Tune up if noisy environment causes false triggers; tune down if mic
    is very quiet. Default: ``300``.
vad_silence_ms : int
    Milliseconds of silence after speech that triggers end-of-utterance.
    Default: ``700`` (700ms).
vad_max_duration_sec : float
    Maximum seconds to wait for an utterance before giving up.
    Default: ``8.0``.
wake_cooldown_sec : float
    Seconds to ignore wake word detections after handling one, preventing
    immediate re-triggering on residual OWW scores. Default: ``2.0``.
sample_rate : int
    Microphone sample rate in Hz. Must match openWakeWord's expected rate
    of 16000 Hz. Default: ``16000``.

Topics Published
----------------
/jetauto/detection/enable : std_msgs/Bool
    ``True`` to enable YOLO detection, ``False`` to disable.
/jetauto/detection/target : std_msgs/String
    Canonical YOLO COCO class label to search for (e.g. ``'bottle'``).
/tts/speak : std_msgs/String
    Human-readable TTS response text.
"""

import os
import queue
import threading
from typing import Optional

# Suppress HuggingFace Hub unauthenticated request warnings
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from jetauto_voice.intent_mapper import (
    extract_target,
    is_enable_command,
    is_disable_command,
)

# openWakeWord expects 80 ms frames at 16 kHz = 1280 samples
_OWW_CHUNK_SAMPLES = 1280
_OWW_SAMPLE_RATE = 16000


class VoiceCommanderNode(Node):
    """Offline voice commander using openWakeWord and faster-whisper.

    This node runs a background audio listener thread that continuously
    feeds audio chunks to openWakeWord. When the wake word is detected,
    it captures a full utterance, runs faster-whisper STT, parses the
    transcript for intent, and publishes to the appropriate ROS2 topics.
    """

    def __init__(self) -> None:
        super().__init__("voice_commander_node")

        # -- Declare parameters --
        self.declare_parameter("wake_word_model", "hey_jarvis")
        self.declare_parameter("wake_word_threshold", 0.5)
        self.declare_parameter("stt_model_size", "base.en")
        self.declare_parameter("stt_device", "cuda")
        self.declare_parameter("stt_compute_type", "float16")
        self.declare_parameter("mic_device_index", -1)
        self.declare_parameter("vad_aggressiveness", 2)
        self.declare_parameter("vad_drain_ms", 350)
        self.declare_parameter("vad_speech_start_frames", 6)
        self.declare_parameter("vad_speech_end_frames", 20)
        self.declare_parameter("vad_min_speech_ms", 300)
        self.declare_parameter("vad_listen_timeout_sec", 4.0)
        self.declare_parameter("vad_max_duration_sec", 8.0)
        self.declare_parameter("wake_cooldown_sec", 3.0)
        self.declare_parameter("sample_rate", 16000)

        # -- Read parameters --
        self._wake_word_model: str = self.get_parameter("wake_word_model").value
        self._wake_word_threshold: float = self.get_parameter("wake_word_threshold").value
        self._stt_model_size: str = self.get_parameter("stt_model_size").value
        self._stt_device: str = self.get_parameter("stt_device").value
        self._stt_compute_type: str = self.get_parameter("stt_compute_type").value
        self._mic_device_index: int = self.get_parameter("mic_device_index").value
        self._vad_aggressiveness: int = self.get_parameter("vad_aggressiveness").value
        self._vad_drain_ms: int = self.get_parameter("vad_drain_ms").value
        self._vad_speech_start_frames: int = self.get_parameter("vad_speech_start_frames").value
        self._vad_speech_end_frames: int = self.get_parameter("vad_speech_end_frames").value
        self._vad_min_speech_ms: int = self.get_parameter("vad_min_speech_ms").value
        self._vad_listen_timeout_sec: float = self.get_parameter("vad_listen_timeout_sec").value
        self._vad_max_duration_sec: float = self.get_parameter("vad_max_duration_sec").value
        self._wake_cooldown_sec: float = self.get_parameter("wake_cooldown_sec").value
        self._sample_rate: int = self.get_parameter("sample_rate").value

        # -- Publishers --
        self._detection_pub = self.create_publisher(
            Bool, "/jetauto/detection/enable", 1
        )
        self._target_pub = self.create_publisher(
            String, "/jetauto/detection/target", 1
        )
        self._tts_pub = self.create_publisher(String, "/tts/speak", 1)

        # -- Internal state --
        self._shutdown_event = threading.Event()
        self._wake_word_detector = None
        self._stt_model = None
        self._stt_actual_device = self._stt_device  # may change after fallback

        # -- Load models --
        self._init_wakeword()
        self._init_stt()

        # -- Start background audio thread --
        self._listener_thread = threading.Thread(
            target=self._audio_loop,
            daemon=True,
            name="voice_listener",
        )
        self._listener_thread.start()

        self.get_logger().info(
            f"VoiceCommanderNode ready — wake_word='{self._wake_word_model}' "
            f"(threshold={self._wake_word_threshold}), "
            f"STT={self._stt_model_size}@{self._stt_actual_device}"
        )

    # ------------------------------------------------------------------ #
    # Model initialization
    # ------------------------------------------------------------------ #

    def _init_wakeword(self) -> None:
        """Load the openWakeWord model, downloading it first if needed."""
        try:
            from openwakeword.model import Model  # type: ignore[import]
            from openwakeword.utils import download_models  # type: ignore[import]
        except ImportError:
            self.get_logger().error(
                "openwakeword not installed! Run: pip3 install openwakeword"
            )
            return

        # Models are NOT bundled with the pip package — download on first run.
        self.get_logger().info(
            f"Downloading openWakeWord model '{self._wake_word_model}' if needed..."
        )
        try:
            download_models([self._wake_word_model])
        except Exception as exc:
            self.get_logger().warn(
                f"Model download for '{self._wake_word_model}' failed ({exc}). "
                "Will attempt to load from cache anyway."
            )

        try:
            self._wake_word_detector = Model(
                wakeword_models=[self._wake_word_model],
                inference_framework="onnx",
            )
            self.get_logger().info(
                f"openWakeWord loaded: '{self._wake_word_model}'"
            )
        except Exception as exc:
            self.get_logger().error(f"openWakeWord init failed: {exc}")

    def _init_stt(self) -> None:
        """Load the faster-whisper STT model, with automatic CPU fallback."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            self._stt_model = WhisperModel(
                self._stt_model_size,
                device=self._stt_device,
                compute_type=self._stt_compute_type,
            )
            self.get_logger().info(
                f"faster-whisper loaded: size={self._stt_model_size}, "
                f"device={self._stt_device}, compute={self._stt_compute_type}"
            )
        except ImportError:
            self.get_logger().error(
                "faster-whisper not installed! Run: pip3 install faster-whisper"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"faster-whisper on {self._stt_device} failed ({exc}) — "
                "trying CPU/int8 fallback"
            )
            self._init_stt_cpu_fallback()

    def _init_stt_cpu_fallback(self) -> None:
        """Fallback: load faster-whisper on CPU with int8 quantization."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            self._stt_model = WhisperModel(
                self._stt_model_size,
                device="cpu",
                compute_type="int8",
            )
            self._stt_actual_device = "cpu"
            self.get_logger().info(
                f"faster-whisper CPU fallback: size={self._stt_model_size}, int8"
            )
        except Exception as exc:
            self.get_logger().error(f"faster-whisper CPU fallback failed: {exc}")

    # ------------------------------------------------------------------ #
    # Background audio loop
    # ------------------------------------------------------------------ #

    def _audio_loop(self) -> None:
        """Continuously read mic audio and run wake word detection.

        Runs on a background daemon thread. When openWakeWord detects the
        wake word above the configured threshold, calls ``_handle_wake_word``
        to capture and transcribe a full utterance.

        A cooldown period after each detection prevents OWW's residual
        buffered scores from immediately re-triggering.

        Exits cleanly when ``_shutdown_event`` is set.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            self.get_logger().error(
                "sounddevice not installed! Run: pip3 install sounddevice"
            )
            return

        if self._wake_word_detector is None:
            self.get_logger().error(
                "Wake word detector not loaded — audio loop aborted"
            )
            return

        device: Optional[int] = (
            self._mic_device_index if self._mic_device_index >= 0 else None
        )

        # How many OWW chunks to skip after a detection (prevents re-triggering)
        cooldown_total = int(
            self._wake_cooldown_sec * self._sample_rate / _OWW_CHUNK_SAMPLES
        )

        self.get_logger().info(
            f"Audio loop started — rate={self._sample_rate}Hz, "
            f"chunk={_OWW_CHUNK_SAMPLES} samples, device={device}"
        )

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=_OWW_CHUNK_SAMPLES,
            device=device,
        ) as stream:
            cooldown_remaining = 0
            while not self._shutdown_event.is_set():
                audio_chunk, _ = stream.read(_OWW_CHUNK_SAMPLES)
                audio_flat = audio_chunk.flatten()

                # Skip OWW evaluation during cooldown; still drain the stream
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue

                # openWakeWord expects int16 samples
                prediction: dict = self._wake_word_detector.predict(audio_flat)

                # Pick the maximum score across all loaded models
                score: float = max(prediction.values()) if prediction else 0.0

                if score >= self._wake_word_threshold:
                    self.get_logger().info(
                        f"Wake word detected (score={score:.3f})"
                    )
                    self._handle_wake_word(stream)
                    # Apply cooldown after returning from capture to suppress
                    # any residual high scores still in OWW's sliding window
                    cooldown_remaining = cooldown_total

    def _handle_wake_word(self, stream) -> None:
        """Greeting + conversation loop activated by wake word.

        First activation says "Yes, how may I help you?" then enters a listen
        loop. Each subsequent listen within the same session uses a beep.
        The loop exits (returning to wake word listening) when:
        - No speech is detected (timeout)
        - STT produces an empty or hallucinated transcript
        - The user says "stop" / "cancel" / "never mind"
        """
        from jetauto_voice.intent_mapper import _HALLUCINATIONS

        # Greeting on first activation
        self._publish_tts("Yes, how may I help you?")
        self.get_logger().info("*** SPEAK NOW (greeting) ***")

        first_listen = True

        while not self._shutdown_event.is_set():
            if not first_listen:
                self._play_beep()
                self.get_logger().info("*** SPEAK NOW ***")

            first_listen = False

            audio_float = self._capture_with_vad(stream)

            if audio_float is None or len(audio_float) == 0:
                self.get_logger().info("No speech — returning to wake word")
                break

            text = self._transcribe(audio_float)
            if not text:
                self.get_logger().info("Empty transcript — returning to wake word")
                break

            # Filter Whisper hallucinations
            if text.lower().rstrip("?.!,") in _HALLUCINATIONS:
                self.get_logger().info(f'Hallucination filtered: "{text}" — returning to wake word')
                break

            self.get_logger().info(f'Transcribed: "{text}"')

            # Stop command exits the loop
            if self._is_stop_command(text):
                self.get_logger().info("Stop command — returning to wake word")
                self._publish_tts("Okay.")
                break

            # Dispatch intent; loop continues regardless of match result
            self._dispatch_intent(text)

    def _play_beep(self, freq: float = 880.0, duration: float = 0.15, volume: float = 0.4) -> None:
        """Play a short sine-wave beep directly via sounddevice.

        Gives the user an immediate audible 'speak now' cue without
        depending on the TTS node being running.

        Args:
            freq: Frequency in Hz. Default 880 Hz (A5).
            duration: Duration in seconds. Default 0.15s.
            volume: Amplitude in [0.0, 1.0]. Default 0.4.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
            t = np.linspace(0, duration, int(self._sample_rate * duration), endpoint=False)
            tone = (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)
            sd.play(tone, samplerate=self._sample_rate)
            sd.wait()
        except Exception as exc:
            self.get_logger().debug(f"Beep playback failed: {exc}")

    def _capture_with_vad(self, stream) -> Optional[np.ndarray]:
        """Capture speech using WebRTC VAD endpoint detection.

        Reads 20ms frames (320 samples at 16kHz) and runs webrtcvad on each.
        Returns audio from first speech frame through end-of-speech silence.
        Returns None if speech never starts within the listen timeout.

        Falls back to 5-second fixed capture if webrtcvad is unavailable.
        """
        # webrtcvad requires 10/20/30ms frames at 8/16/32kHz
        # 20ms at 16kHz = 320 samples
        VAD_FRAME_SAMPLES = 320

        try:
            import webrtcvad  # type: ignore[import]
            vad = webrtcvad.Vad(self._vad_aggressiveness)
        except ImportError:
            self.get_logger().warn(
                "webrtcvad not installed — using 5s fixed capture. "
                "Run: pip3 install webrtcvad"
            )
            audio_data, _ = stream.read(int(self._sample_rate * 5.0))
            return audio_data.flatten().astype(np.float32) / 32768.0

        # Drain audio buffer to clear the tail of the wake word before VAD starts
        drain_samples = int(self._vad_drain_ms * self._sample_rate / 1000)
        stream.read(drain_samples)

        timeout_frames = int(self._vad_listen_timeout_sec * self._sample_rate / VAD_FRAME_SAMPLES)
        max_frames = int(self._vad_max_duration_sec * self._sample_rate / VAD_FRAME_SAMPLES)
        min_speech_frames = max(1, self._vad_min_speech_ms // 20)  # 20ms per frame

        audio_frames = []
        speech_frames_total = 0
        speech_run = 0    # consecutive speech frames
        silence_run = 0   # consecutive silence frames after speech
        speech_started = False

        for i in range(max_frames):
            if self._shutdown_event.is_set():
                return None

            chunk, _ = stream.read(VAD_FRAME_SAMPLES)
            chunk_flat = chunk.flatten()
            audio_frames.append(chunk_flat)

            try:
                is_speech = vad.is_speech(chunk_flat.tobytes(), self._sample_rate)
            except Exception:
                is_speech = False

            if is_speech:
                speech_run += 1
                speech_frames_total += 1
                silence_run = 0
                if speech_run >= self._vad_speech_start_frames and not speech_started:
                    speech_started = True
                    self.get_logger().info("Speech detected — listening...")
            else:
                speech_run = 0
                if speech_started:
                    silence_run += 1
                    if silence_run >= self._vad_speech_end_frames:
                        elapsed_ms = len(audio_frames) * VAD_FRAME_SAMPLES * 1000 // self._sample_rate
                        self.get_logger().info(f"*** DONE LISTENING — captured {elapsed_ms}ms, transcribing...")
                        break
                else:
                    if i >= timeout_frames:
                        self.get_logger().info("Listen timeout — no speech detected, returning to wake word")
                        return None

        if not audio_frames or not speech_started:
            return None

        # Require a minimum amount of actual speech to avoid transcribing noise
        if speech_frames_total < min_speech_frames:
            self.get_logger().info(
                f"Too little speech ({speech_frames_total * 20}ms) — ignoring"
            )
            return None

        audio = np.concatenate(audio_frames).astype(np.float32) / 32768.0
        return audio

    # ------------------------------------------------------------------ #
    # STT
    # ------------------------------------------------------------------ #

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 audio array using faster-whisper.

        Args:
            audio: float32 numpy array, normalized to [-1.0, 1.0], 16 kHz mono.

        Returns:
            Stripped transcription string, or ``""`` on failure.
        """
        if self._stt_model is None:
            self.get_logger().error("STT model not loaded — cannot transcribe")
            return ""

        try:
            segments, _info = self._stt_model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            self.get_logger().error(f"STT transcription failed: {exc}")
            return ""

    # ------------------------------------------------------------------ #
    # Intent dispatch
    # ------------------------------------------------------------------ #

    def _is_stop_command(self, text: str) -> bool:
        """Return True if the user wants to stop the listening session."""
        t = text.lower().rstrip("?.!,")
        return any(phrase in t for phrase in [
            "stop", "cancel", "never mind", "nevermind", "quit", "exit",
            "that's all", "thats all", "goodbye", "bye", "done",
        ])

    def _dispatch_intent(self, text: str) -> None:
        """Route a transcript to the appropriate action.

        Checks for:
        1. Find-object intent  → enable detection + publish target + TTS
        2. Enable-detection command → enable detection + TTS
        3. Disable-detection command → disable detection + TTS
        4. Unknown → TTS apology

        Args:
            text: Transcribed speech string.
        """
        # 1. Find-object intent
        result = extract_target(text)
        if result is not None:
            yolo_label, raw_object = result
            self.get_logger().info(
                f'Intent: find "{raw_object}" → YOLO class "{yolo_label}"'
            )
            self._publish_detection_enable(True)
            self._publish_target(yolo_label)
            self._publish_tts(f"Looking for {raw_object}.")
            return

        # 2. Enable detection
        if is_enable_command(text):
            self.get_logger().info("Intent: enable detection")
            self._publish_detection_enable(True)
            self._publish_tts("Detection enabled.")
            return

        # 3. Disable detection
        if is_disable_command(text):
            self.get_logger().info("Intent: disable detection")
            self._publish_detection_enable(False)
            self._publish_tts("Detection disabled.")
            return

        # 4. Unknown
        self.get_logger().info(f'No intent matched for: "{text}"')
        self._publish_tts("Sorry, I didn't understand that.")

    # ------------------------------------------------------------------ #
    # Publishers
    # ------------------------------------------------------------------ #

    def _publish_detection_enable(self, enabled: bool) -> None:
        """Publish to /jetauto/detection/enable."""
        msg = Bool()
        msg.data = enabled
        self._detection_pub.publish(msg)
        state = "ENABLED" if enabled else "DISABLED"
        self.get_logger().info(f"Detection {state}")

    def _publish_target(self, label: str) -> None:
        """Publish target YOLO class label to /jetauto/detection/target."""
        msg = String()
        msg.data = label
        self._target_pub.publish(msg)

    def _publish_tts(self, text: str) -> None:
        """Publish text to /tts/speak for the TTS node."""
        msg = String()
        msg.data = text
        self._tts_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def destroy_node(self) -> None:
        """Signal the audio thread to stop before destroying the node."""
        self._shutdown_event.set()
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
