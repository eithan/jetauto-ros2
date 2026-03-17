#!/usr/bin/env python3
"""
Voice Commander Node — fully offline, open-source voice pipeline.

Flow::

    mic audio  →  openWakeWord (continuous, 80ms chunks)
                      │ wake word detected
                      ▼
               greeting TTS → beep → continuous listen session
                      │
              ┌───────┴───── VAD captures utterance ─────┐
              │              faster-whisper STT           │
              │              intent_mapper                │
              │                                           │
              │  find X / start vision → execute, END     │
              │  stop/disable vision   → execute, LISTEN  │
              │  Jarvis stop           → END              │
              │  unmatched / noise     → LISTEN           │
              └───────────────────────────────────────────┘

"END" = return to passive wake word listening.
"LISTEN" = keep the session open and capture next utterance.
"""

import os
import time
import threading
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String

from std_msgs.msg import Float32 as Float32Msg  # noqa: F401 — used via type hint
from jetauto_voice.intent_mapper import (
    extract_target,
    is_enable_command,
    is_disable_command,
    is_what_do_you_see_command,
    _HALLUCINATIONS,
)

_OWW_CHUNK_SAMPLES = 1280   # 80 ms @ 16 kHz
_OWW_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Timing constants (seconds)
# ---------------------------------------------------------------------------
_TTS_WPS = 2.5          # pyttsx3 approximate words-per-second
_TTS_OVERHEAD = 1.2     # pyttsx3 startup + buffer + reverb decay


class VoiceCommanderNode(Node):

    def __init__(self) -> None:
        super().__init__("voice_commander_node")

        # -- Parameters --
        self.declare_parameter("wake_word_model", "hey_jarvis")
        self.declare_parameter("wake_word_threshold", 0.5)
        self.declare_parameter("stt_model_size", "base.en")
        self.declare_parameter("stt_device", "cuda")
        self.declare_parameter("stt_compute_type", "float16")
        self.declare_parameter("mic_device_index", 1)
        self.declare_parameter("vad_aggressiveness", 2)
        self.declare_parameter("vad_speech_start_frames", 8)
        self.declare_parameter("vad_speech_end_frames", 30)
        self.declare_parameter("vad_min_speech_ms", 250)
        self.declare_parameter("vad_listen_timeout_sec", 30.0)
        self.declare_parameter("vad_max_duration_sec", 10.0)
        self.declare_parameter("wake_cooldown_sec", 5.0)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("session_timeout_sec", 300.0)

        # -- Read all parameters --
        p = self.get_parameter
        self._wake_word_model: str = p("wake_word_model").value
        self._wake_word_threshold: float = p("wake_word_threshold").value
        self._stt_model_size: str = p("stt_model_size").value
        self._stt_device: str = p("stt_device").value
        self._stt_compute_type: str = p("stt_compute_type").value
        self._mic_device_index: int = p("mic_device_index").value
        self._vad_aggressiveness: int = p("vad_aggressiveness").value
        self._vad_speech_start_frames: int = p("vad_speech_start_frames").value
        self._vad_speech_end_frames: int = p("vad_speech_end_frames").value
        self._vad_min_speech_ms: int = p("vad_min_speech_ms").value
        self._vad_listen_timeout_sec: float = p("vad_listen_timeout_sec").value
        self._vad_max_duration_sec: float = p("vad_max_duration_sec").value
        self._wake_cooldown_sec: float = p("wake_cooldown_sec").value
        self._sample_rate: int = p("sample_rate").value
        self._session_timeout_sec: float = p("session_timeout_sec").value

        # -- Publishers --
        self._detection_pub = self.create_publisher(Bool, "/jetauto/detection/enable", 1)
        self._target_pub = self.create_publisher(String, "/jetauto/detection/target", 1)
        self._tts_pub = self.create_publisher(String, "/tts/speak", 1)
        # TRANSIENT_LOCAL matches the dashboard subscriber — ensures the first
        # state publish is never lost due to DDS discovery timing.
        _qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._voice_state_pub = self.create_publisher(String, "/jetauto/voice/state", _qos_latched)

        # -- Subscriptions --
        self._caption_sub = self.create_subscription(
            String, '/scene_caption', self._on_scene_caption, 1
        )

        # -- State --
        self._shutdown_event = threading.Event()
        self._wake_word_detector = None
        self._stt_model = None
        self._stt_actual_device = self._stt_device
        self._last_caption: str = ''

        # -- Load models --
        self._init_wakeword()
        self._init_stt()

        # -- Background audio thread --
        self._listener_thread = threading.Thread(
            target=self._audio_loop, daemon=True, name="voice_listener",
        )
        self._listener_thread.start()

        self.get_logger().info(
            f"VoiceCommanderNode ready — wake='{self._wake_word_model}' "
            f"(thr={self._wake_word_threshold}), "
            f"STT={self._stt_model_size}@{self._stt_actual_device}"
        )

    # ================================================================== #
    # Model init
    # ================================================================== #

    def _init_wakeword(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            self.get_logger().error("openwakeword not installed!")
            return

        oww_dir = os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models"
        )
        local = os.path.join(oww_dir, f"{self._wake_word_model}_v0.1.onnx")

        if os.path.exists(local):
            source = local
        else:
            self.get_logger().info(f"Downloading OWW model '{self._wake_word_model}'…")
            try:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                from openwakeword.utils import download_models
                download_models([self._wake_word_model])
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            except Exception as e:
                self.get_logger().warn(f"OWW download failed: {e}")
            source = self._wake_word_model

        try:
            self._wake_word_detector = Model(
                wakeword_models=[source], inference_framework="onnx",
            )
            self.get_logger().info(f"openWakeWord loaded: '{self._wake_word_model}'")
        except Exception as e:
            self.get_logger().error(f"openWakeWord init failed: {e}")

    def _init_stt(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.get_logger().error("faster-whisper not installed!")
            return

        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            self._stt_model = WhisperModel(
                self._stt_model_size,
                device=self._stt_device,
                compute_type=self._stt_compute_type,
            )
            self.get_logger().info(
                f"faster-whisper: {self._stt_model_size} "
                f"on {self._stt_device}/{self._stt_compute_type}"
            )
        except Exception as e:
            self.get_logger().warn(f"STT on {self._stt_device} failed ({e}) — trying CPU")
            try:
                self._stt_model = WhisperModel(
                    self._stt_model_size, device="cpu", compute_type="int8",
                )
                self._stt_actual_device = "cpu"
                self.get_logger().info(f"faster-whisper CPU fallback: {self._stt_model_size}/int8")
            except Exception as e2:
                self.get_logger().error(f"STT CPU fallback failed: {e2}")
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ================================================================== #
    # Audio loop — wake word detection
    # ================================================================== #

    def _audio_loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            self.get_logger().error("sounddevice not installed!")
            return

        if self._wake_word_detector is None:
            self.get_logger().error("OWW not loaded — aborting audio loop")
            return

        device = self._mic_device_index if self._mic_device_index >= 0 else None
        cooldown_total = int(
            self._wake_cooldown_sec * self._sample_rate / _OWW_CHUNK_SAMPLES
        )

        self.get_logger().info(
            f"Audio loop started — {self._sample_rate}Hz, device={device}"
        )

        # Signal dashboard that voice pipeline is ready (dismisses loading overlay)
        self._pub_voice_state('ready')
        time.sleep(0.1)
        self._pub_voice_state('idle')

        with sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16",
            blocksize=_OWW_CHUNK_SAMPLES, device=device,
        ) as stream:
            cooldown = 0
            dbg_interval = int(5.0 * self._sample_rate / _OWW_CHUNK_SAMPLES)
            dbg_count = 0
            dbg_peak = 0.0

            while not self._shutdown_event.is_set():
                chunk, _ = stream.read(_OWW_CHUNK_SAMPLES)
                flat = chunk.flatten()

                if cooldown > 0:
                    cooldown -= 1
                    self._wake_word_detector.predict(flat)
                    continue

                pred = self._wake_word_detector.predict(flat)
                score = max(pred.values()) if pred else 0.0

                if score > dbg_peak:
                    dbg_peak = score
                dbg_count += 1
                if dbg_count >= dbg_interval:
                    self.get_logger().info(
                        f"[OWW] peak={dbg_peak:.3f} thr={self._wake_word_threshold}"
                    )
                    dbg_count = 0
                    dbg_peak = 0.0

                if score >= self._wake_word_threshold:
                    self.get_logger().info(f"Wake word (score={score:.3f})")
                    self._run_session(stream)
                    cooldown = cooldown_total
                    dbg_count = 0
                    dbg_peak = 0.0

    # ================================================================== #
    # Voice session — continuous listen after wake word
    # ================================================================== #

    def _run_session(self, stream) -> None:
        """Run a voice command session.

        1. Greeting TTS → wait → drain → beep
        2. Continuous VAD listen loop:
           - noise/empty/hallucination → silently loop (zero overhead)
           - matched command → execute → if detection-on, END; else drain+beep+loop
           - "Jarvis stop" → END
           - 5-min timeout → END
        """
        # --- Greeting ---
        self._pub_voice_state('speaking')
        self._speak_and_wait("Yes?")
        self._drain_stream(stream, 800)
        self._play_beep()
        self._pub_voice_state('listening')
        self.get_logger().info("Session open — listening for commands")

        t0 = time.time()

        while not self._shutdown_event.is_set():
            # Session clock
            if time.time() - t0 > self._session_timeout_sec:
                self.get_logger().info("Session timeout — returning to wake word")
                break

            # --- Capture speech ---
            audio = self._capture_speech(stream)
            if audio is None:
                # Silence timeout — no speech for vad_listen_timeout_sec
                # Just loop — session stays open until 5-min timeout
                continue

            # --- Transcribe ---
            self._pub_voice_state('processing')
            text = self._transcribe(audio)
            if not text:
                self._pub_voice_state('listening')
                continue
            if text.lower().strip().rstrip("?.!,") in _HALLUCINATIONS:
                self._pub_voice_state('listening')
                continue

            self.get_logger().info(f'Heard: "{text}"')

            # --- Jarvis stop ---
            if self._is_stop_command(text):
                self.get_logger().info("Jarvis stop — ending session")
                self._pub_voice_state('speaking')
                self._speak_and_wait("Okay.")
                break

            # --- Dispatch intent ---
            end_session = self._dispatch_intent(text)
            if end_session:
                self.get_logger().info("Session ending after command")
                break

            # Command executed, drain TTS echo then keep listening
            self._drain_stream(stream, 800)
            self._play_beep()
            self._pub_voice_state('listening')

        self._pub_voice_state('idle')

    # ================================================================== #
    # Speech capture (VAD only — no drains, no beeps)
    # ================================================================== #

    def _capture_speech(self, stream) -> Optional[np.ndarray]:
        """Capture one utterance using WebRTC VAD.

        Returns float32 audio on success, None on silence timeout.
        Silently ignores noise bursts (too-short detections).

        This method does NO draining or beeping — the caller manages those.
        """
        VAD_MS = 20
        VAD_SAMPLES = self._sample_rate * VAD_MS // 1000

        try:
            import webrtcvad
            vad = webrtcvad.Vad(self._vad_aggressiveness)
        except ImportError:
            self.get_logger().warn("webrtcvad missing — fixed 5s capture")
            data, _ = stream.read(int(self._sample_rate * 5.0))
            return data.flatten().astype(np.float32) / 32768.0

        timeout_frames = int(self._vad_listen_timeout_sec * 1000 / VAD_MS)
        max_frames = int(self._vad_max_duration_sec * 1000 / VAD_MS)
        min_speech = max(1, self._vad_min_speech_ms // VAD_MS)

        frames = []
        speech_run = 0
        silence_run = 0
        speech_total = 0
        started = False

        for i in range(max_frames):
            if self._shutdown_event.is_set():
                return None

            chunk, _ = stream.read(VAD_SAMPLES)
            flat = chunk.flatten()
            frames.append(flat)

            try:
                is_speech = vad.is_speech(flat.tobytes(), self._sample_rate)
            except Exception:
                is_speech = False

            if is_speech:
                speech_run += 1
                speech_total += 1
                silence_run = 0
                if speech_run >= self._vad_speech_start_frames and not started:
                    started = True
            else:
                speech_run = 0
                if started:
                    silence_run += 1
                    if silence_run >= self._vad_speech_end_frames:
                        ms = len(frames) * VAD_MS
                        self.get_logger().info(f"Captured {ms}ms ({speech_total * VAD_MS}ms speech)")
                        break
                elif i >= timeout_frames:
                    return None   # true silence — no speech at all

        if not started:
            return None

        if speech_total < min_speech:
            return None   # noise burst — caller will silently retry

        return np.concatenate(frames).astype(np.float32) / 32768.0

    # ================================================================== #
    # Helpers
    # ================================================================== #

    def _drain_stream(self, stream, ms: int) -> None:
        """Discard mic audio for `ms` wall-clock milliseconds."""
        deadline = time.time() + ms / 1000.0
        while time.time() < deadline and not self._shutdown_event.is_set():
            try:
                stream.read(_OWW_CHUNK_SAMPLES)
            except Exception:
                time.sleep(0.01)

    def _play_beep(self, freq: float = 880.0, dur: float = 0.15, vol: float = 0.4):
        """Audible 'your turn' cue."""
        try:
            import sounddevice as sd
            t = np.linspace(0, dur, int(self._sample_rate * dur), endpoint=False)
            sd.play((np.sin(2 * np.pi * freq * t) * vol).astype(np.float32),
                     samplerate=self._sample_rate)
            sd.wait()
        except Exception:
            pass

    def _speak_and_wait(self, text: str) -> None:
        """Publish TTS and sleep for the estimated speech duration."""
        msg = String()
        msg.data = text
        self._tts_pub.publish(msg)
        # Generous estimate: word count / WPS + overhead for pyttsx3 latency
        wait = max(1.5, len(text.split()) / _TTS_WPS + _TTS_OVERHEAD)
        time.sleep(wait)

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._stt_model is None:
            self.get_logger().error("STT not loaded")
            return ""
        try:
            segs, _ = self._stt_model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True,
            )
            return " ".join(s.text for s in segs).strip()
        except Exception as e:
            self.get_logger().error(f"STT error: {e}")
            return ""

    # ================================================================== #
    # Intent dispatch
    # ================================================================== #

    def _on_scene_caption(self, msg: String) -> None:
        """Cache the latest Florence-2 caption for on-demand retrieval."""
        if msg.data.strip():
            self._last_caption = msg.data.strip()

    def _is_stop_command(self, text: str) -> bool:
        # Strip all punctuation so "Jarvis, stop." matches same as "Jarvis stop"
        import re
        t = re.sub(r'[^\w\s]', '', text.lower().strip())
        return any(p in t for p in [
            "jarvis stop", "stop jarvis", "hey jarvis stop",
            "jarvis quit", "jarvis done", "jarvis bye",
        ])

    def _dispatch_intent(self, text: str) -> bool:
        """Execute a voice command. Returns True if session should end."""

        # 0. "What do you see?" → speak latest Florence-2 caption
        if is_what_do_you_see_command(text):
            if self._last_caption:
                self.get_logger().info(f'What do you see → "{self._last_caption}"')
                self._pub_voice_state('speaking')
                self._speak_and_wait(self._last_caption)
            else:
                self.get_logger().info('What do you see → no caption yet')
                self._pub_voice_state('speaking')
                self._speak_and_wait("I haven't captured a scene yet. Make sure vision is enabled.")
            return False  # stay in session

        # 1. Find object → announce only, no action (vision must be enabled separately)
        result = extract_target(text)
        if result is not None:
            yolo_label, raw_object = result
            self.get_logger().info(f'Find intent (announce only): "{raw_object}"')
            self._pub_voice_state('speaking')
            self._speak_and_wait(f"I'll look for {raw_object} when detection is enabled.")
            return False   # stay in session

        # 2. Enable detection → end session
        if is_enable_command(text):
            self.get_logger().info("Enable detection")
            self._pub_enable(True)
            self._pub_voice_state('speaking')
            self._speak_and_wait("Detection enabled.")
            return True

        # 3. Disable detection → stay in session
        if is_disable_command(text):
            self.get_logger().info("Disable detection")
            self._pub_enable(False)
            self._pub_voice_state('speaking')
            self._speak_and_wait("Detection disabled.")
            return False

        # 4. Disable voice → end session
        if self._is_disable_voice_command(text):
            self.get_logger().info("Disable voice — ending session")
            self._pub_voice_state('speaking')
            self._speak_and_wait("Voice disabled.")
            return True

        # 5. Unmatched — silently ignore, keep listening
        self.get_logger().info(f'No intent for: "{text}"')
        return False

    def _pub_voice_state(self, state: str) -> None:
        """Publish voice state to dashboard ('idle'|'listening'|'processing'|'speaking')."""
        msg = String()
        msg.data = state
        self._voice_state_pub.publish(msg)
        self.get_logger().debug(f'Voice state → {state}')

    def _is_disable_voice_command(self, text: str) -> bool:
        t = text.lower().strip()
        return any(p in t for p in [
            "disable voice", "turn off voice", "stop voice", "voice off",
        ])

    def _pub_enable(self, on: bool) -> None:
        msg = Bool()
        msg.data = on
        self._detection_pub.publish(msg)
        self.get_logger().info(f"Detection {'ENABLED' if on else 'DISABLED'}")

    def _pub_target(self, label: str) -> None:
        msg = String()
        msg.data = label
        self._target_pub.publish(msg)

    # ================================================================== #
    # Cleanup
    # ================================================================== #

    def destroy_node(self) -> None:
        self._shutdown_event.set()
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down by signal handler


if __name__ == "__main__":
    main()
