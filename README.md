# JetAuto ROS2 — Vision + Voice + TTS

Camera-based object detection with spoken announcements for the JetAuto robot car.

## Hardware

- **Robot:** JetAuto (Hiwonder)
- **Compute:** NVIDIA Orin Nano 8GB
- **Sensors:** Depth Camera, Lidar
- **OS:** Ubuntu 22.04 + ROS2 Humble (pre-installed)

## Architecture

```
┌──────────────────┐  /detected_objects        ┌─────────────┐
│  jetauto_vision  │ ──────────────────────►   │ jetauto_tts │
│  (YOLOv8)        │  DetectedObjectArray      │ (pyttsx3)   │
│                  │  (jetauto_msgs)           │             │
│  depth_cam ─────►│                           │──► speakers │
└──────────────────┘                           └─────────────┘
         ▲
         │ /jetauto/detection/enable (Bool)
         │ /jetauto/detection/target (String)
         │
┌──────────────────┐
│  jetauto_voice   │◄── mic array (ALSA)
│  voice_commander │
│  (openWakeWord   │
│   + faster-      │──► /tts/speak ──► jetauto_tts ──► speakers
│     whisper)     │
└──────────────────┘
```

**jetauto_vision** — Subscribes to the depth camera topic, runs YOLOv8 inference, publishes typed `DetectedObjectArray` messages.

**jetauto_tts** — Subscribes to detected objects, generates spoken descriptions ("I can see a cup and a keyboard"), plays audio through speakers. Also listens on `/tts/speak` for arbitrary text.

**jetauto_voice** — Offline voice pipeline: openWakeWord (always-on wake word) + faster-whisper STT → intent parsing → publishes detection commands and target class labels. No iFlyTek or network dependency.

**jetauto_msgs** — Shared message definitions (`DetectedObject`, `DetectedObjectArray`).

## Packages

| Package | Type | Description |
| --- | --- | --- |
| `jetauto_msgs` | ament_cmake | Custom message types (DetectedObject, DetectedObjectArray) |
| `jetauto_vision` | ament_python | Object detection lifecycle node (YOLOv8 via ultralytics) |
| `jetauto_tts` | ament_python | Text-to-speech lifecycle node (pyttsx3 offline TTS) |
| `jetauto_voice` | ament_python | Offline voice commander (openWakeWord + faster-whisper) |

## Quick Start

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/eithan/jetauto-ros2.git

# 2. Install system dependency for sounddevice
sudo apt-get install libportaudio2 portaudio19-dev

# 3. Install Python dependencies
pip3 install -r jetauto-ros2/requirements.txt

# 4. Build (msgs must build first)
cd ~/ros2_ws
colcon build --packages-select jetauto_msgs jetauto_vision jetauto_tts jetauto_voice
source install/setup.bash

# 5. Launch everything (vision + TTS)
ros2 launch jetauto_tts tts_launch.py

# Launch vision only
ros2 launch jetauto_vision vision_launch.py

# Launch the offline voice commander
ros2 launch jetauto_voice voice_control.launch.py
```

## Deployment

This repo is developed off-robot and deployed via rsync. Edit `deploy.sh` to set your robot's IP, then:

```bash
# Sync code and build on robot
./deploy.sh

# Or just sync without building
./deploy.sh --sync-only
```

## Configuration

Edit YAML files to tune behavior:

**`src/jetauto_vision/config/vision_params.yaml`** — Model selection (yolov8n/s/m), confidence threshold, camera topic, inference rate.

**`src/jetauto_tts/config/tts_params.yaml`** — TTS voice/rate/volume, cooldown between announcements, confidence filter.

## Topics

| Topic | Type | Publisher | Description |
| --- | --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | camera driver | Input camera feed |
| `/detected_objects` | `jetauto_msgs/DetectedObjectArray` | `jetauto_vision` | Typed detection results |
| `/detected_objects/image` | `sensor_msgs/Image` | `jetauto_vision` | Annotated image with bounding boxes |
| `/tts/speak` | `std_msgs/String` | `jetauto_voice` / user | Manual TTS trigger |
| `/jetauto/detection/enable` | `std_msgs/Bool` | `jetauto_voice` | Enable/disable YOLO detection |
| `/jetauto/detection/target` | `std_msgs/String` | `jetauto_voice` | Target YOLO class to search for |

## Lifecycle Management

Both nodes are ROS2 **lifecycle nodes**. The launch files auto-transition them through `configure → activate`. You can also control them manually:

```bash
# Check state
ros2 lifecycle get /detector_node

# Deactivate (pause processing)
ros2 lifecycle set /detector_node deactivate

# Re-activate
ros2 lifecycle set /detector_node activate
```

## Voice Commander

The `jetauto_voice` package provides a fully offline, open-source voice pipeline that replaces the original iFlyTek/asr_node dependency.

### How it works

1. **Always-on wake word** — `openWakeWord` runs on a background thread consuming ~1–3% CPU, listening for the configured wake word (default: `hey_jarvis`).
2. **Capture** — on detection, `N` seconds of audio are captured from the mic.
3. **STT** — `faster-whisper` transcribes the audio locally (CUDA on Orin Nano, falls back to CPU/int8 automatically).
4. **Intent parsing** — regex patterns extract the target object ("find the bottle" → `bottle`).
5. **YOLO class mapping** — the object name is mapped to a canonical COCO class label.
6. **ROS2 publish** — detection is enabled and the target label is published, so `jetauto_vision` picks it up.

### Installing on Orin Nano

```bash
# System dependency (PortAudio for sounddevice)
sudo apt-get install libportaudio2 portaudio19-dev

# Python packages
pip3 install openwakeword faster-whisper sounddevice numpy

# openWakeWord downloads small ONNX models on first run (~10 MB)
# To pre-download:
python3 -c "from openwakeword.model import Model; Model(wakeword_models=['hey_jarvis'])"
```

### Running

```bash
# Default: hey_jarvis wake word, base Whisper model, CUDA
ros2 launch jetauto_voice voice_control.launch.py

# Custom wake word and sensitivity
ros2 launch jetauto_voice voice_control.launch.py \
    wake_word_model:=alexa \
    wake_word_threshold:=0.4

# Smaller/faster STT model (less accurate but fast on CPU)
ros2 launch jetauto_voice voice_control.launch.py \
    stt_model_size:=tiny \
    stt_device:=cpu

# Use the legacy iFlyTek bridge instead (requires asr_node running)
ros2 launch jetauto_voice voice_control.launch.py use_iflytek:=true
```

### Tuning wake word sensitivity

The `wake_word_threshold` parameter controls sensitivity:

| Value | Effect |
|-------|--------|
| `0.3` | More sensitive — fewer missed detections, more false positives |
| `0.5` | Default — balanced |
| `0.7` | Less sensitive — fewer false positives, may miss soft speech |

Increase the threshold in noisy environments; decrease it if the robot doesn't respond reliably.

### Training a custom wake word

openWakeWord supports custom models trained with just a few minutes of positive examples. See the [openWakeWord training guide](https://github.com/dscripka/openWakeWord#training-new-models) for instructions. Place the `.onnx` model file on the robot and set:

```bash
ros2 launch jetauto_voice voice_control.launch.py \
    wake_word_model:=/path/to/my_wake_word.onnx
```

### Voice commands

| Say... | Action |
|--------|--------|
| `hey jarvis, find the bottle` | Enable detection, set target to `bottle` |
| `hey jarvis, look for a person` | Enable detection, set target to `person` |
| `hey jarvis, where is the phone` | Enable detection, set target to `cell phone` |
| `hey jarvis, start detection` | Enable detection (no specific target) |
| `hey jarvis, stop detection` | Disable detection |

### COCO class mapping (selected examples)

| Spoken | YOLO class |
|--------|-----------|
| phone, mobile, smartphone | `cell phone` |
| fridge | `refrigerator` |
| sofa | `couch` |
| bike | `bicycle` |
| tv, television, monitor | `tv` |
| teddy | `teddy bear` |
| bag, rucksack | `backpack` |
| bottles (plural) | `bottle` |

Full mapping is defined in `src/jetauto_voice/jetauto_voice/intent_mapper.py`.

## Tests

```bash
# Voice commander tests (no ROS2 or hardware required)
python3 -m pytest src/jetauto_voice/test/ -v

# All packages via colcon
cd ~/ros2_ws
colcon test --packages-select jetauto_vision jetauto_tts jetauto_voice
colcon test-result --verbose
```

## License

MIT
