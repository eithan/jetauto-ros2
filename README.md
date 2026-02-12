# JetAuto ROS2 — Vision + TTS

Camera-based object detection with spoken announcements for the JetAuto robot car.

## Hardware

- **Robot:** JetAuto (Hiwonder)
- **Compute:** NVIDIA Orin Nano 8GB
- **Sensors:** Depth Camera, Lidar
- **OS:** Ubuntu 22.04 + ROS2 Humble (pre-installed)

## Architecture

```
┌─────────────┐  /detected_objects        ┌─────────────┐
│  jetauto_    │ ──────────────────────►   │  jetauto_   │
│  vision      │  DetectedObjectArray      │  tts         │
│              │  (jetauto_msgs)           │              │
│ depth_cam ──►│                           │──► speakers  │
└─────────────┘                           └─────────────┘
```

**jetauto_vision** — Subscribes to the depth camera topic, runs YOLOv8 inference, publishes typed `DetectedObjectArray` messages.

**jetauto_tts** — Subscribes to detected objects, generates spoken descriptions ("I can see a cup and a keyboard"), plays audio through speakers.

**jetauto_msgs** — Shared message definitions (`DetectedObject`, `DetectedObjectArray`).

## Packages

| Package | Type | Description |
| --- | --- | --- |
| `jetauto_msgs` | ament_cmake | Custom message types (DetectedObject, DetectedObjectArray) |
| `jetauto_vision` | ament_python | Object detection lifecycle node (YOLOv8 via ultralytics) |
| `jetauto_tts` | ament_python | Text-to-speech lifecycle node (pyttsx3 offline TTS) |

## Quick Start

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/eithan/jetauto-ros2.git

# 2. Install Python dependencies
pip3 install -r jetauto-ros2/requirements.txt

# 3. Build (msgs must build first)
cd ~/ros2_ws
colcon build --packages-select jetauto_msgs jetauto_vision jetauto_tts
source install/setup.bash

# 4. Launch everything (vision + TTS)
ros2 launch jetauto_tts tts_launch.py

# Or launch vision only
ros2 launch jetauto_vision vision_launch.py
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

| Topic | Type | Description |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Input camera feed |
| `/detected_objects` | `jetauto_msgs/DetectedObjectArray` | Typed detection results |
| `/detected_objects/image` | `sensor_msgs/Image` | Annotated image with bounding boxes |
| `/tts/speak` | `std_msgs/String` | Manual TTS trigger |

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

## Tests

```bash
cd ~/ros2_ws
colcon test --packages-select jetauto_vision jetauto_tts
colcon test-result --verbose
```

## License

MIT
