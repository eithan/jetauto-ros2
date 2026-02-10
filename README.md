# JetAuto ROS2 — Vision + TTS

Camera-based object detection with spoken answers for the JetAuto robot car.

## Hardware

- **Robot:** JetAuto (Hiwonder)
- **Compute:** NVIDIA Orin Nano 8GB
- **Sensors:** Depth Camera, Lidar
- **OS:** Ubuntu 22.04 + ROS2 Humble (pre-installed)

## Architecture

```
┌─────────────┐    /detected_objects     ┌─────────────┐
│  jetauto_    │ ──────────────────────►  │  jetauto_   │
│  vision      │   DetectedObjectArray    │  tts         │
│              │                          │              │
│ depth_cam ──►│                          │──► speakers  │
└─────────────┘                          └─────────────┘
```

**jetauto_vision** — Subscribes to the depth camera topic, runs YOLO (or MobileNet-SSD) inference, publishes detected objects with labels and confidence scores.

**jetauto_tts** — Subscribes to detected objects, generates spoken descriptions ("I see a cup and a keyboard"), plays audio through speakers.

## Packages

| Package | Description |
|---------|-------------|
| `jetauto_vision` | Object detection node (YOLOv8 via ultralytics) |
| `jetauto_tts` | Text-to-speech node (pyttsx3 offline TTS) |

## Quick Start

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/eithan/jetauto-ros2.git

# 2. Install Python dependencies
pip3 install ultralytics pyttsx3 opencv-python-headless

# 3. Build
cd ~/ros2_ws
colcon build --packages-select jetauto_vision jetauto_tts
source install/setup.bash

# 4. Launch everything
ros2 launch jetauto_vision vision_launch.py

# Or launch vision + TTS together
ros2 launch jetauto_tts tts_launch.py
```

## Configuration

Edit `src/jetauto_vision/config/vision_params.yaml` and `src/jetauto_tts/config/tts_params.yaml` to tune:
- Model selection (yolov8n/s/m)
- Confidence thresholds
- Camera topic
- TTS voice/rate/volume
- Cooldown between announcements

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Input camera feed |
| `/detected_objects` | `std_msgs/String` | JSON array of detections |
| `/tts/speak` | `std_msgs/String` | Manual TTS trigger |

## Development

This repo is developed off-robot (code-only) and deployed to the JetAuto for testing. No ROS2 installation needed for editing.

## License

MIT
