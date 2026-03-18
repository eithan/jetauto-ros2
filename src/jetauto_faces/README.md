# jetauto_faces — Face Recognition Package

Face recognition for the JetAuto robot using [InsightFace](https://github.com/deepinsight/insightface). Detects and identifies people in real-time, greets them by name via TTS.

## Architecture

```
detector_node (YOLO)          camera image
    │ /detected_objects            │ /depth_cam/rgb/image_raw
    ▼                              ▼
┌──────────────────────────────────────┐
│       face_recognition_node          │
│  • crops person regions from frame   │
│  • runs InsightFace detection        │
│  • extracts face embeddings          │
│  • compares against enrolled faces   │
│  • publishes RecognizedFaceArray     │
│  • greets known people via TTS       │
└──────────────┬───────────────────────┘
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
/recognized  /recognized  /tts/speak
  _faces     _faces/image  (greeting)
```

## Dependencies

Install on the Jetson (or any machine for testing):

```bash
pip3 install insightface onnxruntime-gpu
# For CPU-only testing:
# pip3 install insightface onnxruntime
```

The InsightFace model (`buffalo_l` by default) downloads automatically on first run (~600MB).

## Face Enrollment

Before the robot can recognize anyone, you need to enroll faces.

### From images (recommended)

Take 5-10 photos of each person from different angles and lighting conditions. Place them in a directory.

```bash
# Enroll from a directory of images
ros2 run jetauto_faces enroll_face -- --name "Eithan" --images ~/faces/eithan/

# Enroll from a single image
ros2 run jetauto_faces enroll_face -- --name "Mom" --images ~/faces/mom/photo.jpg
```

### From webcam (interactive)

```bash
# Capture 5 photos interactively
ros2 run jetauto_faces enroll_face -- --name "Eithan" --capture 5
```

Press SPACE to capture each photo, Q to quit.

### Managing enrollments

```bash
# List all enrolled faces
ros2 run jetauto_faces enroll_face -- --list

# Delete an enrollment
ros2 run jetauto_faces enroll_face -- --delete "Eithan"
```

### Tips for good enrollment

- Use **5-10 diverse images** per person (different angles, lighting, expressions)
- Ensure the face is clearly visible (not occluded, not blurry)
- Include photos from the distance/angle the robot will typically see people
- One face per image works best; if multiple faces are present, the largest is used

## Running the Node

### Standalone

```bash
ros2 launch jetauto_faces face_recognition.launch.py
```

### With the full vision pipeline

The face recognition node works alongside the existing detector_node. It subscribes to `/detected_objects` to know when a person is in frame, then runs face recognition on the camera image.

## Configuration

See `config/face_params.yaml` for all parameters. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `buffalo_l` | InsightFace model (`buffalo_l`, `buffalo_s`) |
| `recognition_threshold` | `0.4` | Min cosine similarity for a match |
| `recognition_interval` | `0.5` | Seconds between recognition attempts |
| `greeting_cooldown` | `60.0` | Seconds before re-greeting same person |
| `auto_greet` | `true` | Automatically greet recognized people |
| `greeting_template` | `Hello {name}!` | TTS greeting text |
| `faces_db_path` | `data/faces` | Directory with enrolled `.npz` files |

## ROS2 Topics

### Subscribed
- `/detected_objects` (DetectedObjectArray) — person detections from YOLO
- `/depth_cam/rgb/image_raw` (Image) — camera frames
- `/faces/reload` (String) — trigger face database reload

### Published
- `/recognized_faces` (RecognizedFaceArray) — recognized face results
- `/recognized_faces/image` (Image) — annotated image with face bounding boxes
- `/tts/speak` (String) — greeting text for TTS node

## Performance

On the Jetson Orin Nano (8GB):
- InsightFace `buffalo_l`: ~300MB GPU memory, ~10 FPS face detection
- InsightFace `buffalo_s`: ~200MB GPU memory, ~15 FPS face detection
- Face embedding comparison: <1ms per face (CPU)

## Testing

```bash
cd src/jetauto_faces
pytest test/
```
