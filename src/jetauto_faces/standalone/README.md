# Standalone Testing (No ROS2 Required)

Test the face recognition system on any machine with a webcam — Mac, Linux, Windows.

## Setup

```bash
pip3 install insightface onnxruntime opencv-python numpy
# Optional for voice greetings:
pip3 install pyttsx3
```

The InsightFace model (`buffalo_l`, ~600MB) downloads automatically on first run.

## Step 1: Enroll Faces

The enrollment tool has zero ROS2 dependencies — run it directly:

```bash
cd src/jetauto_faces

# From webcam (interactive — press SPACE to capture, Q to quit)
python3 jetauto_faces/enroll_face.py --name "Eithan" --capture 5 --gpu-id -1

# From images
python3 jetauto_faces/enroll_face.py --name "Eithan" --images ~/faces/eithan/ --gpu-id -1

# List enrolled faces
python3 jetauto_faces/enroll_face.py --list

# Delete
python3 jetauto_faces/enroll_face.py --delete "Eithan"
```

> **Mac note:** Use `--gpu-id -1` for CPU mode (no CUDA on Mac).

## Step 2: Run Face Recognition Demo

```bash
python3 standalone/demo_recognition.py --gpu-id -1
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--db PATH` | `../data/faces` | Face database directory |
| `--model NAME` | `buffalo_l` | InsightFace model (`buffalo_l`, `buffalo_s`) |
| `--threshold N` | `0.4` | Min similarity for a match |
| `--camera N` | `0` | Camera device index |
| `--gpu-id N` | `-1` | GPU id (-1 = CPU) |
| `--greeting-cooldown N` | `30.0` | Seconds before re-greeting |
| `--no-greet` | off | Disable TTS greetings |
| `--no-display` | off | Headless mode (console only) |

### Keyboard Controls

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| R | Reload face database |
| G | Toggle greetings on/off |
| +/- | Adjust recognition threshold |
| SPACE | Freeze/unfreeze frame |

## What This Tests

The standalone demo exercises the **exact same code paths** as the robot:

| Component | ROS2 Node | Standalone Demo |
|-----------|-----------|-----------------|
| InsightFace model loading | `_load_model()` | Same `FaceAnalysis` init |
| Face database loading | `_load_face_db()` | Same `.npz` loading + normalization |
| Cosine similarity matching | `_identify_face()` | Same `identify_face()` algorithm |
| Greeting with cooldown | `_maybe_greet()` | Same cooldown logic |
| Annotated display | `_publish_annotated()` | Same color scheme + drawing |

The only difference: on the robot, YOLO detects "person" first and the face node crops that region. In the standalone demo, InsightFace detects faces directly in the full frame (it has its own face detector built in). The recognition and matching logic is identical.
