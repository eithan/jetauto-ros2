# JetAuto ROS2 Roadmap

## ✅ Phase 1: Basic Object Detection + TTS (DONE)
- YOLOv8n running on Orin Nano GPU
- pyttsx3 offline TTS announcing detected objects
- Camera topic: `/depth_cam/rgb/image_raw`

## 🔜 Phase 2: Improve Recognition Accuracy
- Upgrade model from `yolov8n` to `yolov8s` or `yolov8m` (edit `vision_params.yaml`)
- Raise `confidence_threshold` from 0.5 to 0.65-0.7 (reduce false positives like phantom toilets)
- Both are config-only changes, no rebuild needed

## 🔜 Phase 3: Better Voice
- Replace pyttsx3 with **Piper TTS** (offline, natural-sounding, lightweight)
- Alternative: Coqui TTS (heavier but more voices)
- Runs locally on Orin Nano

## 🔜 Phase 4: Conversation / Chat
- Add microphone input with **Whisper** (small model) for speech-to-text
- Connect to an LLM for conversational responses
  - Option A: Local model via **Ollama** on the Orin Nano
  - Option B: API call to cloud LLM
- Goal: ask the robot questions, get spoken answers

## 🔜 Phase 5: Recognize Specific People & Pets
- Face recognition via **InsightFace** (embedding-based, few-shot)
  - Take a few photos of each person/pet
  - Store embeddings, match at runtime
- Alternatively: fine-tune YOLO on labeled photos
- Teach it names — "That's Eithan" / "That's [dog name]"

---

*Created: 2026-03-01*
