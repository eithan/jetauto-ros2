#!/usr/bin/env python3
"""
Standalone face recognition demo — no ROS2 required.

Opens a webcam, runs InsightFace detection + recognition against enrolled
faces, draws annotated bounding boxes, and optionally speaks greetings.

Exercises the same core logic as face_recognition_node.py:
  - InsightFace model loading (same model, same providers)
  - Face database loading (same .npz format)
  - Cosine similarity matching (same algorithm + threshold)
  - Greeting with cooldown (same logic)
  - Annotated display (same color scheme)

Usage:
    python3 demo_recognition.py                          # defaults
    python3 demo_recognition.py --db ./data/faces        # custom db path
    python3 demo_recognition.py --threshold 0.5          # stricter matching
    python3 demo_recognition.py --no-greet               # disable TTS
    python3 demo_recognition.py --model buffalo_s        # lighter model
    python3 demo_recognition.py --camera 1               # different camera

Controls:
    Q or ESC  — quit
    R         — reload face database
    G         — toggle greetings on/off
    +/-       — adjust recognition threshold
    SPACE     — freeze/unfreeze frame
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def load_face_db(db_path: str) -> dict:
    """Load enrolled face embeddings from .npz files.

    Same format as face_recognition_node._load_face_db().
    Returns {name: normalized_embedding}.
    """
    face_db = {}

    if not os.path.isdir(db_path):
        print(f'⚠ Face database not found: {db_path}')
        print('  Enroll faces first:')
        print(f'    python3 enroll_face.py --name "YourName" --capture 5 --db {db_path}')
        return face_db

    for filename in os.listdir(db_path):
        if not filename.endswith('.npz'):
            continue
        filepath = os.path.join(db_path, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
            name = str(data['name'])
            embedding = data['embedding'].astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            face_db[name] = embedding
            num_samples = int(data.get('num_samples', 0))
            print(f'  Loaded: {name} ({num_samples} enrollment samples)')
        except Exception as e:
            print(f'  ⚠ Failed to load {filename}: {e}')

    return face_db


def identify_face(embedding: np.ndarray, face_db: dict, threshold: float) -> tuple:
    """Compare a face embedding against the enrolled database.

    Same algorithm as face_recognition_node._identify_face().
    Returns (name, confidence).
    """
    embedding = embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    best_name = 'unknown'
    best_score = 0.0

    for name, db_embedding in face_db.items():
        score = float(np.dot(embedding, db_embedding))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score

    return 'unknown', best_score


def init_tts():
    """Initialize pyttsx3 TTS engine. Returns None if unavailable."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        print(f'⚠ TTS unavailable ({e}) — greetings will be text-only')
        return None


def speak(engine, text: str):
    """Non-blocking speech (best-effort)."""
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def draw_face(frame, bbox, name: str, confidence: float, threshold: float):
    """Draw face bounding box and label. Same color scheme as the ROS node."""
    x1, y1, x2, y2 = bbox.astype(int)

    if name != 'unknown':
        color = (0, 255, 0)  # green for recognized
        label = f'{name} ({confidence:.2f})'
    else:
        color = (0, 0, 255)  # red for unknown
        label = f'unknown ({confidence:.2f})'

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(
        frame, label, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )


def draw_hud(frame, fps: float, threshold: float, greet_enabled: bool,
             num_enrolled: int, frozen: bool):
    """Draw heads-up display with stats."""
    h, w = frame.shape[:2]
    lines = [
        f'FPS: {fps:.1f}',
        f'Threshold: {threshold:.2f}',
        f'Enrolled: {num_enrolled}',
        f'Greetings: {"ON" if greet_enabled else "OFF"}',
    ]
    if frozen:
        lines.append('*** FROZEN ***')

    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (200, 200, 200), 1)

    # Controls hint at bottom
    controls = 'Q=quit  R=reload  G=greet  +/-=threshold  SPACE=freeze'
    cv2.putText(frame, controls, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                 0.4, (150, 150, 150), 1)


def main():
    parser = argparse.ArgumentParser(
        description='Standalone face recognition demo (no ROS2 required)',
    )
    parser.add_argument(
        '--db', type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )), 'data', 'faces'),
        help='Face database directory (default: ../data/faces)',
    )
    parser.add_argument('--model', type=str, default='buffalo_l',
                        help='InsightFace model name')
    parser.add_argument('--det-size', type=int, default=640,
                        help='Detection size')
    parser.add_argument('--gpu-id', type=int, default=-1,
                        help='GPU id (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Recognition threshold')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--greeting-cooldown', type=float, default=30.0,
                        help='Seconds before re-greeting same person')
    parser.add_argument('--greeting-template', type=str, default='Hello {name}!',
                        help='Greeting template ({name} replaced)')
    parser.add_argument('--no-greet', action='store_true',
                        help='Disable TTS greetings')
    parser.add_argument('--no-display', action='store_true',
                        help='Run headless (print detections to console)')
    args = parser.parse_args()

    # --- Load InsightFace ---
    print('Loading InsightFace model...')
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print('Error: insightface not installed.')
        print('Run: pip3 install insightface onnxruntime')
        sys.exit(1)

    providers = ['CPUExecutionProvider']
    if args.gpu_id >= 0:
        providers = [
            ('CUDAExecutionProvider', {'device_id': args.gpu_id}),
            'CPUExecutionProvider',
        ]

    app = FaceAnalysis(name=args.model, providers=providers)
    app.prepare(ctx_id=args.gpu_id, det_size=(args.det_size, args.det_size))
    print(f'Model loaded: {args.model} (det_size={args.det_size})\n')

    # --- Load face database ---
    print('Loading face database...')
    face_db = load_face_db(args.db)
    print(f'{len(face_db)} face(s) enrolled\n')

    # --- Init TTS ---
    tts_engine = None
    greet_enabled = not args.no_greet
    if greet_enabled:
        tts_engine = init_tts()
    greeting_times: dict = {}  # {name: last_greeting_time}

    # --- Open camera ---
    print(f'Opening camera {args.camera}...')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Error: could not open camera {args.camera}')
        sys.exit(1)
    print('Camera opened. Running...\n')

    threshold = args.threshold
    frozen = False
    frozen_frame = None
    frame_times = []

    try:
        while True:
            if not frozen:
                ret, frame = cap.read()
                if not ret:
                    print('Camera read failed')
                    break
            else:
                frame = frozen_frame.copy()

            t0 = time.time()

            # --- Run InsightFace ---
            faces = app.get(frame)

            # --- Process each face ---
            for face in faces:
                if face.embedding is None:
                    continue

                name, confidence = identify_face(
                    face.embedding, face_db, threshold
                )

                # Draw on frame
                if not args.no_display:
                    draw_face(frame, face.bbox, name, confidence, threshold)

                # Console output
                if name != 'unknown':
                    print(f'  ✓ {name} ({confidence:.3f})')
                else:
                    print(f'  ? unknown (best: {confidence:.3f})')

                # Greeting
                if name != 'unknown' and greet_enabled:
                    now = time.time()
                    last = greeting_times.get(name, 0.0)
                    if now - last >= args.greeting_cooldown:
                        greeting_times[name] = now
                        greeting = args.greeting_template.replace('{name}', name)
                        print(f'  🗣 "{greeting}"')
                        speak(tts_engine, greeting)

            # --- FPS tracking ---
            elapsed = time.time() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            # --- Display ---
            if not args.no_display:
                draw_hud(frame, fps, threshold, greet_enabled,
                         len(face_db), frozen)
                cv2.imshow('Face Recognition Demo', frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):  # Q or ESC
                    break

                elif key == ord('r'):  # Reload database
                    print('\nReloading face database...')
                    face_db = load_face_db(args.db)
                    print(f'{len(face_db)} face(s) enrolled\n')

                elif key == ord('g'):  # Toggle greetings
                    greet_enabled = not greet_enabled
                    print(f'Greetings {"enabled" if greet_enabled else "disabled"}')

                elif key in (ord('+'), ord('=')):  # Increase threshold
                    threshold = min(1.0, threshold + 0.05)
                    print(f'Threshold: {threshold:.2f}')

                elif key == ord('-'):  # Decrease threshold
                    threshold = max(0.0, threshold - 0.05)
                    print(f'Threshold: {threshold:.2f}')

                elif key == ord(' '):  # Freeze/unfreeze
                    frozen = not frozen
                    if frozen:
                        frozen_frame = frame.copy()
                    print('Frozen' if frozen else 'Unfrozen')
            else:
                # Headless mode — small delay to avoid CPU spin
                time.sleep(0.03)

    except KeyboardInterrupt:
        print('\nInterrupted')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('Done.')


if __name__ == '__main__':
    main()
