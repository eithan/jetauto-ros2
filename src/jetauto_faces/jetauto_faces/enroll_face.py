#!/usr/bin/env python3
"""
Face enrollment CLI tool for the JetAuto face recognition system.

Processes a directory of face images for a given person, extracts
InsightFace embeddings, averages them, and saves to the face database.

Usage:
    # Enroll from a directory of images
    ros2 run jetauto_faces enroll_face -- --name "Eithan" --images ~/faces/eithan/

    # Enroll from a single image
    ros2 run jetauto_faces enroll_face -- --name "Eithan" --images ~/faces/eithan/photo1.jpg

    # Enroll using webcam capture (interactive)
    ros2 run jetauto_faces enroll_face -- --name "Eithan" --capture 5

    # List enrolled faces
    ros2 run jetauto_faces enroll_face -- --list

    # Delete an enrolled face
    ros2 run jetauto_faces enroll_face -- --delete "Eithan"

Options:
    --name NAME          Person's name for enrollment
    --images PATH        Path to image file or directory of images
    --capture N          Capture N photos from webcam for enrollment
    --db PATH            Face database directory (default: ./data/faces)
    --model MODEL        InsightFace model name (default: buffalo_l)
    --det-size SIZE      Detection size (default: 640)
    --gpu-id ID          GPU device id, -1 for CPU (default: 0)
    --list               List all enrolled faces
    --delete NAME        Delete an enrolled face
    --show               Display images with detected faces during enrollment
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def get_insightface_app(model_name: str, det_size: int, gpu_id: int):
    """Initialize and return an InsightFace FaceAnalysis app."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print('Error: insightface not installed.')
        print('Run: pip3 install insightface onnxruntime-gpu')
        sys.exit(1)

    providers = ['CPUExecutionProvider']
    if gpu_id >= 0:
        providers = [
            ('CUDAExecutionProvider', {'device_id': gpu_id}),
            'CPUExecutionProvider',
        ]

    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=gpu_id, det_size=(det_size, det_size))
    return app


def enroll_from_images(app, image_path: str, show: bool = False) -> list:
    """Extract face embeddings from image(s).

    Args:
        app: InsightFace FaceAnalysis instance
        image_path: Path to single image or directory of images
        show: If True, display images with detected faces

    Returns:
        List of (embedding, image_path) tuples
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    embeddings = []

    if os.path.isfile(image_path):
        image_files = [image_path]
    elif os.path.isdir(image_path):
        image_files = sorted([
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
    else:
        print(f'Error: path not found: {image_path}')
        return embeddings

    if not image_files:
        print(f'Error: no image files found in {image_path}')
        return embeddings

    print(f'Processing {len(image_files)} image(s)...')

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f'  ⚠ Could not read: {os.path.basename(img_path)}')
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f'  ⚠ No face detected: {os.path.basename(img_path)}')
            continue

        if len(faces) > 1:
            print(
                f'  ⚠ Multiple faces ({len(faces)}) in {os.path.basename(img_path)}'
                ' — using largest face'
            )
            # Pick the face with the largest bounding box area
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        face = faces[0]
        if face.embedding is None:
            print(f'  ⚠ No embedding extracted: {os.path.basename(img_path)}')
            continue

        embeddings.append((face.embedding, img_path))
        det_score = face.det_score if hasattr(face, 'det_score') else 0.0
        print(f'  ✓ {os.path.basename(img_path)} (detection score: {det_score:.3f})')

        if show:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Enrollment', img)
            cv2.waitKey(1000)

    if show:
        cv2.destroyAllWindows()

    return embeddings


def enroll_from_webcam(app, num_captures: int, show: bool = True) -> list:
    """Capture photos from webcam and extract face embeddings.

    Args:
        app: InsightFace FaceAnalysis instance
        num_captures: Number of photos to capture
        show: If True, display webcam feed with face detection

    Returns:
        List of (embedding, capture_index) tuples
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return []

    print(f'\nWebcam enrollment: will capture {num_captures} face(s)')
    print('Position your face in the camera. Press SPACE to capture, Q to quit.\n')

    embeddings = []
    captured = 0

    while captured < num_captures:
        ret, frame = cap.read()
        if not ret:
            print('Error: could not read from webcam')
            break

        # Run face detection on current frame for preview
        display = frame.copy()
        faces = app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            color = (0, 255, 0) if face.embedding is not None else (0, 0, 255)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        status = f'Captured: {captured}/{num_captures} | SPACE=capture Q=quit'
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show:
            cv2.imshow('Face Enrollment', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print('Enrollment cancelled by user')
            break

        if key == ord(' '):
            if len(faces) == 0:
                print('  ⚠ No face detected — try again')
                continue

            face = faces[0]
            if len(faces) > 1:
                # Pick largest
                faces.sort(
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )
                face = faces[0]

            if face.embedding is None:
                print('  ⚠ No embedding extracted — try again')
                continue

            captured += 1
            embeddings.append((face.embedding, f'webcam_capture_{captured}'))
            print(f'  ✓ Capture {captured}/{num_captures}')

    cap.release()
    cv2.destroyAllWindows()

    return embeddings


def save_enrollment(name: str, embeddings: list, db_path: str):
    """Average embeddings and save to the face database.

    Args:
        name: Person's name
        embeddings: List of (embedding, source) tuples
        db_path: Path to face database directory
    """
    if not embeddings:
        print('Error: no valid embeddings to save')
        return False

    os.makedirs(db_path, exist_ok=True)

    # Average all embeddings
    all_embeddings = np.array([e[0] for e in embeddings], dtype=np.float32)
    avg_embedding = np.mean(all_embeddings, axis=0)

    # Normalize
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    # Save as .npz
    safe_name = name.lower().replace(' ', '_')
    filepath = os.path.join(db_path, f'{safe_name}.npz')

    np.savez(
        filepath,
        name=name,
        embedding=avg_embedding,
        num_samples=len(embeddings),
    )

    print(f'\n✅ Enrolled "{name}" with {len(embeddings)} sample(s)')
    print(f'   Saved to: {filepath}')
    print(f'   Embedding dimension: {avg_embedding.shape[0]}')

    # Show per-sample similarity to the averaged embedding
    if len(embeddings) > 1:
        print('\n   Per-sample similarity to average:')
        for emb, source in embeddings:
            emb_norm = emb / np.linalg.norm(emb)
            sim = float(np.dot(emb_norm, avg_embedding))
            src_name = os.path.basename(source) if os.path.exists(str(source)) else source
            print(f'     {src_name}: {sim:.4f}')

    return True


def list_enrolled(db_path: str):
    """List all enrolled faces in the database."""
    if not os.path.isdir(db_path):
        print(f'Face database not found: {db_path}')
        return

    files = [f for f in os.listdir(db_path) if f.endswith('.npz')]

    if not files:
        print('No faces enrolled yet.')
        return

    print(f'\nEnrolled faces ({len(files)}):')
    print('-' * 50)

    for filename in sorted(files):
        filepath = os.path.join(db_path, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
            name = str(data['name'])
            num = int(data.get('num_samples', 0))
            dim = data['embedding'].shape[0]
            print(f'  {name:20s}  {num:3d} samples  ({dim}d embedding)')
        except Exception as e:
            print(f'  {filename}: error loading — {e}')


def delete_enrolled(name: str, db_path: str):
    """Delete an enrolled face from the database."""
    if not os.path.isdir(db_path):
        print(f'Face database not found: {db_path}')
        return

    safe_name = name.lower().replace(' ', '_')
    filepath = os.path.join(db_path, f'{safe_name}.npz')

    if not os.path.exists(filepath):
        print(f'No enrollment found for "{name}" at {filepath}')
        return

    os.remove(filepath)
    print(f'✅ Deleted enrollment for "{name}"')


def main():
    parser = argparse.ArgumentParser(
        description='Face enrollment tool for JetAuto face recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Enroll from images
  ros2 run jetauto_faces enroll_face -- --name "Eithan" --images ~/faces/eithan/

  # Enroll from webcam (capture 5 photos)
  ros2 run jetauto_faces enroll_face -- --name "Eithan" --capture 5

  # List enrolled faces
  ros2 run jetauto_faces enroll_face -- --list

  # Delete enrollment
  ros2 run jetauto_faces enroll_face -- --delete "Eithan"
        ''',
    )

    parser.add_argument('--name', type=str, help="Person's name")
    parser.add_argument('--images', type=str, help='Path to image(s)')
    parser.add_argument('--capture', type=int, help='Capture N webcam photos')
    parser.add_argument(
        '--db', type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))), 'data', 'faces'),
        help='Face database directory',
    )
    parser.add_argument('--model', type=str, default='buffalo_l', help='InsightFace model')
    parser.add_argument('--det-size', type=int, default=640, help='Detection size')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--list', action='store_true', help='List enrolled faces')
    parser.add_argument('--delete', type=str, help='Delete enrolled face')
    parser.add_argument('--show', action='store_true', help='Show images during enrollment')

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_enrolled(args.db)
        return

    # Handle --delete
    if args.delete:
        delete_enrolled(args.delete, args.db)
        return

    # Enrollment requires --name
    if not args.name:
        parser.error('--name is required for enrollment')

    if not args.images and args.capture is None:
        parser.error('Provide --images PATH or --capture N')

    print(f'Face Enrollment Tool')
    print(f'  Name:    {args.name}')
    print(f'  Model:   {args.model}')
    print(f'  DB:      {args.db}')
    print()

    # Initialize InsightFace
    print('Loading InsightFace model...')
    app = get_insightface_app(args.model, args.det_size, args.gpu_id)
    print('Model loaded.\n')

    # Collect embeddings
    embeddings = []

    if args.images:
        embeddings.extend(enroll_from_images(app, args.images, show=args.show))

    if args.capture:
        embeddings.extend(enroll_from_webcam(app, args.capture, show=True))

    # Save
    if embeddings:
        save_enrollment(args.name, embeddings, args.db)
    else:
        print('\n❌ No valid face embeddings collected. Enrollment failed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
