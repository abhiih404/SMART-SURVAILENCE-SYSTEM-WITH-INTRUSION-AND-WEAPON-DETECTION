"""
capture_faces.py
────────────────────────────────────────────────────────────
Capture face images of known persons for LBPH training.

Usage:
    python capture_faces.py --name "Ashwin Krish" --samples 300
    python capture_faces.py --name "Manu" --source video.mp4

Controls:
    Q  → Quit early
────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time
import cv2

from config.settings import KNOWN_FACES_DIR, VIDEO_SOURCE
from utils.preprocessing import preprocess_face_roi
from utils.logger import get_logger

log = get_logger("capture")

# Haar face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def capture_faces(name: str, num_samples: int = 300, source=None):

    src = source if source is not None else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        log.error(f"Cannot open video source: {src}")
        sys.exit(1)

    # Folder for this person
    save_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    capture_delay = 0.2  # seconds between captures
    last_capture = 0

    log.info(f"Capturing {num_samples} face samples for '{name}'")
    log.info("Press Q to quit early")

    while count < num_samples:

        ret, frame = cap.read()

        if not ret:
            log.warning("Frame read failed – end of stream?")
            break

        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Capturing: {name} | Saved: {count}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        cv2.putText(
            display,
            "Move head slowly (left/right/up/down)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        cv2.imshow("Face Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Auto capture faces with delay
        if len(faces) > 0 and (time.time() - last_capture) > capture_delay:

            # Use largest detected face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

            roi = frame[y:y + h, x:x + w]

            processed = preprocess_face_roi(roi)

            filepath = os.path.join(
                save_dir,
                f"{name.replace(' ', '_')}_{count:04d}.jpg"
            )

            cv2.imwrite(filepath, processed)

            count += 1
            last_capture = time.time()

            log.info(f"Saved sample {count}/{num_samples} -> {filepath}")

    cap.release()
    cv2.destroyAllWindows()

    log.info(f"Done. {count} samples saved to {save_dir}")
    log.info("Run: python train_model.py to retrain the recogniser.")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Capture face samples for a known person"
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Person name (e.g. 'John Doe')"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of samples to capture (default 300)"
    )

    parser.add_argument(
        "--source",
        default=None,
        help="Video source (webcam index or video file)"
    )

    args = parser.parse_args()

    if args.name is None:
        args.name = input("Enter person's name: ")

    src = args.source

    if src is not None and src.isdigit():
        src = int(src)

    capture_faces(args.name, args.samples, src)
