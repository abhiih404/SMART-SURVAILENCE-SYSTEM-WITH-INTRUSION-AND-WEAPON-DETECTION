"""
main.py
──────────────────────────────────────────────────────────────────────────────
Entry point for the Smart Surveillance System.

Usage:
    python main.py                         # use source from .env
    python main.py --source 0              # webcam 0
    python main.py --source /path/video.mp4
    python main.py --source 0 --record     # save output to alerts/

Controls:
    Q  – quit
    P  – pause / resume
    S  – take screenshot
──────────────────────────────────────────────────────────────────────────────
"""
import argparse
import os
import sys
import time
from datetime import datetime

import cv2

from config.settings import VIDEO_SOURCE, ALERTS_DIR
from surveillance_engine import SurveillanceEngine
from utils.logger import get_logger

log = get_logger("main")


def run(source, record: bool = False) -> None:
    log.info("Initialising surveillance engine …")
    engine = SurveillanceEngine()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Stream: {width}×{height} @ {fps_in:.1f} fps")

    writer = None
    if record:
        out_path = os.path.join(
            ALERTS_DIR,
            f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        log.info(f"Recording to {out_path}")

    paused   = False
    frame_n  = 0
    t_start  = time.time()

    log.info("Surveillance running. Press Q to quit, P to pause, S for screenshot.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                log.info("End of stream or read error.")
                break

            annotated = engine.process_frame(frame)

            # FPS overlay
            elapsed = time.time() - t_start
            fps_live = frame_n / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"FPS: {fps_live:.1f}", (width - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if writer:
                writer.write(annotated)

            cv2.imshow("Smart Surveillance", annotated)
            frame_n += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("Quit requested.")
            break
        elif key == ord("p"):
            paused = not paused
            log.info("Paused." if paused else "Resumed.")
        elif key == ord("s"):
            shot_path = os.path.join(
                ALERTS_DIR,
                f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            )
            cv2.imwrite(shot_path, annotated if not paused else frame)
            log.info(f"Screenshot saved: {shot_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log.info(f"Session ended. Processed {frame_n} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Surveillance System")
    parser.add_argument("--source", default=None,
                        help="Video source: webcam index (0,1,…) or file path")
    parser.add_argument("--record", action="store_true",
                        help="Record annotated output to alerts/")
    args = parser.parse_args()

    src = args.source
    if src is None:
        src = VIDEO_SOURCE
    elif src.isdigit():
        src = int(src)
    run(src, record=args.record)

