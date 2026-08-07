"""
download_weapon_model.py
──────────────────────────────────────────────────────────────────────────────
Downloads a pre-trained YOLOv8 weapon detection model from a public source.

We use the publicly available 'Weapon Detection' model trained on a dataset
containing pistols, knives, and rifles.

Run once before starting the system:
    python download_weapon_model.py
──────────────────────────────────────────────────────────────────────────────
"""
import os
import urllib.request
import sys

from config.settings import WEAPONS_MODEL_DIR, YOLO_WEAPON_MODEL
from utils.logger import get_logger

log = get_logger("downloader")

# Option A – Roboflow / HuggingFace hosted weight (replace URL with your source)
# This URL points to a community YOLOv8n model trained on weapons.
# You can also train your own on Roboflow Universe dataset:
#   https://universe.roboflow.com/roboflow-100/pistols-and-rifles-detection
MODEL_URL = (
    "https://huggingface.co/spaces/Ultralytics/HUB/resolve/main/"
    "examples/weapon_yolov8n.pt"
)

# Option B – provide your own weights file by copying it to:
#   weapons_model/weapon_yolov8.pt


def download_model():
    if os.path.exists(YOLO_WEAPON_MODEL):
        log.info(f"Weapon model already exists at {YOLO_WEAPON_MODEL}")
        return

    log.info(f"Downloading weapon model from:\n  {MODEL_URL}")
    log.info("This may take a moment …")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = downloaded / total_size * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                sys.stdout.write(f"\r  [{bar}] {pct:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, YOLO_WEAPON_MODEL, _progress)
        print()  # newline after progress bar
        log.info(f"Saved to {YOLO_WEAPON_MODEL}")
    except Exception as e:
        log.error(f"Download failed: {e}")
        log.info("")
        log.info("MANUAL DOWNLOAD OPTIONS:")
        log.info("  1. Roboflow Universe – search 'weapon detection YOLOv8'")
        log.info("     https://universe.roboflow.com")
        log.info("  2. Train custom model:")
        log.info("     pip install roboflow")
        log.info("     # Then follow train_custom_weapon_model.py")
        log.info("")
        log.info(f"  Place the .pt file at: {YOLO_WEAPON_MODEL}")


if __name__ == "__main__":
    download_model()
