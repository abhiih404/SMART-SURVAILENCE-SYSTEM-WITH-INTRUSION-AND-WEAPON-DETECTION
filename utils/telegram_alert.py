"""
utils/telegram_alert.py
Sends Telegram alerts asynchronously with an annotated image attachment.
Uses python-telegram-bot v20 (asyncio-based).
"""
import asyncio
import io
import os
from datetime import datetime

import cv2
import numpy as np

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ALERTS_DIR
from utils.logger import get_logger

log = get_logger("telegram")

_EMOJI = {"SUSPICIOUS": "⚠️", "HIGH": "🚨"}


async def _send(level: str, label: str, frame: np.ndarray, extra: str = "") -> None:
    try:
        from telegram import Bot

        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        emoji = _EMOJI.get(level, "ℹ️")
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        caption = (
            f"{emoji} *SURVEILLANCE ALERT*\n"
            f"Level : `{level}`\n"
            f"Person: `{label}`\n"
            f"Time  : `{ts}`\n"
        )
        if extra:
            caption += f"Detail: {extra}\n"

        # Save alert image locally
        alert_path = os.path.join(ALERTS_DIR, f"alert_{level}_{ts.replace(' ', '_').replace(':', '-')}.jpg")
        cv2.imwrite(alert_path, frame)

        # Encode to bytes for Telegram
        _, buf = cv2.imencode(".jpg", frame)
        photo_bytes = io.BytesIO(buf.tobytes())
        photo_bytes.name = "alert.jpg"

        await bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=photo_bytes,
            caption=caption,
            parse_mode="Markdown",
        )
        log.info(f"Telegram alert sent: {level} – {label}")
    except Exception as exc:
        log.error(f"Telegram send failed: {exc}")


def send_alert(level: str, label: str, frame: np.ndarray, extra: str = "") -> None:
    """
    Thread-safe entry point. Runs the async sender in a new event loop
    so it can be called from any thread without blocking the video loop.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured – alert skipped.")
        return
    if level not in ("SUSPICIOUS", "HIGH"):
        return
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(level, label, frame, extra))
        loop.close()
    except Exception as exc:
        log.error(f"Alert loop error: {exc}")
