"""
surveillance_engine.py
──────────────────────────────────────────────────────────────────────────────
Core surveillance pipeline:

  Frame → Gaussian Blur → YOLOv8 (person detection) + NMS
        → Haar face crop → LBPH face recognition
        → YOLOv8 weapon detection
        → Loitering tracker
        → SVM threat classifier
        → Telegram alert (SUSPICIOUS / HIGH)
──────────────────────────────────────────────────────────────────────────────
"""
import pickle
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    CONFIDENCE_THRESHOLD, FACE_RECOGNITION_THRESHOLD,
    LOITERING_TIME_THRESHOLD, WEAPON_CONFIDENCE_THRESHOLD,
    LBPH_MODEL_PATH, SVM_MODEL_PATH, LABEL_MAP_PATH,
    YOLO_PERSON_MODEL, YOLO_WEAPON_MODEL,
    USE_GPU,
)
from utils.preprocessing import preprocess_frame, preprocess_face_roi, non_maximum_suppression
from utils.logger import get_logger
from utils.telegram_alert import send_alert

log = get_logger("engine")

# Colour scheme
_COLOURS = {
    "NORMAL":     (0, 200, 0),
    "SUSPICIOUS": (0, 165, 255),
    "HIGH":       (0, 0, 255),
    "UNKNOWN":    (100, 100, 255),
}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Weapon class names that appear in COCO / custom weapon dataset
_WEAPON_CLASS_NAMES = {"knife", "scissors", "gun", "pistol", "rifle", "weapon"}

# COCO class id for 'person'
_COCO_PERSON_ID = 0


class LoiteringTracker:
    """
    Tracks how long each unique track-id has been in the same region.
    A person is considered 'loitering' when their centroid hasn't moved
    more than `move_threshold` pixels for `time_threshold` seconds.
    """

    def __init__(self, time_threshold: float = 10.0, move_threshold: float = 60.0):
        self.time_threshold  = time_threshold
        self.move_threshold  = move_threshold

        self._first_seen: dict[int, float]         = {}
        self._last_centroid: dict[int, tuple]       = {}
        self._stationary_since: dict[int, float]    = {}

    def update(self, track_id: int, centroid: tuple[float, float]) -> bool:
        now = time.time()
        cx, cy = centroid

        if track_id not in self._first_seen:
            self._first_seen[track_id]       = now
            self._last_centroid[track_id]    = (cx, cy)
            self._stationary_since[track_id] = now
            return False

        lx, ly = self._last_centroid[track_id]
        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5

        if dist > self.move_threshold:
            # Person moved – reset stationary timer
            self._last_centroid[track_id]    = (cx, cy)
            self._stationary_since[track_id] = now

        presence_time = now - self._first_seen[track_id]
        elapsed = now - self._stationary_since[track_id]

        # must stay at least 5 seconds before loitering logic starts
        if presence_time < 5:
            return False

        return elapsed >= self.time_threshold

    def cleanup(self, active_ids: set):
        stale = set(self._first_seen.keys()) - active_ids
        for sid in stale:
            self._first_seen.pop(sid, None)
            self._last_centroid.pop(sid, None)
            self._stationary_since.pop(sid, None)


class SurveillanceEngine:
    def __init__(self):
        device = "cuda" if USE_GPU else "cpu"
        log.info(f"Using device: {device}")
        self.identity_history = defaultdict(list)
        self.threat_history = defaultdict(list)
        self.track_identity = {}
        self.track_identity_conf = {}
        self.track_last_seen = {}
        self.track_memory_time = 8  # seconds to remember identity
        # YOLOv8 – person detection (downloads yolov8n.pt automatically)
        log.info("Loading YOLOv8 person model …")
        self.person_model = YOLO(YOLO_PERSON_MODEL)
        self.device = device#jsfksdj

        # YOLOv8 – weapon detection
        import os
        self.weapon_model: Optional[YOLO] = None
        if os.path.exists(YOLO_WEAPON_MODEL):
            log.info("Loading YOLOv8 weapon model …")
            self.weapon_model = YOLO(YOLO_WEAPON_MODEL)
            self.weapon_model.to(device)
        else:
            log.warning(
                f"Weapon model not found at {YOLO_WEAPON_MODEL}. "
                "Weapon detection disabled. See README for download instructions."
            )

        # LBPH face recogniser
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read(LBPH_MODEL_PATH)
            log.info(f"LBPH model loaded from {LBPH_MODEL_PATH}")
        except Exception:
            log.warning("LBPH model not found – run train_model.py first. Face recognition disabled.")
            self.recognizer = None

        # Label map
        try:
            with open(LABEL_MAP_PATH, "rb") as f:
                self.label_map: dict[int, str] = pickle.load(f)
        except Exception:
            self.label_map = {}

        # SVM classifier
        try:
            with open(SVM_MODEL_PATH, "rb") as f:
                self.svm: object = pickle.load(f)
            log.info("SVM classifier loaded.")
        except Exception:
            log.warning("SVM model not found – run train_model.py first.")
            self.svm = None

        self.loitering_tracker = LoiteringTracker(
            time_threshold=LOITERING_TIME_THRESHOLD
        )

        # Alert cooldown: don't spam the same track-id within 60 s
        self._last_alert: dict[int, float] = defaultdict(float)
        self._alert_cooldown = 60.0

        # Class labels for SVM output
        self._threat_labels = {0: "NORMAL", 1: "SUSPICIOUS", 2: "HIGH"}

    # ─────────────────────────────────────────────────────────────────────────

    def _detect_persons(self, frame: np.ndarray):
        """
        Run YOLOv8 with tracking enabled, apply custom NMS,
        return list of dicts with keys: box, track_id, confidence.
        """
        results = self.person_model.track(
            frame,
            persist=True,
            classes=[_COCO_PERSON_ID],
            conf=CONFIDENCE_THRESHOLD,
            device=self.device,#sjfld
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes_raw = results[0].boxes
            xyxy   = boxes_raw.xyxy.cpu().numpy()   # (N,4)
            confs  = boxes_raw.conf.cpu().numpy()   # (N,)
            ids    = boxes_raw.id                   # may be None if tracking lost
            track_ids = ids.cpu().numpy().astype(int) if ids is not None else list(range(len(xyxy)))

            kept = non_maximum_suppression(xyxy, confs, iou_threshold=0.45,
                                           score_threshold=CONFIDENCE_THRESHOLD)

            for i in kept:
                detections.append({
                    "box":       xyxy[i].astype(int),
                    "track_id":  int(track_ids[i]),
                    "confidence": float(confs[i]),
                })
        return detections

    def _detect_weapons(self, frame: np.ndarray) -> bool:
        """Returns True if any weapon is detected in the frame."""
        if self.weapon_model is None:
            return False
        results = self.weapon_model(frame, conf=WEAPON_CONFIDENCE_THRESHOLD,device=self.device, verbose=False)
        if results and results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.cpu().numpy():
                cls_name = self.weapon_model.names.get(int(cls_id), "").lower()
                print("Weapon detected:", cls_name)   # debug

                if cls_name in ["pistol", "knife"]:
                    return True
                '''if any(w in cls_name for w in _WEAPON_CLASS_NAMES):
                    return True'''
        return False

    def _recognize_face(self, frame: np.ndarray, box: np.ndarray) -> tuple[str, float]:
        """
        Crop person ROI, detect face, run LBPH.
        Returns (name, confidence) – name='Unknown' if not recognised.
        """
        if self.recognizer is None:
            return "Unknown", 100.0

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # Add margin around face to include hair and forehead
        margin = 30

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        roi = frame[y1:y2, x1:x2]
        '''x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]'''
        if roi.size == 0:
            return "Unknown", 100.0

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_roi, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return "Unknown", 100.0

        # Use largest face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray_roi[fy: fy + fh, fx: fx + fw]
        face_crop = cv2.resize(face_crop, (100, 100))
        face_crop = cv2.equalizeHist(face_crop)

        label_id, confidence = self.recognizer.predict(face_crop)
        if confidence > 1000:
            confidence = 100

        # Debug print to tune threshold
        print(f"Face confidence: {confidence}")

        # Better threshold control
        if confidence < FACE_RECOGNITION_THRESHOLD:
            name = self.label_map.get(label_id, f"Person_{label_id}")
        else:
            name = "Unknown"
        return name, float(confidence)

    def _classify_threat(self, is_unknown: bool, loitering: bool,
                         weapon: bool, face_conf: float) -> str:
        """Use SVM to classify threat level."""
        # Hard rule override: weapon always → HIGH
        if weapon:
            return "HIGH"

        if self.svm is None:
            # Fallback rule-based
            if is_unknown and loitering:
                return "SUSPICIOUS"
            return "NORMAL"

        # Normalise confidence to 0-1 (LBPH: 0=perfect, 100=worst)
        conf_norm = min(face_conf / 100.0, 1.0)
        X = np.array([[int(is_unknown), int(loitering), int(weapon), conf_norm]])
        pred = self.svm.predict(X)[0]
        return self._threat_labels.get(int(pred), "NORMAL")

    def _send_alert_async(self, level: str, name: str, frame: np.ndarray, extra: str):
        t = threading.Thread(target=send_alert, args=(level, name, frame.copy(), extra), daemon=True)
        t.start()

    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline for a single frame.
        Returns annotated frame.
        """
        preprocessed = preprocess_frame(frame)

        # Weapon scan on full frame
        weapon_detected = self._detect_weapons(preprocessed)

        # Person detection
        persons = self._detect_persons(preprocessed)
        active_ids = {p["track_id"] for p in persons}
        self.loitering_tracker.cleanup(active_ids)

        annotated = frame.copy()

        for person in persons:
            box       = person["box"]
            track_id  = person["track_id"]
            det_conf  = person["confidence"]

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Face recognition
            now = time.time()

            # If we already know this track identity, reuse it
            if track_id in self.track_identity:
                if now - self.track_last_seen[track_id] < self.track_memory_time:
                    name = self.track_identity[track_id]
                    face_conf = self.track_identity_conf[track_id]
                else:
                    name, face_conf = self._recognize_face(frame, box)
            else:
                name, face_conf = self._recognize_face(frame, box)

            if name != "Unknown":
                self.track_identity[track_id] = name
                self.track_identity_conf[track_id] = face_conf
            self.track_last_seen[track_id] = now

            history = self.identity_history[track_id]
            history.append(name)

            if len(history) > 10:
                history.pop(0)

            name = max(set(history), key=history.count)
            is_unknown = (name == "Unknown" and face_conf > FACE_RECOGNITION_THRESHOLD)

            # Apply loitering only for unknown persons
            if is_unknown:
                loitering = self.loitering_tracker.update(track_id, (cx, cy))
            else:
                loitering = False
                self.loitering_tracker._stationary_since.pop(track_id, None)
                self.loitering_tracker._first_seen.pop(track_id, None)
                self.loitering_tracker._last_centroid.pop(track_id, None)

            # Threat classification
            threat = self._classify_threat(is_unknown, loitering, weapon_detected, face_conf)
            # Store prediction for evaluation
            #with open("logs/predictions.csv", "a") as f:
            #    f.write(f"{track_id},{name},{is_unknown},{loitering},{weapon_detected},{threat}\n")
            # Log face recognition prediction
            #with open("logs/face_recognition_log.csv", "a") as f:
            #    f.write(f"{track_id},{name},{face_conf}\n")
            history = self.threat_history[track_id]
            history.append(threat)

            if len(history) > 10:
                history.pop(0)

            # choose most frequent threat in last 10 frames
            threat = max(set(history), key=history.count)
            colour = _COLOURS.get(threat, (200, 200, 200))

            # ── Draw annotation ──────────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            label_text = f"ID:{track_id} {name} [{threat}]"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if loitering:
                cv2.putText(annotated, "LOITERING", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # ── Send Telegram alert ──────────────────────────────────────────
            #if threat in ("SUSPICIOUS", "HIGH") and history.count(threat) >= 8:
                
                 # ensure person has been present for some time
                now = time.time()
                if now - self._last_alert[track_id] > self._alert_cooldown:
                    if threat == "HIGH"and weapon_detected:#jkkkkkkkjjjjjjjjjjj

                        self._last_alert[track_id] = now
                        self._send_alert_async(threat, name, annotated, "Weapon detected")
                        return annotated
                    elif threat == "SUSPICIOUS" and loitering:
                         if track_id in self.loitering_tracker._first_seen:

                            presence_time = now - self.loitering_tracker._first_seen[track_id]

                            if presence_time >= 4:   # must stay at least 4 seconds
                                self._last_alert[track_id] = now
                                self._send_alert_async("SUSPICIOUS", name, annotated, "Loitering detected")
                    '''presence_time = time.time() - self.loitering_tracker._first_seen[track_id]

                    if presence_time < 4:   # must stay at least 5 seconds
                        continue
                    now = time.time()
                    if now - self._last_alert[track_id] > self._alert_cooldown:
                        self._last_alert[track_id] = now
                        extra = []
                        if loitering:  extra.append("Loitering detected")
                        if weapon_detected: extra.append("Weapon detected")
                        self._send_alert_async(threat, name, annotated, " | ".join(extra))'''

        # ── HUD overlays ────────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if weapon_detected:
            cv2.putText(annotated, "WEAPON DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        person_count = len(persons)
        cv2.putText(annotated, f"Persons: {person_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


        return annotated

      """Bytetrack""""
