"""
train_model.py
──────────────────────────────────────────────────────────────────────────────
Trains:
  1. LBPH face recogniser  (OpenCV, fast on CPU/GPU)
  2. SVM threat classifier (scikit-learn)
     Features: [face_confidence, is_unknown, loitering_flag, weapon_flag]
     Labels  : NORMAL / SUSPICIOUS / HIGH

Run after adding new known persons:
    python train_model.py
──────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
import pickle

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from config.settings import (
    KNOWN_FACES_DIR, MODELS_DIR,
    LBPH_MODEL_PATH, SVM_MODEL_PATH, LABEL_MAP_PATH,
)
from utils.preprocessing import preprocess_face_roi
from utils.logger import get_logger

log = get_logger("trainer")


# ── 1.  LBPH face recogniser ──────────────────────────────────────────────────

def load_face_dataset() -> tuple[list, list, dict]:
    """
    Walk KNOWN_FACES_DIR/<PersonName>/*.jpg and return
    (faces_list, labels_list, label_map {int -> name}).
    """
    faces, labels = [], []
    label_map: dict[int, str] = {}
    label_id = 0

    persons = sorted(os.listdir(KNOWN_FACES_DIR))
    if not persons:
        log.error(f"No persons found in {KNOWN_FACES_DIR}. Run capture_faces.py first.")
        sys.exit(1)

    for person_name in persons:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))]
        if not images:
            log.warning(f"No images in {person_dir} – skipping.")
            continue

        label_map[label_id] = person_name
        for img_file in images:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            faces.append(img)
            labels.append(label_id)

        log.info(f"  Loaded {len(images)} images for '{person_name}' (id={label_id})")
        label_id += 1

    return faces, labels, label_map


def train_lbph(faces: list, labels: list) -> cv2.face.LBPHFaceRecognizer:
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(LBPH_MODEL_PATH)
    log.info(f"LBPH model saved → {LBPH_MODEL_PATH}")
    return recognizer


# ── 2.  SVM threat classifier ─────────────────────────────────────────────────
# We generate synthetic but physically meaningful training data here.
# In production you can replace / augment this with real logged events.

def generate_svm_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Feature vector: [is_unknown (0/1), loitering (0/1), weapon (0/1), face_conf_norm]
    face_conf_norm: LBPH confidence normalised 0-1 (higher → less certain)
    
    Labels: 0=NORMAL, 1=SUSPICIOUS, 2=HIGH
    """
    rng = np.random.default_rng(42)

    def _samples(n, is_unknown, loitering, weapon, conf_range):
        conf = rng.uniform(*conf_range, size=n)
        X = np.column_stack([
            np.full(n, is_unknown),
            np.full(n, loitering),
            np.full(n, weapon),
            conf,
        ])
        return X

    # NORMAL  – known person, no loitering, no weapon
    X_normal = _samples(300, 0, 0, 0, (0.0, 0.4))

    # SUSPICIOUS – unknown person + loitering, no weapon
    X_susp1 = _samples(200, 1, 1, 0, (0.4, 1.0))
    # SUSPICIOUS – unknown person, short stay
    X_susp2 = _samples(100, 1, 0, 0, (0.5, 1.0))

    # HIGH – weapon detected (regardless of identity)
    X_high1 = _samples(200, 1, 1, 1, (0.4, 1.0))
    X_high2 = _samples(100, 0, 0, 1, (0.0, 0.5))   # known person with weapon = still HIGH
    X_high3 = _samples(100, 1, 0, 1, (0.4, 1.0))

    X = np.vstack([X_normal, X_susp1, X_susp2, X_high1, X_high2, X_high3])
    y = np.concatenate([
        np.zeros(300),          # NORMAL
        np.ones(300),           # SUSPICIOUS
        np.full(400, 2),        # HIGH
    ])
    return X, y


def train_svm(X: np.ndarray, y: np.ndarray) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, class_weight="balanced")),
    ])
    scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    log.info(f"SVM cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(X, y)
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    log.info(f"SVM model saved → {SVM_MODEL_PATH}")
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("═══ Loading face dataset ═══")
    faces, labels, label_map = load_face_dataset()
    log.info(f"Total samples: {len(faces)} across {len(label_map)} persons")

    log.info("═══ Training LBPH face recogniser ═══")
    train_lbph(faces, labels)

    # Save label map
    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump(label_map, f)
    log.info(f"Label map saved → {LABEL_MAP_PATH}  {label_map}")

    log.info("═══ Training SVM threat classifier ═══")
    X, y = generate_svm_training_data()
    train_svm(X, y)

    log.info("═══ Training complete ═══")


if __name__ == "__main__":
    main()
