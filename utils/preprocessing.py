"""
utils/preprocessing.py
Frame preprocessing pipeline:
  1. Gaussian blur  – reduces sensor noise
  2. NMS            – applied after YOLO to suppress duplicate bounding boxes
"""
import cv2
import numpy as np


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Light Gaussian blur to suppress high-frequency noise before detection.
    A small kernel (3x3) keeps edges intact while smoothing grain.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.50,
) -> list[int]:
    """
    Pure-Python NMS on top of YOLO results so we can tune thresholds
    independently per class.

    Parameters
    ----------
    boxes  : (N, 4) array  [x1, y1, x2, y2]
    scores : (N,)   array  confidence values
    iou_threshold   : boxes with IoU > this are suppressed
    score_threshold : boxes with score < this are dropped first

    Returns
    -------
    List of kept indices.
    """
    if len(boxes) == 0:
        return []

    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Drop low-confidence detections
    keep_mask = scores >= score_threshold
    boxes, scores = boxes[keep_mask], scores[keep_mask]
    original_indices = np.where(keep_mask)[0].tolist()

    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(original_indices[i])
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        iw  = np.maximum(0.0, ix2 - ix1 + 1)
        ih  = np.maximum(0.0, iy2 - iy1 + 1)
        inter = iw * ih
        iou   = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]

    return kept


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline applied to every captured frame."""
    return apply_gaussian_blur(frame)


def preprocess_face_roi(face_roi: np.ndarray, size: tuple = (100, 100)) -> np.ndarray:
    """Normalize a face crop for LBPH training / prediction."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    resized = cv2.resize(gray, size)
    equalized = cv2.equalizeHist(resized)
    return equalized
