"""
face_matching.py — robust face matcher for FaceGuard AI (supports fr_128 & dct_256)

- Detect best face -> crop -> compute BOTH probe embeddings:
    * fr_128  (if `face_recognition` available)  -> Euclidean distance (<= 0.60)
    * dct_256 (always available)                 -> Cosine similarity  (>= 0.84)
- Compare probe with each user's stored embedding using the correct metric
- Hard thresholds (per-model) — NO defaulting to first user
- Confidence is normalized per-model to [0..1] for UI
"""

from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# repo root on path
sys.path.append(str(Path(__file__).parent))

from src.face_detector import FaceDetector
from database import db

logger = logging.getLogger(__name__)

# Optional face_recognition (for fr_128)
try:
    import face_recognition  # type: ignore
    _HAS_FR = True
except Exception:
    _HAS_FR = False


# ---------------------- bbox helpers (same as registration) ------------------
def _normalize_bbox(bbox, W: int, H: int) -> Tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return 0, 0, W, H
    x0, y0, a, b = map(float, bbox)

    def is_norm(*vals): return all(0.0 <= v <= 1.5 for v in vals)

    if ((a <= 1.5 and b <= 1.5 and is_norm(x0, y0, a, b)) or (a > 1.5 and b > 1.5)):
        if is_norm(x0, y0, a, b):
            x, y, w, h = x0 * W, y0 * H, a * W, b * H
        else:
            x, y, w, h = x0, y0, a, b
    else:
        if is_norm(x0, y0, a, b):
            x1, y1, x2, y2 = x0 * W, y0 * H, a * W, b * H
        else:
            x1, y1, x2, y2 = x0, y0, a, b
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(2, min(w, W - x)))
    h = int(max(2, min(h, H - y)))
    return x, y, w, h


def _expand_bbox(x: int, y: int, w: int, h: int, W: int, H: int, margin: float = 0.12) -> Tuple[int, int, int, int]:
    dx, dy = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(W, x + w + dx), min(H, y + h + dy)
    return x1, y1, max(2, x2 - x1), max(2, y2 - y1)


def _best_face_224(img: np.ndarray, detector: FaceDetector) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]], float]:
    faces = detector.detect_faces(img)
    if not faces:
        return None, None, 0.0
    best = max(faces, key=lambda f: f.get("confidence", 0.0))
    H, W = img.shape[:2]
    x, y, w, h = _normalize_bbox(best["bbox"], W, H)
    x, y, w, h = _expand_bbox(x, y, w, h, W, H, 0.12)
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None, None, 0.0
    face224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    return face224, (x, y, w, h), float(best.get("confidence", 0.0))


# --------------------------- probe embeddings --------------------------------
def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return v.astype(np.float32) / n


def _probe_fr_128(face_bgr_224: np.ndarray) -> Optional[np.ndarray]:
    if not _HAS_FR:
        return None
    try:
        rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        locs = [(0, w, h, 0)]
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs, num_jitters=1, model="small")
        if not encs:
            encs = face_recognition.face_encodings(rgb, num_jitters=1, model="small")
        if not encs:
            return None
        return _l2_normalize(np.asarray(encs[0], dtype=np.float32))
    except Exception as e:
        logger.warning(f"probe fr_128 failed: {e}")
        return None


def _probe_dct_256(face_bgr_224: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Use 128x128 DCT for better SNR, then take 16x16 low-freq block
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(g)
    block = dct[:16, :16].reshape(-1)
    block -= block.mean()
    block /= (block.std() + 1e-6)
    return block.astype(np.float32)  # (256,)


# --------------------------- scoring + thresholds ----------------------------
# face_recognition 128D: standard Euclidean distance threshold
FR_DISTANCE_MAX = 0.60  # <= 0.60 matches
# DCT-256 cosine similarity threshold
DCT_COSINE_MIN = 0.82   # >= 0.82 matches

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.ravel().astype(np.float32) - b.ravel().astype(np.float32)))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a) + 1e-12; nb = np.linalg.norm(b) + 1e-12
    return float(np.clip(np.dot(a/na, b/nb), -1.0, 1.0))


def _score_and_pass(model: str, probe: np.ndarray, gallery: np.ndarray) -> Tuple[float, bool, float]:
    """
    Returns (raw_score, passed, normalized_confidence[0..1])

    - fr_128: lower distance is better; pass if dist <= FR_DISTANCE_MAX
              normalized_conf = max(0, 1 - dist / FR_DISTANCE_MAX)
    - dct_256: higher cosine is better; pass if cos >= DCT_COSINE_MIN
               normalized_conf = max(0, (cos - DCT_COSINE_MIN) / (1 - DCT_COSINE_MIN))
    """
    if model == "fr_128":
        dist = _euclidean(probe, gallery)
        passed = dist <= FR_DISTANCE_MAX
        conf = max(0.0, 1.0 - dist / FR_DISTANCE_MAX)
        return dist, passed, conf
    elif model == "dct_256":
        cos = _cosine(probe, gallery)
        passed = cos >= DCT_COSINE_MIN
        conf = max(0.0, (cos - DCT_COSINE_MIN) / max(1e-6, (1.0 - DCT_COSINE_MIN)))
        return cos, passed, conf
    else:
        return 0.0, False, 0.0


# -------------------------- gallery embedding fetch --------------------------
def _load_user_embedding(user: Dict) -> Tuple[Optional[np.ndarray], Optional[str]]:
    rec = db.get_user_by_id(user["id"])
    if not rec:
        return None, None
    feats = (rec.get("face_features") or {})
    emb = feats.get("embedding")
    model = feats.get("embedding_model")

    if isinstance(emb, (list, tuple)) and model in ("fr_128", "dct_256"):
        arr = np.asarray(emb, dtype=np.float32)
        if model == "fr_128":
            arr = _l2_normalize(arr)
        return arr, model

    # Fallback: compute from stored 224x224 crop bytes (face_encoding)
    face_blob = rec.get("face_encoding") or rec.get("face_image")
    if face_blob:
        try:
            arr = np.frombuffer(face_blob, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None, None
            face224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # Prefer fr_128 if available
            if _HAS_FR:
                fr = _probe_fr_128(face224)
                if fr is not None:
                    return fr, "fr_128"
            dct = _probe_dct_256(face224)
            return dct, "dct_256"
        except Exception:
            return None, None

    return None, None


# --------------------------------- Matcher -----------------------------------
class FaceMatcher:
    def __init__(self,
                 fr_distance_max: float = FR_DISTANCE_MAX,
                 dct_cosine_min: float = DCT_COSINE_MIN):
        self.face_detector = FaceDetector()
        self.fr_distance_max = float(fr_distance_max)
        self.dct_cosine_min = float(dct_cosine_min)

    def match_face_to_users(self, image: np.ndarray) -> List[Dict]:
        """Return all users that pass their model threshold, sorted by confidence."""
        face224, bbox, det_conf = _best_face_224(image, self.face_detector)
        if face224 is None:
            return []

        # compute BOTH probe embeddings once
        probe_fr = _probe_fr_128(face224) if _HAS_FR else None
        probe_dct = _probe_dct_256(face224)

        matches: List[Dict] = []
        for u in (db.get_all_users() or []):
            gal_vec, model = _load_user_embedding(u)
            if gal_vec is None or model not in ("fr_128", "dct_256"):
                continue

            if model == "fr_128":
                if probe_fr is None:
                    continue
                raw, passed, conf = _score_and_pass("fr_128", probe_fr, gal_vec)
            else:  # dct_256
                raw, passed, conf = _score_and_pass("dct_256", probe_dct, gal_vec)

            if passed:
                matches.append({
                    "user_id": u["id"],
                    "username": u.get("username"),
                    "full_name": u.get("full_name"),
                    "model": model,
                    "raw_score": float(raw),
                    "confidence": float(conf),
                    "face_bbox": bbox,
                    "detection_confidence": float(det_conf),
                })

        # sort by confidence descending
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        return matches

    def authenticate_user(self, image: np.ndarray, username: Optional[str] = None) -> Dict:
        """Authenticate with hard thresholds; never return a wrong user."""
        faces = self.face_detector.detect_faces(image)
        if not faces:
            return {'success': False, 'error': 'No face detected', 'confidence': 0.0}

        matches = self.match_face_to_users(image)
        if not matches:
            return {'success': False, 'error': 'No matching user cleared threshold', 'confidence': 0.0}

        if username:
            hit = next((m for m in matches if m["username"] == username), None)
            if not hit:
                return {'success': False, 'error': f'Face does not meet threshold for {username}', 'confidence': 0.0}
            top = hit
        else:
            top = matches[0]

        # Log attempt
        try:
            db.log_auth_attempt(
                top["user_id"],
                'face_match',
                True,
                top["confidence"],
                {
                    'model': top["model"],
                    'raw_score': top["raw_score"],
                    'thresholds': {'fr_distance_max': self.fr_distance_max, 'dct_cosine_min': self.dct_cosine_min},
                    'detection_confidence': top["detection_confidence"],
                }
            )
        except Exception:
            pass

        return {
            'success': True,
            'user_id': top["user_id"],
            'username': top["username"],
            'full_name': top.get("full_name"),
            'confidence': top["confidence"],
            'face_info': {
                'bbox': top["face_bbox"],
                'detection_confidence': top["detection_confidence"],
                'model': top["model"],
                'raw_score': top["raw_score"],
            }
        }


# global instance
face_matcher = FaceMatcher()
