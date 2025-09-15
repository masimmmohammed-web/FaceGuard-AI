"""
Face registration system for FaceGuard AI authentication (hardened).
- Robust bbox normalization + optional expansion
- Per-session AntiSpoofingDetector to avoid cross-session history bleed
- Store reduced frames for motion to save memory
- Pass landmarks into anti-spoof for stable EAR (no NaNs)
- ✅ Store an embedding at registration time (fr_128 or dct_256) with z-score normalization for DCT
"""

from flask import Blueprint, request, jsonify, render_template
import cv2
import numpy as np
import base64
import sys
import time
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.eye_tracker import EyeTracker
from src.anti_spoofing import AntiSpoofingDetector
from database import db

import dlib
from imutils import face_utils

# Optional: face_recognition (for 128-D embeddings). If missing, we fall back to DCT.
try:
    import face_recognition  # type: ignore
    _HAS_FR = True
except Exception:
    _HAS_FR = False

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Blueprint & core components
# -----------------------------------------------------------------------------
face_registration = Blueprint('face_registration', __name__)

face_detector = FaceDetector()

# Global (in-process) store for active ~20s validation sessions
_liveness_sessions: Dict[str, dict] = {}

# Limits
_MAX_DECODED_BYTES = 5 * 1024 * 1024  # 5MB decoded payload cap

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _resolve_predictor_path() -> Optional[str]:
    """Find a usable dlib 68 landmarks predictor on disk or via env."""
    import os
    env_path = os.getenv('PREDICTOR_PATH')
    if env_path and Path(env_path).exists():
        return env_path
    candidates = [
        Path(__file__).parent.parent / 'shape_predictor_68_face_landmarks.dat',
        Path.cwd() / 'shape_predictor_68_face_landmarks.dat',
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

_PRED_PATH = _resolve_predictor_path()
try:
    _dlib_predictor = dlib.shape_predictor(_PRED_PATH) if _PRED_PATH else None
except Exception as e:
    logger.error(f"Failed to load dlib shape predictor: {e}")
    _dlib_predictor = None

try:
    eye_tracker = EyeTracker(_PRED_PATH) if _PRED_PATH else None
    if eye_tracker:
        logger.info(f"✅ Using dlib landmarks for face registration: {_PRED_PATH}")
    else:
        logger.warning("⚠️  dlib predictor not found, using basic face features")
except Exception as e:
    logger.error(f"EyeTracker init failed: {e}")
    eye_tracker = None

def _decode_data_url_to_bgr(data_url: str) -> Tuple[bool, Optional[np.ndarray], str]:
    """Strict, chatty decoder with actionable error messages + size guard."""
    if not data_url or ',' not in data_url:
        return False, None, 'Invalid frame data format'
    try:
        _, b64 = data_url.split(',', 1)
    except ValueError:
        return False, None, 'Malformed frame data'

    # Size guard (estimate decoded size from base64 length ~ 3/4)
    est_decoded = (len(b64) * 3) // 4
    if est_decoded > _MAX_DECODED_BYTES:
        return False, None, f'Image too large (>{_MAX_DECODED_BYTES // (1024*1024)}MB)'

    try:
        raw = base64.b64decode(b64)
        if not raw:
            return False, None, 'Empty decoded image data'
    except Exception:
        return False, None, 'Failed to decode base64 image data'
    try:
        nparr = np.frombuffer(raw, np.uint8)
        if nparr.size == 0:
            return False, None, 'Empty image buffer'
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, None, 'OpenCV failed to decode image'
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False, None, 'Invalid image dimensions'
    except Exception:
        return False, None, 'Failed to process image with OpenCV'
    return True, img, ''

def _normalize_bbox(bbox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Accept [x,y,w,h] or [x1,y1,x2,y2], in either pixels or normalized [0..1].
    Return clamped integer [x,y,w,h] in pixels.
    """
    if not bbox or len(bbox) != 4:
        return 0, 0, img_w, img_h

    x0, y0, a, b = map(float, bbox)

    def _is_norm(vals):  # allow tiny >1 noise
        return all(0.0 <= v <= 1.5 for v in vals)

    # Treat as [x,y,w,h] if (a,b) look like sizes (both <=1.5 or both >1.5)
    if ((a <= 1.5 and b <= 1.5 and _is_norm([x0, y0, a, b])) or (a > 1.5 and b > 1.5)):
        if _is_norm([x0, y0, a, b]):
            x = x0 * img_w
            y = y0 * img_h
            w = a * img_w
            h = b * img_h
        else:
            x, y, w, h = x0, y0, a, b
    else:
        # Otherwise treat as corners [x1,y1,x2,y2]
        if _is_norm([x0, y0, a, b]):
            x1, y1, x2, y2 = x0 * img_w, y0 * img_h, a * img_w, b * img_h
        else:
            x1, y1, x2, y2 = x0, y0, a, b
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

    # Clamp + ensure min size
    x = int(max(0, min(x, img_w - 1)))
    y = int(max(0, min(y, img_h - 1)))
    w = int(max(2, min(w, img_w - x)))
    h = int(max(2, min(h, img_h - y)))
    return x, y, w, h

def _expand_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int, margin: float = 0.15) -> Tuple[int, int, int, int]:
    """Expand bbox by margin on each side, clamped to image."""
    dx = int(w * margin)
    dy = int(h * margin)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)
    return x1, y1, max(2, x2 - x1), max(2, y2 - y1)

def _reduce_frame_for_motion(frame: np.ndarray) -> np.ndarray:
    """Downsampled grayscale for cheap motion checks; keeps memory tiny."""
    try:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        return small
    except Exception:
        return np.empty((0, 0), dtype=np.uint8)

def _cleanup_sessions(ttl_seconds: float = 30.0) -> None:
    """Drop inactive or old sessions so memory doesn't grow without bound."""
    now = time.time()
    for k, v in list(_liveness_sessions.items()):
        if (not v.get('is_active')) or (now - v.get('start_time', now) > ttl_seconds):
            _liveness_sessions.pop(k, None)

# --- Embedding helpers -------------------------------------------------------
def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(vec) + eps)
    return vec.astype(np.float32) / n

def _compute_dct_embedding(face_bgr: np.ndarray) -> Tuple[List[float], str]:
    """
    Fallback descriptor: grayscale->128x128->2D DCT->top-left 16x16 (256-D)
    ► z-score normalize (mean/std) to match the tester, then use cosine in matcher.
    """
    if face_bgr is None or face_bgr.size == 0:
        return [], "none"
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    f32 = np.float32(gray) / 255.0
    dct = cv2.dct(f32)
    block = dct[:16, :16].reshape(-1).astype(np.float32)
    block -= float(block.mean())
    block /= float(block.std() + 1e-6)  # <-- z-score (not L2) to align with tester
    return block.tolist(), "dct_256"

def _compute_fr_embedding(face_bgr: np.ndarray) -> Tuple[List[float], str]:
    """
    Preferred descriptor using face_recognition (128-D).
    Passes the whole crop as the known location; falls back to auto-detect.
    """
    if not _HAS_FR or face_bgr is None or face_bgr.size == 0:
        return [], "none"
    try:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        locs = [(0, w, h, 0)]  # top, right, bottom, left (whole crop)
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs, num_jitters=1, model="small")
        if not encs:
            encs = face_recognition.face_encodings(rgb, num_jitters=1, model="small")
        if encs:
            vec = _l2_normalize(np.asarray(encs[0], dtype=np.float32))
            return vec.tolist(), "fr_128"
        return [], "none"
    except Exception as e:
        logger.warning(f"face_recognition embedding failed: {e}")
        return [], "none"

def compute_face_embedding(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[List[float], str]:
    """
    Returns (embedding_list, model_name) — prefers fr_128, falls back to dct_256.
    """
    try:
        H, W = image.shape[:2]
        x, y, w, h = _normalize_bbox(bbox, W, H)
        # Small expansion tends to help downstream matching
        x, y, w, h = _expand_bbox(x, y, w, h, W, H, margin=0.10)
        crop = image[y:y+h, x:x+w]

        emb, model = _compute_fr_embedding(crop)
        if emb:
            return emb, model

        # Fallback
        return _compute_dct_embedding(crop)
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return [], "none"

# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------
def calculate_eye_aspect_ratio(eye_points: np.ndarray) -> float:
    try:
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3]) + 1e-6
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

def calculate_symmetry(left_points: np.ndarray, right_points: np.ndarray) -> float:
    try:
        if len(left_points) != len(right_points):
            return 0.0
        center_x = (np.mean(left_points[:, 0]) + np.mean(right_points[:, 0])) / 2.0
        left_distances = np.abs(left_points[:, 0] - center_x)
        right_distances = np.abs(right_points[:, 0] - center_x)
        return float(np.mean(np.abs(left_distances - right_distances)))
    except Exception:
        return 0.0

def extract_basic_features(gray_face: np.ndarray) -> Dict:
    """Extract basic features when dlib landmarks are not available."""
    try:
        features: Dict[str, float] = {}
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        features['histogram_mean'] = float(np.mean(hist))
        features['histogram_std'] = float(np.std(hist))
        features['texture_variance'] = float(np.var(gray_face))
        features['texture_mean'] = float(np.mean(gray_face))
        edges = cv2.Canny(gray_face, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        return features
    except Exception as e:
        logger.error(f"Basic feature extraction failed: {e}")
        return {}

def extract_landmark_features(points_img_xy: np.ndarray, face_xywh: Tuple[int, int, int, int]) -> Dict:
    """
    Extract features from facial landmarks (points in image coords).
    Normalize positions relative to face box for robustness.
    """
    try:
        x, y, w, h = face_xywh
        pts = points_img_xy.astype(np.float32)
        # Normalize into face-local coords
        pts_local = pts.copy()
        pts_local[:, 0] -= x
        pts_local[:, 1] -= y

        features: Dict[str, float] = {}

        features['landmark_count'] = float(len(pts_local))
        features['landmark_center_x'] = float(np.mean(pts_local[:, 0]) / max(1.0, w))
        features['landmark_center_y'] = float(np.mean(pts_local[:, 1]) / max(1.0, h))
        features['landmark_std_x'] = float(np.std(pts_local[:, 0]) / max(1.0, w))
        features['landmark_std_y'] = float(np.std(pts_local[:, 1]) / max(1.0, h))

        # Jawline 0-16
        jawline = pts_local[0:17]
        if len(jawline) >= 2:
            features['jawline_length_norm'] = float(
                np.sum(np.linalg.norm(np.diff(jawline, axis=0), axis=1)) / max(1.0, (w + h) * 0.5)
            )
        else:
            features['jawline_length_norm'] = 0.0

        # Eyes
        if len(pts_local) >= 48:
            left_eye = pts_local[36:42]
            right_eye = pts_local[42:48]
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            features['eye_distance_norm'] = float(eye_distance / max(1.0, (w + h) * 0.5))

            left_eye_ar = calculate_eye_aspect_ratio(left_eye)
            right_eye_ar = calculate_eye_aspect_ratio(right_eye)
            features['left_eye_ar'] = float(left_eye_ar)
            features['right_eye_ar'] = float(right_eye_ar)
            features['avg_eye_ar'] = float((left_eye_ar + right_eye_ar) * 0.5)
        else:
            features['eye_distance_norm'] = 0.0
            features['left_eye_ar'] = 0.0
            features['right_eye_ar'] = 0.0
            features['avg_eye_ar'] = 0.0

        # Nose 27-35
        if len(pts_local) >= 36:
            nose = pts_local[27:36]
            nose_center = np.mean(nose, axis=0)
            features['nose_center_x'] = float(nose_center[0] / max(1.0, w))
            features['nose_center_y'] = float(nose_center[1] / max(1.0, h))
        else:
            features['nose_center_x'] = 0.0
            features['nose_center_y'] = 0.0

        # Mouth 48-67
        if len(pts_local) >= 68:
            mouth = pts_local[48:68]
            mouth_center = np.mean(mouth, axis=0)
            features['mouth_center_x'] = float(mouth_center[0] / max(1.0, w))
            features['mouth_center_y'] = float(mouth_center[1] / max(1.0, h))
        else:
            features['mouth_center_x'] = 0.0
            features['mouth_center_y'] = 0.0

        # Face symmetry (use raw image coords but compute relative difference)
        if len(pts) >= 17:
            left_face = pts[0:9]
            right_face = pts[8:17]
            features['face_symmetry'] = float(calculate_symmetry(left_face, right_face))
        else:
            features['face_symmetry'] = 0.0

        # Relative position features
        diag = np.linalg.norm(np.array([w, h], dtype=np.float32))
        if diag > 0:
            features['eye_to_nose_ratio'] = float(features['eye_distance_norm'])  # already normalized
            if len(pts_local) >= 68:
                nose_center = np.mean(pts_local[27:36], axis=0)
                mouth_center = np.mean(pts_local[48:68], axis=0)
                features['nose_to_mouth_ratio'] = float(np.linalg.norm(nose_center - mouth_center) / diag)
            else:
                features['nose_to_mouth_ratio'] = 0.0
        else:
            features['eye_to_nose_ratio'] = 0.0
            features['nose_to_mouth_ratio'] = 0.0

        return features
    except Exception as e:
        logger.error(f"Landmark feature extraction failed: {e}")
        return {}

def extract_face_features(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
    """Extract face features using dlib landmarks for registration (if available)."""
    try:
        H, W = image.shape[:2]
        x, y, w, h = _normalize_bbox(bbox, W, H)
        x, y, w, h = _expand_bbox(x, y, w, h, W, H, margin=0.10)

        face_region = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (224, 224))
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # predictor expects full-image coords

        # Basic features
        features: Dict = {
            'face_size': (int(w), int(h)),
            'face_aspect_ratio': float(w / max(1, h)),  # renamed for clarity
            'face_area': int(w * h),
            'image_shape': tuple(image.shape[:2]),
            'face_center': (int(x + w // 2), int(y + h // 2)),
            'face_region_shape': tuple(face_region.shape[:2]),
        }

        # Landmark-based features if predictor is available
        if _dlib_predictor is not None:
            try:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shp = _dlib_predictor(gray_full, rect)
                pts = face_utils.shape_to_np(shp)  # (68,2) in image coords
                landmark_features = extract_landmark_features(pts, (x, y, w, h))
                features.update(landmark_features)
                logger.info("✅ Extracted landmark features for registration")
            except Exception as e:
                logger.warning(f"dlib landmark extraction failed: {e}")
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                features.update(extract_basic_features(gray_face))
        else:
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            features.update(extract_basic_features(gray_face))

        return features
    except Exception as e:
        logger.error(f"Face feature extraction failed: {e}")
        return {}

def encode_face_data(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bytes:
    """Encode face patch for storage (JPEG, 224x224)."""
    try:
        H, W = image.shape[:2]
        x, y, w, h = _normalize_bbox(bbox, W, H)
        x, y, w, h = _expand_bbox(x, y, w, h, W, H, margin=0.10)
        face_region = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (224, 224))
        ok, encoded = cv2.imencode('.jpg', face_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return encoded.tobytes() if ok else b''
    except Exception as e:
        logger.error(f"Face encoding failed: {e}")
        return b''

def analyze_liveness_session(session: dict) -> dict:
    """Analyze the complete liveness validation session (20s window)."""
    try:
        results = session.get('validation_results', [])
        frames = session.get('frames', [])
        anti: Optional[AntiSpoofingDetector] = session.get('anti')

        if not results:
            return {'valid': False, 'error': 'No validation data collected', 'confidence': 0.0}

        total_frames = len(results)
        authentic_frames = sum(1 for r in results if r['spoofing_result'].get('is_authentic', False))
        live_frames = sum(1 for r in results if r['spoofing_result'].get('liveness', {}).get('is_live', False))
        avg_face_confidence = sum(r.get('face_confidence', 0.0) for r in results) / max(1, total_frames)

        # Simple inter-frame motion via absdiff mean on reduced frames
        movement_detected = 0
        if len(frames) > 1:
            for i in range(1, len(frames)):
                if frames[i-1].size and frames[i-1].shape == frames[i].shape:
                    diff = cv2.absdiff(frames[i-1], frames[i])
                    mean_diff = float(np.mean(diff))
                    if mean_diff > 2.0:  # sensitive small movement threshold
                        movement_detected += 1
        movement_ratio = movement_detected / max(1, len(frames) - 1)

        authenticity_score = authentic_frames / max(1, total_frames)
        liveness_score = live_frames / max(1, total_frames)
        movement_score = movement_ratio

        final_score = (authenticity_score * 0.4 + liveness_score * 0.4 + movement_score * 0.2)

        is_valid = (
            final_score > 0.5 and
            authenticity_score > 0.6 and
            liveness_score > 0.4 and
            movement_score > 0.2 and
            avg_face_confidence > 0.6
        )

        risk_factors: List[str] = []
        if authenticity_score < 0.6:
            risk_factors.append(f"Low authenticity: {authenticity_score:.2f}")
        if liveness_score < 0.4:
            risk_factors.append(f"Low liveness: {liveness_score:.2f}")
        if movement_score < 0.2:
            risk_factors.append(f"Low movement: {movement_score:.2f}")
        if avg_face_confidence < 0.6:
            risk_factors.append(f"Low face confidence: {avg_face_confidence:.2f}")

        # Clean anti history for safety (this anti is per-session, but be tidy)
        try:
            if anti:
                anti.clear_history()
        except Exception:
            pass

        return {
            'valid': is_valid,
            'confidence': final_score,
            'statistics': {
                'total_frames': total_frames,
                'authentic_frames': authentic_frames,
                'live_frames': live_frames,
                'movement_frames': movement_detected,
                'authenticity_score': authenticity_score,
                'liveness_score': liveness_score,
                'movement_score': movement_score,
                'avg_face_confidence': avg_face_confidence
            },
            'risk_factors': risk_factors,
            'message': 'Liveness validation completed successfully' if is_valid else 'Liveness validation failed'
        }
    except Exception as e:
        logger.error(f"Error analyzing liveness session: {str(e)}")
        return {'valid': False, 'error': 'Failed to analyze liveness session', 'confidence': 0.0}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@face_registration.route('/register')
def registration_page():
    """Face registration page."""
    return render_template('face_registration.html')

@face_registration.route('/test_validation', methods=['GET'])
def test_validation():
    """Test endpoint to verify the validation route is working."""
    return jsonify({'valid': True, 'message': 'Validation endpoint is working'})

@face_registration.route('/start_liveness_validation', methods=['POST'])
def start_liveness_validation():
    """Start ~20-second continuous liveness validation process."""
    try:
        _cleanup_sessions()  # keep memory tidy

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        frame_data = data.get('frame_data', '')
        if not frame_data:
            return jsonify({'error': 'Face image is required'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok:
            return jsonify({'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({'error': 'No face detected. Please ensure your face is clearly visible.'}), 400

        best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)

        face_area = bw * bh
        image_area = H * W
        face_fill_ratio = face_area / max(1, image_area)

        if face_fill_ratio < 0.01:
            return jsonify({'error': 'Face is too small. Please move closer to the camera.'}), 400
        if face_fill_ratio > 0.5:
            return jsonify({'error': 'Face is too large. Please move away from the camera.'}), 400

        # Per-session anti-spoofing instance so history is isolated
        anti = AntiSpoofingDetector()
        anti.clear_history()

        session_id = f"liveness_{int(time.time()*1000)}"
        _liveness_sessions[session_id] = {
            'frames': [ _reduce_frame_for_motion(image) ],  # reduced grayscale frames
            'bbox': (bx, by, bw, bh),
            'start_time': time.time(),
            'validation_results': [],
            'is_active': True,
            'anti': anti
        }
        logger.info(f"Started liveness validation session: {session_id}")

        return jsonify({
            'session_id': session_id,
            'message': 'Liveness validation started. Please maintain your position and follow the instructions.',
            'duration': 20,
            'face_info': {
                'confidence': best_face.get('confidence', 0.0),
                'bbox': [bx, by, bw, bh],
                'face_ratio': face_fill_ratio,        # kept for backward-compat
                'face_fill_ratio': face_fill_ratio     # explicit name
            }
        })
    except Exception as e:
        logger.error(f"Error starting liveness validation: {str(e)}")
        return jsonify({'error': 'Failed to start liveness validation. Please try again.'}), 500

@face_registration.route('/validate_liveness_frame', methods=['POST'])
def validate_liveness_frame():
    """Validate a single frame during the ~20s liveness validation."""
    try:
        _cleanup_sessions()  # keep memory tidy

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        session_id = data.get('session_id', '')
        frame_data = data.get('frame_data', '')
        if not session_id or not frame_data:
            return jsonify({'error': 'Session ID and frame data are required'}), 400

        session = _liveness_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Invalid session ID'}), 400
        if not session.get('is_active', False):
            return jsonify({'error': 'Session expired'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok:
            return jsonify({'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({'error': 'No face detected in this frame'}), 200  # soft error

        best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
        bbox = (bx, by, bw, bh)

        # Landmarks (better blink cue in anti-spoofing)
        landmarks: List[Tuple[int, int]] = []
        if _dlib_predictor is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(int(bx), int(by), int(bx + bw), int(by + bh))
                shp = _dlib_predictor(gray, rect)
                landmarks = [(p.x, p.y) for p in shp.parts()]
            except Exception as e:
                logger.warning(f"Could not extract landmarks: {e}")

        anti: AntiSpoofingDetector = session['anti']
        spoofing_result = anti.comprehensive_analysis(image, bbox, landmarks)

        # Append reduced frame for motion stats
        session['frames'].append(_reduce_frame_for_motion(image))
        session['validation_results'].append({
            'timestamp': time.time(),
            'spoofing_result': spoofing_result,
            'face_confidence': best_face.get('confidence', 0.0),
            'bbox': bbox
        })

        # Cap stored frames to ~150 to limit memory (25fps * 6s worst-case bursts)
        if len(session['frames']) > 150:
            session['frames'] = session['frames'][-120:]
            session['validation_results'] = session['validation_results'][-120:]

        elapsed_time = time.time() - session['start_time']
        remaining_time = max(0.0, 20.0 - elapsed_time)

        if remaining_time <= 0:
            session['is_active'] = False
            final_result = analyze_liveness_session(session)
            return jsonify({
                'validation_complete': True,
                'result': final_result,
                'total_frames': len(session['frames']),
                'elapsed_time': elapsed_time
            })
        else:
            return jsonify({
                'validation_complete': False,
                'remaining_time': remaining_time,
                'frame_count': len(session['frames']),
                'current_result': {
                    'is_authentic': bool(spoofing_result.get('is_authentic', False)),
                    'is_live': bool(spoofing_result.get('liveness', {}).get('is_live', False)),
                    'risk_level': spoofing_result.get('risk_level', 'UNKNOWN')
                }
            })
    except Exception as e:
        logger.error(f"Error validating liveness frame: {str(e)}")
        return jsonify({'error': 'Failed to validate frame. Please try again.'}), 500

@face_registration.route('/validate_face_capture', methods=['POST'])
def validate_face_capture():
    """Validate a single still capture for anti-spoofing before allowing registration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'valid': False, 'error': 'No data provided'}), 400

        frame_data = data.get('frame_data', '')
        if not frame_data:
            return jsonify({'valid': False, 'error': 'Face image is required'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok:
            return jsonify({'valid': False, 'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({'valid': False, 'error': 'No face detected. Please ensure your face is clearly visible.'}), 400

        best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
        bbox = (bx, by, bw, bh)

        face_fill_ratio = (bw * bh) / float(max(1, H * W))
        if face_fill_ratio < 0.01:
            return jsonify({'valid': False, 'error': 'Face is too small. Please move closer to the camera.'}), 400
        if face_fill_ratio > 0.5:
            return jsonify({'valid': False, 'error': 'Face is too large. Please move away from the camera.'}), 400

        # Landmarks (optional but recommended)
        landmarks: List[Tuple[int, int]] = []
        if _dlib_predictor is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(int(bx), int(by), int(bx + bw), int(by + bh))
                shp = _dlib_predictor(gray, rect)
                landmarks = [(p.x, p.y) for p in shp.parts()]
            except Exception as e:
                logger.warning(f"Could not extract landmarks for anti-spoofing: {e}")

        # Use a fresh anti instance (single-shot)
        anti = AntiSpoofingDetector()
        anti.clear_history()
        spoofing_result = anti.comprehensive_analysis(image, bbox, landmarks)
        logger.info(f"Anti-spoofing results (capture): {spoofing_result}")

        overall_risk = float(spoofing_result.get('overall_risk', 0.0))
        if overall_risk > 0.95:
            risk_details = ', '.join([f"{name}: {conf:.2f}" for name, conf in spoofing_result.get('risk_factors', [])])
            logger.warning(f"Extremely high spoofing risk detected during face capture: {risk_details} (Risk: {overall_risk:.2f})")
            return jsonify({
                'valid': False,
                'error': 'Spoofing attack detected! Please use a real face, not a photo or video.',
                'details': f"Detected risks: {risk_details}",
                'spoofing_confidence': overall_risk
            }), 400

        liveness_score = float(spoofing_result.get('liveness', {}).get('confidence', 0.0))
        if liveness_score < 0.05:
            logger.warning(f"Extremely low liveness detected during face capture: {liveness_score:.2f}")
            return jsonify({
                'valid': False,
                'error': 'Please ensure you are a real person. Try blinking or moving slightly.',
                'details': 'The system detected extremely low liveness. Please try again with natural movements.'
            }), 400

        logger.info("Anti-spoofing validation passed for face capture")
        return jsonify({
            'valid': True,
            'message': 'Face validation successful',
            'face_info': {
                'confidence': best_face.get('confidence', 0.0),
                'bbox': [bx, by, bw, bh],
                'face_ratio': face_fill_ratio,        # kept for backward-compat
                'face_fill_ratio': face_fill_ratio
            }
        })
    except Exception as e:
        logger.error(f"Error in face capture validation: {str(e)}")
        return jsonify({'valid': False, 'error': 'Face validation failed. Please try again.'}), 500

@face_registration.route('/register_face', methods=['POST'])
def register_face():
    """Register a new user with face data (now also stores an embedding)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        frame_data = data.get('frame_data', '')

        if not username:
            return jsonify({'error': 'Username is required'}), 400
        if not frame_data:
            return jsonify({'error': 'Face image is required'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok:
            return jsonify({'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({'error': 'No face detected. Please ensure your face is clearly visible.'}), 400

        best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
        bbox = (bx, by, bw, bh)

        face_fill_ratio = (bw * bh) / float(max(1, H * W))
        if face_fill_ratio < 0.01:
            return jsonify({'error': 'Face is too small. Please move closer to the camera.'}), 400

        # Anti-spoofing (block obvious attacks only)
        landmarks: List[Tuple[int, int]] = []
        if _dlib_predictor is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(int(bx), int(by), int(bx + bw), int(by + bh))
                shp = _dlib_predictor(gray, rect)
                landmarks = [(p.x, p.y) for p in shp.parts()]
            except Exception as e:
                logger.warning(f"Could not extract landmarks for anti-spoofing: {e}")

        anti = AntiSpoofingDetector()
        anti.clear_history()
        spoofing_result = anti.comprehensive_analysis(image, bbox, landmarks)

        if not bool(spoofing_result.get('is_authentic', True)):
            risk_details = ', '.join([f"{name}: {conf:.2f}" for name, conf in spoofing_result.get('risk_factors', [])])
            overall_risk = float(spoofing_result.get('overall_risk', 0.0))
            logger.warning(f"Spoofing attack detected for user {username}: {risk_details} (Risk: {overall_risk:.2f})")
            return jsonify({
                'error': 'Spoofing attack detected! Please use a real face, not a photo or video.',
                'details': f"Detected risks: {risk_details}",
                'spoofing_confidence': overall_risk
            }), 400

        if not bool(spoofing_result.get('liveness', {}).get('is_live', False)):
            logger.warning(f"No liveness detected for user {username}")
            return jsonify({
                'error': 'No liveness detected! Please ensure you are a real person and try blinking or moving slightly.',
                'details': 'The system requires a live person to register. Photos, videos, or static images are not allowed.'
            }), 400

        if face_fill_ratio > 0.5:
            return jsonify({'error': 'Face is too large. Please move away from the camera.'}), 400

        # === Features + embedding + encoded face ===
        face_features = extract_face_features(image, bbox)
        embedding, emb_model = compute_face_embedding(image, bbox)
        face_features['embedding'] = embedding  # <- stored in DB for matching
        face_features['embedding_model'] = emb_model

        face_encoding = encode_face_data(image, bbox)
        if not face_features or not face_encoding or not embedding:
            return jsonify({'error': 'Failed to extract face features/embedding'}), 500

        try:
            user_id = db.register_user(username, email, full_name, face_encoding, face_features)
            db.log_auth_attempt(
                user_id, 'registration', True, best_face.get('confidence', 0.0),
                {
                    'face_size': (int(bw), int(bh)),
                    'face_ratio': face_fill_ratio,  # kept for backward-compat
                    'face_fill_ratio': face_fill_ratio,
                    'confidence': best_face.get('confidence', 0.0),
                    'embedding_model': emb_model
                }
            )
            return jsonify({
                'success': True,
                'message': f'User {username} registered successfully!',
                'user_id': user_id,
                'face_info': {
                    'bbox': [bx, by, bw, bh],
                    'confidence': best_face.get('confidence', 0.0),
                    'face_ratio': face_fill_ratio,        # kept for backward-compat
                    'face_fill_ratio': face_fill_ratio
                }
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Database registration failed: {e}")
            return jsonify({'error': 'Registration failed. Please try again.'}), 500
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@face_registration.route('/users')
def list_users():
    """List all registered users (admin function)."""
    try:
        users = db.get_all_users()
        return jsonify({'success': True, 'users': users, 'count': len(users)})
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return jsonify({'error': 'Failed to retrieve users'}), 500

@face_registration.route('/user/<username>')
def get_user(username):
    """Get user information by username."""
    try:
        user = db.get_user_by_username(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        user_data = {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'full_name': user['full_name'],
            'registration_date': user['registration_date'],
            'last_login': user['last_login']
        }
        return jsonify({'success': True, 'user': user_data})
    except Exception as e:
        logger.error(f"Failed to get user {username}: {e}")
        return jsonify({'error': 'Failed to retrieve user'}), 500
