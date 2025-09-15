# web_app/authentication.py
"""
Complete authentication system integrating face matching and blink detection
with robust bbox handling, outlier guards, a live blink endpoint, and an
overlay-friendly /analyze_frame route (with anti-spoofing metrics).

This version includes:
- Dynamic mEAR blink detection with EMA smoothing
- Self-healing 'eyes-open' baseline after calibration
- IOD-aware threshold floor + global cap (default floor aligned at 0.28 when IOD unknown)
- Ratio-gated closure detection for soft blinks
- Clean, consistent logging
- Updated integration with AntiSpoofingDetector (supports `ear_threshold`/`ear_thresh` + `tag`)
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify, render_template, session
import cv2
import numpy as np
import base64
import os
import sys
import time
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict

import dlib

# Optional import for predictor path reuse
try:
    from web_app.live_analyzer import PREDICTOR_PATH  # type: ignore
except Exception:
    PREDICTOR_PATH = None  # fallback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.eye_tracker import EyeTracker
from database import db
from face_matching import face_matcher

# Optional anti-spoofing (safe if missing)
try:
    from src.anti_spoofing import AntiSpoofingDetector  # type: ignore
except Exception:
    AntiSpoofingDetector = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint & components
# ---------------------------------------------------------------------------
authentication = Blueprint('authentication', __name__)
face_detector = FaceDetector()
anti = AntiSpoofingDetector(history=30) if AntiSpoofingDetector else None

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    coords = np.zeros((shape.num_parts, 2), dtype=np.int32)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def _resolve_predictor_path() -> Optional[str]:
    env_path = os.getenv('PREDICTOR_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    if PREDICTOR_PATH and os.path.exists(str(PREDICTOR_PATH)):
        return str(PREDICTOR_PATH)
    try:
        root = Path(__file__).parent.parent
        candidates = [
            root / 'shape_predictor_68_face_landmarks.dat',
            Path.cwd() / 'shape_predictor_68_face_landmarks.dat',
        ]
        for p in candidates:
            if p.exists():
                return str(p)
    except Exception:
        pass
    return None

_PRED_PATH = _resolve_predictor_path()
try:
    _ear_predictor = dlib.shape_predictor(_PRED_PATH) if _PRED_PATH else None
except Exception as _e:
    logger.error(f"Failed to load dlib shape predictor: {_e}")
    _ear_predictor = None

# Eye tracker (optional)
try:
    eye_tracker = EyeTracker(predictor_path=_PRED_PATH)
except Exception as _e:
    logger.error(f"EyeTracker init failed: {_e}")
    eye_tracker = None

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
auth_sessions: Dict[str, "AuthenticationSession"] = {}  # session_id -> AuthenticationSession

class AuthenticationSession:
    """Manages authentication session state."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.face_matched = False
        self.blink_detected = False
        self.blink_count: int = 0
        self.required_blinks: int = 3
        self.user_id = None
        self.username = None
        self.attempts = 0
        self.max_attempts = 3
        self.timeout = 300  # 5 minutes

        # Blink state (EMA + dynamic threshold + time-based refractory)
        self.ema_ear: Optional[float] = None
        self.ema_mear: Optional[float] = None

        # Dynamic calibration
        self.calibrate_seconds = 2.0  # increased for better open-eye sampling
        self.calib_until = self.created_at + self.calibrate_seconds
        self.open_baseline: Optional[float] = None
        self.dyn_ratio = 0.85  # dynamic threshold = baseline * ratio

        # Time-based gating
        self.closed_since: Optional[float] = None
        self.last_blink_at: float = 0.0
        self.closed_min_ms = 110
        self.refractory_ms = 450

        self.prev_closed: bool = False

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout

    def is_complete(self) -> bool:
        return self.face_matched and self.blink_detected

    def can_attempt(self) -> bool:
        return self.attempts < self.max_attempts and not self.is_expired()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_bbox(bbox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Accept [x,y,w,h] or [x1,y1,x2,y2], in either pixels or normalized [0..1].
    Return clamped integer [x,y,w,h] in pixels.
    """
    if not bbox or len(bbox) != 4:
        return 0, 0, img_w, img_h

    x0, y0, a, b = map(float, bbox)
    def is_norm(*vals): return all(0.0 <= v <= 1.5 for v in vals)

    # Case A: [x,y,w,h]
    if ((a <= 1.5 and b <= 1.5 and is_norm(x0, y0, a, b))
        or (a > 1.5 and b > 1.5)):
        if is_norm(x0, y0, a, b):  # normalized
            x = x0 * img_w; y = y0 * img_h; w = a * img_w; h = b * img_h
        else:  # pixels
            x, y, w, h = x0, y0, a, b
    # Case B: [x1,y1,x2,y2]
    else:
        if is_norm(x0, y0, a, b):  # normalized corners
            x1, y1, x2, y2 = x0 * img_w, y0 * img_h, a * img_w, b * img_h
        else:  # pixel corners
            x1, y1, x2, y2 = x0, y0, a, b
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

    x = int(max(0, min(x, img_w - 1)))
    y = int(max(0, min(y, img_h - 1)))
    w = int(max(2, min(w, img_w - x)))
    h = int(max(2, min(h, img_h - y)))
    return x, y, w, h

def _expand_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int, margin: float = 0.15):
    dx = int(w * margin); dy = int(h * margin)
    x1 = max(0, x - dx); y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx); y2 = min(img_h, y + h + dy)
    return x1, y1, max(2, x2 - x1), max(2, y2 - y1)

def _rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (points - center) @ R.T + center

def _ear_from_shape(pts: np.ndarray) -> float:
    left = pts[36:42]; right = pts[42:48]
    def ear(eye):
        y1 = np.linalg.norm(eye[1] - eye[5])
        y2 = np.linalg.norm(eye[2] - eye[4])
        x1 = np.linalg.norm(eye[0] - eye[3])
        return (y1 + y2) / (2.0 * x1 + 1e-6)
    return (ear(left) + ear(right)) / 2.0

def _mear_from_shape(pts: np.ndarray) -> float:
    left = pts[36:42]; right = pts[42:48]
    def mear(eye):
        p0, p3 = eye[0], eye[3]
        angle = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        center = eye.mean(axis=0)
        rot = _rotate_points(eye, -angle, center)
        y1 = np.linalg.norm(rot[1] - rot[5])
        y2 = np.linalg.norm(rot[2] - rot[4])
        x1 = np.linalg.norm(rot[0] - rot[3])
        return (y1 + y2) / (2.0 * x1 + 1e-6)
    return (mear(left) + mear(right)) / 2.0

def _decode_data_url_to_bgr(data_url: str) -> Tuple[bool, Optional[np.ndarray], str]:
    if not data_url or ',' not in data_url:
        return False, None, 'Invalid frame data format'
    try:
        _, b64 = data_url.split(',', 1)
    except ValueError:
        return False, None, 'Malformed frame data'
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

def _bbox_from_landmarks(landmarks: List[List[int]] | List[Tuple[int, int]], W: int, H: int, pad: int = 18) -> Tuple[int, int, int, int]:
    pts = np.asarray(landmarks, dtype=np.int32)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

def _extract_pupil_value(pupil):
    if isinstance(pupil, dict):
        return pupil.get('center') or pupil.get('xy') or None
    if isinstance(pupil, (list, tuple)) and len(pupil) == 2:
        return tuple(pupil)
    return None

# ---------------------------------------------------------------------------
# Blink via dynamic mEAR (EMA + calibration + time-based refractory)
# ---------------------------------------------------------------------------
BLINK_THRESH_EAR = 0.25
BLINK_THRESH_MEAR = 0.25
EMA_ALPHA = 0.4
BASELINE_UP_ALPHA = 0.08       # slow, one-way (up) adaptation after calibration
DYN_RATIO = 0.83               # threshold = baseline * ratio
ABS_THR_MAX = 0.34             # cap absolute threshold

def _iod_min_threshold(iod_px: float | None) -> float:
    """
    Small faces (low IOD) need a higher minimum threshold.
    Map IOD ~ 30..70 px -> thr_min ~ 0.25..0.29 (clamped).
    When IOD unknown, align to 0.28 (same as anti_spoofing default).
    """
    if iod_px is None or not np.isfinite(iod_px):
        return 0.28
    thr = 0.23 + 0.002 * max(0.0, iod_px - 30.0)
    return float(np.clip(thr, 0.23, 0.32))

def _blink_via_mear(frame_bgr: np.ndarray, bbox_xywh, auth_session: AuthenticationSession) -> dict:
    if _ear_predictor is None:
        return {'blink_detected': False, 'available': False, 'method': 'mear'}

    H, W = frame_bgr.shape[:2]
    try:
        x, y, w, h = _normalize_bbox(bbox_xywh, W, H)
    except Exception:
        x, y, w, h = bbox_xywh

    x, y, w, h = _expand_bbox(x, y, w, h, W, H, margin=0.15)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    try:
        shp = _ear_predictor(gray, rect)
        pts = _shape_to_np(shp)

        # Inter-ocular sanity
        try:
            iod = float(np.linalg.norm(pts[42] - pts[39]))
            if iod < 20:
                logger.warning(f"[BLINK] IOD too small ({iod:.1f}px); skipping.")
                return {'blink_detected': False, 'available': False, 'method': 'mear', 'reason': 'face_too_small'}
        except Exception:
            iod = float('nan')

        ear_raw  = float(_ear_from_shape(pts))
        mear_raw = float(_mear_from_shape(pts))

        invalid = (
            not np.isfinite(ear_raw) or not np.isfinite(mear_raw) or
            ear_raw <= 0.0 or mear_raw <= 0.0 or
            ear_raw > 1.0 or mear_raw > 1.0
        )
        if not invalid and (auth_session.ema_mear is not None):
            if abs(mear_raw - auth_session.ema_mear) > 0.5:
                invalid = True
        if invalid:
            logger.warning(f"[BLINK] invalid EAR/mEAR (ear={ear_raw:.3f}, mear={mear_raw:.3f}); skipping.")
            return {'blink_detected': False, 'available': False, 'method': 'mear', 'reason': 'invalid_ear_mear'}

        # EMA smoothing
        auth_session.ema_ear  = ear_raw  if auth_session.ema_ear  is None else (EMA_ALPHA * ear_raw  + (1-EMA_ALPHA) * auth_session.ema_ear)
        auth_session.ema_mear = mear_raw if auth_session.ema_mear is None else (EMA_ALPHA * mear_raw + (1-EMA_ALPHA) * auth_session.ema_mear)

        # Calibration + adaptive baseline
        now = time.time()
        cur_for_calib = auth_session.ema_mear if auth_session.ema_mear is not None else mear_raw

        # Initial calibration: only when clearly open to avoid depressing baseline
        if now <= auth_session.calib_until and cur_for_calib is not None and mear_raw > 0.28:
            auth_session.open_baseline = cur_for_calib if auth_session.open_baseline is None else max(auth_session.open_baseline, cur_for_calib)

        # Self-healing baseline after calibration: slow upward tracking on clearly-open frames
        if (auth_session.open_baseline is not None) and (now > auth_session.calib_until):
            if cur_for_calib is not None and mear_raw > (0.02 + (auth_session.open_baseline * auth_session.dyn_ratio)):
                auth_session.open_baseline = (1.0 - BASELINE_UP_ALPHA) * auth_session.open_baseline + BASELINE_UP_ALPHA * cur_for_calib

        # Thresholds: IOD-aware floor + global cap; clamp dynamic threshold to avoid overshoot
        thr_min = _iod_min_threshold(iod if np.isfinite(iod) else None)
        thr_max = thr_min + 0.10  # allow dynamic threshold to be only slightly above IOD min

        if auth_session.open_baseline is not None and now > auth_session.calib_until:
            dyn = float(auth_session.open_baseline * auth_session.dyn_ratio)
            thr_mear = float(np.clip(dyn, thr_min, thr_max))
            thr_ear  = thr_mear
        else:
            thr_mear = max(0.26, thr_min)
            thr_ear  = thr_mear

        # Closed decision: absolute OR ratio gate (soft blink)
        ratio_closed = False
        if auth_session.open_baseline is not None and auth_session.open_baseline > 1e-6:
            ratio_closed = (mear_raw / auth_session.open_baseline) < 0.80

        closed_ear_raw  = ear_raw  < thr_ear
        closed_mear_raw = mear_raw < thr_mear
        closed = bool(closed_mear_raw or closed_ear_raw or ratio_closed)

        # State transitions
        pct_state = 'none'
        if closed and not auth_session.prev_closed: pct_state = 'drop'
        elif (not closed) and auth_session.prev_closed: pct_state = 'rise'
        auth_session.prev_closed = closed

        # Time-gated blink detection
        if closed and auth_session.closed_since is None:
            auth_session.closed_since = now

        blink_detected = False
        if (not closed) and (auth_session.closed_since is not None):
            closed_ms = (now - auth_session.closed_since) * 1000.0
            since_last_ms = (now - auth_session.last_blink_at) * 1000.0
            if closed_ms >= auth_session.closed_min_ms and since_last_ms >= auth_session.refractory_ms:
                blink_detected = True
                auth_session.last_blink_at = now
            auth_session.closed_since = None

        # Confidence based on depth below threshold and ratio drop
        depth = max(0.0, (thr_mear - mear_raw))
        ratio_gap = 0.0
        if auth_session.open_baseline:
            ratio_gap = 0.70 - (mear_raw / max(auth_session.open_baseline, 1e-6))
        confidence = float(np.clip(0.5 + 0.8 * max(depth, 0.08 * ratio_gap), 0.05, 0.98))

        logger.info(
            f"[BLINK] IOD={iod:.1f}px ear_raw={ear_raw:.3f} mear_raw={mear_raw:.3f} "
            f"thr=({thr_ear:.3f},{thr_mear:.3f}) abs_closed={closed} "
            f"pct_state={pct_state} detected={blink_detected}"
        )

        return {
            'blink_detected': bool(blink_detected),
            'available': True,
            'method': 'mear',
            'ear': float(auth_session.ema_ear) if auth_session.ema_ear is not None else ear_raw,
            'mear': float(auth_session.ema_mear) if auth_session.ema_mear is not None else mear_raw,
            'ear_raw': float(ear_raw),
            'mear_raw': float(mear_raw),
            'confidence': float(confidence),
            'thresholds': {
                'ear': float(thr_ear),
                'mear': float(thr_mear),
                'baseline': float(auth_session.open_baseline) if auth_session.open_baseline is not None else None,
                'calibrated': bool(auth_session.open_baseline is not None and now > auth_session.calib_until),
                'closed_min_ms': int(auth_session.closed_min_ms),
                'refractory_ms': int(auth_session.refractory_ms),
                'ratio_gate': 0.75,
                'thr_min': float(thr_min),
            },
            'closed': {
                'ear': bool(closed_ear_raw),
                'mear': bool(closed_mear_raw),
                'ratio': bool(ratio_closed),
                'any': bool(closed),
                'pct_state': pct_state
            }
        }
    except Exception as e:
        logger.error(f"EAR/mEAR blink error: {e}")
        return {'blink_detected': False, 'available': False, 'method': 'mear'}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@authentication.route('/login')
def login_page():
    return render_template('login.html')

def create_auth_session() -> str:
    import secrets
    session_id = secrets.token_urlsafe(16)
    auth_sessions[session_id] = AuthenticationSession(session_id)
    # Optional: clear anti-spoofing temporal buffers when a new auth starts
    try:
        if anti:
            anti.clear_history()
    except Exception:
        pass
    return session_id

def get_auth_session(session_id: str) -> Optional[AuthenticationSession]:
    sess = auth_sessions.get(session_id)
    if not sess:
        return None
    if sess.is_expired():
        try: del auth_sessions[session_id]
        except Exception: pass
        return None
    return sess

@authentication.route('/start_auth', methods=['POST'])
def start_auth():
    try:
        session_id = create_auth_session()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Authentication session started. Please look at the camera.'
        })
    except Exception as e:
        logger.error(f"Failed to start authentication: {e}")
        return jsonify({'error': 'Failed to start authentication'}), 500

@authentication.route('/authenticate_face', methods=['POST'])
def authenticate_face():
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        frame_data = data.get('frame_data', '')
        if not session_id: return jsonify({'error': 'Session ID required'}), 400
        if not frame_data: return jsonify({'error': 'Face image required'}), 400

        auth_session = get_auth_session(session_id)
        if not auth_session: return jsonify({'error': 'Invalid or expired session'}), 400
        if not auth_session.can_attempt(): return jsonify({'error': 'Maximum attempts exceeded or session expired'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok: return jsonify({'error': err}), 400

        # Face matcher
        auth_result = face_matcher.authenticate_user(image)
        if auth_result.get('success'):
            auth_session.face_matched = True
            auth_session.user_id = auth_result.get('user_id')
            auth_session.username = auth_result.get('username')

            # Optional enrich
            pupil_coords = None
            try:
                faces = face_detector.detect_faces(image)
                if faces:
                    best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
                    H, W = image.shape[:2]
                    bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
                    bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
                    if eye_tracker is not None:
                        eyes = eye_tracker.detect_eyes(image, (bx, by, bw, bh))
                        pupils = eye_tracker.track_pupils(eyes)
                        lp = _extract_pupil_value((pupils or {}).get('left_pupil') if isinstance(pupils, dict) else None)
                        rp = _extract_pupil_value((pupils or {}).get('right_pupil') if isinstance(pupils, dict) else None)

                        pupil_coords = {
                            'left_pupil': lp,
                            'right_pupil': rp,
                            'left_pupil_relative': (pupils.get('left_pupil') or {}).get('relative_position') if isinstance(pupils, dict) else None,
                            'right_pupil_relative': (pupils.get('right_pupil') or {}).get('relative_position') if isinstance(pupils, dict) else None,
                            'left_eye_bbox': (eyes.get('left_eye') or {}).get('bbox'),
                            'right_eye_bbox': (eyes.get('right_eye') or {}).get('bbox'),
                            'left_eye_landmarks': (eyes.get('left_eye') or {}).get('landmarks'),
                            'right_eye_landmarks': (eyes.get('right_eye') or {}).get('landmarks'),
                            'face_bbox': [bx, by, bw, bh],
                            'eyes_detected': bool(eyes.get('eyes_detected', False)),
                            'detection_method': eyes.get('method', 'none'),
                            'gaze_direction': (pupils or {}).get('gaze_direction') if isinstance(pupils, dict) else None
                        }
            except Exception as e:
                logger.error(f"Error getting pupil coordinates for face auth: {e}")
                pupil_coords = None

            return jsonify({
                'success': True,
                'message': f'Face recognized as {auth_result.get("username")}',
                'user_info': {
                    'user_id': auth_result.get('user_id'),
                    'username': auth_result.get('username'),
                    'full_name': auth_result.get('full_name')
                },
                'next_step': 'blink_detection',
                'face_info': auth_result.get('face_info'),
                'pupil_coords': pupil_coords
            })
        else:
            auth_session.attempts += 1
            return jsonify({
                'success': False,
                'error': auth_result.get('error', 'Face not recognized'),
                'attempts_remaining': auth_session.max_attempts - auth_session.attempts,
                'can_retry': auth_session.can_attempt(),
                'best_candidate': auth_result.get('best_candidate')
            }), 400

    except Exception as e:
        logger.error(f"Face authentication error: {e}")
        return jsonify({'error': 'Face authentication failed'}), 500

@authentication.route('/authenticate_blink', methods=['POST'])
def authenticate_blink():
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        frame_data = data.get('frame_data', '')

        if not session_id: return jsonify({'error': 'Session ID required'}), 400
        if not frame_data: return jsonify({'error': 'Face image required'}), 400

        auth_session = get_auth_session(session_id)
        if not auth_session: return jsonify({'error': 'Invalid or expired session'}), 400
        if not auth_session.face_matched: return jsonify({'error': 'Face authentication required first'}), 400
        if not auth_session.can_attempt(): return jsonify({'error': 'Maximum attempts exceeded or session expired'}), 400

        # If we already have enough blinks from live polling, finalize quickly
        if auth_session.blink_count >= auth_session.required_blinks:
            session_token = db.create_session(auth_session.user_id)
            db.update_last_login(auth_session.user_id)

            try:
                db.log_auth_attempt(
                    auth_session.user_id,
                    'blink_detection',
                    True,
                    0.9,
                    {
                        'blink_confidence': 0.9,
                        'method': 'mear',
                        'thresholds': {}
                    }
                )
            except Exception as e:
                logger.error(f"Logging auth attempt failed: {e}")

            try: del auth_sessions[session_id]
            except Exception: pass

            resp = jsonify({
                'success': True,
                'message': 'Authentication successful!',
                'session_token': session_token,
                'redirect_url': '/dashboard',
                'user_info': {
                    'user_id': auth_session.user_id,
                    'username': auth_session.username
                },
                'blink_info': {
                    'detected': True,
                    'confidence': 0.9,
                    'ear': None,
                    'mear': None,
                    'method': 'mear',
                    'thresholds': {}
                }
            })
            try:
                resp.set_cookie('session_token', session_token, max_age=60*60*8, secure=False, httponly=False, samesite='Lax', path='/')
            except Exception:
                pass
            return resp

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok: return jsonify({'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces: return jsonify({'error': 'No face detected'}), 400

        best_face = max(faces, key=lambda x: x.get('confidence', 0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
        bbox = (bx, by, bw, bh)

        blink = _blink_via_mear(image, bbox, auth_session)
        if not blink.get('available') and eye_tracker is not None:
            try:
                eyes = eye_tracker.detect_eyes(image, bbox)
                blink = eye_tracker.detect_blink(eyes)
            except Exception as _e:
                logger.error(f"Fallback blink failed: {_e}")
                blink = {'blink_detected': False, 'confidence': 0.0, 'method': 'unknown'}

        blink_detected = bool(blink.get('blink_detected', False))
        blink_confidence = float(blink.get('confidence', 0.0))

        # Require N blinks; do not finalize until met
        if auth_session.blink_count < auth_session.required_blinks:
            return jsonify({
                'success': False,
                'error': f'More blinks required ({auth_session.blink_count}/{auth_session.required_blinks}).',
                'blink_info': {
                    'detected': blink_detected,
                    'confidence': blink_confidence,
                    'ear': blink.get('ear'),
                    'mear': blink.get('mear'),
                    'method': blink.get('method', 'unknown'),
                    'thresholds': blink.get('thresholds')
                },
                'progress': {
                    'blink_count': auth_session.blink_count,
                    'required_blinks': auth_session.required_blinks
                }
            }), 400

        if blink_detected and blink_confidence >= 0.5:  # type: ignore[func-returns-value]
            auth_session.blink_detected = True

            session_token = db.create_session(auth_session.user_id)
            db.update_last_login(auth_session.user_id)

            try:
                db.log_auth_attempt(
                    auth_session.user_id,
                    'blink_detection',
                    True,
                    blink_confidence,
                    {
                        'blink_confidence': blink_confidence,
                        'method': blink.get('method', 'unknown'),
                        'thresholds': blink.get('thresholds', {})
                    }
                )
            except Exception as e:
                logger.error(f"Logging auth attempt failed: {e}")

            try: del auth_sessions[session_id]
            except Exception: pass

            resp = jsonify({
                'success': True,
                'message': 'Authentication successful!',
                'session_token': session_token,
                'redirect_url': '/dashboard',
                'user_info': {
                    'user_id': auth_session.user_id,
                    'username': auth_session.username
                },
                'blink_info': {
                    'detected': blink_detected,
                    'confidence': blink_confidence,
                    'ear': blink.get('ear'),
                    'mear': blink.get('mear'),
                    'method': blink.get('method', 'unknown'),
                    'thresholds': blink.get('thresholds')
                }
            })
            try:
                resp.set_cookie('session_token', session_token, max_age=60*60*8, secure=False, httponly=False, samesite='Lax', path='/')
            except Exception:
                pass
            return resp
        else:
            auth_session.attempts += 1
            return jsonify({
                'success': False,
                'error': 'Blink not detected. Please blink naturally.',
                'blink_info': {
                    'detected': blink_detected,
                    'confidence': blink_confidence,
                    'ear': blink.get('ear'),
                    'mear': blink.get('mear'),
                    'method': blink.get('method', 'unknown'),
                    'thresholds': blink.get('thresholds')
                },
                'attempts_remaining': auth_session.max_attempts - auth_session.attempts,
                'can_retry': auth_session.can_attempt()
            }), 400

    except Exception as e:
        logger.error(f"Blink authentication error: {e}")
        try:
            data = request.get_json() or {}
            sess = get_auth_session(data.get('session_id', ''))
            if sess: sess.attempts += 1
        except Exception:
            pass
        return jsonify({'error': 'Blink authentication failed'}), 500

@authentication.route('/blink_live', methods=['POST'])
def blink_live():
    """
    Live blink analysis (poll per frame).
    - Does NOT increment attempts or finalize auth.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        frame_data = data.get('frame_data', '')

        if not session_id: return jsonify({'error': 'Session ID required'}), 400
        if not frame_data: return jsonify({'error': 'Face image required'}), 400

        auth_session = get_auth_session(session_id)
        if not auth_session: return jsonify({'error': 'Invalid or expired session'}), 400

        ok, image, err = _decode_data_url_to_bgr(frame_data)
        if not ok: return jsonify({'error': err}), 400

        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({'success': True, 'error': 'No face detected'}), 200  # soft for UI

        best_face = max(faces, key=lambda x: x.get('confidence', 0.0))
        H, W = image.shape[:2]
        bx, by, bw, bh = _normalize_bbox(best_face['bbox'], W, H)
        bx, by, bw, bh = _expand_bbox(bx, by, bw, bh, W, H, margin=0.15)
        bbox = (bx, by, bw, bh)

        blink = _blink_via_mear(image, bbox, auth_session)

        # Persist strong blink in session to allow fast finalize; count blinks
        try:
            if bool(blink.get('blink_detected', False)) and float(blink.get('confidence', 0.0)) >= 0.5:
                auth_session.blink_detected = True
                auth_session.blink_count = int(auth_session.blink_count) + 1
        except Exception:
            pass

        return jsonify({
            'success': True,
            'blink_info': {
                'detected': bool(blink.get('blink_detected', False)),
                'confidence': float(blink.get('confidence', 0.0)),
                'method': blink.get('method', 'mear'),
                'ear': float(blink.get('ear')) if blink.get('ear') is not None else None,
                'mear': float(blink.get('mear')) if blink.get('mear') is not None else None,
                'ear_raw': float(blink.get('ear_raw')) if blink.get('ear_raw') is not None else None,
                'mear_raw': float(blink.get('mear_raw')) if blink.get('mear_raw') is not None else None,
                'thresholds': blink.get('thresholds', {}),
                'closed': blink.get('closed', {}),
                'available': blink.get('available', True),
                'reason': blink.get('reason')
            },
            'face_bbox': [bx, by, bw, bh],
            'session_status': {
                'face_matched': auth_session.face_matched,
                'blink_detected': auth_session.blink_detected,
                'blink_count': auth_session.blink_count,
                'required_blinks': auth_session.required_blinks,
                'attempts': auth_session.attempts,
                'max_attempts': auth_session.max_attempts,
                'time_remaining': max(0, auth_session.timeout - (time.time() - auth_session.created_at))
            }
        })
    except Exception as e:
        logger.error(f"Live blink error: {e}")
        return jsonify({'error': 'Live blink analysis failed'}), 500

@authentication.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """
    Lightweight analyzer for overlay and anti-spoofing logs.
    """
    try:
        data = request.get_json() or {}
        frame_data = data.get('frame_data', '')

        ok, img, err = _decode_data_url_to_bgr(frame_data)
        if not ok or img is None:
            return jsonify({'success': False, 'error': err}), 200  # soft fail for UI

        H, W = img.shape[:2]
        eyes = None
        face_bbox = (0, 0, W, H)
        landmarks: List[Tuple[int, int]] = []

        if eye_tracker:
            try:
                eyes = eye_tracker.detect_eyes(img, None)
                if eyes.get('eyes_detected'):
                    landmarks = eyes.get('landmarks') or []
                    if landmarks:
                        face_bbox = _bbox_from_landmarks(landmarks, W, H, pad=18)
            except Exception as e:
                logger.error(f"detect_eyes error: {e}")

        comp = {}
        if anti:
            try:
                # Compute IOD from landmarks if available
                iod_px = None
                if landmarks and len(landmarks) >= 48:
                    pts = np.asarray(landmarks)
                    iod_px = float(np.linalg.norm(pts[42] - pts[39]))
                ear_thr = _iod_min_threshold(iod_px)
                comp = anti.comprehensive_analysis(
                    img, face_bbox, landmarks,
                    blink_hint=None,
                    ear_threshold=ear_thr,   # alias ear_thresh also supported in AntiSpoofingDetector
                    iod_px=iod_px,
                    tag='analyze_frame'
                )
            except Exception as e:
                logger.error(f"anti-spoof error: {e}")
                comp = {}

        eyes_dict = {}
        pupils_dict = {'pupils_detected': False, 'gaze_direction': None}
        if isinstance(eyes, dict):
            eyes_dict = {
                'detected': bool(eyes.get('eyes_detected')),
                'left_bbox': (eyes.get('left_eye') or {}).get('bbox'),
                'right_bbox': (eyes.get('right_eye') or {}).get('bbox'),
                'left_landmarks': (eyes.get('left_eye') or {}).get('landmarks'),
                'right_landmarks': (eyes.get('right_eye') or {}).get('landmarks'),
                'method': eyes.get('method', 'dlib_landmarks'),
            }

        return jsonify({
            'success': True,
            'eye_tracking': {'eyes': eyes_dict, 'pupils': pupils_dict},
            'face_detection': {'bbox': list(face_bbox) if face_bbox else None},
            'anti_spoof': comp
        })
    except Exception as e:
        logger.error(f"analyze_frame error: {e}")
        return jsonify({'success': False, 'error': 'analyze_frame failed'}), 200

@authentication.route('/check_auth_status', methods=['POST'])
def check_auth_status():
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        auth_session = get_auth_session(session_id)
        if not auth_session:
            return jsonify({'error': 'Invalid or expired session'}), 400

        return jsonify({
            'success': True,
            'session_status': {
                'face_matched': auth_session.face_matched,
                'blink_detected': auth_session.blink_detected,
                'attempts': auth_session.attempts,
                'max_attempts': auth_session.max_attempts,
                'can_attempt': auth_session.can_attempt(),
                'is_complete': auth_session.is_complete(),
                'time_remaining': max(0, auth_session.timeout - (time.time() - auth_session.created_at))
            }
        })
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return jsonify({'error': 'Failed to check authentication status'}), 500

@authentication.route('/logout', methods=['POST'])
def logout():
    try:
        data = request.get_json() or {}
        session_token = data.get('session_token')
        if session_token:
            db.logout_session(session_token)
        session.clear()
        try:
            if anti:
                anti.clear_history()
        except Exception:
            pass
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not session_token:
            # fallback to cookie
            session_token = request.cookies.get('session_token', '')
        if not session_token:
            return jsonify({'error': 'Authentication required'}), 401
        user_data = db.validate_session(session_token)
        if not user_data:
            return jsonify({'error': 'Invalid or expired session'}), 401
        request.user_data = user_data  # type: ignore[attr-defined]
        return f(*args, **kwargs)
    return decorated_function

@authentication.route('/dashboard')
@require_auth
def dashboard():
    user = getattr(request, 'user_data', None)
    return render_template('dashboard.html', user=user)
