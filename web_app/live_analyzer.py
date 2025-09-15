"""
Dedicated Live Analyzer for FaceGuard AI
Handles live camera feed and real-time analysis only.
"""

from flask import Blueprint, request, jsonify, Response
import cv2
import numpy as np
import base64
import os
import sys
import time
from pathlib import Path
import dlib
from imutils import face_utils
from collections import deque  # for low-FPS percent-drop fallback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.anti_spoofing import AntiSpoofingDetector
from src.eye_tracker import EyeTracker
from web_app.face_hf_engine import HFDeepfakeEngine

# ----------------------------- Flask Blueprint ------------------------------
live_analyzer = Blueprint('live_analyzer', __name__)

# ----------------------------- Components -----------------------------------
face_detector = FaceDetector()
hf_engine = HFDeepfakeEngine()
anti_spoofing = AntiSpoofingDetector()

# Eye tracker setup (ONNX optional)
PREDICTOR_PATH = os.getenv('PREDICTOR_PATH')
if not PREDICTOR_PATH or not os.path.exists(PREDICTOR_PATH):
    default_predictor_path = os.path.join('..', 'shape_predictor_68_face_landmarks.dat')
    if os.path.exists(default_predictor_path):
        PREDICTOR_PATH = default_predictor_path

eye_tracker = EyeTracker(PREDICTOR_PATH, onnx_model_path=None, onnx_input_size=224)

# dlib predictor (for robust EAR/mEAR landmarks)
ear_predictor = None
if PREDICTOR_PATH and os.path.exists(PREDICTOR_PATH):
    try:
        ear_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except Exception:
        ear_predictor = None

# ----------------------------- Config ---------------------------------------
# EAR-based blink detection config (using STANDARD EAR = (v1+v2)/(2*h))
EAR_BLINK_ENABLED   = True
BLINK_THRESH        = 0.23     # static fallback threshold for standard EAR
MEAR_BLINK_ENABLED  = True
MEAR_THRESH         = 0.23     # static fallback threshold for mEAR
BLINK_CONSEC_FRAMES = 1       # legacy informational; main logic is time-based
EMA_ALPHA           = 0.6     # smoothing for stability (for display only)

# Debug overlay config
DEBUG_OVERLAY_DEFAULT = True   # set False in production
DEBUG_JPEG_QUALITY    = 85

# Landmark index constants for dlib 68-point model (for reference)
L_start, L_end = 36, 42
R_start, R_end = 42, 48

# Per-client blink state cache
blink_states = {}

# --- Blink calibration & timing (time-based edge trigger) ---
CALIBRATE_SECONDS = 0.7   # faster start
CLOSED_MIN_MS     = 90    # shorter closures count as a blink
REFRACTORY_MS     = 400   # avoid double-count on slow FPS
DYN_THRESH_RATIO  = 0.72  # slightly lower dynamic threshold

# --- Percent-drop fallback (low-FPS tolerant) ---
BUF_WINDOW_MS      = 2000   # keep ~2s of raw samples
DROP_RATIO_MIN     = 0.28   # need >=28% drop from recent-open reference
REBOUND_RATIO_MIN  = 0.18   # rebound near-open to confirm blink
DROP_MIN_MS        = 40     # drop must last at least 40ms
DROP_MAX_MS        = 700    # and at most 700ms

# ----------------------------- Helpers --------------------------------------
def _client_key(data: dict) -> str:
    key = (data or {}).get('client_key')
    if not key:
        ua = (request.headers.get('User-Agent') or '')[:32]
        key = f"{request.remote_addr}:{ua}"
    return key

def _normalize_bbox(bbox, img_w, img_h):
    """
    Accept [x,y,w,h] or [x1,y1,x2,y2]; return clamped [x,y,w,h].
    """
    if bbox is None or len(bbox) != 4:
        return 0, 0, img_w, img_h
    x0, y0, a, b = bbox
    # Heuristic: if a,b look like x2,y2 inside image, convert to w,h
    if a > x0 and b > y0 and a <= img_w and b <= img_h:
        x, y, w, h = x0, y0, a - x0, b - y0
    else:
        x, y, w, h = x0, y0, a, b
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > img_w: w = img_w - x
    if y + h > img_h: h = img_h - y
    return x, y, w, h

def _as_dlib_rect(x, y, w, h):
    return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

def _rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (points - center) @ R.T + center

def _compute_ear_from_shape(shape_points: np.ndarray) -> float:
    left  = shape_points[36:42]
    right = shape_points[42:48]
    def ear(eye):
        y1 = np.linalg.norm(eye[1] - eye[5])
        y2 = np.linalg.norm(eye[2] - eye[4])
        x1 = np.linalg.norm(eye[0] - eye[3])
        return (y1 + y2) / (2.0 * x1 + 1e-6)  # STANDARD EAR
    return (ear(left) + ear(right)) / 2.0

def _compute_mear_from_shape(shape_points: np.ndarray) -> float:
    left  = shape_points[36:42]
    right = shape_points[42:48]
    def ear_rot(eye):
        p0, p3 = eye[0], eye[3]
        angle  = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        center = eye.mean(axis=0)
        eye_r  = _rotate_points(eye, -angle, center)  # unroll (pose-normalize)
        y1 = np.linalg.norm(eye_r[1] - eye_r[5])
        y2 = np.linalg.norm(eye_r[2] - eye_r[4])
        x1 = np.linalg.norm(eye_r[0] - eye_r[3])
        return (y1 + y2) / (2.0 * x1 + 1e-6)
    return (ear_rot(left) + ear_rot(right)) / 2.0

def _image_to_data_url(img_bgr, quality=85):
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"

def _draw_eye_overlay(frame_bgr, shape_np, bbox_xywh=None,
                      ear_val=None, mear_val=None, ema_ear=None, ema_mear=None,
                      thresholds=None):
    """
    Draws:
      - yellow points for eye landmarks
      - blue convex hulls
      - white baseline (p0->p3) and cyan/green vertical segments (p1->p5, p2->p4)
      - EAR/mEAR/threshold text and face bbox
    """
    if shape_np is None:
        return frame_bgr

    # colors (BGR)
    C_PT  = (0, 255, 255)   # yellow points
    C_HUL = (255, 0, 0)     # blue hull
    C_BL  = (245, 245, 245) # white baseline
    C_V1  = (255, 255, 0)   # cyan-ish vertical 1
    C_V2  = (0, 220, 0)     # green vertical 2
    C_BOX = (0, 200, 0)     # bbox

    left  = shape_np[36:42]
    right = shape_np[42:48]

    def draw_eye(eye):
        for (x, y) in eye:
            cv2.circle(frame_bgr, (int(x), int(y)), 1, C_PT, -1)
        hull = cv2.convexHull(eye)
        cv2.drawContours(frame_bgr, [hull], -1, C_HUL, 1)
        p0, p1, p2, p3, p4, p5 = [tuple(map(int, p)) for p in eye]
        cv2.line(frame_bgr, p0, p3, C_BL, 1)  # baseline
        cv2.line(frame_bgr, p1, p5, C_V1, 1)  # verticals
        cv2.line(frame_bgr, p2, p4, C_V2, 1)

    draw_eye(left)
    draw_eye(right)

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = [int(v) for v in bbox_xywh]
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), C_BOX, 1)

    y0 = 22
    dy = 20
    def put(txt, y):
        cv2.putText(frame_bgr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2)

    ear_txt  = f"EAR:  {ear_val:.3f}"   if ear_val  is not None else "EAR:  -"
    mear_txt = f"mEAR: {mear_val:.3f}" if mear_val is not None else "mEAR: -"
    if ema_ear  is not None: ear_txt  += f" (EMA {ema_ear:.3f})"
    if ema_mear is not None: mear_txt += f" (EMA {ema_mear:.3f})"
    put(ear_txt,  y0)
    put(mear_txt, y0+dy)

    if thresholds:
        t_ear  = thresholds.get('ear',  None)
        t_mear = thresholds.get('mear', None)
        consec = thresholds.get('consec_frames', None)
        th_txt = f"Thr: EAR={t_ear:.2f} mEAR={t_mear:.2f} N={consec}" if (t_ear is not None and t_mear is not None and consec is not None) else ""
        if th_txt:
            put(th_txt, y0+2*dy)

    return frame_bgr

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def _detect_faces_preferring_dlib(image):
    try:
        return face_detector.detect_faces(image)
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

# ----------------------------- Blink via EAR/mEAR ---------------------------
def _blink_via_ear(frame_bgr: np.ndarray, bbox_xywh, data: dict, debug_overlay: bool = False) -> dict:
    """
    Time-based edge-triggered blink detection with EAR/mEAR, EMA smoothing (display),
    automatic open-eye calibration, and percent-drop fallback for low-FPS streams.
    Fires when eyes reopen after being closed for >= CLOSED_MIN_MS OR when a
    significant percent-drop + rebound sequence is observed.
    """
    if ear_predictor is None:
        return {'blink_detected': False, 'eye_aspect_ratio': None, 'method': 'ear', 'available': False}

    H, W = frame_bgr.shape[:2]
    try:
        x, y, w, h = _normalize_bbox(bbox_xywh, W, H)
    except Exception:
        x, y, w, h = [int(v) for v in bbox_xywh]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rect = _as_dlib_rect(x, y, w, h)

    try:
        shape    = ear_predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        ear_raw  = float(_compute_ear_from_shape(shape_np))
        mear_raw = float(_compute_mear_from_shape(shape_np)) if MEAR_BLINK_ENABLED else None

        key = _client_key(data)
        now = time.time()
        st = blink_states.get(key) or {
            # EMA (display only)
            'ema_ear': None, 'ema_mear': None,
            # calibration
            'calib_until': now + CALIBRATE_SECONDS,
            'open_ear_baseline': None,
            # time-based edge trigger (absolute threshold path)
            'closed_since': None,
            'last_blink_at': 0.0,
            # percent-drop fallback state
            'buf': deque(),                # (t, raw_measure)
            'drop_started_at': None,
            'drop_from': None,
            'drop_min': None,
            # overlay/debug
            'last_thresh_ear': BLINK_THRESH,
            'last_thresh_mear': MEAR_THRESH
        }

        # EMA for overlay
        st['ema_ear']  = ear_raw  if st['ema_ear']  is None else (EMA_ALPHA*ear_raw  + (1-EMA_ALPHA)*st['ema_ear'])
        if mear_raw is not None:
            st['ema_mear'] = mear_raw if st['ema_mear'] is None else (EMA_ALPHA*mear_raw + (1-EMA_ALPHA)*st['ema_mear'])

        cur_ear  = st['ema_ear']  if st['ema_ear']  is not None else ear_raw
        cur_mear = st['ema_mear'] if st['ema_mear'] is not None else mear_raw

        # calibration: capture max open in early window
        if now <= st['calib_until']:
            candidate = cur_mear if (MEAR_BLINK_ENABLED and cur_mear is not None) else cur_ear
            if candidate is not None:
                st['open_ear_baseline'] = candidate if st['open_ear_baseline'] is None else max(st['open_ear_baseline'], candidate)

        # optional static override
        force_static = bool((data or {}).get('force_static_thresh', False))
        if (st['open_ear_baseline'] is not None and now > st['calib_until']) and not force_static:
            dyn = float(st['open_ear_baseline'] * DYN_THRESH_RATIO)
            dyn_ear  = dyn
            dyn_mear = dyn
        else:
            dyn_ear  = BLINK_THRESH
            dyn_mear = MEAR_THRESH
        st['last_thresh_ear']  = float(dyn_ear)
        st['last_thresh_mear'] = float(dyn_mear)

        # ---------- Absolute-threshold path (raw) ----------
        closed_ear_raw  = (ear_raw  is not None) and (ear_raw  < dyn_ear)
        closed_mear_raw = (mear_raw is not None) and (mear_raw < dyn_mear)
        closed_abs = closed_mear_raw if (MEAR_BLINK_ENABLED and (mear_raw is not None)) else closed_ear_raw

        blink_detected = False

        # start/end closed window
        if closed_abs and st['closed_since'] is None:
            st['closed_since'] = now
        if (not closed_abs) and (st['closed_since'] is not None):
            closed_ms = (now - st['closed_since']) * 1000.0
            if closed_ms >= CLOSED_MIN_MS and (now - st['last_blink_at']) * 1000.0 >= REFRACTORY_MS:
                blink_detected = True
                st['last_blink_at'] = now
            st['closed_since'] = None

        # ---------- Percent-drop fallback (low-FPS tolerant) ----------
        # choose measure: prefer mEAR if available
        measure = mear_raw if (MEAR_BLINK_ENABLED and mear_raw is not None) else ear_raw

        # update buffer
        st['buf'].append((now, measure))
        cutoff = now - (BUF_WINDOW_MS / 1000.0)
        while st['buf'] and st['buf'][0][0] < cutoff:
            st['buf'].popleft()

        # recent "open" reference = max value in buffer (or baseline if available)
        if st['buf']:
            recent_max = max(v for _, v in st['buf'])
        else:
            recent_max = st['open_ear_baseline'] if st['open_ear_baseline'] is not None else measure

        # start drop if we fell by DROP_RATIO_MIN from recent open
        if st['drop_started_at'] is None and recent_max is not None and measure is not None and recent_max > 0:
            if (recent_max - measure) / recent_max >= DROP_RATIO_MIN:
                st['drop_started_at'] = now
                st['drop_from'] = recent_max
                st['drop_min'] = measure

        # update drop, check rebound
        if st['drop_started_at'] is not None:
            # track minimum during drop
            if measure is not None:
                st['drop_min'] = min(st['drop_min'], measure) if st['drop_min'] is not None else measure

            drop_ms = (now - st['drop_started_at']) * 1000.0
            valid_duration = (drop_ms >= DROP_MIN_MS) and (drop_ms <= DROP_MAX_MS)

            # rebound if we climbed back near pre-drop level
            if st['drop_from'] and measure is not None and st['drop_from'] > 0:
                # rebound ratio toward open (1.0 == fully back)
                rebound = 1.0 - (st['drop_from'] - measure) / st['drop_from']
                if valid_duration and rebound >= (1 - REBOUND_RATIO_MIN) and ((now - st['last_blink_at']) * 1000.0 >= REFRACTORY_MS):
                    blink_detected = True or blink_detected
                    st['last_blink_at'] = now
                    st['drop_started_at'] = None
                    st['drop_from'] = None
                    st['drop_min'] = None

            # timeout: abandon drop window
            if drop_ms > DROP_MAX_MS:
                st['drop_started_at'] = None
                st['drop_from'] = None
                st['drop_min'] = None

        blink_states[key] = st

        # log line (helps verify behavior)
        try:
            print(
                f"[BLINK] ear_raw={ear_raw:.3f} mear_raw={(mear_raw if mear_raw is not None else float('nan')):.3f} "
                f"thr=({dyn_ear:.3f},{dyn_mear:.3f}) "
                f"abs_closed={closed_abs} pct_state={('none' if st.get('drop_started_at') is None else 'drop')} "
                f"detected={blink_detected}"
            )
        except Exception:
            pass

        out = {
            'blink_detected': bool(blink_detected),
            # EMA shown for UI readability
            'eye_aspect_ratio': float(cur_ear)  if cur_ear  is not None else None,
            'mear':             float(cur_mear) if cur_mear is not None else None,
            'method': 'mear' if (MEAR_BLINK_ENABLED and cur_mear is not None) else 'ear',
            'available': True,
            'thresholds': {
                'ear': st['last_thresh_ear'],
                'mear': st['last_thresh_mear'],
                'consec_frames': BLINK_CONSEC_FRAMES,
                'calibrated': bool(st['open_ear_baseline'] is not None and now > st['calib_until'])
            },
            'closed': {'ear': bool(closed_ear_raw), 'mear': bool(closed_mear_raw)},
            'debug': {
                'ear_raw': ear_raw, 'mear_raw': mear_raw,
                'recent_max': recent_max,
                'drop_started': st.get('drop_started_at') is not None,
                'drop_from': st.get('drop_from'),
                'drop_min': st.get('drop_min')
            }
        }

        if debug_overlay:
            vis = frame_bgr.copy()
            _draw_eye_overlay(
                vis, shape_np, bbox_xywh=(x, y, w, h),
                ear_val=ear_raw, mear_val=mear_raw,
                ema_ear=st['ema_ear'], ema_mear=st['ema_mear'],
                thresholds={'ear': st['last_thresh_ear'], 'mear': st['last_thresh_mear'], 'consec_frames': BLINK_CONSEC_FRAMES}
            )
            out['debug_image'] = _image_to_data_url(vis, quality=DEBUG_JPEG_QUALITY)

        return out

    except Exception as _e:
        return {'blink_detected': False, 'eye_aspect_ratio': None, 'method': 'ear', 'available': False, 'error': str(_e)}

# ----------------------------- Routes ---------------------------------------
@live_analyzer.route('/detect_faces', methods=['POST'])
def detect_faces():
    try:
        data = request.get_json()
        if not data or 'frame_data' not in data:
            return jsonify({'error': 'No frame data provided'})
        frame_data = data['frame_data']
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'})
        faces = _detect_faces_preferring_dlib(frame)
        if not faces:
            return jsonify({'success': True, 'faces': [], 'face_count': 0, 'message': 'No faces detected'})
        faces_info = [{'bbox': f['bbox'], 'confidence': f.get('confidence', 0.8)} for f in faces]
        return jsonify({'success': True, 'faces': faces_info, 'face_count': len(faces_info)})
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return jsonify({'error': str(e)})

@live_analyzer.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        if not data or 'frame_data' not in data:
            return jsonify({'error': 'No frame data provided'})
        frame_data = data['frame_data']
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'})

        # Downscale for faster processing (keep aspect ratio)
        try:
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 720:
                scale = 720.0 / float(max_dim)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        except Exception:
            pass

        debug_overlay = bool((data or {}).get('debug_overlay', DEBUG_OVERLAY_DEFAULT))

        faces = _detect_faces_preferring_dlib(frame)
        if not faces:
            return jsonify({'success': False, 'error': 'No faces detected in frame'})

        # Use first face (You can extend to multi-face if desired)
        face = faces[0]
        raw_bbox = face['bbox']
        H, W = frame.shape[:2]
        x, y, bw, bh = _normalize_bbox(raw_bbox, W, H)
        bbox = [x, y, bw, bh]

        # --- Eye tracking + blink detection ---
        eye_tracking_result = None
        eye_auth_ok = False
        overlay_img = None

        if eye_tracker is not None:
            try:
                eyes = eye_tracker.detect_eyes(frame, bbox)

                landmarks_list = []
                if eyes and isinstance(eyes.get('landmarks'), list) and len(eyes['landmarks']) >= 68:
                    # convert [[x,y], ...] -> [(x,y), ...]
                    landmarks_list = [tuple(map(int, p)) for p in eyes['landmarks']]
                pupils = eye_tracker.track_pupils(eyes)

                blink = _blink_via_ear(frame, bbox, data, debug_overlay=debug_overlay) if EAR_BLINK_ENABLED else None
                eye_auth_ok = bool(blink and blink.get('blink_detected', False))
                overlay_img = blink.get('debug_image') if blink else None

                # Optional: reuse the current EAR threshold you’re displaying
                ear_thr = (blink.get('thresholds', {}) or {}).get('ear', 0.24) if blink else 0.24

                # --- Anti-spoofing (now with blink + landmarks) ---
                anti_spoofing_result = anti_spoofing.comprehensive_analysis(
                    frame,
                    tuple(bbox),
                    landmarks_list,
                    blink_hint=eye_auth_ok,
                    ear_threshold=float(ear_thr),
                )
                if blink and blink.get('available'):
                    eye_auth_ok = bool(blink.get('blink_detected', False))
                    overlay_img = blink.get('debug_image')
                else:
                    # Fallback to EyeTracker’s internal blink detector
                    blink = eye_tracker.detect_blink(eyes)
                    eye_auth_ok = bool(blink.get('blink_detected', False))

                eye_tracking_result = {
                    'eyes': {
                        'detected': eyes.get('eyes_detected', False),
                        'method': eyes.get('method', 'none'),
                        'left_bbox': (eyes.get('left_eye') or {}).get('bbox') if eyes.get('left_eye') else None,
                        'right_bbox': (eyes.get('right_eye') or {}).get('bbox') if eyes.get('right_eye') else None,
                        'left_landmarks': (eyes.get('left_eye') or {}).get('landmarks') if eyes.get('left_eye') else None,
                        'right_landmarks': (eyes.get('right_eye') or {}).get('landmarks') if eyes.get('right_eye') else None,
                    },
                    'pupils': pupils,
                    'blink': blink
                }
            except Exception as e:
                print(f"Eye tracking error: {e}")

        # --- Anti-spoofing ---
        anti_spoofing_result = anti_spoofing.comprehensive_analysis(frame, bbox, [])
        anti_spoofing_risk = float(anti_spoofing_result.get('overall_risk', 0.15))

        # --- Deepfake detection (GATED on blink) ---
        deepfake_result = {'overall_risk': 0.0, 'assessment': 'REAL', 'engine': 'gated'}
        deepfake_details = None
        if eye_auth_ok:
            deepfake_raw = hf_engine.score_frame(frame, bbox)
            df_risk = float(deepfake_raw.get('overall_risk', 0.0))
            deepfake_result = {
                'overall_risk': df_risk,
                'assessment': 'REAL' if df_risk < 0.5 else 'FAKE',
                'engine': 'hf'
            }
            deepfake_details = deepfake_raw.get('details', {})

        deepfake_risk = float(deepfake_result.get('overall_risk', 0.12))

        # --- Risk fusion & decision ---
        overall_risk = (anti_spoofing_risk * 0.6) + (deepfake_risk * 0.4)
        if overall_risk < 0.3:
            assessment = 'AUTHENTIC'
            is_authentic = True
        elif overall_risk < 0.6:
            assessment = 'SUSPICIOUS'
            is_authentic = False
        else:
            assessment = 'DEEPFAKE'
            is_authentic = False

        result = {
            'success': True,
            'face_detected': True,
            'bbox': bbox,
            'confidence': face.get('confidence', 0.8),
            'anti_spoofing': {
                'overall_risk': anti_spoofing_risk,
                'assessment': 'PASS' if anti_spoofing_risk < 0.5 else 'FAIL'
            },
            'deepfake_detection': {
                'overall_risk': deepfake_risk,
                'assessment': deepfake_result.get('assessment', 'REAL'),
                'engine': deepfake_result.get('engine', 'gated')
            },
            'deepfake_details': deepfake_details if eye_auth_ok else None,
            'eye_tracking': eye_tracking_result,
            'eye_auth_ok': eye_auth_ok,
            'overall_risk': overall_risk,
            'is_authentic': is_authentic,
            'assessment': assessment,
            'debug_image': overlay_img  # data URL for quick preview in UI
        }

        result = convert_numpy_types(result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in frame analysis: {str(e)}")
        return jsonify({'error': str(e)})
    finally:
        try:
            import gc
            gc.collect()
        except Exception:
            pass

@live_analyzer.route('/health_check', methods=['GET'])
def health_check():
    try:
        components_status = {
            'face_detector': face_detector is not None,
            'deepfake_hf_engine': hf_engine is not None,
            'anti_spoofing': anti_spoofing is not None,
            'eye_tracker': eye_tracker is not None,
            'dlib_predictor_loaded': ear_predictor is not None
        }
        return jsonify({
            'status': 'healthy',
            'components': components_status,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e), 'timestamp': time.time()}), 500
