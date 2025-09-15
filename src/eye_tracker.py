"""
Minimal Eye Tracker using dlib 68-point landmarks and mEAR blink logic.

Public API:
- EyeTracker.detect_eyes(frame, face_bbox) -> dict with left/right eye bboxes and landmarks
- EyeTracker.track_pupils(eye_data) -> stub result (no pupils)
- EyeTracker.detect_blink(eye_data) -> dict with blink info (ear, mear, thresholds)
"""

from typing import Dict, Optional, Tuple, List
import math
import numpy as np
import cv2
import dlib


def _shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    """Convert dlib shape to (68,2) array (no imutils needed)."""
    coords = np.zeros((shape.num_parts, 2), dtype=np.int32)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


class EyeTracker:
    def __init__(
        self,
        predictor_path: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
        onnx_input_size: int = 224,
        onnx_provider: Optional[str] = None,
    ):
        # dlib detector + predictor only
        self.detector = dlib.get_frontal_face_detector()
        if not predictor_path:
            raise RuntimeError("predictor_path is required for EyeTracker (dlib)")
        self.predictor = dlib.shape_predictor(predictor_path)

        # Blink tuning
        self.blink_thresh_ear = 0.22
        self.blink_thresh_mear = 0.23
        self.blink_consec_frames = 2
        self.ema_alpha = 0.4

        # EMA + gating state
        self._ema_ear: Optional[float] = None
        self._ema_mear: Optional[float] = None
        self._consec_low = 0

        # dlib 68 indices
        self.L_start, self.L_end = 36, 42
        self.R_start, self.R_end = 42, 48

    # ---------- helpers ----------
    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    @staticmethod
    def _rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]], dtype=float)
        return (points - center) @ R.T + center

    def _mear(self, eye_pts: np.ndarray) -> float:
        p0, p3 = eye_pts[0], eye_pts[3]
        angle = math.atan2(p3[1] - p0[1], p3[0] - p0[0])
        center = eye_pts.mean(axis=0)
        eye_rot = self._rotate_points(eye_pts, -angle, center)
        return self._ear(eye_rot)

    # ---------- API ----------
    def detect_eyes(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_bbox is None:
            faces = self.detector(gray, 0)
            if len(faces) == 0:
                return {
                    'eyes_detected': False,
                    'left_eye': None,
                    'right_eye': None,
                    'confidence': 0.0,
                    'method': 'dlib_no_face'
                }
            face = faces[0]
        else:
            x, y, w, h = face_bbox
            face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = self.predictor(gray, face)
        shape_np = _shape_to_np(shape)

        left_pts = shape_np[self.L_start:self.L_end]
        right_pts = shape_np[self.R_start:self.R_end]

        def _bbox(pts: np.ndarray) -> Tuple[int, int, int, int]:
            xs, ys = pts[:, 0], pts[:, 1]
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            pad = 6
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(frame.shape[1], x1 + pad)
            y1 = min(frame.shape[0], y1 + pad)
            return (x0, y0, x1 - x0, y1 - y0)

        left_bbox = _bbox(left_pts)
        right_bbox = _bbox(right_pts)

        return {
            'eyes_detected': True,
            'left_eye': {'bbox': left_bbox, 'landmarks': left_pts.tolist()},
            'right_eye': {'bbox': right_bbox, 'landmarks': right_pts.tolist()},
            'confidence': 0.9,
            'method': 'dlib_landmarks',
            'landmarks': shape_np.tolist()
        }

    def track_pupils(self, eye_data: Dict) -> Dict:
        # Stub (API-compatible)
        return {
            'pupils_detected': False,
            'left_pupil': None,
            'right_pupil': None,
            'gaze_direction': None,
            'confidence': 0.0
        }

    def detect_blink(self, eye_data: Dict) -> Dict:
        if not eye_data or not eye_data.get('eyes_detected'):
            return {'blink_detected': False, 'eye_aspect_ratio': 0.0, 'mear': 0.0, 'blink_count': 0, 'confidence': 0.0}

        left = np.array((eye_data.get('left_eye') or {}).get('landmarks') or [], dtype=float)
        right = np.array((eye_data.get('right_eye') or {}).get('landmarks') or [], dtype=float)
        if left.shape != (6, 2) or right.shape != (6, 2):
            return {'blink_detected': False, 'eye_aspect_ratio': 0.0, 'mear': 0.0, 'blink_count': 0, 'confidence': 0.0}

        raw_ear = (self._ear(left) + self._ear(right)) / 2.0
        raw_mear = (self._mear(left) + self._mear(right)) / 2.0

        # EMA smoothing
        self._ema_ear = raw_ear if self._ema_ear is None else (self.ema_alpha * raw_ear + (1 - self.ema_alpha) * self._ema_ear)
        self._ema_mear = raw_mear if self._ema_mear is None else (self.ema_alpha * raw_mear + (1 - self.ema_alpha) * self._ema_mear)

        closed_ear = (self._ema_ear if self._ema_ear is not None else raw_ear) < self.blink_thresh_ear
        closed_mear = (self._ema_mear if self._ema_mear is not None else raw_mear) < self.blink_thresh_mear

        if closed_ear or closed_mear:
            self._consec_low += 1
        else:
            self._consec_low = 0

        blink_detected = self._consec_low >= self.blink_consec_frames
        if blink_detected:
            self._consec_low = 0

        return {
            'blink_detected': bool(blink_detected),
            'eye_aspect_ratio': float(self._ema_ear) if self._ema_ear is not None else raw_ear,
            'mear': float(self._ema_mear) if self._ema_mear is not None else raw_mear,
            'blink_count': 1 if blink_detected else 0,
            'confidence': 0.85 if blink_detected else 0.7,
            'method': 'mear',
            'thresholds': {
                'ear': self.blink_thresh_ear,
                'mear': self.blink_thresh_mear,
                'consec_frames': self.blink_consec_frames
            },
            'closed': {'ear': closed_ear, 'mear': closed_mear}
        }
