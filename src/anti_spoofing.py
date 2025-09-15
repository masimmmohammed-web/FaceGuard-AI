"""
Anti-spoofing module for detecting photo/video attacks and ensuring liveness.
Safe, dependency-light version with robust guards and sensible defaults.
"""

from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Optional utils (used if present) ---------------------------------------------------------
try:
    from .utils import calculate_optical_flow as _utils_calc_flow   # type: ignore
except Exception:
    _utils_calc_flow = None

try:
    from .utils import detect_blinking as _utils_detect_blinking     # type: ignore
except Exception:
    _utils_detect_blinking = None


def _iod_min_threshold(iod_px: float | None) -> float:
    """
    For small faces (low IOD), use a reasonable minimum threshold.
    Keep default consistent with the auth pipeline (None -> 0.28).
    """
    if iod_px is None or not np.isfinite(iod_px):
        return 0.28
    thr = 0.23 + 0.002 * max(0.0, iod_px - 30.0)
    return float(np.clip(thr, 0.23, 0.32))


class AntiSpoofingDetector:
    """Detects photo/video/3D mask spoofing via simple spatial + temporal signals."""

    def __init__(self, history: int = 30):
        # History (use deque to bound memory)
        self.frame_history: deque[np.ndarray] = deque(maxlen=max(5, history))
        # Tunables (picked to be conservative and easily adjustable)
        self._flow_pts = 80               # number of corners for flow
        self._min_face_px = 24 * 24       # ignore tiny faces
        self._movement_eps = 0.6          # px median movement threshold for "very still"
        self._still_frames_for_photo = 2  # how many consecutive still deltas suggest a photo
        self._recent_flow_medians: deque[float] = deque(maxlen=5)

    # ----------------------- utilities -----------------------

    @staticmethod
    def _clamp_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(0, min(w, W - x))
        h = max(0, min(h, H - y))
        return x, y, w, h

    @staticmethod
    def _safe_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y:y + h, x:x + w]

    @staticmethod
    def _fast_hash(frame: np.ndarray) -> int:
        """
        Very cheap near-duplicate detector: downsample + single channel sum.
        Used only to avoid awarding 'stillness' on self-compares / duplicate frames.
        """
        # Ensure we have at least 1 channel; frames here are BGR (H,W,3)
        sub = frame[::8, ::8, 0]  # take blue channel for speed; any channel is fine
        return int(sub.sum()) & 0x7FFFFFFF

    def _median_optical_flow(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, bool]:
        """
        Median per-point displacement (pixels) inside bbox; falls back gracefully.

        Returns
        -------
        (median_flow, is_duplicate)
            is_duplicate=True means prev and curr look identical (same shape + same fast hash).
            Callers should NOT treat (0.0, True) as evidence of stillness.
        """
        # Duplicate-frame guard (shape + cheap hash check)
        if prev.shape == curr.shape and self._fast_hash(prev) == self._fast_hash(curr):
            return 0.0, True

        x, y, w, h = bbox
        if w * h < self._min_face_px:
            return 0.0, False

        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        roi_prev = prev_g[y:y + h, x:x + w]
        if roi_prev.size == 0:
            return 0.0, False

        # Use external flow util if available (we've already dup-guarded above)
        if _utils_calc_flow is not None:
            try:
                return float(_utils_calc_flow(prev, curr, bbox)), False  # type: ignore[arg-type]
            except Exception:
                pass

        # Native: track good features within ROI
        pts = cv2.goodFeaturesToTrack(roi_prev, maxCorners=self._flow_pts, qualityLevel=0.01, minDistance=4)
        if pts is None or len(pts) == 0:
            return 0.0, False

        # Shift ROI points to image coords
        pts = pts.reshape(-1, 2)
        pts[:, 0] += x
        pts[:, 1] += y
        pts = pts.astype(np.float32)

        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts, None, winSize=(15, 15), maxLevel=2)
        if nxt is None or st is None:
            return 0.0, False

        ok = st.reshape(-1) == 1
        if not np.any(ok):
            return 0.0, False

        disp = np.linalg.norm((nxt[ok] - pts[ok]), axis=1)
        return (float(np.median(disp)) if disp.size else 0.0), False

    @staticmethod
    def _landmarks_have_eyes(landmarks: List[Tuple[int, int]]) -> bool:
        # Dlib-68 eye indices exist if we have >= 48 points
        return len(landmarks) >= 48

    @staticmethod
    def _ear(eye: np.ndarray) -> float:
        # eye: (6,2)
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def _blink_from_landmarks(self, landmarks: List[Tuple[int, int]], ear_thresh: float = 0.24) -> Tuple[bool, float]:
        """
        One-shot blink cue from 68-pt landmarks.
        Returns (blink_detected, mean_ear).
        """
        if not self._landmarks_have_eyes(landmarks):
            return False, float("nan")
        pts = np.asarray(landmarks, dtype=np.float32)
        L = pts[36:42]  # 36-41
        R = pts[42:48]  # 42-47
        ear = (self._ear(L) + self._ear(R)) * 0.5
        return (ear < ear_thresh), float(ear)

    # ----------------------- public API -----------------------

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to history for temporal analysis."""
        if frame is not None and frame.size:
            self.frame_history.append(frame.copy())

    def detect_photo_attack(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """Detects cues consistent with a static photo or screen presentation."""
        try:
            face_bbox = self._clamp_bbox(frame, face_bbox)
            x, y, w, h = face_bbox
            face_roi = self._safe_roi(frame, face_bbox)
            H, W = face_roi.shape[:2]
            area = float(max(1, W * H))
            if area <= 1:
                return {'is_photo_attack': False, 'confidence': 0.0, 'metrics': {}}

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # --- Spatial signals -------------------------------------------------
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # "sharpness"
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / area

            color_var = float(np.var(face_roi.reshape(-1, 3).astype(np.float32), axis=0).mean())

            # Local texture variance (lower => more uniform/printed)
            kernel = np.ones((3, 3), np.float32) / 9.0
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            texture_var = float(cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel).mean())

            # Screen-like saturation/brightness spikes
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            sat_hi = float(np.mean(sat > 200))
            val_hi = float(np.mean(val > 240))

            # Rectangle / bezel-ish contour near full ROI
            screen_rect = False
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area_ratio = cv2.contourArea(c) / area
                eps = 0.02 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True)
                if len(approx) == 4 and 0.6 <= area_ratio <= 0.98:
                    # Check rectangularity
                    rect = cv2.minAreaRect(c)
                    (rw, rh) = rect[1]
                    if rw > 1 and rh > 1:
                        ar = max(rw, rh) / min(rw, rh)
                        screen_rect = 1.4 <= ar <= 2.4  # ~phone/tablet aspect range

            # --- Temporal signals (stillness) -----------------------------------
            still_bonus = 0.0
            if len(self.frame_history) >= 1:
                prev = self.frame_history[-1]
                if prev.shape == frame.shape:
                    med_flow, is_dup = self._median_optical_flow(prev, frame, face_bbox)
                    # Only track & reward stillness if this is NOT a self-compare
                    if not is_dup:
                        self._recent_flow_medians.append(med_flow)
                        very_still = med_flow < self._movement_eps
                        # If *several* consecutive medians are tiny, stronger photo suspicion
                        if very_still and sum(m < self._movement_eps for m in self._recent_flow_medians) >= self._still_frames_for_photo:
                            still_bonus = 0.25
                else:
                    # shape mismatch – can't compare
                    self._recent_flow_medians.clear()

            # --- Heuristic fusion (bounded) -------------------------------------
            w_sharp = 0.20 if lap_var > 700 else 0.0
            w_edge  = 0.20 if edge_density > 0.35 else 0.0
            w_color = 0.15 if color_var < 160 else 0.0
            w_text  = 0.15 if texture_var < 12 else 0.0
            w_sat   = 0.15 if sat_hi > 0.65 else 0.0
            w_val   = 0.10 if val_hi > 0.75 else 0.0
            w_rect  = 0.15 if screen_rect else 0.0

            confidence = min(1.0, w_sharp + w_edge + w_color + w_text + w_sat + w_val + w_rect + still_bonus)
            is_photo = confidence >= 0.5  # tune as needed

            metrics = {
                'sharpness': lap_var,
                'edge_density': edge_density,
                'color_variance': color_var,
                'texture_variance': texture_var,
                'sat_hi_ratio': sat_hi,
                'val_hi_ratio': val_hi,
                'screen_rect': bool(screen_rect),
                'recent_flow_median': float(self._recent_flow_medians[-1]) if self._recent_flow_medians else None,
                'still_bonus': still_bonus,
            }

            logger.info(f"[photo] conf={confidence:.2f} metrics={metrics}")
            return {'is_photo_attack': bool(is_photo), 'confidence': float(confidence), 'metrics': metrics}

        except Exception as e:
            logger.exception("Error in detect_photo_attack")
            return {'is_photo_attack': False, 'confidence': 0.0, 'metrics': {'error': str(e)}}

    def detect_liveness(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks: List[Tuple[int, int]] = (),
        *,
        blink_hint: Optional[bool] = None,
        ear_threshold: Optional[float] = None,
        ear_thresh: Optional[float] = None,  # alias for robustness
        iod_px: Optional[float] = None,
        tag: str = "",
    ) -> Dict:
        """
        Checks quick liveness cues (blink cue + micro-movement + texture/color variety).

        Parameters
        ----------
        blink_hint : Optional[bool]
            If you already computed blink externally (e.g., your EAR tracker prints
            `[BLINK] ... detected=...`), pass that boolean here and it will be used directly.
            If None, we will try to infer blink from 68-pt landmarks or optional utils.
        ear_threshold : float
            EAR threshold used when inferring blink from landmarks (default: dynamic via IOD).
        ear_thresh : float
            Alias for `ear_threshold`. If both are provided, `ear_threshold` wins.
        tag : str
            Optional tag to disambiguate callers in logs (e.g., "analyze_frame").
        """
        # Compute/resolve EAR threshold (alias-friendly)
        src = "provided"
        if ear_threshold is None and ear_thresh is not None:
            ear_threshold = ear_thresh
        if ear_threshold is None:
            ear_threshold = _iod_min_threshold(iod_px)
            src = "iod_dynamic"

        try:
            face_bbox = self._clamp_bbox(frame, face_bbox)
            x, y, w, h = face_bbox
            face_roi = self._safe_roi(frame, face_bbox)
            if face_roi.size == 0:
                return {'is_live': False, 'confidence': 0.0, 'metrics': {'reason': 'empty_roi'}}

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Blink: prefer explicit hint; else infer
            blink = bool(blink_hint) if blink_hint is not None else False
            ear_val = float("nan")

            if blink_hint is None:
                if self._landmarks_have_eyes(landmarks):
                    blink, ear_val = self._blink_from_landmarks(landmarks, ear_thresh=ear_threshold)
                elif _utils_detect_blinking is not None:
                    try:
                        blink = bool(_utils_detect_blinking(landmarks))
                    except Exception:
                        blink = False

            # Temporal micro-movement (reuse median flow)
            micro_mv = 0.0
            if len(self.frame_history) >= 1 and self.frame_history[-1].shape == frame.shape:
                mv, is_dup = self._median_optical_flow(self.frame_history[-1], frame, face_bbox)
                if not is_dup:
                    micro_mv = mv  # ignore self-compare zeros

            # Variety cues
            texture_var = float(np.var(gray))
            color_var = float(np.var(face_roi.reshape(-1, 3).astype(np.float32), axis=0).mean())

            # Score (bounded sum of small contributions)
            score = 0.0
            score += 0.45 if blink else 0.0
            score += 0.25 if micro_mv > 0.15 else 0.0         # small natural motion
            score += 0.15 if texture_var > 60 else 0.0        # some skin texture
            score += 0.10 if color_var > 220 else 0.0         # natural chroma variance

            is_live = score >= 0.25
            metrics = {
                'blink_detected': bool(blink),
                'ear': ear_val,
                'ear_threshold': ear_threshold,
                'ear_threshold_source': src,
                'micro_movement_px': float(micro_mv),
                'texture_variance': texture_var,
                'color_variance': color_var,
            }

            if tag:
                logger.info(f"[liveness:{tag}] score={score:.2f} live={is_live} metrics={metrics}")
            else:
                logger.info(f"[liveness] score={score:.2f} live={is_live} metrics={metrics}")
            return {'is_live': bool(is_live), 'confidence': float(score), 'metrics': metrics}

        except Exception as e:
            logger.exception("Error in detect_liveness")
            return {'is_live': False, 'confidence': 0.0, 'metrics': {'error': str(e)}}

    def comprehensive_analysis(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks: List[Tuple[int, int]] = (),
        *,
        blink_hint: Optional[bool] = None,
        ear_threshold: Optional[float] = None,
        ear_thresh: Optional[float] = None,  # alias passthrough
        iod_px: Optional[float] = None,
        tag: str = "default",
    ) -> Dict:
        """
        End-to-end pass that (a) uses the previous frame for temporal features and (b) updates history
        AFTER computing on the current frame. This avoids self-compare artifacts.

        `tag` helps disambiguate multiple callers in logs (e.g., "analyze_frame").
        """
        # Compute dynamic EAR threshold if not provided (alias-aware)
        if ear_threshold is None and ear_thresh is not None:
            ear_threshold = ear_thresh
        if ear_threshold is None:
            ear_threshold = _iod_min_threshold(iod_px)
        try:
            # 1) Use existing history (previous frame) for computations
            photo_res = self.detect_photo_attack(frame, face_bbox)
            live_res = self.detect_liveness(
                frame, face_bbox, landmarks,
                blink_hint=blink_hint,
                ear_threshold=ear_threshold,
                iod_px=iod_px,
                tag=tag,
            )

            # 2) Now update history with the current frame (for next call)
            self.add_frame(frame)

            risk_factors: List[Tuple[str, float]] = []
            if photo_res.get('is_photo_attack'):
                risk_factors.append(('Photo Attack', float(photo_res.get('confidence', 0.0))))
            if not live_res.get('is_live', False):
                risk_factors.append(('No Liveness', 1.0 - float(live_res.get('confidence', 0.0))))

            overall_risk = max((r for _, r in risk_factors), default=0.0)
            risk_level = 'HIGH' if overall_risk > 0.6 else 'MEDIUM' if overall_risk > 0.35 else 'LOW'
            is_authentic = (not photo_res.get('is_photo_attack', False)) and live_res.get('is_live', False) and overall_risk <= 0.35

            logger.info(f"[comprehensive:{tag}] risk={overall_risk:.2f} level={risk_level} factors={risk_factors}")

            return {
                'overall_risk': float(overall_risk),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'is_authentic': bool(is_authentic),
                'photo_attack': photo_res,
                'liveness': live_res,
            }
        except Exception as e:
            logger.exception("Error in comprehensive_analysis")
            return {'overall_risk': 1.0, 'risk_level': 'ERROR', 'is_authentic': False, 'error': str(e)}

    # Maintenance helpers
    def clear_history(self) -> None:
        self.frame_history.clear()
        self._recent_flow_medians.clear()
