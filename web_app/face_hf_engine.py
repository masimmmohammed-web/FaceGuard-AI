from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image

# Import the HF model utilities from images.final_deepfake_detection
from images.final_deepfake_detection import (
    load_model,
    predict_image,
    detect_faces_cv2,
    expand_box,
)


class HFDeepfakeEngine:
    """Thin wrapper that exposes a simple API for deepfake scoring.

    Methods accept OpenCV BGR images for seamless integration with Flask routes.
    Returns a dict with 'overall_risk' in [0,1] and a small details payload.
    """

    def __init__(self):
        self._processor = None
        self._model = None

    def ensure_loaded(self) -> None:
        if self._processor is None or self._model is None:
            self._processor, self._model = load_model()

    def score_frame(self, bgr_image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> Dict:
        """Score a frame or an optional face crop for deepfake risk.

        - If bbox is provided: use an expanded crop around the face and score it.
        - Always compute whole-frame score as a fallback.
        """
        self.ensure_loaded()

        # Convert to PIL
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Whole image prediction
        whole_probs = predict_image(pil_img, self._processor, self._model)
        p_fake_whole = float(whole_probs.get("Deepfake", 0.0))

        face_probs = None
        face_box = None

        if bbox is not None:
            x, y, w, h = bbox
            # Expand similar to final_deepfake_detection.expand_box semantics
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2, _ = expand_box((x1, y1, x2, y2, 1.0), pil_img.width, pil_img.height, margin=0.25)
            face_box = (x1, y1, x2, y2)
            crop = pil_img.crop(face_box)
            face_probs = predict_image(crop, self._processor, self._model)

        # Derive risk: prefer face score if available, else whole
        p_fake_face = float(face_probs.get("Deepfake", 0.0)) if face_probs else None
        overall_risk = p_fake_face if p_fake_face is not None else p_fake_whole

        return {
            "overall_risk": overall_risk,
            "details": {
                "whole": whole_probs,
                "face": face_probs,
                "face_box": face_box,
            },
            "engine": "prithivMLmods/Deepfake-vs-Real-8000",
        }


