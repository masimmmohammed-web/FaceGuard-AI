The file `src/anti_spoofing.py` implements an **anti-spoofing module** for face authentication systems. Its main purpose is to **detect photo, video, and mask attacks** (i.e., attempts to bypass facial recognition using images, screens, or 3D masks) and to **ensure liveness**—that is, to confirm that a real, live person is present.

### Key Features

- **Photo Attack Detection**
  - Uses spatial signals (image sharpness, edge density, color and texture variance, saturation/brightness, screen aspect ratio) and temporal signals (frame-to-frame stillness) to identify static images or screen presentations being used as spoofing attacks.
  - Returns a confidence score and metrics for the likelihood of a photo attack.

- **Liveness Detection**
  - Verifies that the subject is live by checking for **blinks** (using eye landmarks or external blink detectors), **micro-movements** (subtle natural motion between frames), and texture/color variety typical of real skin.
  - Returns a confidence score and metrics indicating whether the subject is likely live.

- **Comprehensive Analysis**
  - Combines both photo attack and liveness detection for a risk assessment: computes risk level (LOW/MEDIUM/HIGH) and determines if the subject is authentic.
  - Updates its internal frame history after analysis to avoid false positives from comparing frames to themselves.

- **Robustness and Safety**
  - Handles missing dependencies gracefully (optional utilities for optical flow and blinking).
  - Bounded history (memory-safe), sensible thresholds, and error handling with logging.
  - Utility functions for safe bounding box handling, fast frame duplication checks, and eye aspect ratio calculations.

- **Usage**
  - Designed to be used in a face authentication pipeline, where each incoming video frame is analyzed for spoofing risks and liveness cues.

### Example (Simplified Usage)
```python
detector = AntiSpoofingDetector()
detector.add_frame(current_frame)
photo_results = detector.detect_photo_attack(current_frame, face_bbox)
live_results = detector.detect_liveness(current_frame, face_bbox, landmarks)
analysis = detector.comprehensive_analysis(current_frame, face_bbox, landmarks)
```

### Summary

**This module helps protect face recognition systems from being tricked by photos, videos, or masks, and ensures that the face detected is a real, live person—making authentication more secure.**
