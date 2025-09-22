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

### **Eye Tracking**
This file, `eye_tracker.py`, implements a **minimal eye tracking module** using [dlib](http://dlib.net/)’s 68-point facial landmark detector and a blink detection algorithm based on the Eye Aspect Ratio (EAR) and a custom “mEAR” (rotationally normalized EAR) metric.

### **Main Features**

**Class:** `EyeTracker`
- **Initialization:**  
  Loads dlib’s face detector and a shape predictor (68-point facial landmarks). You must supply a path to a trained dlib shape predictor model (e.g., `shape_predictor_68_face_landmarks.dat`).

- **Blink Detection Tuning:**  
  Sets thresholds for EAR/mEAR to classify blinks, consecutive frame requirement, and smoothing via exponential moving average (EMA).

---

### **API Methods**

#### 1. `detect_eyes(frame, face_bbox=None)`
- **Purpose:** Detects eyes in a given image (frame).
- **How:**  
  - Converts frame to grayscale.
  - Detects face, unless a bounding box is provided.
  - Finds 68 facial landmarks.
  - Extracts left and right eye landmarks (6 points each).
  - Calculates bounding boxes for each eye.
- **Returns:**  
  Dictionary containing:
  - Whether eyes were detected
  - Left/right eye bounding boxes and landmarks
  - Confidence score
  - Method used
  - All 68 facial landmarks

#### 2. `track_pupils(eye_data)`
- **Purpose:** Stub method — does not actually track pupils.
- **Returns:** Dictionary with default values for pupil detection and gaze (all set to `None` or `False`).

#### 3. `detect_blink(eye_data)`
- **Purpose:** Detects blinks using eye aspect ratio (EAR) and mEAR.
- **How:**  
  - Computes EAR and mEAR for both eyes.
  - Applies EMA smoothing to metrics.
  - Checks if EAR/mEAR fall below blink thresholds for consecutive frames.
  - If threshold met, blink is detected.
- **Returns:**  
  Dictionary containing:
  - Whether a blink was detected
  - Smoothed EAR/mEAR values
  - Blink count (1 if detected, else 0)
  - Confidence score
  - Thresholds used
  - Whether eyes are closed according to each metric

---

### **Key Algorithms**

- **EAR (Eye Aspect Ratio):**  
  Measures eye openness using distances between landmarks.
- **mEAR:**  
  Rotates eye landmarks to normalize for head tilt, then computes EAR.
- **EMA Smoothing:**  
  Reduces false positives by smoothing blink detection metrics across frames.

---

### **Usage**

Typical usage in an application would involve:
1. Initializing `EyeTracker` with the path to the shape predictor.
2. Calling `detect_eyes()` on each frame to get eye locations and landmarks.
3. Passing the result to `detect_blink()` to determine if a blink occurred.

---

**Summary:**  
This module provides a basic, real-time blink detection capability using facial landmarks, suitable for webcam-based attentiveness monitoring, fatigue detection, or simple human-computer interaction. It does **not** track gaze direction or pupil location (those are stubbed).
