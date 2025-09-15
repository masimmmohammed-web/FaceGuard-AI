import cv2
import dlib
import imutils
import numpy as np
from math import atan2, cos, sin
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# ---------- Camera ----------
cam = cv2.VideoCapture(0)  # change to 1/2 if needed
if not cam.isOpened():
    raise RuntimeError("Could not open webcam. Try cam index 1 or 2.")

# ---------- EAR helpers ----------
def ear(eye):
    """
    Standard Eye Aspect Ratio (Soukupová & Čech 2016)
    eye: np.ndarray of shape (6, 2) in order [p1..p6]
    """
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h  = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def rotate_points(points, angle, center):
    """
    Rotate an array of (x,y) points by 'angle' radians around 'center'.
    """
    c, s = cos(angle), sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (points - center) @ R.T + center

def modified_ear_pose_normalized(eye):
    """
    Pose-normalized EAR (mEAR):
    1) compute corner-to-corner angle
    2) rotate eye to make that baseline horizontal
    3) compute standard EAR in the rotated frame
    """
    p0, p3 = eye[0], eye[3]
    angle = atan2(p3[1] - p0[1], p3[0] - p0[0])  # roll/tilt of the eye line
    center = eye.mean(axis=0)
    eye_rot = rotate_points(eye, -angle, center)  # unroll
    return ear(eye_rot)

# ---------- Config ----------
blink_thresh_ear  = 0.22      # typical: 0.18–0.25 (tune)
blink_thresh_mear = 0.23      # mEAR may sit a hair higher; tune in your setup
succ_frame        = 2
count_frame       = 0

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ---------- Models ----------
detector = dlib.get_frontal_face_detector()
predictor_path = "C:/Users/masim/OneDrive/Desktop/FinalProject/shape_predictor_68_face_landmarks.dat"
landmark_predict = dlib.shape_predictor(predictor_path)

# smoothing (EMA) for stability
alpha = 0.4
ema_ear  = None
ema_mear = None

# FPS
t0, frames = time.time(), 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=720)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces:
        shape = landmark_predict(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye  = shape[L_start:L_end]
        right_eye = shape[R_start:R_end]

        # --- compute EAR + mEAR for both eyes and average ---
        l_ear  = ear(left_eye)
        r_ear  = ear(right_eye)
        l_mear = modified_ear_pose_normalized(left_eye)
        r_mear = modified_ear_pose_normalized(right_eye)

        EAR  = (l_ear + r_ear) / 2.0
        MEAR = (l_mear + r_mear) / 2.0

        # --- exponential moving average (optional but recommended) ---
        ema_ear  = EAR  if ema_ear  is None else (alpha*EAR  + (1-alpha)*ema_ear)
        ema_mear = MEAR if ema_mear is None else (alpha*MEAR + (1-alpha)*ema_mear)

        # --- draw landmarks (small circles) ---
        for (x, y) in np.vstack([left_eye, right_eye]):
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)  # yellow

        # --- draw smooth eye outlines (convex hulls) ---
        left_hull  = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_hull],  -1, (255, 0, 0), 1)      # blue
        cv2.drawContours(frame, [right_hull], -1, (255, 0, 0), 1)

        # --- blink logic (use mEAR primarily, fallback to EAR) ---
        closed_mear = (ema_mear if ema_mear is not None else MEAR) < blink_thresh_mear
        closed_ear  = (ema_ear  if ema_ear  is not None else EAR)  < blink_thresh_ear

        if closed_mear or closed_ear:
            count_frame += 1
        else:
            if count_frame >= succ_frame:
                cv2.putText(frame, "Blink Detected", (20, 42),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            count_frame = 0

        # --- display metrics ---
        cv2.putText(frame, f"EAR:  {EAR:.3f}  (EMA {ema_ear:.3f})" if ema_ear else f"EAR:  {EAR:.3f}",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)
        cv2.putText(frame, f"mEAR: {MEAR:.3f} (EMA {ema_mear:.3f})" if ema_mear else f"mEAR: {MEAR:.3f}",
                    (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)

        # optional: face box
        cv2.rectangle(frame, (face.left(), face.top()),
                      (face.right(), face.bottom()), (0, 200, 0), 1)

    # FPS (approx)
    frames += 1
    if frames % 10 == 0:
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Live Blink Detection (Eye Markers + mEAR)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
