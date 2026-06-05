import cv2
import time
import mediapipe as mp
import pyautogui

# -----------------------------
# Face Mesh Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Screen Setup
# -----------------------------
screen_w, screen_h = pyautogui.size()

# Safety
pyautogui.FAILSAFE = False

# -----------------------------
# Eye Tracking Landmarks
# -----------------------------
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_IRIS = 468

# -----------------------------
# Blink Variables
# -----------------------------
blink_count = 0
eye_closed = False

# -----------------------------
# Cursor Smoothing
# -----------------------------
smooth_x = screen_w // 2
smooth_y = screen_h // 2

SMOOTHING = 0.35

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # Draw Face Mesh
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION
        )

        landmarks = face_landmarks.landmark

        # -----------------------------
        # Eye Tracking
        # -----------------------------
        left_x = landmarks[LEFT_EYE_LEFT].x
        right_x = landmarks[LEFT_EYE_RIGHT].x
        iris_x = landmarks[LEFT_IRIS].x
        iris_y = landmarks[LEFT_IRIS].y

        eye_center = (left_x + right_x) / 2
        eye_width = right_x - left_x

        relative_x = (iris_x - eye_center) / eye_width

        # Amplify eye movement
        amplified_x = 0.5 + (relative_x * 3)

        amplified_x = max(0, min(1, amplified_x))
        iris_y = max(0, min(1, iris_y))

        target_x = int(amplified_x * screen_w)
        target_y = int(iris_y * screen_h)

        # -----------------------------
        # Smooth Mouse Movement
        # -----------------------------
        smooth_x += (target_x - smooth_x) * SMOOTHING
        smooth_y += (target_y - smooth_y) * SMOOTHING

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        # -----------------------------
        # Blink Detection
        # -----------------------------
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y

        eye_gap = left_eye_bottom - left_eye_top

        if eye_gap < 0.008:

            if not eye_closed:

                blink_count += 1
                eye_closed = True

                # Mouse Click
                pyautogui.click()

        else:
            eye_closed = False

        # -----------------------------
        # Eye Direction Display
        # -----------------------------
        if relative_x < -0.15:
            eye_direction = "LEFT"
        elif relative_x > 0.15:
            eye_direction = "RIGHT"
        else:
            eye_direction = "CENTER"

        # -----------------------------
        # Display Information
        # -----------------------------
        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eye: {eye_direction}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "MOUSE CONTROL ACTIVE",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # FPS Counter
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("FaceMesh V5 - Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()