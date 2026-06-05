import cv2
import mediapipe as mp
import pyautogui
import time

# ----------------------------------
# Face Mesh Setup
# ----------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------------
# Screen Size
# ----------------------------------
screen_w, screen_h = pyautogui.size()

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# ----------------------------------
# Webcam
# ----------------------------------
cap = cv2.VideoCapture(0)

# ----------------------------------
# Blink Variables
# ----------------------------------
blink_count = 0
eye_closed = False

# ----------------------------------
# FPS
# ----------------------------------
prev_time = time.time()

# ----------------------------------
# Cursor Smoothing
# ----------------------------------
smooth_x = screen_w // 2
smooth_y = screen_h // 2

SMOOTHING = 0.35

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        landmarks = result.multi_face_landmarks[0].landmark

        # ----------------------------------
        # Eye Tracking
        # ----------------------------------

        left_corner = landmarks[33].x
        right_corner = landmarks[133].x

        iris_x = landmarks[468].x
        iris_y = landmarks[468].y

        eye_center = (left_corner + right_corner) / 2

        eye_width = right_corner - left_corner

        relative_x = (iris_x - eye_center) / eye_width

        # Amplify movement
        amplified_x = 0.5 + (relative_x * 3)

        amplified_x = max(0, min(1, amplified_x))
        iris_y = max(0, min(1, iris_y))

        target_x = int(amplified_x * screen_w)
        target_y = int(iris_y * screen_h)

        # ----------------------------------
        # Smoothing
        # ----------------------------------

        smooth_x = smooth_x + (target_x - smooth_x) * SMOOTHING
        smooth_y = smooth_y + (target_y - smooth_y) * SMOOTHING

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        # ----------------------------------
        # Blink Detection
        # ----------------------------------

        top = landmarks[159].y
        bottom = landmarks[145].y

        eye_gap = bottom - top

        if eye_gap < 0.008:

            if not eye_closed:

                blink_count += 1
                eye_closed = True

                pyautogui.click()

        else:
            eye_closed = False

        # ----------------------------------
        # Display
        # ----------------------------------

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "MOUSE CONTROL ACTIVE",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ----------------------------------
    # FPS
    # ----------------------------------

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Eye Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()