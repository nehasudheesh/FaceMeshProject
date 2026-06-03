import cv2
import time
import mediapipe as mp
import pyautogui

# -----------------------------
# Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

# -----------------------------
# Blink control
# -----------------------------
blink_count = 0
eye_closed = False

prev_time = 0

# -----------------------------
# Eye indices
# -----------------------------
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_IRIS = 468

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # -----------------------------
        # Iris tracking
        # -----------------------------
        left_x = lm[LEFT_EYE_LEFT].x
        right_x = lm[LEFT_EYE_RIGHT].x
        iris_x = lm[LEFT_IRIS].x
        iris_y = lm[LEFT_IRIS].y

        eye_center_x = (left_x + right_x) / 2

        # Normalize
        norm_x = (iris_x - eye_center_x) / (right_x - left_x)

        # Map to screen
        screen_x = int(screen_w * iris_x)
        screen_y = int(screen_h * iris_y)

        # Move mouse
        pyautogui.moveTo(screen_x, screen_y, duration=0.05)

        # -----------------------------
        # Blink detection (click)
        # -----------------------------
        top = lm[159].y
        bottom = lm[145].y

        gap = bottom - top

        if gap < 0.008:
            if not eye_closed:
                blink_count += 1
                eye_closed = True

                # CLICK on blink
                pyautogui.click()

        else:
            eye_closed = False

        # -----------------------------
        # UI display
        # -----------------------------
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "MOUSE CONTROL ACTIVE", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Eye Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()