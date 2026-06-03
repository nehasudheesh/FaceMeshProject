import cv2
import time
import mediapipe as mp
import numpy as np

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
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# -----------------------------
# Landmark indices
# -----------------------------
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_IRIS_CENTER = 468

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_FACE = 234
RIGHT_FACE = 454

# -----------------------------
# Blink variables
# -----------------------------
blink_count = 0
eye_closed = False

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        mp_drawing.draw_landmarks(
            frame,
            result.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            drawing_spec,
            drawing_spec
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
        else:
            eye_closed = False

        # -----------------------------
        # Eye Tracking (stable version)
        # -----------------------------
        left_x = landmarks[33].x
        right_x = landmarks[133].x
        iris_x = landmarks[468].x

        eye_center = (left_x + right_x) / 2
        offset = iris_x - eye_center
        eye_width = right_x - left_x

        normalized_offset = offset / eye_width

        eye_direction = "CENTER"

        if normalized_offset < -0.15:
            eye_direction = "LEFT"
        elif normalized_offset > 0.15:
            eye_direction = "RIGHT"

        # -----------------------------
        # Simple Head Pose (stable fallback)
        # -----------------------------
        nose_y = landmarks[NOSE_TIP].y
        chin_y = landmarks[CHIN].y

        head_direction = "CENTER"

        if nose_y < chin_y - 0.08:
            head_direction = "HEAD UP"
        elif nose_y > chin_y - 0.02:
            head_direction = "HEAD DOWN"

        # -----------------------------
        # Display
        # -----------------------------
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Eye: {eye_direction}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Head: {head_direction}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # -----------------------------
    # FPS Counter
    # -----------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Mesh Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()