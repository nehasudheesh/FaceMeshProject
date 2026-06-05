import cv2
import time
import mediapipe as mp

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
# Eye Tracking Landmarks
# -----------------------------
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_IRIS = 468

# -----------------------------
# Head Pose Landmarks
# -----------------------------
NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454

# -----------------------------
# Blink Variables
# -----------------------------
blink_count = 0
eye_closed = False

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

    h, w, _ = frame.shape

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
        # Eye Tracking
        # -----------------------------
        left_x = landmarks[LEFT_EYE_LEFT].x
        right_x = landmarks[LEFT_EYE_RIGHT].x
        iris_x = landmarks[LEFT_IRIS].x

        eye_center = (left_x + right_x) / 2
        eye_width = right_x - left_x

        offset = iris_x - eye_center
        ratio = offset / eye_width

        if ratio < -0.15:
            eye_direction = "LEFT"
        elif ratio > 0.15:
            eye_direction = "RIGHT"
        else:
            eye_direction = "CENTER"

        # -----------------------------
        # Head Pose
        # -----------------------------
        nose_x = landmarks[NOSE_TIP].x
        left_face_x = landmarks[LEFT_FACE].x
        right_face_x = landmarks[RIGHT_FACE].x

        face_center = (left_face_x + right_face_x) / 2

        head_offset = nose_x - face_center

        if head_offset < -0.03:
            head_direction = "LEFT"
        elif head_offset > 0.03:
            head_direction = "RIGHT"
        else:
            head_direction = "CENTER"

        nose_y = landmarks[NOSE_TIP].y

        if nose_y < 0.40:
            head_direction = "UP"
        elif nose_y > 0.60:
            head_direction = "DOWN"

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
            f"Head: {head_direction}",
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

    cv2.imshow("FaceMesh V4 - Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()