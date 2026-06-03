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
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# -----------------------------
# Blink Detection Variables
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Draw face mesh
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
        # Improved Eye Tracking
        # -----------------------------
        left_corner_x = landmarks[33].x
        right_corner_x = landmarks[133].x
        iris_x = landmarks[468].x

        # Eye center
        eye_center = (left_corner_x + right_corner_x) / 2

        # Normalize deviation
        offset = iris_x - eye_center

        # Scale factor
        eye_width = right_corner_x - left_corner_x
        normalized_offset = offset / eye_width

        eye_direction = "CENTER"

        if normalized_offset < -0.15:
            eye_direction = "LEFT"
        elif normalized_offset > 0.15:
            eye_direction = "RIGHT"

        # -----------------------------
        # Display Info
        # -----------------------------
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Eye: {eye_direction}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

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