import cv2
import time
import mediapipe as mp

# -----------------------------
# Face Mesh Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh Processing
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # -----------------------------
        # Draw Face Contours
        # -----------------------------
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=1
            )
        )

        # -----------------------------
        # Blink Detection
        # -----------------------------
        landmarks = face_landmarks.landmark

        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y

        eye_gap = left_eye_bottom - left_eye_top

        # Adjust threshold if needed
        if eye_gap < 0.008:

            if not eye_closed:
                blink_count += 1
                eye_closed = True

        else:
            eye_closed = False

        # -----------------------------
        # Display Landmark Count
        # -----------------------------
        cv2.putText(
            frame,
            f"Points: {len(landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # -----------------------------
        # Display Blink Count
        # -----------------------------
        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # -----------------------------
    # FPS Calculation
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
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Show Window
    # -----------------------------
    cv2.imshow("Face Mesh Detection", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()