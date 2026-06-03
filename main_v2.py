import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# Face Outline Connections
# ----------------------------
FACE_OUTLINE = [
    (10, 338), (338, 297), (297, 332),
    (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323),
    (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149),
    (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93),
    (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103),
    (103, 67), (67, 109), (109, 10)
]

# ----------------------------
# Draw Connection Function
# ----------------------------
def draw_connections(frame, landmarks, connections, w, h):

    for start_idx, end_idx in connections:

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)

        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            1
        )

# ----------------------------
# Load Face Landmarker Model
# ----------------------------
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)

p_time = 0

# ----------------------------
# Main Loop
# ----------------------------
while True:

    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:

        h, w, _ = frame.shape

        landmarks = result.face_landmarks[0]

        # Draw Face Outline
        draw_connections(
            frame,
            landmarks,
            FACE_OUTLINE,
            w,
            h
        )

        # Draw Landmarks
        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        # Landmark Count
        cv2.putText(
            frame,
            f"Points: {len(landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    # FPS Calculation
    c_time = time.time()

    fps = 1 / (c_time - p_time) if p_time != 0 else 0

    p_time = c_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Mesh Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()