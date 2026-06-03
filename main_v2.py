import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect Face Landmarks
    result = detector.detect(mp_image)

    # Draw Landmarks
    if result.face_landmarks:

        h, w, _ = frame.shape

        landmarks = result.face_landmarks[0]
       
        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

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

    # Show Frame
    cv2.imshow("Face Mesh Detection", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()