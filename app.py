
import cv2
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

with HandLandmarker.create_from_options(options) as detector:

    start_time = time.monotonic()

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                points = []

                for landmark in hand:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))

                for a, b in [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17)
                ]:
                    cv2.line(frame, points[a], points[b], (0,255,0), 2)

                for point in points:
                    cv2.circle(frame, point, 5, (0,0,255), -1)

        cv2.imshow("AI Virtual Air Canvas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
