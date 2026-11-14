
import cv2
import mediapipe as mp
import numpy as np
import time
import os

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
canvas = None

colors = [
    ("BLUE", (255, 0, 0)),
    ("GREEN", (0, 255, 0)),
    ("RED", (0, 0, 255)),
    ("YELLOW", (0, 255, 255)),
    ("WHITE", (255, 255, 255))
]

color_index = 0
eraser = False
prev_x, prev_y = 0, 0
brush_size = 8
eraser_size = 35

os.makedirs("saved_drawings", exist_ok=True)

with HandLandmarker.create_from_options(options) as detector:

    start_time = time.monotonic()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp)

        cv2.rectangle(frame, (0, 0), (w, 110), (35, 35, 35), -1)

        cv2.putText(frame, "AIR CANVAS", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        for i, (name, color) in enumerate(colors):
            x1 = 10 + i * 105
            cv2.rectangle(frame, (x1, 42), (x1 + 95, 85), color, -1)
            cv2.putText(frame, name, (x1 + 5, 69),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1)

        cv2.putText(frame, "ERASER", (535, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        cv2.putText(frame, "CLEAR", (640, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        cv2.putText(frame, "SAVE", (735, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        drawing_color = colors[color_index][1]

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            index_tip = hand[8]
            index_pip = hand[6]
            middle_tip = hand[12]
            middle_pip = hand[10]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            if index_up and middle_up:

                prev_x, prev_y = 0, 0

                if 35 <= y <= 100:

                    if 10 <= x <= 535:
                        selected = (x - 10) // 105

                        if 0 <= selected < len(colors):
                            color_index = selected
                            eraser = False

                    elif 535 < x <= 635:
                        eraser = True

                    elif 635 < x <= 735:
                        canvas[:] = 0

                    elif 735 < x <= 835:
                        filename = f"saved_drawings/drawing_{int(time.time())}.png"
                        cv2.imwrite(filename, canvas)
                        print("Drawing saved:", filename)

            elif index_up and not middle_up:

                if y > 110:

                    if prev_x != 0 and prev_y != 0:

                        if eraser:
                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (x, y),
                                (0, 0, 0),
                                eraser_size
                            )

                        else:
                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (x, y),
                                drawing_color,
                                brush_size
                            )

                    prev_x, prev_y = x, y

                    if eraser:
                        cv2.circle(
                            frame, (x, y), eraser_size,
                            (255, 255, 255), 2
                        )
                    else:
                        cv2.circle(
                            frame, (x, y), 8,
                            drawing_color, -1
                        )

            else:
                prev_x, prev_y = 0, 0

        else:
            prev_x, prev_y = 0, 0

        combined = cv2.add(frame, canvas)

        mode = "ERASER" if eraser else colors[color_index][0]

        cv2.putText(
            combined,
            f"MODE: {mode}",
            (15, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            combined,
            "One Finger: Draw | Two Fingers: Select",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("AI Virtual Air Canvas", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            color_index = 0
            eraser = False

        elif key == ord('g'):
            color_index = 1
            eraser = False

        elif key == ord('r'):
            color_index = 2
            eraser = False

        elif key == ord('y'):
            color_index = 3
            eraser = False

        elif key == ord('w'):
            color_index = 4
            eraser = False

        elif key == ord('e'):
            eraser = True

        elif key == ord('c'):
            canvas[:] = 0

        elif key == ord('s'):
            filename = f"saved_drawings/drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print("Drawing saved:", filename)

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
