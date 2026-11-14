
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera could not be opened!")
    cap.release()
    raise SystemExit

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

prev_x, prev_y = None, None
smooth_x, smooth_y = None, None

smoothing = 0.65
brush_size = 12
eraser_size = 35

os.makedirs("saved_drawings", exist_ok=True)
save_folder = os.path.abspath("saved_drawings")

recognized_text = "Press T to recognize text"


def select_color(index):

    global color_index, eraser

    color_index = index
    eraser = False

    print(colors[index][0], "selected")


def save_canvas(image):

    filename = os.path.join(
        save_folder,
        f"drawing_{int(time.time() * 1000)}.png"
    )

    saved = cv2.imwrite(filename, image)

    if saved:
        print("Drawing saved successfully:", filename)
    else:
        print("Drawing could not be saved!")



def recognize_handwriting(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 8, 255, cv2.THRESH_BINARY
    )

    pixel_count = cv2.countNonZero(binary)

    if pixel_count < 250:
        return ""

    coords = cv2.findNonZero(binary)

    if coords is None:
        return ""

    x, y, w, h = cv2.boundingRect(coords)

    if w < 30 or h < 15:
        return ""

    padding_x = max(40, int(w * 0.10))
    padding_y = max(40, int(h * 0.30))

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image.shape[1], x + w + padding_x)
    y2 = min(image.shape[0], y + h + padding_y)

    cropped = binary[y1:y2, x1:x2]

    cropped = cv2.morphologyEx(
        cropped,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8)
    )

    scale = max(
        3,
        min(6, int(180 / max(cropped.shape[0], 1)))
    )

    enlarged = cv2.resize(
        cropped,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    processed = cv2.bitwise_not(enlarged)

    processed = cv2.copyMakeBorder(
        processed,
        50, 50, 50, 50,
        cv2.BORDER_CONSTANT,
        value=255
    )

    debug_path = os.path.join(
        save_folder, "ocr_input.png"
    )

    cv2.imwrite(debug_path, processed)

    text = pytesseract.image_to_string(
        processed,
        lang="eng",
        config=(
            "--oem 3 --psm 8 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
    )

    print("OCR input image saved:", debug_path)

    return " ".join(text.split())



print("AI Virtual Air Canvas Started!")
print("B/G/R/Y/W or 1/2/3/4/5: Select colors")
print("E: Eraser | C: Clear | S: Save | T: Recognize | Q: Quit")

try:

    with HandLandmarker.create_from_options(options) as detector:

        start_time = time.monotonic()

        while True:

            success, frame = cap.read()

            if not success:
                print("Camera frame not received!")
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

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image, timestamp
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (w, 110),
                (35, 35, 35),
                -1
            )

            cv2.putText(
                frame,
                "AI VIRTUAL AIR CANVAS",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            for i, (name, color) in enumerate(colors):

                x1 = 10 + i * 105

                cv2.rectangle(
                    frame,
                    (x1, 42),
                    (x1 + 95, 85),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    name,
                    (x1 + 5, 69),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )

                if color_index == i and not eraser:

                    cv2.rectangle(
                        frame,
                        (x1, 42),
                        (x1 + 95, 85),
                        (255, 255, 255),
                        3
                    )

            cv2.putText(
                frame,
                "ERASER",
                (535, 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "CLEAR",
                (640, 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SAVE",
                (735, 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

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

                    prev_x, prev_y = None, None
                    smooth_x, smooth_y = None, None

                    if 35 <= y <= 100:

                        selected = None

                        for i in range(len(colors)):

                            x1 = 10 + i * 105

                            if x1 <= x <= x1 + 95:
                                selected = i
                                break

                        if selected is not None:
                            select_color(selected)

                        elif 535 <= x <= 635:
                            eraser = True

                        elif 635 < x <= 735:
                            canvas[:] = 0
                            print("Canvas cleared")

                        elif 735 < x <= 835:
                            save_canvas(canvas)

                elif index_up and not middle_up:

                    if y > 110:

                        if smooth_x is None:
                            smooth_x, smooth_y = x, y

                        smooth_x = int(
                            smoothing * x +
                            (1 - smoothing) * smooth_x
                        )

                        smooth_y = int(
                            smoothing * y +
                            (1 - smoothing) * smooth_y
                        )

                        if prev_x is not None and prev_y is not None:

                            draw_color = (
                                (0, 0, 0)
                                if eraser
                                else colors[color_index][1]
                            )

                            thickness = (
                                eraser_size
                                if eraser
                                else brush_size
                            )

                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (smooth_x, smooth_y),
                                draw_color,
                                thickness,
                                cv2.LINE_AA
                            )

                        else:

                            draw_color = (
                                (0, 0, 0)
                                if eraser
                                else colors[color_index][1]
                            )

                            thickness = (
                                eraser_size // 2
                                if eraser
                                else brush_size // 2
                            )

                            cv2.circle(
                                canvas,
                                (smooth_x, smooth_y),
                                thickness,
                                draw_color,
                                -1
                            )

                        prev_x, prev_y = smooth_x, smooth_y

                        if eraser:

                            cv2.circle(
                                frame,
                                (smooth_x, smooth_y),
                                eraser_size,
                                (255, 255, 255),
                                2
                            )

                        else:

                            cv2.circle(
                                frame,
                                (smooth_x, smooth_y),
                                6,
                                colors[color_index][1],
                                -1
                            )

                else:

                    prev_x, prev_y = None, None
                    smooth_x, smooth_y = None, None

            else:

                prev_x, prev_y = None, None
                smooth_x, smooth_y = None, None

            combined = cv2.add(frame, canvas)

            mode = (
                "ERASER"
                if eraser
                else colors[color_index][0]
            )

            cv2.putText(
                combined,
                f"MODE: {mode}",
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            display_text = recognized_text[:55]

            if len(recognized_text) > 55:
                display_text += "..."

            cv2.putText(
                combined,
                f"OCR: {display_text}",
                (15, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                combined,
                "One Finger: Draw | Two Fingers: Select",
                (15, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.putText(
                combined,
                "B/G/R/Y/W or 1/2/3/4/5: Colors | E: Eraser",
                (15, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            cv2.putText(
                combined,
                "T: Recognize | S: Save | C: Clear | Q: Quit",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "AI Virtual Air Canvas",
                combined
            )

            key = cv2.waitKey(1) & 0xFF

            if 65 <= key <= 90:
                key += 32

            if key == ord('b') or key == ord('1'):
                select_color(0)

            elif key == ord('g') or key == ord('2'):
                select_color(1)

            elif key == ord('r') or key == ord('3'):
                select_color(2)

            elif key == ord('y') or key == ord('4'):
                select_color(3)

            elif key == ord('w') or key == ord('5'):
                select_color(4)

            elif key == ord('e'):
                eraser = True
                print("ERASER selected")

            elif key == ord('c'):
                canvas[:] = 0
                recognized_text = "Canvas cleared"
                print("Canvas cleared")

            elif key == ord('s'):
                save_canvas(canvas)

            elif key == ord('t'):

                print("Recognizing handwriting...")

                try:

                    recognized_text = recognize_handwriting(canvas)

                    if recognized_text:
                        print("Recognized Text:", recognized_text)
                    else:
                        recognized_text = "No text found"
                        print("No text found. Try writing clearly.")

                except Exception as error:

                    recognized_text = "OCR error - check terminal"
                    print("OCR Error:", error)

            elif key == ord('q') or key == 27:
                print("Closing Air Canvas...")
                break

finally:

    cap.release()
    cv2.destroyAllWindows()
    print("Air Canvas closed successfully!")