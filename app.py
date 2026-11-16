
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
min_brush_size = 2
max_brush_size = 40
eraser_size = 35

backgrounds = [
    ("WHITE", (255, 255, 255)),
    ("BLACK", (0, 0, 0)),
    ("LIGHT BLUE", (173, 216, 230)),
    ("LIGHT GREEN", (144, 238, 144)),
    ("CREAM", (240, 248, 255)),
    ("PINK", (203, 192, 255))
]

background_index = 0

save_folder = os.path.abspath("saved_drawings")
os.makedirs(save_folder, exist_ok=True)

last_saved_path = None
recognized_text = "Press T for OCR"

undo_stack = []
redo_stack = []
max_history = 20
drawing_active = False


def reset_stroke():
    global drawing_active
    global prev_x, prev_y
    global smooth_x, smooth_y

    drawing_active = False
    prev_x, prev_y = None, None
    smooth_x, smooth_y = None, None


def remember_canvas():
    undo_stack.append(canvas.copy())

    if len(undo_stack) > max_history:
        undo_stack.pop(0)

    redo_stack.clear()


def select_color(index):
    global color_index, eraser

    color_index = index
    eraser = False
    reset_stroke()

    print(colors[index][0], "selected")


def select_background(index):
    global background_index

    background_index = index

    print("Background selected:", backgrounds[index][0])


def create_final_image(image):
    bg_color = backgrounds[background_index][1]

    output = np.full(
        image.shape,
        bg_color,
        dtype=np.uint8
    )

    mask = np.any(image != 0, axis=2)
    output[mask] = image[mask]

    return output


def save_canvas(image):
    global last_saved_path

    output = create_final_image(image)

    filename = os.path.join(
        save_folder,
        f"drawing_{int(time.time() * 1000)}.png"
    )

    if cv2.imwrite(filename, output):
        last_saved_path = filename

        print("PNG saved successfully:", filename)
        print("Background:", backgrounds[background_index][0])

        return filename

    print("PNG could not be saved!")
    return None


def export_jpg(image):
    output = create_final_image(image)

    filename = os.path.join(
        save_folder,
        f"drawing_{int(time.time() * 1000)}.jpg"
    )

    if cv2.imwrite(
        filename,
        output,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    ):
        print("JPG exported successfully:", filename)
    else:
        print("JPG export failed!")


def prepare_ocr_image(image, is_canvas=False):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if is_canvas:
        _, binary = cv2.threshold(
            gray,
            8,
            255,
            cv2.THRESH_BINARY
        )
    else:
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    if w < 30 or h < 15:
        return None

    padding_x = max(40, int(w * 0.10))
    padding_y = max(40, int(h * 0.30))

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(image.shape[1], x + w + padding_x)
    y2 = min(image.shape[0], y + h + padding_y)

    cropped = binary[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

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

    return processed


def recognize_handwriting(image, is_canvas=False):
    processed = prepare_ocr_image(
        image,
        is_canvas
    )

    if processed is None:
        return ""

    debug_path = os.path.join(
        save_folder,
        "ocr_input.png"
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

    print("Processed OCR image saved:", debug_path)

    return " ".join(text.split())


def recognize_latest_saved_drawing():
    global last_saved_path

    if last_saved_path and os.path.isfile(last_saved_path):
        image_path = last_saved_path
    else:
        files = [
            os.path.join(save_folder, name)
            for name in os.listdir(save_folder)
            if name.lower().endswith(".png")
            and name.lower().startswith("drawing_")
        ]

        if not files:
            print("No saved PNG found!")
            return ""

        image_path = max(
            files,
            key=os.path.getmtime
        )

        last_saved_path = image_path

    image = cv2.imread(image_path)

    if image is None:
        print("Could not open saved image!")
        return ""

    print("Reading saved drawing:", image_path)

    return recognize_handwriting(
        image,
        is_canvas=False
    )


print("AI VIRTUAL AIR CANVAS STARTED")
print("--------------------------------")
print("1-5: Drawing Colors")
print("6: White Background")
print("7: Black Background")
print("8: Light Blue Background")
print("9: Light Green Background")
print("0: Cream Background")
print("P: Pink Background")
print("[ / -: Smaller Brush")
print("] / =: Bigger Brush")
print("E: Eraser")
print("C: Clear")
print("S: Save PNG")
print("J: Export JPG")
print("T: OCR Current Canvas")
print("O: OCR Latest Saved PNG")
print("Z: Undo")
print("X: Redo")
print("Q: Quit")
print("--------------------------------")


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
                canvas = np.zeros(
                    (h, w, 3),
                    dtype=np.uint8
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp
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

                    reset_stroke()

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
                            reset_stroke()
                            print("ERASER selected")

                        elif 635 < x <= 735:

                            if np.any(canvas):
                                remember_canvas()
                                canvas[:] = 0
                                print("Canvas cleared")

                        elif 735 < x <= 835:

                            save_canvas(canvas)

                elif index_up and not middle_up:

                    if y > 110:

                        if not drawing_active:
                            remember_canvas()
                            drawing_active = True

                        if smooth_x is None:
                            smooth_x = x
                            smooth_y = y

                        smooth_x = int(
                            smoothing * x +
                            (1 - smoothing) * smooth_x
                        )

                        smooth_y = int(
                            smoothing * y +
                            (1 - smoothing) * smooth_y
                        )

                        if prev_x is not None and prev_y is not None:

                            if eraser:
                                draw_color = (0, 0, 0)
                                thickness = eraser_size
                            else:
                                draw_color = colors[color_index][1]
                                thickness = brush_size

                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (smooth_x, smooth_y),
                                draw_color,
                                thickness,
                                cv2.LINE_AA
                            )

                        else:

                            if eraser:
                                draw_color = (0, 0, 0)
                                thickness = eraser_size // 2
                            else:
                                draw_color = colors[color_index][1]
                                thickness = max(1, brush_size // 2)

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
                                max(2, brush_size // 2),
                                colors[color_index][1],
                                -1
                            )

                    else:
                        reset_stroke()

                else:
                    reset_stroke()

            else:
                reset_stroke()

            combined = cv2.add(frame, canvas)

            mode = (
                "ERASER"
                if eraser
                else colors[color_index][0]
            )

            cv2.putText(
                combined,
                f"BACKGROUND: {backgrounds[background_index][0]}",
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
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

            cv2.putText(
                combined,
                f"OCR: {recognized_text[:55]}",
                (15, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                combined,
                f"BRUSH SIZE: {brush_size}",
                (15, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            instructions = [
                "One Finger: Draw | Two Fingers: Select",
                "1-5: Colors | E: Eraser",
                "6-0: Backgrounds | P: Pink",
                "[: Smaller Brush | ]: Bigger Brush",
                "T: OCR Canvas | O: OCR Saved PNG",
                "S: Save PNG | J: Export JPG | C: Clear",
                "Z: Undo | X: Redo | Q: Quit"
            ]

            y_position = h - 155

            for instruction in instructions:

                cv2.putText(
                    combined,
                    instruction,
                    (15, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                y_position += 22

            cv2.imshow(
                "AI Virtual Air Canvas",
                combined
            )

            key = cv2.waitKey(1) & 0xFF

            if 65 <= key <= 90:
                key += 32

            if key in (ord('b'), ord('1')):
                select_color(0)

            elif key in (ord('g'), ord('2')):
                select_color(1)

            elif key in (ord('r'), ord('3')):
                select_color(2)

            elif key in (ord('y'), ord('4')):
                select_color(3)

            elif key in (ord('w'), ord('5')):
                select_color(4)

            elif key == ord('6'):
                select_background(0)

            elif key == ord('7'):
                select_background(1)

            elif key == ord('8'):
                select_background(2)

            elif key == ord('9'):
                select_background(3)

            elif key == ord('0'):
                select_background(4)

            elif key == ord('p'):
                select_background(5)

            elif key in (ord('['), ord('-')):
                brush_size = max(
                    min_brush_size,
                    brush_size - 2
                )
                reset_stroke()
                print("Brush size:", brush_size)

            elif key in (ord(']'), ord('='), ord('+')):
                brush_size = min(
                    max_brush_size,
                    brush_size + 2
                )
                reset_stroke()
                print("Brush size:", brush_size)

            elif key == ord('e'):
                eraser = True
                reset_stroke()
                print("ERASER selected")

            elif key == ord('c'):
                if np.any(canvas):
                    remember_canvas()

                canvas[:] = 0
                recognized_text = "Canvas cleared"
                reset_stroke()
                print("Canvas cleared")

            elif key == ord('z'):
                if undo_stack:
                    redo_stack.append(canvas.copy())

                    if len(redo_stack) > max_history:
                        redo_stack.pop(0)

                    canvas = undo_stack.pop()
                    reset_stroke()
                    print("Undo successful")
                else:
                    print("Nothing to undo")

            elif key == ord('x'):
                if redo_stack:
                    undo_stack.append(canvas.copy())

                    if len(undo_stack) > max_history:
                        undo_stack.pop(0)

                    canvas = redo_stack.pop()
                    reset_stroke()
                    print("Redo successful")
                else:
                    print("Nothing to redo")

            elif key == ord('s'):
                save_canvas(canvas)

            elif key == ord('j'):
                export_jpg(canvas)

            elif key == ord('t'):
                print("Recognizing current canvas...")

                try:
                    recognized_text = recognize_handwriting(
                        canvas,
                        is_canvas=True
                    )

                    if recognized_text:
                        print("Recognized Text:", recognized_text)
                    else:
                        recognized_text = "No text found"
                        print("No text found. Try writing clearly.")

                except Exception as error:
                    recognized_text = "OCR error"
                    print("OCR Error:", error)

            elif key == ord('o'):
                print("Recognizing latest saved PNG...")

                try:
                    recognized_text = recognize_latest_saved_drawing()

                    if recognized_text:
                        print("Saved PNG OCR Result:", recognized_text)
                    else:
                        recognized_text = "No text found"
                        print("No text found in saved PNG.")

                except Exception as error:
                    recognized_text = "Saved PNG OCR error"
                    print("Saved PNG OCR Error:", error)

            elif key == ord('q') or key == 27:
                print("Closing Air Canvas...")
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Air Canvas closed successfully!")
