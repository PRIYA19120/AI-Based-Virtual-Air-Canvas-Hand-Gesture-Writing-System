
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

backgrounds = [
    ("WHITE", (255, 255, 255)),
    ("BLACK", (0, 0, 0)),
    ("LIGHT BLUE", (173, 216, 230)),
    ("LIGHT GREEN", (144, 238, 144)),
    ("CREAM", (240, 248, 255)),
    ("PINK", (203, 192, 255))
]

background_index = 0

prev_x, prev_y = None, None
smooth_x, smooth_y = None, None

smoothing = 0.65
brush_size = 12
min_brush_size = 2
max_brush_size = 40
eraser_size = 35

save_folder = os.path.abspath("saved_drawings")
os.makedirs(save_folder, exist_ok=True)

last_saved_path = None
recognized_text = "Press T for OCR"
status_message = "Welcome to Air Canvas"

undo_stack = []
redo_stack = []
max_history = 20
drawing_active = False

toolbar_height = 155


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
    global color_index, eraser, status_message

    color_index = index
    eraser = False
    reset_stroke()

    status_message = colors[index][0] + " selected"
    print(status_message)


def select_background(index):
    global background_index, status_message

    background_index = index
    status_message = backgrounds[index][0] + " background selected"

    print(status_message)


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
    global last_saved_path, status_message

    output = create_final_image(image)

    filename = os.path.join(
        save_folder,
        f"drawing_{int(time.time() * 1000)}.png"
    )

    if cv2.imwrite(filename, output):
        last_saved_path = filename
        status_message = "PNG saved successfully"

        print("PNG saved:", filename)
        return filename

    status_message = "PNG save failed"
    return None


def export_jpg(image):
    global status_message

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
        status_message = "JPG exported successfully"
        print("JPG exported:", filename)
    else:
        status_message = "JPG export failed"


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
        config="--oem 3 --psm 8"
    )

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
            return ""

        image_path = max(
            files,
            key=os.path.getmtime
        )

        last_saved_path = image_path

    image = cv2.imread(image_path)

    if image is None:
        return ""

    return recognize_handwriting(
        image,
        is_canvas=False
    )


def undo_canvas():
    global canvas, status_message

    if undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()
        reset_stroke()
        status_message = "Undo successful"
    else:
        status_message = "Nothing to undo"


def redo_canvas():
    global canvas, status_message

    if redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()
        reset_stroke()
        status_message = "Redo successful"
    else:
        status_message = "Nothing to redo"


def clear_canvas():
    global canvas, recognized_text, status_message

    if np.any(canvas):
        remember_canvas()

    canvas[:] = 0
    recognized_text = "Canvas cleared"
    reset_stroke()
    status_message = "Canvas cleared"


def draw_button(frame, x1, y1, x2, y2, label, color, selected=False):
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    border_color = (255, 255, 255) if selected else (100, 100, 100)
    border_size = 3 if selected else 1

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        border_color,
        border_size
    )

    text_size = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        1
    )[0]

    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0) if color != (0, 0, 0) else (255, 255, 255),
        1
    )


def draw_toolbar(frame):
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], toolbar_height),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        frame,
        "AI VIRTUAL AIR CANVAS",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    for i, (name, color) in enumerate(colors):
        x1 = 10 + i * 92

        draw_button(
            frame,
            x1, 38,
            x1 + 82, 75,
            name,
            color,
            color_index == i and not eraser
        )

    tools = [
        ("ERASER", (180, 180, 180)),
        ("CLEAR", (80, 80, 220)),
        ("SAVE", (80, 180, 80)),
        ("JPG", (100, 160, 220)),
        ("UNDO", (180, 140, 80)),
        ("REDO", (180, 140, 80)),
        ("OCR", (100, 200, 200)),
        ("SAVED OCR", (100, 200, 200))
    ]

    tool_positions = {}

    for i, (label, color) in enumerate(tools):
        x1 = 475 + i * 98

        draw_button(
            frame,
            x1, 38,
            x1 + 88, 75,
            label,
            color,
            eraser if label == "ERASER" else False
        )

        tool_positions[label] = (
            x1, 38, x1 + 88, 75
        )

    cv2.putText(
        frame,
        "BACKGROUND:",
        (12, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    for i, (name, color) in enumerate(backgrounds):
        x1 = 125 + i * 112

        draw_button(
            frame,
            x1, 88,
            x1 + 102, 121,
            name,
            color,
            background_index == i
        )

    draw_button(
        frame, 805, 88, 875, 121,
        "BRUSH -", (160, 160, 160)
    )

    draw_button(
        frame, 882, 88, 952, 121,
        "BRUSH +", (160, 160, 160)
    )

    cv2.putText(
        frame,
        f"SIZE: {brush_size}",
        (970, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        status_message[:45],
        (15, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
        1
    )

    return tool_positions


def mouse_callback(event, x, y, flags, param):
    global eraser, brush_size
    global recognized_text, status_message

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if y < 38 or y > 121:
        return

    if 38 <= y <= 75:

        for i in range(len(colors)):
            x1 = 10 + i * 92

            if x1 <= x <= x1 + 82:
                select_color(i)
                return

        tool_positions = [
            ("ERASER", 475),
            ("CLEAR", 573),
            ("SAVE", 671),
            ("JPG", 769),
            ("UNDO", 867),
            ("REDO", 965),
            ("OCR", 1063),
            ("SAVED OCR", 1161)
        ]

        for label, x1 in tool_positions:

            if x1 <= x <= x1 + 88:

                if label == "ERASER":
                    eraser = True
                    reset_stroke()
                    status_message = "Eraser selected"

                elif label == "CLEAR":
                    clear_canvas()

                elif label == "SAVE":
                    save_canvas(canvas)

                elif label == "JPG":
                    export_jpg(canvas)

                elif label == "UNDO":
                    undo_canvas()

                elif label == "REDO":
                    redo_canvas()

                elif label == "OCR":
                    try:
                        recognized_text = recognize_handwriting(
                            canvas,
                            is_canvas=True
                        )

                        status_message = (
                            "OCR: " + recognized_text[:35]
                            if recognized_text
                            else "No text found"
                        )

                    except Exception as error:
                        status_message = "OCR error"
                        print(error)

                elif label == "SAVED OCR":
                    try:
                        recognized_text = recognize_latest_saved_drawing()

                        status_message = (
                            "Saved OCR: " + recognized_text[:30]
                            if recognized_text
                            else "No saved text found"
                        )

                    except Exception as error:
                        status_message = "Saved OCR error"
                        print(error)

                return

    if 88 <= y <= 121:

        for i in range(len(backgrounds)):
            x1 = 125 + i * 112

            if x1 <= x <= x1 + 102:
                select_background(i)
                return

        if 805 <= x <= 875:
            brush_size = max(
                min_brush_size,
                brush_size - 2
            )
            status_message = f"Brush size: {brush_size}"
            reset_stroke()

        elif 882 <= x <= 952:
            brush_size = min(
                max_brush_size,
                brush_size + 2
            )
            status_message = f"Brush size: {brush_size}"
            reset_stroke()


print("AI VIRTUAL AIR CANVAS STARTED")
print("--------------------------------")
print("Mouse toolbar: Click buttons on screen")
print("B/G/R/Y/W: Colors")
print("1-5: Colors")
print("6-0: Backgrounds | P: Pink")
print("E: Eraser | C: Clear")
print("[ / -: Smaller Brush")
print("] / =: Bigger Brush")
print("S: Save PNG | J: Export JPG")
print("T: Current Canvas OCR")
print("O: Saved PNG OCR")
print("Z: Undo | X: Redo")
print("Q: Quit")
print("--------------------------------")


try:
    cv2.namedWindow("AI Virtual Air Canvas")
    cv2.setMouseCallback(
        "AI Virtual Air Canvas",
        mouse_callback
    )

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

                elif index_up and not middle_up:

                    if y > toolbar_height:

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
                                thickness = eraser_size // 2
                                draw_color = (0, 0, 0)
                            else:
                                thickness = max(1, brush_size // 2)
                                draw_color = colors[color_index][1]

                            cv2.circle(
                                canvas,
                                (smooth_x, smooth_y),
                                thickness,
                                draw_color,
                                -1
                            )

                        prev_x, prev_y = smooth_x, smooth_y

                    else:
                        reset_stroke()

                else:
                    reset_stroke()

            else:
                reset_stroke()

            combined = cv2.add(
                frame,
                canvas
            )

            draw_toolbar(combined)

            cv2.putText(
                combined,
                f"MODE: {'ERASER' if eraser else colors[color_index][0]}",
                (15, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                combined,
                f"OCR: {recognized_text[:55]}",
                (15, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1
            )

            cv2.putText(
                combined,
                "One Finger: Draw | Two Fingers: Stop Drawing",
                (15, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                combined,
                "Mouse: Toolbar | Keyboard: Shortcuts | Q: Quit",
                (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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

            elif key in (ord(']'), ord('='), ord('+')):
                brush_size = min(
                    max_brush_size,
                    brush_size + 2
                )
                reset_stroke()

            elif key == ord('e'):
                eraser = True
                reset_stroke()
                status_message = "Eraser selected"

            elif key == ord('c'):
                clear_canvas()

            elif key == ord('z'):
                undo_canvas()

            elif key == ord('x'):
                redo_canvas()

            elif key == ord('s'):
                save_canvas(canvas)

            elif key == ord('j'):
                export_jpg(canvas)

            elif key == ord('t'):
                try:
                    recognized_text = recognize_handwriting(
                        canvas,
                        is_canvas=True
                    )

                    status_message = (
                        "OCR: " + recognized_text[:35]
                        if recognized_text
                        else "No text found"
                    )

                    print("OCR:", recognized_text)

                except Exception as error:
                    status_message = "OCR error"
                    print("OCR Error:", error)

            elif key == ord('o'):
                try:
                    recognized_text = recognize_latest_saved_drawing()

                    status_message = (
                        "Saved OCR: " + recognized_text[:30]
                        if recognized_text
                        else "No saved text found"
                    )

                    print("Saved OCR:", recognized_text)

                except Exception as error:
                    status_message = "Saved OCR error"
                    print("Saved OCR Error:", error)

            elif key == ord('q') or key == 27:
                print("Closing Air Canvas...")
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Air Canvas closed successfully!")
