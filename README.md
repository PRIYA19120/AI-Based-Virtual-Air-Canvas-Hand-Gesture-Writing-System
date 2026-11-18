
# 🎨 AI Virtual Air Canvas

An AI-powered virtual drawing application that allows users to create digital drawings in the air using hand gestures. The system uses a webcam to track hand movements and converts them into real-time drawings on a virtual canvas.

## 📌 About the Project

AI Virtual Air Canvas is a computer vision-based project developed using Python. It provides a touch-free drawing experience through hand gesture recognition.

Users can draw, erase, change colors, select backgrounds, save their artwork, and recognize handwritten text using OCR technology.

## ✨ Features

- ✋ **Hand Gesture Recognition:** Draw using real-time hand movements.
- 🎨 **Multiple Colors:** Choose from different drawing colors.
- 🖌️ **Adjustable Brush Size:** Increase or decrease brush thickness.
- 🧽 **Adjustable Eraser:** Control eraser size from 5 to 100.
- 🌈 **Background Selection:** Choose different canvas backgrounds.
- ↩️ **Undo and Redo:** Restore previous drawing actions.
- 🗑️ **Clear Canvas:** Remove all drawings instantly.
- 📷 **OCR Recognition:** Convert handwritten drawings into digital text.
- 💾 **Save Drawings:** Export artwork in PNG format.
- 🖼️ **JPG Export:** Save drawings as JPG images.
- 🗂️ **Drawing Gallery:** View previously saved drawings.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| OpenCV | Image processing and webcam access |
| MediaPipe | Hand landmark detection |
| NumPy | Canvas and image operations |
| Pytesseract | Handwriting recognition using OCR |

## 📂 Project Structure

```text
AI-Virtual-Air-Canvas/
│
├── app.py
├── gallery.py
├── hand_landmarker.task
├── saved_drawings/
└── README.md
```

## ⚙️ Installation

**Step 1: Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**Step 2: Open the project folder**

```bash
cd AI-Virtual-Air-Canvas
```

**Step 3: Install required libraries**

```bash
pip install opencv-python mediapipe numpy pytesseract
```

**Step 4: Install Tesseract OCR**

Install Tesseract OCR separately and configure its executable path in `app.py`.

**Step 5: Download the MediaPipe hand landmark model**

Keep `hand_landmarker.task` inside the project folder.

**Step 6: Run the application**

```bash
python app.py
```

## ⌨️ Keyboard Shortcuts

| Key | Function |
|---|---|
| B / G / R / Y / W | Select drawing color |
| 1–5 | Select colors |
| 6–0 | Select background |
| P | Pink background |
| E | Activate eraser |
| C | Clear canvas |
| [ / - | Decrease brush size |
| ] / = | Increase brush size |
| , | Decrease eraser size |
| . | Increase eraser size |
| S | Save drawing as PNG |
| J | Export drawing as JPG |
| T | Recognize current canvas text |
| O | Recognize saved drawing text |
| Z | Undo |
| X | Redo |
| Q | Exit application |

## 🚀 Future Enhancements

- Advanced hand gesture controls.
- Shape recognition and drawing assistance.
- Text-to-Speech for recognized text.
- More drawing tools and customization options.
- Improved handwriting recognition accuracy.

## 👩‍💻 Developer

**Priya Joshi**

BCA – Artificial Intelligence and Data Science

Graphic Era University, Dehradun

---

⭐ If you find this project interesting, feel free to explore the repository!
