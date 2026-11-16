
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

FOLDER = "saved_drawings"

root = tk.Tk()
root.title("Air Canvas - Saved Drawings Gallery")
root.geometry("1000x700")
root.configure(bg="#f1f5f9")

photos = []

header = tk.Frame(root, bg="#1e293b", height=80)
header.pack(fill="x")

title = tk.Label(
    header,
    text="Saved Drawings Gallery",
    font=("Arial", 22, "bold"),
    fg="white",
    bg="#1e293b"
)
title.pack(pady=20)

top = tk.Frame(root, bg="#f1f5f9")
top.pack(fill="x", padx=20, pady=12)

info = tk.Label(
    top,
    text="Your saved Air Canvas drawings",
    font=("Arial", 12),
    bg="#f1f5f9",
    fg="#334155"
)
info.pack(side="left")

refresh_button = tk.Button(
    top,
    text="Refresh Gallery",
    font=("Arial", 10, "bold"),
    bg="#2563eb",
    fg="white",
    command=lambda: load_gallery()
)
refresh_button.pack(side="right")

container = tk.Frame(root, bg="#f1f5f9")
container.pack(fill="both", expand=True, padx=20, pady=10)

gallery_canvas = tk.Canvas(
    container,
    bg="#f1f5f9",
    highlightthickness=0
)

scrollbar = tk.Scrollbar(
    container,
    orient="vertical",
    command=gallery_canvas.yview
)

gallery_frame = tk.Frame(gallery_canvas, bg="#f1f5f9")

gallery_window = gallery_canvas.create_window(
    (0, 0),
    window=gallery_frame,
    anchor="nw"
)

gallery_canvas.configure(yscrollcommand=scrollbar.set)

gallery_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


def update_scroll(event=None):
    gallery_canvas.configure(
        scrollregion=gallery_canvas.bbox("all")
    )


def resize_frame(event):
    gallery_canvas.itemconfig(
        gallery_window,
        width=event.width
    )


gallery_frame.bind("<Configure>", update_scroll)
gallery_canvas.bind("<Configure>", resize_frame)


def open_image(path):
    if os.path.exists(path):
        os.startfile(os.path.abspath(path))


def delete_image(path):
    answer = messagebox.askyesno(
        "Delete Drawing",
        "Are you sure you want to delete this drawing?"
    )

    if answer:
        try:
            os.remove(path)
            messagebox.showinfo(
                "Deleted",
                "Drawing deleted successfully!"
            )
            load_gallery()
        except Exception as e:
            messagebox.showerror("Error", str(e))


def load_gallery():
    global photos

    photos = []

    for widget in gallery_frame.winfo_children():
        widget.destroy()

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    files = [
        file for file in os.listdir(FOLDER)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    files.sort(key=lambda x: os.path.getmtime(
        os.path.join(FOLDER, x)
    ), reverse=True)

    info.config(text=f"Total Saved Drawings: {len(files)}")

    if not files:
        empty_label = tk.Label(
            gallery_frame,
            text="No drawings saved yet!\nSave a drawing from Air Canvas first.",
            font=("Arial", 16),
            bg="#f1f5f9",
            fg="#64748b"
        )
        empty_label.pack(pady=100)
        return

    for index, file in enumerate(files):
        path = os.path.join(FOLDER, file)

        card = tk.Frame(
            gallery_frame,
            bg="white",
            bd=1,
            relief="solid",
            padx=12,
            pady=12
        )

        row = index // 3
        column = index % 3

        card.grid(
            row=row,
            column=column,
            padx=12,
            pady=12,
            sticky="nsew"
        )

        try:
            img = Image.open(path)
            img.thumbnail((230, 160))

            photo = ImageTk.PhotoImage(img)
            photos.append(photo)

            image_label = tk.Label(
                card,
                image=photo,
                bg="white"
            )
            image_label.pack(pady=5)

        except Exception:
            tk.Label(
                card,
                text="Preview not available",
                width=28,
                height=8,
                bg="#e2e8f0"
            ).pack(pady=5)

        name_label = tk.Label(
            card,
            text=file,
            font=("Arial", 10, "bold"),
            bg="white",
            wraplength=220
        )
        name_label.pack(pady=8)

        button_frame = tk.Frame(card, bg="white")
        button_frame.pack(pady=5)

        open_button = tk.Button(
            button_frame,
            text="Open",
            bg="#2563eb",
            fg="white",
            command=lambda p=path: open_image(p)
        )
        open_button.pack(side="left", padx=5)

        delete_button = tk.Button(
            button_frame,
            text="Delete",
            bg="#dc2626",
            fg="white",
            command=lambda p=path: delete_image(p)
        )
        delete_button.pack(side="left", padx=5)

    for column in range(3):
        gallery_frame.grid_columnconfigure(
            column,
            weight=1
        )


load_gallery()

root.mainloop()
