import json
import os
import queue
import sys
import time
import threading
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.model import DenseNetModel
from app.preprocessing import OPGPreprocessor
from app.predictor import OPGPredictor


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "opg_model.pth"
WATCH_DIR = BASE_DIR / "watch_folder"
RESULTS_DIR = BASE_DIR / "results"

WATCH_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

file_queue = queue.Queue()
observer = None


preprocessor = OPGPreprocessor()

predictor = OPGPredictor(
    model_path=MODEL_PATH,
    model_class=DenseNetModel,
    preprocessor=preprocessor,
    threshold=0.5
)


def save_result(filename, result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    result_file = RESULTS_DIR / f"{safe_name}_{timestamp}.json"

    data = {
        "filename": filename,
        "created_at": timestamp,
        "result": result
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return result_file


def wait_until_file_ready(file_path, timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        try:
            with open(file_path, "rb"):
                return True
        except PermissionError:
            time.sleep(0.5)

    return False


class OPGFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            return

        file_queue.put(file_path)


def start_observer():
    global observer

    event_handler = OPGFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_DIR), recursive=False)
    observer.start()


def stop_observer():
    global observer

    if observer:
        observer.stop()
        observer.join()


def add_result_card(parent, title, status, confidence, color):
    card = tk.Frame(parent, bg="#f8fafc", padx=14, pady=12)
    card.pack(fill="x", padx=16, pady=8)

    label_title = tk.Label(
        card,
        text=title,
        font=("Arial", 13, "bold"),
        bg="#f8fafc",
        fg="#172033"
    )
    label_title.pack(anchor="w")

    label_status = tk.Label(
        card,
        text=status,
        font=("Arial", 12),
        bg="#f8fafc",
        fg=color
    )
    label_status.pack(anchor="w", pady=3)

    label_conf = tk.Label(
        card,
        text=f"Уверенность: {confidence}%",
        font=("Arial", 10),
        bg="#f8fafc",
        fg="#64748b"
    )
    label_conf.pack(anchor="w")

    progress = ttk.Progressbar(
        card,
        orient="horizontal",
        length=360,
        mode="determinate"
    )
    progress["value"] = confidence
    progress.pack(anchor="w", pady=6)


def show_result_window(root, image_path, result, result_file):
    window = tk.Toplevel(root)
    window.title("Результат анализа ортопантомограммы")
    window.geometry("1050x720")
    window.configure(bg="#eef3f8")
    window.lift()
    window.focus_force()

    title = tk.Label(
        window,
        text="Анализ ортопантомограммы",
        font=("Arial", 22, "bold"),
        bg="#eef3f8",
        fg="#172033"
    )
    title.pack(pady=15)

    main_frame = tk.Frame(window, bg="#eef3f8")
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    left_frame = tk.Frame(main_frame, bg="white")
    left_frame.pack(side="left", fill="both", expand=True, padx=10)

    right_frame = tk.Frame(main_frame, bg="white")
    right_frame.pack(side="right", fill="both", expand=True, padx=10)

    img_title = tk.Label(
        left_frame,
        text="Исходный снимок",
        font=("Arial", 16, "bold"),
        bg="white",
        fg="#172033"
    )
    img_title.pack(pady=10)

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((580, 430))

    photo = ImageTk.PhotoImage(img)

    img_label = tk.Label(left_frame, image=photo, bg="white")
    img_label.image = photo
    img_label.pack(pady=10)

    file_label = tk.Label(
        left_frame,
        text=f"Файл: {Path(image_path).name}",
        font=("Arial", 11),
        bg="white",
        fg="#64748b"
    )
    file_label.pack(pady=10)

    result_title = tk.Label(
        right_frame,
        text="Результаты анализа",
        font=("Arial", 16, "bold"),
        bg="white",
        fg="#172033"
    )
    result_title.pack(pady=10)

    add_result_card(
        right_frame,
        "Кариес",
        result["caries"]["status"],
        result["caries"]["confidence_percent"],
        "#dc2626" if result["caries"]["detected"] else "#16a34a"
    )

    add_result_card(
        right_frame,
        "Пульпит",
        result["pulpitis"]["status"],
        result["pulpitis"]["confidence_percent"],
        "#dc2626" if result["pulpitis"]["detected"] else "#16a34a"
    )

    add_result_card(
        right_frame,
        "Наличие 8-х зубов",
        result["wisdom_teeth"]["class_name"],
        result["wisdom_teeth"]["confidence_percent"],
        "#2563eb"
    )

    add_result_card(
        right_frame,
        "Ретенированные 8-е зубы",
        result["impacted_wisdom_teeth"]["class_name"],
        result["impacted_wisdom_teeth"]["confidence_percent"],
        "#f59e0b"
    )

    saved_label = tk.Label(
        right_frame,
        text=f"Результат сохранён:\n{result_file.name}",
        font=("Arial", 10),
        bg="white",
        fg="#166534"
    )
    saved_label.pack(pady=15)

    warning = tk.Label(
        window,
        text="Результаты являются вспомогательными и не заменяют заключение врача-специалиста.",
        font=("Arial", 10),
        bg="#eef3f8",
        fg="#64748b"
    )
    warning.pack(pady=10)


def analyze_file(root, status_label, file_path):
    status_label.config(text=f"Найден файл: {file_path.name}. Выполняется анализ...")

    if not wait_until_file_ready(file_path):
        status_label.config(text=f"Ошибка: файл занят или недоступен: {file_path.name}")
        return

    try:
        result = predictor.predict(str(file_path))
        result_file = save_result(file_path.name, result)

        status_label.config(text=f"Анализ выполнен: {file_path.name}")

        show_result_window(root, file_path, result, result_file)

    except Exception as e:
        status_label.config(text=f"Ошибка анализа: {e}")


def process_queue(root, status_label):
    try:
        file_path = file_queue.get_nowait()
        analyze_file(root, status_label, file_path)

    except queue.Empty:
        pass

    root.after(1000, lambda: process_queue(root, status_label))


def open_watch_folder():
    os.startfile(str(WATCH_DIR))


def close_app(root):
    stop_observer()
    root.destroy()


def start_desktop_app():
    root = tk.Tk()
    root.title("OPG AI Plugin")
    root.geometry("720x420")
    root.configure(bg="#eef3f8")

    title = tk.Label(
        root,
        text="OPG AI Plugin",
        font=("Arial", 24, "bold"),
        bg="#eef3f8",
        fg="#172033"
    )
    title.pack(pady=25)

    description = tk.Label(
        root,
        text="Плагин отслеживает папку и автоматически анализирует новые ортопантомограммы.",
        font=("Arial", 12),
        bg="#eef3f8",
        fg="#475569",
        wraplength=620,
        justify="center"
    )
    description.pack(pady=5)

    folder_label = tk.Label(
        root,
        text=f"Папка наблюдения:\n{WATCH_DIR}",
        font=("Arial", 10),
        bg="#eef3f8",
        fg="#64748b",
        wraplength=660,
        justify="center"
    )
    folder_label.pack(pady=15)

    status_label = tk.Label(
        root,
        text="Статус: плагин запущен и ожидает снимки",
        font=("Arial", 12, "bold"),
        bg="#eef3f8",
        fg="#166534"
    )
    status_label.pack(pady=15)

    buttons_frame = tk.Frame(root, bg="#eef3f8")
    buttons_frame.pack(pady=20)

    open_button = tk.Button(
        buttons_frame,
        text="Открыть папку",
        font=("Arial", 12),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=10,
        command=open_watch_folder
    )
    open_button.pack(side="left", padx=10)

    exit_button = tk.Button(
        buttons_frame,
        text="Выход",
        font=("Arial", 12),
        bg="#dc2626",
        fg="white",
        padx=20,
        pady=10,
        command=lambda: close_app(root)
    )
    exit_button.pack(side="left", padx=10)

    start_observer()

    root.after(1000, lambda: process_queue(root, status_label))

    root.protocol("WM_DELETE_WINDOW", lambda: close_app(root))

    root.mainloop()


if __name__ == "__main__":
    start_desktop_app()