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

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.local_dataset import (
    count_labels,
    ensure_dataset_dirs,
    load_learning_settings,
    pending_label_count,
    save_doctor_label,
    should_start_training,
    update_train_trigger_count,
)
from app.local_training import train_local_model
from app.model import DenseNetModel
from app.preprocessing import OPGPreprocessor
from app.predictor import OPGPredictor


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_base_model_path():
    candidates = [
        BASE_DIR / "models" / "opg_model.pth",
        Path(getattr(sys, "_MEIPASS", BASE_DIR)) / "models" / "opg_model.pth",
        BASE_DIR.parent / "models" / "opg_model.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


MODEL_PATH = resolve_base_model_path()
LOCAL_MODEL_PATH = BASE_DIR / "models" / "opg_model_local.pth"
WATCH_DIR = BASE_DIR / "watch_folder"
RESULTS_DIR = BASE_DIR / "results"

(BASE_DIR / "models").mkdir(exist_ok=True)
WATCH_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
ensure_dataset_dirs(BASE_DIR)

file_queue = queue.Queue()
observer = None


preprocessor = OPGPreprocessor()

predictor = OPGPredictor(
    model_path=MODEL_PATH,
    local_model_path=LOCAL_MODEL_PATH,
    model_class=DenseNetModel,
    preprocessor=preprocessor,
    threshold=0.5,
)


def save_result(filename, result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    result_file = RESULTS_DIR / f"{safe_name}_{timestamp}.json"

    data = {
        "filename": filename,
        "created_at": timestamp,
        "result": result,
        "model_source": predictor.model_source,
        "model_path": str(predictor.active_model_path),
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


def learning_status_text():
    settings = load_learning_settings(BASE_DIR)
    return (
        f"{predictor.get_model_status()}. "
        f"Локальный датасет: {count_labels(BASE_DIR)}; "
        f"ожидает дообучения: {pending_label_count(BASE_DIR)}; "
        f"режим: после {settings['train_trigger_count']} снимков."
    )


def update_main_status(status_label, message=None):
    prefix = f"{message}\n" if message else ""
    status_label.config(text=prefix + learning_status_text())


def add_result_card(parent, title, status, confidence, color):
    card = tk.Frame(parent, bg="#f8fafc", padx=14, pady=12)
    card.pack(fill="x", padx=16, pady=8)

    label_title = tk.Label(
        card,
        text=title,
        font=("Arial", 13, "bold"),
        bg="#f8fafc",
        fg="#172033",
    )
    label_title.pack(anchor="w")

    label_status = tk.Label(
        card,
        text=status,
        font=("Arial", 12),
        bg="#f8fafc",
        fg=color,
    )
    label_status.pack(anchor="w", pady=3)

    label_conf = tk.Label(
        card,
        text=f"Уверенность: {confidence}%",
        font=("Arial", 10),
        bg="#f8fafc",
        fg="#64748b",
    )
    label_conf.pack(anchor="w")

    progress = ttk.Progressbar(
        card,
        orient="horizontal",
        length=360,
        mode="determinate",
    )
    progress["value"] = confidence
    progress.pack(anchor="w", pady=6)


def run_local_training_in_background(root, status_label, main_status_label=None):
    def worker():
        result = train_local_model(
            base_dir=BASE_DIR,
            model_class=DenseNetModel,
            base_model_path=MODEL_PATH,
            local_model_path=LOCAL_MODEL_PATH,
            epochs=3,
            batch_size=8,
        )

        if result.get("status") == "success":
            predictor.reload_model()

        def show_result():
            status_label.config(text=result.get("message", "Локальное дообучение завершено."))
            if main_status_label is not None:
                update_main_status(main_status_label, "Локальное дообучение завершено.")

        try:
            root.after(0, show_result)
        except tk.TclError:
            pass

    status_label.config(text="Локальное дообучение запущено в фоне.")
    threading.Thread(target=worker, daemon=True).start()


def add_doctor_form(parent, root, image_path, result, main_status_label=None):
    doctor_frame = tk.Frame(parent, bg="white", padx=18, pady=16)
    doctor_frame.pack(fill="x", padx=16, pady=(8, 16))

    tk.Label(
        doctor_frame,
        text="Разметка врача",
        font=("Arial", 16, "bold"),
        bg="white",
        fg="#172033",
    ).pack(anchor="w", pady=(0, 8))

    info_text = (
        "Проверьте результат модели и заполните фактические признаки по снимку. "
        "Эти данные сохраняются только на этом компьютере и используются для локального дообучения."
    )

    tk.Label(
        doctor_frame,
        text=info_text,
        font=("Arial", 10),
        bg="white",
        fg="#475569",
        justify="left",
        wraplength=620,
    ).pack(anchor="w", fill="x")

    form = tk.Frame(doctor_frame, bg="white")
    form.pack(fill="x", pady=(14, 0))

    caries_var = tk.StringVar(value="Да" if result["caries"]["detected"] else "Нет")
    pulpitis_var = tk.StringVar(value="Да" if result["pulpitis"]["detected"] else "Нет")
    wisdom_var = tk.IntVar(value=int(result["wisdom_teeth"]["class_id"]))
    impacted_var = tk.IntVar(value=int(result["impacted_wisdom_teeth"]["class_id"]))
    settings = load_learning_settings(BASE_DIR)
    trigger_var = tk.IntVar(value=1 if settings["train_trigger_count"] == 1 else 8)

    def add_combo(row, label, variable, values):
        tk.Label(
            form,
            text=label,
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#172033",
        ).grid(row=row, column=0, sticky="w", pady=6)

        combo = ttk.Combobox(
            form,
            textvariable=variable,
            values=values,
            state="readonly",
            width=18,
        )
        combo.grid(row=row, column=1, sticky="ew", pady=6)
        return combo

    form.columnconfigure(0, weight=0)
    form.columnconfigure(1, weight=1)
    add_combo(0, "Кариес", caries_var, ["Нет", "Да"])
    add_combo(1, "Пульпит", pulpitis_var, ["Нет", "Да"])
    add_combo(2, "Количество 8-х зубов", wisdom_var, [0, 1, 2, 3, 4])
    add_combo(3, "Ретинированные 8-е зубы", impacted_var, [0, 1, 2, 3, 4])

    tk.Label(
        form,
        text="Комментарий врача",
        font=("Arial", 10, "bold"),
        bg="white",
        fg="#172033",
    ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4))

    comment = tk.Text(form, height=4, width=42, wrap="word")
    comment.grid(row=5, column=0, columnspan=2, sticky="ew")

    tk.Label(
        form,
        text="Запуск локального дообучения",
        font=("Arial", 10, "bold"),
        bg="white",
        fg="#172033",
    ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 4))

    radio_frame = tk.Frame(form, bg="white")
    radio_frame.grid(row=7, column=0, columnspan=2, sticky="w")

    tk.Radiobutton(
        radio_frame,
        text="после каждого снимка",
        variable=trigger_var,
        value=1,
        bg="white",
    ).pack(side="left", padx=(0, 14))

    tk.Radiobutton(
        radio_frame,
        text="после 8 снимков",
        variable=trigger_var,
        value=8,
        bg="white",
    ).pack(side="left")

    form_status = tk.Label(
        form,
        text="",
        font=("Arial", 10),
        bg="white",
        fg="#166534",
        wraplength=430,
        justify="left",
    )
    form_status.grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def save_label():
        try:
            update_train_trigger_count(BASE_DIR, trigger_var.get())
            saved = save_doctor_label(
                image_path=image_path,
                base_dir=BASE_DIR,
                caries=1 if caries_var.get() == "Да" else 0,
                pulpitis=1 if pulpitis_var.get() == "Да" else 0,
                wisdom_teeth_count=wisdom_var.get(),
                impacted_wisdom_teeth_count=impacted_var.get(),
                doctor_comment=comment.get("1.0", "end").strip(),
            )

            trigger_count = trigger_var.get()
            message = (
                f"Разметка сохранена. В датасете: {saved['label_count']} снимков. "
                f"До запуска дообучения: {max(0, trigger_count - saved['pending_count'])}."
            )
            form_status.config(text=message, fg="#166534")

            if main_status_label is not None:
                update_main_status(main_status_label, "Разметка врача сохранена.")

            if should_start_training(BASE_DIR):
                run_local_training_in_background(root, form_status, main_status_label)

        except Exception as exc:
            form_status.config(text=f"Ошибка сохранения разметки: {exc}", fg="#991b1b")

    save_button = tk.Button(
        form,
        text="Сохранить разметку",
        font=("Arial", 11, "bold"),
        bg="#2563eb",
        fg="white",
        padx=16,
        pady=8,
        command=save_label,
    )
    save_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 0))


def show_result_window(root, image_path, result, result_file, main_status_label=None):
    window = tk.Toplevel(root)
    window.title("Результат анализа ортопантомограммы")
    window.geometry("1180x860")
    window.configure(bg="#eef3f8")
    window.lift()
    window.focus_force()

    canvas = tk.Canvas(window, bg="#eef3f8", highlightthickness=0)
    scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
    content = tk.Frame(canvas, bg="#eef3f8")

    content.bind(
        "<Configure>",
        lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=content, anchor="nw", width=1160)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    title = tk.Label(
        content,
        text="Анализ ортопантомограммы",
        font=("Arial", 22, "bold"),
        bg="#eef3f8",
        fg="#172033",
    )
    title.pack(pady=15)

    model_label = tk.Label(
        content,
        text=learning_status_text(),
        font=("Arial", 10),
        bg="#eef3f8",
        fg="#475569",
        wraplength=1000,
        justify="center",
    )
    model_label.pack(pady=(0, 10))

    main_frame = tk.Frame(content, bg="#eef3f8")
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
        fg="#172033",
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
        fg="#64748b",
    )
    file_label.pack(pady=10)

    result_title = tk.Label(
        right_frame,
        text="Результаты анализа",
        font=("Arial", 16, "bold"),
        bg="white",
        fg="#172033",
    )
    result_title.pack(pady=10)

    add_result_card(
        right_frame,
        "Кариес",
        result["caries"]["status"],
        result["caries"]["confidence_percent"],
        "#dc2626" if result["caries"]["detected"] else "#16a34a",
    )

    add_result_card(
        right_frame,
        "Пульпит",
        result["pulpitis"]["status"],
        result["pulpitis"]["confidence_percent"],
        "#dc2626" if result["pulpitis"]["detected"] else "#16a34a",
    )

    add_result_card(
        right_frame,
        "Наличие 8-х зубов",
        result["wisdom_teeth"]["class_name"],
        result["wisdom_teeth"]["confidence_percent"],
        "#2563eb",
    )

    add_result_card(
        right_frame,
        "Ретинированные 8-е зубы",
        result["impacted_wisdom_teeth"]["class_name"],
        result["impacted_wisdom_teeth"]["confidence_percent"],
        "#f59e0b",
    )

    saved_label = tk.Label(
        right_frame,
        text=f"Результат сохранен:\n{result_file.name}",
        font=("Arial", 10),
        bg="white",
        fg="#166534",
    )
    saved_label.pack(pady=15)

    add_doctor_form(left_frame, root, image_path, result, main_status_label)

    warning = tk.Label(
        content,
        text="Результаты являются вспомогательными и не заменяют заключение врача-специалиста.",
        font=("Arial", 10),
        bg="#eef3f8",
        fg="#64748b",
    )
    warning.pack(pady=10)


def analyze_file(root, status_label, file_path):
    update_main_status(status_label, f"Найден файл: {file_path.name}. Выполняется анализ...")

    if not wait_until_file_ready(file_path):
        update_main_status(status_label, f"Ошибка: файл занят или недоступен: {file_path.name}")
        return

    try:
        result = predictor.predict(str(file_path))
        result_file = save_result(file_path.name, result)

        update_main_status(status_label, f"Анализ выполнен: {file_path.name}")

        show_result_window(root, file_path, result, result_file, status_label)

    except Exception as e:
        update_main_status(status_label, f"Ошибка анализа: {e}")


def process_queue(root, status_label):
    try:
        file_path = file_queue.get_nowait()
        analyze_file(root, status_label, file_path)

    except queue.Empty:
        pass

    root.after(1000, lambda: process_queue(root, status_label))


def open_watch_folder():
    os.startfile(str(WATCH_DIR))


def open_local_dataset_folder():
    paths = ensure_dataset_dirs(BASE_DIR)
    os.startfile(str(paths["dataset_dir"]))


def close_app(root):
    stop_observer()
    root.destroy()


def start_desktop_app():
    root = tk.Tk()
    root.title("OPG AI Plugin")
    root.geometry("760x500")
    root.configure(bg="#eef3f8")

    title = tk.Label(
        root,
        text="OPG AI Plugin",
        font=("Arial", 24, "bold"),
        bg="#eef3f8",
        fg="#172033",
    )
    title.pack(pady=25)

    description = tk.Label(
        root,
        text="Плагин отслеживает папку и автоматически анализирует новые ортопантомограммы.",
        font=("Arial", 12),
        bg="#eef3f8",
        fg="#475569",
        wraplength=620,
        justify="center",
    )
    description.pack(pady=5)

    folder_label = tk.Label(
        root,
        text=f"Папка наблюдения:\n{WATCH_DIR}",
        font=("Arial", 10),
        bg="#eef3f8",
        fg="#64748b",
        wraplength=700,
        justify="center",
    )
    folder_label.pack(pady=15)

    status_label = tk.Label(
        root,
        text="",
        font=("Arial", 11, "bold"),
        bg="#eef3f8",
        fg="#166534",
        wraplength=700,
        justify="center",
    )
    status_label.pack(pady=15)
    update_main_status(status_label, "Статус: плагин запущен и ожидает снимки.")

    buttons_frame = tk.Frame(root, bg="#eef3f8")
    buttons_frame.pack(pady=20)

    open_button = tk.Button(
        buttons_frame,
        text="Открыть папку снимков",
        font=("Arial", 12),
        bg="#2563eb",
        fg="white",
        padx=20,
        pady=10,
        command=open_watch_folder,
    )
    open_button.pack(side="left", padx=8)

    dataset_button = tk.Button(
        buttons_frame,
        text="Открыть локальный датасет",
        font=("Arial", 12),
        bg="#0f766e",
        fg="white",
        padx=20,
        pady=10,
        command=open_local_dataset_folder,
    )
    dataset_button.pack(side="left", padx=8)

    exit_button = tk.Button(
        buttons_frame,
        text="Выход",
        font=("Arial", 12),
        bg="#dc2626",
        fg="white",
        padx=20,
        pady=10,
        command=lambda: close_app(root),
    )
    exit_button.pack(side="left", padx=8)

    start_observer()

    root.after(1000, lambda: process_queue(root, status_label))

    root.protocol("WM_DELETE_WINDOW", lambda: close_app(root))

    root.mainloop()


if __name__ == "__main__":
    start_desktop_app()
