import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_TRAIN_TRIGGER_COUNT = 8
SETTINGS_FILE_NAME = "settings.json"
LABELS_FILE_NAME = "labels.csv"
LABELS_META_FILE_NAME = "labels_meta.jsonl"

LEGACY_LABEL_FIELDNAMES = [
    "image",
    "caries",
    "pulpitis",
    "wisdom_teeth_count",
    "impacted_wisdom_teeth_count",
]


def get_dataset_paths(base_dir):
    dataset_dir = Path(base_dir) / "local_dataset"
    images_dir = dataset_dir / "images"
    labels_file = dataset_dir / LABELS_FILE_NAME
    labels_meta_file = dataset_dir / LABELS_META_FILE_NAME
    settings_file = dataset_dir / SETTINGS_FILE_NAME

    return {
        "dataset_dir": dataset_dir,
        "images_dir": images_dir,
        "labels_file": labels_file,
        "labels_meta_file": labels_meta_file,
        "settings_file": settings_file,
    }


def ensure_dataset_dirs(base_dir):
    paths = get_dataset_paths(base_dir)
    paths["images_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def _default_settings():
    return {
        "auto_train_enabled": True,
        "train_trigger_count": DEFAULT_TRAIN_TRIGGER_COUNT,
        "last_trained_label_count": 0,
        "last_training_status": "not_started",
        "last_training_message": "",
        "last_training_at": None,
    }


def load_learning_settings(base_dir):
    paths = ensure_dataset_dirs(base_dir)
    settings = _default_settings()

    if paths["settings_file"].exists():
        try:
            with open(paths["settings_file"], "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                settings.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    settings["train_trigger_count"] = max(1, int(settings.get("train_trigger_count", DEFAULT_TRAIN_TRIGGER_COUNT)))
    settings["last_trained_label_count"] = max(0, int(settings.get("last_trained_label_count", 0)))
    settings["auto_train_enabled"] = bool(settings.get("auto_train_enabled", True))

    return settings


def save_learning_settings(base_dir, settings):
    paths = ensure_dataset_dirs(base_dir)
    merged = _default_settings()
    merged.update(settings)
    merged["train_trigger_count"] = max(1, int(merged["train_trigger_count"]))
    merged["last_trained_label_count"] = max(0, int(merged["last_trained_label_count"]))

    with open(paths["settings_file"], "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    return merged


def update_train_trigger_count(base_dir, train_trigger_count):
    settings = load_learning_settings(base_dir)
    settings["train_trigger_count"] = max(1, int(train_trigger_count))
    return save_learning_settings(base_dir, settings)


def count_labels(base_dir):
    return len(load_label_records(base_dir))


def _is_header(row):
    if not row:
        return False
    return row[0].strip().lower() in {"image", "filename", "file", "name"}


def _normalize_label_row(row):
    if len(row) < 5:
        return None

    try:
        return {
            "image": row[0].strip(),
            "caries": _safe_int(row[1], 0, 1),
            "pulpitis": _safe_int(row[2], 0, 1),
            "wisdom_teeth_count": _safe_int(row[3], 0, 4),
            "impacted_wisdom_teeth_count": _safe_int(row[4], 0, 4),
        }
    except (TypeError, ValueError):
        return None


def load_label_records(base_dir):
    paths = ensure_dataset_dirs(base_dir)

    if not paths["labels_file"].exists():
        return []

    text = paths["labels_file"].read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    records = []

    for line in text.splitlines():
        delimiter = ";" if line.count(";") > line.count(",") else ","
        row = next(csv.reader([line], delimiter=delimiter))
        if _is_header(row):
            continue

        record = _normalize_label_row(row)
        if record:
            records.append(record)

    return records


def ensure_labels_csv_format(base_dir):
    paths = ensure_dataset_dirs(base_dir)
    if not paths["labels_file"].exists():
        return

    text = paths["labels_file"].read_text(encoding="utf-8-sig")
    if not text.strip():
        return

    first_line = text.splitlines()[0]
    already_target_format = "," in first_line and ";" not in first_line and not _is_header(next(csv.reader([first_line])))
    if already_target_format:
        return

    records = load_label_records(base_dir)
    with open(paths["labels_file"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for record in records:
            writer.writerow([
                record["image"],
                record["caries"],
                record["pulpitis"],
                record["wisdom_teeth_count"],
                record["impacted_wisdom_teeth_count"],
            ])


def pending_label_count(base_dir):
    settings = load_learning_settings(base_dir)
    return max(0, count_labels(base_dir) - settings["last_trained_label_count"])


def should_start_training(base_dir):
    settings = load_learning_settings(base_dir)
    if not settings["auto_train_enabled"]:
        return False
    return pending_label_count(base_dir) >= settings["train_trigger_count"]


def mark_training_started(base_dir):
    settings = load_learning_settings(base_dir)
    settings["last_training_status"] = "running"
    settings["last_training_message"] = "Локальное дообучение выполняется."
    settings["last_training_at"] = datetime.now().isoformat(timespec="seconds")
    return save_learning_settings(base_dir, settings)


def mark_training_success(base_dir, message):
    settings = load_learning_settings(base_dir)
    settings["last_trained_label_count"] = count_labels(base_dir)
    settings["last_training_status"] = "success"
    settings["last_training_message"] = message
    settings["last_training_at"] = datetime.now().isoformat(timespec="seconds")
    return save_learning_settings(base_dir, settings)


def mark_training_failed(base_dir, message):
    settings = load_learning_settings(base_dir)
    settings["last_training_status"] = "failed"
    settings["last_training_message"] = message
    settings["last_training_at"] = datetime.now().isoformat(timespec="seconds")
    return save_learning_settings(base_dir, settings)


def _safe_int(value, min_value=0, max_value=4):
    number = int(value)
    return min(max(number, min_value), max_value)


def save_doctor_label(
    image_path,
    base_dir,
    caries,
    pulpitis,
    wisdom_teeth_count,
    impacted_wisdom_teeth_count,
    doctor_comment="",
):
    paths = ensure_dataset_dirs(base_dir)
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Файл снимка не найден: {image_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_source_name = image_path.name.replace(" ", "_")
    new_image_name = f"{timestamp}_{safe_source_name}"
    new_image_path = paths["images_dir"] / new_image_name

    shutil.copy2(image_path, new_image_path)
    ensure_labels_csv_format(base_dir)

    row = [
        new_image_name,
        _safe_int(caries, 0, 1),
        _safe_int(pulpitis, 0, 1),
        _safe_int(wisdom_teeth_count, 0, 4),
        _safe_int(impacted_wisdom_teeth_count, 0, 4),
    ]

    with open(paths["labels_file"], "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    metadata = {
        "image": new_image_name,
        "source_filename": image_path.name,
        "doctor_comment": str(doctor_comment or "").strip(),
        "created_at": timestamp,
    }
    with open(paths["labels_meta_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    return {
        "labels_file": paths["labels_file"],
        "image_file": new_image_path,
        "label_count": count_labels(base_dir),
        "pending_count": pending_label_count(base_dir),
    }
