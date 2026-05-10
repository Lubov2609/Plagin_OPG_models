import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.local_dataset import (
    ensure_dataset_dirs,
    load_label_records,
    mark_training_failed,
    mark_training_started,
    mark_training_success,
)
from app.preprocessing import OPGPreprocessor


_TRAINING_LOCK = threading.Lock()


class LocalOPGDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        paths = ensure_dataset_dirs(base_dir)
        self.images_dir = paths["images_dir"]
        self.labels_file = paths["labels_file"]
        self.preprocessor = OPGPreprocessor()
        self.records = self._load_records()

    def _load_records(self):
        records = []
        for row in load_label_records(self.base_dir):
            image_name = row["image"]
            image_path = self.images_dir / image_name
            if not image_name or not image_path.exists():
                continue

            records.append({
                "image_path": image_path,
                "labels": torch.tensor([
                    int(row["caries"]),
                    int(row["pulpitis"]),
                    int(row["wisdom_teeth_count"]),
                    int(row["impacted_wisdom_teeth_count"]),
                ], dtype=torch.float32),
            })

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("L")
        image = image.resize(self.preprocessor.size)
        image_tensor = self.preprocessor.transform(image)
        return image_tensor, record["labels"]


def _load_state_dict(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)


def _freeze_for_safe_local_training(model):
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("backbone.")


def _keep_batch_norm_stable(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _compute_loss(outputs, targets):
    binary_loss = 0
    for task_index in range(2):
        logits = outputs[task_index].squeeze(1)
        labels = targets[:, task_index].float()
        binary_loss = binary_loss + nn.functional.binary_cross_entropy_with_logits(logits, labels)

    multiclass_loss = 0
    for task_index in range(2):
        logits = outputs[task_index + 2]
        labels = targets[:, task_index + 2].long()
        multiclass_loss = multiclass_loss + nn.functional.cross_entropy(logits, labels)

    return (binary_loss / 2) + (multiclass_loss / 2)


def train_local_model(
    base_dir,
    model_class,
    base_model_path,
    local_model_path,
    epochs=3,
    batch_size=8,
    learning_rate=1e-4,
):
    if not _TRAINING_LOCK.acquire(blocking=False):
        return {
            "status": "skipped",
            "message": "Локальное дообучение уже выполняется.",
        }

    try:
        mark_training_started(base_dir)

        dataset = LocalOPGDataset(base_dir)
        if len(dataset) == 0:
            message = "Нет сохраненных врачебных разметок для дообучения."
            mark_training_failed(base_dir, message)
            return {"status": "failed", "message": message}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_class()
        source_model_path = local_model_path if Path(local_model_path).exists() else base_model_path
        _load_state_dict(model, source_model_path, device)

        _freeze_for_safe_local_training(model)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=1e-5,
        )

        loader = DataLoader(
            dataset,
            batch_size=max(1, min(int(batch_size), len(dataset))),
            shuffle=True,
            num_workers=0,
        )

        losses = []
        for _ in range(max(1, int(epochs))):
            epoch_loss = 0.0
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                _keep_batch_norm_stable(model)
                outputs = model(images)
                loss = _compute_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            losses.append(round(epoch_loss / max(1, len(loader)), 6))

        local_model_path = Path(local_model_path)
        local_model_path.parent.mkdir(parents=True, exist_ok=True)

        training_dir = Path(base_dir) / "local_training"
        backups_dir = training_dir / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)

        if local_model_path.exists():
            backup_name = datetime.now().strftime("opg_model_local_%Y-%m-%d_%H-%M-%S.pth")
            shutil.copy2(local_model_path, backups_dir / backup_name)

        temp_model_path = local_model_path.with_suffix(".tmp")
        torch.save(model.state_dict(), temp_model_path)
        temp_model_path.replace(local_model_path)

        history_file = training_dir / "history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    saved_history = json.load(f)
                if isinstance(saved_history, list):
                    history = saved_history
            except (json.JSONDecodeError, OSError):
                history = []

        history.append({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_size": len(dataset),
            "source_model": str(source_model_path),
            "local_model": str(local_model_path),
            "epochs": max(1, int(epochs)),
            "batch_size": max(1, min(int(batch_size), len(dataset))),
            "losses": losses,
            "device": device,
        })

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        message = f"Локальная модель обновлена на {len(dataset)} размеченных снимках."
        mark_training_success(base_dir, message)
        return {
            "status": "success",
            "message": message,
            "losses": losses,
            "model_path": str(local_model_path),
        }

    except Exception as exc:
        message = f"Ошибка локального дообучения: {exc}"
        mark_training_failed(base_dir, message)
        return {"status": "failed", "message": message}

    finally:
        _TRAINING_LOCK.release()
