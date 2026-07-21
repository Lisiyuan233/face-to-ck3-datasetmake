from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260718,
    "data": {
        "root": "face_to_ck3_dataset_male_small/processed_front",
        "schema": "face_to_ck3_dataset_male_small/dna_schema.json",
        "manifest": "face_to_ck3_dataset_male_small/processed_front/manifest.json",
        "train_label_stats": "face_to_ck3_dataset_male_small/processed_front/train_label_stats.json",
        "image_height": 384,
        "image_width": 256,
        "shuffle_buffer": 2000,
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": True,
        "dual_view": True,
        "dropout": 0.1,
    },
    "train": {
        "output_dir": "runs/convnext_tiny_dual_view",
        "epochs": 30,
        "freeze_backbone_epochs": 2,
        "batch_size": 32,
        "gradient_accumulation": 1,
        "num_workers": 8,
        "val_num_workers": 4,
        "backbone_lr": 0.0001,
        "head_lr": 0.0003,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.02,
        "grad_clip": 1.0,
        "amp": "bf16",
        "ema_decay": 0.9999,
        "log_every": 50,
        "early_stopping_patience": 5,
        "max_train_steps": None,
        "max_val_steps": None,
    },
    "loss": {
        "smooth_l1_beta": 0.05,
        "signed_weight": 1.0,
        "class_weight": 1.0,
        "strength_weight": 1.0,
        "color_weight": 0.2,
        "consistency_weight": 0.05,
        "minimum_class_visibility": 0.05,
        "class_weight_min": 0.5,
        "class_weight_max": 3.0,
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "rotation_degrees": 3.0,
        "translate_fraction": 0.03,
        "scale_min": 0.95,
        "scale_max": 1.05,
        "geometry_brightness": 0.20,
        "geometry_contrast": 0.20,
        "geometry_saturation": 0.10,
        "geometry_grayscale": 0.35,
        "blur_probability": 0.10,
        "noise_std": 0.015,
        "upper_occlusion_probability": 0.15,
        "color_brightness": 0.08,
        "color_contrast": 0.08,
        "color_saturation": 0.05,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    return _deep_update(config, user)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["model"].update(
        {"backbone": "resnet18", "pretrained": False, "dual_view": False}
    )
    config["data"].update(
        {"image_height": 192, "image_width": 128, "shuffle_buffer": 64}
    )
    config["train"].update(
        {
            "output_dir": "runs/smoke_test",
            "epochs": 1,
            "freeze_backbone_epochs": 0,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "num_workers": 0,
            "val_num_workers": 0,
            "amp": "off",
            "ema_decay": 0.0,
            "log_every": 1,
            "max_train_steps": 2,
            "max_val_steps": 2,
        }
    )
    config["loss"]["consistency_weight"] = 0.0
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    train = config["train"]
    model = config["model"]
    if int(data["image_height"]) < 32 or int(data["image_width"]) < 32:
        raise ValueError("image dimensions must be at least 32 pixels")
    for key in ("epochs", "batch_size", "gradient_accumulation"):
        if int(train[key]) < 1:
            raise ValueError(f"train.{key} must be >= 1")
    for key in ("num_workers", "val_num_workers", "freeze_backbone_epochs"):
        if int(train[key]) < 0:
            raise ValueError(f"train.{key} must be >= 0")
    if str(train["amp"]).lower() not in {"off", "fp16", "bf16"}:
        raise ValueError("train.amp must be off, fp16, or bf16")
    if str(model["backbone"]) not in {"resnet18", "convnext_tiny"}:
        raise ValueError("model.backbone must be resnet18 or convnext_tiny")
    if int(data["shuffle_buffer"]) < 1:
        raise ValueError("data.shuffle_buffer must be >= 1")
