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
        "fraction": 1.0,
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": True,
        "dual_view": True,
        "side_view": False,
        "dropout": 0.2,
        "geometry_branch": {
            "enabled": False,
            "targets": ["signed", "strength", "categorical"],
            "grid_height": 48,
            "grid_width": 32,
            "hidden_dim": 256,
            "dropout": 0.1,
            "gate_bias": -2.0,
            "foreground_margin": 0.06,
            "foreground_softness": 0.03,
        },
    },
    "train": {
        "output_dir": "runs/convnext_tiny_geometry_v1",
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
        "early_stopping_min_delta": 0.0,
        "max_train_steps": None,
        "max_val_steps": None,
    },
    "loss": {
        "smooth_l1_beta": 0.05,
        "scalar_weight": 1.0,
        "signed_weight": 1.0,
        "class_weight": 1.0,
        "strength_weight": 1.0,
        "consistency_weight": 0.1,
        "class_label_smoothing": 0.05,
        "class_visibility_threshold": 0.10,
        "reference_only_categorical_fields": [
            "face_detail_eye_socket"
        ],
        "consistency_excluded_categorical_fields": [
            "face_detail_eye_socket"
        ],
        "class_weight_min": 0.5,
        "class_weight_max": 2.0,
        "field_weights_path": None,
        "texture_metrics_path": None,
        "texture_weight_blend": 0.0,
        "use_schema_visibility_thresholds": False,
    },
    "selection": {
        "scalar_mae_weight": 0.0,
        "signed_mae_weight": 0.40,
        "strength_mae_weight": 0.25,
        "categorical_error_weight": 0.35,
        "categorical_min_observable_count": 0,
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "rotation_degrees": 3.0,
        "translate_fraction": 0.03,
        "scale_min": 0.95,
        "scale_max": 1.05,
        "geometry_brightness": 0.20,
        "geometry_contrast": 0.20,
        "geometry_saturation": 0.35,
        "geometry_hue": 0.08,
        "geometry_grayscale": 0.50,
        "blur_probability": 0.10,
        "noise_std": 0.015,
        "upper_occlusion_probability": 0.15,
        "reference_brightness": 0.08,
        "reference_contrast": 0.08,
        "reference_saturation": 0.05,
        "reference_hue": 0.02,
        "exposure_normalization": {
            "enabled": False,
            "target_mean": 0.45,
            "target_std": 0.20,
            "min_gain": 0.5,
            "max_gain": 2.0,
        },
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
        {
            "backbone": "resnet18",
            "pretrained": False,
            "dual_view": False,
        }
    )
    if config["model"]["geometry_branch"]["enabled"]:
        config["model"]["geometry_branch"]["hidden_dim"] = 64
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
    loss = config["loss"]
    selection = config["selection"]
    augmentation = config["augmentation"]
    if int(data["image_height"]) < 32 or int(data["image_width"]) < 32:
        raise ValueError("image dimensions must be at least 32 pixels")
    for key in (
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "early_stopping_patience",
    ):
        if int(train[key]) < 1:
            raise ValueError(f"train.{key} must be >= 1")
    for key in ("num_workers", "val_num_workers", "freeze_backbone_epochs"):
        if int(train[key]) < 0:
            raise ValueError(f"train.{key} must be >= 0")
    if str(train["amp"]).lower() not in {"off", "fp16", "bf16"}:
        raise ValueError("train.amp must be off, fp16, or bf16")
    if float(train["early_stopping_min_delta"]) < 0:
        raise ValueError("train.early_stopping_min_delta must be >= 0")
    if str(model["backbone"]) not in {"resnet18", "convnext_tiny"}:
        raise ValueError("model.backbone must be resnet18 or convnext_tiny")
    if not isinstance(model.get("side_view"), bool):
        raise ValueError("model.side_view must be a boolean")
    geometry_branch = model.get("geometry_branch")
    if not isinstance(geometry_branch, dict):
        raise ValueError("model.geometry_branch must be an object")
    if not isinstance(geometry_branch.get("enabled"), bool):
        raise ValueError("model.geometry_branch.enabled must be a boolean")
    targets = geometry_branch.get("targets")
    allowed_targets = {"scalar", "signed", "strength", "categorical"}
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) for target in targets)
        or len(targets) != len(set(targets))
        or not set(targets).issubset(allowed_targets)
    ):
        raise ValueError(
            "model.geometry_branch.targets must be a non-empty list without "
            "duplicates containing only scalar, signed, strength, and categorical"
        )
    for key in ("grid_height", "grid_width"):
        if int(geometry_branch[key]) < 8:
            raise ValueError(f"model.geometry_branch.{key} must be >= 8")
    if int(geometry_branch["hidden_dim"]) < 16:
        raise ValueError("model.geometry_branch.hidden_dim must be >= 16")
    if not 0.0 <= float(geometry_branch["dropout"]) < 1.0:
        raise ValueError("model.geometry_branch.dropout must be in [0, 1)")
    if float(geometry_branch["foreground_margin"]) < 0:
        raise ValueError(
            "model.geometry_branch.foreground_margin must be >= 0"
        )
    if float(geometry_branch["foreground_softness"]) <= 0:
        raise ValueError(
            "model.geometry_branch.foreground_softness must be > 0"
        )
    if int(data["shuffle_buffer"]) < 1:
        raise ValueError("data.shuffle_buffer must be >= 1")
    if not 0.0 < float(data["fraction"]) <= 1.0:
        raise ValueError("data.fraction must be in (0, 1]")
    if not 0.0 <= float(loss["class_label_smoothing"]) < 1.0:
        raise ValueError("loss.class_label_smoothing must be in [0, 1)")
    if not 0.0 <= float(loss["class_visibility_threshold"]) <= 1.0:
        raise ValueError("loss.class_visibility_threshold must be in [0, 1]")
    for key in (
        "scalar_weight",
        "signed_weight",
        "class_weight",
        "strength_weight",
        "consistency_weight",
    ):
        if float(loss[key]) < 0:
            raise ValueError(f"loss.{key} must be >= 0")
    if not 0.0 <= float(loss["texture_weight_blend"]) <= 1.0:
        raise ValueError("loss.texture_weight_blend must be in [0, 1]")
    for key in ("field_weights_path", "texture_metrics_path"):
        if loss.get(key) is not None and not isinstance(loss[key], str):
            raise ValueError(f"loss.{key} must be a path string or null")
    if not isinstance(loss.get("use_schema_visibility_thresholds"), bool):
        raise ValueError("loss.use_schema_visibility_thresholds must be a boolean")
    selection_keys = (
        "scalar_mae_weight",
        "signed_mae_weight",
        "strength_mae_weight",
        "categorical_error_weight",
    )
    if any(float(selection[key]) < 0 for key in selection_keys):
        raise ValueError("selection weights must be >= 0")
    if sum(float(selection[key]) for key in selection_keys) <= 0:
        raise ValueError("at least one selection weight must be > 0")
    if int(selection["categorical_min_observable_count"]) < 0:
        raise ValueError(
            "selection.categorical_min_observable_count must be >= 0"
        )
    for key in ("geometry_hue", "reference_hue"):
        if not 0.0 <= float(augmentation[key]) <= 0.5:
            raise ValueError(f"augmentation.{key} must be in [0, 0.5]")
    exposure = augmentation.get("exposure_normalization")
    if not isinstance(exposure, dict):
        raise ValueError("augmentation.exposure_normalization must be an object")
    if not isinstance(exposure.get("enabled"), bool):
        raise ValueError("augmentation.exposure_normalization.enabled must be boolean")
    for key in ("target_mean", "target_std"):
        if not 0.0 < float(exposure[key]) < 1.0:
            raise ValueError(
                f"augmentation.exposure_normalization.{key} must be in (0, 1)"
            )
    if not 0.0 < float(exposure["min_gain"]) <= float(exposure["max_gain"]):
        raise ValueError(
            "augmentation.exposure_normalization gains must satisfy 0 < min <= max"
        )
