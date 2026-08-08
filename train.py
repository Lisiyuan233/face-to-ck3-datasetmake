#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
except ImportError as error:
    print(
        "PyTorch is not installed. Install requirements-train.txt in a Python 3.10+ "
        "environment before running train.py.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error

from ck3_training.config import (
    apply_smoke_overrides,
    load_config,
    save_config,
    validate_config,
)
from ck3_training.data import (
    DualViewTransform,
    TarShardDataset,
    discover_all_shards,
    discover_shards,
)
from ck3_training.engine import (
    ExponentialMovingAverage,
    append_jsonl,
    atomic_checkpoint,
    cosine_warmup_scheduler,
    evaluate,
    raw_model,
    seed_everything,
    seed_worker,
    train_one_epoch,
)
from ck3_training.losses import MultitaskLoss
from ck3_training.metrics import (
    categorical_selection_score,
    continuous_selection_score,
    selection_score,
)
from ck3_training.model import FaceToCK3Model
from ck3_training.sampling import evenly_spaced_fraction
from ck3_training.schema import TARGET_FAMILY, load_schema
from ck3_training.split_index import (
    file_sha256,
    load_split_ids,
    load_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-task face-to-CK3-DNA model."
    )
    parser.add_argument(
        "--config", default="configs/train_convnext_tiny.json", type=Path
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-selection-state",
        action="store_true",
        help=(
            "recompute checkpoint-selection baselines from the resumed epoch "
            "and clear early-stopping patience"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run two CPU-friendly train/validation steps with ResNet-18",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        help=(
            "use this fraction of train/validation data, for example 0.1 for "
            "a reproducible 10%% subset"
        ),
    )
    return parser.parse_args()


def distributed_context(device_choice: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and device_choice != "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def make_loader(
    dataset: TarShardDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "num_workers": int(workers),
        "pin_memory": device.type == "cuda",
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_resume(
    path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    ema: ExponentialMovingAverage,
    schema_sha256: str,
    device: torch.device,
    target_family: str = TARGET_FAMILY,
    split_index_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_split_index_sha256 = checkpoint.get("split_index_sha256")
    if saved_split_index_sha256 != split_index_sha256:
        raise RuntimeError(
            "checkpoint split index does not match the current data split; "
            "start a new run"
        )
    saved_target_family = checkpoint.get("schema", {}).get("target_family")
    if saved_target_family != target_family:
        raise RuntimeError(
            f"checkpoint target family {saved_target_family!r} does not match "
            f"current target family {target_family!r}; start a new run"
        )
    saved_side_view = bool(
        checkpoint.get("config", {}).get("model", {}).get("side_view", False)
    )
    current_side_view = bool(getattr(model, "use_side_view", False))
    if saved_side_view != current_side_view:
        raise RuntimeError(
            "checkpoint side-view architecture does not match the current config; "
            "start a new run"
        )
    saved_geometry_branch = bool(
        checkpoint.get("config", {})
        .get("model", {})
        .get("geometry_branch", {})
        .get("enabled", False)
    )
    current_geometry_branch = bool(
        getattr(model, "use_geometry_branch", False)
    )
    if saved_geometry_branch != current_geometry_branch:
        raise RuntimeError(
            "checkpoint geometry-branch architecture does not match the current "
            "config; start a new run"
        )
    if current_geometry_branch:
        saved_geometry_config = (
            checkpoint.get("config", {})
            .get("model", {})
            .get("geometry_branch", {})
        )
        saved_geometry_targets = frozenset(
            saved_geometry_config.get(
                "targets", ("signed", "strength", "categorical")
            )
        )
        current_geometry_targets = frozenset(
            getattr(model, "geometry_targets", ())
        )
        if saved_geometry_targets != current_geometry_targets:
            raise RuntimeError(
                "checkpoint geometry-branch targets do not match the current "
                "config; start a new run"
            )
    saved_schema = checkpoint.get("schema", {}).get("schema_sha256")
    if saved_schema != schema_sha256:
        raise RuntimeError(
            f"checkpoint schema {saved_schema} does not match current schema {schema_sha256}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("ema"):
        ema.load_state_dict(checkpoint["ema"])
    return checkpoint


def _selection_signature(config: dict[str, Any]) -> dict[str, float | int]:
    return {
        "scalar_mae_weight": float(config.get("scalar_mae_weight", 0.0)),
        "signed_mae_weight": float(config["signed_mae_weight"]),
        "strength_mae_weight": float(config["strength_mae_weight"]),
        "categorical_error_weight": float(config["categorical_error_weight"]),
        "categorical_min_observable_count": int(
            config.get("categorical_min_observable_count", 0)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.reset_selection_state and args.resume is None:
        raise ValueError("--reset-selection-state requires --resume")
    rank, world_size, local_rank, device = distributed_context(args.device)
    config = load_config(args.config)
    if args.data_fraction is not None:
        config["data"]["fraction"] = args.data_fraction
    if args.smoke_test:
        config = apply_smoke_overrides(config)
    validate_config(config)
    seed_everything(int(config["seed"]), rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    train_config = config["train"]
    schema = load_schema(data_config["schema"])
    manifest_path = Path(data_config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_index_path = data_config.get("split_index")
    split_ids: dict[str, frozenset[str]] = {}
    if split_index_path:
        split_index_path = Path(split_index_path)
        split_manifest = load_split_manifest(split_index_path)
        if str(split_manifest.get("schema_sha256", "")) != schema.sha256:
            raise RuntimeError("split index uses a different training schema")
        counts = {
            key: int(value) for key, value in split_manifest["counts"].items()
        }
        split_ids = {
            split: load_split_ids(split_index_path, split)
            for split in ("train", "val")
        }
        overlap = split_ids["train"] & split_ids["val"]
        if overlap:
            raise RuntimeError(
                f"split index has {len(overlap)} sample ids in both train and val"
            )
        grouped_split_sha256 = file_sha256(split_index_path)
    else:
        counts = {
            key: int(value)
            for key, value in manifest["split"]["counts"].items()
        }
        grouped_split_sha256 = None
    use_side_view = bool(config["model"].get("side_view", False))
    if use_side_view and "side" not in manifest.get("crops", {}):
        raise RuntimeError(
            "model.side_view=true requires a paired front/side dataset; "
            "regenerate shards with image_preprocessor.py v2"
        )
    stats_path = Path(data_config["train_label_stats"])
    if not stats_path.is_file():
        raise RuntimeError(
            f"missing {stats_path}; run tools/build_training_label_stats.py first"
        )
    label_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if label_stats.get("split") != "train":
        raise RuntimeError("label statistics must be generated from the train split")
    stats_schema_sha256 = str(label_stats.get("schema_sha256", ""))
    if not schema.accepts_statistics_schema(stats_schema_sha256):
        raise RuntimeError("train label statistics use an incompatible schema")
    if int(label_stats.get("sample_count", -1)) != counts["train"]:
        raise RuntimeError("train label statistics sample count does not match manifest")
    if grouped_split_sha256 is not None and str(
        label_stats.get("split_index_sha256", "")
    ) != grouped_split_sha256:
        raise RuntimeError("train label statistics use a different split index")
    use_schema_visibility = bool(
        config["loss"].get("use_schema_visibility_thresholds", False)
    )
    observable_class_counts = None
    if not use_schema_visibility:
        observable_class_counts = label_stats.get(
            "categorical_observable_class_counts"
        )
    if observable_class_counts is not None:
        stats_threshold = float(label_stats.get("observable_threshold", -1.0))
        configured_threshold = float(
            config["loss"]["class_visibility_threshold"]
        )
        if abs(stats_threshold - configured_threshold) > 1e-9:
            raise RuntimeError(
                "train label statistics observable threshold does not match config"
            )
    else:
        observable_class_counts = label_stats["categorical_class_counts"]
    observable_class_counts = schema.adapt_class_counts(
        observable_class_counts
    )
    data_root = Path(data_config["root"])
    if split_index_path:
        train_shards = discover_all_shards(data_root)
        val_shards = list(train_shards)
    else:
        train_shards = discover_shards(data_root, "train")
        val_shards = discover_shards(data_root, "val")
    data_fraction = float(data_config["fraction"])
    minimum_train_shards = world_size * max(1, int(train_config["num_workers"]))
    train_shards = evenly_spaced_fraction(
        train_shards,
        data_fraction,
        minimum=minimum_train_shards,
    )
    if minimum_train_shards > len(train_shards):
        raise RuntimeError(
            "world_size * num_workers exceeds the number of training shards"
        )

    train_transform = DualViewTransform(
        data_config["image_height"],
        data_config["image_width"],
        config["augmentation"],
        training=True,
        dual_view=bool(config["model"]["dual_view"]),
        geometry_map_config=config["model"]["geometry_branch"],
    )
    eval_transform = DualViewTransform(
        data_config["image_height"],
        data_config["image_width"],
        config["augmentation"],
        training=False,
        dual_view=bool(config["model"]["dual_view"]),
        geometry_map_config=config["model"]["geometry_branch"],
    )
    train_dataset = TarShardDataset(
        train_shards,
        schema,
        train_transform,
        training=True,
        repeat=True,
        shuffle_buffer=int(data_config["shuffle_buffer"]),
        seed=int(config["seed"]),
        sample_fraction=1.0,
        sample_ids=split_ids.get("train"),
        require_side_view=use_side_view,
        rank=rank,
        world_size=world_size,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        workers=int(train_config["num_workers"]),
        device=device,
        drop_last=True,
    )

    validation_loader = None
    if rank == 0:
        validation_dataset = TarShardDataset(
            val_shards,
            schema,
            eval_transform,
            training=False,
            repeat=False,
            shuffle_buffer=1,
            seed=int(config["seed"]),
            sample_fraction=data_fraction,
            sample_ids=split_ids.get("val"),
            require_side_view=use_side_view,
            rank=0,
            world_size=1,
        )
        validation_loader = make_loader(
            validation_dataset,
            batch_size=int(train_config["batch_size"]),
            workers=int(train_config["val_num_workers"]),
            device=device,
            drop_last=False,
        )

    model_config = dict(config["model"])
    if args.resume:
        model_config["pretrained"] = False
    model = FaceToCK3Model(schema, model_config).to(device)
    criterion = MultitaskLoss(
        schema, config["loss"], observable_class_counts
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            train_config["backbone_lr"],
            train_config["head_lr"],
            train_config["weight_decay"],
        ),
        betas=(0.9, 0.999),
    )

    effective_train_count = max(1, math.ceil(counts["train"] * data_fraction))
    expected_val_count = max(1, math.ceil(counts["val"] * data_fraction))
    micro_steps = math.ceil(
        effective_train_count
        / (int(train_config["batch_size"]) * max(1, world_size))
    )
    if train_config.get("max_train_steps") is not None:
        micro_steps = int(train_config["max_train_steps"])
    accumulation = int(train_config["gradient_accumulation"])
    optimizer_steps_per_epoch = math.ceil(micro_steps / accumulation)
    total_optimizer_steps = optimizer_steps_per_epoch * int(train_config["epochs"])
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_optimizer_steps,
        train_config["warmup_ratio"],
        train_config["min_lr_ratio"],
    )
    amp_mode = str(train_config["amp"]).lower()
    scaler_enabled = device.type == "cuda" and amp_mode == "fp16"
    scaler = make_scaler(device, scaler_enabled)
    ema = ExponentialMovingAverage(model, float(train_config["ema_decay"]))

    start_epoch = 0
    global_step = 0
    best_score = float("inf")
    best_continuous_score = float("inf")
    best_categorical_score = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = load_resume(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            schema_sha256=schema.sha256,
            device=device,
            target_family=schema.target_family,
            split_index_sha256=grouped_split_sha256,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        saved_selection = _selection_signature(
            checkpoint.get("config", {}).get("selection", {})
        )
        current_selection = _selection_signature(config["selection"])
        selection_changed = saved_selection != current_selection
        if selection_changed and not args.reset_selection_state:
            raise RuntimeError(
                "selection config changed since the checkpoint; resume with "
                "--reset-selection-state"
            )
        if args.reset_selection_state:
            validation = checkpoint.get("validation")
            if not isinstance(validation, dict):
                raise RuntimeError(
                    "cannot reset selection state: checkpoint has no validation metrics"
                )
            best_score = selection_score(validation, config["selection"])
            best_continuous_score = continuous_selection_score(
                validation, config["selection"]
            )
            categorical_score = categorical_selection_score(
                validation, config["selection"]
            )
            if categorical_score is not None:
                best_categorical_score = categorical_score
            epochs_without_improvement = 0
        else:
            best_score = float(checkpoint.get("best_score", float("inf")))
            best_continuous_score = float(
                checkpoint.get("best_continuous_score", float("inf"))
            )
            best_categorical_score = float(
                checkpoint.get("best_categorical_score", float("inf"))
            )
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=int(train_config["freeze_backbone_epochs"]) > 0,
        )

    output_dir = Path(train_config["output_dir"])
    log_path = output_dir / "metrics.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "resolved_config.json")
        (output_dir / "schema_metadata.json").write_text(
            json.dumps(schema.checkpoint_metadata(), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "loss_profile.json").write_text(
            json.dumps(
                criterion.profile_metadata(), ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"device={device} world_size={world_size} "
            f"data_fraction={data_fraction:.4f} "
            f"train~={effective_train_count} val~={expected_val_count} "
            f"train_shards={len(train_shards)} micro_steps/epoch={micro_steps}",
            flush=True,
        )

    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_dataset.set_epoch(epoch)
        frozen = epoch < int(train_config["freeze_backbone_epochs"])
        raw_model(model).set_backbone_trainable(not frozen)
        train_metrics, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            micro_steps=micro_steps,
            accumulation_steps=accumulation,
            grad_clip=float(train_config["grad_clip"]),
            amp_mode=amp_mode,
            log_every=int(train_config["log_every"]),
            log_path=log_path,
            rank=rank,
            global_step=global_step,
            backbone_frozen=frozen,
        )

        stop = False
        if rank == 0:
            assert validation_loader is not None
            with ema.average_parameters(raw_model(model)):
                validation_metrics = evaluate(
                    model=raw_model(model),
                    criterion=criterion,
                    loader=validation_loader,
                    schema=schema,
                    device=device,
                    amp_mode=amp_mode,
                    max_steps=train_config.get("max_val_steps"),
                    observable_threshold=criterion.observable_thresholds(),
                )
            score = selection_score(validation_metrics, config["selection"])
            continuous_score = continuous_selection_score(
                validation_metrics, config["selection"]
            )
            categorical_score = categorical_selection_score(
                validation_metrics, config["selection"]
            )
            min_delta = float(train_config["early_stopping_min_delta"])
            improved = score < best_score - min_delta
            continuous_improved = (
                continuous_score < best_continuous_score - min_delta
            )
            categorical_improved = (
                categorical_score is not None
                and categorical_score < best_categorical_score - min_delta
            )
            if improved:
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if continuous_improved:
                best_continuous_score = continuous_score
            if categorical_improved:
                assert categorical_score is not None
                best_categorical_score = categorical_score

            record = {
                "type": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "continuous_selection_score": continuous_score,
                "categorical_selection_score": categorical_score,
                "best_score": best_score,
                "best_continuous_score": best_continuous_score,
                "best_categorical_score": (
                    best_categorical_score
                    if best_categorical_score < float("inf")
                    else None
                ),
                "improved": improved,
                "continuous_improved": continuous_improved,
                "categorical_improved": categorical_improved,
            }
            append_jsonl(log_path, record)
            (output_dir / f"validation-epoch-{epoch:03d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "best_continuous_score": best_continuous_score,
                "best_categorical_score": best_categorical_score,
                "epochs_without_improvement": epochs_without_improvement,
                "model": raw_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
                "config": config,
                "schema": schema.checkpoint_metadata(),
                "split_index_sha256": grouped_split_sha256,
                "validation": validation_metrics,
            }
            atomic_checkpoint(output_dir / "last.pt", state)
            if improved:
                atomic_checkpoint(output_dir / "best.pt", state)
            if continuous_improved:
                atomic_checkpoint(output_dir / "best_continuous.pt", state)
            if categorical_improved:
                atomic_checkpoint(output_dir / "best_categorical.pt", state)
            geometry_gate_text = ""
            if "geometry_gate_mean" in validation_metrics:
                geometry_gate_text = (
                    " geometry_gate="
                    f"{validation_metrics['geometry_gate_mean']:.4f}"
                )
            scalar_text = ""
            if "scalar_mae" in validation_metrics:
                scalar_text = (
                    f"scalar_mae={validation_metrics['scalar_mae']:.5f} "
                )
            print(
                f"epoch={epoch} val_score={score:.6f} "
                f"{scalar_text}"
                f"signed_mae={validation_metrics['signed_mae']:.5f} "
                "observable_macro_f1="
                f"{validation_metrics['categorical_observable_macro_f1']:.4f}"
                f"{geometry_gate_text}",
                flush=True,
            )
            stop = epochs_without_improvement >= int(
                train_config["early_stopping_patience"]
            )

        if world_size > 1:
            stop_tensor = torch.tensor(
                [1 if stop else 0], device=device, dtype=torch.int32
            )
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
            dist.barrier()
        if stop:
            if rank == 0:
                print("early stopping", flush=True)
            break

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
