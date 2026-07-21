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
    discover_shards,
    manifest_counts,
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
from ck3_training.metrics import selection_score
from ck3_training.model import FaceToCK3Model
from ck3_training.schema import load_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-task face-to-CK3-DNA model."
    )
    parser.add_argument(
        "--config", default="configs/train_convnext_tiny.json", type=Path
    )
    parser.add_argument("--resume", type=Path)
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
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_context(args.device)
    config = load_config(args.config)
    if args.smoke_test:
        config = apply_smoke_overrides(config)
    validate_config(config)
    seed_everything(int(config["seed"]), rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    train_config = config["train"]
    schema = load_schema(data_config["schema"])
    counts = manifest_counts(data_config["manifest"])
    stats_path = Path(data_config["train_label_stats"])
    if not stats_path.is_file():
        raise RuntimeError(
            f"missing {stats_path}; run tools/build_training_label_stats.py first"
        )
    label_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if label_stats.get("split") != "train":
        raise RuntimeError("label statistics must be generated from the train split")
    if label_stats.get("schema_sha256") != schema.sha256:
        raise RuntimeError("train label statistics use a different schema")
    if int(label_stats.get("sample_count", -1)) != counts["train"]:
        raise RuntimeError("train label statistics sample count does not match manifest")
    data_root = Path(data_config["root"])
    train_shards = discover_shards(data_root, "train")
    val_shards = discover_shards(data_root, "val")
    if world_size * max(1, int(train_config["num_workers"])) > len(train_shards):
        raise RuntimeError(
            "world_size * num_workers exceeds the number of training shards"
        )

    train_transform = DualViewTransform(
        data_config["image_height"],
        data_config["image_width"],
        config["augmentation"],
        training=True,
        dual_view=bool(config["model"]["dual_view"]),
    )
    eval_transform = DualViewTransform(
        data_config["image_height"],
        data_config["image_width"],
        config["augmentation"],
        training=False,
        dual_view=bool(config["model"]["dual_view"]),
    )
    train_dataset = TarShardDataset(
        train_shards,
        schema,
        train_transform,
        training=True,
        repeat=True,
        shuffle_buffer=int(data_config["shuffle_buffer"]),
        seed=int(config["seed"]),
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
        schema, config["loss"], label_stats["categorical_class_counts"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            train_config["backbone_lr"],
            train_config["head_lr"],
            train_config["weight_decay"],
        ),
        betas=(0.9, 0.999),
    )

    micro_steps = math.ceil(
        counts["train"]
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
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("best_score", float("inf")))
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
        print(
            f"device={device} world_size={world_size} train={counts['train']} "
            f"val={counts['val']} micro_steps/epoch={micro_steps}",
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
                )
            score = selection_score(validation_metrics)
            improved = score < best_score
            if improved:
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            record = {
                "type": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "best_score": best_score,
                "improved": improved,
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
                "epochs_without_improvement": epochs_without_improvement,
                "model": raw_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
                "config": config,
                "schema": schema.checkpoint_metadata(),
                "validation": validation_metrics,
            }
            atomic_checkpoint(output_dir / "last.pt", state)
            if improved:
                atomic_checkpoint(output_dir / "best.pt", state)
            print(
                f"epoch={epoch} val_score={score:.6f} "
                f"signed_mae={validation_metrics['signed_mae']:.5f} "
                f"macro_f1={validation_metrics['categorical_macro_f1']:.4f}",
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
