from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import MetricAccumulator


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank) * 1009
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def move_targets(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "signed",
        "categorical_class",
        "categorical_strength",
        "race_group",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def amp_settings(device: torch.device, mode: str) -> tuple[bool, torch.dtype, bool]:
    mode = mode.lower()
    if device.type != "cuda" or mode == "off":
        return False, torch.float32, False
    if mode == "bf16":
        supported = bool(
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if supported:
            return True, torch.bfloat16, False
        mode = "fp16"
    if mode == "fp16":
        return True, torch.float16, True
    raise ValueError(f"unsupported AMP mode: {mode}")


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()
            else:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        if self.decay <= 0 or not self.shadow:
            yield
            return
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = round(total_steps * float(warmup_ratio))
    total_steps = max(1, int(total_steps))

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def raw_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    ema: ExponentialMovingAverage,
    device: torch.device,
    epoch: int,
    micro_steps: int,
    accumulation_steps: int,
    grad_clip: float,
    amp_mode: str,
    log_every: int,
    log_path: Path,
    rank: int,
    global_step: int,
    backbone_frozen: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    if backbone_frozen:
        raw_model(model).backbone.eval()
    enabled, dtype, use_scaler = amp_settings(device, amp_mode)
    optimizer.zero_grad(set_to_none=True)
    component_sums: dict[str, float] = {}
    sample_count = 0
    started = time.perf_counter()
    iterator = iter(loader)

    for micro_step in range(micro_steps):
        batch = next(iterator)
        geometry = batch["geometry_view"].to(device, non_blocking=True)
        reference = batch["reference_view"].to(device, non_blocking=True)
        side = batch.get("side_view")
        if side is not None:
            side = side.to(device, non_blocking=True)
        targets = move_targets(batch, device)
        final_micro_step = micro_step + 1 == micro_steps
        should_update = ((micro_step + 1) % accumulation_steps == 0) or final_micro_step
        sync_context = contextlib.nullcontext()
        if hasattr(model, "no_sync") and not should_update:
            sync_context = model.no_sync()
        with sync_context:
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                outputs = model(geometry, reference, side)
                loss, components = criterion(outputs, targets)
                accumulation_group_start = (
                    micro_step // accumulation_steps
                ) * accumulation_steps
                accumulation_group_size = min(
                    accumulation_steps, micro_steps - accumulation_group_start
                )
                scaled_loss = loss / accumulation_group_size
            scaler.scale(scaled_loss).backward()

        batch_size = geometry.shape[0]
        sample_count += batch_size
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + float(
                value.detach().item()
            )

        if should_update:
            if use_scaler:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(raw_model(model))
            global_step += 1

        if rank == 0 and (micro_step + 1) % max(1, log_every) == 0:
            elapsed = max(1e-6, time.perf_counter() - started)
            record = {
                "type": "train_step",
                "epoch": epoch,
                "micro_step": micro_step + 1,
                "global_step": global_step,
                "samples_per_second": sample_count / elapsed,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "loss": {
                    key: value / (micro_step + 1)
                    for key, value in component_sums.items()
                },
            }
            append_jsonl(log_path, record)
            print(
                f"epoch={epoch} step={micro_step + 1}/{micro_steps} "
                f"loss={record['loss'].get('total', 0):.5f} "
                f"samples/s={record['samples_per_second']:.1f}",
                flush=True,
            )

    elapsed = max(1e-6, time.perf_counter() - started)
    summary = {
        key: value / micro_steps for key, value in component_sums.items()
    }
    summary["samples_per_second"] = sample_count / elapsed
    return summary, global_step


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    schema,
    device: torch.device,
    amp_mode: str,
    max_steps: int | None,
    observable_threshold: float = 0.10,
) -> dict[str, Any]:
    model.eval()
    enabled, dtype, _ = amp_settings(device, amp_mode)
    metrics = MetricAccumulator(
        schema, device, observable_threshold=observable_threshold
    )
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= int(max_steps):
            break
        geometry = batch["geometry_view"].to(device, non_blocking=True)
        reference = batch["reference_view"].to(device, non_blocking=True)
        side = batch.get("side_view")
        if side is not None:
            side = side.to(device, non_blocking=True)
        targets = move_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            outputs = model(geometry, reference, side)
            _, components = criterion(outputs, targets)
        metrics.update(outputs, targets, components)
    return metrics.compute()
