#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a held-out shard split.

The evaluator is intentionally read-only: it restores model or EMA weights,
computes the same metrics used during validation, and compares continuous
targets with a train-median baseline. It never updates the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as error:
    raise SystemExit("PyTorch is required; install requirements-train.txt") from error

from ck3_training.data import DualViewTransform, TarShardDataset, discover_shards
from ck3_training.engine import evaluate, seed_worker
from ck3_training.losses import MultitaskLoss
from ck3_training.metrics import (
    categorical_selection_score,
    continuous_selection_score,
    selection_score,
)
from ck3_training.model import FaceToCK3Model
from ck3_training.schema import CK3Schema, load_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--field-csv", type=Path)
    return parser.parse_args()


def _device(choice: str) -> torch.device:
    use_cuda = torch.cuda.is_available() and choice != "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if use_cuda else "cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _load_train_statistics(
    path: Path, schema: CK3Schema, use_schema_visibility: bool
) -> list[list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("split") != "train":
        raise RuntimeError("label statistics are not from the train split")
    if not schema.accepts_statistics_schema(str(value.get("schema_sha256", ""))):
        raise RuntimeError("label statistics use an incompatible schema")
    key = (
        "categorical_class_counts"
        if use_schema_visibility
        else "categorical_observable_class_counts"
    )
    counts = value.get(key, value["categorical_class_counts"])
    return schema.adapt_class_counts(counts)


def _load_model(
    checkpoint: dict[str, Any], schema: CK3Schema, device: torch.device, raw: bool
) -> tuple[FaceToCK3Model, str]:
    saved = checkpoint.get("schema", {})
    if saved.get("target_family") != schema.target_family:
        raise RuntimeError("checkpoint target family does not match schema")
    if saved.get("schema_sha256") != schema.sha256:
        raise RuntimeError("checkpoint schema hash does not match schema")
    model_config = dict(checkpoint["config"]["model"])
    model_config["pretrained"] = False
    model = FaceToCK3Model(schema, model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    source = "raw"
    shadow = checkpoint.get("ema", {}).get("shadow", {})
    if not raw and shadow:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in shadow:
                    parameter.copy_(shadow[name].to(device=device, dtype=parameter.dtype))
        source = "ema"
    model.eval()
    return model, source


def _iter_labels(shards: Iterable[Path], schema: CK3Schema) -> Iterable[dict[str, Any]]:
    for path in shards:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".json"):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot extract {member.name} from {path}")
                source = json.loads(stream.read().decode("utf-8"))
                yield schema.adapt_label(source)


def _target_matrices(
    shards: Iterable[Path], schema: CK3Schema
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[list[float]]] = {
        "scalar": [],
        "signed": [],
        "strength": [],
    }
    for label in _iter_labels(shards, schema):
        rows["scalar"].append([float(value) for value in label.get("scalar", ())])
        rows["signed"].append([float(value) for value in label["signed"]])
        rows["strength"].append(
            [float(value) for value in label["categorical_strength"]]
        )
    return {
        key: torch.tensor(value, dtype=torch.float64)
        for key, value in rows.items()
    }


def _median_baseline(
    *,
    train_shards: list[Path],
    target_shards: list[Path],
    schema: CK3Schema,
    metrics: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = _target_matrices(train_shards, schema)
    target = _target_matrices(target_shards, schema)
    families = (
        ("scalar", list(schema.scalar_fields), "scalar_mae_by_field"),
        ("signed", list(schema.signed_fields), "signed_mae_by_field"),
        (
            "strength",
            [field.name for field in schema.categorical_fields],
            "strength_mae_by_field",
        ),
    )
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    baseline_means: dict[str, float] = {}
    for family, names, metric_key in families:
        if not names:
            continue
        median = torch.quantile(train[family], 0.5, dim=0)
        baseline = (target[family] - median).abs().mean(dim=0)
        model = torch.tensor(metrics[metric_key], dtype=torch.float64)
        improvement = (baseline - model) / baseline.clamp_min(1e-12)
        baseline_means[family] = float(baseline.mean().item())
        family_rows = []
        for name, baseline_mae, model_mae, gain in zip(
            names, baseline.tolist(), model.tolist(), improvement.tolist()
        ):
            row = {
                "family": family,
                "field": name,
                "baseline_mae": float(baseline_mae),
                "model_mae": float(model_mae),
                "improvement": float(gain),
            }
            rows.append(row)
            family_rows.append(row)
        gains = improvement.sort().values
        summary[family] = {
            "baseline_mae": baseline_means[family],
            "model_mae": float(model.mean().item()),
            "improvement": float(
                (baseline.mean() - model.mean()) / baseline.mean().clamp_min(1e-12)
            ),
            "positive_field_count": int((improvement > 0).sum().item()),
            "field_count": len(names),
            "gain_ge_10pct_count": int((improvement >= 0.10).sum().item()),
            "gain_ge_25pct_count": int((improvement >= 0.25).sum().item()),
            "median_field_improvement": float(
                gains[len(gains) // 2].item()
                if len(gains) % 2
                else gains[len(gains) // 2 - 1 : len(gains) // 2 + 1].mean().item()
            ),
            "best_fields": sorted(
                family_rows, key=lambda row: row["improvement"], reverse=True
            )[:8],
            "worst_fields": sorted(
                family_rows, key=lambda row: row["improvement"]
            )[:8],
        }
    weights = {
        family: weight
        for family, weight in {
            "scalar": float(selection.get("scalar_mae_weight", 0.0)),
            "signed": float(selection["signed_mae_weight"]),
            "strength": float(selection["strength_mae_weight"]),
        }.items()
        if weight > 0.0 and family in baseline_means
    }
    denominator = sum(weights.values())
    summary["continuous_score"] = sum(
        weights[family] * baseline_means[family] for family in weights
    ) / denominator
    summary["target_sample_count"] = int(target["signed"].shape[0])
    summary["train_sample_count"] = int(train["signed"].shape[0])
    return summary, rows


def _write_field_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "family",
                    "field",
                    "baseline_mae",
                    "model_mae",
                    "improvement",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.workers is not None and args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    checkpoint_path = args.checkpoint.resolve()
    device = _device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_config = config["data"]
    train_config = config["train"]
    schema = load_schema(Path(data_config["schema"]))
    data_root = Path(data_config["root"])
    target_shards = discover_shards(data_root, args.split)
    train_shards = discover_shards(data_root, "train")
    model, weight_source = _load_model(
        checkpoint, schema, device, raw=args.raw_weights
    )

    loss_config = config["loss"]
    class_counts = _load_train_statistics(
        Path(data_config["train_label_stats"]),
        schema,
        bool(loss_config.get("use_schema_visibility_thresholds", False)),
    )
    criterion = MultitaskLoss(schema, loss_config, class_counts).to(device)
    model_config = config["model"]
    transform = DualViewTransform(
        data_config["image_height"],
        data_config["image_width"],
        config["augmentation"],
        training=False,
        dual_view=bool(model_config["dual_view"]),
        geometry_map_config=model_config.get("geometry_branch"),
    )
    dataset = TarShardDataset(
        target_shards,
        schema,
        transform,
        training=False,
        repeat=False,
        shuffle_buffer=1,
        seed=int(config["seed"]),
        require_side_view=bool(model_config.get("side_view", False)),
    )
    workers = (
        int(args.workers)
        if args.workers is not None
        else min(int(train_config.get("val_num_workers", 0)), len(target_shards))
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size or train_config["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    if workers:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    metrics = evaluate(
        model=model,
        criterion=criterion,
        loader=loader,
        schema=schema,
        device=device,
        amp_mode=str(train_config["amp"]),
        max_steps=None,
        observable_threshold=criterion.observable_thresholds(),
    )
    selection_config = config["selection"]
    baseline, field_rows = _median_baseline(
        train_shards=train_shards,
        target_shards=target_shards,
        schema=schema,
        metrics=metrics,
        selection=selection_config,
    )
    test_score = selection_score(metrics, selection_config)
    categorical_score = categorical_selection_score(metrics, selection_config)
    output = args.output or checkpoint_path.parent / f"{args.split}-evaluation.json"
    field_csv = args.field_csv or checkpoint_path.parent / f"{args.split}-field-improvement.csv"
    result = {
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "weight_source": weight_source,
        "split": args.split,
        "device": str(device),
        "batch_size": int(loader_kwargs["batch_size"]),
        "workers": workers,
        "schema_sha256": schema.sha256,
        "metrics": metrics,
        "selection_score": test_score,
        "continuous_selection_score": continuous_selection_score(
            metrics, selection_config
        ),
        "categorical_selection_score": categorical_score,
        "train_median_baseline": baseline,
        "continuous_score_improvement": (
            baseline["continuous_score"] - test_score
        ) / baseline["continuous_score"],
        "notes": {
            "test_split_was_not_used_for_training_or_checkpoint_selection": True,
            "categorical_class_loss_weight": float(loss_config["class_weight"]),
            "categorical_metrics_are_report_only": float(loss_config["class_weight"])
            == 0.0,
        },
    }
    _atomic_json(output, result)
    _write_field_csv(field_csv, field_rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "field_csv": str(field_csv),
                "split": args.split,
                "samples": metrics["sample_count"],
                "weight_source": weight_source,
                "selection_score": test_score,
                "baseline_score": baseline["continuous_score"],
                "improvement": result["continuous_score_improvement"],
                "scalar_mae": metrics.get("scalar_mae"),
                "signed_mae": metrics["signed_mae"],
                "strength_mae": metrics["strength_mae"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
