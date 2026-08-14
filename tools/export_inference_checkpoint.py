#!/usr/bin/env python3
"""Export a compact, EMA-baked checkpoint for the packaged inference app."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


FORMAT_NAME = "face-to-ck3-inference-v1"


def _cpu_state_dict(
    model_state: Mapping[str, Any], ema_state: Mapping[str, Any]
) -> OrderedDict[str, Any]:
    missing = sorted(set(ema_state) - set(model_state))
    if missing:
        raise ValueError("EMA 中存在模型没有的参数: " + ", ".join(missing[:5]))

    result: OrderedDict[str, Any] = OrderedDict()
    for name, value in model_state.items():
        selected = ema_state.get(name, value)
        result[name] = selected.detach().cpu() if torch.is_tensor(selected) else selected
    return result


def export_inference_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("输出路径不能覆盖训练 checkpoint")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    ema_state = checkpoint.get("ema", {}).get("shadow")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint 缺少 model state dict")
    if not isinstance(ema_state, Mapping) or not ema_state:
        raise ValueError("checkpoint 缺少 EMA state dict")
    for required in ("config", "schema"):
        if required not in checkpoint:
            raise ValueError(f"checkpoint 缺少 {required}")

    payload = {
        "format": FORMAT_NAME,
        "inference_weight_source": "ema",
        "model": _cpu_state_dict(model_state, ema_state),
        "config": checkpoint["config"],
        "schema": checkpoint["schema"],
        "source": {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "best_score": checkpoint.get("best_score"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = export_inference_checkpoint(args.source, args.output)
    parameter_count = sum(
        value.numel() for value in payload["model"].values() if torch.is_tensor(value)
    )
    size_mib = args.output.resolve().stat().st_size / (1024 * 1024)
    print(
        f"exported {parameter_count:,} values to {args.output.resolve()} "
        f"({size_mib:.1f} MiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
