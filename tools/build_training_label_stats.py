#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ck3_training.schema import CK3Schema, load_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-only target statistics from CK3 tar shards."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("face_to_ck3_dataset_male_small/processed_front"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("face_to_ck3_dataset_male_small/dna_schema.json"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def empty_stats(schema: CK3Schema) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "signed_sum": [0.0] * schema.signed_dim,
        "strength_sum": [0.0] * schema.categorical_dim,
        "color_sum": [0.0] * schema.color_dim,
        "class_counts": [
            [0] * len(field.classes) for field in schema.categorical_fields
        ],
    }


def add_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["sample_count"] += source["sample_count"]
    for key in ("signed_sum", "strength_sum", "color_sum"):
        for index, value in enumerate(source[key]):
            target[key][index] += value
    for field_index, counts in enumerate(source["class_counts"]):
        for class_index, value in enumerate(counts):
            target["class_counts"][field_index][class_index] += value


def scan_shard(path: Path, schema: CK3Schema) -> dict[str, Any]:
    stats = empty_stats(schema)
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot extract {member.name} from {path}")
            label = json.loads(stream.read().decode("utf-8"))
            schema.validate_label(label)
            stats["sample_count"] += 1
            for index, value in enumerate(label["signed"]):
                stats["signed_sum"][index] += float(value)
            for index, value in enumerate(label["categorical_strength"]):
                stats["strength_sum"][index] += float(value)
            for index, value in enumerate(label["colors"]):
                stats["color_sum"][index] += float(value)
            for field_index, class_id in enumerate(label["categorical_class"]):
                stats["class_counts"][field_index][int(class_id)] += 1
    return stats


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    schema = load_schema(args.schema)
    shards = sorted((args.data_root / args.split).glob(f"{args.split}-*.tar"))
    if not shards:
        raise SystemExit(f"no {args.split} shards under {args.data_root}")
    total = empty_stats(schema)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, stats in enumerate(
            executor.map(lambda path: scan_shard(path, schema), shards), start=1
        ):
            add_stats(total, stats)
            if index % 10 == 0 or index == len(shards):
                print(
                    f"scanned {index}/{len(shards)} shards, "
                    f"samples={total['sample_count']}",
                    flush=True,
                )
    count = max(1, total["sample_count"])
    output = {
        "version": 1,
        "split": args.split,
        "schema_sha256": schema.sha256,
        "sample_count": total["sample_count"],
        "signed_mean": [value / count for value in total["signed_sum"]],
        "categorical_strength_mean": [
            value / count for value in total["strength_sum"]
        ],
        "color_mean": [value / count for value in total["color_sum"]],
        "categorical_class_counts": total["class_counts"],
    }
    destination = args.output or args.data_root / f"{args.split}_label_stats.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
