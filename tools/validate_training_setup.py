#!/usr/bin/env python3
"""Validate a training config, schema, statistics, and one label per split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ck3_training.config import load_config, validate_config
from ck3_training.schema import CK3Schema, load_schema
from ck3_training.split_index import (
    file_sha256,
    load_split_ids,
    load_split_manifest,
)


def _first_json_label(
    paths: list[Path], sample_ids: frozenset[str] | None = None
) -> tuple[dict[str, Any], Path]:
    for path in paths:
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".json"):
                    continue
                if sample_ids is not None and Path(member.name).stem not in sample_ids:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot extract {member.name} from {path}")
                return json.loads(stream.read().decode("utf-8")), path
    raise RuntimeError("no matching JSON label in configured shards")


def _validate_split(
    shards: list[Path],
    split: str,
    schema: CK3Schema,
    sample_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not shards:
        raise FileNotFoundError(f"no shards configured for {split}")
    source, source_shard = _first_json_label(shards, sample_ids)
    adapted = schema.adapt_label(source)
    schema.validate_label(adapted)
    return {
        "shard_count": len(shards),
        "first_shard": str(source_shard),
        "sample_id": adapted.get("sample_id"),
        "source_signed_dim": len(source.get("signed", ())),
        "scalar_dim": len(adapted.get("scalar", ())),
        "signed_dim": len(adapted["signed"]),
        "categorical_dim": len(adapted["categorical_class"]),
    }


def validate_setup(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    data = config["data"]
    schema = load_schema(data["schema"])
    manifest = json.loads(Path(data["manifest"]).read_text(encoding="utf-8"))
    stats = json.loads(Path(data["train_label_stats"]).read_text(encoding="utf-8"))
    if not schema.accepts_statistics_schema(str(stats.get("schema_sha256", ""))):
        raise RuntimeError("training statistics are incompatible with schema")
    split_index_path = data.get("split_index")
    split_ids: dict[str, frozenset[str]] = {}
    if split_index_path:
        split_index_path = Path(split_index_path)
        split_manifest = load_split_manifest(split_index_path)
        if str(split_manifest.get("schema_sha256", "")) != schema.sha256:
            raise RuntimeError("split index uses a different schema")
        split_ids = {
            split: load_split_ids(split_index_path, split)
            for split in ("train", "val")
        }
        overlap = split_ids["train"] & split_ids["val"]
        if overlap:
            raise RuntimeError(
                f"split index has {len(overlap)} sample ids in both train and val"
            )
        train_count = int(split_manifest["counts"]["train"])
        if str(stats.get("split_index_sha256", "")) != file_sha256(
            split_index_path
        ):
            raise RuntimeError("training statistics use a different split index")
    else:
        train_count = int(manifest["split"]["counts"]["train"])
    if int(stats.get("sample_count", -1)) != train_count:
        raise RuntimeError("training statistics sample count does not match manifest")
    schema.adapt_class_counts(stats["categorical_class_counts"])
    profile_coverage: dict[str, int] = {}
    target_fields = set(schema.scalar_fields) | set(schema.signed_fields) | {
        field.name for field in schema.categorical_fields
    }
    for key in ("field_weights_path", "texture_metrics_path"):
        configured = config["loss"].get(key)
        if configured and not Path(configured).is_file():
            raise FileNotFoundError(f"missing loss profile {key}: {configured}")
        if not configured:
            continue
        if key == "field_weights_path":
            profile_fields = set(
                json.loads(Path(configured).read_text(encoding="utf-8"))
            )
        else:
            with Path(configured).open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                profile_fields = {str(row["field"]) for row in csv.DictReader(stream)}
        missing = target_fields - profile_fields
        if missing:
            raise RuntimeError(
                f"{key} is missing fields: {', '.join(sorted(missing))}"
            )
        profile_coverage[key] = len(profile_fields & target_fields)
    data_root = Path(data["root"])
    if split_index_path:
        all_shards = []
        for physical_split in ("train", "val", "test"):
            all_shards.extend(
                sorted(
                    (data_root / physical_split).glob(
                        f"{physical_split}-*.tar"
                    )
                )
            )
        split_shards = {"train": all_shards, "val": all_shards}
    else:
        split_shards = {
            split: sorted((data_root / split).glob(f"{split}-*.tar"))
            for split in ("train", "val")
        }
    splits = {
        split: _validate_split(
            split_shards[split], split, schema, split_ids.get(split)
        )
        for split in ("train", "val")
    }
    return {
        "config": str(config_path.resolve()),
        "schema": str(schema.path),
        "schema_version": schema.version,
        "target_family": schema.target_family,
        "sample_count": schema.sample_count,
        "scalar_dim": schema.scalar_dim,
        "signed_dim": schema.signed_dim,
        "categorical_dim": schema.categorical_dim,
        "legacy_statistics_adapted": stats.get("schema_sha256") != schema.sha256,
        "grouped_split": bool(split_index_path),
        "loss_profile_coverage": profile_coverage,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_setup(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
