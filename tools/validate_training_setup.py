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


def _first_json_label(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".json"):
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot extract {member.name} from {path}")
                return json.loads(stream.read().decode("utf-8"))
    raise RuntimeError(f"no JSON label in {path}")


def _validate_split(data_root: Path, split: str, schema: CK3Schema) -> dict[str, Any]:
    shards = sorted((data_root / split).glob(f"{split}-*.tar"))
    if not shards:
        raise FileNotFoundError(f"no {split} shards under {data_root}")
    source = _first_json_label(shards[0])
    adapted = schema.adapt_label(source)
    schema.validate_label(adapted)
    return {
        "shard_count": len(shards),
        "first_shard": str(shards[0]),
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
    splits = {
        split: _validate_split(data_root, split, schema)
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
