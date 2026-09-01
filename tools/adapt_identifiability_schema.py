#!/usr/bin/env python3
"""Bind an identifiability target schema to a newly collected source schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


STAT_KEYS = (
    "observed_min",
    "observed_max",
    "missing_count",
    "pair_mismatch_count",
    "negative_count",
    "positive_count",
    "zero_allele",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"schema is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_name(fields: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in fields:
        name = field.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} contains a field without a name")
        if name in result:
            raise ValueError(f"{label} contains duplicate field {name}")
        result[name] = field
    return result


def _refresh_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in STAT_KEYS:
        if key in source:
            target[key] = source[key]


def adapt_schema(
    source_path: Path,
    template_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    template_path = template_path.resolve()
    output_path = output_path.resolve()
    source = _load(source_path)
    template = _load(template_path)
    if source.get("schema_version") != 1:
        raise ValueError("source schema must have schema_version=1")
    if template.get("schema_version") != 2:
        raise ValueError("identifiability template must have schema_version=2")

    source_signed = _by_name(list(source["signed_fields"]), "source signed_fields")
    source_categorical = _by_name(
        list(source["categorical_fields"]), "source categorical_fields"
    )
    result = deepcopy(template)

    for target in result.get("scalar_fields", []):
        name = str(target["name"])
        if name not in source_signed:
            raise ValueError(f"scalar field {name} is absent from source schema")
        current = source_signed[name]
        source_alleles = {
            str(current["negative_allele"]), str(current["positive_allele"])
        }
        if set(target.get("alleles", [])) != source_alleles:
            raise ValueError(f"scalar field {name} has incompatible alleles")
        _refresh_stats(target, current)

    for target in result["signed_fields"]:
        name = str(target["name"])
        if name not in source_signed:
            raise ValueError(f"signed field {name} is absent from source schema")
        current = source_signed[name]
        for key in ("negative_allele", "positive_allele"):
            if target.get(key) != current.get(key):
                raise ValueError(f"signed field {name} has incompatible {key}")
        _refresh_stats(target, current)

    for target in result["categorical_fields"]:
        name = str(target["name"])
        if name not in source_categorical:
            raise ValueError(f"categorical field {name} is absent from source schema")
        current = source_categorical[name]
        target_classes = list(target["classes"])
        source_classes = list(current["classes"])
        if set(target_classes) != set(source_classes):
            raise ValueError(f"categorical field {name} has incompatible classes")
        source_counts = dict(zip(source_classes, current.get("class_counts", [])))
        if len(source_counts) == len(source_classes):
            target["class_counts"] = [source_counts[value] for value in target_classes]
        _refresh_stats(target, current)

    result["sample_count"] = int(source["sample_count"])
    result["source_schema_path"] = Path(
        os.path.relpath(source_path, output_path.parent)
    ).as_posix()
    result["source_schema_sha256"] = _sha256(source_path)
    result["source_collection"] = {
        "schema_path": result["source_schema_path"],
        "schema_sha256": result["source_schema_sha256"],
        "sample_count": result["sample_count"],
        "identifiability_template_path": Path(
            os.path.relpath(template_path, output_path.parent)
        ).as_posix(),
        "identifiability_template_sha256": _sha256(template_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = adapt_schema(args.source, args.template, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    try:
        partial.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(
        f"adapted: {result['sample_count']} samples, "
        f"{len(result.get('scalar_fields', []))} scalar, "
        f"{len(result['signed_fields'])} signed, "
        f"{len(result['categorical_fields'])} categorical -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
