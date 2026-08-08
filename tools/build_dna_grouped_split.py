#!/usr/bin/env python3
"""Build leakage-safe split indexes by grouping identical normalized DNA targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ck3_training.schema import CK3Schema, load_schema
from ck3_training.split_index import SPLIT_NAMES


def parse_ratios(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ratios must contain numbers") from error
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("ratios must contain train,val,test values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError("ratios must sum to 1")
    return ratios  # type: ignore[return-value]


def normalized_target_fingerprint(
    source_label: dict[str, Any], schema: CK3Schema
) -> str:
    """Hash only model targets, excluding sample id, colors, and population metadata."""
    label = schema.adapt_label(source_label)
    schema.validate_label(label)
    payload = bytearray(b"ck3-normalized-target-v1\0")
    payload.extend(round(float(value) * 255) for value in label.get("scalar", ()))
    for value in label["signed"]:
        payload.extend(struct.pack("<h", round(float(value) * 255)))
    for value in label["categorical_class"]:
        payload.extend(struct.pack("<H", int(value)))
    payload.extend(
        round(float(value) * 255) for value in label["categorical_strength"]
    )
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def assigned_split(
    fingerprint: str, ratios: Sequence[float], seed: int
) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{fingerprint}".encode("ascii"), digest_size=8
    ).digest()
    position = int.from_bytes(digest, "big") / 2**64
    if position < float(ratios[0]):
        return "train"
    if position < float(ratios[0]) + float(ratios[1]):
        return "val"
    return "test"


def build_grouped_split(
    labels_path: Path,
    schema: CK3Schema,
    output_dir: Path,
    *,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 20260718,
    limit: int | None = None,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        split: output_dir / f"{split}.ids.txt" for split in SPLIT_NAMES
    }
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (*destinations.values(), manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite split index: {existing[:3]}")
    partials = {
        split: path.with_suffix(path.suffix + ".partial")
        for split, path in destinations.items()
    }
    manifest_partial = manifest_path.with_suffix(".json.partial")
    counts = {split: 0 for split in SPLIT_NAMES}
    group_counts: dict[str, int] = {}
    source_digest = hashlib.sha256()
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in partials.items()
    }
    sample_count = 0
    try:
        with labels_path.open("rb") as stream:
            for index, raw in enumerate(stream, start=1):
                if limit is not None and index > limit:
                    break
                source_digest.update(raw)
                source = json.loads(raw)
                sample_id = str(source.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"label line {index} has no sample_id")
                fingerprint = normalized_target_fingerprint(source, schema)
                split = assigned_split(fingerprint, ratios, seed)
                handles[split].write(sample_id + "\n")
                counts[split] += 1
                group_counts[fingerprint] = group_counts.get(fingerprint, 0) + 1
                sample_count += 1
                if progress_every > 0 and index % progress_every == 0:
                    print(f"processed {index} labels", flush=True)
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in (*partials.values(), manifest_partial):
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()

    manifest = {
        "version": 1,
        "method": "normalized_target_fingerprint",
        "fingerprint_version": 1,
        "schema_path": os.path.relpath(schema.path, output_dir).replace("\\", "/"),
        "schema_sha256": schema.sha256,
        "source_labels": str(labels_path.resolve()),
        "source_labels_scanned_sha256": source_digest.hexdigest(),
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "counts": counts,
        "sample_count": sample_count,
        "target_group_count": len(group_counts),
        "duplicated_sample_count": sample_count - len(group_counts),
        "maximum_group_size": max(group_counts.values(), default=0),
        "cross_split_duplicate_groups": 0,
        "files": {split: path.name for split, path in destinations.items()},
    }
    manifest_partial.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for split in SPLIT_NAMES:
        os.replace(partials[split], destinations[split])
    os.replace(manifest_partial, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("face_to_ck3_dataset_male_small/labels.jsonl"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "experiments/dna_identifiability/recommended_training_schema.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", type=parse_ratios, default=(0.9, 0.05, 0.05))
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    schema = load_schema(args.schema)
    manifest = build_grouped_split(
        args.labels,
        schema,
        args.output,
        ratios=args.ratios,
        seed=args.seed,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

