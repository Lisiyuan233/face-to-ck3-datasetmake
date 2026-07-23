#!/usr/bin/env python3
"""CK3 DNA training-label normalizer.

This tool builds an allele schema from CK3 ruler DNA files, converts DNA files
to compact normalized JSONL records, and applies a normalized record back to a
DNA template.

The implementation uses only the Python standard library so dataset inspection
does not require a PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = 1
BYTE_MAX = 255

# These are the facial geometry/detail fields present in the collected male
# dataset. Non-facial fields (body, clothes, expressions, hair, etc.) are
# deliberately excluded from the default training target.
FACE_FIELDS: tuple[str, ...] = (
    "gene_chin_forward",
    "gene_chin_height",
    "gene_chin_width",
    "gene_eye_angle",
    "gene_eye_depth",
    "gene_eye_height",
    "gene_eye_distance",
    "gene_eye_shut",
    "gene_forehead_angle",
    "gene_forehead_brow_height",
    "gene_forehead_roundness",
    "gene_forehead_width",
    "gene_forehead_height",
    "gene_head_height",
    "gene_head_width",
    "gene_head_profile",
    "gene_head_top_height",
    "gene_head_top_width",
    "gene_jaw_angle",
    "gene_jaw_forward",
    "gene_jaw_height",
    "gene_jaw_width",
    "gene_mouth_corner_depth",
    "gene_mouth_corner_height",
    "gene_mouth_forward",
    "gene_mouth_height",
    "gene_mouth_width",
    "gene_mouth_upper_lip_size",
    "gene_mouth_lower_lip_size",
    "gene_mouth_open",
    "gene_neck_length",
    "gene_neck_width",
    "gene_bs_cheek_forward",
    "gene_bs_cheek_height",
    "gene_bs_cheek_width",
    "gene_bs_ear_angle",
    "gene_bs_ear_inner_shape",
    "gene_bs_ear_bend",
    "gene_bs_ear_outward",
    "gene_bs_ear_size",
    "gene_bs_eye_corner_depth",
    "gene_bs_eye_fold_shape",
    "gene_bs_eye_size",
    "gene_bs_eye_upper_lid_size",
    "gene_bs_forehead_brow_curve",
    "gene_bs_forehead_brow_forward",
    "gene_bs_forehead_brow_inner_height",
    "gene_bs_forehead_brow_outer_height",
    "gene_bs_forehead_brow_width",
    "gene_bs_jaw_def",
    "gene_bs_mouth_lower_lip_def",
    "gene_bs_mouth_lower_lip_full",
    "gene_bs_mouth_lower_lip_pad",
    "gene_bs_mouth_lower_lip_width",
    "gene_bs_mouth_philtrum_def",
    "gene_bs_mouth_philtrum_shape",
    "gene_bs_mouth_philtrum_width",
    "gene_bs_mouth_upper_lip_def",
    "gene_bs_mouth_upper_lip_full",
    "gene_bs_mouth_upper_lip_profile",
    "gene_bs_mouth_upper_lip_width",
    "gene_bs_nose_forward",
    "gene_bs_nose_height",
    "gene_bs_nose_length",
    "gene_bs_nose_nostril_height",
    "gene_bs_nose_nostril_width",
    "gene_bs_nose_profile",
    "gene_bs_nose_ridge_angle",
    "gene_bs_nose_ridge_width",
    "gene_bs_nose_size",
    "gene_bs_nose_tip_angle",
    "gene_bs_nose_tip_forward",
    "gene_bs_nose_tip_width",
    "face_detail_cheek_def",
    "face_detail_cheek_fat",
    "face_detail_chin_cleft",
    "face_detail_chin_def",
    "face_detail_eye_lower_lid_def",
    "face_detail_eye_socket",
    "face_detail_nasolabial",
    "face_detail_nose_ridge_def",
    "face_detail_nose_tip_def",
    "face_detail_temple_def",
)

COLOR_FIELDS: tuple[str, ...] = ("hair_color", "skin_color", "eye_color")

QUOTED_FIELD_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_]+)='
    r'\{\s*"(?P<allele1>[^"]+)"\s+(?P<value1>\d+)\s+'
    r'"(?P<allele2>[^"]+)"\s+(?P<value2>\d+)\s*\}'
)


@dataclass(frozen=True)
class GeneValue:
    allele1: str
    value1: int
    allele2: str
    value2: int

    @property
    def pair_matches(self) -> bool:
        return self.allele1 == self.allele2 and self.value1 == self.value2


@dataclass(frozen=True)
class DNARecord:
    genes: dict[str, GeneValue]
    colors: dict[str, tuple[int, int, int, int]]


def _color_re(key: str) -> re.Pattern[str]:
    return re.compile(
        rf'{re.escape(key)}=\{{\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\}}'
    )


def parse_dna(text: str) -> DNARecord:
    """Parse quoted gene fields and CK3 two-coordinate color fields."""
    genes: dict[str, GeneValue] = {}
    for match in QUOTED_FIELD_RE.finditer(text):
        key = match.group("key")
        if key in genes:
            raise ValueError(f"duplicate DNA field: {key}")
        values = GeneValue(
            allele1=match.group("allele1"),
            value1=int(match.group("value1")),
            allele2=match.group("allele2"),
            value2=int(match.group("value2")),
        )
        if not all(0 <= value <= BYTE_MAX for value in (values.value1, values.value2)):
            raise ValueError(f"{key}: value outside 0..255")
        genes[key] = values

    colors: dict[str, tuple[int, int, int, int]] = {}
    for key in COLOR_FIELDS:
        match = _color_re(key).search(text)
        if match:
            values = tuple(int(value) for value in match.groups())
            if not all(0 <= value <= BYTE_MAX for value in values):
                raise ValueError(f"{key}: value outside 0..255")
            colors[key] = values  # type: ignore[assignment]

    return DNARecord(genes=genes, colors=colors)


def iter_dna_paths(inputs: Sequence[Path], stride: int = 1) -> Iterator[Path]:
    """Yield DNA files without materializing a 510k-entry path list."""
    seen = 0
    for input_path in inputs:
        if input_path.is_file():
            candidates: Iterable[Path] = (input_path,)
        elif input_path.is_dir():
            candidates = (
                Path(entry.path)
                for entry in os.scandir(input_path)
                if entry.is_file() and entry.name.lower().endswith(".txt")
            )
        else:
            raise FileNotFoundError(input_path)

        for path in candidates:
            if seen % stride == 0:
                yield path
            seen += 1


def _signed_pair(alleles: set[str]) -> tuple[str, str] | None:
    """Return (negative, positive) only for an exact symmetric allele pair."""
    if len(alleles) != 2:
        return None
    negative = [allele for allele in alleles if allele.endswith("_neg")]
    positive = [allele for allele in alleles if allele.endswith("_pos")]
    if len(negative) != 1 or len(positive) != 1:
        return None
    if negative[0][:-4] != positive[0][:-4]:
        return None
    return negative[0], positive[0]


def build_schema(
    paths: Iterable[Path],
    limit: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    allele_counts = {field: Counter() for field in FACE_FIELDS}
    value_min = {field: BYTE_MAX for field in FACE_FIELDS}
    value_max = {field: 0 for field in FACE_FIELDS}
    missing_counts = Counter()
    mismatch_counts = Counter()
    color_missing_counts = Counter()
    color_mismatch_counts = Counter()
    sample_count = 0

    for path in paths:
        if limit is not None and sample_count >= limit:
            break
        text = path.read_text(encoding="utf-8-sig")
        record = parse_dna(text)
        sample_count += 1
        if progress_every and sample_count % progress_every == 0:
            print(f"schema: scanned {sample_count} files...", file=sys.stderr)

        for field in FACE_FIELDS:
            value = record.genes.get(field)
            if value is None:
                missing_counts[field] += 1
                continue
            allele_counts[field][value.allele1] += 1
            value_min[field] = min(value_min[field], value.value1)
            value_max[field] = max(value_max[field], value.value1)
            if not value.pair_matches:
                mismatch_counts[field] += 1

        for field in COLOR_FIELDS:
            color = record.colors.get(field)
            if color is None:
                color_missing_counts[field] += 1
            elif color[:2] != color[2:]:
                color_mismatch_counts[field] += 1

    if sample_count == 0:
        raise ValueError("no DNA files found")

    signed_fields: list[dict[str, Any]] = []
    categorical_fields: list[dict[str, Any]] = []
    for field in FACE_FIELDS:
        counts = allele_counts[field]
        if not counts:
            raise ValueError(f"field {field!r} was absent from every scanned DNA file")
        common = counts.most_common()
        signed = _signed_pair(set(counts))
        stats = {
            "name": field,
            "observed_min": value_min[field],
            "observed_max": value_max[field],
            "missing_count": missing_counts[field],
            "pair_mismatch_count": mismatch_counts[field],
        }
        if signed:
            negative, positive = signed
            signed_fields.append(
                {
                    **stats,
                    "negative_allele": negative,
                    "positive_allele": positive,
                    "negative_count": counts[negative],
                    "positive_count": counts[positive],
                    "zero_allele": common[0][0],
                }
            )
        else:
            classes = sorted(counts)
            categorical_fields.append(
                {
                    **stats,
                    "classes": classes,
                    "class_counts": [counts[name] for name in classes],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": sample_count,
        "normalization": {
            "byte_divisor": BYTE_MAX,
            "signed_range": [-1.0, 1.0],
            "strength_range": [0.0, 1.0],
            "color_range": [0.0, 1.0],
        },
        "signed_fields": signed_fields,
        "categorical_fields": categorical_fields,
        "color_fields": [
            {
                "name": field,
                "missing_count": color_missing_counts[field],
                "pair_mismatch_count": color_mismatch_counts[field],
            }
            for field in COLOR_FIELDS
        ],
    }


def validate_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={schema.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    divisor = schema.get("normalization", {}).get("byte_divisor")
    if divisor != BYTE_MAX:
        raise ValueError(f"unsupported byte_divisor={divisor!r}")


def normalize_record(
    record: DNARecord,
    schema: dict[str, Any],
    sample_id: str,
    require_matching_pairs: bool = True,
) -> dict[str, Any]:
    validate_schema(schema)
    signed_values: list[float] = []
    categorical_classes: list[int] = []
    categorical_strengths: list[float] = []
    colors: list[float] = []

    for spec in schema["signed_fields"]:
        field = spec["name"]
        value = _required_gene(record, field, require_matching_pairs)
        strength = value.value1 / BYTE_MAX
        if value.allele1 == spec["negative_allele"]:
            signed = -strength
        elif value.allele1 == spec["positive_allele"]:
            signed = strength
        else:
            raise ValueError(f"{field}: allele {value.allele1!r} is absent from schema")
        signed_values.append(signed)

    for spec in schema["categorical_fields"]:
        field = spec["name"]
        value = _required_gene(record, field, require_matching_pairs)
        try:
            class_id = spec["classes"].index(value.allele1)
        except ValueError as error:
            raise ValueError(
                f"{field}: allele {value.allele1!r} is absent from schema"
            ) from error
        categorical_classes.append(class_id)
        categorical_strengths.append(value.value1 / BYTE_MAX)

    for spec in schema["color_fields"]:
        field = spec["name"]
        if field not in record.colors:
            raise ValueError(f"missing color field: {field}")
        value = record.colors[field]
        if require_matching_pairs and value[:2] != value[2:]:
            raise ValueError(f"{field}: color chromosome pairs differ: {value}")
        colors.extend((value[0] / BYTE_MAX, value[1] / BYTE_MAX))

    return {
        "sample_id": sample_id,
        "signed": signed_values,
        "categorical_class": categorical_classes,
        "categorical_strength": categorical_strengths,
        "colors": colors,
    }


def _required_gene(
    record: DNARecord, field: str, require_matching_pairs: bool
) -> GeneValue:
    if field not in record.genes:
        raise ValueError(f"missing DNA field: {field}")
    value = record.genes[field]
    if require_matching_pairs and not value.pair_matches:
        raise ValueError(
            f"{field}: chromosome pairs differ: "
            f"{value.allele1} {value.value1} / {value.allele2} {value.value2}"
        )
    return value


def _unit_to_byte(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError(f"prediction is not finite: {value}")
    return round(min(1.0, max(0.0, value)) * BYTE_MAX)


def apply_normalized_to_template(
    template_text: str, label: dict[str, Any], schema: dict[str, Any]
) -> str:
    """Apply one normalized prediction to a DNA template.

    Both chromosome slots are set to the prediction. At exactly zero, a signed
    field uses the schema's most frequent allele because the allele has no visual
    effect at zero strength.
    """
    validate_schema(schema)
    signed = label["signed"]
    classes = label["categorical_class"]
    strengths = label["categorical_strength"]
    colors = label.get("colors")

    _expect_length("signed", signed, len(schema["signed_fields"]))
    _expect_length("categorical_class", classes, len(schema["categorical_fields"]))
    _expect_length(
        "categorical_strength", strengths, len(schema["categorical_fields"])
    )
    if colors is not None:
        _expect_length("colors", colors, 2 * len(schema["color_fields"]))

    replacements: dict[str, tuple[str, int]] = {}
    for value, spec in zip(signed, schema["signed_fields"]):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{spec['name']}: prediction is not finite")
        if numeric < 0:
            allele = spec["negative_allele"]
        elif numeric > 0:
            allele = spec["positive_allele"]
        else:
            allele = spec["zero_allele"]
        replacements[spec["name"]] = (allele, _unit_to_byte(abs(numeric)))

    for class_id, strength, spec in zip(
        classes, strengths, schema["categorical_fields"]
    ):
        class_id = int(class_id)
        if not 0 <= class_id < len(spec["classes"]):
            raise ValueError(f"{spec['name']}: invalid class id {class_id}")
        replacements[spec["name"]] = (
            spec["classes"][class_id],
            _unit_to_byte(float(strength)),
        )

    replaced_fields: set[str] = set()

    def replace_gene(match: re.Match[str]) -> str:
        key = match.group("key")
        if key not in replacements:
            return match.group(0)
        allele, value = replacements[key]
        replaced_fields.add(key)
        return f'{match.group("indent")}{key}={{ "{allele}" {value} "{allele}" {value} }}'

    output = QUOTED_FIELD_RE.sub(replace_gene, template_text)
    missing = set(replacements) - replaced_fields
    if missing:
        raise ValueError(f"template is missing fields: {', '.join(sorted(missing))}")

    if colors is not None:
        for index, spec in enumerate(schema["color_fields"]):
            key = spec["name"]
            first = _unit_to_byte(float(colors[index * 2]))
            second = _unit_to_byte(float(colors[index * 2 + 1]))
            pattern = _color_re(key)
            output, count = pattern.subn(
                f"{key}={{ {first} {second} {first} {second} }}", output, count=1
            )
            if count != 1:
                raise ValueError(f"template is missing color field: {key}")

    return output


def _expect_length(name: str, values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name}: expected {expected} values, got {len(values)}")


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    validate_schema(schema)
    return schema


def load_label(path: Path, sample_id: str | None) -> dict[str, Any]:
    """Load a JSON object or select one object from a JSONL file."""
    content = path.read_text(encoding="utf-8")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        if sample_id is None or value.get("sample_id") == sample_id:
            return value

    for line_number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            label = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if sample_id is None or label.get("sample_id") == sample_id:
            return label
    if sample_id is None:
        raise ValueError(f"no JSON object found in {path}")
    raise ValueError(f"sample_id {sample_id!r} not found in {path}")


def command_schema(args: argparse.Namespace) -> int:
    paths = iter_dna_paths(args.inputs, stride=args.stride)
    schema = build_schema(
        paths, limit=args.limit, progress_every=args.progress_every
    )
    args.output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"schema: {schema['sample_count']} samples, "
        f"{len(schema['signed_fields'])} signed fields, "
        f"{len(schema['categorical_fields'])} categorical fields -> {args.output}"
    )
    return 0


def command_normalize(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema)
    processed = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for path in iter_dna_paths(args.inputs, stride=args.stride):
            if args.limit is not None and processed + skipped >= args.limit:
                break
            try:
                record = parse_dna(path.read_text(encoding="utf-8-sig"))
                label = normalize_record(
                    record,
                    schema,
                    sample_id=path.stem,
                    require_matching_pairs=not args.allow_pair_mismatch,
                )
                output.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                processed += 1
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"normalize: wrote {processed} labels...", file=sys.stderr)
            except Exception as error:
                if not args.skip_invalid:
                    raise
                skipped += 1
                print(f"skip {path}: {error}", file=sys.stderr)
    print(f"normalized: {processed} samples, skipped: {skipped} -> {args.output}")
    return 0


def command_denormalize(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema)
    label = load_label(args.label, args.sample_id)
    template = args.template.read_text(encoding="utf-8-sig")
    output = apply_normalized_to_template(template, label, schema)
    args.output.write_text(output, encoding="utf-8")
    print(f"DNA: sample_id={label.get('sample_id', '?')} -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and apply normalized training labels for CK3 DNA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser(
        "schema", help="scan DNA files and build allele vocabularies"
    )
    schema_parser.add_argument("inputs", nargs="+", type=Path)
    schema_parser.add_argument("-o", "--output", required=True, type=Path)
    schema_parser.add_argument("--limit", type=int)
    schema_parser.add_argument("--stride", type=int, default=1)
    schema_parser.add_argument("--progress-every", type=int, default=10000)
    schema_parser.set_defaults(handler=command_schema)

    normalize_parser = subparsers.add_parser(
        "normalize", help="convert DNA files to compact normalized JSONL"
    )
    normalize_parser.add_argument("inputs", nargs="+", type=Path)
    normalize_parser.add_argument("--schema", required=True, type=Path)
    normalize_parser.add_argument("-o", "--output", required=True, type=Path)
    normalize_parser.add_argument("--limit", type=int)
    normalize_parser.add_argument("--stride", type=int, default=1)
    normalize_parser.add_argument("--progress-every", type=int, default=10000)
    normalize_parser.add_argument("--skip-invalid", action="store_true")
    normalize_parser.add_argument("--allow-pair-mismatch", action="store_true")
    normalize_parser.set_defaults(handler=command_normalize)

    denormalize_parser = subparsers.add_parser(
        "denormalize", help="apply one normalized JSON/JSONL record to a DNA template"
    )
    denormalize_parser.add_argument("--schema", required=True, type=Path)
    denormalize_parser.add_argument("--label", required=True, type=Path)
    denormalize_parser.add_argument("--sample-id")
    denormalize_parser.add_argument("--template", required=True, type=Path)
    denormalize_parser.add_argument("-o", "--output", required=True, type=Path)
    denormalize_parser.set_defaults(handler=command_denormalize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "stride", 1) < 1:
        parser.error("--stride must be >= 1")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if getattr(args, "progress_every", 1) < 0:
        parser.error("--progress-every must be >= 0")
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
