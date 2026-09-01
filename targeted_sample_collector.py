#!/usr/bin/env python3
"""Prepare, execute, and inspect leakage-safe CK3 targeted sample collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from build_identifiability_variants import interleave_baselines
from dna_field_sweep_tool import (
    WindowsAutomationBackend,
    atomic_write_text,
    automation_config_from_settings,
    load_user_settings,
    parse_value_sequence,
    replace_gene_pair,
    safe_component,
    sha256_text,
    utc_now,
)
from dna_normalizer import parse_dna
from run_identifiability_experiment import (
    PlannedVariant,
    completed_variant_ids,
    load_plan,
    run_experiment,
)


PROTOCOL_VERSION = 1
PROTOCOL_KIND = "targeted_sample_collection"
DEFAULT_STRENGTHS = (0, 64, 128, 192, 255)
DEFAULT_TARGET_FIELDS = (
    "gene_bs_mouth_philtrum_width",
    "gene_bs_mouth_lower_lip_pad",
    "gene_bs_mouth_upper_lip_def",
    "gene_bs_mouth_philtrum_def",
    "gene_mouth_corner_depth",
    "gene_chin_width",
    "gene_mouth_open",
    "face_detail_nasolabial",
    "gene_bs_mouth_philtrum_shape",
    "gene_bs_mouth_upper_lip_profile",
    "gene_bs_mouth_upper_lip_full",
    "gene_bs_mouth_lower_lip_full",
    "gene_eye_shut",
    "gene_eye_distance",
)
DIAGNOSTIC_FIELD = "face_detail_chin_cleft"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class TargetField:
    name: str
    source_type: str
    target_family: str
    alleles: tuple[tuple[str, str, int | None], ...]


@dataclass(frozen=True)
class LockedBase:
    base_index: int
    base_id: str
    source_sample_id: str
    source_dna_path: Path
    copied_dna_path: str
    dna_text: str
    dna_sha256: str
    split: str | None
    race_group: int

    def manifest_row(self) -> dict[str, Any]:
        return {
            "base_index": self.base_index,
            "base_dna_id": self.base_id,
            "source_sample_id": self.source_sample_id,
            "source_dna_path": str(self.source_dna_path),
            "dna_path": self.copied_dna_path,
            "dna_sha256": self.dna_sha256,
            "split": self.split,
            "race_group": self.race_group,
        }


@dataclass(frozen=True)
class BaseSelectionCandidate:
    sample_id: str
    dna_path: Path
    continuous: bytes
    categorical: tuple[int, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            rows.append(row)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("split ratios 必须是数字") from error
    if (
        len(ratios) != 3
        or any(not math.isfinite(item) or item < 0 for item in ratios)
        or abs(sum(ratios) - 1.0) > 1e-9
    ):
        raise argparse.ArgumentTypeError(
            "split ratios 必须是和为 1 的 train,val,test 三个非负数"
        )
    return ratios  # type: ignore[return-value]


def _deduplicate(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _quantize_unit(value: Any, *, signed: bool) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("标签包含非有限数值")
    if signed:
        if number < -1.000001 or number > 1.000001:
            raise ValueError(f"signed 标签超出 [-1,1]: {number}")
        return round((max(-1.0, min(1.0, number)) + 1.0) * 127.5)
    if number < -0.000001 or number > 1.000001:
        raise ValueError(f"unit 标签超出 [0,1]: {number}")
    return round(max(0.0, min(1.0, number)) * 255)


def _candidate_from_label(
    label: dict[str, Any], dna_dir: Path, line_number: int
) -> BaseSelectionCandidate:
    sample_id = str(label.get("sample_id", ""))
    if not sample_id:
        raise ValueError(f"labels 第 {line_number} 行缺少 sample_id")
    dna_path = (dna_dir / f"{sample_id}.txt").resolve()
    if not dna_path.is_file():
        raise FileNotFoundError(dna_path)
    continuous = bytearray()
    for value in label.get("scalar", []):
        continuous.append(_quantize_unit(value, signed=False))
    for value in label.get("signed", []):
        continuous.append(_quantize_unit(value, signed=True))
    for value in label.get("categorical_strength", []):
        continuous.append(_quantize_unit(value, signed=False))
    for value in label.get("colors", []):
        continuous.append(_quantize_unit(value, signed=False))
    categorical = tuple(int(value) for value in label.get("categorical_class", []))
    if not continuous:
        raise ValueError(f"labels 第 {line_number} 行没有连续目标")
    if any(value < 0 for value in categorical):
        raise ValueError(f"labels 第 {line_number} 行 categorical class 小于 0")
    return BaseSelectionCandidate(
        sample_id=sample_id,
        dna_path=dna_path,
        continuous=bytes(continuous),
        categorical=categorical,
    )


def _histogram_median(histogram: Sequence[int]) -> int:
    target = (sum(histogram) - 1) // 2
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative > target:
            return value
    raise ValueError("空直方图没有中位数")


def _candidate_distance(
    left: BaseSelectionCandidate,
    right_continuous: Sequence[int],
    right_categorical: Sequence[int],
) -> float:
    if len(left.continuous) != len(right_continuous):
        raise ValueError("连续标签维度不一致")
    if len(left.categorical) != len(right_categorical):
        raise ValueError("categorical 标签维度不一致")
    continuous_distance = sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left.continuous, right_continuous)
    ) / (255.0 * len(left.continuous))
    if not left.categorical:
        return continuous_distance
    categorical_distance = sum(
        left_value != right_value
        for left_value, right_value in zip(
            left.categorical, right_categorical
        )
    ) / len(left.categorical)
    return 0.5 * continuous_distance + 0.5 * categorical_distance


def select_diverse_bases(
    labels_path: Path,
    dna_dir: Path,
    output_path: Path,
    *,
    count: int = 32,
    included_sample_ids: set[str] | None = None,
    excluded_sample_ids: set[str] | None = None,
    limit: int | None = None,
    progress_every: int = 2000,
) -> list[dict[str, Any]]:
    """Select a median anchor followed by deterministic farthest-point bases."""

    if count < 1:
        raise ValueError("count 必须大于 0")
    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于 0")
    labels_path = labels_path.resolve()
    dna_dir = dna_dir.resolve()
    output_path = output_path.resolve()
    excluded_sample_ids = excluded_sample_ids or set()
    candidates: list[BaseSelectionCandidate] = []
    histograms: list[list[int]] | None = None
    categorical_counts: list[dict[int, int]] | None = None
    seen_sample_ids: set[str] = set()
    with labels_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if limit is not None and line_number > limit:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"labels 第 {line_number} 行必须是 JSON 对象")
            sample_id = str(value.get("sample_id", ""))
            if included_sample_ids is not None and sample_id not in included_sample_ids:
                continue
            if sample_id in excluded_sample_ids:
                continue
            candidate = _candidate_from_label(value, dna_dir, line_number)
            if candidate.sample_id in seen_sample_ids:
                raise ValueError(f"labels 中 sample_id 重复: {candidate.sample_id}")
            seen_sample_ids.add(candidate.sample_id)
            if histograms is None:
                histograms = [[0] * 256 for _ in candidate.continuous]
                categorical_counts = [{} for _ in candidate.categorical]
            if len(candidate.continuous) != len(histograms) or len(
                candidate.categorical
            ) != len(categorical_counts or []):
                raise ValueError(f"labels 第 {line_number} 行目标维度不一致")
            for index, item in enumerate(candidate.continuous):
                histograms[index][item] += 1
            for index, item in enumerate(candidate.categorical):
                counts = categorical_counts[index]  # type: ignore[index]
                counts[item] = counts.get(item, 0) + 1
            candidates.append(candidate)
            if progress_every > 0 and line_number % progress_every == 0:
                print(f"已扫描 {line_number} 条标签", flush=True)
    if len(candidates) < count:
        raise ValueError(
            f"可用且有 DNA 的候选只有 {len(candidates)} 个，少于 count={count}"
        )
    assert histograms is not None
    assert categorical_counts is not None
    median = [_histogram_median(histogram) for histogram in histograms]
    modes = [
        min(counts, key=lambda value: (-counts[value], value))
        for counts in categorical_counts
    ]
    center_distances = [
        _candidate_distance(candidate, median, modes) for candidate in candidates
    ]
    first_index = min(
        range(len(candidates)),
        key=lambda index: (center_distances[index], candidates[index].sample_id),
    )
    selected_indices = [first_index]
    selected_set = {first_index}
    selection_distances = [center_distances[first_index]]
    minimum_distances = [float("inf")] * len(candidates)
    while len(selected_indices) < count:
        latest = candidates[selected_indices[-1]]
        for index, candidate in enumerate(candidates):
            if index in selected_set:
                continue
            distance = _candidate_distance(
                candidate, latest.continuous, latest.categorical
            )
            minimum_distances[index] = min(minimum_distances[index], distance)
        next_index = min(
            (index for index in range(len(candidates)) if index not in selected_set),
            key=lambda index: (-minimum_distances[index], candidates[index].sample_id),
        )
        selected_indices.append(next_index)
        selected_set.add(next_index)
        selection_distances.append(minimum_distances[next_index])

    rows: list[dict[str, Any]] = []
    for rank, (index, distance) in enumerate(
        zip(selected_indices, selection_distances), 1
    ):
        candidate = candidates[index]
        try:
            dna_path = candidate.dna_path.relative_to(output_path.parent).as_posix()
        except ValueError:
            dna_path = str(candidate.dna_path)
        rows.append(
            {
                "base_dna_id": f"base_{rank:03d}_{safe_component(candidate.sample_id)}",
                "sample_id": candidate.sample_id,
                "dna_path": dna_path,
                "selection_rank": rank,
                "selection_method": (
                    "closest_to_component_median"
                    if rank == 1
                    else "farthest_point_maximin"
                ),
                "selection_distance": distance,
                "candidate_count": len(candidates),
            }
        )
    _atomic_jsonl(output_path, rows)
    return rows


def load_sample_id_files(paths: Sequence[Path] | None) -> set[str]:
    result: set[str] = set()
    for path in paths or []:
        with path.resolve().open("r", encoding="utf-8") as stream:
            for line in stream:
                value = line.strip()
                if value:
                    result.add(value)
    return result


def _schema_fields(
    schema: dict[str, Any], family: str
) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for raw in schema.get(family, []):
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError(f"schema {family} 包含无效字段")
        name = str(raw["name"])
        if name in fields:
            raise ValueError(f"schema {family} 字段重复: {name}")
        fields[name] = raw
    return fields


def resolve_target_fields(
    source_schema: dict[str, Any],
    training_schema: dict[str, Any],
    names: Sequence[str],
) -> list[TargetField]:
    if source_schema.get("schema_version") != 1:
        raise ValueError("source schema 必须是 schema_version=1")
    if training_schema.get("schema_version") != 2:
        raise ValueError("training schema 必须是 schema_version=2")

    source_signed = _schema_fields(source_schema, "signed_fields")
    source_categorical = _schema_fields(source_schema, "categorical_fields")
    training_scalar = _schema_fields(training_schema, "scalar_fields")
    training_signed = _schema_fields(training_schema, "signed_fields")
    training_categorical = _schema_fields(training_schema, "categorical_fields")
    result: list[TargetField] = []
    for name in _deduplicate([str(value) for value in names]):
        if name in training_scalar or name in training_signed:
            target_family = "scalar" if name in training_scalar else "signed"
            if name not in source_signed:
                raise ValueError(f"目标字段 {name} 不在 source signed_fields")
            source = source_signed[name]
            alleles = (
                ("negative", str(source["negative_allele"]), None),
                ("positive", str(source["positive_allele"]), None),
            )
            result.append(
                TargetField(
                    name=name,
                    source_type="signed",
                    target_family=target_family,
                    alleles=alleles,
                )
            )
            continue
        if name in training_categorical:
            if name not in source_categorical:
                raise ValueError(f"目标字段 {name} 不在 source categorical_fields")
            source_classes = {str(value) for value in source_categorical[name]["classes"]}
            training_classes = [
                str(value) for value in training_categorical[name]["classes"]
            ]
            if source_classes != set(training_classes):
                raise ValueError(f"目标字段 {name} 的 source/training classes 不一致")
            result.append(
                TargetField(
                    name=name,
                    source_type="categorical",
                    target_family="strength",
                    alleles=tuple(
                        (allele, allele, class_id)
                        for class_id, allele in enumerate(training_classes)
                    ),
                )
            )
            continue
        raise ValueError(f"目标字段不在 training schema 中: {name}")
    if not result:
        raise ValueError("至少需要一个目标字段")
    return result


def assign_base_splits(
    base_ids: Sequence[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign exact split counts after a deterministic hash shuffle."""

    if len(set(base_ids)) != len(base_ids):
        raise ValueError("base_dna_id 重复")
    total = len(base_ids)
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1

    shuffled = sorted(
        base_ids,
        key=lambda base_id: hashlib.blake2b(
            f"{seed}:{base_id}".encode("utf-8"), digest_size=16
        ).digest(),
    )
    result: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts):
        for base_id in shuffled[offset : offset + count]:
            result[base_id] = split
        offset += count
    if len(result) != total:
        raise RuntimeError("内部错误：base split 分配数量不一致")
    return result


def load_and_lock_bases(
    manifest_path: Path,
    verification_fields: Sequence[str],
    *,
    split_ratios: tuple[float, float, float],
    split_seed: int,
) -> list[LockedBase]:
    manifest_path = manifest_path.resolve()
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise ValueError("bases manifest 为空")
    required_fields = set(verification_fields)
    bases: list[LockedBase] = []
    seen_ids: set[str] = set()
    seen_samples: set[str] = set()
    explicit_split_count = 0
    for index, row in enumerate(rows, 1):
        raw_path = row.get("dna_path")
        if not raw_path:
            raise ValueError(f"bases manifest 第 {index} 行缺少 dna_path")
        source_path = Path(str(raw_path))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sample_id = str(row.get("sample_id") or source_path.stem)
        base_id = str(
            row.get("base_dna_id")
            or row.get("base_id")
            or f"base_{index:03d}_{safe_component(source_sample_id)}"
        )
        if base_id in seen_ids:
            raise ValueError(f"base_dna_id 重复: {base_id}")
        if source_sample_id in seen_samples:
            raise ValueError(f"source sample_id 重复: {source_sample_id}")
        seen_ids.add(base_id)
        seen_samples.add(source_sample_id)
        split = row.get("split")
        if split is not None:
            split = str(split)
            if split not in SPLIT_NAMES:
                raise ValueError(f"base {base_id} 的 split 无效: {split}")
            explicit_split_count += 1
        dna_text = source_path.read_text(encoding="utf-8-sig")
        record = parse_dna(dna_text)
        missing = required_fields - set(record.genes)
        if missing:
            raise ValueError(
                f"基础 DNA {base_id} 缺少 schema 字段: "
                + ", ".join(sorted(missing))
            )
        copied_path = f"bases/{safe_component(base_id)}.txt"
        bases.append(
            LockedBase(
                base_index=index,
                base_id=base_id,
                source_sample_id=source_sample_id,
                source_dna_path=source_path,
                copied_dna_path=copied_path,
                dna_text=dna_text,
                dna_sha256=sha256_text(dna_text),
                split=split,
                race_group=int(row.get("race_group", -1)),
            )
        )
    if explicit_split_count not in {0, len(bases)}:
        raise ValueError("bases manifest 的 split 必须全部填写或全部省略")
    if explicit_split_count == 0:
        assigned = assign_base_splits(
            [base.base_id for base in bases], split_ratios, split_seed
        )
        bases = [replace(base, split=assigned[base.base_id]) for base in bases]

    return bases


def _field_blueprints(
    base_dna: str,
    targets: Sequence[TargetField],
    strengths: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intervention_index = 0
    for target in targets:
        for class_or_sign, allele, class_id in target.alleles:
            for strength in strengths:
                intervention_index += 1
                rows.append(
                    {
                        "kind": "field",
                        "intervention_index": intervention_index,
                        "source_type": "targeted_intervention",
                        "field": target.name,
                        "intervention_field": target.name,
                        "field_type": target.source_type,
                        "target_family": target.target_family,
                        "class_id": class_id,
                        "class_or_sign": class_or_sign,
                        "allele": allele,
                        "strength": int(strength),
                        "training_eligible": True,
                        "loss_mask": [
                            {"family": target.target_family, "field": target.name}
                        ],
                        "dna_text": replace_gene_pair(
                            base_dna, target.name, allele, int(strength)
                        ),
                    }
                )
    return rows


def _plan_identity(
    source_schema_sha256: str,
    training_schema_sha256: str,
    bases: Sequence[LockedBase],
    targets: Sequence[TargetField],
    strengths: Sequence[int],
    baseline_repeats: int,
    split_ratios: tuple[float, float, float],
    split_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_kind": PROTOCOL_KIND,
        "source_schema_sha256": source_schema_sha256,
        "training_schema_sha256": training_schema_sha256,
        "bases": [
            {
                "base_dna_id": base.base_id,
                "source_sample_id": base.source_sample_id,
                "dna_sha256": base.dna_sha256,
                "split": base.split,
            }
            for base in bases
        ],
        "target_fields": [
            {
                "name": target.name,
                "source_type": target.source_type,
                "target_family": target.target_family,
                "alleles": [value[1] for value in target.alleles],
            }
            for target in targets
        ],
        "strengths": list(strengths),
        "baseline_repeats": int(baseline_repeats),
        "split_ratios": dict(zip(SPLIT_NAMES, split_ratios)),
        "split_seed": int(split_seed),
        "ordering": "base_then_field_then_allele_or_class_then_strength",
    }


def prepare_targeted_protocol(
    source_schema_path: Path,
    training_schema_path: Path,
    bases_manifest_path: Path,
    output_dir: Path,
    *,
    fields: Sequence[str] = DEFAULT_TARGET_FIELDS,
    strengths: Sequence[int] = DEFAULT_STRENGTHS,
    baseline_repeats: int = 5,
    split_ratios: tuple[float, float, float] = (0.75, 0.125, 0.125),
    split_seed: int = 20260718,
) -> dict[str, Any]:
    source_schema_path = source_schema_path.resolve()
    training_schema_path = training_schema_path.resolve()
    bases_manifest_path = bases_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if baseline_repeats < 0:
        raise ValueError("baseline_repeats 不能小于 0")
    strengths = tuple(
        parse_value_sequence(",".join(str(value) for value in strengths))
    )
    source_schema = _read_json(source_schema_path)
    training_schema = _read_json(training_schema_path)
    targets = resolve_target_fields(source_schema, training_schema, fields)
    verification_fields = [
        str(spec["name"])
        for family in ("signed_fields", "categorical_fields")
        for spec in source_schema.get(family, [])
    ]
    bases = load_and_lock_bases(
        bases_manifest_path,
        verification_fields,
        split_ratios=split_ratios,
        split_seed=split_seed,
    )
    source_schema_sha256 = _sha256_file(source_schema_path)
    training_schema_sha256 = _sha256_file(training_schema_path)
    identity = _plan_identity(
        source_schema_sha256,
        training_schema_sha256,
        bases,
        targets,
        strengths,
        baseline_repeats,
        split_ratios,
        split_seed,
    )
    plan_sha256 = sha256_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True)
    )
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        existing = _read_json(protocol_path)
        if existing.get("plan_sha256") != plan_sha256:
            raise RuntimeError(
                "输出目录已有不同的 protocol.json；请使用新目录，避免混写样本"
            )

    for base in bases:
        copied = output_dir / base.copied_dna_path
        if copied.is_file() and sha256_text(
            copied.read_text(encoding="utf-8")
        ) != base.dna_sha256:
            raise RuntimeError(f"锁定基础 DNA 内容冲突: {copied}")
        atomic_write_text(copied, base.dna_text)

    variant_rows: list[dict[str, Any]] = []
    global_index = 0
    for base in bases:
        field_rows = _field_blueprints(base.dna_text, targets, strengths)
        scheduled = interleave_baselines(
            field_rows, base.dna_text, baseline_repeats
        )
        rows_for_base: list[dict[str, Any]] = []
        for sequence_index, raw_item in enumerate(scheduled, 1):
            global_index += 1
            item = dict(raw_item)
            if item["kind"] == "baseline":
                item.update(
                    {
                        "source_type": "targeted_baseline",
                        "intervention_field": None,
                        "target_family": None,
                        "class_id": None,
                        "training_eligible": False,
                        "loss_mask": [],
                    }
                )
                suffix = f"baseline_{int(item['baseline_repeat']):02d}"
            else:
                suffix = (
                    f"{safe_component(str(item['field']))}_"
                    f"{safe_component(str(item['class_or_sign']))}_"
                    f"{int(item['strength']):03d}"
                )
            variant_id = f"tb{base.base_index:03d}_{sequence_index:04d}_{suffix}"
            targeted_sample_id = f"targeted_{plan_sha256[:10]}_{global_index:06d}"
            dna_relative = f"dna/{safe_component(base.base_id)}/{variant_id}.txt"
            render_relative = (
                f"renders/{safe_component(base.base_id)}/{variant_id}.png"
            )
            dna_text = str(item.pop("dna_text"))
            row = {
                "global_index": global_index,
                "base_sequence_index": sequence_index,
                "variant_id": variant_id,
                "sample_id": targeted_sample_id,
                "base_id": base.base_id,
                "base_dna_id": base.base_id,
                "base_index": base.base_index,
                "base_split": base.split,
                "source_sample_id": base.source_sample_id,
                "race_group": base.race_group,
                **item,
                "base_dna_sha256": base.dna_sha256,
                "dna_path": dna_relative,
                "render_path": render_relative,
                "dna_sha256": sha256_text(dna_text),
            }
            dna_path = output_dir / dna_relative
            if dna_path.is_file() and sha256_text(
                dna_path.read_text(encoding="utf-8")
            ) != row["dna_sha256"]:
                raise RuntimeError(f"变体 DNA 内容冲突: {dna_path}")
            atomic_write_text(dna_path, dna_text)
            rows_for_base.append(row)

        zero_references = {
            (row["field"], row["allele"]): row["variant_id"]
            for row in rows_for_base
            if row["kind"] == "field" and row["strength"] == 0
        }
        baseline_ids = [
            row["variant_id"]
            for row in rows_for_base
            if row["kind"] == "baseline"
        ]
        for row in rows_for_base:
            if row["kind"] == "field":
                row["zero_reference_variant_id"] = zero_references[
                    (row["field"], row["allele"])
                ]
            else:
                row["zero_reference_variant_id"] = None
            row["baseline_reference_variant_ids"] = baseline_ids
        variant_rows.extend(rows_for_base)

    split_base_counts = {
        split: sum(base.split == split for base in bases) for split in SPLIT_NAMES
    }
    split_variant_counts = {
        split: sum(row["base_split"] == split for row in variant_rows)
        for split in SPLIT_NAMES
    }
    field_counts = {
        target.name: sum(
            row.get("intervention_field") == target.name for row in variant_rows
        )
        for target in targets
    }
    _atomic_jsonl(
        output_dir / "bases.jsonl", [base.manifest_row() for base in bases]
    )
    _atomic_jsonl(output_dir / "variants.jsonl", variant_rows)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_kind": PROTOCOL_KIND,
        "created_at": utc_now(),
        "plan_sha256": plan_sha256,
        "source_schema_path": str(source_schema_path),
        "source_schema_sha256": source_schema_sha256,
        "training_schema_path": str(training_schema_path),
        "training_schema_sha256": training_schema_sha256,
        "bases_manifest_path": str(bases_manifest_path),
        "base_count": len(bases),
        "target_fields": [target.name for target in targets],
        "target_field_count": len(targets),
        "strengths": list(strengths),
        "baseline_repeats_per_base": baseline_repeats,
        "total_variants": len(variant_rows),
        "training_eligible_variants": sum(
            bool(row["training_eligible"]) for row in variant_rows
        ),
        "field_variant_counts": field_counts,
        "base_split_counts": split_base_counts,
        "variant_split_counts": split_variant_counts,
        "base_split_policy": "explicit_or_exact_hash_shuffle",
        "split_ratios": dict(zip(SPLIT_NAMES, split_ratios)),
        "split_seed": split_seed,
        "verification_policy": "schema_fields_and_colors_round_trip_required",
        "verification_fields": verification_fields,
        "training_policy": {
            "random_to_targeted_batch_ratio": "4:1_initial",
            "intervention_field_loss_weight": 1.0,
            "non_intervention_field_loss_weight": "0_to_0.1",
            "baseline_training_eligible": False,
            "split_group": "base_dna_id",
        },
        "paths": {
            "bases": "bases.jsonl",
            "variants": "variants.jsonl",
            "dna": "dna",
            "renders": "renders",
            "render_manifest": "render_manifest.jsonl",
            "errors": "errors.jsonl",
        },
    }
    _atomic_json(protocol_path, protocol)
    return protocol


def _filter_variants(
    variants: Sequence[PlannedVariant],
    *,
    base_ids: Sequence[str] | None,
    splits: Sequence[str] | None,
    fields: Sequence[str] | None,
) -> list[PlannedVariant]:
    selected = list(variants)
    if base_ids:
        requested = set(base_ids)
        available = {variant.base_id for variant in variants}
        missing = requested - available
        if missing:
            raise ValueError("未知 base_id: " + ", ".join(sorted(missing)))
        selected = [variant for variant in selected if variant.base_id in requested]
    if splits:
        requested_splits = set(splits)
        selected = [
            variant
            for variant in selected
            if variant.metadata.get("base_split") in requested_splits
        ]
    if fields:
        requested_fields = set(fields)
        available_fields = {
            str(variant.field) for variant in variants if variant.field is not None
        }
        missing_fields = requested_fields - available_fields
        if missing_fields:
            raise ValueError("未知 field: " + ", ".join(sorted(missing_fields)))
        selected = [
            variant
            for variant in selected
            if variant.kind == "baseline" or variant.field in requested_fields
        ]
    return selected


def collection_status(experiment_dir: Path) -> dict[str, Any]:
    protocol, variants = load_plan(experiment_dir)
    if protocol.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("这不是定向样本采集计划")
    completed = completed_variant_ids(experiment_dir.resolve())
    by_split: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        subset = [
            variant
            for variant in variants
            if variant.metadata.get("base_split") == split
        ]
        by_split[split] = {
            "total": len(subset),
            "completed": sum(variant.variant_id in completed for variant in subset),
        }
    by_field: dict[str, dict[str, int]] = {}
    for field in protocol.get("target_fields", []):
        subset = [variant for variant in variants if variant.field == field]
        by_field[str(field)] = {
            "total": len(subset),
            "completed": sum(variant.variant_id in completed for variant in subset),
        }
    return {
        "protocol_kind": PROTOCOL_KIND,
        "plan_sha256": protocol.get("plan_sha256"),
        "total": len(variants),
        "completed": len(completed & {variant.variant_id for variant in variants}),
        "remaining": len(variants)
        - len(completed & {variant.variant_id for variant in variants}),
        "by_split": by_split,
        "by_field": by_field,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select-bases", help="从现有标签中选择形态分散的基础 DNA"
    )
    select.add_argument("--labels", type=Path, required=True)
    select.add_argument("--dna-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--count", type=int, default=32)
    select.add_argument(
        "--include-ids",
        type=Path,
        action="append",
        help="每行一个允许选作 base 的 sample ID，可重复指定；建议传旧 train IDs",
    )
    select.add_argument(
        "--exclude-ids",
        type=Path,
        action="append",
        help="每行一个禁止选作 base 的 sample ID，可重复指定",
    )
    select.add_argument("--limit", type=int)
    select.add_argument("--progress-every", type=int, default=2000)

    prepare = subparsers.add_parser("prepare", help="生成并锁定定向采集计划")
    prepare.add_argument("--source-schema", type=Path, required=True)
    prepare.add_argument("--training-schema", type=Path, required=True)
    prepare.add_argument("--bases-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--field",
        action="append",
        help="只采指定字段，可重复；默认使用扩展方案中的 14 个弱字段",
    )
    prepare.add_argument(
        "--include-diagnostic-chin-cleft",
        action="store_true",
        help="额外加入仅建议小规模诊断的 face_detail_chin_cleft",
    )
    prepare.add_argument(
        "--strengths",
        default=",".join(str(value) for value in DEFAULT_STRENGTHS),
        help="0..255 强度列表或 start:end:step；默认 0,64,128,192,255",
    )
    prepare.add_argument("--baseline-repeats", type=int, default=5)
    prepare.add_argument(
        "--split-ratios",
        type=parse_split_ratios,
        default=(0.75, 0.125, 0.125),
        metavar="TRAIN,VAL,TEST",
    )
    prepare.add_argument("--split-seed", type=int, default=20260718)

    run = subparsers.add_parser("run", help="运行或恢复已准备的采集计划")
    run.add_argument("experiment", type=Path)
    run.add_argument("--settings", type=Path)
    run.add_argument("--base-id", action="append")
    run.add_argument("--split", action="append", choices=SPLIT_NAMES)
    run.add_argument("--field", action="append")
    run.add_argument("--limit", type=int)
    run.add_argument("--progress-every", type=int, default=10)
    run.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="显示计划完成度")
    status.add_argument("experiment", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select-bases":
        rows = select_diverse_bases(
            args.labels,
            args.dna_dir,
            args.output,
            count=args.count,
            included_sample_ids=(
                load_sample_id_files(args.include_ids)
                if args.include_ids is not None
                else None
            ),
            excluded_sample_ids=load_sample_id_files(args.exclude_ids),
            limit=args.limit,
            progress_every=args.progress_every,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "selected": len(rows),
                    "first_base": rows[0]["base_dna_id"],
                    "last_base": rows[-1]["base_dna_id"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "prepare":
        fields = list(args.field or DEFAULT_TARGET_FIELDS)
        if args.include_diagnostic_chin_cleft:
            fields.append(DIAGNOSTIC_FIELD)
        protocol = prepare_targeted_protocol(
            args.source_schema,
            args.training_schema,
            args.bases_manifest,
            args.output,
            fields=fields,
            strengths=parse_value_sequence(args.strengths),
            baseline_repeats=args.baseline_repeats,
            split_ratios=args.split_ratios,
            split_seed=args.split_seed,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "plan_sha256": protocol["plan_sha256"],
                    "bases": protocol["base_count"],
                    "fields": protocol["target_field_count"],
                    "total_variants": protocol["total_variants"],
                    "training_eligible": protocol["training_eligible_variants"],
                    "base_split_counts": protocol["base_split_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "status":
        print(
            json.dumps(
                collection_status(args.experiment.resolve()),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    experiment_dir = args.experiment.resolve()
    protocol, variants = load_plan(experiment_dir)
    if protocol.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("这不是 targeted_sample_collector.py 生成的计划")
    variants = _filter_variants(
        variants,
        base_ids=args.base_id,
        splits=args.split,
        fields=args.field,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        variants = variants[: args.limit]
    if not variants:
        raise ValueError("筛选后没有待运行的变体")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan_sha256": protocol.get("plan_sha256"),
                    "selected_variants": len(variants),
                    "selected_bases": len({variant.base_id for variant in variants}),
                    "selected_fields": sorted(
                        {variant.field for variant in variants if variant.field}
                    ),
                    "baselines": sum(
                        variant.kind == "baseline" for variant in variants
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    settings = (
        _read_json(args.settings.resolve())
        if args.settings is not None
        else load_user_settings()
    )
    config = automation_config_from_settings(settings)
    if config.verify_copy_button is None:
        raise ValueError(
            "正式定向采集必须先在 dna_field_sweep_tool.py 中记录复制 DNA 验证按钮"
        )
    backend = WindowsAutomationBackend(config)
    progress_every = max(1, int(args.progress_every))

    def progress(
        done: int,
        total: int,
        variant: PlannedVariant,
        status_value: str,
    ) -> None:
        if status_value == "failed" or done % progress_every == 0 or done == total:
            print(
                f"[{done}/{total}] {variant.base_id} "
                f"{variant.variant_id} [{status_value}]",
                flush=True,
            )

    result = run_experiment(
        experiment_dir,
        protocol,
        variants,
        backend,
        retries=config.retries,
        verification_fields=protocol["verification_fields"],
        on_progress=progress,
    )
    print(
        f"运行完成: completed={result.completed}, skipped={result.skipped}, "
        f"attempted={result.attempted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
