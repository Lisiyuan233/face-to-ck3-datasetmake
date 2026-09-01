#!/usr/bin/env python3
"""Build the controlled CK3 DNA field-identifiability experiment.

The builder has two jobs:

* select one deterministic, median-like real sample from every race group; and
* expand the training schema into signed negative/positive and categorical
  class variants, with unchanged baseline renders interleaved through each
  base-face plan.

NumPy is used when available for a single-pass, vectorized base selection; a
standard-library two-pass fallback is retained.  The generated protocol is
immutable: rerunning into an existing directory is allowed only when the plan
hash is unchanged, which keeps render manifests safe to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from dna_field_sweep_tool import parse_value_sequence, replace_gene_pair, safe_component
from dna_normalizer import BYTE_MAX, parse_dna, validate_schema


PROTOCOL_VERSION = 1
DEFAULT_GROUP_COUNT = 17
DEFAULT_GROUP_SIZE = 30_000
DEFAULT_STRENGTHS = (0, 128, 255)
DEFAULT_BASELINE_REPEATS = 5
SAMPLE_NUMBER_RE = re.compile(r"(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            yield line_number, value


def load_raw_schema(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    schema = json.loads(raw.decode("utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("schema 顶层必须是 JSON 对象")
    validate_schema(schema)
    names = [
        str(spec["name"])
        for family in ("signed_fields", "categorical_fields")
        for spec in schema[family]
    ]
    if len(names) != len(set(names)):
        raise ValueError("schema 中存在重复字段")
    return schema, sha256_bytes(raw)


def race_group_for_label(
    label: dict[str, Any], sample_id: str, group_size: int
) -> int:
    explicit = label.get("race_group")
    if explicit is not None:
        return int(explicit)
    match = SAMPLE_NUMBER_RE.search(sample_id)
    if match is None:
        raise ValueError(f"sample_id 无法推导 race_group: {sample_id!r}")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError(f"sample_id 编号必须从 1 开始: {sample_id!r}")
    return (number - 1) // group_size


def _quantize_unit(value: Any, *, signed: bool) -> int:
    numeric = float(value)
    minimum, maximum = (-1.0, 1.0) if signed else (0.0, 1.0)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"归一化值超出 [{minimum}, {maximum}]: {value!r}")
    return round(numeric * BYTE_MAX)


def _validate_label_shape(label: dict[str, Any], schema: dict[str, Any]) -> None:
    expected = {
        "signed": len(schema["signed_fields"]),
        "categorical_class": len(schema["categorical_fields"]),
        "categorical_strength": len(schema["categorical_fields"]),
        "colors": 2 * len(schema.get("color_fields", [])),
    }
    for key, length in expected.items():
        values = label.get(key)
        if not isinstance(values, list) or len(values) != length:
            actual = len(values) if isinstance(values, list) else type(values).__name__
            raise ValueError(f"label {key} 长度为 {actual}，期望 {length}")


def _histogram_median(histogram: Sequence[int], offset: int = 0) -> float:
    total = sum(histogram)
    if total <= 0:
        raise ValueError("不能从空直方图计算中位数")

    def order_statistic(rank: int) -> int:
        cumulative = 0
        for index, count in enumerate(histogram):
            cumulative += count
            if cumulative > rank:
                return index - offset
        raise RuntimeError("直方图计数不一致")

    lower = order_statistic((total - 1) // 2)
    upper = order_statistic(total // 2)
    return (lower + upper) / 2.0


@dataclass
class _GroupHistogram:
    signed: list[list[int]]
    categorical_class: list[list[int]]
    categorical_strength: list[list[int]]
    colors: list[list[int]]
    count: int = 0

    @classmethod
    def create(cls, schema: dict[str, Any]) -> "_GroupHistogram":
        return cls(
            signed=[[0] * (BYTE_MAX * 2 + 1) for _ in schema["signed_fields"]],
            categorical_class=[
                [0] * len(spec["classes"]) for spec in schema["categorical_fields"]
            ],
            categorical_strength=[
                [0] * (BYTE_MAX + 1) for _ in schema["categorical_fields"]
            ],
            colors=[
                [0] * (BYTE_MAX + 1)
                for _ in range(2 * len(schema.get("color_fields", [])))
            ],
        )

    def add(self, label: dict[str, Any], schema: dict[str, Any]) -> None:
        _validate_label_shape(label, schema)
        for histogram, value in zip(self.signed, label["signed"]):
            histogram[_quantize_unit(value, signed=True) + BYTE_MAX] += 1
        for field_index, (histogram, value) in enumerate(
            zip(self.categorical_class, label["categorical_class"])
        ):
            class_id = int(value)
            if not 0 <= class_id < len(histogram):
                name = schema["categorical_fields"][field_index]["name"]
                raise ValueError(f"{name}: class id {class_id} 超出 schema")
            histogram[class_id] += 1
        for histogram, value in zip(
            self.categorical_strength, label["categorical_strength"]
        ):
            histogram[_quantize_unit(value, signed=False)] += 1
        for histogram, value in zip(self.colors, label["colors"]):
            histogram[_quantize_unit(value, signed=False)] += 1
        self.count += 1


@dataclass(frozen=True)
class _GroupTarget:
    signed: tuple[float, ...]
    categorical_class: tuple[int, ...]
    categorical_strength: tuple[float, ...]
    colors: tuple[float, ...]


def _target_from_histogram(histogram: _GroupHistogram) -> _GroupTarget:
    return _GroupTarget(
        signed=tuple(
            _histogram_median(values, BYTE_MAX) for values in histogram.signed
        ),
        categorical_class=tuple(
            max(range(len(values)), key=lambda index: (values[index], -index))
            for values in histogram.categorical_class
        ),
        categorical_strength=tuple(
            _histogram_median(values) for values in histogram.categorical_strength
        ),
        colors=tuple(_histogram_median(values) for values in histogram.colors),
    )


def _distance_to_target(label: dict[str, Any], target: _GroupTarget) -> float:
    distance = 0.0
    dimensions = 0
    for value, median in zip(label["signed"], target.signed):
        distance += abs(_quantize_unit(value, signed=True) - median) / (2 * BYTE_MAX)
        dimensions += 1
    for value, mode in zip(label["categorical_class"], target.categorical_class):
        distance += float(int(value) != mode)
        dimensions += 1
    for value, median in zip(label["categorical_strength"], target.categorical_strength):
        distance += abs(_quantize_unit(value, signed=False) - median) / BYTE_MAX
        dimensions += 1
    for value, median in zip(label["colors"], target.colors):
        distance += abs(_quantize_unit(value, signed=False) - median) / BYTE_MAX
        dimensions += 1
    return distance / max(1, dimensions)


@dataclass(frozen=True)
class BaseCandidate:
    race_group: int
    sample_id: str
    source_dna_path: Path
    selection_method: str
    selection_distance: float | None = None
    group_sample_count: int | None = None


def _select_median_bases_python(
    labels_path: Path,
    dna_dir: Path,
    schema: dict[str, Any],
    *,
    group_count: int = DEFAULT_GROUP_COUNT,
    group_size: int = DEFAULT_GROUP_SIZE,
    progress_every: int = 0,
    on_progress: Callable[[str, int], None] | None = None,
) -> list[BaseCandidate]:
    """Select the real sample closest to each group's mixed-type median.

    Continuous label medians are computed from byte-resolution histograms and
    categorical classes use their group mode.  A second streaming pass finds
    the sample with minimum mean normalized L1/Hamming distance.  Ties are
    resolved by sample_id, making the selection reproducible without NumPy.
    """

    if group_count <= 0 or group_size <= 0:
        raise ValueError("group_count 和 group_size 必须大于 0")
    histograms = [_GroupHistogram.create(schema) for _ in range(group_count)]
    scanned = 0
    for line_number, label in _read_jsonl(labels_path):
        scanned += 1
        sample_id = str(label.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{labels_path} 第 {line_number} 行缺少 sample_id")
        group = race_group_for_label(label, sample_id, group_size)
        if 0 <= group < group_count:
            try:
                histograms[group].add(label, schema)
            except ValueError as error:
                raise ValueError(
                    f"{labels_path} 第 {line_number} 行 ({sample_id}): {error}"
                ) from error
        if progress_every and scanned % progress_every == 0 and on_progress:
            on_progress("median_histograms", scanned)
    empty = [index for index, histogram in enumerate(histograms) if histogram.count == 0]
    if empty:
        raise ValueError("以下 race_group 没有标签: " + ", ".join(map(str, empty)))

    targets = [_target_from_histogram(histogram) for histogram in histograms]
    best: list[tuple[float, str, Path] | None] = [None] * group_count
    scanned = 0
    for line_number, label in _read_jsonl(labels_path):
        scanned += 1
        sample_id = str(label.get("sample_id", ""))
        group = race_group_for_label(label, sample_id, group_size)
        if not 0 <= group < group_count:
            continue
        try:
            _validate_label_shape(label, schema)
            distance = _distance_to_target(label, targets[group])
        except ValueError as error:
            raise ValueError(
                f"{labels_path} 第 {line_number} 行 ({sample_id}): {error}"
            ) from error
        dna_path = dna_dir / f"{sample_id}.txt"
        if not dna_path.is_file():
            continue
        candidate = (distance, sample_id, dna_path.resolve())
        if best[group] is None or candidate[:2] < best[group][:2]:
            best[group] = candidate
        if progress_every and scanned % progress_every == 0 and on_progress:
            on_progress("nearest_samples", scanned)

    missing = [index for index, value in enumerate(best) if value is None]
    if missing:
        raise ValueError(
            "以下 race_group 没有同时存在标签和 DNA 的样本: "
            + ", ".join(map(str, missing))
        )
    return [
        BaseCandidate(
            race_group=group,
            sample_id=value[1],
            source_dna_path=value[2],
            selection_method="closest_to_group_mixed_median",
            selection_distance=round(value[0], 10),
            group_sample_count=histograms[group].count,
        )
        for group, item in enumerate(best)
        for value in [item]
        if value is not None
    ]


def _select_median_bases_numpy(
    labels_path: Path,
    dna_dir: Path,
    schema: dict[str, Any],
    *,
    group_count: int,
    group_size: int,
    progress_every: int,
    on_progress: Callable[[str, int], None] | None,
) -> list[BaseCandidate]:
    """Single-pass vectorized selector used when NumPy is available."""

    import numpy as np

    signed_dim = len(schema["signed_fields"])
    categorical_dim = len(schema["categorical_fields"])
    color_dim = 2 * len(schema.get("color_fields", []))
    signed_values = np.empty(
        (group_count, group_size, signed_dim), dtype=np.int16
    )
    categorical_classes = np.empty(
        (group_count, group_size, categorical_dim), dtype=np.uint8
    )
    categorical_strengths = np.empty(
        (group_count, group_size, categorical_dim), dtype=np.uint8
    )
    colors = np.empty((group_count, group_size, color_dim), dtype=np.uint8)
    sample_ids: list[list[str | None]] = [
        [None] * group_size for _ in range(group_count)
    ]
    counts = [0] * group_count
    batch_groups: list[int] = []
    batch_slots: list[int] = []
    batch_signed: list[list[Any]] = []
    batch_classes: list[list[Any]] = []
    batch_strengths: list[list[Any]] = []
    batch_colors: list[list[Any]] = []

    def flush_batch() -> None:
        if not batch_groups:
            return
        groups = np.asarray(batch_groups, dtype=np.intp)
        slots = np.asarray(batch_slots, dtype=np.intp)
        try:
            signed = np.asarray(batch_signed, dtype=np.float64)
            classes = np.asarray(batch_classes, dtype=np.int64)
            strengths = np.asarray(batch_strengths, dtype=np.float64)
            color_array = np.asarray(batch_colors, dtype=np.float64)
        except ValueError as error:
            raise ValueError("labels 中存在长度不一致的向量") from error
        expected_shapes = {
            "signed": (len(groups), signed_dim),
            "categorical_class": (len(groups), categorical_dim),
            "categorical_strength": (len(groups), categorical_dim),
            "colors": (len(groups), color_dim),
        }
        actual_shapes = {
            "signed": signed.shape,
            "categorical_class": classes.shape,
            "categorical_strength": strengths.shape,
            "colors": color_array.shape,
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(
                    f"label {name} shape={actual_shapes[name]}，期望 {expected}"
                )
        if not np.isfinite(signed).all() or (signed < -1.0).any() or (signed > 1.0).any():
            raise ValueError("signed label 超出 [-1, 1]")
        if (
            not np.isfinite(strengths).all()
            or (strengths < 0.0).any()
            or (strengths > 1.0).any()
        ):
            raise ValueError("categorical_strength label 超出 [0, 1]")
        if (
            not np.isfinite(color_array).all()
            or (color_array < 0.0).any()
            or (color_array > 1.0).any()
        ):
            raise ValueError("colors label 超出 [0, 1]")
        for field_index, spec in enumerate(schema["categorical_fields"]):
            field_classes = classes[:, field_index]
            if (field_classes < 0).any() or (
                field_classes >= len(spec["classes"])
            ).any():
                raise ValueError(f"{spec['name']}: categorical class 超出 schema")
        signed_values[groups, slots] = np.rint(signed * BYTE_MAX).astype(np.int16)
        categorical_classes[groups, slots] = classes.astype(np.uint8)
        categorical_strengths[groups, slots] = np.rint(
            strengths * BYTE_MAX
        ).astype(np.uint8)
        colors[groups, slots] = np.rint(color_array * BYTE_MAX).astype(np.uint8)
        batch_groups.clear()
        batch_slots.clear()
        batch_signed.clear()
        batch_classes.clear()
        batch_strengths.clear()
        batch_colors.clear()

    scanned = 0
    for line_number, label in _read_jsonl(labels_path):
        scanned += 1
        sample_id = str(label.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{labels_path} 第 {line_number} 行缺少 sample_id")
        group = race_group_for_label(label, sample_id, group_size)
        if not 0 <= group < group_count:
            continue
        slot = counts[group]
        if slot >= group_size:
            raise ValueError(
                f"race_group {group} 超过配置的 group_size={group_size}"
            )
        try:
            batch_signed.append(label["signed"])
            batch_classes.append(label["categorical_class"])
            batch_strengths.append(label["categorical_strength"])
            batch_colors.append(label["colors"])
        except KeyError as error:
            raise ValueError(
                f"{labels_path} 第 {line_number} 行 ({sample_id}) 缺少 {error.args[0]}"
            ) from error
        batch_groups.append(group)
        batch_slots.append(slot)
        sample_ids[group][slot] = sample_id
        counts[group] += 1
        if len(batch_groups) >= 5_000:
            flush_batch()
        if progress_every and scanned % progress_every == 0 and on_progress:
            on_progress("median_and_nearest", scanned)
    flush_batch()

    empty = [group for group, count in enumerate(counts) if count == 0]
    if empty:
        raise ValueError("以下 race_group 没有标签: " + ", ".join(map(str, empty)))

    candidates = []
    dimensions = signed_dim + categorical_dim * 2 + color_dim
    for group, count in enumerate(counts):
        signed = signed_values[group, :count].astype(np.float32)
        classes = categorical_classes[group, :count]
        strengths = categorical_strengths[group, :count].astype(np.float32)
        color_array = colors[group, :count].astype(np.float32)
        signed_median = np.median(signed, axis=0)
        strength_median = np.median(strengths, axis=0)
        color_median = np.median(color_array, axis=0)
        class_modes = np.asarray(
            [
                np.bincount(
                    classes[:, field_index],
                    minlength=len(spec["classes"]),
                ).argmax()
                for field_index, spec in enumerate(schema["categorical_fields"])
            ],
            dtype=np.uint8,
        )
        distance = np.abs(signed - signed_median).sum(axis=1) / (2 * BYTE_MAX)
        distance += (classes != class_modes).sum(axis=1)
        distance += np.abs(strengths - strength_median).sum(axis=1) / BYTE_MAX
        distance += np.abs(color_array - color_median).sum(axis=1) / BYTE_MAX
        distance /= max(1, dimensions)
        ids = [value for value in sample_ids[group][:count] if value is not None]
        if len(ids) != count:
            raise RuntimeError(f"race_group {group} 的 sample_id 缓冲不完整")
        order = np.lexsort((np.asarray(ids), distance))
        selected: tuple[str, Path, float] | None = None
        for index in order:
            sample_id = ids[int(index)]
            dna_path = dna_dir / f"{sample_id}.txt"
            if dna_path.is_file():
                selected = (sample_id, dna_path.resolve(), float(distance[int(index)]))
                break
        if selected is None:
            raise ValueError(f"race_group {group} 没有同时存在标签和 DNA 的样本")
        candidates.append(
            BaseCandidate(
                race_group=group,
                sample_id=selected[0],
                source_dna_path=selected[1],
                selection_method="closest_to_group_mixed_median",
                selection_distance=round(selected[2], 10),
                group_sample_count=count,
            )
        )
    return candidates


def select_median_bases(
    labels_path: Path,
    dna_dir: Path,
    schema: dict[str, Any],
    *,
    group_count: int = DEFAULT_GROUP_COUNT,
    group_size: int = DEFAULT_GROUP_SIZE,
    progress_every: int = 0,
    on_progress: Callable[[str, int], None] | None = None,
) -> list[BaseCandidate]:
    """Select deterministic mixed-median representatives with an optional fast path."""

    try:
        import numpy  # noqa: F401
    except ImportError:
        return _select_median_bases_python(
            labels_path,
            dna_dir,
            schema,
            group_count=group_count,
            group_size=group_size,
            progress_every=progress_every,
            on_progress=on_progress,
        )
    return _select_median_bases_numpy(
        labels_path,
        dna_dir,
        schema,
        group_count=group_count,
        group_size=group_size,
        progress_every=progress_every,
        on_progress=on_progress,
    )


def load_base_manifest(path: Path, *, group_count: int) -> list[BaseCandidate]:
    candidates: list[BaseCandidate] = []
    for line_number, row in _read_jsonl(path):
        try:
            race_group = int(row["race_group"])
            sample_id = str(row["sample_id"])
            raw_path = Path(str(row["dna_path"])).expanduser()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} 第 {line_number} 行 bases 字段无效") from error
        dna_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
        if not dna_path.is_file():
            raise FileNotFoundError(dna_path)
        candidates.append(
            BaseCandidate(
                race_group=race_group,
                sample_id=sample_id,
                source_dna_path=dna_path.resolve(),
                selection_method=str(row.get("selection_method", "explicit_manifest")),
                selection_distance=(
                    float(row["selection_distance"])
                    if row.get("selection_distance") is not None
                    else None
                ),
                group_sample_count=(
                    int(row["group_sample_count"])
                    if row.get("group_sample_count") is not None
                    else None
                ),
            )
        )
    groups = [candidate.race_group for candidate in candidates]
    expected = list(range(group_count))
    if sorted(groups) != expected:
        raise ValueError(
            f"bases manifest 必须恰好包含 race_group {expected}，实际为 {sorted(groups)}"
        )
    if len({candidate.sample_id for candidate in candidates}) != len(candidates):
        raise ValueError("bases manifest 中 sample_id 重复")
    return sorted(candidates, key=lambda value: value.race_group)


@dataclass(frozen=True)
class _LockedBase:
    race_group: int
    base_id: str
    sample_id: str
    source_dna_path: Path
    copied_dna_path: str
    dna_text: str
    dna_sha256: str
    selection_method: str
    selection_distance: float | None
    group_sample_count: int | None

    def manifest_row(self) -> dict[str, Any]:
        return {
            "race_group": self.race_group,
            "base_id": self.base_id,
            "sample_id": self.sample_id,
            "dna_path": self.copied_dna_path,
            "dna_sha256": self.dna_sha256,
            "source_dna_path": str(self.source_dna_path),
            "selection_method": self.selection_method,
            "selection_distance": self.selection_distance,
            "group_sample_count": self.group_sample_count,
        }


def _lock_bases(
    candidates: Sequence[BaseCandidate], schema: dict[str, Any]
) -> list[_LockedBase]:
    required = {
        str(spec["name"])
        for family in ("signed_fields", "categorical_fields")
        for spec in schema[family]
    }
    locked = []
    for candidate in sorted(candidates, key=lambda value: value.race_group):
        text = candidate.source_dna_path.read_text(encoding="utf-8-sig")
        record = parse_dna(text)
        missing = required - set(record.genes)
        if missing:
            raise ValueError(
                f"基础 DNA {candidate.sample_id} 缺少 schema 字段: "
                + ", ".join(sorted(missing))
            )
        base_id = f"race_{candidate.race_group:02d}_{safe_component(candidate.sample_id)}"
        copied_path = f"bases/{base_id}.txt"
        locked.append(
            _LockedBase(
                race_group=candidate.race_group,
                base_id=base_id,
                sample_id=candidate.sample_id,
                source_dna_path=candidate.source_dna_path,
                copied_dna_path=copied_path,
                dna_text=text,
                dna_sha256=sha256_text(text),
                selection_method=candidate.selection_method,
                selection_distance=candidate.selection_distance,
                group_sample_count=candidate.group_sample_count,
            )
        )
    return locked


def _field_blueprints(
    base_dna: str,
    schema: dict[str, Any],
    strengths: Sequence[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    field_variant_index = 0
    for spec in schema["signed_fields"]:
        for sign, allele in (
            ("negative", str(spec["negative_allele"])),
            ("positive", str(spec["positive_allele"])),
        ):
            for strength in strengths:
                field_variant_index += 1
                result.append(
                    {
                        "kind": "field",
                        "field_variant_index": field_variant_index,
                        "field": str(spec["name"]),
                        "field_type": "signed",
                        "class_or_sign": sign,
                        "allele": allele,
                        "strength": int(strength),
                        "dna_text": replace_gene_pair(
                            base_dna, str(spec["name"]), allele, int(strength)
                        ),
                    }
                )
    for spec in schema["categorical_fields"]:
        for class_id, allele in enumerate(spec["classes"]):
            for strength in strengths:
                field_variant_index += 1
                result.append(
                    {
                        "kind": "field",
                        "field_variant_index": field_variant_index,
                        "field": str(spec["name"]),
                        "field_type": "categorical",
                        "class_id": class_id,
                        "class_or_sign": str(allele),
                        "allele": str(allele),
                        "strength": int(strength),
                        "dna_text": replace_gene_pair(
                            base_dna, str(spec["name"]), str(allele), int(strength)
                        ),
                    }
                )
    return result


def interleave_baselines(
    field_variants: Sequence[dict[str, Any]],
    base_dna: str,
    baseline_repeats: int,
) -> list[dict[str, Any]]:
    if baseline_repeats < 0:
        raise ValueError("baseline_repeats 不能小于 0")
    total = len(field_variants)
    if baseline_repeats == 0:
        positions: list[int] = []
    elif baseline_repeats == 1:
        positions = [total // 2]
    else:
        positions = [
            round(repeat * total / (baseline_repeats - 1))
            for repeat in range(baseline_repeats)
        ]
    baselines: dict[int, list[int]] = {}
    for repeat, position in enumerate(positions, 1):
        baselines.setdefault(position, []).append(repeat)

    scheduled: list[dict[str, Any]] = []
    for position in range(total + 1):
        for repeat in baselines.get(position, []):
            scheduled.append(
                {
                    "kind": "baseline",
                    "baseline_repeat": repeat,
                    "field": None,
                    "field_type": None,
                    "class_or_sign": "baseline",
                    "allele": None,
                    "strength": None,
                    "dna_text": base_dna,
                }
            )
        if position < total:
            scheduled.append(dict(field_variants[position]))
    return scheduled


def _plan_identity(
    schema_sha256: str,
    bases: Sequence[_LockedBase],
    strengths: Sequence[int],
    baseline_repeats: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "schema_sha256": schema_sha256,
        "bases": [
            {
                "race_group": base.race_group,
                "base_id": base.base_id,
                "sample_id": base.sample_id,
                "dna_sha256": base.dna_sha256,
            }
            for base in bases
        ],
        "strengths": list(strengths),
        "baseline_repeats": baseline_repeats,
        "ordering": "schema_field_then_allele_or_class_then_strength_with_even_baselines",
    }


def build_protocol(
    schema_path: Path,
    candidates: Sequence[BaseCandidate],
    output_dir: Path,
    *,
    strengths: Sequence[int] = DEFAULT_STRENGTHS,
    baseline_repeats: int = DEFAULT_BASELINE_REPEATS,
) -> dict[str, Any]:
    schema_path = schema_path.resolve()
    output_dir = output_dir.resolve()
    schema, schema_sha256 = load_raw_schema(schema_path)
    strengths = tuple(
        parse_value_sequence(",".join(str(value) for value in strengths))
    )
    if not strengths:
        raise ValueError("至少需要一个强度")
    bases = _lock_bases(candidates, schema)
    if not bases:
        raise ValueError("没有基础 DNA")

    identity = _plan_identity(schema_sha256, bases, strengths, baseline_repeats)
    plan_sha256 = sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        existing = _read_json(protocol_path)
        if existing.get("plan_sha256") != plan_sha256:
            raise RuntimeError(
                "输出目录已有不同的 protocol.json；为避免混写截图，请使用新的输出目录"
            )

    for base in bases:
        base_path = output_dir / base.copied_dna_path
        if base_path.is_file() and sha256_text(
            base_path.read_text(encoding="utf-8")
        ) != base.dna_sha256:
            raise RuntimeError(f"锁定基础 DNA 内容冲突: {base_path}")
        _atomic_write_text(base_path, base.dna_text)

    variant_rows: list[dict[str, Any]] = []
    global_index = 0
    for base in bases:
        field_variants = _field_blueprints(base.dna_text, schema, strengths)
        scheduled = interleave_baselines(
            field_variants, base.dna_text, baseline_repeats
        )
        rows_for_base: list[dict[str, Any]] = []
        for sequence_index, item in enumerate(scheduled, 1):
            global_index += 1
            if item["kind"] == "baseline":
                suffix = f"baseline_{int(item['baseline_repeat']):02d}"
            else:
                suffix = (
                    f"{safe_component(str(item['field']))}_"
                    f"{safe_component(str(item['class_or_sign']))}_"
                    f"{int(item['strength']):03d}"
                )
            variant_id = f"b{base.race_group:02d}_{sequence_index:04d}_{suffix}"
            dna_relative = f"dna/{base.base_id}/{variant_id}.txt"
            render_relative = f"renders/{base.base_id}/{variant_id}.png"
            dna_text = str(item.pop("dna_text"))
            row = {
                "global_index": global_index,
                "base_sequence_index": sequence_index,
                "variant_id": variant_id,
                "base_id": base.base_id,
                "race_group": base.race_group,
                "sample_id": base.sample_id,
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
            _atomic_write_text(dna_path, dna_text)
            rows_for_base.append(row)

        zero_references = {
            (row["field"], row["allele"]): row["variant_id"]
            for row in rows_for_base
            if row["kind"] == "field" and row["strength"] == 0
        }
        baseline_ids = [
            row["variant_id"] for row in rows_for_base if row["kind"] == "baseline"
        ]
        for row in rows_for_base:
            if row["kind"] == "field":
                row["zero_reference_variant_id"] = zero_references[
                    (row["field"], row["allele"])
                ]
                row["baseline_reference_variant_ids"] = baseline_ids
            else:
                row["zero_reference_variant_id"] = None
                row["baseline_reference_variant_ids"] = baseline_ids
        variant_rows.extend(rows_for_base)

    signed_count = len(schema["signed_fields"])
    categorical_count = len(schema["categorical_fields"])
    class_count = sum(len(spec["classes"]) for spec in schema["categorical_fields"])
    field_variants_per_base = (
        signed_count * 2 * len(strengths) + class_count * len(strengths)
    )
    expected_per_base = field_variants_per_base + baseline_repeats
    if any(
        sum(1 for row in variant_rows if row["base_id"] == base.base_id)
        != expected_per_base
        for base in bases
    ):
        raise RuntimeError("内部错误：每个基础脸的计划数量不一致")

    _write_jsonl(output_dir / "bases.jsonl", [base.manifest_row() for base in bases])
    _write_jsonl(output_dir / "variants.jsonl", variant_rows)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "stage": 1,
        "plan_sha256": plan_sha256,
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha256,
        "base_count": len(bases),
        "race_groups": [base.race_group for base in bases],
        "signed_field_count": signed_count,
        "categorical_field_count": categorical_count,
        "categorical_class_count": class_count,
        "strengths": list(strengths),
        "signed_variants_per_base": signed_count * 2 * len(strengths),
        "categorical_variants_per_base": class_count * len(strengths),
        "field_variants_per_base": field_variants_per_base,
        "baseline_repeats_per_base": baseline_repeats,
        "variants_per_base": expected_per_base,
        "total_variants": len(variant_rows),
        "baseline_schedule": "evenly_interleaved_including_start_and_end",
        "verification_policy": "schema_fields_and_colors_round_trip_required",
        "verification_fields": [
            str(spec["name"])
            for family in ("signed_fields", "categorical_fields")
            for spec in schema[family]
        ],
        "paths": {
            "bases": "bases.jsonl",
            "variants": "variants.jsonl",
            "dna": "dna",
            "renders": "renders",
            "render_manifest": "render_manifest.jsonl",
            "errors": "errors.jsonl",
        },
    }
    _write_json(protocol_path, protocol)
    return protocol


def _base_source_rows(candidates: Sequence[BaseCandidate], output_path: Path) -> None:
    rows = []
    for candidate in candidates:
        try:
            relative = candidate.source_dna_path.resolve().relative_to(
                output_path.parent.resolve()
            )
            dna_path = relative.as_posix()
        except ValueError:
            dna_path = str(candidate.source_dna_path.resolve())
        rows.append(
            {
                "race_group": candidate.race_group,
                "sample_id": candidate.sample_id,
                "dna_path": dna_path,
                "selection_method": candidate.selection_method,
                "selection_distance": candidate.selection_distance,
                "group_sample_count": candidate.group_sample_count,
            }
        )
    _write_jsonl(output_path, rows)


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dna-dir", type=Path, required=True)
    parser.add_argument("--group-count", type=int, default=DEFAULT_GROUP_COUNT)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--progress-every", type=int, default=50_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select-bases", help="从 labels.jsonl 为每个 race group 选择中位代表"
    )
    _add_selection_arguments(select_parser)
    select_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="选择/读取 bases 并生成完整第一阶段 DNA 计划"
    )
    prepare_parser.add_argument("--schema", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--bases-manifest", type=Path)
    prepare_parser.add_argument("--labels", type=Path)
    prepare_parser.add_argument("--dna-dir", type=Path)
    prepare_parser.add_argument("--group-count", type=int, default=DEFAULT_GROUP_COUNT)
    prepare_parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    prepare_parser.add_argument("--progress-every", type=int, default=50_000)
    prepare_parser.add_argument(
        "--strengths", default=",".join(map(str, DEFAULT_STRENGTHS))
    )
    prepare_parser.add_argument(
        "--baseline-repeats", type=int, default=DEFAULT_BASELINE_REPEATS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema, _schema_sha256 = load_raw_schema(args.schema.resolve())

    def progress(phase: str, count: int) -> None:
        labels = {
            "median_histograms": "统计组中位数",
            "nearest_samples": "寻找最近真实样本",
            "median_and_nearest": "读取标签并批量选择",
        }
        label = labels.get(phase, phase)
        print(f"{label}: 已扫描 {count} 条标签...", flush=True)

    if args.command == "select-bases":
        candidates = select_median_bases(
            args.labels.resolve(),
            args.dna_dir.resolve(),
            schema,
            group_count=args.group_count,
            group_size=args.group_size,
            progress_every=args.progress_every,
            on_progress=progress,
        )
        _base_source_rows(candidates, args.output.resolve())
        print(f"已选择 {len(candidates)} 个基础 DNA -> {args.output.resolve()}")
        return 0

    if args.bases_manifest is not None:
        if args.labels is not None or args.dna_dir is not None:
            raise ValueError("使用 --bases-manifest 时不要同时提供 --labels/--dna-dir")
        candidates = load_base_manifest(
            args.bases_manifest.resolve(), group_count=args.group_count
        )
    else:
        if args.labels is None or args.dna_dir is None:
            raise ValueError(
                "未提供 --bases-manifest 时，必须同时提供 --labels 和 --dna-dir"
            )
        candidates = select_median_bases(
            args.labels.resolve(),
            args.dna_dir.resolve(),
            schema,
            group_count=args.group_count,
            group_size=args.group_size,
            progress_every=args.progress_every,
            on_progress=progress,
        )
    strengths = parse_value_sequence(args.strengths)
    protocol = build_protocol(
        args.schema,
        candidates,
        args.output,
        strengths=strengths,
        baseline_repeats=args.baseline_repeats,
    )
    print(
        f"已生成 {protocol['base_count']} 个 bases、{protocol['total_variants']} 个变体 "
        f"-> {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
