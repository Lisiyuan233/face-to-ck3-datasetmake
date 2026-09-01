#!/usr/bin/env python3
"""Analyze a completed CK3 DNA identifiability render protocol.

The analyzer deliberately separates geometry from the long-run exposure drift
observed in CK3.  Every screenshot is standardized independently before a
low-resolution gradient feature is measured.  Exact SHA-256 equality on bases
whose repeated baselines are bit-identical is used as the conservative gate for
merging signed alleles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


ANALYSIS_VERSION = 1
DEFAULT_FEATURE_SIZE = (200, 132)
NOISE_FLOOR = 0.0005
DETECTION_SNR = 1.0
CLASS_ALIAS_COSINE = 0.90


REGION_BOXES: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "face": ((0.08, 0.04, 0.95, 0.92), (0.05, 0.04, 0.98, 0.92)),
    "head": ((0.08, 0.03, 0.95, 0.90), (0.05, 0.03, 0.98, 0.90)),
    "forehead": ((0.18, 0.07, 0.88, 0.43), (0.18, 0.07, 0.96, 0.43)),
    "eyes": ((0.18, 0.28, 0.90, 0.52), (0.32, 0.27, 0.98, 0.52)),
    "nose": ((0.34, 0.33, 0.78, 0.69), (0.42, 0.31, 1.00, 0.69)),
    "mouth": ((0.30, 0.51, 0.82, 0.76), (0.34, 0.49, 0.97, 0.76)),
    "chin": ((0.26, 0.59, 0.86, 0.90), (0.20, 0.58, 0.96, 0.90)),
    "jaw": ((0.10, 0.45, 0.94, 0.91), (0.08, 0.44, 0.97, 0.91)),
    "cheeks": ((0.10, 0.38, 0.94, 0.72), (0.12, 0.36, 0.91, 0.72)),
    "ears": ((0.02, 0.31, 0.99, 0.66), (0.02, 0.29, 0.50, 0.66)),
    "neck": ((0.16, 0.69, 0.93, 1.00), (0.04, 0.68, 0.82, 1.00)),
}


@dataclass(frozen=True)
class RenderFeature:
    normalized_gray: np.ndarray

    @classmethod
    def from_path(cls, path: Path, size: tuple[int, int]) -> "RenderFeature":
        with Image.open(path) as image:
            resized = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            value = np.asarray(resized, dtype=np.float32)
        value = (value - float(value.mean())) / (float(value.std()) + 1e-6)
        return cls(value)

    def vector(self, region: str) -> np.ndarray:
        height, width = self.normalized_gray.shape
        split = width // 2
        if region == "front":
            patches = (self.normalized_gray[:, :split],)
        elif region == "side":
            patches = (self.normalized_gray[:, split:],)
        else:
            boxes = REGION_BOXES.get(region, REGION_BOXES["face"])
            patches = []
            for view_index, box in enumerate(boxes):
                view_left = 0 if view_index == 0 else split
                view_width = split if view_index == 0 else width - split
                x0 = view_left + int(round(box[0] * view_width))
                y0 = int(round(box[1] * height))
                x1 = view_left + int(round(box[2] * view_width))
                y1 = int(round(box[3] * height))
                patches.append(self.normalized_gray[y0:y1, x0:x1])
        gradients: list[np.ndarray] = []
        for patch in patches:
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                continue
            gradients.append((patch[:, 1:] - patch[:, :-1]).ravel())
            gradients.append((patch[1:, :] - patch[:-1, :]).ravel())
        if not gradients:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(gradients)


def feature_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"feature shape mismatch: {left.shape} != {right.shape}")
    return float(np.mean(np.abs(left - right)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    position = 0
    while position < len(array):
        end = position + 1
        while end < len(array) and array[order[end]] == array[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def spearman(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = rankdata(list(range(len(values))))
    y = rankdata(values)
    if float(y.std()) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def local_region(field: str) -> str:
    lowered = field.lower()
    if "neck" in lowered:
        return "neck"
    if "ear" in lowered:
        return "ears"
    if "eye" in lowered or "brow" in lowered:
        return "eyes"
    if "nose" in lowered:
        return "nose"
    if any(token in lowered for token in ("mouth", "lip", "philtrum", "nasolabial")):
        return "mouth"
    if "chin" in lowered:
        return "chin"
    if "jaw" in lowered:
        return "jaw"
    if any(token in lowered for token in ("cheek", "temple")):
        return "cheeks"
    if "forehead" in lowered:
        return "forehead"
    if "head" in lowered:
        return "head"
    return "face"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_path(experiment_dir: Path, protocol: dict[str, Any]) -> Path:
    raw = Path(str(protocol["schema_path"]))
    return raw if raw.is_absolute() else experiment_dir / raw


def _field_specs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field_type, family in (
        ("signed", "signed_fields"),
        ("categorical", "categorical_fields"),
    ):
        for spec in schema[family]:
            result[str(spec["name"])] = {"field_type": field_type, **spec}
    return result


def _connected_components(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            component.append(item)
            stack.extend(sorted(graph[item] - seen, reverse=True))
        components.append(sorted(component))
    return components


def analyze_experiment(
    experiment_dir: Path,
    *,
    feature_size: tuple[int, int] = DEFAULT_FEATURE_SIZE,
    output_dir: Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    output_dir = (output_dir or experiment_dir).resolve()
    protocol = _read_json(experiment_dir / "protocol.json")
    schema_path = _schema_path(experiment_dir, protocol).resolve()
    schema = _read_json(schema_path)
    specs = _field_specs(schema)
    manifest = _read_jsonl(experiment_dir / "render_manifest.jsonl")
    completed = [row for row in manifest if row.get("status") == "completed"]
    expected = int(protocol.get("total_variants", -1))
    if len(completed) != expected:
        raise RuntimeError(f"完成记录 {len(completed)} 与计划 {expected} 不一致")
    if len({str(row["variant_id"]) for row in completed}) != len(completed):
        raise RuntimeError("render_manifest.jsonl 存在重复 variant_id")
    missing = [
        row["render_path"]
        for row in completed
        if not (experiment_dir / str(row["render_path"])).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张截图，首个: {missing[0]}")

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_base[str(row["base_id"])].append(row)

    class_samples: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    signed_alias_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_pair_samples: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    quality_rows: list[dict[str, Any]] = []
    stable_bases: set[str] = set()

    for base_index, (base_id, rows) in enumerate(sorted(by_base.items()), 1):
        baseline_rows = sorted(
            (row for row in rows if row["kind"] == "baseline"),
            key=lambda row: int(row["baseline_repeat"]),
        )
        if len(baseline_rows) < 2:
            raise RuntimeError(f"{base_id} 至少需要两张 baseline")
        baseline_features = [
            RenderFeature.from_path(
                experiment_dir / str(row["render_path"]), feature_size
            )
            for row in baseline_rows
        ]
        noise_cache: dict[str, float] = {}

        def noise(region: str) -> float:
            if region not in noise_cache:
                vectors = [feature.vector(region) for feature in baseline_features]
                distances = [
                    feature_distance(left, right)
                    for left, right in itertools.combinations(vectors, 2)
                ]
                noise_cache[region] = max(percentile(distances, 95), NOISE_FLOOR)
            return noise_cache[region]

        baseline_hashes = [str(row["render_sha256"]) for row in baseline_rows]
        bit_exact = len(set(baseline_hashes)) == 1
        if bit_exact:
            stable_bases.add(base_id)
        brightness = []
        for row in baseline_rows:
            with Image.open(experiment_dir / str(row["render_path"])) as image:
                brightness.append(float(np.asarray(image.convert("L")).mean()))
        quality_rows.append(
            {
                "base_id": base_id,
                "baseline_count": len(baseline_rows),
                "unique_baseline_hashes": len(set(baseline_hashes)),
                "bit_exact_baselines": bit_exact,
                "brightness_min": min(brightness),
                "brightness_max": max(brightness),
                "brightness_range": max(brightness) - min(brightness),
                "front_noise_p95": noise("front"),
                "side_noise_p95": noise("side"),
                "face_noise_p95": noise("face"),
            }
        )

        field_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["kind"] == "field":
                field_groups[(str(row["field"]), str(row["field_type"]))].append(row)

        for (field, field_type), field_rows in sorted(field_groups.items()):
            region = local_region(field)
            features = {
                (str(row["class_or_sign"]), int(row["strength"])): (
                    RenderFeature.from_path(
                        experiment_dir / str(row["render_path"]), feature_size
                    ),
                    row,
                )
                for row in field_rows
            }
            classes = sorted({key[0] for key in features})
            for class_or_sign in classes:
                samples = {
                    strength: features[(class_or_sign, strength)][0]
                    for strength in (0, 128, 255)
                }
                metrics: dict[str, Any] = {
                    "base_id": base_id,
                    "field": field,
                    "field_type": field_type,
                    "class_or_sign": class_or_sign,
                }
                for metric_region in ("front", "side", region):
                    name = "local" if metric_region == region else metric_region
                    vectors = {
                        strength: sample.vector(metric_region)
                        for strength, sample in samples.items()
                    }
                    distance_128 = feature_distance(vectors[0], vectors[128])
                    distance_255 = feature_distance(vectors[0], vectors[255])
                    region_noise = noise(metric_region)
                    metrics[f"{name}_distance_128"] = distance_128
                    metrics[f"{name}_distance_255"] = distance_255
                    metrics[f"{name}_snr_128"] = distance_128 / region_noise
                    metrics[f"{name}_snr_255"] = distance_255 / region_noise
                    if name == "local":
                        metrics["monotonic"] = distance_255 >= distance_128
                        metrics["strength_correlation"] = spearman(
                            [0.0, distance_128, distance_255]
                        )
                class_samples[(field, field_type, class_or_sign)].append(metrics)

            if field_type == "signed":
                neg = {
                    strength: features[("negative", strength)]
                    for strength in (0, 128, 255)
                }
                pos = {
                    strength: features[("positive", strength)]
                    for strength in (0, 128, 255)
                }
                local_vectors_neg = {
                    strength: item[0].vector(region) for strength, item in neg.items()
                }
                local_vectors_pos = {
                    strength: item[0].vector(region) for strength, item in pos.items()
                }
                effect_neg = local_vectors_neg[255] - local_vectors_neg[0]
                effect_pos = local_vectors_pos[255] - local_vectors_pos[0]
                signed_alias_samples[field].append(
                    {
                        "base_id": base_id,
                        "stable_base": base_id in stable_bases,
                        "exact_0": neg[0][1]["render_sha256"] == pos[0][1]["render_sha256"],
                        "exact_128": neg[128][1]["render_sha256"]
                        == pos[128][1]["render_sha256"],
                        "exact_255": neg[255][1]["render_sha256"]
                        == pos[255][1]["render_sha256"],
                        "effect_cosine": cosine_similarity(effect_neg, effect_pos),
                        "effect_separation": feature_distance(effect_neg, effect_pos),
                    }
                )
            elif len(classes) >= 2:
                effects = {}
                magnitudes = {}
                for class_name in classes:
                    zero = features[(class_name, 0)][0].vector(region)
                    full = features[(class_name, 255)][0].vector(region)
                    effects[class_name] = full - zero
                    magnitudes[class_name] = float(np.mean(np.abs(effects[class_name])))
                for left_index, left in enumerate(classes):
                    for right in classes[left_index + 1 :]:
                        separation = feature_distance(effects[left], effects[right])
                        magnitude = max(
                            (magnitudes[left] + magnitudes[right]) / 2.0,
                            NOISE_FLOOR,
                        )
                        class_pair_samples[(field, left, right)].append(
                            {
                                "base_id": base_id,
                                "effect_cosine": cosine_similarity(
                                    effects[left], effects[right]
                                ),
                                "effect_separation": separation,
                                "separation_over_effect": separation / magnitude,
                            }
                        )
        if progress:
            print(f"[{base_index}/{len(by_base)}] analyzed {base_id}", flush=True)

    signed_decisions: dict[str, dict[str, Any]] = {}
    for field, samples in signed_alias_samples.items():
        stable = [sample for sample in samples if sample["stable_base"]]
        exact_all = bool(stable) and all(
            sample["exact_0"] and sample["exact_128"] and sample["exact_255"]
            for sample in stable
        )
        exact_mid_high = bool(stable) and all(
            sample["exact_128"] and sample["exact_255"] for sample in stable
        )
        signed_decisions[field] = {
            "robust_alias": exact_all,
            "context_dependent_alias": exact_mid_high and not exact_all,
            "stable_base_count": len(stable),
            "stable_exact_all_count": sum(
                sample["exact_0"] and sample["exact_128"] and sample["exact_255"]
                for sample in stable
            ),
            "median_effect_cosine": median(
                [float(sample["effect_cosine"]) for sample in samples]
            ),
            "median_effect_separation": median(
                [float(sample["effect_separation"]) for sample in samples]
            ),
        }

    alias_pair_rows: list[dict[str, Any]] = []
    class_alias_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (field, left, right), samples in sorted(class_pair_samples.items()):
        cosine = median([float(sample["effect_cosine"]) for sample in samples])
        separation = median(
            [float(sample["effect_separation"]) for sample in samples]
        )
        separation_ratio = median(
            [float(sample["separation_over_effect"]) for sample in samples]
        )
        candidate = cosine >= CLASS_ALIAS_COSINE
        if candidate:
            class_alias_edges[field].append((left, right))
        alias_pair_rows.append(
            {
                "field": field,
                "left_class": left,
                "right_class": right,
                "median_effect_cosine": cosine,
                "median_effect_separation": separation,
                "median_separation_over_effect": separation_ratio,
                "base_count": len(samples),
                "alias_candidate": candidate,
            }
        )
    class_alias_groups = {
        field: _connected_components(edges)
        for field, edges in sorted(class_alias_edges.items())
    }

    metric_rows: list[dict[str, Any]] = []
    class_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for (field, field_type, class_or_sign), samples in sorted(class_samples.items()):
        local_snr_128 = [float(sample["local_snr_128"]) for sample in samples]
        local_snr_255 = [float(sample["local_snr_255"]) for sample in samples]
        front_snr = [float(sample["front_snr_255"]) for sample in samples]
        side_snr = [float(sample["side_snr_255"]) for sample in samples]
        detection_128 = sum(value >= DETECTION_SNR for value in local_snr_128) / len(samples)
        detection_255 = sum(value >= DETECTION_SNR for value in local_snr_255) / len(samples)
        if detection_128 >= 0.80:
            threshold = 128
        elif detection_255 >= 0.80:
            threshold = 255
        else:
            threshold = 256
        value = {
            "field": field,
            "type": field_type,
            "class_or_sign": class_or_sign,
            "front_snr": median(front_snr),
            "side_snr": median(side_snr),
            "local_region": local_region(field),
            "local_snr": median(local_snr_255),
            "detection_rate": detection_255,
            "detection_rate_128": detection_128,
            "monotonicity": sum(bool(sample["monotonic"]) for sample in samples)
            / len(samples),
            "probe_accuracy": "",
            "strength_correlation": median(
                [float(sample["strength_correlation"]) for sample in samples]
            ),
            "worst_group_detection": min(local_snr_255),
            "base_count": len(samples),
            "recommended_visibility_threshold": threshold,
        }
        metric_rows.append(value)
        class_metrics[(field, class_or_sign)] = value

    field_rows: list[dict[str, Any]] = []
    tiers: dict[str, list[str]] = defaultdict(list)
    field_groups: dict[str, list[str]] = defaultdict(list)
    weights: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    robust_aliases: list[str] = []
    context_aliases: list[str] = []
    single_class_fields: list[str] = []

    for field, spec in sorted(specs.items()):
        field_type = str(spec["field_type"])
        metrics = [row for row in metric_rows if row["field"] == field]
        detection = sum(float(row["detection_rate"]) for row in metrics) / len(metrics)
        detection_128 = sum(float(row["detection_rate_128"]) for row in metrics) / len(metrics)
        monotonicity = sum(float(row["monotonicity"]) for row in metrics) / len(metrics)
        front_snr = median([float(row["front_snr"]) for row in metrics])
        side_snr = median([float(row["side_snr"]) for row in metrics])
        local_snr = median([float(row["local_snr"]) for row in metrics])
        worst = min(float(row["worst_group_detection"]) for row in metrics)
        class_count = 2 if field_type == "signed" else len(spec["classes"])
        alias_group = ""
        source_weight: float | None = None
        merged_weight: float | None = None

        signed_decision = signed_decisions.get(field, {})
        if signed_decision.get("robust_alias"):
            tier = "D"
            strategy = "merge_alleles_scalar"
            alias_group = f"{field}:negative=positive"
            robust_aliases.append(field)
            field_groups["signed_allele_alias"].append(field)
            source_weight = 0.0
        elif signed_decision.get("context_dependent_alias"):
            tier = "C"
            strategy = "piecewise_signed_with_visibility_mask"
            alias_group = f"{field}:mid_high_alias_low_strength_context"
            context_aliases.append(field)
            field_groups["context_dependent"].append(field)
        elif field_type == "categorical" and class_count == 1:
            single_class_fields.append(field)
            strategy = "strength_only"
            tier = "A" if detection >= 0.80 else "B" if detection >= 0.50 else "C"
            source_weight = 0.0
            field_groups["single_class_strength_only"].append(field)
        elif field in class_alias_groups:
            tier = "D"
            strategy = "group_similar_classes_plus_strength"
            alias_group = ";".join("=".join(group) for group in class_alias_groups[field])
            field_groups["categorical_alias_candidate"].append(field)
        elif detection >= 0.80 and monotonicity >= 0.80:
            tier = "A"
            strategy = "independent_prediction"
            field_groups["independent_prediction"].append(field)
        elif detection >= 0.50:
            tier = "B"
            strategy = "local_region_head"
            field_groups["local_region_head"].append(field)
        elif detection >= 0.20:
            tier = "C"
            strategy = "conditioned_local_head"
            field_groups["context_dependent"].append(field)
        else:
            tier = "E"
            strategy = "template_or_median_pending_local_probe"
            field_groups["weak_or_unresolved"].append(field)

        if detection_128 >= 0.80:
            visibility = 128
        elif detection >= 0.80:
            visibility = 255
        else:
            visibility = 256
        base_weight = round(min(1.0, max(0.10, detection)), 3)
        if tier == "E":
            base_weight = 0.0
        if strategy == "merge_alleles_scalar":
            merged_weight = base_weight
        recommended_weight = base_weight
        tiers[tier].append(field)
        row = {
            "field": field,
            "type": (
                "scalar_alias" if strategy == "merge_alleles_scalar" else field_type
            ),
            "class_count": class_count,
            "front_snr": front_snr,
            "side_snr": side_snr,
            "local_region": local_region(field),
            "local_snr": local_snr,
            "detection_rate": detection,
            "detection_rate_128": detection_128,
            "monotonicity": monotonicity,
            "probe_accuracy": "",
            "strength_correlation": median(
                [float(row["strength_correlation"]) for row in metrics]
            ),
            "worst_group_detection": worst,
            "alias_group": alias_group,
            "final_tier": tier,
            "recommended_strategy": strategy,
            "recommended_weight": recommended_weight,
            "source_head_weight": source_weight if source_weight is not None else recommended_weight,
            "merged_head_weight": merged_weight if merged_weight is not None else "",
            "recommended_visibility_threshold": visibility,
            "decision_status": "provisional_no_probe_exposure_normalized",
        }
        field_rows.append(row)
        weights[field] = {
            "tier": tier,
            "strategy": strategy,
            "recommended_weight": recommended_weight,
            "source_head_weight": row["source_head_weight"],
            "merged_head_weight": row["merged_head_weight"],
        }
        thresholds[field] = {
            "strength_byte": visibility,
            "normalized": None if visibility == 256 else visibility / 255.0,
            "detection_rate_128": detection_128,
            "detection_rate_255": detection,
        }

    field_lookup = {row["field"]: row for row in field_rows}
    for row in metric_rows:
        decision = field_lookup[str(row["field"])]
        row["alias_group"] = decision["alias_group"]
        row["final_tier"] = decision["final_tier"]
        row["recommended_strategy"] = decision["recommended_strategy"]
        row["recommended_weight"] = decision["recommended_weight"]
        row["decision_status"] = decision["decision_status"]

    scalar_fields = []
    retained_signed_fields = []
    for spec in schema["signed_fields"]:
        field = str(spec["name"])
        if field in robust_aliases:
            scalar_fields.append(
                {
                    "name": field,
                    "alleles": [spec["negative_allele"], spec["positive_allele"]],
                    "canonical_allele": spec["positive_allele"],
                    "normalization": "strength_div_255",
                    "observed_min": spec.get("observed_min"),
                    "observed_max": spec.get("observed_max"),
                    "recommended_visibility_threshold": thresholds[field]["strength_byte"],
                    "evidence": "bit_exact_all_strengths_on_stable_bases",
                }
            )
        else:
            copied = dict(spec)
            copied["recommended_visibility_threshold"] = thresholds[field]["strength_byte"]
            copied["prediction_strategy"] = field_lookup[field]["recommended_strategy"]
            retained_signed_fields.append(copied)

    recommended_schema = {
        "schema_version": 2,
        "sample_count": schema.get("sample_count"),
        "source_schema_path": Path(
            os.path.relpath(schema_path, output_dir)
        ).as_posix(),
        "source_schema_sha256": _sha256(schema_path),
        "analysis_version": ANALYSIS_VERSION,
        "normalization": {
            **schema.get("normalization", {}),
            "scalar_range": [0.0, 1.0],
        },
        "scalar_fields": scalar_fields,
        "signed_fields": retained_signed_fields,
        "categorical_fields": [
            {
                **spec,
                "prediction_strategy": field_lookup[str(spec["name"])][
                    "recommended_strategy"
                ],
                "recommended_visibility_threshold": thresholds[str(spec["name"])][
                    "strength_byte"
                ],
            }
            for spec in schema["categorical_fields"]
        ],
        "color_fields": schema.get("color_fields", []),
        "compatibility": {
            "status": "normalizer_and_training_v2_supported",
            "original_schema_is_not_modified": True,
            "legacy_v1_shard_labels_are_adapted_online": True,
        },
    }

    field_columns = [
        "field",
        "type",
        "class_count",
        "front_snr",
        "side_snr",
        "local_region",
        "local_snr",
        "detection_rate",
        "detection_rate_128",
        "monotonicity",
        "probe_accuracy",
        "strength_correlation",
        "worst_group_detection",
        "alias_group",
        "final_tier",
        "recommended_strategy",
        "recommended_weight",
        "source_head_weight",
        "merged_head_weight",
        "recommended_visibility_threshold",
        "decision_status",
    ]
    metric_columns = [
        "field",
        "type",
        "class_or_sign",
        "front_snr",
        "side_snr",
        "local_region",
        "local_snr",
        "detection_rate",
        "detection_rate_128",
        "monotonicity",
        "probe_accuracy",
        "strength_correlation",
        "worst_group_detection",
        "base_count",
        "alias_group",
        "final_tier",
        "recommended_strategy",
        "recommended_weight",
        "recommended_visibility_threshold",
        "decision_status",
    ]
    alias_columns = [
        "field",
        "left_class",
        "right_class",
        "median_effect_cosine",
        "median_effect_separation",
        "median_separation_over_effect",
        "base_count",
        "alias_candidate",
    ]
    quality_columns = [
        "base_id",
        "baseline_count",
        "unique_baseline_hashes",
        "bit_exact_baselines",
        "brightness_min",
        "brightness_max",
        "brightness_range",
        "front_noise_p95",
        "side_noise_p95",
        "face_noise_p95",
    ]
    _write_csv(output_dir / "field_identifiability.csv", field_rows, field_columns)
    _write_csv(
        output_dir / "field_class_identifiability.csv", metric_rows, metric_columns
    )
    _write_csv(
        output_dir / "allele_class_alias_matrix.csv", alias_pair_rows, alias_columns
    )
    _write_csv(output_dir / "render_quality.csv", quality_rows, quality_columns)
    _write_json(
        output_dir / "field_groups.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "robust_signed_allele_aliases": sorted(robust_aliases),
            "context_dependent_signed_aliases": sorted(context_aliases),
            "single_class_categorical_fields": sorted(single_class_fields),
            "categorical_alias_groups": class_alias_groups,
            "training_groups": {
                key: sorted(set(value)) for key, value in sorted(field_groups.items())
            },
            "tiers": {key: sorted(value) for key, value in sorted(tiers.items())},
        },
    )
    _write_json(output_dir / "recommended_loss_weights.json", weights)
    _write_json(output_dir / "recommended_visibility_thresholds.json", thresholds)
    _write_json(output_dir / "recommended_training_schema.json", recommended_schema)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_dir": str(experiment_dir),
        "plan_sha256": protocol.get("plan_sha256"),
        "render_count": len(completed),
        "base_count": len(by_base),
        "field_count": len(field_rows),
        "stable_baseline_base_count": len(stable_bases),
        "exposure_drift_base_count": len(by_base) - len(stable_bases),
        "robust_signed_alias_count": len(robust_aliases),
        "context_dependent_alias_count": len(context_aliases),
        "single_class_categorical_count": len(single_class_fields),
        "categorical_alias_field_count": len(class_alias_groups),
        "tier_counts": {tier: len(fields) for tier, fields in sorted(tiers.items())},
        "limitations": [
            "probe_accuracy_not_computed",
            "landmark_detector_not_computed",
            "lighting_drift_normalized_with_per_image_gradient_features",
            "categorical_aliases_are_candidates_until_rgb_texture_probe",
        ],
        "outputs": {
            "fields": "field_identifiability.csv",
            "classes": "field_class_identifiability.csv",
            "aliases": "allele_class_alias_matrix.csv",
            "quality": "render_quality.csv",
            "groups": "field_groups.json",
            "loss_weights": "recommended_loss_weights.json",
            "visibility_thresholds": "recommended_visibility_thresholds.json",
            "training_schema": "recommended_training_schema.json",
        },
    }
    _write_json(output_dir / "analysis_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, help="默认写入 experiment 目录")
    parser.add_argument("--feature-width", type=int, default=DEFAULT_FEATURE_SIZE[0])
    parser.add_argument("--feature-height", type=int, default=DEFAULT_FEATURE_SIZE[1])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.feature_width < 32 or args.feature_height < 32:
        raise ValueError("feature width/height 不能小于 32")
    summary = analyze_experiment(
        args.experiment,
        feature_size=(args.feature_width, args.feature_height),
        output_dir=args.output,
        progress=not args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
