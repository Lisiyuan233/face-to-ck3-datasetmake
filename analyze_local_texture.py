#!/usr/bin/env python3
"""Measure local CK3 texture signals in a completed identifiability sweep.

The analysis is deliberately dependency-light: Pillow and NumPy are enough.
Each target field is evaluated inside a fixed front/side facial region using
three complementary, per-crop illumination-normalized distances:

* block SSIM dissimilarity;
* gradient/edge difference;
* high-frequency (discrete Laplacian) residual difference.

All distances are divided by the P95 distance between repeated baseline
renders of the same base DNA and region.  The generated texture tiers are
therefore evidence about visible local signal, not a claim that the signal is
pure texture rather than a small local geometry change.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from analyze_identifiability_experiment import REGION_BOXES, local_region


ANALYSIS_VERSION = 1
DEFAULT_SIZE = (400, 264)
STRENGTHS = (0, 128, 255)
METRICS = ("ssim", "edge", "highpass")
NOISE_FLOORS = {"ssim": 0.0005, "edge": 0.001, "highpass": 0.002}
TEXTURE_DETECTION_SNR = 1.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
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


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(np.median(np.asarray(materialized, dtype=np.float64)))


def _percentile(values: Iterable[float], quantile: float) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(np.percentile(np.asarray(materialized, dtype=np.float64), quantile))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ranked = _rankdata(values)
    if float(ranked.std()) == 0:
        return 0.0
    expected = _rankdata(list(range(len(values))))
    return float(np.corrcoef(expected, ranked)[0, 1])


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    value = np.asarray(patch, dtype=np.float32)
    mean = float(value.mean())
    std = float(value.std())
    if std < 1e-6:
        return np.full(value.shape, 0.5, dtype=np.float32)
    normalized = np.clip((value - mean) / std, -3.0, 3.0)
    return normalized / 6.0 + 0.5


def block_ssim(left: np.ndarray, right: np.ndarray, block: int = 8) -> float:
    """Return mean non-overlapping block SSIM for two normalized patches."""

    if left.shape != right.shape:
        raise ValueError(f"patch shape mismatch: {left.shape} != {right.shape}")
    height = left.shape[0] - left.shape[0] % block
    width = left.shape[1] - left.shape[1] % block
    if height < block or width < block:
        return 1.0 if np.array_equal(left, right) else 0.0

    def windows(value: np.ndarray) -> np.ndarray:
        cropped = value[:height, :width]
        return cropped.reshape(height // block, block, width // block, block).transpose(
            0, 2, 1, 3
        )

    left_windows = windows(left)
    right_windows = windows(right)
    left_mean = left_windows.mean(axis=(2, 3))
    right_mean = right_windows.mean(axis=(2, 3))
    left_centered = left_windows - left_mean[:, :, None, None]
    right_centered = right_windows - right_mean[:, :, None, None]
    left_var = np.mean(left_centered * left_centered, axis=(2, 3))
    right_var = np.mean(right_centered * right_centered, axis=(2, 3))
    covariance = np.mean(left_centered * right_centered, axis=(2, 3))
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2.0 * left_mean * right_mean + c1)
        * (2.0 * covariance + c2)
        / (
            (left_mean * left_mean + right_mean * right_mean + c1)
            * (left_var + right_var + c2)
        )
    )
    return float(np.clip(score.mean(), -1.0, 1.0))


def _edge_map(value: np.ndarray) -> np.ndarray:
    x = value[1:, 1:] - value[1:, :-1]
    y = value[1:, 1:] - value[:-1, 1:]
    return np.sqrt(x * x + y * y)


def _highpass_map(value: np.ndarray) -> np.ndarray:
    center = value[1:-1, 1:-1]
    return (
        4.0 * center
        - value[:-2, 1:-1]
        - value[2:, 1:-1]
        - value[1:-1, :-2]
        - value[1:-1, 2:]
    )


def texture_distances(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"patch shape mismatch: {left.shape} != {right.shape}")
    return {
        "ssim": max(0.0, 1.0 - block_ssim(left, right)),
        "edge": float(np.mean(np.abs(_edge_map(left) - _edge_map(right)))),
        "highpass": float(
            np.mean(np.abs(_highpass_map(left) - _highpass_map(right)))
        ),
    }


@dataclass
class TextureFeature:
    gray: np.ndarray
    patches: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path, size: tuple[int, int]) -> "TextureFeature":
        with Image.open(path) as image:
            resized = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            gray = np.asarray(resized, dtype=np.float32)
        return cls(gray=gray)

    def patch(self, region: str, view: str) -> np.ndarray:
        key = (region, view)
        if key in self.patches:
            return self.patches[key]
        split = self.gray.shape[1] // 2
        view_index = 0 if view == "front" else 1
        view_left = 0 if view_index == 0 else split
        view_width = split if view_index == 0 else self.gray.shape[1] - split
        box = REGION_BOXES.get(region, REGION_BOXES["face"])[view_index]
        x0 = view_left + int(round(box[0] * view_width))
        y0 = int(round(box[1] * self.gray.shape[0]))
        x1 = view_left + int(round(box[2] * view_width))
        y1 = int(round(box[3] * self.gray.shape[0]))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"texture crop is too small: {region}/{view}")
        patch = _normalize_patch(self.gray[y0:y1, x0:x1])
        self.patches[key] = patch
        return patch


def _load_feature(
    experiment_dir: Path, row: dict[str, Any], size: tuple[int, int]
) -> TextureFeature:
    render_path = experiment_dir / str(row["render_path"])
    if not render_path.is_file():
        raise FileNotFoundError(render_path)
    return TextureFeature.from_path(render_path, size)


def _noise_for_baselines(
    features: Sequence[TextureFeature], region: str, view: str
) -> dict[str, float]:
    distances: dict[str, list[float]] = {name: [] for name in METRICS}
    for left, right in itertools.combinations(features, 2):
        measured = texture_distances(left.patch(region, view), right.patch(region, view))
        for name in METRICS:
            distances[name].append(measured[name])
    return {
        name: max(_percentile(distances[name], 95), NOISE_FLOORS[name])
        for name in METRICS
    }


def _texture_tier(detection_rate: float, texture_snr: float) -> tuple[str, str]:
    if detection_rate >= 0.80 and texture_snr >= TEXTURE_DETECTION_SNR:
        return "T1", "texture_auxiliary_loss"
    if detection_rate >= 0.50:
        return "T2", "conditioned_region_texture_head"
    if detection_rate >= 0.20:
        return "T3", "low_weight_texture_regularizer"
    return "T4", "no_texture_specific_loss"


def _aggregate_rows(
    rows: Sequence[dict[str, Any]], key_columns: Sequence[str]
) -> dict[str, Any]:
    first = rows[0]
    result = {column: first[column] for column in key_columns}
    for view in ("front", "side"):
        for metric in METRICS:
            result[f"{view}_{metric}_distance"] = _median(
                float(row[f"{view}_{metric}_255"]) for row in rows
            )
            result[f"{view}_{metric}_snr"] = _median(
                float(row[f"{view}_{metric}_snr_255"]) for row in rows
            )
        result[f"{view}_texture_snr"] = _median(
            float(row[f"{view}_texture_snr_255"]) for row in rows
        )
    result["texture_snr"] = _median(float(row["texture_snr_255"]) for row in rows)
    result["texture_detection_rate"] = float(
        np.mean(
            [
                float(row["texture_snr_255"]) >= TEXTURE_DETECTION_SNR
                for row in rows
            ]
        )
    )
    result["texture_detection_rate_128"] = float(
        np.mean(
            [
                float(row["texture_snr_128"]) >= TEXTURE_DETECTION_SNR
                for row in rows
            ]
        )
    )
    result["monotonicity"] = _median(float(row["monotonicity"]) for row in rows)
    result["base_count"] = len({str(row["base_id"]) for row in rows})
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    result["worst_base_texture_snr"] = min(
        _median(float(row["texture_snr_255"]) for row in base_rows)
        for base_rows in by_base.values()
    )
    tier, strategy = _texture_tier(
        float(result["texture_detection_rate"]), float(result["texture_snr"])
    )
    result["texture_tier"] = tier
    result["recommended_texture_strategy"] = strategy
    result["recommended_texture_weight"] = min(
        1.0,
        float(result["texture_detection_rate"])
        * math.sqrt(max(0.0, float(result["texture_snr"])) / 3.0),
    )
    result["status"] = "provisional_fixed_regions_no_frozen_model"
    return result


def _colorize_heatmap(value: np.ndarray) -> Image.Image:
    scale = _percentile(value.ravel(), 99)
    if scale <= 1e-9:
        normalized = np.zeros(value.shape, dtype=np.float32)
    else:
        normalized = np.clip(value / scale, 0.0, 1.0)
    red = np.clip(3.0 * normalized, 0.0, 1.0)
    green = np.clip(3.0 * normalized - 1.0, 0.0, 1.0)
    blue = np.clip(1.5 - 3.0 * np.abs(normalized - 0.25), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.asarray(rgb * 255.0, dtype=np.uint8), mode="RGB")


def _save_heatmaps(
    output_dir: Path,
    heat_sums: dict[tuple[str, str], np.ndarray],
    heat_counts: dict[tuple[str, str], int],
) -> dict[str, str]:
    heatmap_dir = output_dir / "texture_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    fields = sorted({field for field, _view in heat_sums})
    for field_name in fields:
        panels: list[Image.Image] = []
        for view in ("front", "side"):
            key = (field_name, view)
            mean_diff = heat_sums[key] / max(1, heat_counts[key])
            panel = _colorize_heatmap(mean_diff).resize(
                (256, 192), Image.Resampling.BILINEAR
            )
            panels.append(panel)
        combined = Image.new("RGB", (520, 192), color=(255, 255, 255))
        combined.paste(panels[0], (0, 0))
        combined.paste(panels[1], (264, 0))
        relative = Path("texture_heatmaps") / f"{field_name}.png"
        combined.save(output_dir / relative)
        index[field_name] = relative.as_posix()
    _write_json(output_dir / "texture_heatmap_index.json", index)
    return index


FIELD_COLUMNS = [
    "field",
    "type",
    "local_region",
    "class_count",
    "base_count",
    "front_ssim_distance",
    "front_ssim_snr",
    "front_edge_distance",
    "front_edge_snr",
    "front_highpass_distance",
    "front_highpass_snr",
    "front_texture_snr",
    "side_ssim_distance",
    "side_ssim_snr",
    "side_edge_distance",
    "side_edge_snr",
    "side_highpass_distance",
    "side_highpass_snr",
    "side_texture_snr",
    "texture_snr",
    "texture_detection_rate",
    "texture_detection_rate_128",
    "monotonicity",
    "worst_base_texture_snr",
    "texture_tier",
    "recommended_texture_strategy",
    "recommended_texture_weight",
    "status",
]


CLASS_COLUMNS = [
    "field",
    "type",
    "class_or_sign",
    "local_region",
    "base_count",
    *FIELD_COLUMNS[5:],
]


def analyze_local_texture(
    experiment_dir: Path,
    *,
    output_dir: Path | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    save_heatmaps: bool = True,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    output_dir = (output_dir or experiment_dir).resolve()
    protocol = _read_json(experiment_dir / "protocol.json")
    manifest = _read_jsonl(experiment_dir / "render_manifest.jsonl")
    completed = [row for row in manifest if row.get("status") == "completed"]
    expected = int(protocol["total_variants"])
    if len(completed) != expected:
        raise ValueError(f"completed render count mismatch: {len(completed)} != {expected}")

    baselines_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fields_by_base: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        base_id = str(row["base_id"])
        if row["kind"] == "baseline":
            baselines_by_base[base_id].append(row)
        elif row["kind"] == "field":
            fields_by_base[(base_id, str(row["field"]))].append(row)

    fields = sorted({field_name for _base_id, field_name in fields_by_base})
    regions = sorted({local_region(field_name) for field_name in fields})
    noise: dict[tuple[str, str, str], dict[str, float]] = {}
    noise_rows: list[dict[str, Any]] = []
    for base_id in sorted(baselines_by_base):
        baseline_rows = sorted(
            baselines_by_base[base_id], key=lambda row: int(row["baseline_repeat"])
        )
        features = [
            _load_feature(experiment_dir, row, size) for row in baseline_rows
        ]
        bit_exact = len({str(row["render_sha256"]) for row in baseline_rows}) == 1
        for region in regions:
            for view in ("front", "side"):
                measured = _noise_for_baselines(features, region, view)
                noise[(base_id, region, view)] = measured
                noise_rows.append(
                    {
                        "base_id": base_id,
                        "region": region,
                        "view": view,
                        "ssim_noise_p95": measured["ssim"],
                        "edge_noise_p95": measured["edge"],
                        "highpass_noise_p95": measured["highpass"],
                        "baseline_pair_count": len(features) * (len(features) - 1) // 2,
                        "bit_exact_baseline": bit_exact,
                    }
                )

    sample_rows: list[dict[str, Any]] = []
    heat_sums: dict[tuple[str, str], np.ndarray] = {}
    heat_counts: dict[tuple[str, str], int] = defaultdict(int)
    grouped_items = sorted(fields_by_base.items())
    for item_index, ((base_id, field_name), variant_rows) in enumerate(grouped_items, 1):
        region = local_region(field_name)
        loaded: dict[tuple[str, int], TextureFeature] = {}
        row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in variant_rows:
            key = (str(row["class_or_sign"]), int(row["strength"]))
            loaded[key] = _load_feature(experiment_dir, row, size)
            row_by_key[key] = row
        classes = sorted({key[0] for key in loaded})
        for class_or_sign in classes:
            missing = [
                strength
                for strength in STRENGTHS
                if (class_or_sign, strength) not in loaded
            ]
            if missing:
                raise ValueError(
                    f"missing strengths for {base_id}/{field_name}/{class_or_sign}: {missing}"
                )
            zero = loaded[(class_or_sign, 0)]
            sample: dict[str, Any] = {
                "base_id": base_id,
                "field": field_name,
                "type": str(row_by_key[(class_or_sign, 0)]["field_type"]),
                "class_or_sign": class_or_sign,
                "local_region": region,
            }
            for view in ("front", "side"):
                zero_patch = zero.patch(region, view)
                for strength in (128, 255):
                    target_patch = loaded[(class_or_sign, strength)].patch(region, view)
                    measured = texture_distances(zero_patch, target_patch)
                    normalized: list[float] = []
                    for metric in METRICS:
                        sample[f"{view}_{metric}_{strength}"] = measured[metric]
                        snr = measured[metric] / noise[(base_id, region, view)][metric]
                        sample[f"{view}_{metric}_snr_{strength}"] = snr
                        normalized.append(snr)
                    sample[f"{view}_texture_snr_{strength}"] = _median(normalized)
                    if strength == 255:
                        key = (field_name, view)
                        difference = np.abs(zero_patch - target_patch)
                        if key not in heat_sums:
                            heat_sums[key] = np.zeros_like(difference, dtype=np.float64)
                        heat_sums[key] += difference
                        heat_counts[key] += 1
            for strength in (128, 255):
                sample[f"texture_snr_{strength}"] = max(
                    float(sample[f"front_texture_snr_{strength}"]),
                    float(sample[f"side_texture_snr_{strength}"]),
                )
            sample["monotonicity"] = _spearman(
                [0.0, float(sample["texture_snr_128"]), float(sample["texture_snr_255"])]
            )
            sample_rows.append(sample)
        if item_index % 100 == 0:
            print(f"texture analysis: {item_index}/{len(grouped_items)} base-field groups")

    class_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    field_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        class_groups[(str(row["field"]), str(row["class_or_sign"]))].append(row)
        field_groups[str(row["field"])].append(row)

    class_rows = [
        _aggregate_rows(rows, ("field", "type", "class_or_sign", "local_region"))
        for _key, rows in sorted(class_groups.items())
    ]
    field_rows: list[dict[str, Any]] = []
    for field_name, rows in sorted(field_groups.items()):
        result = _aggregate_rows(rows, ("field", "type", "local_region"))
        result["class_count"] = len({str(row["class_or_sign"]) for row in rows})
        field_rows.append(result)

    _write_csv(output_dir / "local_texture_identifiability.csv", field_rows, FIELD_COLUMNS)
    _write_csv(
        output_dir / "local_texture_class_metrics.csv", class_rows, CLASS_COLUMNS
    )
    _write_csv(
        output_dir / "local_texture_noise.csv",
        noise_rows,
        (
            "base_id",
            "region",
            "view",
            "ssim_noise_p95",
            "edge_noise_p95",
            "highpass_noise_p95",
            "baseline_pair_count",
            "bit_exact_baseline",
        ),
    )
    heatmap_index = (
        _save_heatmaps(output_dir, heat_sums, heat_counts) if save_heatmaps else {}
    )
    tier_counts = {
        tier: sum(row["texture_tier"] == tier for row in field_rows)
        for tier in ("T1", "T2", "T3", "T4")
    }
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_dir": str(experiment_dir),
        "render_count": len(completed),
        "base_count": len(baselines_by_base),
        "field_count": len(field_rows),
        "class_or_sign_count": len(class_rows),
        "feature_size": list(size),
        "metrics": list(METRICS),
        "detection_snr": TEXTURE_DETECTION_SNR,
        "tier_counts": tier_counts,
        "heatmap_count": len(heatmap_index),
        "limitations": [
            "fixed_regions_are_not_landmark_aligned",
            "no_frozen_learned_visual_features_available",
            "local_geometry_edges_can_contribute_to_texture_distance",
            "per_crop_standardization_suppresses_global_exposure_drift",
        ],
        "outputs": {
            "fields": "local_texture_identifiability.csv",
            "classes": "local_texture_class_metrics.csv",
            "noise": "local_texture_noise.csv",
            "heatmaps": "texture_heatmap_index.json" if save_heatmaps else None,
        },
    }
    _write_json(output_dir / "local_texture_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--no-heatmaps", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.width < 64 or args.height < 64:
        raise ValueError("texture analysis width/height must be at least 64")
    summary = analyze_local_texture(
        args.experiment,
        output_dir=args.output_dir,
        size=(args.width, args.height),
        save_heatmaps=not args.no_heatmaps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
