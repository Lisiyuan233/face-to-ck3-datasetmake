#!/usr/bin/env python3
"""Re-render sampled duplicate CK3 DNA groups and compare them with history.

The script has three resumable phases:

* ``prepare`` samples exact, adjacent duplicate DNA groups across numeric blocks;
* ``run`` reuses the field-sweep GUI backend to import, verify, and capture DNA;
* ``analyze`` ranks same-DNA history against the preceding/following DNA groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from analyze_identifiability_experiment import (
    DEFAULT_FEATURE_SIZE,
    RenderFeature,
    feature_distance,
)
from dna_field_sweep_tool import (
    EmergencyStop,
    WindowsAutomationBackend,
    automation_config_from_settings,
    load_user_settings,
)
from dna_normalizer import FACE_FIELDS


PROTOCOL_VERSION = 1
DEFAULT_BLOCK_SIZE = 30_000
DEFAULT_BLOCK_COUNT = 17
REGION_WEIGHTS = {
    "head": 0.22,
    "nose": 0.16,
    "jaw": 0.20,
    "ears": 0.14,
    "front": 0.12,
    "side": 0.16,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            values.append(value)
    return values


def sample_id(number: int) -> str:
    return f"face_{number:04d}"


def resolve_dataset_root(experiment_dir: Path, protocol: dict[str, Any]) -> Path:
    configured = Path(str(protocol["dataset_root"]))
    if not configured.is_absolute():
        configured = experiment_dir.resolve() / configured
    return configured.resolve()


class DatasetLookup:
    def __init__(self, dataset_root: Path) -> None:
        self.root = dataset_root.resolve()
        self._dna_cache: dict[int, bytes] = {}
        self._hash_cache: dict[int, str] = {}

    def dna_path(self, number: int) -> Path:
        return self.root / "dna" / f"{sample_id(number)}.txt"

    def image_path(self, number: int) -> Path:
        return self.root / "face" / f"{sample_id(number)}.png"

    def dna_bytes(self, number: int) -> bytes:
        if number not in self._dna_cache:
            path = self.dna_path(number)
            if not path.is_file():
                raise FileNotFoundError(path)
            self._dna_cache[number] = path.read_bytes()
        return self._dna_cache[number]

    def dna_hash(self, number: int) -> str:
        if number not in self._hash_cache:
            self._hash_cache[number] = sha256_bytes(self.dna_bytes(number))
        return self._hash_cache[number]

    def group_around(
        self, number: int, *, lower: int, upper: int
    ) -> tuple[int, int, str]:
        target = self.dna_hash(number)
        start = number
        while start > lower and self.dna_hash(start - 1) == target:
            start -= 1
        end = number
        while end < upper and self.dna_hash(end + 1) == target:
            end += 1
        return start, end, target

    def group_payload(
        self,
        number: int,
        *,
        lower: int,
        upper: int,
    ) -> dict[str, Any]:
        start, end, digest = self.group_around(number, lower=lower, upper=upper)
        members = list(range(start, end + 1))
        return {
            "dna_sha256": digest,
            "member_ids": [sample_id(value) for value in members],
            "image_paths": [
                self.image_path(value).relative_to(self.root).as_posix()
                for value in members
            ],
        }


def _quotas(total: int, block_count: int) -> list[int]:
    if total < block_count:
        raise ValueError("group_count 必须至少等于 block_count，以保证每块都有样本")
    base, remainder = divmod(total, block_count)
    return [base + (1 if index < remainder else 0) for index in range(block_count)]


def prepare_plan(
    dataset_root: Path,
    experiment_dir: Path,
    *,
    group_count: int = 100,
    repeats: int = 1,
    seed: int = 20260808,
    block_size: int = DEFAULT_BLOCK_SIZE,
    block_count: int = DEFAULT_BLOCK_COUNT,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    if block_size < 2 or block_count < 1:
        raise ValueError("block_size/block_count 无效")
    experiment_dir = experiment_dir.resolve()
    plan_path = experiment_dir / "plan.jsonl"
    protocol_path = experiment_dir / "protocol.json"
    existing = [path for path in (plan_path, protocol_path) if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有验证计划: {existing}")

    lookup = DatasetLookup(dataset_root)
    quotas = _quotas(group_count, block_count)
    rows: list[dict[str, Any]] = []
    selected_ranges: set[tuple[int, int]] = set()
    for block, quota in enumerate(quotas):
        lower = block * block_size + 1
        upper = (block + 1) * block_size
        candidates = list(range(lower, upper))
        random.Random(seed + block * 1009).shuffle(candidates)
        selected = 0
        for number in candidates:
            if lookup.dna_hash(number) != lookup.dna_hash(number + 1):
                continue
            start, end, digest = lookup.group_around(
                number, lower=lower, upper=upper
            )
            key = (start, end)
            if key in selected_ranges:
                continue
            member_paths = [lookup.image_path(value) for value in range(start, end + 1)]
            missing = [path for path in member_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])

            previous = (
                lookup.group_payload(start - 1, lower=lower, upper=upper)
                if start > lower
                else None
            )
            following = (
                lookup.group_payload(end + 1, lower=lower, upper=upper)
                if end < upper
                else None
            )
            group_id = f"b{block:02d}_g{selected + 1:02d}_{sample_id(start)}"
            rows.append(
                {
                    "group_id": group_id,
                    "block": block,
                    "group_start": start,
                    "group_end": end,
                    "dna_sha256": digest,
                    "dna_path": lookup.dna_path(start)
                    .relative_to(lookup.root)
                    .as_posix(),
                    "member_ids": [
                        sample_id(value) for value in range(start, end + 1)
                    ],
                    "image_paths": [
                        path.relative_to(lookup.root).as_posix()
                        for path in member_paths
                    ],
                    "previous_group": previous,
                    "next_group": following,
                }
            )
            selected_ranges.add(key)
            selected += 1
            if selected == quota:
                break
        if selected != quota:
            raise RuntimeError(
                f"block {block} 只找到 {selected}/{quota} 组相邻重复 DNA"
            )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "renders").mkdir(exist_ok=True)
    plan_partial = plan_path.with_suffix(plan_path.suffix + ".partial")
    with plan_partial.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(plan_partial, plan_path)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "dataset_root": Path(
            os.path.relpath(lookup.root, experiment_dir)
        ).as_posix(),
        "group_count": len(rows),
        "repeats": repeats,
        "seed": seed,
        "block_size": block_size,
        "block_count": block_count,
        "block_quotas": quotas,
        "plan_path": "plan.jsonl",
        "plan_sha256": sha256_file(plan_path),
        "render_count": len(rows) * repeats,
        "comparison_scope": "same_dna_vs_previous_and_next_dna_groups",
    }
    protocol_partial = protocol_path.with_suffix(protocol_path.suffix + ".partial")
    protocol_partial.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(protocol_partial, protocol_path)
    return protocol


def load_plan(experiment_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_dir = experiment_dir.resolve()
    protocol = json.loads(
        (experiment_dir / "protocol.json").read_text(encoding="utf-8")
    )
    if int(protocol.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("不支持的重复 DNA 验证协议版本")
    plan_path = experiment_dir / str(protocol["plan_path"])
    if sha256_file(plan_path) != str(protocol.get("plan_sha256", "")):
        raise RuntimeError("plan.jsonl SHA-256 与 protocol.json 不一致")
    rows = read_jsonl(plan_path)
    if len(rows) != int(protocol["group_count"]):
        raise RuntimeError("计划组数与 protocol.json 不一致")
    return protocol, rows


def _completed_render_ids(experiment_dir: Path, plan_sha256: str) -> set[str]:
    path = experiment_dir / "render_manifest.jsonl"
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        if row.get("status") != "completed":
            continue
        if row.get("protocol_plan_sha256") != plan_sha256:
            continue
        render_path = experiment_dir / str(row.get("render_path", ""))
        if render_path.is_file() and sha256_file(render_path) == row.get(
            "render_sha256"
        ):
            completed.add(str(row["render_id"]))
    return completed


def run_validation(
    experiment_dir: Path,
    backend: Any,
    *,
    limit: int | None = None,
    retries: int = 1,
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, int]:
    experiment_dir = experiment_dir.resolve()
    protocol, groups = load_plan(experiment_dir)
    dataset_root = resolve_dataset_root(experiment_dir, protocol)
    repeats = int(protocol["repeats"])
    variants = [
        (group, repeat)
        for group in groups
        for repeat in range(1, repeats + 1)
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        variants = variants[:limit]
    completed_ids = _completed_render_ids(
        experiment_dir, str(protocol["plan_sha256"])
    )
    completed = skipped = attempted = 0
    for group, repeat in variants:
        render_id = f"{group['group_id']}_r{repeat:02d}"
        if render_id in completed_ids:
            completed += 1
            skipped += 1
            if on_progress:
                on_progress(completed, len(variants), render_id, "skipped")
            continue
        attempted += 1
        dna_path = dataset_root / str(group["dna_path"])
        dna_bytes = dna_path.read_bytes()
        if sha256_bytes(dna_bytes) != group["dna_sha256"]:
            raise RuntimeError(f"计划 DNA 已变化: {dna_path}")
        dna_text = dna_bytes.decode("utf-8")
        render_path = experiment_dir / "renders" / f"{render_id}.png"
        started_at = utc_now()
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                backend.apply_dna(dna_text, FACE_FIELDS)
                backend.capture(render_path)
                last_error = None
                break
            except EmergencyStop:
                raise
            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.5)
        if last_error is not None:
            append_jsonl(
                experiment_dir / "errors.jsonl",
                {
                    "protocol_plan_sha256": protocol["plan_sha256"],
                    "render_id": render_id,
                    "group_id": group["group_id"],
                    "status": "failed",
                    "started_at": started_at,
                    "failed_at": utc_now(),
                    "attempts": retries + 1,
                    "error": f"{type(last_error).__name__}: {last_error}",
                },
            )
            if on_progress:
                on_progress(completed, len(variants), render_id, "failed")
            raise RuntimeError(
                f"{render_id} 连续失败 {retries + 1} 次，已停止: {last_error}"
            ) from last_error
        render_hash = sha256_file(render_path)
        append_jsonl(
            experiment_dir / "render_manifest.jsonl",
            {
                "protocol_plan_sha256": protocol["plan_sha256"],
                "render_id": render_id,
                "group_id": group["group_id"],
                "repeat": repeat,
                "block": group["block"],
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "dna_path": group["dna_path"],
                "dna_sha256": group["dna_sha256"],
                "render_path": render_path.relative_to(experiment_dir).as_posix(),
                "render_sha256": render_hash,
                "verification_field_count": len(FACE_FIELDS),
            },
        )
        completed += 1
        if on_progress:
            on_progress(completed, len(variants), render_id, "completed")
    return {"completed": completed, "skipped": skipped, "attempted": attempted}


def _candidate_rows(group: dict[str, Any]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for relation, payload in (
        ("same", group),
        ("previous", group.get("previous_group")),
        ("next", group.get("next_group")),
    ):
        if not payload:
            continue
        for member_id, path in zip(payload["member_ids"], payload["image_paths"]):
            values.append(
                {"relation": relation, "sample_id": member_id, "image_path": path}
            )
    return values


class FeatureCache:
    def __init__(self, feature_size: tuple[int, int]) -> None:
        self.feature_size = feature_size
        self.features: dict[Path, RenderFeature] = {}
        self.vectors: dict[tuple[Path, str], np.ndarray] = {}

    def vector(self, path: Path, region: str) -> np.ndarray:
        key = (path.resolve(), region)
        if key not in self.vectors:
            if key[0] not in self.features:
                self.features[key[0]] = RenderFeature.from_path(
                    key[0], self.feature_size
                )
            self.vectors[key] = self.features[key[0]].vector(region)
        return self.vectors[key]

    def distance(self, left: Path, right: Path) -> tuple[float, dict[str, float]]:
        regions = {
            region: feature_distance(
                self.vector(left, region), self.vector(right, region)
            )
            for region in REGION_WEIGHTS
        }
        combined = sum(REGION_WEIGHTS[key] * value for key, value in regions.items())
        return combined, regions


def _classification(relation: str, relative_margin: float) -> str:
    if relation == "same":
        return "aligned" if relative_margin >= 0.05 else "aligned_weak"
    if relative_margin > -0.05:
        return "ambiguous"
    if relation == "previous":
        return "suspected_previous_dna_lag"
    if relation == "next":
        return "suspected_next_image_shift"
    return "ambiguous"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _contact_sheet(
    destination: Path,
    render_path: Path,
    group: dict[str, Any],
    dataset_root: Path,
) -> None:
    candidates = _candidate_rows(group)
    items = [("rerender", render_path)] + [
        (f"{item['relation']}:{item['sample_id']}", dataset_root / item["image_path"])
        for item in candidates
    ]
    width, height = 300, 202
    canvas = Image.new("RGB", (width * len(items), height + 26), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(items):
        with Image.open(path) as image:
            value = image.convert("RGB")
            value.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = index * width + (width - value.width) // 2
            canvas.paste(value, (x, 26))
        draw.text((index * width + 4, 5), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90)


def analyze_validation(
    experiment_dir: Path,
    *,
    feature_size: tuple[int, int] = DEFAULT_FEATURE_SIZE,
    review_count: int = 20,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    protocol, groups = load_plan(experiment_dir)
    dataset_root = resolve_dataset_root(experiment_dir, protocol)
    manifest_path = experiment_dir / "render_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("尚无 render_manifest.jsonl，请先运行 run")
    completed = {
        str(row["render_id"]): row
        for row in read_jsonl(manifest_path)
        if row.get("status") == "completed"
        and row.get("protocol_plan_sha256") == protocol["plan_sha256"]
    }
    expected = [
        (group, repeat, f"{group['group_id']}_r{repeat:02d}")
        for group in groups
        for repeat in range(1, int(protocol["repeats"]) + 1)
    ]
    missing = [render_id for _group, _repeat, render_id in expected if render_id not in completed]
    if missing:
        raise RuntimeError(f"缺少 {len(missing)} 个重渲染，首个: {missing[0]}")

    cache = FeatureCache(feature_size)
    render_rows: list[dict[str, Any]] = []
    group_map = {str(group["group_id"]): group for group in groups}
    for group, repeat, render_id in expected:
        render_path = experiment_dir / str(completed[render_id]["render_path"])
        distances: list[dict[str, Any]] = []
        for candidate in _candidate_rows(group):
            candidate_path = dataset_root / candidate["image_path"]
            combined, regions = cache.distance(render_path, candidate_path)
            distances.append(
                {
                    **candidate,
                    "distance": combined,
                    **{f"{key}_distance": value for key, value in regions.items()},
                }
            )
        distances.sort(key=lambda value: float(value["distance"]))
        same = [value for value in distances if value["relation"] == "same"]
        negative = [value for value in distances if value["relation"] != "same"]
        if not same or not negative:
            raise RuntimeError(f"{group['group_id']} 缺少正样本或邻组负样本")
        best = distances[0]
        positive = min(same, key=lambda value: float(value["distance"]))
        negative_best = min(negative, key=lambda value: float(value["distance"]))
        positive_distance = float(positive["distance"])
        negative_distance = float(negative_best["distance"])
        relative_margin = (negative_distance - positive_distance) / max(
            negative_distance, 1e-9
        )
        classification = _classification(str(best["relation"]), relative_margin)
        render_rows.append(
            {
                "group_id": group["group_id"],
                "block": group["block"],
                "repeat": repeat,
                "render_id": render_id,
                "member_ids": ";".join(group["member_ids"]),
                "nearest_relation": best["relation"],
                "nearest_sample_id": best["sample_id"],
                "nearest_distance": best["distance"],
                "positive_best_id": positive["sample_id"],
                "positive_distance": positive_distance,
                "negative_best_relation": negative_best["relation"],
                "negative_best_id": negative_best["sample_id"],
                "negative_distance": negative_distance,
                "relative_margin": relative_margin,
                "classification": classification,
                **{
                    f"positive_{region}_distance": positive[f"{region}_distance"]
                    for region in REGION_WEIGHTS
                },
            }
        )

    _write_csv(experiment_dir / "render_comparison.csv", render_rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in render_rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    group_rows: list[dict[str, Any]] = []
    for group_id, rows in grouped.items():
        class_counts = Counter(str(row["classification"]) for row in rows)
        nearest_counts = Counter(str(row["nearest_relation"]) for row in rows)
        group_rows.append(
            {
                "group_id": group_id,
                "block": rows[0]["block"],
                "repeat_count": len(rows),
                "same_top1_count": nearest_counts.get("same", 0),
                "previous_top1_count": nearest_counts.get("previous", 0),
                "next_top1_count": nearest_counts.get("next", 0),
                "median_relative_margin": median(
                    float(row["relative_margin"]) for row in rows
                ),
                "final_classification": class_counts.most_common(1)[0][0],
            }
        )
    group_rows.sort(key=lambda row: (int(row["block"]), str(row["group_id"])))
    _write_csv(experiment_dir / "group_comparison.csv", group_rows)

    review_rows = sorted(
        render_rows, key=lambda row: float(row["relative_margin"])
    )[: max(0, review_count)]
    for row in review_rows:
        group = group_map[str(row["group_id"])]
        render_path = experiment_dir / str(completed[str(row["render_id"])]["render_path"])
        _contact_sheet(
            experiment_dir / "review" / f"{row['render_id']}.jpg",
            render_path,
            group,
            dataset_root,
        )

    classifications = Counter(str(row["final_classification"]) for row in group_rows)
    same_top1 = sum(int(row["same_top1_count"]) for row in group_rows)
    total_renders = len(render_rows)
    summary = {
        "analysis_version": 1,
        "created_at": utc_now(),
        "group_count": len(group_rows),
        "render_count": total_renders,
        "same_dna_top1_rate": same_top1 / max(1, total_renders),
        "median_relative_margin": median(
            float(row["relative_margin"]) for row in render_rows
        ),
        "classification_counts": dict(sorted(classifications.items())),
        "interpretation": {
            "aligned": "重渲染明显更接近同 DNA 历史截图",
            "aligned_weak": "同 DNA 排名第一，但与邻组间隔较小",
            "suspected_previous_dna_lag": "重渲染更接近前一 DNA 组，疑似复制滞后",
            "suspected_next_image_shift": "重渲染更接近后一 DNA 组，疑似截图/编号前移",
            "ambiguous": "动画或族群内相似度过高，自动指标不能定论",
        },
        "outputs": {
            "renders": "renders/",
            "render_comparison": "render_comparison.csv",
            "group_comparison": "group_comparison.csv",
            "review": "review/",
        },
    }
    path = experiment_dir / "analysis_summary.json"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="抽取重复 DNA 并生成验证计划")
    prepare.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        default=Path("experiments/duplicate_dna_validation"),
    )
    prepare.add_argument(
        "--dataset", type=Path, default=Path("face_to_ck3_dataset_male_small")
    )
    prepare.add_argument("--groups", type=int, default=100)
    prepare.add_argument("--repeats", type=int, default=1)
    prepare.add_argument("--seed", type=int, default=20260808)

    run = subparsers.add_parser("run", help="导入计划 DNA 并在 CK3 中重渲染")
    run.add_argument("experiment", type=Path)
    run.add_argument("--settings", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--progress-every", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")

    analyze = subparsers.add_parser("analyze", help="与历史截图和前后邻组比较")
    analyze.add_argument("experiment", type=Path)
    analyze.add_argument("--feature-width", type=int, default=DEFAULT_FEATURE_SIZE[0])
    analyze.add_argument("--feature-height", type=int, default=DEFAULT_FEATURE_SIZE[1])
    analyze.add_argument("--review-count", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        protocol = prepare_plan(
            args.dataset,
            args.experiment,
            group_count=args.groups,
            repeats=args.repeats,
            seed=args.seed,
        )
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        protocol, groups = load_plan(args.experiment)
        total = len(groups) * int(protocol["repeats"])
        if args.limit is not None:
            total = min(total, args.limit)
        if args.dry_run:
            print(
                f"计划校验通过: groups={len(groups)}, renders={total}, "
                f"plan={protocol['plan_sha256']}"
            )
            return 0
        settings = (
            json.loads(args.settings.read_text(encoding="utf-8"))
            if args.settings
            else load_user_settings()
        )
        config = automation_config_from_settings(settings)
        if config.verify_copy_button is None:
            raise ValueError("必须先在字段扫描 GUI 中记录复制 DNA 验证按钮")
        backend = WindowsAutomationBackend(config)
        progress_every = max(1, int(args.progress_every))

        def progress(done: int, total: int, render_id: str, status: str) -> None:
            if status == "failed" or done % progress_every == 0 or done == total:
                print(f"[{done}/{total}] {render_id} [{status}]", flush=True)

        result = run_validation(
            args.experiment,
            backend,
            limit=args.limit,
            retries=config.retries,
            on_progress=progress,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.feature_width < 32 or args.feature_height < 32:
        raise ValueError("feature width/height 不能小于 32")
    summary = analyze_validation(
        args.experiment,
        feature_size=(args.feature_width, args.feature_height),
        review_count=args.review_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
