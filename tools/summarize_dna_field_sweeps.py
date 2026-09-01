#!/usr/bin/env python3
"""Merge CK3 DNA field sweep batches into canonical result summaries.

The source experiment tree is read-only. Duplicate/restarted sessions are
merged by (field, allele, value), preferring the latest completed record whose
DNA and render files still exist. Reports are written to a separate summary
directory under the experiment root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageStat


VariantKey = tuple[str, int]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if isinstance(value, dict):
            result.append(value)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    # Session DNA hashes are calculated from the in-memory text. Reading with
    # universal-newline translation restores LF on Windows before hashing.
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_timestamp(record: dict[str, Any], fallback: float) -> float:
    for key in ("completed_at", "failed_at", "started_at"):
        text = record.get(key)
        if isinstance(text, str):
            try:
                return datetime.fromisoformat(text).timestamp()
            except ValueError:
                pass
    return fallback


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def variant_key(value: dict[str, Any]) -> VariantKey:
    return str(value["allele"]), int(value["value"])


@dataclass(frozen=True)
class SessionData:
    path: Path
    relative_path: str
    field: str
    session: dict[str, Any]
    variants: tuple[dict[str, Any], ...]
    completed: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]
    verified: bool
    modified_at: float


def discover_sessions(root: Path) -> list[SessionData]:
    sessions: list[SessionData] = []
    for session_path in sorted(root.glob("*/session.json")):
        session_dir = session_path.parent
        session = read_json(session_path)
        field = session.get("field")
        if not isinstance(field, str) or not field:
            continue
        variants = tuple(
            value
            for value in session.get("variants", [])
            if isinstance(value, dict)
        )
        automation = session.get("automation")
        verified = bool(
            isinstance(automation, dict)
            and automation.get("verify_copy_button") is not None
        )
        sessions.append(
            SessionData(
                path=session_dir,
                relative_path=session_dir.relative_to(root).as_posix(),
                field=field,
                session=session,
                variants=variants,
                completed=tuple(read_jsonl(session_dir / "manifest.jsonl")),
                errors=tuple(read_jsonl(session_dir / "errors.jsonl")),
                verified=verified,
                modified_at=max(
                    (path.stat().st_mtime for path in session_dir.glob("*.json*")),
                    default=session_path.stat().st_mtime,
                ),
            )
        )
    return sessions


def discover_master_field_order(root: Path) -> tuple[list[str], str | None]:
    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
    for path in root.glob("*_batch.json"):
        batch = read_json(path)
        fields = batch.get("fields")
        if not isinstance(fields, list):
            continue
        candidates.append((len(fields), path.stat().st_mtime, path, batch))
    if not candidates:
        return [], None
    _count, _mtime, path, batch = max(candidates, key=lambda item: (item[0], item[1]))
    order = [
        str(item["field"])
        for item in batch["fields"]
        if isinstance(item, dict) and item.get("field")
    ]
    return order, path.name


def image_mae_percent(first: Path, second: Path) -> dict[str, float]:
    with Image.open(first) as left_image, Image.open(second) as right_image:
        left = left_image.convert("RGB")
        right = right_image.convert("RGB")
        if left.size != right.size:
            raise ValueError(
                f"render size mismatch: {first}={left.size}, {second}={right.size}"
            )
        width, height = left.size

        def metric(box: tuple[int, int, int, int]) -> float:
            difference = ImageChops.difference(left.crop(box), right.crop(box))
            means = ImageStat.Stat(difference).mean
            return round(sum(means) / len(means) / 255.0 * 100.0, 6)

        split = width // 2
        return {
            "whole_percent": metric((0, 0, width, height)),
            "front_half_percent": metric((0, 0, split, height)),
            "side_half_percent": metric((split, 0, width, height)),
        }


def choose_latest(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if current is None or candidate["_sort_time"] > current["_sort_time"]:
        return candidate
    return current


def field_status(completed: int, expected: int, errors: int) -> str:
    if expected > 0 and completed == expected:
        return "complete"
    if completed > 0:
        return "partial"
    if errors > 0:
        return "failed"
    return "missing"


def visual_status(completed: int, unique_hashes: int) -> str:
    if completed == 0:
        return "unavailable"
    if completed == 1:
        return "insufficient_variants"
    if unique_hashes == 1:
        return "no_pixel_change"
    if unique_hashes < completed:
        return "quantized_or_saturated"
    return "all_steps_distinct"


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}%"


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    sessions = discover_sessions(root)
    if not sessions:
        raise ValueError(f"no sweep sessions found under {root}")

    master_order, master_batch = discover_master_field_order(root)
    sessions_by_field: dict[str, list[SessionData]] = {}
    for session in sessions:
        sessions_by_field.setdefault(session.field, []).append(session)
    field_order = list(master_order)
    field_order.extend(
        field
        for field in sessions_by_field
        if field not in set(field_order)
    )

    variant_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for order_index, field in enumerate(field_order, 1):
        field_sessions = sessions_by_field.get(field, [])
        expected: dict[VariantKey, dict[str, Any]] = {}
        completed: dict[VariantKey, dict[str, Any]] = {}
        errors: list[dict[str, Any]] = []
        allele_order: list[str] = []

        for session in sorted(field_sessions, key=lambda value: value.modified_at):
            for variant in session.variants:
                key = variant_key(variant)
                if key[0] not in allele_order:
                    allele_order.append(key[0])
                expected.setdefault(key, dict(variant))
            for record in session.completed:
                if record.get("status") != "completed":
                    continue
                key = variant_key(record)
                render_relative = str(record.get("render_path", ""))
                dna_relative = str(record.get("dna_path", ""))
                render_path = session.path / render_relative
                dna_path = session.path / dna_relative
                if not render_path.is_file() or not dna_path.is_file():
                    continue
                candidate = dict(record)
                candidate.update(
                    {
                        "_session": session,
                        "_render_path": render_path,
                        "_dna_path": dna_path,
                        "_sort_time": record_timestamp(record, session.modified_at),
                    }
                )
                completed[key] = choose_latest(completed.get(key), candidate)
            for error in session.errors:
                item = dict(error)
                item["session_path"] = session.relative_path
                item["_sort_time"] = record_timestamp(error, session.modified_at)
                errors.append(item)

        sorted_keys = sorted(
            expected,
            key=lambda key: (
                allele_order.index(key[0]) if key[0] in allele_order else 9999,
                key[1],
            ),
        )
        hashes: list[str] = []
        verified_count = 0
        integrity_errors: list[str] = []
        endpoint_metrics: list[dict[str, Any]] = []

        for key in sorted_keys:
            expected_variant = expected[key]
            record = completed.get(key)
            row: dict[str, Any] = {
                "field_order": order_index,
                "field": field,
                "allele": key[0],
                "value": key[1],
                "expected_variant_id": expected_variant.get("variant_id"),
                "status": "missing",
            }
            if record is not None:
                session = record["_session"]
                render_path = record["_render_path"]
                dna_path = record["_dna_path"]
                actual_render_sha256 = sha256_file(render_path)
                recorded_render_sha256 = record.get("render_sha256")
                render_hash_ok = (
                    not recorded_render_sha256
                    or actual_render_sha256 == recorded_render_sha256
                )
                actual_dna_sha256 = sha256_text_file(dna_path)
                expected_dna_sha256 = expected_variant.get("dna_sha256")
                dna_hash_ok = (
                    not expected_dna_sha256
                    or actual_dna_sha256 == expected_dna_sha256
                )
                if not render_hash_ok:
                    integrity_errors.append(
                        f"{key[0]}={key[1]} render SHA-256 mismatch"
                    )
                if not dna_hash_ok:
                    integrity_errors.append(
                        f"{key[0]}={key[1]} DNA SHA-256 mismatch"
                    )
                hashes.append(actual_render_sha256)
                verified_count += int(session.verified)
                row.update(
                    {
                        "status": "completed",
                        "variant_id": record.get("variant_id"),
                        "session_path": session.relative_path,
                        "dna_path": dna_path.relative_to(root).as_posix(),
                        "render_path": render_path.relative_to(root).as_posix(),
                        "dna_sha256": actual_dna_sha256,
                        "render_sha256": actual_render_sha256,
                        "dna_hash_ok": dna_hash_ok,
                        "render_hash_ok": render_hash_ok,
                        "round_trip_verified": session.verified,
                        "completed_at": record.get("completed_at"),
                    }
                )
            variant_rows.append(row)

        for allele in allele_order:
            allele_records = [
                (key[1], completed[key])
                for key in sorted_keys
                if key[0] == allele and key in completed
            ]
            allele_records.sort(key=lambda item: item[0])
            if len(allele_records) < 2:
                continue
            low_value, low_record = allele_records[0]
            high_value, high_record = allele_records[-1]
            metric = image_mae_percent(
                low_record["_render_path"],
                high_record["_render_path"],
            )
            endpoint_metrics.append(
                {
                    "allele": allele,
                    "low_value": low_value,
                    "high_value": high_value,
                    **metric,
                }
            )

        unique_hashes = len(set(hashes))
        completed_count = len(completed)
        expected_count = len(expected)
        latest_error = max(errors, key=lambda item: item["_sort_time"]) if errors else None
        strongest_metric = (
            max(endpoint_metrics, key=lambda item: item["whole_percent"])
            if endpoint_metrics
            else None
        )
        preferred_session = (
            max(
                field_sessions,
                key=lambda session: (len(session.completed), session.modified_at),
            ).relative_path
            if field_sessions
            else None
        )
        field_rows.append(
            {
                "order": order_index,
                "field": field,
                "status": field_status(completed_count, expected_count, len(errors)),
                "visual_status": visual_status(completed_count, unique_hashes),
                "expected_variants": expected_count,
                "completed_variants": completed_count,
                "missing_variants": max(0, expected_count - completed_count),
                "verified_variants": verified_count,
                "unique_render_hashes": unique_hashes,
                "alleles": allele_order,
                "endpoint_metrics": endpoint_metrics,
                "strongest_endpoint_metric": strongest_metric,
                "integrity_ok": not integrity_errors,
                "integrity_errors": integrity_errors,
                "error_count": len(errors),
                "latest_error": (
                    {
                        key: value
                        for key, value in latest_error.items()
                        if not key.startswith("_")
                    }
                    if latest_error
                    else None
                ),
                "preferred_session": preferred_session,
                "all_sessions": [session.relative_path for session in field_sessions],
            }
        )

    status_counts = {
        status: sum(1 for row in field_rows if row["status"] == status)
        for status in ("complete", "partial", "failed", "missing")
    }
    visual_counts = {
        status: sum(1 for row in field_rows if row["visual_status"] == status)
        for status in (
            "all_steps_distinct",
            "quantized_or_saturated",
            "no_pixel_change",
            "insufficient_variants",
            "unavailable",
        )
    }
    expected_total = sum(row["expected_variants"] for row in field_rows)
    completed_total = sum(row["completed_variants"] for row in field_rows)
    summary = {
        "generated_at": utc_now(),
        "experiment_root": str(root),
        "master_batch": master_batch,
        "session_count": len(sessions),
        "field_count": len(field_rows),
        "expected_variants": expected_total,
        "completed_variants": completed_total,
        "variant_coverage_percent": round(
            completed_total / expected_total * 100.0 if expected_total else 0.0,
            4,
        ),
        "round_trip_verified_variants": sum(
            row["verified_variants"] for row in field_rows
        ),
        "integrity_error_count": sum(
            len(row["integrity_errors"]) for row in field_rows
        ),
        "field_status_counts": status_counts,
        "visual_status_counts": visual_counts,
        "fields": field_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "variant_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in variant_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    status_labels = {
        "complete": "完整",
        "partial": "部分",
        "failed": "失败",
        "missing": "未执行",
    }
    visual_labels = {
        "all_steps_distinct": "各档不同",
        "quantized_or_saturated": "存在量化/饱和",
        "no_pixel_change": "无像素变化",
        "insufficient_variants": "样本不足",
        "unavailable": "无图像",
    }
    markdown = [
        "# DNA 字段扫描总表",
        "",
        f"生成时间：`{summary['generated_at']}`  ",
        f"主批次：`{master_batch or '-'}`  ",
        f"字段：**{len(field_rows)}**；变体覆盖：**{completed_total}/{expected_total} "
        f"({summary['variant_coverage_percent']:.2f}%)**。",
        "",
        "| # | 字段 | 完成状态 | 变体 | 唯一截图 | 视觉状态 | 端点 Δ 全图 | 正面 | 侧面 |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in field_rows:
        metric = row["strongest_endpoint_metric"]
        whole = metric["whole_percent"] if metric else None
        front = metric["front_half_percent"] if metric else None
        side = metric["side_half_percent"] if metric else None
        field_link = (
            f"[`{row['field']}`](../{row['preferred_session']})"
            if row["preferred_session"]
            else f"`{row['field']}`"
        )
        markdown.append(
            f"| {row['order']} | {field_link} | {status_labels[row['status']]} | "
            f"{row['completed_variants']}/{row['expected_variants']} | "
            f"{row['unique_render_hashes']} | {visual_labels[row['visual_status']]} | "
            f"{format_percent(whole)} | {format_percent(front)} | "
            f"{format_percent(side)} |"
        )
    markdown.extend(
        [
            "",
            "> 端点 Δ 是同一 allele 的最低/最高已完成取值之间的 RGB 平均绝对像素差，",
            "> 除以 255 后以百分比显示。它适合做同一采集条件内的相对排序，不是模型准确率。",
            "",
        ]
    )
    (output_dir / "field_summary.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_root",
        type=Path,
        nargs="?",
        default=Path("experiments/dna_field_sweeps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <experiment_root>/summary",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.experiment_root / "summary"
    summary = summarize(args.experiment_root, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "fields": summary["field_count"],
                "completed_variants": summary["completed_variants"],
                "expected_variants": summary["expected_variants"],
                "coverage_percent": summary["variant_coverage_percent"],
                "field_status_counts": summary["field_status_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
