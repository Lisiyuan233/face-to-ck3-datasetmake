#!/usr/bin/env python3
"""Execute a prepared CK3 DNA identifiability protocol on the Windows desktop."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from dna_field_sweep_tool import (
    EmergencyStop,
    WindowsAutomationBackend,
    append_jsonl,
    automation_config_from_settings,
    load_user_settings,
    sha256_file,
    sha256_text,
    utc_now,
)


class IdentifiabilityBackend(Protocol):
    def apply_dna(self, dna_text: str, field: str | None) -> None: ...

    def capture(self, path: Path) -> None: ...


@dataclass(frozen=True)
class PlannedVariant:
    global_index: int
    variant_id: str
    base_id: str
    race_group: int
    kind: str
    field: str | None
    dna_path: Path
    render_path: Path
    dna_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    completed: int
    skipped: int
    attempted: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
            rows.append(value)
    return rows


def load_plan(experiment_dir: Path, *, verify_dna_hashes: bool = True) -> tuple[dict[str, Any], list[PlannedVariant]]:
    experiment_dir = experiment_dir.resolve()
    protocol = _read_json(experiment_dir / "protocol.json")
    if int(protocol.get("protocol_version", 0)) != 1:
        raise ValueError(
            f"不支持 protocol_version={protocol.get('protocol_version')!r}"
        )
    variants_relative = protocol.get("paths", {}).get("variants", "variants.jsonl")
    rows = _read_jsonl(experiment_dir / str(variants_relative))
    if len(rows) != int(protocol.get("total_variants", -1)):
        raise ValueError(
            f"variants 数量 {len(rows)} 与 protocol 的 "
            f"{protocol.get('total_variants')} 不一致"
        )

    variants = []
    seen_ids: set[str] = set()
    expected_index = 1
    for row in rows:
        try:
            global_index = int(row["global_index"])
            variant_id = str(row["variant_id"])
            base_id = str(row["base_id"])
            race_group = int(row["race_group"])
            kind = str(row["kind"])
            field = str(row["field"]) if row.get("field") is not None else None
            dna_path = experiment_dir / str(row["dna_path"])
            render_path = experiment_dir / str(row["render_path"])
            dna_sha256 = str(row["dna_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"变体记录字段无效: {row!r}") from error
        if global_index != expected_index:
            raise ValueError(
                f"global_index 应为 {expected_index}，实际为 {global_index}"
            )
        expected_index += 1
        if variant_id in seen_ids:
            raise ValueError(f"variant_id 重复: {variant_id}")
        seen_ids.add(variant_id)
        if kind not in {"field", "baseline"}:
            raise ValueError(f"未知变体 kind: {kind}")
        if kind == "field" and not field:
            raise ValueError(f"字段变体缺少 field: {variant_id}")
        if kind == "baseline" and field is not None:
            raise ValueError(f"baseline 不应包含 field: {variant_id}")
        if not dna_path.is_file():
            raise FileNotFoundError(dna_path)
        if verify_dna_hashes:
            actual_dna_sha256 = sha256_text(dna_path.read_text(encoding="utf-8"))
            if actual_dna_sha256 != dna_sha256:
                raise RuntimeError(f"变体 DNA SHA-256 不一致: {dna_path}")
        variants.append(
            PlannedVariant(
                global_index=global_index,
                variant_id=variant_id,
                base_id=base_id,
                race_group=race_group,
                kind=kind,
                field=field,
                dna_path=dna_path,
                render_path=render_path,
                dna_sha256=dna_sha256,
                metadata=dict(row),
            )
        )
    return protocol, variants


def completed_variant_ids(experiment_dir: Path) -> set[str]:
    path = experiment_dir / "render_manifest.jsonl"
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in _read_jsonl(path):
        if row.get("status") != "completed":
            continue
        render_relative = row.get("render_path")
        recorded_sha256 = row.get("render_sha256")
        if not render_relative or not recorded_sha256:
            continue
        render_path = experiment_dir / str(render_relative)
        if render_path.is_file() and sha256_file(render_path) == recorded_sha256:
            completed.add(str(row.get("variant_id", "")))
    return completed


def run_experiment(
    experiment_dir: Path,
    protocol: dict[str, Any],
    variants: Sequence[PlannedVariant],
    backend: IdentifiabilityBackend,
    *,
    retries: int,
    on_progress: Callable[[int, int, PlannedVariant, str], None] | None = None,
) -> RunResult:
    """Run or resume variants, requiring a full parsed-DNA round trip each time."""

    if retries < 0:
        raise ValueError("retries 不能小于 0")
    experiment_dir = experiment_dir.resolve()
    manifest_path = experiment_dir / "render_manifest.jsonl"
    error_path = experiment_dir / "errors.jsonl"
    completed_ids = completed_variant_ids(experiment_dir)
    completed = 0
    skipped = 0
    attempted = 0
    total = len(variants)

    for variant in variants:
        if variant.variant_id in completed_ids:
            completed += 1
            skipped += 1
            if on_progress:
                on_progress(completed, total, variant, "skipped")
            continue
        attempted += 1
        started_at = utc_now()
        last_error: Exception | None = None
        render_sha256: str | None = None
        for attempt in range(retries + 1):
            try:
                dna_text = variant.dna_path.read_text(encoding="utf-8")
                if sha256_text(dna_text) != variant.dna_sha256:
                    raise RuntimeError(f"运行前 DNA SHA-256 不一致: {variant.dna_path}")
                # None deliberately requests full-record verification.  This is
                # stronger than the single-field exploratory sweep and also
                # verifies that interleaved baselines truly restored the base.
                backend.apply_dna(dna_text, None)
                backend.capture(variant.render_path)
                render_sha256 = sha256_file(variant.render_path)
                last_error = None
                break
            except EmergencyStop:
                raise
            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.50)
        if last_error is not None:
            append_jsonl(
                error_path,
                {
                    "protocol_plan_sha256": protocol.get("plan_sha256"),
                    "global_index": variant.global_index,
                    "variant_id": variant.variant_id,
                    "base_id": variant.base_id,
                    "race_group": variant.race_group,
                    "kind": variant.kind,
                    "field": variant.field,
                    "status": "failed",
                    "started_at": started_at,
                    "failed_at": utc_now(),
                    "attempts": retries + 1,
                    "error": f"{type(last_error).__name__}: {last_error}",
                },
            )
            if on_progress:
                on_progress(completed, total, variant, "failed")
            raise RuntimeError(
                f"{variant.variant_id} 连续失败 {retries + 1} 次，已停止：{last_error}"
            ) from last_error

        append_jsonl(
            manifest_path,
            {
                "protocol_plan_sha256": protocol.get("plan_sha256"),
                "global_index": variant.global_index,
                "variant_id": variant.variant_id,
                "base_id": variant.base_id,
                "race_group": variant.race_group,
                "sample_id": variant.metadata.get("sample_id"),
                "kind": variant.kind,
                "field": variant.field,
                "field_type": variant.metadata.get("field_type"),
                "class_or_sign": variant.metadata.get("class_or_sign"),
                "allele": variant.metadata.get("allele"),
                "strength": variant.metadata.get("strength"),
                "baseline_repeat": variant.metadata.get("baseline_repeat"),
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "dna_path": variant.dna_path.relative_to(experiment_dir).as_posix(),
                "dna_sha256": variant.dna_sha256,
                "render_path": variant.render_path.relative_to(experiment_dir).as_posix(),
                "render_sha256": render_sha256,
                "round_trip_scope": "full_parsed_dna",
            },
        )
        completed += 1
        if on_progress:
            on_progress(completed, total, variant, "completed")
    return RunResult(completed=completed, skipped=skipped, attempted=attempted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument(
        "--settings",
        type=Path,
        help="自动化 settings.json；默认读取 CK3DNAFieldSweep GUI 保存的设置",
    )
    parser.add_argument(
        "--base-id",
        action="append",
        help="只运行指定 base_id，可重复；默认运行全部 bases",
    )
    parser.add_argument("--limit", type=int, help="仅运行筛选后前 N 个变体，用于小规模门禁")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验 protocol、DNA 文件和哈希，不控制 CK3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = args.experiment.resolve()
    protocol, variants = load_plan(experiment_dir)
    if args.base_id:
        requested = set(args.base_id)
        available = {variant.base_id for variant in variants}
        missing = requested - available
        if missing:
            raise ValueError("未知 base_id: " + ", ".join(sorted(missing)))
        variants = [variant for variant in variants if variant.base_id in requested]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        variants = variants[: args.limit]
    if args.dry_run:
        baselines = sum(variant.kind == "baseline" for variant in variants)
        print(
            f"计划校验通过: {len(variants)} 个变体，{baselines} 个 baseline，"
            f"plan={protocol.get('plan_sha256')}"
        )
        return 0

    if args.settings is not None:
        settings = _read_json(args.settings.resolve())
    else:
        settings = load_user_settings()
    config = automation_config_from_settings(settings)
    if config.verify_copy_button is None:
        raise ValueError(
            "正式可辨识度实验必须先在 dna_field_sweep_tool.py 中记录“复制 DNA 验证按钮”"
        )
    backend = WindowsAutomationBackend(config)
    progress_every = max(1, int(args.progress_every))

    def progress(done: int, total: int, variant: PlannedVariant, status: str) -> None:
        if status == "failed" or done % progress_every == 0 or done == total:
            print(
                f"[{done}/{total}] {variant.base_id} {variant.variant_id} [{status}]",
                flush=True,
            )

    result = run_experiment(
        experiment_dir,
        protocol,
        variants,
        backend,
        retries=config.retries,
        on_progress=progress,
    )
    print(
        f"运行完成: completed={result.completed}, skipped={result.skipped}, "
        f"attempted={result.attempted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
