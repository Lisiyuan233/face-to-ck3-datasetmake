#!/usr/bin/env python3
"""Preprocess the CK3 composite portraits into front-face WebDataset shards.

The script is intentionally deterministic. It crops and resizes the collected
PNG files, encodes them as JPEG, joins normalized JSONL labels by sample_id, and
writes atomic, uncompressed tar shards suitable for sequential training I/O.
Random data augmentation belongs in the training loader, not in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image, ImageOps


SCRIPT_VERSION = 1
IMAGE_NAME_RE = re.compile(r"^face_(\d+)\.png$", re.IGNORECASE)
DEFAULT_CROP = (150, 20, 690, 830)
DEFAULT_OUTPUT_SIZE = (256, 384)
DEFAULT_EXPECTED_SIZE = (1326, 891)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ImageTask:
    path: str
    sample_id: str
    crop: tuple[int, int, int, int]
    output_size: tuple[int, int]
    expected_size: tuple[int, int]
    allow_size_mismatch: bool
    jpeg_quality: int
    jpeg_subsampling: int


@dataclass(frozen=True)
class ProcessedImage:
    sample_id: str
    jpeg: bytes | None
    source_size: tuple[int, int] | None
    error: str | None = None


def parse_int_tuple(value: str, length: int, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if len(values) != length:
        raise argparse.ArgumentTypeError(
            f"{name} requires {length} comma-separated integers"
        )
    return values


def parse_crop(value: str) -> tuple[int, int, int, int]:
    values = parse_int_tuple(value, 4, "crop")
    left, top, right, bottom = values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError(
            "crop must be left,top,right,bottom with positive area"
        )
    return left, top, right, bottom


def parse_size(value: str) -> tuple[int, int]:
    values = parse_int_tuple(value, 2, "size")
    width, height = values
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def parse_splits(value: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("splits must contain numbers") from error
    if len(values) != 3:
        raise argparse.ArgumentTypeError("splits requires train,val,test ratios")
    if any(ratio < 0 for ratio in values) or abs(sum(values) - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError("split ratios must be non-negative and sum to 1")
    return values  # type: ignore[return-value]


def discover_images(input_dir: Path) -> tuple[list[Path], list[str]]:
    """Return numerically sorted dataset portraits and excluded PNG names."""
    images: list[tuple[int, Path]] = []
    excluded: list[str] = []
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".png"):
                continue
            match = IMAGE_NAME_RE.fullmatch(entry.name)
            if match:
                images.append((int(match.group(1)), Path(entry.path)))
            else:
                excluded.append(entry.name)
    images.sort(key=lambda item: item[0])
    excluded.sort()
    return [path for _, path in images], excluded


def sample_number(sample_id: str) -> int:
    match = re.fullmatch(r"face_(\d+)", sample_id, re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid sample_id: {sample_id}")
    number = int(match.group(1))
    if number < 1:
        raise ValueError(f"sample_id index must be >= 1: {sample_id}")
    return number


def race_position(sample_id: str, group_size: int) -> tuple[int, int]:
    """Return zero-based (race_group, offset_within_group)."""
    zero_based = sample_number(sample_id) - 1
    return zero_based // group_size, zero_based % group_size


def _affine_permutation(value: int, modulus: int, seed: int, group: int) -> int:
    """Deterministically permute 0..modulus-1 without allocating a shuffle."""
    digest = hashlib.blake2b(
        f"{seed}:race:{group}".encode("utf-8"), digest_size=16
    ).digest()
    multiplier = int.from_bytes(digest[:8], "big") % modulus
    while math.gcd(multiplier, modulus) != 1:
        multiplier = (multiplier + 1) % modulus
    offset = int.from_bytes(digest[8:], "big") % modulus
    return (multiplier * value + offset) % modulus


def split_for_sample(
    sample_id: str,
    ratios: tuple[float, float, float],
    seed: int,
    race_group_size: int,
) -> str:
    """Assign a stable split, exactly balanced inside complete race blocks."""
    if race_group_size:
        group, offset = race_position(sample_id, race_group_size)
        shuffled = _affine_permutation(offset, race_group_size, seed, group)
        train_count = round(ratios[0] * race_group_size)
        val_count = round(ratios[1] * race_group_size)
        if shuffled < train_count:
            return "train"
        if shuffled < train_count + val_count:
            return "val"
        return "test"
    else:
        # Fallback for datasets without known sequential population blocks.
        digest = hashlib.blake2b(
            f"{seed}:{sample_id}".encode("utf-8"), digest_size=8
        ).digest()
        position = int.from_bytes(digest, "big") / 2**64
    if position < ratios[0]:
        return "train"
    if position < ratios[0] + ratios[1]:
        return "val"
    return "test"


def attach_sample_metadata(
    sample_id: str, label: bytes | None, race_group_size: int
) -> bytes:
    """Add grouping metadata to normalized labels or create a metadata record."""
    if label is None:
        value: dict[str, Any] = {"sample_id": sample_id}
    else:
        decoded = json.loads(label)
        if not isinstance(decoded, dict):
            raise ValueError(f"label for {sample_id} is not a JSON object")
        if decoded.get("sample_id") != sample_id:
            raise ValueError(
                f"label sample_id mismatch: expected {sample_id}, "
                f"got {decoded.get('sample_id')!r}"
            )
        value = decoded
    if race_group_size:
        group, _ = race_position(sample_id, race_group_size)
        value["race_group"] = group
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def process_image(task: ImageTask) -> ProcessedImage:
    """Worker-safe image decode, validation, crop, resize and JPEG encode."""
    try:
        with Image.open(task.path) as source:
            source = ImageOps.exif_transpose(source)
            source_size = source.size
            if not task.allow_size_mismatch and source_size != task.expected_size:
                raise ValueError(
                    f"expected {task.expected_size}, got {source_size}"
                )
            left, top, right, bottom = task.crop
            if right > source.width or bottom > source.height:
                raise ValueError(
                    f"crop {task.crop} exceeds image size {source_size}"
                )
            image = source.convert("RGB").crop(task.crop)
            image = image.resize(task.output_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=task.jpeg_quality,
                subsampling=task.jpeg_subsampling,
                optimize=False,
                progressive=False,
            )
        return ProcessedImage(
            sample_id=task.sample_id,
            jpeg=output.getvalue(),
            source_size=source_size,
        )
    except Exception as error:
        return ProcessedImage(
            sample_id=task.sample_id,
            jpeg=None,
            source_size=None,
            error=f"{task.path}: {error}",
        )


class LabelIndex:
    """Disk-backed sample_id -> JSONL byte-range index.

    The SQLite database stores only offsets and lengths. Label payload remains
    in the original JSONL, avoiding both high RAM use and a duplicate 1 GB DB.
    """

    def __init__(
        self, labels_path: Path, temp_dir: Path | None, progress_every: int
    ) -> None:
        self.labels_path = labels_path
        self.progress_every = progress_every
        self._temporary = tempfile.TemporaryDirectory(
            prefix="ck3_label_index_", dir=str(temp_dir) if temp_dir else None
        )
        database_path = Path(self._temporary.name) / "labels.sqlite3"
        self.connection = sqlite3.connect(database_path)
        self.handle = labels_path.open("rb")
        self.count = 0
        self._build()

    def _build(self) -> None:
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(
            "CREATE TABLE labels (sample_id TEXT PRIMARY KEY, offset INTEGER, length INTEGER)"
        )
        batch: list[tuple[str, int, int]] = []
        offset = 0
        line_number = 0
        while True:
            line = self.handle.readline()
            if not line:
                break
            line_number += 1
            length = len(line)
            stripped = line.strip()
            if not stripped:
                offset += length
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{self.labels_path}:{line_number}: invalid JSON: {error}"
                ) from error
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"{self.labels_path}:{line_number}: missing string sample_id"
                )
            batch.append((sample_id, offset, length))
            if len(batch) >= 5000:
                self._insert_batch(batch)
                batch.clear()
                if self.progress_every and self.count % self.progress_every == 0:
                    print(
                        f"labels: indexed {self.count} records...", file=sys.stderr
                    )
            offset += length
        if batch:
            self._insert_batch(batch)
        self.connection.commit()
        self.handle.seek(0)

    def _insert_batch(self, batch: list[tuple[str, int, int]]) -> None:
        try:
            self.connection.executemany(
                "INSERT INTO labels(sample_id, offset, length) VALUES (?, ?, ?)", batch
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"duplicate sample_id in {self.labels_path}") from error
        self.count += len(batch)

    def get(self, sample_id: str) -> bytes | None:
        row = self.connection.execute(
            "SELECT offset, length FROM labels WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        if row is None:
            return None
        offset, length = row
        self.handle.seek(offset)
        return self.handle.read(length).strip()

    def close(self) -> None:
        self.handle.close()
        self.connection.close()
        self._temporary.cleanup()

    def __enter__(self) -> "LabelIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class TarShardWriter:
    """Write deterministic WebDataset-style tar shards atomically."""

    def __init__(self, output_dir: Path, split: str, max_samples: int) -> None:
        self.output_dir = output_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_samples = max_samples
        self.shard_index = 0
        self.current_count = 0
        self.total_count = 0
        self.completed_shards = 0
        self._tar: tarfile.TarFile | None = None
        self._partial_path: Path | None = None
        self._final_path: Path | None = None

    def _open(self) -> None:
        final_name = f"{self.split}-{self.shard_index:06d}.tar"
        self._final_path = self.output_dir / final_name
        self._partial_path = self.output_dir / f"{final_name}.partial"
        if self._final_path.exists() or self._partial_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing shard: {self._final_path}"
            )
        self._tar = tarfile.open(self._partial_path, mode="w", format=tarfile.PAX_FORMAT)
        self.current_count = 0

    def add(self, sample_id: str, jpeg: bytes, label: bytes | None) -> None:
        if self._tar is None:
            self._open()
        assert self._tar is not None
        self._add_bytes(f"{sample_id}.jpg", jpeg)
        if label is not None:
            self._add_bytes(f"{sample_id}.json", label)
        self.current_count += 1
        self.total_count += 1
        if self.current_count >= self.max_samples:
            self._finalize_current()

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        self._tar.addfile(info, io.BytesIO(payload))

    def _finalize_current(self) -> None:
        if self._tar is None:
            return
        assert self._partial_path is not None and self._final_path is not None
        self._tar.close()
        self._tar = None
        os.replace(self._partial_path, self._final_path)
        self.completed_shards += 1
        self.shard_index += 1
        self.current_count = 0
        self._partial_path = None
        self._final_path = None

    def close(self, success: bool) -> None:
        if self._tar is None:
            return
        if success:
            self._finalize_current()
        else:
            # Close the handle but retain the explicitly named .partial file so
            # an interrupted run can never be mistaken for a valid shard.
            self._tar.close()
            self._tar = None


def batched(values: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def ensure_output_is_safe(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                f"output directory is not empty: {output_dir}; use a new directory"
            )
    else:
        output_dir.mkdir(parents=True)


def build_task(path: Path, args: argparse.Namespace) -> ImageTask:
    return ImageTask(
        path=str(path),
        sample_id=path.stem,
        crop=args.crop,
        output_size=args.size,
        expected_size=args.expected_size,
        allow_size_mismatch=args.allow_size_mismatch,
        jpeg_quality=args.jpeg_quality,
        jpeg_subsampling=args.jpeg_subsampling,
    )


def process_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {args.input_dir}")
    ensure_output_is_safe(args.output_dir)
    images, excluded_pngs = discover_images(args.input_dir)
    discovered_count = len(images)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError("no face_<number>.png files found")

    print(
        f"images: discovered {discovered_count}, selected {len(images)}, "
        f"excluded PNGs: {len(excluded_pngs)}"
    )
    writers = {
        split: TarShardWriter(args.output_dir, split, args.shard_size)
        for split in SPLIT_NAMES
    }
    split_counts = {split: 0 for split in SPLIT_NAMES}
    race_split_counts: dict[int, dict[str, int]] = {}
    skipped: list[dict[str, str]] = []
    matched_labels = 0
    source_sizes: set[tuple[int, int]] = set()
    started = time.monotonic()
    success = False

    label_context: Any
    if args.labels:
        label_context = LabelIndex(args.labels, args.temp_dir, args.progress_every)
    else:
        label_context = _NullLabelIndex()

    try:
        with label_context as labels, ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            for path_batch in batched(images, max(args.workers * 4, 1)):
                tasks = [build_task(path, args) for path in path_batch]
                results = executor.map(process_image, tasks, chunksize=1)
                for path, result in zip(path_batch, results):
                    try:
                        if result.error is not None:
                            raise ValueError(result.error)
                        assert result.jpeg is not None and result.source_size is not None
                        label = labels.get(result.sample_id)
                        if args.labels and label is None:
                            raise ValueError(
                                f"no label found for sample_id={result.sample_id}"
                            )
                        if label is not None:
                            matched_labels += 1
                        split = split_for_sample(
                            result.sample_id,
                            args.splits,
                            args.split_seed,
                            args.race_group_size,
                        )
                        metadata = attach_sample_metadata(
                            result.sample_id, label, args.race_group_size
                        )
                        writers[split].add(result.sample_id, result.jpeg, metadata)
                        split_counts[split] += 1
                        if args.race_group_size:
                            group, _ = race_position(
                                result.sample_id, args.race_group_size
                            )
                            counts = race_split_counts.setdefault(
                                group,
                                {"total": 0, "train": 0, "val": 0, "test": 0},
                            )
                            counts["total"] += 1
                            counts[split] += 1
                        source_sizes.add(result.source_size)
                    except Exception as error:
                        if not args.skip_invalid:
                            raise
                        skipped.append({"path": str(path), "error": str(error)})

                    completed = sum(split_counts.values()) + len(skipped)
                    if args.progress_every and completed % args.progress_every == 0:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        rate = completed / elapsed
                        print(
                            f"images: {completed}/{len(images)} "
                            f"({rate:.1f} images/s)...",
                            file=sys.stderr,
                        )
        success = True
    finally:
        for writer in writers.values():
            writer.close(success=success)

    elapsed = time.monotonic() - started
    processed = sum(split_counts.values())
    label_count = label_context.count if args.labels else None
    manifest = {
        "preprocessing_version": SCRIPT_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "labels": str(args.labels.resolve()) if args.labels else None,
        "source_images_discovered": discovered_count,
        "source_images_selected": len(images),
        "processed": processed,
        "skipped": len(skipped),
        "excluded_pngs": excluded_pngs,
        "source_sizes": [list(size) for size in sorted(source_sizes)],
        "crop": list(args.crop),
        "output_size": list(args.size),
        "jpeg": {
            "quality": args.jpeg_quality,
            "subsampling": args.jpeg_subsampling,
        },
        "split": {
            "ratios": dict(zip(SPLIT_NAMES, args.splits)),
            "seed": args.split_seed,
            "counts": split_counts,
        },
        "race_groups": {
            "group_size": args.race_group_size or None,
            "group_count": len(race_split_counts) if args.race_group_size else None,
            "counts": {
                str(group): counts
                for group, counts in sorted(race_split_counts.items())
            },
        },
        "shard_size": args.shard_size,
        "shards": {
            split: writers[split].completed_shards for split in SPLIT_NAMES
        },
        "label_count": label_count,
        "matched_labels": matched_labels if args.labels else None,
        "unmatched_labels": (
            label_count - matched_labels
            if args.labels and args.limit is None and label_count is not None
            else None
        ),
        "elapsed_seconds": round(elapsed, 3),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if skipped:
        with (args.output_dir / "errors.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for error in skipped:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    return manifest


class _NullLabelIndex:
    count = 0

    def get(self, sample_id: str) -> None:
        return None

    def __enter__(self) -> "_NullLabelIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop CK3 composite PNG portraits into front-face JPEG WebDataset shards."
        )
    )
    parser.add_argument("input_dir", type=Path, help="directory containing face_*.png")
    parser.add_argument("output_dir", type=Path, help="new, empty output directory")
    parser.add_argument("--labels", type=Path, help="normalized labels.jsonl")
    parser.add_argument(
        "--crop", type=parse_crop, default=DEFAULT_CROP, metavar="L,T,R,B"
    )
    parser.add_argument(
        "--size", type=parse_size, default=DEFAULT_OUTPUT_SIZE, metavar="W,H"
    )
    parser.add_argument(
        "--expected-size",
        type=parse_size,
        default=DEFAULT_EXPECTED_SIZE,
        metavar="W,H",
    )
    parser.add_argument("--allow-size-mismatch", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--jpeg-subsampling",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="0 preserves full chroma; 2 is smallest",
    )
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=(0.90, 0.05, 0.05),
        metavar="TRAIN,VAL,TEST",
    )
    parser.add_argument("--split-seed", type=int, default=20260718)
    parser.add_argument(
        "--race-group-size",
        type=int,
        default=30000,
        help="sequential samples per race; use 0 to disable grouped splitting",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument("--limit", type=int, help="process only the first N images")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="temporary directory for the compact label-offset index",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    left, top, right, bottom = args.crop
    expected_width, expected_height = args.expected_size
    if not args.allow_size_mismatch and (
        right > expected_width or bottom > expected_height
    ):
        parser.error("crop exceeds --expected-size")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.shard_size < 1:
        parser.error("--shard-size must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.race_group_size < 0:
        parser.error("--race-group-size must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.labels and not args.labels.is_file():
        parser.error(f"labels file does not exist: {args.labels}")
    if args.temp_dir and not args.temp_dir.is_dir():
        parser.error(f"temporary directory does not exist: {args.temp_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        manifest = process_dataset(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    counts = manifest["split"]["counts"]
    print(
        f"done: {manifest['processed']} images, skipped {manifest['skipped']}, "
        f"train/val/test={counts['train']}/{counts['val']}/{counts['test']} "
        f"in {manifest['elapsed_seconds']:.1f}s -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
