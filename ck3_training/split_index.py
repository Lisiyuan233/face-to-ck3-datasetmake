from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "val", "test")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(value.get("version", -1)) != 1:
        raise ValueError(f"unsupported split-index version: {value.get('version')}")
    if value.get("method") != "normalized_target_fingerprint":
        raise ValueError(f"unsupported split-index method: {value.get('method')}")
    for split in SPLIT_NAMES:
        if split not in value.get("files", {}) or split not in value.get("counts", {}):
            raise ValueError(f"split-index manifest is missing {split}")
    return value


def split_ids_path(manifest_path: str | Path, split: str) -> Path:
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid split: {split}")
    manifest_path = Path(manifest_path)
    manifest = load_split_manifest(manifest_path)
    path = Path(str(manifest["files"][split]))
    return path if path.is_absolute() else manifest_path.parent / path


def load_split_ids(manifest_path: str | Path, split: str) -> frozenset[str]:
    manifest_path = Path(manifest_path)
    manifest = load_split_manifest(manifest_path)
    path = split_ids_path(manifest_path, split)
    with path.open(encoding="utf-8") as stream:
        values = frozenset(line.strip() for line in stream if line.strip())
    expected = int(manifest["counts"][split])
    if len(values) != expected:
        raise RuntimeError(
            f"split-index {split} contains {len(values)} ids, expected {expected}"
        )
    return values

