from __future__ import annotations

import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from ck3_training.data import TarShardDataset
from ck3_training.schema import load_schema
from ck3_training.split_index import load_split_ids, load_split_manifest
from tools.build_training_label_stats import scan_shard
from tools.build_dna_grouped_split import (
    build_grouped_split,
    normalized_target_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "experiments"
    / "dna_identifiability"
    / "recommended_training_schema.json"
)
LABELS = ROOT / "face_to_ck3_dataset_male_small" / "labels.jsonl"


class GroupedSplitTests(unittest.TestCase):
    @staticmethod
    def _write_tar(path: Path, rows: list[dict]) -> None:
        with tarfile.open(path, "w") as archive:
            for row in rows:
                sample_id = row["sample_id"]
                for suffix, payload in (
                    ("front.jpg", b"front"),
                    ("side.jpg", b"side"),
                    ("json", json.dumps(row).encode("utf-8")),
                ):
                    info = tarfile.TarInfo(f"{sample_id}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

    def test_identical_targets_never_cross_splits(self) -> None:
        schema = load_schema(SCHEMA)
        with LABELS.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
            next(stream)
            second = json.loads(next(stream))
        rows = []
        for index, source in enumerate((first, first, second, second), start=1):
            copied = dict(source)
            copied["sample_id"] = f"synthetic_{index}"
            rows.append(copied)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels.jsonl"
            labels.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "split"
            manifest = build_grouped_split(
                labels, schema, output, seed=7, progress_every=0
            )
            loaded = load_split_manifest(output / "manifest.json")
            self.assertEqual(manifest["target_group_count"], 2)
            self.assertEqual(loaded["cross_split_duplicate_groups"], 0)
            assignments = {}
            for split in ("train", "val", "test"):
                for sample_id in load_split_ids(output / "manifest.json", split):
                    assignments[sample_id] = split
            self.assertEqual(assignments["synthetic_1"], assignments["synthetic_2"])
            self.assertEqual(assignments["synthetic_3"], assignments["synthetic_4"])
            self.assertEqual(sum(manifest["counts"].values()), 4)

    def test_fingerprint_ignores_sample_metadata(self) -> None:
        schema = load_schema(SCHEMA)
        with LABELS.open(encoding="utf-8") as stream:
            source = json.loads(next(stream))
        changed = dict(source)
        changed["sample_id"] = "different"
        changed["race_group"] = 99
        changed["colors"] = [0.0] * len(source.get("colors", ()))
        self.assertEqual(
            normalized_target_fingerprint(source, schema),
            normalized_target_fingerprint(changed, schema),
        )

    def test_tar_reader_and_stats_filter_before_decode(self) -> None:
        schema = load_schema(SCHEMA)
        with LABELS.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
            second = json.loads(next(stream))
        first["sample_id"] = "keep"
        second["sample_id"] = "skip"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.tar"
            self._write_tar(path, [first, second])
            dataset = TarShardDataset(
                [path],
                schema,
                transform=object(),  # _read_shard does not decode images
                training=False,
                repeat=False,
                shuffle_buffer=1,
                seed=1,
                sample_ids=frozenset(("keep",)),
                require_side_view=True,
            )
            raw = list(dataset._read_shard(path))
            self.assertEqual([row["key"] for row in raw], ["keep"])
            stats = scan_shard(
                path, schema, 0.1, sample_ids=frozenset(("keep",))
            )
            self.assertEqual(stats["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
