from __future__ import annotations

import json
import tarfile
import unittest
from pathlib import Path

from PIL import Image

from ck3_training.schema import load_schema


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "face_to_ck3_dataset_male_small"


class SchemaAndShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema(DATASET / "dna_schema.json")

    def test_schema_dimensions_and_constant_fields(self) -> None:
        self.assertEqual(self.schema.sample_count, 510000)
        self.assertEqual(self.schema.signed_dim, 67)
        self.assertEqual(self.schema.categorical_dim, 16)
        self.assertEqual(
            self.schema.checkpoint_metadata()["target_family"],
            "geometry_only_v1",
        )
        self.assertEqual(len(self.schema.active_categorical_indices), 10)
        self.assertEqual(
            sum(field.is_constant for field in self.schema.categorical_fields), 6
        )

    def test_manifest_counts(self) -> None:
        manifest = json.loads(
            (DATASET / "processed_front" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["split"]["counts"],
            {"train": 459000, "val": 25500, "test": 25500},
        )
        self.assertEqual(manifest["processed"], 510000)
        self.assertEqual(manifest["skipped"], 0)
        self.assertEqual(manifest["unmatched_labels"], 0)

    def test_train_only_label_statistics(self) -> None:
        stats = json.loads(
            (DATASET / "processed_front" / "train_label_stats.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stats["split"], "train")
        self.assertEqual(stats["sample_count"], 459000)
        self.assertEqual(stats["schema_sha256"], self.schema.sha256)
        self.assertEqual(len(stats["categorical_class_counts"]), 16)
        for field, counts in zip(
            self.schema.categorical_fields, stats["categorical_class_counts"]
        ):
            self.assertEqual(len(counts), len(field.classes))
            self.assertEqual(sum(counts), 459000)

    def test_first_validation_samples_are_paired_and_valid(self) -> None:
        shard = DATASET / "processed_front" / "val" / "val-000000.tar"
        pairs: dict[str, dict[str, tarfile.TarInfo]] = {}
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = Path(member.name)
                if path.suffix.lower() not in {".jpg", ".json"}:
                    continue
                pairs.setdefault(path.stem, {})[path.suffix.lower()] = member
                complete = [key for key, value in pairs.items() if len(value) == 2]
                if len(complete) >= 3:
                    break

            self.assertGreaterEqual(len(complete), 3)
            for key in complete[:3]:
                members = pairs[key]
                label_stream = archive.extractfile(members[".json"])
                image_stream = archive.extractfile(members[".jpg"])
                self.assertIsNotNone(label_stream)
                self.assertIsNotNone(image_stream)
                label = json.loads(label_stream.read().decode("utf-8"))
                self.schema.validate_label(label)
                self.assertEqual(label["sample_id"], key)
                with Image.open(image_stream) as image:
                    self.assertEqual(image.size, (256, 384))
                    self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
