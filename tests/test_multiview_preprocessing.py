from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_preprocessor import (
    DEFAULT_CROP,
    DEFAULT_EXPECTED_SIZE,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_SIDE_CROP,
    ImageTask,
    TarShardWriter,
    process_image,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "face_to_ck3_dataset_male_small"


class MultiviewPreprocessingTests(unittest.TestCase):
    def test_process_image_extracts_aligned_front_and_side_views(self) -> None:
        result = process_image(
            ImageTask(
                path=str(DATASET / "face" / "face_0001.png"),
                sample_id="face_0001",
                front_crop=DEFAULT_CROP,
                side_crop=DEFAULT_SIDE_CROP,
                output_size=DEFAULT_OUTPUT_SIZE,
                expected_size=DEFAULT_EXPECTED_SIZE,
                allow_size_mismatch=False,
                jpeg_quality=95,
                jpeg_subsampling=0,
            )
        )
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.front_jpeg)
        self.assertIsNotNone(result.side_jpeg)
        for payload in (result.front_jpeg, result.side_jpeg):
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.size, DEFAULT_OUTPUT_SIZE)
                self.assertEqual(image.mode, "RGB")

    def test_writer_pairs_both_views_with_one_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = TarShardWriter(root, "train", max_samples=1)
            writer.add(
                "face_0001",
                b"front",
                b"side",
                json.dumps({"sample_id": "face_0001"}).encode("utf-8"),
            )
            writer.close(success=True)
            shard = root / "train" / "train-000000.tar"
            with tarfile.open(shard, "r:") as archive:
                self.assertEqual(
                    [member.name for member in archive if member.isfile()],
                    [
                        "face_0001.front.jpg",
                        "face_0001.side.jpg",
                        "face_0001.json",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
