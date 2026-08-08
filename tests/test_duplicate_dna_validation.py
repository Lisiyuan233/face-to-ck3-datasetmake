from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from validate_duplicate_dna_renders import (
    analyze_validation,
    prepare_plan,
    read_jsonl,
    run_validation,
)


def dna_text(group: int) -> str:
    value = 10 + group
    return f'''ruler_designer_1={{
    genes={{
        gene_test={{ "group_{group}" {value} "group_{group}" {value} }}
        skin_color={{ 20 30 20 30 }}
    }}
}}
'''


class RenderFromDNA:
    def __init__(self, images: dict[str, Path]) -> None:
        self.images = images
        self.current_hash = ""
        self.applied = 0
        self.captured = 0

    def apply_dna(self, value: str, _fields) -> None:
        self.current_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
        self.applied += 1

    def capture(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(self.images[self.current_hash]) as image:
            image.save(path)
        self.captured += 1


class DuplicateDnaValidationTests(unittest.TestCase):
    def test_prepare_run_resume_and_analyze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            experiment = root / "experiment"
            (dataset / "dna").mkdir(parents=True)
            (dataset / "face").mkdir(parents=True)
            hash_to_image: dict[str, Path] = {}
            # Two numeric blocks, five exact adjacent duplicate groups per block.
            for number in range(1, 21):
                group = (number - 1) // 2
                text = dna_text(group)
                sample = f"face_{number:04d}"
                dna_path = dataset / "dna" / f"{sample}.txt"
                image_path = dataset / "face" / f"{sample}.png"
                dna_path.write_text(text, encoding="utf-8")
                rng = np.random.default_rng(group + 123)
                pixels = rng.integers(0, 256, (72, 108, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(image_path)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                hash_to_image.setdefault(digest, image_path)

            protocol = prepare_plan(
                dataset,
                experiment,
                group_count=4,
                repeats=1,
                seed=7,
                block_size=10,
                block_count=2,
            )
            self.assertEqual(protocol["block_quotas"], [2, 2])
            plan = read_jsonl(experiment / "plan.jsonl")
            self.assertEqual(len(plan), 4)
            self.assertTrue(all(len(row["member_ids"]) == 2 for row in plan))

            backend = RenderFromDNA(hash_to_image)
            first = run_validation(experiment, backend)
            self.assertEqual(first, {"completed": 4, "skipped": 0, "attempted": 4})
            resumed = run_validation(experiment, backend)
            self.assertEqual(resumed, {"completed": 4, "skipped": 4, "attempted": 0})
            self.assertEqual(backend.captured, 4)

            summary = analyze_validation(experiment, review_count=0)
            self.assertEqual(summary["group_count"], 4)
            self.assertEqual(summary["same_dna_top1_rate"], 1.0)
            self.assertEqual(summary["classification_counts"], {"aligned": 4})
            self.assertTrue((experiment / "render_comparison.csv").is_file())
            self.assertTrue((experiment / "group_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()

