from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from analyze_identifiability_experiment import analyze_experiment


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IdentifiabilityAnalysisTests(unittest.TestCase):
    def test_analysis_finds_bit_exact_signed_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_path = root / "schema.json"
            schema = {
                "schema_version": 1,
                "normalization": {"byte_divisor": 255},
                "signed_fields": [
                    {
                        "name": "gene_chin_forward",
                        "negative_allele": "chin_neg",
                        "positive_allele": "chin_pos",
                        "zero_allele": "chin_neg",
                    },
                    {
                        "name": "gene_bs_nose_forward",
                        "negative_allele": "nose_neg",
                        "positive_allele": "nose_pos",
                        "zero_allele": "nose_neg",
                    },
                ],
                "categorical_fields": [
                    {
                        "name": "gene_bs_eye_fold_shape",
                        "classes": ["fold_a", "fold_b"],
                    }
                ],
                "color_fields": [],
            }
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            render_dir = root / "renders"
            render_dir.mkdir()
            rows = []
            index = 0

            def save_image(name: str, shape: str, strength: int) -> Path:
                path = render_dir / f"{name}.png"
                image = Image.new("RGB", (80, 48), (100, 110, 120))
                draw = ImageDraw.Draw(image)
                amount = max(1, round(strength / 32))
                if shape == "center":
                    draw.rectangle((35 - amount, 16, 44 + amount, 34), fill=(180, 80, 60))
                elif shape == "left":
                    draw.rectangle((8, 14, 8 + amount * 2, 36), fill=(200, 70, 50))
                elif shape == "right":
                    draw.rectangle((71 - amount * 2, 14, 71, 36), fill=(200, 70, 50))
                elif shape == "horizontal":
                    draw.rectangle((20, 22 - amount, 60, 25 + amount), fill=(60, 190, 80))
                elif shape == "vertical":
                    draw.rectangle((38 - amount, 8, 41 + amount, 40), fill=(60, 190, 80))
                image.save(path)
                return path

            for base_number in range(2):
                base_id = f"base_{base_number}"
                baseline = save_image(f"{base_id}_baseline", "center", 0)
                for repeat in range(2):
                    index += 1
                    rows.append(
                        {
                            "global_index": index,
                            "variant_id": f"{base_id}_baseline_{repeat}",
                            "base_id": base_id,
                            "kind": "baseline",
                            "baseline_repeat": repeat + 1,
                            "status": "completed",
                            "render_path": baseline.relative_to(root).as_posix(),
                            "render_sha256": image_hash(baseline),
                        }
                    )
                plans = [
                    ("gene_chin_forward", "signed", "negative", "center"),
                    ("gene_chin_forward", "signed", "positive", "center"),
                    ("gene_bs_nose_forward", "signed", "negative", "left"),
                    ("gene_bs_nose_forward", "signed", "positive", "right"),
                    ("gene_bs_eye_fold_shape", "categorical", "fold_a", "horizontal"),
                    ("gene_bs_eye_fold_shape", "categorical", "fold_b", "vertical"),
                ]
                for field, field_type, class_name, shape in plans:
                    for strength in (0, 128, 255):
                        # Alias alleles intentionally reuse byte-identical images.
                        image_name = (
                            f"{base_id}_{field}_alias_{strength}"
                            if field == "gene_chin_forward"
                            else f"{base_id}_{field}_{class_name}_{strength}"
                        )
                        path = render_dir / f"{image_name}.png"
                        if not path.exists():
                            path = save_image(image_name, shape, strength)
                        index += 1
                        rows.append(
                            {
                                "global_index": index,
                                "variant_id": f"v{index}",
                                "base_id": base_id,
                                "kind": "field",
                                "field": field,
                                "field_type": field_type,
                                "class_or_sign": class_name,
                                "strength": strength,
                                "status": "completed",
                                "render_path": path.relative_to(root).as_posix(),
                                "render_sha256": image_hash(path),
                            }
                        )

            (root / "render_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "total_variants": len(rows),
                        "schema_path": str(schema_path),
                        "plan_sha256": "test-plan",
                    }
                ),
                encoding="utf-8",
            )

            summary = analyze_experiment(
                root, feature_size=(64, 40), output_dir=root, progress=False
            )
            self.assertEqual(summary["robust_signed_alias_count"], 1)
            groups = json.loads((root / "field_groups.json").read_text(encoding="utf-8"))
            self.assertEqual(
                groups["robust_signed_allele_aliases"], ["gene_chin_forward"]
            )
            self.assertNotIn(
                "gene_bs_nose_forward", groups["robust_signed_allele_aliases"]
            )
            recommended = json.loads(
                (root / "recommended_training_schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [field["name"] for field in recommended["scalar_fields"]],
                ["gene_chin_forward"],
            )
            with (root / "field_identifiability.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                decisions = {row["field"]: row for row in csv.DictReader(stream)}
            self.assertEqual(
                decisions["gene_chin_forward"]["recommended_strategy"],
                "merge_alleles_scalar",
            )
            self.assertEqual(
                decisions["gene_chin_forward"]["source_head_weight"], "0.0"
            )


if __name__ == "__main__":
    unittest.main()
