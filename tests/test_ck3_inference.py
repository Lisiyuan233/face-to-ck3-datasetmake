from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ck3_inference import (
    FieldQuality,
    build_dna_from_prediction,
    center_crop_to_aspect,
    crop_composite_views,
)
from dna_normalizer import normalize_record, parse_dna


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "face_to_ck3_dataset_male_v2" / "recommended_training_schema.json"
TEMPLATE = ROOT / "face_to_ck3_dataset_male_v2" / "dna" / "face_0001.txt"


class CK3InferenceTests(unittest.TestCase):
    def test_center_crop_has_training_aspect(self) -> None:
        image = Image.new("RGB", (1000, 500), "white")
        cropped = center_crop_to_aspect(image, 256, 384)
        self.assertEqual(cropped.size, (333, 500))

    def test_composite_crop_scales_manifest_coordinates(self) -> None:
        image = Image.new("RGB", (2490, 1658), "white")
        front, side = crop_composite_views(
            image,
            front_crop=(80, 19, 620, 829),
            side_crop=(650, 19, 1190, 829),
            expected_size=(1245, 829),
        )
        self.assertEqual(front.size, (1080, 1620))
        self.assertEqual(side.size, (1080, 1620))

    def test_only_selected_field_overrides_template(self) -> None:
        import json

        raw_schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        template_text = TEMPLATE.read_text(encoding="utf-8")
        original = normalize_record(
            parse_dna(template_text), raw_schema, "template", False
        )
        prediction = {
            "scalar": [0.0] * len(raw_schema["scalar_fields"]),
            "signed": [0.0] * len(raw_schema["signed_fields"]),
            "categorical_strength": [0.0] * len(raw_schema["categorical_fields"]),
        }
        selected = raw_schema["scalar_fields"][0]["name"]
        quality = {
            ("scalar", selected): FieldQuality("scalar", selected, 1.0, 0.1, 0.9)
        }
        result = build_dna_from_prediction(
            template_text=template_text,
            prediction=prediction,
            schema_path=SCHEMA,
            quality=quality,
            minimum_improvement=0.25,
            weight_source="ema",
            used_side_fallback=False,
        )
        normalized = normalize_record(
            parse_dna(result.dna), raw_schema, "result", False
        )
        original_record = parse_dna(template_text)
        result_record = parse_dna(result.dna)
        self.assertEqual(result.predicted_fields, (selected,))
        self.assertEqual(normalized["scalar"][0], 0.0)
        self.assertEqual(
            normalized["signed"], original["signed"]
        )
        self.assertEqual(
            normalized["categorical_class"], original["categorical_class"]
        )
        self.assertEqual(normalized["colors"], original["colors"])
        for field, value in original_record.genes.items():
            if field != selected:
                self.assertEqual(result_record.genes[field], value, field)

    def test_strength_prediction_preserves_both_template_alleles(self) -> None:
        import json

        raw_schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        template_text = TEMPLATE.read_text(encoding="utf-8")
        spec = raw_schema["categorical_fields"][1]
        field = spec["name"]
        original = parse_dna(template_text).genes[field]
        prediction = {
            "scalar": [0.5] * len(raw_schema["scalar_fields"]),
            "signed": [0.0] * len(raw_schema["signed_fields"]),
            "categorical_strength": [0.0] * len(raw_schema["categorical_fields"]),
        }
        prediction["categorical_strength"][1] = 1.0
        quality = {
            ("strength", field): FieldQuality("strength", field, 1.0, 0.1, 0.9)
        }
        result = build_dna_from_prediction(
            template_text=template_text,
            prediction=prediction,
            schema_path=SCHEMA,
            quality=quality,
            minimum_improvement=0.25,
            weight_source="ema",
            used_side_fallback=False,
        )
        updated = parse_dna(result.dna).genes[field]
        self.assertEqual(updated.allele1, original.allele1)
        self.assertEqual(updated.allele2, original.allele2)
        self.assertEqual((updated.value1, updated.value2), (255, 255))


if __name__ == "__main__":
    unittest.main()
