from __future__ import annotations

import unittest
from pathlib import Path

from dna_normalizer import (
    apply_normalized_to_template,
    load_schema,
    normalize_record,
    parse_dna,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "face_to_ck3_dataset_male_small"


class TemplateColorTests(unittest.TestCase):
    def test_geometry_prediction_preserves_template_colors(self) -> None:
        schema = load_schema(DATASET / "dna_schema.json")
        template = (DATASET / "dna" / "face_0001.txt").read_text(encoding="utf-8")
        original = parse_dna(template)
        prediction = normalize_record(original, schema, "face_0001")
        prediction.pop("colors")
        prediction["signed"][0] = 0.0

        output = apply_normalized_to_template(template, prediction, schema)

        self.assertEqual(parse_dna(output).colors, original.colors)


if __name__ == "__main__":
    unittest.main()
