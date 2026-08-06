from __future__ import annotations

import unittest
from pathlib import Path

from ck3_training.config import load_config, validate_config
from ck3_training.schema import load_schema
from dna_normalizer import (
    apply_normalized_to_template,
    load_schema as load_normalizer_schema,
    normalize_record,
    parse_dna,
)
from tools.validate_training_setup import validate_setup


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = ROOT / "face_to_ck3_dataset_male_small" / "dna_schema.json"
V2_SCHEMA = (
    ROOT
    / "experiments"
    / "dna_identifiability"
    / "recommended_training_schema.json"
)


class IdentifiabilityTrainingV2Tests(unittest.TestCase):
    def test_schema_v2_adapts_legacy_shard_label(self) -> None:
        source = load_schema(SOURCE_SCHEMA)
        schema = load_schema(V2_SCHEMA)
        self.assertEqual(schema.scalar_dim, 30)
        self.assertEqual(schema.signed_dim, 37)
        self.assertEqual(schema.categorical_dim, 16)
        self.assertEqual(schema.target_family, "geometry_identifiability_v2")
        self.assertEqual(schema.sample_count, 510000)

        legacy = {
            "sample_id": "synthetic",
            "signed": [
                (-0.75 if index % 2 == 0 else 0.25)
                for index in range(source.signed_dim)
            ],
            "categorical_class": [0] * source.categorical_dim,
            "categorical_strength": [0.4] * source.categorical_dim,
            "race_group": 3,
        }
        adapted = schema.adapt_label(legacy)
        schema.validate_label(adapted)
        source_index = {
            name: index for index, name in enumerate(source.signed_fields)
        }
        self.assertEqual(
            adapted["scalar"][0],
            abs(legacy["signed"][source_index[schema.scalar_fields[0]]]),
        )
        self.assertEqual(
            adapted["signed"][0],
            legacy["signed"][source_index[schema.signed_fields[0]]],
        )
        self.assertEqual(adapted["race_group"], 3)

    def test_normalizer_round_trips_schema_v2_scalar_fields(self) -> None:
        schema = load_normalizer_schema(V2_SCHEMA)
        dna_path = ROOT / "face_to_ck3_dataset_male_small" / "dna" / "face_0001.txt"
        template = dna_path.read_text(encoding="utf-8-sig")
        label = normalize_record(parse_dna(template), schema, "face_0001")
        self.assertEqual(len(label["scalar"]), 30)
        self.assertEqual(len(label["signed"]), 37)
        output = apply_normalized_to_template(template, label, schema)
        reparsed = parse_dna(output)
        first = schema["scalar_fields"][0]
        self.assertEqual(
            reparsed.genes[first["name"]].allele1,
            first["canonical_allele"],
        )
        self.assertEqual(
            reparsed.genes[first["name"]].value1,
            round(label["scalar"][0] * 255),
        )

    def test_identifiability_training_config_is_valid(self) -> None:
        config_path = (
            ROOT
            / "configs"
            / "train_convnext_tiny_multiview_identifiability_v2.json"
        )
        config = load_config(config_path)
        validate_config(config)
        self.assertIn("scalar", config["model"]["geometry_branch"]["targets"])
        self.assertTrue(config["loss"]["use_schema_visibility_thresholds"])
        self.assertTrue(
            config["augmentation"]["exposure_normalization"]["enabled"]
        )
        setup = validate_setup(config_path)
        self.assertTrue(setup["legacy_statistics_adapted"])
        self.assertEqual(setup["splits"]["train"]["source_signed_dim"], 67)
        self.assertEqual(
            setup["loss_profile_coverage"],
            {"field_weights_path": 83, "texture_metrics_path": 83},
        )


if __name__ == "__main__":
    unittest.main()
