from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "face_to_ck3_dataset_male_small"


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class GeometryModelTests(unittest.TestCase):
    def test_forward_and_loss_have_no_color_task(self) -> None:
        from ck3_training.config import DEFAULT_CONFIG, apply_smoke_overrides
        from ck3_training.losses import MultitaskLoss
        from ck3_training.model import FaceToCK3Model
        from ck3_training.schema import load_schema

        schema = load_schema(DATASET / "dna_schema.json")
        config = apply_smoke_overrides(DEFAULT_CONFIG)
        config["model"]["dual_view"] = True
        config["loss"]["consistency_weight"] = 0.1
        model = FaceToCK3Model(schema, config["model"])
        self.assertFalse(hasattr(model, "color_head"))

        image = torch.rand(2, 3, 64, 64)
        outputs = model(image, image)
        self.assertNotIn("colors", outputs)
        self.assertIn("reference", outputs)

        stats = json.loads(
            (DATASET / "processed_front" / "train_label_stats.json").read_text(
                encoding="utf-8"
            )
        )
        criterion = MultitaskLoss(
            schema, config["loss"], stats["categorical_class_counts"]
        )
        targets = {
            "signed": torch.zeros(2, schema.signed_dim),
            "categorical_class": torch.zeros(
                2, schema.categorical_dim, dtype=torch.long
            ),
            "categorical_strength": torch.full(
                (2, schema.categorical_dim), 0.5
            ),
        }
        loss, components = criterion(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("color", components)
        self.assertGreaterEqual(float(components["consistency"]), 0.0)


if __name__ == "__main__":
    unittest.main()
