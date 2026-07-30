from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

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
        self.assertGreaterEqual(
            float(components["consistency"].detach()), 0.0
        )

    def test_class_loss_is_normalized_over_observable_samples(self) -> None:
        from ck3_training.losses import MultitaskLoss

        value = MultitaskLoss._observable_weighted_mean(
            torch.tensor([2.0, 100.0]),
            torch.tensor([0.5, 0.0]),
            torch.tensor([True, False]),
        )
        self.assertAlmostEqual(float(value), 2.0)

    def test_side_view_uses_gated_fusion(self) -> None:
        from ck3_training.config import DEFAULT_CONFIG, apply_smoke_overrides
        from ck3_training.model import FaceToCK3Model
        from ck3_training.schema import load_schema

        schema = load_schema(DATASET / "dna_schema.json")
        config = apply_smoke_overrides(DEFAULT_CONFIG)
        config["model"]["side_view"] = True
        model = FaceToCK3Model(schema, config["model"])
        image = torch.rand(2, 3, 64, 64)
        outputs = model(image, image, image)
        self.assertEqual(outputs["signed"].shape, (2, schema.signed_dim))
        self.assertTrue(hasattr(model, "side_gate"))

    def test_geometry_map_exposes_foreground_and_edges(self) -> None:
        from ck3_training.data import build_geometry_map

        image = Image.new("RGB", (64, 96), (34, 36, 40))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 12, 48, 74), fill=(170, 120, 90))
        geometry_map = build_geometry_map(
            image,
            grid_height=24,
            grid_width=16,
            foreground_margin=0.06,
            foreground_softness=0.03,
        )
        self.assertEqual(tuple(geometry_map.shape), (2, 24, 16))
        self.assertTrue(torch.isfinite(geometry_map).all())
        self.assertGreater(
            float(geometry_map[0, 10:18, 6:10].mean()),
            float(geometry_map[0, :3, :3].mean()),
        )
        self.assertGreater(float(geometry_map[1].max()), 0.0)

    def test_geometry_branch_uses_front_and_side_maps(self) -> None:
        from ck3_training.config import DEFAULT_CONFIG, apply_smoke_overrides
        from ck3_training.model import FaceToCK3Model
        from ck3_training.schema import load_schema

        schema = load_schema(DATASET / "dna_schema.json")
        config = apply_smoke_overrides(DEFAULT_CONFIG)
        config["model"]["dual_view"] = True
        config["model"]["side_view"] = True
        config["model"]["geometry_branch"]["enabled"] = True
        model = FaceToCK3Model(schema, config["model"])
        image = torch.rand(2, 3, 64, 64)
        front_map = torch.rand(2, 2, 24, 16)
        side_map = torch.rand(2, 2, 24, 16)
        outputs = model(
            image,
            image,
            image,
            front_map,
            side_map,
        )
        self.assertEqual(outputs["signed"].shape, (2, schema.signed_dim))
        self.assertTrue(hasattr(model, "geometry_gate"))
        self.assertEqual(outputs["geometry_gate_mean"].shape, (2,))
        with self.assertRaisesRegex(ValueError, "side_geometry_map"):
            model(image, image, image, front_map)

    def test_selection_score_uses_observable_macro_f1(self) -> None:
        from ck3_training.metrics import selection_score

        metrics = {
            "signed_mae": 0.2,
            "strength_mae": 0.1,
            "categorical_macro_f1": 0.1,
            "categorical_observable_macro_f1": 0.8,
        }
        score = selection_score(
            metrics,
            {
                "signed_mae_weight": 0.0,
                "strength_mae_weight": 0.0,
                "categorical_error_weight": 1.0,
            },
        )
        self.assertAlmostEqual(score, 0.2)


if __name__ == "__main__":
    unittest.main()
