from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from ck3_training.metrics import MetricAccumulator


class MetricAccumulatorTests(unittest.TestCase):
    def test_reports_manifest_defined_race_groups_above_legacy_limit(self) -> None:
        schema = SimpleNamespace(
            scalar_dim=0,
            signed_dim=1,
            categorical_dim=1,
            active_categorical_indices=(),
            categorical_fields=(),
        )
        metrics = MetricAccumulator(
            schema,
            torch.device("cpu"),
            race_group_count=22,
        )
        outputs = {
            "signed": torch.tensor([[0.0], [0.0]]),
            "categorical_strength": torch.tensor([[0.0], [0.0]]),
        }
        targets = {
            "signed": torch.tensor([[1.0], [2.0]]),
            "categorical_class": torch.zeros((2, 1), dtype=torch.long),
            "categorical_strength": torch.tensor([[0.0], [0.0]]),
            "race_group": torch.tensor([17, 21], dtype=torch.long),
        }

        metrics.update(outputs, targets, {"total": torch.tensor(0.0)})
        result = metrics.compute()

        self.assertEqual(
            [item["race_group"] for item in result["race_groups"]],
            [17, 21],
        )
        self.assertEqual(
            [item["count"] for item in result["race_groups"]],
            [1, 1],
        )

    def test_rejects_negative_race_group_count(self) -> None:
        schema = SimpleNamespace(categorical_dim=0)
        with self.assertRaisesRegex(ValueError, "race_group_count"):
            MetricAccumulator(
                schema,
                torch.device("cpu"),
                race_group_count=-1,
            )


if __name__ == "__main__":
    unittest.main()
