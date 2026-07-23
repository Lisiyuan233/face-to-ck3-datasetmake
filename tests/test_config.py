from __future__ import annotations

import copy
import unittest

from ck3_training.config import (
    DEFAULT_CONFIG,
    apply_smoke_overrides,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        validate_config(DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG["loss"]["class_label_smoothing"], 0.05)
        self.assertEqual(DEFAULT_CONFIG["loss"]["consistency_weight"], 0.1)
        self.assertNotIn("color_weight", DEFAULT_CONFIG["loss"])
        self.assertEqual(
            sum(DEFAULT_CONFIG["selection"].values()),
            1.0,
        )

    def test_smoke_overrides_are_cpu_friendly(self) -> None:
        config = apply_smoke_overrides(DEFAULT_CONFIG)
        validate_config(config)
        self.assertEqual(config["model"]["backbone"], "resnet18")
        self.assertFalse(config["model"]["pretrained"])
        self.assertEqual(config["train"]["max_train_steps"], 2)
        self.assertEqual(config["train"]["num_workers"], 0)

    def test_data_fraction_must_be_in_unit_interval(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["data"]["fraction"] = 0.1
        validate_config(config)
        config["data"]["fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "data.fraction"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
