from __future__ import annotations

import unittest

from ck3_training.config import (
    DEFAULT_CONFIG,
    apply_smoke_overrides,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        validate_config(DEFAULT_CONFIG)

    def test_smoke_overrides_are_cpu_friendly(self) -> None:
        config = apply_smoke_overrides(DEFAULT_CONFIG)
        validate_config(config)
        self.assertEqual(config["model"]["backbone"], "resnet18")
        self.assertFalse(config["model"]["pretrained"])
        self.assertEqual(config["train"]["max_train_steps"], 2)
        self.assertEqual(config["train"]["num_workers"], 0)


if __name__ == "__main__":
    unittest.main()
