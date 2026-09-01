from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from train import load_finetune_weights


class TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("counter", torch.tensor([0.0]))
        self.use_side_view = True
        self.use_geometry_branch = True
        self.geometry_targets = frozenset(("scalar", "signed"))


def checkpoint_for() -> dict:
    raw = TinyCheckpointModel()
    raw.weight.data.fill_(1.0)
    raw.counter.fill_(7.0)
    return {
        "epoch": 12,
        "global_step": 345,
        "model": raw.state_dict(),
        "ema": {"decay": 0.99, "shadow": {"weight": torch.tensor([2.0])}},
        "config": {
            "model": {
                "side_view": True,
                "geometry_branch": {
                    "enabled": True,
                    "targets": ["scalar", "signed"],
                },
            }
        },
        "schema": {"schema_sha256": "schema", "target_family": "family"},
        "split_index_sha256": None,
    }


class FinetuneCheckpointTests(unittest.TestCase):
    def _save(self, value: dict, root: Path) -> Path:
        path = root / "checkpoint.pt"
        torch.save(value, path)
        return path

    def test_finetune_uses_ema_parameters_and_raw_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = TinyCheckpointModel()
            path = self._save(checkpoint_for(), root)
            metadata = load_finetune_weights(
                path,
                model=model,
                schema_sha256="schema",
                target_family="family",
                device=torch.device("cpu"),
            )
            self.assertEqual(model.weight.item(), 2.0)
            self.assertEqual(model.counter.item(), 7.0)
            self.assertEqual(metadata["weight_source"], "ema")
            self.assertEqual(metadata["source_epoch"], 12)

    def test_finetune_can_explicitly_use_raw_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = TinyCheckpointModel()
            path = self._save(checkpoint_for(), root)
            metadata = load_finetune_weights(
                path,
                model=model,
                schema_sha256="schema",
                target_family="family",
                device=torch.device("cpu"),
                raw_weights=True,
            )
            self.assertEqual(model.weight.item(), 1.0)
            self.assertEqual(metadata["weight_source"], "raw")

    def test_finetune_rejects_incompatible_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = TinyCheckpointModel()
            path = self._save(checkpoint_for(), root)
            with self.assertRaisesRegex(RuntimeError, "schema"):
                load_finetune_weights(
                    path,
                    model=model,
                    schema_sha256="different",
                    target_family="family",
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
