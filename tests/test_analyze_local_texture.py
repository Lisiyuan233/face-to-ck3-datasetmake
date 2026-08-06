import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_local_texture import analyze_local_texture, texture_distances


class LocalTextureAnalysisTests(unittest.TestCase):
    def test_identical_patches_have_zero_distance(self) -> None:
        patch = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        measured = texture_distances(patch, patch.copy())
        self.assertEqual(measured["ssim"], 0.0)
        self.assertEqual(measured["edge"], 0.0)
        self.assertEqual(measured["highpass"], 0.0)

    def test_completed_sweep_detects_local_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            renders = root / "renders" / "base_00"
            renders.mkdir(parents=True)
            records = []

            def save(variant_id: str, strength: int) -> tuple[str, str]:
                height, width = 96, 160
                x = np.arange(width, dtype=np.uint8)[None, :]
                y = np.arange(height, dtype=np.uint8)[:, None]
                image = np.asarray(70 + x // 5 + y // 6, dtype=np.uint8).copy()
                if strength:
                    checker = (np.indices((24, 30)).sum(axis=0) % 2) * (
                        35 if strength == 255 else 18
                    )
                    image[50:74, 22:52] = np.clip(
                        image[50:74, 22:52].astype(np.int16) + checker,
                        0,
                        255,
                    ).astype(np.uint8)
                path = renders / f"{variant_id}.png"
                Image.fromarray(image, mode="L").save(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return path.relative_to(root).as_posix(), digest

            for repeat in range(1, 4):
                variant_id = f"baseline_{repeat}"
                render_path, digest = save(variant_id, 0)
                records.append(
                    {
                        "variant_id": variant_id,
                        "base_id": "base_00",
                        "kind": "baseline",
                        "field": None,
                        "field_type": None,
                        "class_or_sign": "baseline",
                        "strength": None,
                        "baseline_repeat": repeat,
                        "status": "completed",
                        "render_path": render_path,
                        "render_sha256": digest,
                    }
                )
            for sign in ("negative", "positive"):
                for strength in (0, 128, 255):
                    variant_id = f"texture_{sign}_{strength}"
                    render_path, digest = save(variant_id, strength)
                    records.append(
                        {
                            "variant_id": variant_id,
                            "base_id": "base_00",
                            "kind": "field",
                            "field": "face_detail_nasolabial",
                            "field_type": "signed",
                            "class_or_sign": sign,
                            "strength": strength,
                            "baseline_repeat": None,
                            "status": "completed",
                            "render_path": render_path,
                            "render_sha256": digest,
                        }
                    )

            (root / "protocol.json").write_text(
                json.dumps({"total_variants": len(records)}), encoding="utf-8"
            )
            (root / "render_manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = analyze_local_texture(
                root, size=(160, 96), save_heatmaps=False
            )
            self.assertEqual(summary["field_count"], 1)
            self.assertEqual(summary["tier_counts"]["T1"], 1)
            with (root / "local_texture_identifiability.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["local_region"], "mouth")
            self.assertEqual(row["texture_tier"], "T1")
            self.assertGreater(float(row["texture_snr"]), 1.0)


if __name__ == "__main__":
    unittest.main()
