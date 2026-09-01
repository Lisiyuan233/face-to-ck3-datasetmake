from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.summarize_dna_field_sweeps import sha256_file, summarize


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


class SweepSummaryTests(unittest.TestCase):
    def test_summary_merges_verified_records_and_checks_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweeps"
            output = root / "summary"
            complete_dir = root / "batch_001_gene_test"
            failed_dir = root / "batch_002_gene_failed"
            dna_lf = 'gene_test={ "test" 0 "test" 0 }\n'
            dna_hash = hashlib.sha256(dna_lf.encode("utf-8")).hexdigest()

            variants = []
            manifest = []
            for index, (value, color) in enumerate(((0, 0), (255, 255)), 1):
                variant_id = f"{index:05d}_gene_test_test_{value:03d}"
                dna_relative = f"dna/{variant_id}.txt"
                render_relative = f"renders/{variant_id}.png"
                dna_path = complete_dir / dna_relative
                render_path = complete_dir / render_relative
                dna_path.parent.mkdir(parents=True, exist_ok=True)
                # Exercise universal-newline normalization used by session hashes.
                dna_path.write_bytes(dna_lf.replace("\n", "\r\n").encode("utf-8"))
                render_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 4), (color, color, color)).save(render_path)
                variants.append(
                    {
                        "index": index,
                        "variant_id": variant_id,
                        "field": "gene_test",
                        "allele": "test",
                        "value": value,
                        "dna_sha256": dna_hash,
                    }
                )
                manifest.append(
                    {
                        **variants[-1],
                        "status": "completed",
                        "completed_at": f"2026-01-01T00:00:0{index}+00:00",
                        "dna_path": dna_relative,
                        "render_path": render_relative,
                        "render_sha256": sha256_file(render_path),
                    }
                )

            write_json(
                complete_dir / "session.json",
                {
                    "field": "gene_test",
                    "variants": variants,
                    "automation": {"verify_copy_button": [1, 2]},
                },
            )
            write_jsonl(complete_dir / "manifest.jsonl", manifest)

            failed_variant = {
                "index": 1,
                "variant_id": "00001_gene_failed_failed_000",
                "field": "gene_failed",
                "allele": "failed",
                "value": 0,
                "dna_sha256": "unused",
            }
            write_json(
                failed_dir / "session.json",
                {
                    "field": "gene_failed",
                    "variants": [failed_variant],
                    "automation": {"verify_copy_button": [1, 2]},
                },
            )
            write_jsonl(
                failed_dir / "errors.jsonl",
                [
                    {
                        **failed_variant,
                        "status": "failed",
                        "failed_at": "2026-01-01T00:00:03+00:00",
                        "error": "RuntimeError: rejected",
                    }
                ],
            )
            write_json(
                root / "batch.json",
                {
                    "fields": [
                        {"index": 1, "field": "gene_test"},
                        {"index": 2, "field": "gene_failed"},
                    ]
                },
            )

            result = summarize(root, output)

            self.assertEqual(result["field_count"], 2)
            self.assertEqual(result["expected_variants"], 3)
            self.assertEqual(result["completed_variants"], 2)
            self.assertEqual(result["round_trip_verified_variants"], 2)
            self.assertEqual(result["integrity_error_count"], 0)
            self.assertEqual(
                result["field_status_counts"],
                {"complete": 1, "partial": 0, "failed": 1, "missing": 0},
            )
            complete = result["fields"][0]
            self.assertEqual(complete["visual_status"], "all_steps_distinct")
            self.assertEqual(
                complete["strongest_endpoint_metric"]["whole_percent"],
                100.0,
            )
            self.assertTrue((output / "field_summary.json").is_file())
            self.assertTrue((output / "variant_summary.jsonl").is_file())
            self.assertTrue((output / "field_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
