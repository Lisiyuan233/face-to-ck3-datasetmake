from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_identifiability_variants import (
    BaseCandidate,
    build_protocol,
    interleave_baselines,
    select_median_bases,
)
from dna_normalizer import parse_dna


def schema_value() -> dict:
    return {
        "schema_version": 1,
        "sample_count": 6,
        "normalization": {
            "byte_divisor": 255,
            "signed_range": [-1.0, 1.0],
            "strength_range": [0.0, 1.0],
            "color_range": [0.0, 1.0],
        },
        "signed_fields": [
            {
                "name": "gene_a",
                "negative_allele": "a_neg",
                "positive_allele": "a_pos",
                "zero_allele": "a_neg",
            }
        ],
        "categorical_fields": [
            {
                "name": "gene_b",
                "classes": ["b_one", "b_two"],
                "class_counts": [3, 3],
            }
        ],
        "color_fields": [{"name": "skin_color"}],
    }


def dna_value(a_value: int = 64, b_value: int = 32) -> str:
    return (
        'ruler_designer_100={\n'
        f'    gene_a={{ "a_neg" {a_value} "a_neg" {a_value} }}\n'
        f'    gene_b={{ "b_one" {b_value} "b_one" {b_value} }}\n'
        "    skin_color={ 10 20 10 20 }\n"
        "}\n"
    )


class IdentifiabilityBuilderTests(unittest.TestCase):
    def test_baselines_are_evenly_interleaved(self) -> None:
        fields = [{"kind": "field", "index": index} for index in range(12)]
        scheduled = interleave_baselines(fields, "base", 5)
        baseline_positions = [
            index for index, row in enumerate(scheduled) if row["kind"] == "baseline"
        ]
        self.assertEqual(baseline_positions, [0, 4, 8, 12, 16])
        self.assertEqual(len(scheduled), 17)
        self.assertEqual(
            [row["baseline_repeat"] for row in scheduled if row["kind"] == "baseline"],
            [1, 2, 3, 4, 5],
        )

    def test_protocol_expands_schema_and_changes_only_target_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema_value()), encoding="utf-8")
            candidates = []
            base_texts = {}
            for group in range(2):
                source = root / f"face_{group + 1:04d}.txt"
                text = dna_value(64 + group, 32 + group)
                source.write_text(text, encoding="utf-8")
                base_texts[group] = text
                candidates.append(
                    BaseCandidate(
                        race_group=group,
                        sample_id=f"face_{group + 1:04d}",
                        source_dna_path=source,
                        selection_method="test",
                    )
                )

            output = root / "experiment"
            protocol = build_protocol(
                schema_path,
                candidates,
                output,
                strengths=[0, 128, 255],
                baseline_repeats=5,
            )
            self.assertEqual(protocol["signed_variants_per_base"], 6)
            self.assertEqual(protocol["categorical_variants_per_base"], 6)
            self.assertEqual(protocol["field_variants_per_base"], 12)
            self.assertEqual(protocol["variants_per_base"], 17)
            self.assertEqual(protocol["total_variants"], 34)
            self.assertEqual(
                protocol["verification_policy"],
                "schema_fields_and_colors_round_trip_required",
            )
            self.assertEqual(protocol["verification_fields"], ["gene_a", "gene_b"])

            rows = [
                json.loads(line)
                for line in (output / "variants.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            for group in range(2):
                group_rows = [row for row in rows if row["race_group"] == group]
                self.assertEqual(sum(row["kind"] == "baseline" for row in group_rows), 5)
                self.assertEqual(group_rows[0]["kind"], "baseline")
                self.assertEqual(group_rows[-1]["kind"], "baseline")
                base_record = parse_dna(base_texts[group])
                for row in group_rows:
                    variant = parse_dna(
                        (output / row["dna_path"]).read_text(encoding="utf-8")
                    )
                    if row["kind"] == "baseline":
                        self.assertEqual(variant, base_record)
                        continue
                    changed = {
                        key
                        for key in base_record.genes
                        if base_record.genes[key] != variant.genes[key]
                    }
                    # Strength/class combinations may equal the original target
                    # value, but no non-target field may ever change.
                    self.assertLessEqual(changed, {row["field"]})
                    self.assertEqual(variant.colors, base_record.colors)

            # The same immutable plan is safe to regenerate/resume.
            repeated = build_protocol(
                schema_path,
                candidates,
                output,
                strengths=[0, 128, 255],
                baseline_repeats=5,
            )
            self.assertEqual(repeated["plan_sha256"], protocol["plan_sha256"])
            with self.assertRaisesRegex(RuntimeError, "不同的 protocol"):
                build_protocol(
                    schema_path,
                    candidates,
                    output,
                    strengths=[0, 255],
                    baseline_repeats=5,
                )

    def test_selects_closest_real_sample_to_each_group_median(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.jsonl"
            dna_dir = root / "dna"
            dna_dir.mkdir()
            labels = []
            signed_values = [-1.0, 0.0, 1.0, -0.5, 0.0, 0.5]
            for index, signed in enumerate(signed_values, 1):
                sample_id = f"face_{index:04d}"
                labels.append(
                    {
                        "sample_id": sample_id,
                        "signed": [signed],
                        "categorical_class": [0],
                        "categorical_strength": [0.5],
                        "colors": [0.25, 0.75],
                    }
                )
                (dna_dir / f"{sample_id}.txt").write_text(
                    dna_value(), encoding="utf-8"
                )
            labels_path.write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )
            candidates = select_median_bases(
                labels_path,
                dna_dir,
                schema_value(),
                group_count=2,
                group_size=3,
            )
            self.assertEqual([item.sample_id for item in candidates], ["face_0002", "face_0005"])
            self.assertEqual([item.group_sample_count for item in candidates], [3, 3])


if __name__ == "__main__":
    unittest.main()
