from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dna_normalizer import parse_dna
from run_identifiability_experiment import load_plan, run_experiment
from targeted_sample_collector import (
    assign_base_splits,
    collection_status,
    prepare_targeted_protocol,
    select_diverse_bases,
)
from tests.test_build_identifiability_variants import dna_value, schema_value


def training_schema_value() -> dict:
    return {
        "schema_version": 2,
        "sample_count": 4,
        "scalar_fields": [
            {
                "name": "gene_a",
                "alleles": ["a_neg", "a_pos"],
                "canonical_allele": "a_pos",
            }
        ],
        "signed_fields": [],
        "categorical_fields": [
            {
                "name": "gene_b",
                "classes": ["b_one", "b_two"],
                "prediction_strategy": "strength_only",
            }
        ],
        "color_fields": [{"name": "skin_color"}],
    }


class FakeBackend:
    def __init__(self) -> None:
        self.applied: list[tuple[str, ...]] = []
        self.captured: list[Path] = []

    def apply_dna(self, dna_text: str, field) -> None:
        self.applied.append(tuple(field))

    def capture(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"render-{len(self.captured)}".encode("ascii"))
        self.captured.append(path)


class TargetedSampleCollectorTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, dict[str, str]]:
        source_schema = root / "source_schema.json"
        training_schema = root / "training_schema.json"
        source_schema.write_text(json.dumps(schema_value()), encoding="utf-8")
        training_schema.write_text(
            json.dumps(training_schema_value()), encoding="utf-8"
        )
        dna_dir = root / "dna"
        dna_dir.mkdir()
        bases = []
        base_texts = {}
        for index in range(1, 5):
            sample_id = f"face_{index:04d}"
            text = dna_value(60 + index, 30 + index)
            path = dna_dir / f"{sample_id}.txt"
            path.write_text(text, encoding="utf-8")
            base_id = f"base_{index:03d}"
            base_texts[base_id] = text
            bases.append(
                {
                    "base_dna_id": base_id,
                    "sample_id": sample_id,
                    "dna_path": path.relative_to(root).as_posix(),
                }
            )
        bases_manifest = root / "bases.jsonl"
        bases_manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in bases),
            encoding="utf-8",
        )
        return source_schema, training_schema, bases_manifest, base_texts

    def test_assigns_exact_deterministic_base_splits(self) -> None:
        base_ids = [f"base_{index:03d}" for index in range(1, 33)]
        first = assign_base_splits(base_ids, (0.75, 0.125, 0.125), 123)
        second = assign_base_splits(list(reversed(base_ids)), (0.75, 0.125, 0.125), 123)
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "train" for value in first.values()), 24)
        self.assertEqual(sum(value == "val" for value in first.values()), 4)
        self.assertEqual(sum(value == "test" for value in first.values()), 4)

    def test_selects_diverse_bases_deterministically_and_honors_exclusions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dna_dir = root / "dna"
            dna_dir.mkdir()
            labels = root / "labels.jsonl"
            rows = []
            for index, value in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 1):
                sample_id = f"face_{index:04d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scalar": [value],
                        "signed": [value * 2.0 - 1.0],
                        "categorical_class": [index % 2],
                        "categorical_strength": [1.0 - value],
                        "colors": [value, 1.0 - value],
                    }
                )
                (dna_dir / f"{sample_id}.txt").write_text(
                    dna_value(60 + index, 30 + index), encoding="utf-8"
                )
            labels.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            first = select_diverse_bases(
                labels,
                dna_dir,
                root / "selected_a.jsonl",
                count=3,
                included_sample_ids={
                    "face_0001",
                    "face_0002",
                    "face_0003",
                    "face_0004",
                    "face_0005",
                    "face_0006",
                },
                excluded_sample_ids={"face_0006"},
                progress_every=0,
            )
            second = select_diverse_bases(
                labels,
                dna_dir,
                root / "selected_b.jsonl",
                count=3,
                included_sample_ids={
                    "face_0001",
                    "face_0002",
                    "face_0003",
                    "face_0004",
                    "face_0005",
                    "face_0006",
                },
                excluded_sample_ids={"face_0006"},
                progress_every=0,
            )

            self.assertEqual(
                [row["sample_id"] for row in first],
                [row["sample_id"] for row in second],
            )
            self.assertEqual(len({row["sample_id"] for row in first}), 3)
            self.assertNotIn("face_0006", {row["sample_id"] for row in first})
            self.assertEqual(first[0]["selection_method"], "closest_to_component_median")
            self.assertTrue(
                all(
                    row["selection_method"] == "farthest_point_maximin"
                    for row in first[1:]
                )
            )
            self.assertTrue(
                all((root / row["dna_path"]).is_file() for row in first)
            )

            with self.assertRaisesRegex(ValueError, "少于 count"):
                select_diverse_bases(
                    labels,
                    dna_dir,
                    root / "too_many.jsonl",
                    count=6,
                    excluded_sample_ids={"face_0006"},
                    progress_every=0,
                )

    def test_prepares_only_selected_fields_with_training_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, training, bases_manifest, base_texts = self._fixture(root)
            experiment = root / "experiment"
            protocol = prepare_targeted_protocol(
                source,
                training,
                bases_manifest,
                experiment,
                fields=["gene_a", "gene_b"],
                strengths=[0, 255],
                baseline_repeats=2,
                split_ratios=(0.5, 0.25, 0.25),
                split_seed=99,
            )

            self.assertEqual(protocol["protocol_kind"], "targeted_sample_collection")
            self.assertEqual(protocol["base_split_counts"], {"train": 2, "val": 1, "test": 1})
            self.assertEqual(protocol["training_eligible_variants"], 32)
            self.assertEqual(protocol["total_variants"], 40)
            rows = [
                json.loads(line)
                for line in (experiment / "variants.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len({row["sample_id"] for row in rows}), 40)
            for base_id, base_text in base_texts.items():
                base_rows = [row for row in rows if row["base_dna_id"] == base_id]
                self.assertEqual(len({row["base_split"] for row in base_rows}), 1)
                original = parse_dna(base_text)
                for row in base_rows:
                    variant = parse_dna(
                        (experiment / row["dna_path"]).read_text(encoding="utf-8")
                    )
                    if row["kind"] == "baseline":
                        self.assertFalse(row["training_eligible"])
                        self.assertEqual(row["loss_mask"], [])
                        self.assertEqual(variant, original)
                        continue
                    self.assertEqual(row["source_type"], "targeted_intervention")
                    self.assertEqual(row["intervention_field"], row["field"])
                    self.assertTrue(row["training_eligible"])
                    self.assertEqual(
                        row["loss_mask"],
                        [{"family": row["target_family"], "field": row["field"]}],
                    )
                    changed = {
                        name
                        for name in original.genes
                        if original.genes[name] != variant.genes[name]
                    }
                    self.assertLessEqual(changed, {row["field"]})
                    self.assertEqual(variant.colors, original.colors)

            repeated = prepare_targeted_protocol(
                source,
                training,
                bases_manifest,
                experiment,
                fields=["gene_a", "gene_b"],
                strengths=[0, 255],
                baseline_repeats=2,
                split_ratios=(0.5, 0.25, 0.25),
                split_seed=99,
            )
            self.assertEqual(repeated["plan_sha256"], protocol["plan_sha256"])
            with self.assertRaisesRegex(RuntimeError, "不同的 protocol"):
                prepare_targeted_protocol(
                    source,
                    training,
                    bases_manifest,
                    experiment,
                    fields=["gene_a"],
                    strengths=[0, 255],
                    baseline_repeats=2,
                    split_ratios=(0.5, 0.25, 0.25),
                    split_seed=99,
                )

    def test_runner_propagates_targeted_metadata_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, training, bases_manifest, _base_texts = self._fixture(root)
            experiment = root / "experiment"
            prepare_targeted_protocol(
                source,
                training,
                bases_manifest,
                experiment,
                fields=["gene_a"],
                strengths=[0, 255],
                baseline_repeats=1,
                split_ratios=(0.5, 0.25, 0.25),
                split_seed=9,
            )
            protocol, variants = load_plan(experiment)
            selected = variants[:3]
            backend = FakeBackend()
            result = run_experiment(
                experiment,
                protocol,
                selected,
                backend,
                retries=0,
                verification_fields=protocol["verification_fields"],
            )
            self.assertEqual(result.completed, 3)
            manifest = [
                json.loads(line)
                for line in (experiment / "render_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            baseline_row = next(row for row in manifest if row["kind"] == "baseline")
            self.assertEqual(baseline_row["source_type"], "targeted_baseline")
            self.assertFalse(baseline_row["training_eligible"])
            field_row = next(row for row in manifest if row["kind"] == "field")
            self.assertEqual(field_row["base_dna_id"], field_row["base_id"])
            self.assertEqual(field_row["intervention_field"], "gene_a")
            self.assertEqual(
                field_row["loss_mask"], [{"family": "scalar", "field": "gene_a"}]
            )
            status = collection_status(experiment)
            self.assertEqual(status["completed"], 3)
            self.assertEqual(status["remaining"], len(variants) - 3)

    def test_rejects_partially_explicit_base_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, training, bases_manifest, _base_texts = self._fixture(root)
            rows = [json.loads(line) for line in bases_manifest.read_text().splitlines()]
            rows[0]["split"] = "train"
            bases_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "全部填写或全部省略"):
                prepare_targeted_protocol(
                    source,
                    training,
                    bases_manifest,
                    root / "experiment",
                    fields=["gene_a"],
                    strengths=[0, 255],
                    baseline_repeats=1,
                )


if __name__ == "__main__":
    unittest.main()
