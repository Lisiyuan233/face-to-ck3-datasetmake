from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_identifiability_variants import BaseCandidate, build_protocol
from run_identifiability_experiment import load_plan, run_experiment
from tests.test_build_identifiability_variants import dna_value, schema_value


class FakeBackend:
    def __init__(self) -> None:
        self.applied: list[tuple[str, str | None]] = []
        self.captured: list[Path] = []

    def apply_dna(self, dna_text: str, field: str | None) -> None:
        self.applied.append((dna_text, field))

    def capture(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"render-{len(self.captured)}".encode("ascii"))
        self.captured.append(path)


class IdentifiabilityRunnerTests(unittest.TestCase):
    def test_runner_uses_full_round_trip_and_resumes_valid_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema_value()), encoding="utf-8")
            base_path = root / "face_0001.txt"
            base_path.write_text(dna_value(), encoding="utf-8")
            experiment = root / "experiment"
            build_protocol(
                schema_path,
                [
                    BaseCandidate(
                        race_group=0,
                        sample_id="face_0001",
                        source_dna_path=base_path,
                        selection_method="test",
                    )
                ],
                experiment,
                strengths=[0, 255],
                baseline_repeats=2,
            )
            protocol, variants = load_plan(experiment)
            selected = variants[:3]
            backend = FakeBackend()
            first = run_experiment(
                experiment, protocol, selected, backend, retries=1
            )
            self.assertEqual(first.completed, 3)
            self.assertEqual(first.skipped, 0)
            self.assertEqual(first.attempted, 3)
            self.assertEqual([field for _dna, field in backend.applied], [None, None, None])

            resumed_backend = FakeBackend()
            resumed = run_experiment(
                experiment, protocol, selected, resumed_backend, retries=1
            )
            self.assertEqual(resumed.completed, 3)
            self.assertEqual(resumed.skipped, 3)
            self.assertEqual(resumed.attempted, 0)
            self.assertEqual(resumed_backend.applied, [])
            manifest_rows = [
                json.loads(line)
                for line in (experiment / "render_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(manifest_rows), 3)
            self.assertTrue(
                all(row["round_trip_scope"] == "full_parsed_dna" for row in manifest_rows)
            )


if __name__ == "__main__":
    unittest.main()
