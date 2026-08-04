from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dna_field_sweep_tool import (
    AutomationConfig,
    WindowsAutomationBackend,
    build_sweep_variants,
    parse_allele_sequence,
    parse_value_sequence,
    prepare_session,
    replace_gene_pair,
    run_sweep,
)
from dna_normalizer import parse_dna


BASE_DNA = """ruler_designer_123={
    genes={
        gene_test={ "test_neg" 10 "test_neg" 10 }
        gene_other={ "other_pos" 99 "other_pos" 99 }
        skin_color={ 20 30 20 30 }
    }
}
"""


class FakeBackend:
    def __init__(self) -> None:
        self.applied: list[tuple[str, int]] = []
        self.captured: list[Path] = []

    def apply_dna(self, dna_text: str, field: str) -> None:
        gene = parse_dna(dna_text).genes[field]
        self.applied.append((gene.allele1, gene.value1))

    def capture(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-png")
        self.captured.append(path)


class DnaFieldSweepTests(unittest.TestCase):
    def test_value_sequence_supports_list_and_inclusive_range(self) -> None:
        self.assertEqual(parse_value_sequence("0, 64；128 255"), [0, 64, 128, 255])
        self.assertEqual(
            parse_value_sequence("0:255:64"),
            [0, 64, 128, 192, 255],
        )
        self.assertEqual(parse_value_sequence("255:0:-128"), [255, 127, 0])
        with self.assertRaisesRegex(ValueError, "0..255"):
            parse_value_sequence("0,256")

    def test_alleles_are_deduplicated_and_validated(self) -> None:
        self.assertEqual(
            parse_allele_sequence("test_neg,test_pos,test_neg", "unused"),
            ["test_neg", "test_pos"],
        )
        self.assertEqual(parse_allele_sequence("", "test_neg"), ["test_neg"])
        with self.assertRaisesRegex(ValueError, "allele"):
            parse_allele_sequence('bad"allele', "test_neg")

    def test_replace_gene_pair_changes_only_selected_gene(self) -> None:
        original = parse_dna(BASE_DNA)
        output = replace_gene_pair(BASE_DNA, "gene_test", "test_pos", 192)
        changed = parse_dna(output)
        self.assertEqual(changed.genes["gene_test"].allele1, "test_pos")
        self.assertEqual(changed.genes["gene_test"].allele2, "test_pos")
        self.assertEqual(changed.genes["gene_test"].value1, 192)
        self.assertEqual(changed.genes["gene_test"].value2, 192)
        self.assertEqual(changed.genes["gene_other"], original.genes["gene_other"])
        self.assertEqual(changed.colors, original.colors)

    def test_build_variants_crosses_alleles_and_values(self) -> None:
        variants = build_sweep_variants(
            BASE_DNA,
            "gene_test",
            ["test_neg", "test_pos"],
            [0, 255],
        )
        self.assertEqual(len(variants), 4)
        self.assertEqual(
            [(variant.allele, variant.value) for variant in variants],
            [
                ("test_neg", 0),
                ("test_neg", 255),
                ("test_pos", 0),
                ("test_pos", 255),
            ],
        )
        self.assertEqual(len({variant.variant_id for variant in variants}), 4)

    def test_runner_saves_manifest_and_resumes_completed_variants(self) -> None:
        variants = build_sweep_variants(
            BASE_DNA,
            "gene_test",
            ["test_neg"],
            [0, 128, 255],
        )
        config = AutomationConfig(
            paste_button=(10, 20),
            confirm_button=(30, 40),
            screenshot_region=(30, 40, 500, 300),
            retries=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "session"
            session = prepare_session(session_dir, BASE_DNA, variants, config)
            self.assertEqual(session["variant_count"], 3)

            first_backend = FakeBackend()
            first = run_sweep(
                variants,
                session_dir,
                first_backend,
                retries=0,
            )
            self.assertEqual(first.completed, 3)
            self.assertEqual(first.skipped, 0)
            self.assertEqual(first_backend.applied, [("test_neg", 0), ("test_neg", 128), ("test_neg", 255)])

            records = [
                json.loads(line)
                for line in (session_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 3)
            self.assertTrue(
                all((session_dir / record["render_path"]).is_file() for record in records)
            )

            resumed_backend = FakeBackend()
            resumed = run_sweep(
                variants,
                session_dir,
                resumed_backend,
                retries=0,
            )
            self.assertEqual(resumed.completed, 3)
            self.assertEqual(resumed.skipped, 3)
            self.assertEqual(resumed_backend.applied, [])

    def test_runner_stops_after_three_identical_unverified_renders(self) -> None:
        variants = build_sweep_variants(
            BASE_DNA,
            "gene_test",
            ["test_neg"],
            [0, 128, 255],
        )
        config = AutomationConfig(
            paste_button=(10, 20),
            confirm_button=(30, 40),
            screenshot_region=(30, 40, 500, 300),
            retries=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "session"
            prepare_session(session_dir, BASE_DNA, variants, config)

            with self.assertRaisesRegex(RuntimeError, "连续 3 张截图完全相同"):
                run_sweep(
                    variants,
                    session_dir,
                    FakeBackend(),
                    retries=0,
                    identical_render_limit=3,
                )

            records = [
                json.loads(line)
                for line in (session_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["render_sha256"] for record in records))
            errors = [
                json.loads(line)
                for line in (session_dir / "errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("疑似后续粘贴未生效", errors[-1]["error"])

    def test_confirmation_button_is_required(self) -> None:
        config = AutomationConfig(
            paste_button=(10, 20),
            screenshot_region=(30, 40, 500, 300),
        )
        with self.assertRaisesRegex(ValueError, "确定"):
            config.validate()

    def test_backend_clicks_paste_then_confirmation(self) -> None:
        class FakePyAutoGui:
            class FailSafeException(Exception):
                pass

            def __init__(self) -> None:
                self.position = (0, 0)
                self.clicks: list[tuple[int, int]] = []

            def moveTo(self, x: int, y: int, *, duration: float) -> None:
                self.position = (x, y)

            def mouseDown(self) -> None:
                pass

            def mouseUp(self) -> None:
                self.clicks.append(self.position)

        class FakeClipboard:
            def copy(self, _value: str) -> None:
                pass

        config = AutomationConfig(
            paste_button=(10, 20),
            confirm_button=(30, 40),
            screenshot_region=(0, 0, 500, 300),
            clipboard_delay=0,
            confirm_delay=0,
            settle_delay=0,
            screenshot_delay=0,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
            inter_variant_delay=0,
        )
        backend = object.__new__(WindowsAutomationBackend)
        backend.config = config
        backend.pyautogui = FakePyAutoGui()
        backend.pyperclip = FakeClipboard()

        with patch("dna_field_sweep_tool.time.sleep"):
            backend.apply_dna(BASE_DNA, "gene_test")

        self.assertEqual(backend.pyautogui.clicks, [(10, 20), (30, 40)])

    def test_verification_polls_until_game_updates_clipboard(self) -> None:
        class FakePyAutoGui:
            class FailSafeException(Exception):
                pass

            def __init__(self) -> None:
                self.position = (0, 0)
                self.clicks: list[tuple[int, int]] = []

            def moveTo(self, x: int, y: int, *, duration: float) -> None:
                self.position = (x, y)

            def mouseDown(self) -> None:
                pass

            def mouseUp(self) -> None:
                self.clicks.append(self.position)

        class DelayedClipboard:
            def __init__(self) -> None:
                self.value = ""
                self.paste_count = 0

            def copy(self, value: str) -> None:
                self.value = value
                if value.startswith("CK3_DNA_VERIFY_"):
                    self.paste_count = 0

            def paste(self) -> str:
                self.paste_count += 1
                if self.value.startswith("CK3_DNA_VERIFY_") and self.paste_count >= 3:
                    return BASE_DNA
                return self.value

        config = AutomationConfig(
            paste_button=(10, 20),
            confirm_button=(30, 40),
            verify_copy_button=(50, 60),
            screenshot_region=(0, 0, 500, 300),
            clipboard_delay=0,
            confirm_delay=0,
            settle_delay=0,
            screenshot_delay=0,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
            verification_timeout=1,
            inter_variant_delay=0,
        )
        backend = object.__new__(WindowsAutomationBackend)
        backend.config = config
        backend.pyautogui = FakePyAutoGui()
        backend.pyperclip = DelayedClipboard()

        backend.apply_dna(BASE_DNA, "gene_test")

        self.assertEqual(
            backend.pyautogui.clicks,
            [(10, 20), (30, 40), (50, 60)],
        )
        self.assertGreaterEqual(backend.pyperclip.paste_count, 3)


if __name__ == "__main__":
    unittest.main()
