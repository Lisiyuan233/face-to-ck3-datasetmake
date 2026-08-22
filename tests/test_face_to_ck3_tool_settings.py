from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_to_ck3_tool import FaceToCK3Tool, SETTINGS_FILENAME


class FaceToCK3ToolSettingsTests(unittest.TestCase):
    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = FaceToCK3Tool(temporary)
            tool.update_settings(
                region=(10, 20, 1245, 829),
                copy_dna_button_pos=(100, 200),
                random_generate_button_pos=(300, 400),
                race_button_pos=(40, 80),
                race_first_option_pos=(1900, 210),
                race_second_option_pos=(1900, 263),
                show_hair_beard_checkbox_pos=(62, 814),
                facial_structure_button_pos=(220, 452),
                clipboard_delay=0.2,
                clipboard_timeout=4.5,
                ui_update_delay=2.0,
                stability_check_delay=0.75,
                stability_timeout=9.0,
                screenshot_delay=0.8,
                randomize_retries=5,
                sample_retries=3,
                render_check_delay=0.6,
                render_stability_timeout=10.0,
                render_stability_threshold=2.5,
                render_min_contrast=36.0,
                render_min_quality_ratio=0.72,
                auto_switch_race=True,
                race_group_size=30000,
                race_count=17,
                default_count=2500,
            )

            restored = FaceToCK3Tool(temporary)
            self.assertEqual(restored.region, (10, 20, 1245, 829))
            self.assertEqual(restored.copy_dna_button_pos, (100, 200))
            self.assertEqual(restored.random_generate_button_pos, (300, 400))
            self.assertEqual(restored.race_button_pos, (40, 80))
            self.assertEqual(restored.race_first_option_pos, (1900, 210))
            self.assertEqual(restored.race_second_option_pos, (1900, 263))
            self.assertEqual(restored.show_hair_beard_checkbox_pos, (62, 814))
            self.assertEqual(restored.facial_structure_button_pos, (220, 452))
            self.assertEqual(restored.clipboard_delay, 0.2)
            self.assertEqual(restored.clipboard_timeout, 4.5)
            self.assertEqual(restored.ui_update_delay, 2.0)
            self.assertEqual(restored.stability_check_delay, 0.75)
            self.assertEqual(restored.stability_timeout, 9.0)
            self.assertEqual(restored.screenshot_delay, 0.8)
            self.assertEqual(restored.randomize_retries, 5)
            self.assertEqual(restored.sample_retries, 3)
            self.assertEqual(restored.render_check_delay, 0.6)
            self.assertEqual(restored.render_stability_timeout, 10.0)
            self.assertEqual(restored.render_stability_threshold, 2.5)
            self.assertEqual(restored.render_min_contrast, 36.0)
            self.assertEqual(restored.render_min_quality_ratio, 0.72)
            self.assertTrue(restored.auto_switch_race)
            self.assertEqual(restored.race_group_size, 30000)
            self.assertEqual(restored.race_count, 17)
            self.assertEqual(restored.default_count, 2500)
            self.assertIsNone(restored.settings_load_error)
            self.assertFalse(
                (Path(temporary) / f"{SETTINGS_FILENAME}.partial").exists()
            )

    def test_corrupt_settings_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_path = Path(temporary) / SETTINGS_FILENAME
            settings_path.write_text("{not valid json", encoding="utf-8")

            tool = FaceToCK3Tool(temporary)

            self.assertIsNone(tool.region)
            self.assertIsNone(tool.copy_dna_button_pos)
            self.assertEqual(tool.clipboard_timeout, 3.0)
            self.assertEqual(tool.default_count, 1000)
            self.assertFalse(tool.auto_switch_race)
            self.assertIsNotNone(tool.settings_load_error)
            self.assertEqual(
                settings_path.read_text(encoding="utf-8"), "{not valid json"
            )

    def test_invalid_update_is_not_applied_or_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = FaceToCK3Tool(temporary)
            tool.update_settings(clipboard_timeout=4.0)
            saved = json.loads(
                (Path(temporary) / SETTINGS_FILENAME).read_text(encoding="utf-8")
            )

            with self.assertRaisesRegex(ValueError, "clipboard_timeout"):
                tool.update_settings(clipboard_timeout=0)

            self.assertEqual(tool.clipboard_timeout, 4.0)
            self.assertEqual(
                json.loads(
                    (Path(temporary) / SETTINGS_FILENAME).read_text(
                        encoding="utf-8"
                    )
                ),
                saved,
            )

            with self.assertRaisesRegex(ValueError, "randomize_retries"):
                tool.update_settings(randomize_retries=1.5)
            self.assertEqual(tool.randomize_retries, 4)

    def test_version_one_settings_migrate_with_race_switch_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_path = Path(temporary) / SETTINGS_FILENAME
            settings_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "region": [10, 20, 30, 40],
                        "copy_dna_button_pos": [50, 60],
                        "random_generate_button_pos": [70, 80],
                    }
                ),
                encoding="utf-8",
            )

            tool = FaceToCK3Tool(temporary)

            self.assertEqual(tool.region, (10, 20, 30, 40))
            self.assertFalse(tool.auto_switch_race)
            self.assertIsNone(tool.race_button_pos)
            self.assertIsNone(tool.settings_load_error)


if __name__ == "__main__":
    unittest.main()
