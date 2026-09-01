from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from ck3_collection import (
    CollectionConfig,
    RenderStabilityError,
    VerifiedCollector,
    discover_collection_state,
    dna_fingerprint,
)


DNA_A = """ruler_designer_1={
    genes={
        gene_test={ "test_a" 10 "test_a" 10 }
        skin_color={ 20 30 20 30 }
    }
}
"""

DNA_B = """ruler_designer_1={
    genes={
        gene_test={ "test_b" 20 "test_b" 20 }
        skin_color={ 20 30 20 30 }
    }
}
"""

DNA_C = """ruler_designer_1={
    genes={
        gene_test={ "test_c" 30 "test_c" 30 }
        skin_color={ 20 30 20 30 }
    }
}
"""


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def sleep(self, seconds: float) -> None:
        self.value += max(0.001, float(seconds))

    def monotonic(self) -> float:
        return self.value


class FakeClipboard:
    def __init__(self) -> None:
        self.value = ""

    def copy(self, value: str) -> None:
        self.value = value

    def paste(self) -> str:
        return self.value


class FakePyAutoGUI:
    class FailSafeException(Exception):
        pass

    FAILSAFE = False
    PAUSE = 0.0

    def __init__(
        self,
        clipboard: FakeClipboard,
        random_dnas: list[str],
        *,
        change_after_screenshot: str | None = None,
        render_levels: list[int] | None = None,
    ) -> None:
        self.clipboard = clipboard
        self.random_dnas = list(random_dnas)
        self.change_after_screenshot = change_after_screenshot
        self.render_levels = list(render_levels or [100])
        self.last_render_level = self.render_levels[-1]
        self.current_dna = DNA_A
        self.position = (0, 0)
        self.random_clicks = 0
        self.screenshot_count = 0
        self.click_positions: list[tuple[int, int]] = []

    def moveTo(self, x: int, y: int, *, duration: float) -> None:
        self.position = (x, y)

    def mouseDown(self) -> None:
        pass

    def mouseUp(self) -> None:
        self.click_positions.append(self.position)
        if self.position == (10, 10):
            self.clipboard.value = self.current_dna
        elif self.position == (20, 20):
            self.random_clicks += 1
            if self.random_dnas:
                self.current_dna = self.random_dnas.pop(0)

    def screenshot(self, *, region):
        self.screenshot_count += 1
        image = Image.new("RGB", (region[2], region[3]), (40, 40, 40))
        if self.render_levels:
            self.last_render_level = self.render_levels.pop(0)
        level = self.last_render_level
        if level > 40:
            image.paste(
                (level, max(40, level - 20), max(40, level - 35)),
                (4, 4, region[2] - 2, region[3] - 2),
            )
        if self.change_after_screenshot is not None:
            self.current_dna = self.change_after_screenshot
        return image


def make_collector(
    gui: FakePyAutoGUI,
    clipboard: FakeClipboard,
    clock: FakeClock,
    *,
    sample_retries: int = 0,
) -> VerifiedCollector:
    config = CollectionConfig(
        screenshot_region=(100, 100, 32, 24),
        copy_dna_button=(10, 10),
        random_generate_button=(20, 20),
        clipboard_settle_delay=0.01,
        clipboard_timeout=1.0,
        ui_settle_delay=0.01,
        stability_check_delay=0.05,
        stability_timeout=0.20,
        screenshot_delay=0,
        render_check_delay=0.01,
        render_stability_timeout=0.20,
        render_stability_threshold=2.0,
        render_min_contrast=20.0,
        post_capture_delay=0,
        inter_sample_delay=0,
        randomize_retries=3,
        sample_retries=sample_retries,
        mouse_move_duration=0,
        click_hover_delay=0,
        click_hold_delay=0,
    )
    return VerifiedCollector(
        config,
        pyautogui_module=gui,
        pyperclip_module=clipboard,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


class CK3CollectionTests(unittest.TestCase):
    def test_race_transition_uses_absolute_target_and_required_order(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [])
        config = CollectionConfig(
            screenshot_region=(100, 100, 32, 24),
            copy_dna_button=(10, 10),
            random_generate_button=(20, 20),
            auto_switch_race=True,
            race_group_size=3,
            race_count=4,
            race_button=(30, 30),
            race_first_option=(100, 100),
            race_second_option=(102, 112),
            show_hair_beard_checkbox=(40, 40),
            facial_structure_button=(50, 50),
            ui_settle_delay=0,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
        )
        events: list[str] = []
        collector = VerifiedCollector(
            config,
            pyautogui_module=gui,
            pyperclip_module=clipboard,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            on_event=events.append,
        )

        self.assertFalse(collector.prepare_race_for_sample(3))
        self.assertTrue(collector.prepare_race_for_sample(4))
        self.assertEqual(
            gui.click_positions,
            [(30, 30), (100, 112), (40, 40), (50, 50)],
        )
        self.assertIn("点击种族按钮", events)
        self.assertIn("取消显示头发与胡须", events)
        self.assertIn("打开面部结构", events)

    def test_race_transition_stops_beyond_configured_list(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [])
        config = CollectionConfig(
            screenshot_region=(100, 100, 32, 24),
            copy_dna_button=(10, 10),
            random_generate_button=(20, 20),
            auto_switch_race=True,
            race_group_size=2,
            race_count=2,
            race_button=(30, 30),
            race_first_option=(100, 100),
            race_second_option=(100, 110),
            show_hair_beard_checkbox=(40, 40),
            facial_structure_button=(50, 50),
        )
        collector = VerifiedCollector(
            config,
            pyautogui_module=gui,
            pyperclip_module=clipboard,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        with self.assertRaisesRegex(RuntimeError, "只配置了 2 个种族"):
            collector.prepare_race_for_sample(5)
        self.assertEqual(gui.click_positions, [])

    def test_duplicate_previous_dna_is_retried_before_capture(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [DNA_A, DNA_B])
        collector = make_collector(gui, clipboard, clock)

        with tempfile.TemporaryDirectory() as temporary:
            result = collector.collect_sample(temporary, 1, DNA_A)
            self.assertEqual(result.randomize_attempts, 2)
            self.assertEqual(gui.random_clicks, 2)
            self.assertEqual(gui.screenshot_count, 2)
            self.assertEqual(
                (Path(temporary) / "dna" / "face_0001.txt").read_text(
                    encoding="utf-8"
                ),
                DNA_B,
            )
            self.assertTrue(
                (Path(temporary) / "face" / "face_0001.png").is_file()
            )
            self.assertFalse(list(Path(temporary).rglob("*.partial")))

    def test_dna_change_after_capture_discards_transaction(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(
            clipboard,
            [DNA_B],
            change_after_screenshot=DNA_C,
        )
        collector = make_collector(gui, clipboard, clock)

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "截图前后的 DNA 不一致"):
                collector.collect_sample(temporary, 1, DNA_A)
            self.assertFalse(list(Path(temporary).rglob("face_0001.*")))

    def test_render_waits_for_two_matching_healthy_frames(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(
            clipboard,
            [DNA_B],
            render_levels=[90, 130, 130],
        )
        collector = make_collector(gui, clipboard, clock)

        with tempfile.TemporaryDirectory() as temporary:
            result = collector.collect_sample(temporary, 1, DNA_A)

        self.assertEqual(gui.screenshot_count, 3)
        self.assertLessEqual(result.render_difference, 2.0)
        self.assertGreaterEqual(result.render_contrast, 20.0)

    def test_healthy_animation_uses_calmest_frame_instead_of_stopping(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(
            clipboard,
            [DNA_B],
            render_levels=[90, 130] * 20,
        )
        collector = make_collector(gui, clipboard, clock)
        collector.config = replace(
            collector.config,
            render_stability_timeout=0.055,
        )

        with tempfile.TemporaryDirectory() as temporary:
            result = collector.collect_sample(temporary, 1, DNA_A)

        self.assertGreater(
            result.render_difference,
            collector.config.render_stability_threshold,
        )
        self.assertGreaterEqual(result.render_contrast, 20.0)

    def test_changed_dna_with_unchanged_render_stops_and_saves_diagnostics(
        self,
    ) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [], render_levels=[100])
        collector = make_collector(gui, clipboard, clock)
        collector.config = replace(
            collector.config,
            render_stability_timeout=0.035,
        )

        with tempfile.TemporaryDirectory() as temporary:
            face_dir = Path(temporary) / "face"
            face_dir.mkdir()
            previous = Image.new("RGB", (32, 24), (40, 40, 40))
            previous.paste((100, 80, 65), (4, 4, 30, 22))
            previous.save(face_dir / "face_0001.png")

            with self.assertRaisesRegex(
                RenderStabilityError, "仍与上一样本近似相同"
            ):
                collector.wait_for_stable_render(temporary, 2)

            diagnostics = Path(temporary) / "collection_diagnostics"
            self.assertTrue(
                (diagnostics / "render_failure_face_0002.png").is_file()
            )
            self.assertTrue(
                (diagnostics / "render_failure_face_0002.json").is_file()
            )

    def test_persistent_render_anomaly_stops_without_retry_or_write(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [DNA_B], render_levels=[40])
        config = CollectionConfig(
            screenshot_region=(100, 100, 32, 24),
            copy_dna_button=(10, 10),
            random_generate_button=(20, 20),
            clipboard_settle_delay=0.01,
            clipboard_timeout=1.0,
            ui_settle_delay=0.01,
            stability_check_delay=0.05,
            stability_timeout=0.20,
            screenshot_delay=0,
            render_check_delay=0.05,
            render_stability_timeout=0.11,
            render_stability_threshold=2.0,
            render_min_contrast=20.0,
            post_capture_delay=0,
            inter_sample_delay=0,
            randomize_retries=1,
            sample_retries=3,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
        )
        collector = VerifiedCollector(
            config,
            pyautogui_module=gui,
            pyperclip_module=clipboard,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RenderStabilityError, "自动停机"):
                collector.collect_sample(temporary, 1, DNA_A)
            self.assertFalse(list(Path(temporary).rglob("face_0001.*")))

        self.assertEqual(gui.random_clicks, 1)

    def test_historical_baseline_detects_relative_quality_collapse(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [], render_levels=[80])
        config = CollectionConfig(
            screenshot_region=(100, 100, 32, 24),
            copy_dna_button=(10, 10),
            random_generate_button=(20, 20),
            screenshot_delay=0,
            render_check_delay=0.05,
            render_stability_timeout=0.11,
            render_stability_threshold=2.0,
            render_min_contrast=10.0,
            render_min_quality_ratio=0.90,
            render_baseline_window=5,
            render_baseline_min_samples=5,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
        )
        collector = VerifiedCollector(
            config,
            pyautogui_module=gui,
            pyperclip_module=clipboard,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            face_dir = Path(temporary) / "face"
            face_dir.mkdir()
            for index in range(1, 6):
                image = Image.new("RGB", (32, 24), (40, 40, 40))
                image.paste((150, 130, 115), (4, 4, 30, 22))
                image.save(face_dir / f"face_{index:04d}.png")

            with self.assertRaisesRegex(RenderStabilityError, "人物对比度"):
                collector.wait_for_stable_render(temporary, 6)

    def test_historical_low_percentile_allows_healthy_dark_portrait(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [], render_levels=[80])
        config = CollectionConfig(
            screenshot_region=(100, 100, 32, 24),
            copy_dna_button=(10, 10),
            random_generate_button=(20, 20),
            screenshot_delay=0,
            render_check_delay=0.01,
            render_stability_timeout=0.10,
            render_stability_threshold=2.0,
            render_min_contrast=10.0,
            render_min_quality_ratio=0.90,
            render_baseline_window=5,
            render_baseline_min_samples=5,
            mouse_move_duration=0,
            click_hover_delay=0,
            click_hold_delay=0,
        )
        collector = VerifiedCollector(
            config,
            pyautogui_module=gui,
            pyperclip_module=clipboard,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            face_dir = Path(temporary) / "face"
            face_dir.mkdir()
            for index, level in enumerate([80, 150, 150, 150, 150], start=1):
                image = Image.new("RGB", (32, 24), (40, 40, 40))
                image.paste(
                    (level, max(40, level - 20), max(40, level - 35)),
                    (4, 4, 30, 22),
                )
                image.save(face_dir / f"face_{index:04d}.png")

            _image, difference, change, contrast = (
                collector.wait_for_stable_render(temporary, 6)
            )

        self.assertLessEqual(difference, 2.0)
        self.assertGreaterEqual(change, config.render_min_change)
        self.assertGreaterEqual(contrast, config.render_min_contrast)

    def test_collection_state_rejects_unpaired_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "face").mkdir()
            (root / "dna").mkdir()
            Image.new("RGB", (4, 4)).save(root / "face" / "face_0001.png")
            with self.assertRaisesRegex(RuntimeError, "未配对"):
                discover_collection_state(root)

    def test_fingerprint_ignores_dna_whitespace(self) -> None:
        reformatted = DNA_A.replace(
            '        gene_test={ "test_a" 10 "test_a" 10 }',
            '\tgene_test={  "test_a"   10   "test_a"  10  }',
        )
        self.assertEqual(dna_fingerprint(DNA_A), dna_fingerprint(reformatted))


if __name__ == "__main__":
    unittest.main()
