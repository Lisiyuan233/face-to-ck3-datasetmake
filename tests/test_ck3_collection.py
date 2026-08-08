from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ck3_collection import (
    CollectionConfig,
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
    ) -> None:
        self.clipboard = clipboard
        self.random_dnas = list(random_dnas)
        self.change_after_screenshot = change_after_screenshot
        self.current_dna = DNA_A
        self.position = (0, 0)
        self.random_clicks = 0
        self.screenshot_count = 0

    def moveTo(self, x: int, y: int, *, duration: float) -> None:
        self.position = (x, y)

    def mouseDown(self) -> None:
        pass

    def mouseUp(self) -> None:
        if self.position == (10, 10):
            self.clipboard.value = self.current_dna
        elif self.position == (20, 20):
            self.random_clicks += 1
            if self.random_dnas:
                self.current_dna = self.random_dnas.pop(0)

    def screenshot(self, *, region):
        self.screenshot_count += 1
        image = Image.new("RGB", (region[2], region[3]), (40, 80, 120))
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
    def test_duplicate_previous_dna_is_retried_before_capture(self) -> None:
        clipboard = FakeClipboard()
        clock = FakeClock()
        gui = FakePyAutoGUI(clipboard, [DNA_A, DNA_B])
        collector = make_collector(gui, clipboard, clock)

        with tempfile.TemporaryDirectory() as temporary:
            result = collector.collect_sample(temporary, 1, DNA_A)
            self.assertEqual(result.randomize_attempts, 2)
            self.assertEqual(gui.random_clicks, 2)
            self.assertEqual(gui.screenshot_count, 1)
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
