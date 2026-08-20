from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from dna_normalizer import DNARecord, parse_dna


SAMPLE_RE = re.compile(r"^face_(\d+)\.(?:png|txt)$", re.IGNORECASE)


class CollectionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionConfig:
    screenshot_region: tuple[int, int, int, int]
    copy_dna_button: tuple[int, int]
    random_generate_button: tuple[int, int]
    auto_switch_race: bool = False
    race_group_size: int = 30_000
    race_count: int = 17
    race_button: tuple[int, int] | None = None
    race_first_option: tuple[int, int] | None = None
    race_second_option: tuple[int, int] | None = None
    show_hair_beard_checkbox: tuple[int, int] | None = None
    facial_structure_button: tuple[int, int] | None = None
    clipboard_settle_delay: float = 0.10
    clipboard_timeout: float = 3.0
    ui_settle_delay: float = 1.5
    stability_check_delay: float = 0.50
    stability_timeout: float = 8.0
    screenshot_delay: float = 0.50
    post_capture_delay: float = 0.20
    inter_sample_delay: float = 0.20
    randomize_retries: int = 4
    sample_retries: int = 2
    mouse_move_duration: float = 0.12
    click_hover_delay: float = 0.05
    click_hold_delay: float = 0.08

    def validate(self) -> None:
        _left, _top, width, height = self.screenshot_region
        if width <= 0 or height <= 0:
            raise ValueError("截图区域宽高必须大于 0")
        for name, value in (
            ("clipboard_settle_delay", self.clipboard_settle_delay),
            ("ui_settle_delay", self.ui_settle_delay),
            ("stability_check_delay", self.stability_check_delay),
            ("screenshot_delay", self.screenshot_delay),
            ("post_capture_delay", self.post_capture_delay),
            ("inter_sample_delay", self.inter_sample_delay),
            ("mouse_move_duration", self.mouse_move_duration),
            ("click_hover_delay", self.click_hover_delay),
            ("click_hold_delay", self.click_hold_delay),
        ):
            if value < 0:
                raise ValueError(f"{name} 不能小于 0")
        if self.clipboard_timeout <= 0 or self.stability_timeout <= 0:
            raise ValueError("clipboard_timeout 和 stability_timeout 必须大于 0")
        if self.randomize_retries < 1:
            raise ValueError("randomize_retries 必须至少为 1")
        if self.sample_retries < 0:
            raise ValueError("sample_retries 不能小于 0")
        if type(self.auto_switch_race) is not bool:
            raise ValueError("auto_switch_race 必须是布尔值")
        if self.race_group_size < 1:
            raise ValueError("race_group_size 必须至少为 1")
        if self.race_count < 2:
            raise ValueError("race_count 必须至少为 2")
        if self.auto_switch_race:
            positions = {
                "种族按钮": self.race_button,
                "种族列表第一项": self.race_first_option,
                "种族列表第二项": self.race_second_option,
                "显示头发与胡须复选框": self.show_hair_beard_checkbox,
                "面部结构按钮": self.facial_structure_button,
            }
            missing = [name for name, value in positions.items() if value is None]
            if missing:
                raise ValueError(
                    "已启用自动切换种族，但以下位置未设置：" + "、".join(missing)
                )
            if self.race_first_option[1] == self.race_second_option[1]:
                raise ValueError("种族列表第一项和第二项必须位于不同行")


@dataclass(frozen=True)
class CollectedSample:
    sample_id: str
    image_path: Path
    dna_path: Path
    dna_text: str
    dna_fingerprint: str
    randomize_attempts: int
    transaction_attempts: int


@dataclass(frozen=True)
class CollectionState:
    next_index: int
    previous_dna: str | None


def _validated_record(dna_text: str) -> DNARecord:
    if not dna_text.strip():
        raise ValueError("复制到的 DNA 为空")
    record = parse_dna(dna_text)
    if not record.genes:
        raise ValueError("复制内容不是有效的 CK3 DNA")
    return record


def dna_fingerprint(dna_text: str) -> str:
    record = _validated_record(dna_text)
    canonical = json.dumps(
        asdict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def discover_collection_state(base_dir: str | Path) -> CollectionState:
    base_dir = Path(base_dir)
    face_dir = base_dir / "face"
    dna_dir = base_dir / "dna"
    face_dir.mkdir(parents=True, exist_ok=True)
    dna_dir.mkdir(parents=True, exist_ok=True)

    image_ids = {
        int(match.group(1))
        for path in face_dir.glob("face_*.png")
        if (match := SAMPLE_RE.fullmatch(path.name))
    }
    dna_ids = {
        int(match.group(1))
        for path in dna_dir.glob("face_*.txt")
        if (match := SAMPLE_RE.fullmatch(path.name))
    }
    if image_ids != dna_ids:
        image_only = sorted(image_ids - dna_ids)[:5]
        dna_only = sorted(dna_ids - image_ids)[:5]
        raise RuntimeError(
            "检测到未配对的历史采集文件；请先处理再继续: "
            f"image_only={image_only}, dna_only={dna_only}"
        )
    if not image_ids:
        return CollectionState(next_index=1, previous_dna=None)
    last_index = max(image_ids)
    sample_id = f"face_{last_index:04d}"
    previous = (dna_dir / f"{sample_id}.txt").read_text(encoding="utf-8")
    _validated_record(previous)
    return CollectionState(next_index=last_index + 1, previous_dna=previous)


class VerifiedCollector:
    """Collect image/DNA pairs only after bidirectional synchronization checks."""

    def __init__(
        self,
        config: CollectionConfig,
        *,
        pyautogui_module: Any,
        pyperclip_module: Any,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        cancelled: Callable[[], bool] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.pyautogui = pyautogui_module
        self.pyperclip = pyperclip_module
        self.sleep = sleep
        self.monotonic = monotonic
        self.cancelled = cancelled or (lambda: False)
        self.on_event = on_event or (lambda _message: None)
        if hasattr(self.pyautogui, "FAILSAFE"):
            self.pyautogui.FAILSAFE = True
        if hasattr(self.pyautogui, "PAUSE"):
            self.pyautogui.PAUSE = 0.05

    def _check_cancelled(self) -> None:
        if self.cancelled():
            raise CollectionCancelled("采集已取消")

    def _click(self, position: tuple[int, int], *, short_hover: bool = False) -> None:
        self._check_cancelled()
        try:
            self.pyautogui.moveTo(
                position[0],
                position[1],
                duration=self.config.mouse_move_duration,
            )
            hover = min(self.config.click_hover_delay, 0.05) if short_hover else self.config.click_hover_delay
            self.sleep(hover)
            self.pyautogui.mouseDown()
            self.sleep(self.config.click_hold_delay)
            self.pyautogui.mouseUp()
        except self.pyautogui.FailSafeException as error:
            raise CollectionCancelled("检测到 PyAutoGUI 安全停止") from error

    def _park_mouse(self) -> None:
        left, top, width, height = self.config.screenshot_region
        try:
            self.pyautogui.moveTo(
                left + width // 2,
                top + height // 2,
                duration=min(self.config.mouse_move_duration, 0.10),
            )
        except self.pyautogui.FailSafeException as error:
            raise CollectionCancelled("移动鼠标离开工具栏时触发安全停止") from error

    def prepare_race_for_sample(self, sample_index: int) -> bool:
        """Select the deterministic race at a race-block boundary.

        Selecting the absolute target entry, instead of blindly clicking a
        relative "next" position, makes a boundary retry select the same race.
        The first two configured entries define the list's vertical row step.
        """
        if sample_index < 1:
            raise ValueError("sample_index 必须至少为 1")
        if not self.config.auto_switch_race:
            return False
        zero_based = sample_index - 1
        if sample_index == 1 or zero_based % self.config.race_group_size:
            return False

        target_group = zero_based // self.config.race_group_size
        if target_group >= self.config.race_count:
            raise RuntimeError(
                f"样本 {sample_index} 需要种族组 {target_group + 1}，"
                f"但只配置了 {self.config.race_count} 个种族"
            )

        first = self.config.race_first_option
        second = self.config.race_second_option
        assert first is not None and second is not None
        target_option = (
            first[0],
            first[1] + (second[1] - first[1]) * target_group,
        )
        steps = (
            ("点击种族按钮", self.config.race_button),
            (f"选择第 {target_group + 1} 个种族", target_option),
            ("取消显示头发与胡须", self.config.show_hair_beard_checkbox),
            ("打开面部结构", self.config.facial_structure_button),
        )
        self.on_event(
            f"到达种族边界：为 face_{sample_index:04d} 切换到"
            f"第 {target_group + 1}/{self.config.race_count} 个种族"
        )
        for label, position in steps:
            assert position is not None
            self.on_event(label)
            self._click(position)
            self.sleep(self.config.ui_settle_delay)
        self._park_mouse()
        return True

    def copy_current_dna(self) -> tuple[str, DNARecord]:
        """Copy DNA with a sentinel so a missed/stale click cannot pass."""
        sentinel = f"CK3_COLLECTION_SENTINEL_{time.time_ns()}"
        self.pyperclip.copy(sentinel)
        self.sleep(self.config.clipboard_settle_delay)
        self._click(self.config.copy_dna_button, short_hover=True)
        # Leave the control immediately: CK3 tooltips can otherwise cover the
        # neighboring random/paste controls while clipboard polling continues.
        self._park_mouse()

        deadline = self.monotonic() + self.config.clipboard_timeout
        poll_interval = max(0.05, min(self.config.clipboard_settle_delay, 0.25))
        while True:
            self._check_cancelled()
            copied = self.pyperclip.paste()
            if copied and copied != sentinel:
                return copied, _validated_record(copied)
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "复制 DNA 后剪贴板未更新；可能是点击未生效或 tooltip 遮挡"
                )
            self.sleep(min(poll_interval, remaining))

    def _randomize(self) -> None:
        self._click(self.config.random_generate_button)
        self._park_mouse()
        self.sleep(self.config.ui_settle_delay)

    def wait_for_new_stable_dna(
        self, previous_dna: str | None
    ) -> tuple[str, DNARecord, int]:
        previous_record = _validated_record(previous_dna) if previous_dna else None
        for randomize_attempt in range(1, self.config.randomize_retries + 1):
            self.on_event(f"随机生成并校验 DNA（尝试 {randomize_attempt}）")
            self._randomize()
            deadline = self.monotonic() + self.config.stability_timeout
            candidate_text: str | None = None
            candidate_record: DNARecord | None = None
            while self.monotonic() < deadline:
                copied, record = self.copy_current_dna()
                if previous_record is not None and record == previous_record:
                    candidate_text = None
                    candidate_record = None
                    self.on_event("DNA 尚未变化，继续等待")
                elif candidate_record is not None and record == candidate_record:
                    return candidate_text or copied, record, randomize_attempt
                else:
                    candidate_text = copied
                    candidate_record = record
                    self.on_event("检测到新 DNA，等待第二次一致性确认")
                self.sleep(self.config.stability_check_delay)
            self.on_event("本次随机生成未得到稳定的新 DNA，重新点击")
        raise RuntimeError(
            f"连续 {self.config.randomize_retries} 次随机生成都未得到稳定的新 DNA"
        )

    def _capture(self) -> Any:
        self._park_mouse()
        self.sleep(self.config.screenshot_delay)
        self._check_cancelled()
        try:
            return self.pyautogui.screenshot(region=self.config.screenshot_region)
        except self.pyautogui.FailSafeException as error:
            raise CollectionCancelled("截图时触发 PyAutoGUI 安全停止") from error

    @staticmethod
    def _save_atomic(
        base_dir: Path, index: int, image: Any, dna_text: str
    ) -> tuple[Path, Path]:
        sample_id = f"face_{index:04d}"
        image_path = base_dir / "face" / f"{sample_id}.png"
        dna_path = base_dir / "dna" / f"{sample_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        dna_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() or dna_path.exists():
            raise FileExistsError(f"拒绝覆盖已有样本: {sample_id}")
        image_partial = image_path.with_name(image_path.name + ".partial")
        dna_partial = dna_path.with_name(dna_path.name + ".partial")
        try:
            image.save(image_partial, format="PNG")
            dna_partial.write_text(dna_text, encoding="utf-8", newline="\n")
            # The image is committed last. discover_collection_state therefore
            # never treats a half-written transaction as a completed sample.
            os.replace(dna_partial, dna_path)
            os.replace(image_partial, image_path)
        finally:
            image_partial.unlink(missing_ok=True)
            dna_partial.unlink(missing_ok=True)
        return image_path, dna_path

    def collect_sample(
        self,
        base_dir: str | Path,
        index: int,
        previous_dna: str | None,
    ) -> CollectedSample:
        errors: list[str] = []
        for transaction_attempt in range(1, self.config.sample_retries + 2):
            self._check_cancelled()
            try:
                candidate_text, candidate_record, randomize_attempts = (
                    self.wait_for_new_stable_dna(previous_dna)
                )
                image = self._capture()
                self.sleep(self.config.post_capture_delay)
                verified_text, verified_record = self.copy_current_dna()
                if verified_record != candidate_record:
                    raise RuntimeError(
                        "截图前后的 DNA 不一致；本次截图已丢弃，不写入 sample ID"
                    )
                if previous_dna and verified_record == _validated_record(previous_dna):
                    raise RuntimeError("截图 DNA 与上一条相同；本次截图已丢弃")
                image_path, dna_path = self._save_atomic(
                    Path(base_dir), index, image, verified_text
                )
                self.sleep(self.config.inter_sample_delay)
                return CollectedSample(
                    sample_id=f"face_{index:04d}",
                    image_path=image_path,
                    dna_path=dna_path,
                    dna_text=verified_text,
                    dna_fingerprint=dna_fingerprint(verified_text),
                    randomize_attempts=randomize_attempts,
                    transaction_attempts=transaction_attempt,
                )
            except CollectionCancelled:
                raise
            except Exception as error:
                errors.append(str(error))
                self.on_event(
                    f"样本事务失败（{transaction_attempt}/{self.config.sample_retries + 1}）：{error}"
                )
        raise RuntimeError("样本连续校验失败: " + " | ".join(errors[-3:]))
