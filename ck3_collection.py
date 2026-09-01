from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from dna_normalizer import DNARecord, parse_dna


SAMPLE_RE = re.compile(r"^face_(\d+)\.(?:png|txt)$", re.IGNORECASE)


class CollectionCancelled(RuntimeError):
    pass


class RenderStabilityError(RuntimeError):
    """The CK3 portrait renderer did not reach a trustworthy state."""

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
    render_check_delay: float = 0.50
    render_stability_timeout: float = 8.0
    render_stability_threshold: float = 2.0
    render_min_change: float = 2.5
    render_min_contrast: float = 35.0
    render_min_quality_ratio: float = 0.70
    render_baseline_window: int = 20
    render_baseline_min_samples: int = 5
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
            ("render_check_delay", self.render_check_delay),
            ("post_capture_delay", self.post_capture_delay),
            ("inter_sample_delay", self.inter_sample_delay),
            ("mouse_move_duration", self.mouse_move_duration),
            ("click_hover_delay", self.click_hover_delay),
            ("click_hold_delay", self.click_hold_delay),
        ):
            if value < 0:
                raise ValueError(f"{name} 不能小于 0")
        if (
            self.clipboard_timeout <= 0
            or self.stability_timeout <= 0
            or self.render_stability_timeout <= 0
            or self.render_check_delay <= 0
        ):
            raise ValueError(
                "clipboard_timeout、stability_timeout 和 "
                "render_stability_timeout、render_check_delay 必须大于 0"
            )
        if self.render_stability_threshold <= 0:
            raise ValueError("render_stability_threshold 必须大于 0")
        if self.render_min_change <= 0:
            raise ValueError("render_min_change 必须大于 0")
        if self.render_min_contrast <= 0:
            raise ValueError("render_min_contrast 必须大于 0")
        if not 0 < self.render_min_quality_ratio <= 1:
            raise ValueError("render_min_quality_ratio 必须在 (0, 1] 范围内")
        if self.render_baseline_window < 1:
            raise ValueError("render_baseline_window 必须至少为 1")
        if not (
            1
            <= self.render_baseline_min_samples
            <= self.render_baseline_window
        ):
            raise ValueError(
                "render_baseline_min_samples 必须在 1 到 "
                "render_baseline_window 之间"
            )
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
    render_difference: float
    render_change: float
    render_contrast: float


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
        self._render_quality_baseline: deque[float] = deque(
            maxlen=self.config.render_baseline_window
        )
        self._render_baseline_initialized = False
        self._render_group: int | None = None
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

    def _screenshot_now(self) -> Any:
        self._park_mouse()
        self._check_cancelled()
        try:
            return self.pyautogui.screenshot(region=self.config.screenshot_region)
        except self.pyautogui.FailSafeException as error:
            raise CollectionCancelled("截图时触发 PyAutoGUI 安全停止") from error

    @staticmethod
    def _render_metrics(image: Any) -> tuple[bytes, float]:
        """Return a low-resolution signature and portrait/background contrast."""
        width, height = 164, 99
        gray = image.convert("L").resize((width, height))
        pixels = gray.tobytes()

        def region(x0: int, y0: int, x1: int, y1: int) -> list[int]:
            return [
                pixels[y * width + x]
                for y in range(y0, y1)
                for x in range(x0, x1)
            ]

        # The capture layout is fixed: two portraits occupy the lower center,
        # while these top patches contain only the static CK3 background.
        background = region(0, 0, 20, 13) + region(75, 0, 95, 13)
        portrait = region(29, 18, 72, 77) + region(113, 15, 150, 74)
        portrait.sort()
        highlight = portrait[int(0.90 * (len(portrait) - 1))]
        contrast = highlight - statistics.fmean(background)
        return pixels, contrast

    @staticmethod
    def _signature_difference(previous: bytes, current: bytes) -> float:
        if len(previous) != len(current):
            return float("inf")
        total = sum(abs(left - right) for left, right in zip(previous, current))
        return total / len(current)

    def _prime_render_baseline(self, base_dir: Path, index: int) -> None:
        race_group = (index - 1) // self.config.race_group_size
        if self._render_group != race_group:
            self._render_group = race_group
            self._render_quality_baseline.clear()
            self._render_baseline_initialized = False
        if self._render_baseline_initialized:
            return

        group_start = race_group * self.config.race_group_size + 1
        first = max(group_start, index - self.config.render_baseline_window)
        for previous_index in range(first, index):
            image_path = base_dir / "face" / f"face_{previous_index:04d}.png"
            if not image_path.is_file():
                continue
            try:
                with Image.open(image_path) as previous_image:
                    _signature, contrast = self._render_metrics(previous_image)
                self._render_quality_baseline.append(contrast)
            except (OSError, ValueError):
                # Pair/state validation owns corrupt-file handling. A single
                # unreadable history image must not disable live safeguards.
                continue
        self._render_baseline_initialized = True
        if self._render_quality_baseline:
            self.on_event(
                "已从最近 "
                f"{len(self._render_quality_baseline)} 张同种族截图恢复渲染基线"
            )

    def _required_render_contrast(self) -> float:
        required = self.config.render_min_contrast
        if (
            len(self._render_quality_baseline)
            >= self.config.render_baseline_min_samples
        ):
            # Skin tone and lighting legitimately vary much more than renderer
            # quality.  Using the median made a healthy dark portrait look like
            # a quality collapse after a run of brighter portraits.  A low
            # historical percentile still catches a real collapse while
            # respecting the healthy range already observed for this race.
            ordered = sorted(self._render_quality_baseline)
            low_index = int(0.10 * (len(ordered) - 1))
            rolling_low = ordered[low_index]
            required = max(
                required,
                rolling_low * self.config.render_min_quality_ratio,
            )
        return required

    def _previous_render_signature(
        self, base_dir: Path, index: int
    ) -> bytes | None:
        if index <= 1:
            return None
        image_path = base_dir / "face" / f"face_{index - 1:04d}.png"
        if not image_path.is_file():
            return None
        try:
            with Image.open(image_path) as previous_image:
                signature, _contrast = self._render_metrics(previous_image)
            return signature
        except (OSError, ValueError):
            return None

    def _save_render_diagnostics(
        self,
        base_dir: Path,
        index: int,
        image: Any,
        *,
        difference: float,
        change: float,
        contrast: float,
        required_contrast: float,
        reason: str,
    ) -> tuple[Path, Path]:
        diagnostics_dir = base_dir / "collection_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        sample_id = f"face_{index:04d}"
        image_path = diagnostics_dir / f"render_failure_{sample_id}.png"
        metrics_path = diagnostics_dir / f"render_failure_{sample_id}.json"
        image.save(image_path, format="PNG")
        payload = {
            "sample_id": sample_id,
            "reason": reason,
            "render_difference": None if difference == float("inf") else difference,
            "render_stability_threshold": self.config.render_stability_threshold,
            "change_from_previous_sample": None if change == float("inf") else change,
            "render_min_change": self.config.render_min_change,
            "render_contrast": contrast,
            "required_render_contrast": required_contrast,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_path": str(image_path),
        }
        temporary = metrics_path.with_name(metrics_path.name + ".partial")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, metrics_path)
        finally:
            temporary.unlink(missing_ok=True)
        return image_path, metrics_path

    def wait_for_stable_render(
        self, base_dir: str | Path, index: int
    ) -> tuple[Any, float, float, float]:
        """Wait for a changed, healthy render and prefer its calmest frame."""
        base_path = Path(base_dir)
        self._prime_render_baseline(base_path, index)
        previous_sample_signature = self._previous_render_signature(base_path, index)
        self._park_mouse()
        self.sleep(self.config.screenshot_delay)
        deadline = self.monotonic() + self.config.render_stability_timeout
        previous_signature: bytes | None = None
        best_candidate: tuple[Any, float, float, float] | None = None

        while True:
            image = self._screenshot_now()
            signature, contrast = self._render_metrics(image)
            required_contrast = self._required_render_contrast()
            difference = (
                self._signature_difference(previous_signature, signature)
                if previous_signature is not None
                else float("inf")
            )
            change = (
                self._signature_difference(previous_sample_signature, signature)
                if previous_sample_signature is not None
                else 0.0
            )
            stable = difference <= self.config.render_stability_threshold
            healthy = contrast >= required_contrast
            changed = (
                previous_sample_signature is None
                or change >= self.config.render_min_change
            )
            trustworthy = healthy and changed
            if previous_signature is not None and trustworthy:
                if best_candidate is None or difference < best_candidate[1]:
                    best_candidate = (image, difference, change, contrast)
            if stable and trustworthy:
                self.on_event(
                    "渲染稳定校验通过："
                    f"帧差 {difference:.2f}，较上一样本变化 {change:.2f}，"
                    f"对比度 {contrast:.1f}"
                )
                return image, difference, change, contrast

            remaining = deadline - self.monotonic()
            if remaining <= 0:
                if best_candidate is not None:
                    best_image, best_difference, best_change, best_contrast = (
                        best_candidate
                    )
                    self.on_event(
                        "画面健康且已切换，但人物动画使连续帧未达到严格阈值；"
                        "使用等待期间最稳定帧："
                        f"帧差 {best_difference:.2f}，"
                        f"较上一样本变化 {best_change:.2f}，"
                        f"对比度 {best_contrast:.1f}"
                    )
                    return (
                        best_image,
                        best_difference,
                        best_change,
                        best_contrast,
                    )
                difference_text = (
                    "尚无第二帧" if previous_signature is None else f"{difference:.2f}"
                )
                if not healthy:
                    reason = (
                        f"人物对比度持续过低 {contrast:.1f}<"
                        f"{required_contrast:.1f}"
                    )
                elif not changed:
                    reason = (
                        "DNA 已变化，但人物画面仍与上一样本近似相同 "
                        f"{change:.2f}<{self.config.render_min_change:.2f}"
                    )
                else:
                    reason = "等待期间没有得到可比较的第二个健康帧"
                _diagnostic_image, diagnostic_metrics = (
                    self._save_render_diagnostics(
                        base_path,
                        index,
                        image,
                        difference=difference,
                        change=change,
                        contrast=contrast,
                        required_contrast=required_contrast,
                        reason=reason,
                    )
                )
                raise RenderStabilityError(
                    "检测到 CK3 渲染持续异常，已自动停机且未保存当前样本："
                    f"{reason}；"
                    f"帧差={difference_text}（要求≤"
                    f"{self.config.render_stability_threshold:.2f}），"
                    f"较上一样本变化={change:.2f}（要求≥"
                    f"{self.config.render_min_change:.2f}），"
                    f"人物对比度={contrast:.1f}（要求≥{required_contrast:.1f}）。"
                    f"诊断文件：{diagnostic_metrics}。"
                    "请重启 CK3，并检查诊断截图。"
                )

            reasons = []
            if not stable:
                reasons.append(
                    "画面仍在变化"
                    if previous_signature is not None
                    else "等待第二帧"
                )
            if not healthy:
                reasons.append(
                    f"人物对比度过低 {contrast:.1f}<{required_contrast:.1f}"
                )
            if not changed:
                reasons.append(
                    "画面尚未切换 "
                    f"{change:.1f}<{self.config.render_min_change:.1f}"
                )
            self.on_event("渲染校验：" + "；".join(reasons))
            previous_signature = signature
            self.sleep(min(self.config.render_check_delay, remaining))

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
                image, render_difference, render_change, render_contrast = (
                    self.wait_for_stable_render(base_dir, index)
                )
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
                self._render_quality_baseline.append(render_contrast)
                self.sleep(self.config.inter_sample_delay)
                return CollectedSample(
                    sample_id=f"face_{index:04d}",
                    image_path=image_path,
                    dna_path=dna_path,
                    dna_text=verified_text,
                    dna_fingerprint=dna_fingerprint(verified_text),
                    randomize_attempts=randomize_attempts,
                    transaction_attempts=transaction_attempt,
                    render_difference=render_difference,
                    render_change=render_change,
                    render_contrast=render_contrast,
                )
            except (CollectionCancelled, RenderStabilityError):
                raise
            except Exception as error:
                errors.append(str(error))
                self.on_event(
                    f"样本事务失败（{transaction_attempt}/{self.config.sample_retries + 1}）：{error}"
                )
        raise RuntimeError("样本连续校验失败: " + " | ".join(errors[-3:]))
