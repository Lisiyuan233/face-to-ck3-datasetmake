#!/usr/bin/env python3
"""GUI automation for controlled one-field CK3 DNA render sweeps.

The pure mutation/runner functions intentionally avoid GUI imports so they can
be unit-tested on machines without a desktop session. PyAutoGUI, pyperclip and
tkinter are loaded only when the Windows GUI is launched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from dna_normalizer import BYTE_MAX, QUOTED_FIELD_RE, parse_dna


TOOL_VERSION = 2
SETTINGS_VERSION = 2
ALLELE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str, maximum: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or "item")[:maximum]


def _deduplicate(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def parse_value_sequence(text: str) -> list[int]:
    """Parse `0,64,128` or an inclusive `0:255:32` byte sequence."""
    text = text.strip()
    if not text:
        raise ValueError("DNA 数值序列不能为空")
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("范围格式必须是 start:end:step，例如 0:255:32")
        try:
            start, end, step = (int(part) for part in parts)
        except ValueError as error:
            raise ValueError("范围中的 start、end、step 必须是整数") from error
        if step == 0:
            raise ValueError("step 不能为 0")
        if (end - start) * step < 0:
            raise ValueError("step 方向与 start/end 不一致")
        values = []
        current = start
        if step > 0:
            while current <= end:
                values.append(current)
                current += step
        else:
            while current >= end:
                values.append(current)
                current += step
        if values[-1] != end:
            values.append(end)
    else:
        tokens = [
            token
            for token in re.split(r"[,;，；\s]+", text)
            if token
        ]
        try:
            values = [int(token) for token in tokens]
        except ValueError as error:
            raise ValueError("DNA 数值必须是 0..255 的整数") from error
    values = _deduplicate(values)
    if not values or any(value < 0 or value > BYTE_MAX for value in values):
        raise ValueError("DNA 数值必须位于 0..255")
    return values


def parse_allele_sequence(text: str, default: str) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[,;，；\s]+", text.strip())
        if token
    ]
    alleles = _deduplicate(tokens or [default])
    invalid = [allele for allele in alleles if not ALLELE_RE.fullmatch(allele)]
    if invalid:
        raise ValueError(
            "allele 只能包含英文字母、数字和下划线: " + ", ".join(invalid)
        )
    return alleles


def replace_gene_pair(
    dna_text: str,
    field: str,
    allele: str,
    value: int,
) -> str:
    """Replace exactly one gene and synchronize both chromosome slots."""
    if not ALLELE_RE.fullmatch(allele):
        raise ValueError(f"invalid allele: {allele!r}")
    if not 0 <= int(value) <= BYTE_MAX:
        raise ValueError(f"DNA value must be in 0..{BYTE_MAX}: {value}")
    original = parse_dna(dna_text)
    if field not in original.genes:
        raise ValueError(f"DNA 中不存在字段: {field}")
    replaced = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal replaced
        if match.group("key") != field:
            return match.group(0)
        replaced += 1
        return (
            f'{match.group("indent")}{field}={{ "{allele}" {int(value)} '
            f'"{allele}" {int(value)} }}'
        )

    output = QUOTED_FIELD_RE.sub(substitute, dna_text)
    if replaced != 1:
        raise ValueError(f"字段 {field} 应匹配一次，实际匹配 {replaced} 次")
    mutated = parse_dna(output)
    expected = (allele, int(value), allele, int(value))
    actual_value = mutated.genes[field]
    actual = (
        actual_value.allele1,
        actual_value.value1,
        actual_value.allele2,
        actual_value.value2,
    )
    if actual != expected:
        raise RuntimeError(f"字段 {field} 修改校验失败: {actual!r}")
    for key, gene in original.genes.items():
        if key != field and mutated.genes.get(key) != gene:
            raise RuntimeError(f"修改 {field} 时意外改变了 {key}")
    if mutated.colors != original.colors:
        raise RuntimeError(f"修改 {field} 时意外改变了颜色字段")
    return output


@dataclass(frozen=True)
class SweepVariant:
    index: int
    variant_id: str
    field: str
    allele: str
    value: int
    dna_text: str
    dna_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "variant_id": self.variant_id,
            "field": self.field,
            "allele": self.allele,
            "value": self.value,
            "dna_sha256": self.dna_sha256,
        }


def build_sweep_variants(
    base_dna: str,
    field: str,
    alleles: Sequence[str],
    values: Sequence[int],
) -> list[SweepVariant]:
    record = parse_dna(base_dna)
    if not record.genes:
        raise ValueError("没有从输入文本中解析到 DNA gene 字段")
    if field not in record.genes:
        raise ValueError(f"DNA 中不存在字段: {field}")
    alleles = parse_allele_sequence(",".join(alleles), record.genes[field].allele1)
    checked_values = parse_value_sequence(",".join(str(value) for value in values))
    variants = []
    index = 0
    for allele in alleles:
        for value in checked_values:
            index += 1
            dna_text = replace_gene_pair(base_dna, field, allele, value)
            variant_id = (
                f"{index:05d}_{safe_component(field)}_"
                f"{safe_component(allele)}_{value:03d}"
            )
            variants.append(
                SweepVariant(
                    index=index,
                    variant_id=variant_id,
                    field=field,
                    allele=allele,
                    value=value,
                    dna_text=dna_text,
                    dna_sha256=sha256_text(dna_text),
                )
            )
    return variants


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class AutomationConfig:
    paste_button: tuple[int, int]
    screenshot_region: tuple[int, int, int, int]
    confirm_button: tuple[int, int] | None = None
    verify_copy_button: tuple[int, int] | None = None
    clipboard_delay: float = 0.20
    confirm_delay: float = 0.80
    settle_delay: float = 1.50
    screenshot_delay: float = 0.30
    mouse_move_duration: float = 0.15
    click_hover_delay: float = 0.20
    click_hold_delay: float = 0.08
    verification_timeout: float = 3.00
    inter_variant_delay: float = 0.50
    retries: int = 1

    def validate(self) -> None:
        _left, _top, width, height = self.screenshot_region
        if width <= 0 or height <= 0:
            raise ValueError("截图区域必须是有效的 left, top, width, height")
        if self.confirm_button is None:
            raise ValueError("必须记录粘贴 DNA 后弹窗中的“确定”按钮位置")
        for name, value in (
            ("clipboard_delay", self.clipboard_delay),
            ("confirm_delay", self.confirm_delay),
            ("settle_delay", self.settle_delay),
            ("screenshot_delay", self.screenshot_delay),
            ("mouse_move_duration", self.mouse_move_duration),
            ("click_hover_delay", self.click_hover_delay),
            ("click_hold_delay", self.click_hold_delay),
            ("inter_variant_delay", self.inter_variant_delay),
        ):
            if value < 0:
                raise ValueError(f"{name} 不能小于 0")
        if self.verification_timeout <= 0:
            raise ValueError("verification_timeout 必须大于 0")
        if self.retries < 0:
            raise ValueError("retries 不能小于 0")


class EmergencyStop(RuntimeError):
    pass


class AutomationBackend(Protocol):
    def apply_dna(self, dna_text: str, field: str) -> None: ...

    def capture(self, path: Path) -> None: ...


class WindowsAutomationBackend:
    """Lazy Windows GUI backend so importing this module remains headless-safe."""

    def __init__(self, config: AutomationConfig) -> None:
        config.validate()
        try:
            import pyautogui
            import pyperclip
        except ImportError as error:
            raise RuntimeError(
                "缺少 GUI 依赖，请运行: pip install pyautogui pillow pyperclip"
            ) from error
        self.config = config
        self.pyautogui = pyautogui
        self.pyperclip = pyperclip
        self.pyautogui.FAILSAFE = True
        self.pyautogui.PAUSE = 0.05

    def _click(self, position: tuple[int, int]) -> None:
        try:
            # CK3's immediate-mode UI can miss a click when the pointer jumps
            # from the confirmation dialog to a toolbar icon and presses in the
            # same frame. Give the game time to establish the hover target, then
            # send a deliberate press/release pair.
            self.pyautogui.moveTo(
                position[0],
                position[1],
                duration=self.config.mouse_move_duration,
            )
            time.sleep(self.config.click_hover_delay)
            self.pyautogui.mouseDown()
            time.sleep(self.config.click_hold_delay)
            self.pyautogui.mouseUp()
        except self.pyautogui.FailSafeException as error:
            raise EmergencyStop("检测到 PyAutoGUI 安全停止（鼠标位于左上角）") from error

    def _wait_for_copied_dna(self, sentinel: str) -> str:
        deadline = time.monotonic() + self.config.verification_timeout
        poll_interval = max(0.05, min(self.config.clipboard_delay, 0.25))
        while True:
            copied = self.pyperclip.paste()
            if copied and copied != sentinel:
                return copied
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "点击复制 DNA 验证按钮后，剪贴板在 "
                    f"{self.config.verification_timeout:g} 秒内没有更新"
                )
            time.sleep(min(poll_interval, remaining))

    def apply_dna(self, dna_text: str, field: str) -> None:
        desired = parse_dna(dna_text).genes[field]
        self.pyperclip.copy(dna_text)
        time.sleep(self.config.clipboard_delay)
        self._click(self.config.paste_button)
        time.sleep(self.config.confirm_delay)
        assert self.config.confirm_button is not None
        self._click(self.config.confirm_button)
        time.sleep(self.config.settle_delay)

        if self.config.verify_copy_button is not None:
            sentinel = f"CK3_DNA_VERIFY_{time.time_ns()}"
            self.pyperclip.copy(sentinel)
            time.sleep(self.config.clipboard_delay)
            self._click(self.config.verify_copy_button)
            copied = self._wait_for_copied_dna(sentinel)
            actual_record = parse_dna(copied)
            actual = actual_record.genes.get(field)
            if actual != desired:
                raise RuntimeError(
                    f"游戏内 DNA 校验失败: {field} 期望 {desired}，实际 {actual}"
                )

    def capture(self, path: Path) -> None:
        time.sleep(self.config.screenshot_delay)
        try:
            image = self.pyautogui.screenshot(region=self.config.screenshot_region)
        except self.pyautogui.FailSafeException as error:
            raise EmergencyStop("检测到 PyAutoGUI 安全停止（鼠标位于左上角）") from error
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
        time.sleep(self.config.inter_variant_delay)


def _plan_hash(base_dna: str, variants: Sequence[SweepVariant]) -> str:
    payload = {
        "base_dna_sha256": sha256_text(base_dna),
        "variants": [variant.metadata() for variant in variants],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def prepare_session(
    session_dir: Path,
    base_dna: str,
    variants: Sequence[SweepVariant],
    automation_config: AutomationConfig,
) -> dict[str, Any]:
    automation_config.validate()
    session_dir.mkdir(parents=True, exist_ok=True)
    plan_hash = _plan_hash(base_dna, variants)
    session_path = session_dir / "session.json"
    if session_path.is_file():
        existing = json.loads(session_path.read_text(encoding="utf-8"))
        if int(existing.get("version", 0)) < TOOL_VERSION:
            raise RuntimeError(
                "该会话由旧版瞬时点击流程创建，completed 截图可能没有实际应用 DNA；"
                "请新建会话重新扫描"
            )
        if existing.get("plan_sha256") != plan_hash:
            raise RuntimeError("恢复目录中的 DNA/字段/数值计划与当前设置不一致")
        previous_automation = existing.get("automation")
        if (
            isinstance(previous_automation, dict)
            and previous_automation.get("confirm_button") is None
        ):
            raise RuntimeError(
                "该旧会话没有设置确认按钮，已有截图可能包含确认弹窗；"
                "请新建会话重新扫描"
            )
        if previous_automation is not None:
            existing.setdefault("automation_history", []).append(
                {
                    "replaced_at": utc_now(),
                    "settings": previous_automation,
                }
            )
        existing["automation"] = asdict(automation_config)
        existing["last_resumed_at"] = utc_now()
        atomic_write_text(
            session_path,
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        )
        return existing
    session = {
        "version": TOOL_VERSION,
        "created_at": utc_now(),
        "base_dna_sha256": sha256_text(base_dna),
        "plan_sha256": plan_hash,
        "field": variants[0].field if variants else None,
        "variant_count": len(variants),
        "variants": [variant.metadata() for variant in variants],
        "automation": asdict(automation_config),
    }
    atomic_write_text(session_dir / "base_dna.txt", base_dna)
    atomic_write_text(
        session_path,
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
    )
    return session


def completed_variant_ids(session_dir: Path) -> set[str]:
    path = session_dir / "manifest.jsonl"
    completed: set[str] = set()
    if not path.is_file():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"manifest.jsonl 第 {line_number} 行损坏") from error
        if record.get("status") == "completed":
            render_path = session_dir / str(record.get("render_path", ""))
            if render_path.is_file():
                completed.add(str(record["variant_id"]))
    return completed


@dataclass(frozen=True)
class SweepResult:
    completed: int
    skipped: int
    stopped: bool


def run_sweep(
    variants: Sequence[SweepVariant],
    session_dir: Path,
    backend: AutomationBackend,
    *,
    retries: int,
    identical_render_limit: int = 0,
    pause_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[int, int, SweepVariant, str], None] | None = None,
) -> SweepResult:
    """Apply variants sequentially, stop on a persistent error, and resume safely."""
    pause_event = pause_event or threading.Event()
    stop_event = stop_event or threading.Event()
    completed_ids = completed_variant_ids(session_dir)
    total = len(variants)
    completed = 0
    skipped = 0
    manifest_path = session_dir / "manifest.jsonl"
    error_path = session_dir / "errors.jsonl"
    previous_render_sha256: str | None = None
    identical_render_count = 0

    for variant in variants:
        if variant.variant_id in completed_ids:
            completed_render = session_dir / "renders" / f"{variant.variant_id}.png"
            if completed_render.is_file():
                completed_hash = sha256_file(completed_render)
                if completed_hash == previous_render_sha256:
                    identical_render_count += 1
                else:
                    previous_render_sha256 = completed_hash
                    identical_render_count = 1
            skipped += 1
            completed += 1
            if on_progress:
                on_progress(completed, total, variant, "skipped")
            continue
        while pause_event.is_set() and not stop_event.is_set():
            stop_event.wait(0.10)
        if stop_event.is_set():
            return SweepResult(completed=completed, skipped=skipped, stopped=True)

        dna_relative = Path("dna") / f"{variant.variant_id}.txt"
        render_relative = Path("renders") / f"{variant.variant_id}.png"
        atomic_write_text(session_dir / dna_relative, variant.dna_text)
        last_error: Exception | None = None
        started_at = utc_now()
        render_sha256: str | None = None
        next_identical_count = identical_render_count
        for attempt in range(int(retries) + 1):
            try:
                backend.apply_dna(variant.dna_text, variant.field)
                backend.capture(session_dir / render_relative)
                render_sha256 = sha256_file(session_dir / render_relative)
                next_identical_count = (
                    identical_render_count + 1
                    if render_sha256 == previous_render_sha256
                    else 1
                )
                if (
                    identical_render_limit > 0
                    and next_identical_count >= identical_render_limit
                ):
                    raise RuntimeError(
                        f"连续 {next_identical_count} 张截图完全相同，"
                        "疑似后续粘贴未生效；请启用 DNA 回读验证或检查点击时序"
                    )
                last_error = None
                break
            except EmergencyStop:
                raise
            except Exception as error:  # GUI failures are recorded with context.
                last_error = error
                if attempt < int(retries):
                    time.sleep(0.50)
        if last_error is not None:
            append_jsonl(
                error_path,
                {
                    **variant.metadata(),
                    "status": "failed",
                    "started_at": started_at,
                    "failed_at": utc_now(),
                    "error": f"{type(last_error).__name__}: {last_error}",
                },
            )
            if on_progress:
                on_progress(completed, total, variant, "failed")
            raise RuntimeError(
                f"{variant.variant_id} 连续失败 {int(retries) + 1} 次，已停止："
                f"{last_error}"
            ) from last_error

        append_jsonl(
            manifest_path,
            {
                **variant.metadata(),
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "dna_path": dna_relative.as_posix(),
                "render_path": render_relative.as_posix(),
                "render_sha256": render_sha256,
            },
        )
        previous_render_sha256 = render_sha256
        identical_render_count = next_identical_count
        completed += 1
        if on_progress:
            on_progress(completed, total, variant, "completed")

    return SweepResult(completed=completed, skipped=skipped, stopped=False)


def settings_path() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / ".config"
    return root / "CK3DNAFieldSweep" / "settings.json"


def load_user_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_user_settings(value: dict[str, Any]) -> None:
    path = settings_path()
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class DnaFieldSweepApp:
    def __init__(self, root: Any, modules: dict[str, Any]) -> None:
        self.root = root
        self.tk = modules["tk"]
        self.ttk = modules["ttk"]
        self.filedialog = modules["filedialog"]
        self.messagebox = modules["messagebox"]
        self.scrolledtext = modules["scrolledtext"]
        self.record = None
        self.variants: list[SweepVariant] = []
        self.session_override: Path | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._build()
        self._load_settings()

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.title("CK3 DNA 单字段扫描与截图工具")
        self.root.geometry("1040x820")
        self.root.minsize(900, 720)

        dna_frame = ttk.LabelFrame(self.root, text="1. 输入基础 DNA")
        dna_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        button_row = ttk.Frame(dna_frame)
        button_row.pack(fill="x", padx=6, pady=5)
        ttk.Button(button_row, text="加载 DNA 文件", command=self._load_dna_file).pack(side="left", padx=3)
        ttk.Button(button_row, text="从剪贴板读取", command=self._paste_dna).pack(side="left", padx=3)
        ttk.Button(button_row, text="解析 DNA", command=self._parse_dna).pack(side="left", padx=3)
        self.dna_status = tk.StringVar(value="尚未解析")
        ttk.Label(button_row, textvariable=self.dna_status).pack(side="left", padx=12)
        self.dna_text = self.scrolledtext.ScrolledText(dna_frame, height=9, wrap="none")
        self.dna_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        sweep_frame = ttk.LabelFrame(self.root, text="2. 单字段扫描计划")
        sweep_frame.pack(fill="x", padx=10, pady=5)
        row = ttk.Frame(sweep_frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="字段:").pack(side="left")
        self.field_var = tk.StringVar()
        self.field_box = ttk.Combobox(row, textvariable=self.field_var, state="readonly", width=42)
        self.field_box.pack(side="left", padx=5)
        self.field_box.bind("<<ComboboxSelected>>", self._field_selected)
        self.current_gene = tk.StringVar(value="当前值: -")
        ttk.Label(row, textvariable=self.current_gene).pack(side="left", padx=8)

        row = ttk.Frame(sweep_frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="数值序列:").pack(side="left")
        self.values_var = tk.StringVar(value="0:255:32")
        ttk.Entry(row, textvariable=self.values_var, width=28).pack(side="left", padx=5)
        ttk.Label(row, text="支持 0,64,128 或 0:255:32（自动包含 255）").pack(side="left", padx=5)

        row = ttk.Frame(sweep_frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Allele 序列:").pack(side="left")
        self.alleles_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.alleles_var, width=55).pack(side="left", padx=5)
        ttk.Label(row, text="留空保持当前；多个用逗号分隔").pack(side="left", padx=5)
        ttk.Button(row, text="预览计划", command=self._preview).pack(side="right", padx=3)

        automation = ttk.LabelFrame(self.root, text="3. 游戏位置与自动化参数")
        automation.pack(fill="x", padx=10, pady=5)
        row = ttk.Frame(automation)
        row.pack(fill="x", padx=6, pady=4)
        self.paste_position = tk.StringVar(value="未设置")
        self.confirm_position = tk.StringVar(value="未设置（必填）")
        self.verify_position = tk.StringVar(value="未设置（可选）")
        self.region_value = tk.StringVar(value="未设置")
        ttk.Button(row, text="记录粘贴 DNA 按钮", command=lambda: self._record_position("paste")).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.paste_position).pack(side="left", padx=5)
        ttk.Button(row, text="记录确认按钮", command=lambda: self._record_position("confirm")).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.confirm_position).pack(side="left", padx=5)

        row = ttk.Frame(automation)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="记录复制 DNA 验证按钮", command=lambda: self._record_position("verify")).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.verify_position).pack(side="left", padx=5)
        ttk.Button(row, text="清除验证按钮", command=self._clear_verify_position).pack(side="left", padx=8)
        ttk.Button(row, text="设置截图区域", command=self._record_region).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.region_value).pack(side="left", padx=5)

        row = ttk.Frame(automation)
        row.pack(fill="x", padx=6, pady=4)
        self.clipboard_delay = tk.StringVar(value="0.20")
        self.confirm_delay = tk.StringVar(value="0.80")
        self.settle_delay = tk.StringVar(value="1.50")
        self.screenshot_delay = tk.StringVar(value="0.30")
        self.retries_var = tk.StringVar(value="1")
        for label, variable in (
            ("剪贴板等待", self.clipboard_delay),
            ("确认等待", self.confirm_delay),
            ("脸部刷新等待", self.settle_delay),
            ("截图前等待", self.screenshot_delay),
            ("失败重试", self.retries_var),
        ):
            ttk.Label(row, text=label + ":").pack(side="left", padx=(6, 1))
            ttk.Entry(row, textvariable=variable, width=7).pack(side="left")

        row = ttk.Frame(automation)
        row.pack(fill="x", padx=6, pady=4)
        self.mouse_move_duration = tk.StringVar(value="0.15")
        self.click_hover_delay = tk.StringVar(value="0.20")
        self.click_hold_delay = tk.StringVar(value="0.08")
        self.verification_timeout = tk.StringVar(value="3.00")
        self.inter_variant_delay = tk.StringVar(value="0.50")
        for label, variable in (
            ("鼠标移动", self.mouse_move_duration),
            ("按钮悬停", self.click_hover_delay),
            ("按住时间", self.click_hold_delay),
            ("验证超时", self.verification_timeout),
            ("轮次间隔", self.inter_variant_delay),
        ):
            ttk.Label(row, text=label + ":").pack(side="left", padx=(6, 1))
            ttk.Entry(row, textvariable=variable, width=7).pack(side="left")

        output_frame = ttk.LabelFrame(self.root, text="4. 输出与运行")
        output_frame.pack(fill="x", padx=10, pady=5)
        row = ttk.Frame(output_frame)
        row.pack(fill="x", padx=6, pady=4)
        self.output_var = tk.StringVar(value=str(Path.cwd() / "experiments" / "dna_field_sweeps"))
        ttk.Label(row, text="输出根目录:").pack(side="left")
        ttk.Entry(row, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row, text="选择", command=self._choose_output).pack(side="left", padx=3)
        ttk.Button(row, text="选择恢复会话", command=self._choose_resume).pack(side="left", padx=3)

        row = ttk.Frame(output_frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="测试粘贴当前 DNA", command=self._test_apply).pack(side="left", padx=3)
        ttk.Button(row, text="测试截图", command=self._test_capture).pack(side="left", padx=3)
        self.start_button = ttk.Button(row, text="开始扫描", command=self._start)
        self.start_button.pack(side="left", padx=12)
        self.pause_button = ttk.Button(row, text="暂停", command=self._toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(row, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        ttk.Label(row, text="紧急停止：把鼠标移到屏幕左上角").pack(side="right", padx=8)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(output_frame, variable=self.progress_var, maximum=1)
        self.progress.pack(fill="x", padx=8, pady=4)
        self.progress_text = tk.StringVar(value="等待开始")
        ttk.Label(output_frame, textvariable=self.progress_text).pack(anchor="w", padx=8, pady=(0, 4))
        self.log = self.scrolledtext.ScrolledText(self.root, height=7, state="disabled")
        self.log.pack(fill="both", padx=10, pady=(5, 10))

        self.paste_button: tuple[int, int] | None = None
        self.confirm_button: tuple[int, int] | None = None
        self.verify_button: tuple[int, int] | None = None
        self.screenshot_region: tuple[int, int, int, int] | None = None

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _load_settings(self) -> None:
        value = load_user_settings()
        try:
            settings_version = int(value.get("settings_version", 0))
        except (TypeError, ValueError):
            settings_version = 0
        self.paste_button = self._tuple_or_none(value.get("paste_button"), 2)
        self.confirm_button = self._tuple_or_none(value.get("confirm_button"), 2)
        self.verify_button = self._tuple_or_none(value.get("verify_button"), 2)
        self.screenshot_region = self._tuple_or_none(value.get("screenshot_region"), 4)
        for key, variable in (
            ("clipboard_delay", self.clipboard_delay),
            ("confirm_delay", self.confirm_delay),
            ("settle_delay", self.settle_delay),
            ("screenshot_delay", self.screenshot_delay),
            ("mouse_move_duration", self.mouse_move_duration),
            ("click_hover_delay", self.click_hover_delay),
            ("click_hold_delay", self.click_hold_delay),
            ("verification_timeout", self.verification_timeout),
            ("inter_variant_delay", self.inter_variant_delay),
            ("retries", self.retries_var),
            ("output", self.output_var),
        ):
            if key in value:
                variable.set(str(value[key]))
        if settings_version < SETTINGS_VERSION:
            # Version 1 used instant clicks and unsafe 0.2 s UI waits. Preserve
            # coordinates/output, but migrate timings to the robust minimums.
            for variable, minimum in (
                (self.confirm_delay, 0.80),
                (self.settle_delay, 1.50),
                (self.screenshot_delay, 0.30),
            ):
                try:
                    current = float(variable.get())
                except (TypeError, ValueError):
                    current = minimum
                variable.set(f"{max(current, minimum):.2f}")
        self._refresh_position_labels()

    @staticmethod
    def _tuple_or_none(value: Any, length: int) -> tuple[int, ...] | None:
        if isinstance(value, (list, tuple)) and len(value) == length:
            return tuple(int(item) for item in value)
        return None

    def _save_settings(self) -> None:
        save_user_settings(
            {
                "settings_version": SETTINGS_VERSION,
                "paste_button": self.paste_button,
                "confirm_button": self.confirm_button,
                "verify_button": self.verify_button,
                "screenshot_region": self.screenshot_region,
                "clipboard_delay": self.clipboard_delay.get(),
                "confirm_delay": self.confirm_delay.get(),
                "settle_delay": self.settle_delay.get(),
                "screenshot_delay": self.screenshot_delay.get(),
                "mouse_move_duration": self.mouse_move_duration.get(),
                "click_hover_delay": self.click_hover_delay.get(),
                "click_hold_delay": self.click_hold_delay.get(),
                "verification_timeout": self.verification_timeout.get(),
                "inter_variant_delay": self.inter_variant_delay.get(),
                "retries": self.retries_var.get(),
                "output": self.output_var.get(),
            }
        )

    def _refresh_position_labels(self) -> None:
        self.paste_position.set(str(self.paste_button) if self.paste_button else "未设置")
        self.confirm_position.set(str(self.confirm_button) if self.confirm_button else "未设置（必填）")
        self.verify_position.set(str(self.verify_button) if self.verify_button else "未设置（可选）")
        self.region_value.set(str(self.screenshot_region) if self.screenshot_region else "未设置")

    def _load_dna_file(self) -> None:
        path = self.filedialog.askopenfilename(filetypes=[("DNA text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except Exception as error:
            self.messagebox.showerror("读取失败", str(error))
            return
        self.dna_text.delete("1.0", "end")
        self.dna_text.insert("1.0", text)
        self._parse_dna()

    def _paste_dna(self) -> None:
        try:
            import pyperclip

            text = pyperclip.paste()
        except Exception as error:
            self.messagebox.showerror("剪贴板失败", str(error))
            return
        self.dna_text.delete("1.0", "end")
        self.dna_text.insert("1.0", text)
        self._parse_dna()

    def _base_dna(self) -> str:
        text = self.dna_text.get("1.0", "end-1c")
        if not text.strip():
            raise ValueError("请先输入或加载基础 DNA")
        return text

    def _parse_dna(self) -> None:
        try:
            self.record = parse_dna(self._base_dna())
            if not self.record.genes:
                raise ValueError("没有解析到 gene 字段")
        except Exception as error:
            self.record = None
            self.dna_status.set("解析失败")
            self.messagebox.showerror("DNA 解析失败", str(error))
            return
        fields = list(self.record.genes)
        self.field_box["values"] = fields
        self.field_var.set(fields[0])
        self.dna_status.set(f"已解析 {len(fields)} 个 gene 字段")
        self._field_selected()

    def _field_selected(self, _event: Any = None) -> None:
        if self.record is None:
            return
        field = self.field_var.get()
        gene = self.record.genes.get(field)
        if gene is None:
            return
        self.current_gene.set(
            f"当前值: {gene.allele1} {gene.value1} / {gene.allele2} {gene.value2}"
        )
        self.alleles_var.set(gene.allele1)

    def _build_variants(self) -> list[SweepVariant]:
        base = self._base_dna()
        record = parse_dna(base)
        field = self.field_var.get()
        if field not in record.genes:
            raise ValueError("请选择有效 DNA 字段")
        values = parse_value_sequence(self.values_var.get())
        alleles = parse_allele_sequence(self.alleles_var.get(), record.genes[field].allele1)
        return build_sweep_variants(base, field, alleles, values)

    def _preview(self) -> None:
        try:
            self.variants = self._build_variants()
        except Exception as error:
            self.messagebox.showerror("计划无效", str(error))
            return
        preview = "\n".join(variant.variant_id for variant in self.variants[:8])
        if len(self.variants) > 8:
            preview += f"\n... 另有 {len(self.variants) - 8} 个"
        self.messagebox.showinfo("扫描计划", f"共 {len(self.variants)} 个变体:\n\n{preview}")

    def _import_pyautogui(self) -> Any:
        try:
            import pyautogui
        except ImportError as error:
            raise RuntimeError("缺少 pyautogui，请先安装 requirements.txt") from error
        pyautogui.FAILSAFE = True
        return pyautogui

    def _record_position(self, kind: str) -> None:
        self.messagebox.showinfo("记录位置", "点击确定后有 3 秒，请把鼠标移动到目标按钮中心。")
        self.root.withdraw()
        try:
            time.sleep(3)
            position = tuple(int(value) for value in self._import_pyautogui().position())
        finally:
            self.root.deiconify()
            self.root.lift()
        if kind == "paste":
            self.paste_button = position
        elif kind == "confirm":
            self.confirm_button = position
        else:
            self.verify_button = position
        self._refresh_position_labels()
        self._save_settings()

    def _record_region(self) -> None:
        points = []
        for label in ("截图区域左上角", "截图区域右下角"):
            self.messagebox.showinfo("记录截图区域", f"点击确定后有 3 秒，请把鼠标移动到{label}。")
            self.root.withdraw()
            try:
                time.sleep(3)
                points.append(tuple(int(value) for value in self._import_pyautogui().position()))
            finally:
                self.root.deiconify()
                self.root.lift()
        (left, top), (right, bottom) = points
        if right <= left or bottom <= top:
            self.messagebox.showerror("截图区域无效", "右下角必须位于左上角的右下方")
            return
        self.screenshot_region = (left, top, right - left, bottom - top)
        self._refresh_position_labels()
        self._save_settings()

    def _clear_verify_position(self) -> None:
        self.verify_button = None
        self._refresh_position_labels()
        self._save_settings()

    def _automation_config(self) -> AutomationConfig:
        if self.paste_button is None:
            raise ValueError("请先记录游戏中的粘贴 DNA 按钮位置")
        if self.confirm_button is None:
            raise ValueError("请先记录粘贴后弹窗中的“确定”按钮位置")
        if self.screenshot_region is None:
            raise ValueError("请先设置截图区域")
        try:
            config = AutomationConfig(
                paste_button=self.paste_button,
                confirm_button=self.confirm_button,
                verify_copy_button=self.verify_button,
                screenshot_region=self.screenshot_region,
                clipboard_delay=float(self.clipboard_delay.get()),
                confirm_delay=float(self.confirm_delay.get()),
                settle_delay=float(self.settle_delay.get()),
                screenshot_delay=float(self.screenshot_delay.get()),
                mouse_move_duration=float(self.mouse_move_duration.get()),
                click_hover_delay=float(self.click_hover_delay.get()),
                click_hold_delay=float(self.click_hold_delay.get()),
                verification_timeout=float(self.verification_timeout.get()),
                inter_variant_delay=float(self.inter_variant_delay.get()),
                retries=int(self.retries_var.get()),
            )
        except ValueError as error:
            raise ValueError("延迟和重试次数必须是有效数字") from error
        config.validate()
        return config

    def _choose_output(self) -> None:
        path = self.filedialog.askdirectory(initialdir=self.output_var.get())
        if path:
            self.output_var.set(path)
            self.session_override = None

    def _choose_resume(self) -> None:
        path = self.filedialog.askdirectory(initialdir=self.output_var.get())
        if path:
            self.session_override = Path(path)
            self._log(f"将恢复会话: {path}；当前 DNA 和扫描计划必须与 session.json 一致")

    def _test_apply(self) -> None:
        try:
            base = self._base_dna()
            field = self.field_var.get()
            if field not in parse_dna(base).genes:
                raise ValueError("请先解析 DNA 并选择一个有效字段")
            config = self._automation_config()
            backend = WindowsAutomationBackend(config)
        except Exception as error:
            self.messagebox.showerror("设置无效", str(error))
            return
        self.messagebox.showinfo("测试粘贴", "点击确定后将把当前基础 DNA 粘贴到游戏。")
        try:
            backend.apply_dna(base, field)
        except Exception as error:
            self.messagebox.showerror("测试失败", str(error))
        else:
            self.messagebox.showinfo("测试完成", "DNA 已粘贴；请人工确认游戏中的脸已刷新。")

    def _test_capture(self) -> None:
        try:
            config = self._automation_config()
            backend = WindowsAutomationBackend(config)
            output = Path(self.output_var.get()).expanduser().resolve() / "test_region.png"
            backend.capture(output)
        except Exception as error:
            self.messagebox.showerror("截图失败", str(error))
        else:
            self.messagebox.showinfo("截图完成", f"已保存到:\n{output}")

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            variants = self._build_variants()
            config = self._automation_config()
            output_root = Path(self.output_var.get()).expanduser().resolve()
            if self.session_override is not None:
                session_dir = self.session_override
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = output_root / f"{timestamp}_{safe_component(variants[0].field)}"
            prepare_session(session_dir, self._base_dna(), variants, config)
            backend = WindowsAutomationBackend(config)
            self._save_settings()
        except Exception as error:
            self.messagebox.showerror("无法开始", str(error))
            return

        self.variants = variants
        self.stop_event.clear()
        self.pause_event.clear()
        self.progress.configure(maximum=max(1, len(variants)))
        self.progress_var.set(0)
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="暂停")
        self.stop_button.configure(state="normal")
        self._log(f"会话目录: {session_dir}")
        self._log(f"开始扫描 {variants[0].field}，共 {len(variants)} 个变体")

        def progress(done: int, total: int, variant: SweepVariant, status: str) -> None:
            self.root.after(
                0,
                lambda d=done, t=total, v=variant, s=status: self._update_progress(d, t, v, s),
            )

        def worker() -> None:
            try:
                result = run_sweep(
                    variants,
                    session_dir,
                    backend,
                    retries=config.retries,
                    identical_render_limit=(3 if config.verify_copy_button is None else 0),
                    pause_event=self.pause_event,
                    stop_event=self.stop_event,
                    on_progress=progress,
                )
            except Exception as error:
                self.root.after(0, lambda e=error: self._finish(error=e))
            else:
                self.root.after(0, lambda r=result: self._finish(result=r))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _update_progress(self, done: int, total: int, variant: SweepVariant, status: str) -> None:
        self.progress_var.set(done)
        self.progress_text.set(
            f"{done}/{total}  {variant.field}  {variant.allele}={variant.value}  [{status}]"
        )
        if status in {"completed", "failed"}:
            self._log(f"{variant.variant_id}: {status}")

    def _toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="暂停")
            self._log("继续运行")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="继续")
            self._log("将在当前步骤完成后暂停")

    def _stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self._log("已请求停止，将在当前步骤完成后停止")

    def _finish(self, result: SweepResult | None = None, error: Exception | None = None) -> None:
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled", text="暂停")
        self.stop_button.configure(state="disabled")
        if error is not None:
            self.progress_text.set(f"运行失败: {error}")
            self._log(f"运行失败: {type(error).__name__}: {error}")
            self.messagebox.showerror("运行失败", str(error))
        elif result is not None and result.stopped:
            self.progress_text.set("已停止，可选择同一会话目录恢复")
            self._log(f"已停止；本次完成/跳过 {result.completed}/{result.skipped}")
        elif result is not None:
            self.progress_text.set("扫描完成")
            self._log(f"扫描完成；完成 {result.completed}，其中恢复跳过 {result.skipped}")
            self.messagebox.showinfo("完成", "DNA 单字段扫描与截图已完成")


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except ImportError as error:
        raise SystemExit("当前 Python 缺少 tkinter，无法启动 GUI") from error
    root = tk.Tk()
    DnaFieldSweepApp(
        root,
        {
            "tk": tk,
            "ttk": ttk,
            "filedialog": filedialog,
            "messagebox": messagebox,
            "scrolledtext": scrolledtext,
        },
    )
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
