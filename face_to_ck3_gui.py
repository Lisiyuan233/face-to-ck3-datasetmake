#!/usr/bin/env python3
"""Tk GUI for predicting paste-ready CK3 DNA from portrait images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


def _frozen_self_test_log(message: str) -> None:
    if not getattr(sys, "frozen", False) or "--self-test" not in sys.argv:
        return
    path = Path(sys.executable).with_name("FaceToCK3-self-test.log")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")


_frozen_self_test_log("START: executable entry reached")

from PIL import Image

_frozen_self_test_log("OK: Pillow imported")

from ck3_inference import (
    CK3Predictor,
    build_dna_from_prediction,
    load_field_quality,
    load_preprocessing_manifest,
    prepare_input_views,
)

_frozen_self_test_log("OK: Torch and inference modules imported")


def resource_root() -> Path:
    """Return the source tree or PyInstaller's temporary bundle directory."""

    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


ROOT = resource_root()
DEFAULT_CHECKPOINT = (
    ROOT
    / "runs"
    / "convnext_tiny_multiview_identifiability_v5_small_clean_finetune"
    / "best.pt"
)
DEFAULT_SCHEMA = ROOT / "face_to_ck3_dataset_male_v2" / "recommended_training_schema.json"
DEFAULT_TEMPLATE = ROOT / "face_to_ck3_dataset_male_v2" / "dna" / "face_0001.txt"
DEFAULT_QUALITY = (
    DEFAULT_CHECKPOINT.parent / "test-field-improvement.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "face_to_ck3_dataset_male_v2" / "processed_multiview" / "manifest.json"
)
WSL_FONTCONFIG = ROOT / "wsl-fontconfig.conf"
WINDOWS_CLIP = Path("/mnt/c/Windows/System32/clip.exe")


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and (
        "microsoft" in platform.release().casefold()
        or "WSL_INTEROP" in os.environ
        or "WSL_DISTRO_NAME" in os.environ
    )


def copy_text_to_clipboard(text: str, tk_root: Any | None = None) -> str:
    """Copy text without using the fragile WSLg X11 clipboard bridge."""

    if not text:
        raise ValueError("没有可复制的内容")
    if "\x00" in text:
        raise ValueError("剪贴板内容不能包含 NUL 字符")
    if is_wsl():
        if not WINDOWS_CLIP.is_file():
            raise RuntimeError(f"找不到 Windows 剪贴板程序: {WINDOWS_CLIP}")
        completed = subprocess.run(
            [str(WINDOWS_CLIP)],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5.0,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Windows 剪贴板写入失败（code={completed.returncode}）: {detail}"
            )
        return "windows_clip.exe"
    if tk_root is None:
        raise RuntimeError("当前平台需要 Tk root 才能写入剪贴板")
    tk_root.clipboard_clear()
    tk_root.clipboard_append(text)
    # update_idletasks avoids the full event-loop synchronization that can tear
    # down the X connection under WSLg. WSL never reaches this branch.
    tk_root.update_idletasks()
    return "tk"


def prepare_wsl_fonts() -> None:
    """Expose Windows CJK fonts to Tk before fontconfig is initialized."""

    if (
        sys.platform.startswith("linux")
        and Path("/mnt/c/Windows/Fonts").is_dir()
        and WSL_FONTCONFIG.is_file()
    ):
        os.environ["FONTCONFIG_FILE"] = str(WSL_FONTCONFIG)


def configure_gui_fonts(root: Any) -> str | None:
    """Select a CJK-capable named font for Tk and ttk widgets."""

    from tkinter import font as tkfont
    from tkinter import ttk

    available = {name.casefold(): name for name in tkfont.families(root)}
    preferred = (
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    )
    selected = next(
        (available[name.casefold()] for name in preferred if name.casefold() in available),
        None,
    )
    if selected is None:
        return None
    for name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(name, root=root).configure(family=selected)
        except Exception:
            pass
    try:
        tkfont.nametofont("TkFixedFont", root=root).configure(
            family="Microsoft YaHei" if "microsoft yahei" in available else selected
        )
    except Exception:
        pass
    ttk.Style(root).configure(".", font=(selected, 10))
    root.option_add("*Font", (selected, 10))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_self_test(args: argparse.Namespace) -> None:
    """Exercise bundled resources and one CPU forward pass without opening Tk."""

    resources = {
        "checkpoint": args.checkpoint,
        "schema": args.schema,
        "template": args.template,
        "quality": args.quality,
        "manifest": args.manifest,
    }
    missing = [f"{name}: {path}" for name, path in resources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少内嵌资源：\n" + "\n".join(missing))
    _frozen_self_test_log("OK: bundled resource files found")

    predictor = CK3Predictor(args.checkpoint, args.schema, device="cpu")
    _frozen_self_test_log("OK: checkpoint loaded")
    try:
        width, height = predictor.model_size
        image = Image.new("RGB", (width, height), "#808080")
        prediction = predictor.predict_normalized(image, image.copy())
        _frozen_self_test_log("OK: CPU forward pass completed")
        build_dna_from_prediction(
            template_text=args.template.read_text(encoding="utf-8-sig"),
            prediction=prediction,
            schema_path=args.schema,
            quality=load_field_quality(args.quality),
            minimum_improvement=0.25,
            weight_source=predictor.weight_source,
            used_side_fallback=True,
        )
        load_preprocessing_manifest(args.manifest)
        _frozen_self_test_log("OK: DNA output and preprocessing manifest validated")
    finally:
        predictor.close()


class FaceToCK3GUI:
    def __init__(self, root: Any, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.ImageTk = ImageTk
        self.root = root
        self.root.title("Face to CK3 DNA - Identifiability v5")
        self.root.geometry("1120x800")
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.predictor: CK3Predictor | None = None
        self.predictor_key: tuple[str, str, str] | None = None
        self.last_dna = ""
        self.preview_refs: list[Any] = []

        self.checkpoint = tk.StringVar(value=str(args.checkpoint.resolve()))
        self.schema = tk.StringVar(value=str(args.schema.resolve()))
        self.template = tk.StringVar(value=str(args.template.resolve()))
        self.quality = tk.StringVar(value=str(args.quality.resolve()))
        self.manifest = tk.StringVar(value=str(args.manifest.resolve()))
        self.device = tk.StringVar(value=args.device)
        self.input_mode = tk.StringVar(value="separate")
        self.front_image = tk.StringVar()
        self.side_image = tk.StringVar()
        self.front_as_side = tk.BooleanVar(value=True)
        self.policy = tk.StringVar(value="0.25")
        self.status = tk.StringVar(value="请选择照片。首次推理需加载约 500 MB 模型。")
        self.summary = tk.StringVar(value="尚未推理")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)

    def _path_row(self, parent: Any, label: str, variable: Any, command: Any) -> None:
        row = self.ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        self.ttk.Label(row, text=label, width=12).pack(side="left")
        self.ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        self.ttk.Button(row, text="浏览…", command=command).pack(side="left", padx=(6, 0))

    def _browse_file(self, variable: Any, kinds: list[tuple[str, str]]) -> None:
        path = self.filedialog.askopenfilename(filetypes=kinds)
        if path:
            variable.set(path)
            if variable in (self.front_image, self.side_image):
                self._update_preview()

    def _build(self) -> None:
        outer = self.ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        source = self.ttk.LabelFrame(outer, text="1. 输入照片")
        source.pack(fill="x")
        mode = self.ttk.Frame(source)
        mode.pack(fill="x", padx=6, pady=4)
        self.ttk.Radiobutton(
            mode,
            text="分别选择正面/右侧脸照片",
            variable=self.input_mode,
            value="separate",
            command=self._update_mode,
        ).pack(side="left")
        self.ttk.Radiobutton(
            mode,
            text="CK3 采集组合截图（自动裁剪）",
            variable=self.input_mode,
            value="composite",
            command=self._update_mode,
        ).pack(side="left", padx=18)
        image_types = [("图片", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")]
        self._path_row(
            source,
            "正面/组合图",
            self.front_image,
            lambda: self._browse_file(self.front_image, image_types),
        )
        self.side_row = self.ttk.Frame(source)
        self.side_row.pack(fill="x", padx=6, pady=3)
        self.ttk.Label(self.side_row, text="右侧脸", width=12).pack(side="left")
        self.ttk.Entry(self.side_row, textvariable=self.side_image).pack(
            side="left", fill="x", expand=True
        )
        self.ttk.Button(
            self.side_row,
            text="浏览…",
            command=lambda: self._browse_file(self.side_image, image_types),
        ).pack(side="left", padx=(6, 0))
        self.side_fallback_check = self.ttk.Checkbutton(
            source,
            text="未提供侧脸时用正脸替代（可以运行，但侧面相关字段精度会下降）",
            variable=self.front_as_side,
        )
        self.side_fallback_check.pack(anchor="w", padx=18, pady=(1, 5))

        preview = self.ttk.Frame(source)
        preview.pack(fill="x", padx=6, pady=(2, 6))
        self.front_preview = self.ttk.Label(preview, text="正面预览", anchor="center")
        self.front_preview.pack(side="left", fill="both", expand=True, padx=(0, 3))
        self.side_preview = self.ttk.Label(preview, text="侧脸预览", anchor="center")
        self.side_preview.pack(side="left", fill="both", expand=True, padx=(3, 0))

        settings = self.ttk.LabelFrame(outer, text="2. 模型、DNA 模板与字段策略")
        settings.pack(fill="x", pady=(8, 0))
        self._path_row(
            settings,
            "最优模型",
            self.checkpoint,
            lambda: self._browse_file(self.checkpoint, [("PyTorch checkpoint", "*.pt")]),
        )
        self._path_row(
            settings,
            "DNA 模板",
            self.template,
            lambda: self._browse_file(self.template, [("DNA 文本", "*.txt"), ("所有文件", "*.*")]),
        )
        options = self.ttk.Frame(settings)
        options.pack(fill="x", padx=6, pady=4)
        self.ttk.Label(options, text="字段策略", width=12).pack(side="left")
        policies = (
            ("可靠字段：68 个，test 改善 ≥25%（推荐）", "0.25"),
            ("含弱信号：76 个，test 改善 ≥10%", "0.10"),
            ("全部连续字段：83 个", "0.0"),
        )
        for text, value in policies:
            self.ttk.Radiobutton(
                options, text=text, variable=self.policy, value=value
            ).pack(side="left", padx=(0, 14))
        device_row = self.ttk.Frame(settings)
        device_row.pack(fill="x", padx=6, pady=(0, 5))
        self.ttk.Label(device_row, text="设备", width=12).pack(side="left")
        self.ttk.Combobox(
            device_row,
            textvariable=self.device,
            values=("auto", "cuda", "cpu"),
            state="readonly",
            width=8,
        ).pack(side="left")
        self.ttk.Label(
            device_row,
            text="颜色、身体、服装和 categorical class 均保留模板；模型只覆盖所选面部连续字段。",
        ).pack(side="left", padx=15)

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=8)
        self.run_button = self.ttk.Button(action, text="生成可粘贴 DNA", command=self._start)
        self.run_button.pack(side="left")
        self.copy_button = self.ttk.Button(action, text="复制 DNA", command=self._copy, state="disabled")
        self.copy_button.pack(side="left", padx=6)
        self.save_button = self.ttk.Button(action, text="保存 DNA…", command=self._save, state="disabled")
        self.save_button.pack(side="left")
        self.ttk.Label(action, textvariable=self.summary).pack(side="right")

        output = self.ttk.LabelFrame(outer, text="3. 可直接粘贴到 CK3 的完整 DNA")
        output.pack(fill="both", expand=True)
        self.output = self.scrolledtext.ScrolledText(output, wrap="none", height=16)
        self.output.pack(fill="both", expand=True, padx=6, pady=6)
        self.ttk.Label(outer, textvariable=self.status, anchor="w").pack(fill="x", pady=(5, 0))
        self.ttk.Label(
            outer,
            text="注意：模型以 CK3 渲染图训练；真实人物照片存在域差异。正/侧脸应为中性表情、无遮挡、接近训练构图。",
            foreground="#8a4b00",
        ).pack(fill="x", pady=(3, 0))

    def _update_mode(self) -> None:
        composite = self.input_mode.get() == "composite"
        state = "disabled" if composite else "normal"
        for child in self.side_row.winfo_children()[1:]:
            try:
                child.configure(state=state)
            except Exception:
                pass
        self.side_fallback_check.configure(state=state)
        self._update_preview()

    def _preview_image(self, image: Image.Image) -> Any:
        copy = image.copy()
        copy.thumbnail((270, 190), Image.Resampling.LANCZOS)
        return self.ImageTk.PhotoImage(copy)

    def _prepared_views(self) -> tuple[Image.Image, Image.Image, bool]:
        front_path = Path(self.front_image.get())
        if not front_path.is_file():
            raise FileNotFoundError("请选择有效的正面照片或组合截图")
        manifest = load_preprocessing_manifest(self.manifest.get())
        crops = manifest["crops"]
        source_sizes = manifest.get("source_sizes", [])
        expected_size = tuple(source_sizes[0]) if source_sizes else None
        return prepare_input_views(
            front_path,
            side_path=self.side_image.get().strip() or None,
            composite=self.input_mode.get() == "composite",
            front_crop=tuple(crops["front"]),
            side_crop=tuple(crops["side"]),
            expected_size=expected_size,
            model_size=(256, 384),
            allow_front_as_side=bool(self.front_as_side.get()),
        )

    def _update_preview(self) -> None:
        try:
            front, side, _ = self._prepared_views()
        except Exception:
            return
        first = self._preview_image(front)
        second = self._preview_image(side)
        self.preview_refs = [first, second]
        self.front_preview.configure(image=first, text="")
        self.side_preview.configure(image=second, text="")

    def _validate_paths(self) -> None:
        for label, raw in (
            ("模型", self.checkpoint.get()),
            ("schema", self.schema.get()),
            ("DNA 模板", self.template.get()),
            ("字段质量表", self.quality.get()),
            ("预处理 manifest", self.manifest.get()),
        ):
            if not Path(raw).is_file():
                raise FileNotFoundError(f"{label}不存在: {raw}")

    def _start(self) -> None:
        try:
            self._validate_paths()
            views = self._prepared_views()
        except Exception as error:
            self.messagebox.showerror("输入无效", str(error))
            return
        self.run_button.configure(state="disabled")
        self.copy_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.status.set("正在加载模型并推理，请稍候…")
        settings = {
            "checkpoint": self.checkpoint.get(),
            "schema": self.schema.get(),
            "template": self.template.get(),
            "quality": self.quality.get(),
            "device": self.device.get(),
            "minimum_improvement": float(self.policy.get()),
        }
        threading.Thread(
            target=self._worker, args=(settings, views), daemon=True
        ).start()

    def _worker(
        self,
        settings: dict[str, Any],
        views: tuple[Image.Image, Image.Image, bool],
    ) -> None:
        try:
            key = (
                str(Path(settings["checkpoint"]).resolve()),
                str(Path(settings["schema"]).resolve()),
                settings["device"],
            )
            if self.predictor is None or self.predictor_key != key:
                self.predictor = CK3Predictor(
                    settings["checkpoint"],
                    settings["schema"],
                    device=settings["device"],
                )
                self.predictor_key = key
            front, side, fallback = views
            normalized = self.predictor.predict_normalized(front, side)
            result = build_dna_from_prediction(
                template_text=Path(settings["template"]).read_text(
                    encoding="utf-8-sig"
                ),
                prediction=normalized,
                schema_path=settings["schema"],
                quality=load_field_quality(settings["quality"]),
                minimum_improvement=settings["minimum_improvement"],
                weight_source=self.predictor.weight_source,
                used_side_fallback=fallback,
            )
            self.events.put(("success", result))
        except Exception as error:
            self.events.put(("error", (error, traceback.format_exc())))

    def _poll_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "success":
                    self.last_dna = value.dna
                    self.output.delete("1.0", "end")
                    self.output.insert("1.0", value.dna)
                    warning = "；正脸代替侧脸，精度下降" if value.used_side_fallback else ""
                    self.summary.set(
                        f"EMA={value.weight_source == 'ema'}，模型字段 {len(value.predicted_fields)}，模板字段 {len(value.preserved_fields)}{warning}"
                    )
                    self.status.set("DNA 已生成。可复制后在 CK3 统治者设计器中粘贴。")
                    self.run_button.configure(state="normal")
                    self.copy_button.configure(state="normal")
                    self.save_button.configure(state="normal")
                else:
                    error, trace = value
                    self.status.set("推理失败")
                    self.run_button.configure(state="normal")
                    self.messagebox.showerror("推理失败", f"{error}\n\n{trace[-1800:]}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _copy(self) -> None:
        if not self.last_dna:
            return
        try:
            backend = copy_text_to_clipboard(self.last_dna, self.root)
        except Exception as error:
            self.messagebox.showerror("复制失败", str(error))
            self.status.set("复制失败；可以使用“保存 DNA”导出文本文件。")
            return
        self.status.set(f"完整 DNA 已复制到剪贴板（{backend}）。")

    def _save(self) -> None:
        if not self.last_dna:
            return
        path = self.filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("DNA 文本", "*.txt")]
        )
        if not path:
            return
        Path(path).write_text(self.last_dna, encoding="utf-8")
        metadata = {
            "checkpoint": self.checkpoint.get(),
            "front_image": self.front_image.get(),
            "side_image": self.side_image.get(),
            "input_mode": self.input_mode.get(),
            "minimum_test_improvement": float(self.policy.get()),
        }
        Path(path).with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.status.set(f"已保存: {path}")

    def _close(self) -> None:
        if self.predictor is not None:
            self.predictor.close()
        self.root.destroy()


def main() -> int:
    args = parse_args()
    if args.self_test:
        try:
            run_self_test(args)
        except Exception:
            if getattr(sys, "frozen", False):
                _frozen_self_test_log(traceback.format_exc())
            raise
        _frozen_self_test_log("PASS: bundled resources loaded and CPU inference completed")
        return 0
    prepare_wsl_fonts()
    try:
        import tkinter as tk
    except ImportError as error:
        raise SystemExit(
            "当前 Python 缺少 tkinter。Ubuntu/WSL 请安装 python3-tk。"
        ) from error
    root = tk.Tk()
    selected_font = configure_gui_fonts(root)
    if selected_font is None:
        root.destroy()
        raise SystemExit(
            "未找到可显示中文的字体。请安装 fonts-noto-cjk 后重新启动："
            " sudo apt install fonts-noto-cjk"
        )
    FaceToCK3GUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
