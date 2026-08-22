from __future__ import annotations

import argparse
import json
import math
import os
import queue
import time
import sys
import threading

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # Allows headless Linux tests to import collection logic.
    tk = None
    messagebox = None
    ttk = None

try:
    import pyautogui
    import pyperclip
except Exception:  # PyAutoGUI can fail on headless systems without DISPLAY.
    pyautogui = None
    pyperclip = None

from ck3_collection import (
    CollectionCancelled,
    CollectionConfig,
    VerifiedCollector,
    discover_collection_state,
)
from feishu_notifier import FeishuNotificationConfig, FeishuNotifier


SETTINGS_FILENAME = "collection_settings.json"
SETTINGS_VERSION = 3
VALIDATION_SETTING_SPECS = (
    ("剪贴板轮询间隔（秒）", "clipboard_delay", float),
    ("剪贴板更新超时（秒）", "clipboard_timeout", float),
    ("随机生成后初始等待（秒）", "ui_update_delay", float),
    ("DNA 稳定复核间隔（秒）", "stability_check_delay", float),
    ("单次稳定等待超时（秒）", "stability_timeout", float),
    ("DNA 稳定后截图等待（秒）", "screenshot_delay", float),
    ("随机生成最大尝试次数", "randomize_retries", int),
    ("样本事务额外重试次数", "sample_retries", int),
    ("渲染帧复核间隔（秒）", "render_check_delay", float),
    ("渲染稳定等待超时（秒）", "render_stability_timeout", float),
    ("允许的连续帧差", "render_stability_threshold", float),
    ("人物最低对比度", "render_min_contrast", float),
    ("相对历史对比度下限", "render_min_quality_ratio", float),
    ("每个种族的样本数", "race_group_size", int),
    ("种族总数", "race_count", int),
)


class FaceToCK3Tool:
    def __init__(self, base_dir=None):
        self.base_dir = os.path.abspath(
            os.fspath(base_dir)
            if base_dir is not None
            else os.path.join(os.getcwd(), "face_to_ck3_dataset_male_small")
        )
        self.face_dir = os.path.join(self.base_dir, "face")
        self.dna_dir = os.path.join(self.base_dir, "dna")
        
        # 确保目录存在
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.dna_dir, exist_ok=True)
        
        # 配置参数
        self.region = None  # 截图区域 (left, top, width, height)
        self.copy_dna_button_pos = None  # 复制DNA按钮位置
        self.random_generate_button_pos = None  # 随机生成外貌按钮位置
        self.race_button_pos = None  # 打开种族列表的位置
        self.race_first_option_pos = None  # 种族列表第一项（用于推导行距）
        self.race_second_option_pos = None  # 种族列表第二项（用于推导行距）
        self.show_hair_beard_checkbox_pos = None
        self.facial_structure_button_pos = None
        
        # 延迟设置
        self.clipboard_delay = 0.10
        self.clipboard_timeout = 3.0
        self.ui_update_delay = 1.5
        self.stability_check_delay = 0.50
        self.stability_timeout = 8.0
        self.screenshot_delay = 0.50
        self.randomize_retries = 4
        self.sample_retries = 2
        self.render_check_delay = 0.50
        self.render_stability_timeout = 8.0
        self.render_stability_threshold = 2.0
        self.render_min_contrast = 35.0
        self.render_min_quality_ratio = 0.70
        # Kept disabled until the race-layout calibration is completed.  The
        # calibration action enables it atomically with all required positions.
        self.auto_switch_race = False
        self.race_group_size = 30_000
        self.race_count = 17
        self.default_count = 1000

        self.settings_load_error = None
        self.load_settings()

    @property
    def settings_path(self):
        return os.path.join(self.base_dir, SETTINGS_FILENAME)

    @staticmethod
    def _optional_integer_tuple(value, length, label):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(f"{label}必须包含 {length} 个整数")
        result = []
        for item in value:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not float(item).is_integer()
            ):
                raise ValueError(f"{label}必须包含 {length} 个整数")
            result.append(int(item))
        return tuple(result)

    @staticmethod
    def _integer_setting(value, label):
        if isinstance(value, bool):
            raise ValueError(f"{label}必须是整数")
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise ValueError(f"{label}必须是整数")
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label}必须是整数") from error

    def _settings_payload(self):
        return {
            "version": SETTINGS_VERSION,
            "region": self.region,
            "copy_dna_button_pos": self.copy_dna_button_pos,
            "random_generate_button_pos": self.random_generate_button_pos,
            "race_button_pos": self.race_button_pos,
            "race_first_option_pos": self.race_first_option_pos,
            "race_second_option_pos": self.race_second_option_pos,
            "show_hair_beard_checkbox_pos": self.show_hair_beard_checkbox_pos,
            "facial_structure_button_pos": self.facial_structure_button_pos,
            "clipboard_delay": self.clipboard_delay,
            "clipboard_timeout": self.clipboard_timeout,
            "ui_update_delay": self.ui_update_delay,
            "stability_check_delay": self.stability_check_delay,
            "stability_timeout": self.stability_timeout,
            "screenshot_delay": self.screenshot_delay,
            "randomize_retries": self.randomize_retries,
            "sample_retries": self.sample_retries,
            "render_check_delay": self.render_check_delay,
            "render_stability_timeout": self.render_stability_timeout,
            "render_stability_threshold": self.render_stability_threshold,
            "render_min_contrast": self.render_min_contrast,
            "render_min_quality_ratio": self.render_min_quality_ratio,
            "auto_switch_race": self.auto_switch_race,
            "race_group_size": self.race_group_size,
            "race_count": self.race_count,
            "default_count": self.default_count,
        }

    def _normalize_settings(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("设置文件内容必须是 JSON 对象")
        if (
            type(payload.get("version")) is not int
            or payload["version"] not in (1, 2, SETTINGS_VERSION)
        ):
            raise ValueError(f"不支持的设置版本: {payload.get('version')!r}")

        current = self._settings_payload()
        normalized = {
            "version": SETTINGS_VERSION,
            "region": self._optional_integer_tuple(
                payload.get("region", current["region"]), 4, "截图区域"
            ),
            "copy_dna_button_pos": self._optional_integer_tuple(
                payload.get(
                    "copy_dna_button_pos", current["copy_dna_button_pos"]
                ),
                2,
                "复制 DNA 按钮位置",
            ),
            "random_generate_button_pos": self._optional_integer_tuple(
                payload.get(
                    "random_generate_button_pos",
                    current["random_generate_button_pos"],
                ),
                2,
                "随机生成按钮位置",
            ),
            "race_button_pos": self._optional_integer_tuple(
                payload.get("race_button_pos", current["race_button_pos"]),
                2,
                "种族按钮位置",
            ),
            "race_first_option_pos": self._optional_integer_tuple(
                payload.get(
                    "race_first_option_pos", current["race_first_option_pos"]
                ),
                2,
                "种族列表第一项位置",
            ),
            "race_second_option_pos": self._optional_integer_tuple(
                payload.get(
                    "race_second_option_pos", current["race_second_option_pos"]
                ),
                2,
                "种族列表第二项位置",
            ),
            "show_hair_beard_checkbox_pos": self._optional_integer_tuple(
                payload.get(
                    "show_hair_beard_checkbox_pos",
                    current["show_hair_beard_checkbox_pos"],
                ),
                2,
                "显示头发与胡须复选框位置",
            ),
            "facial_structure_button_pos": self._optional_integer_tuple(
                payload.get(
                    "facial_structure_button_pos",
                    current["facial_structure_button_pos"],
                ),
                2,
                "面部结构按钮位置",
            ),
        }
        for _label, attribute, converter in VALIDATION_SETTING_SPECS:
            value = payload.get(attribute, current[attribute])
            if converter is int:
                converted = self._integer_setting(value, attribute)
            else:
                if isinstance(value, bool):
                    raise ValueError(f"{attribute} 的值无效")
                try:
                    converted = converter(value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"{attribute} 的值无效") from error
                if not math.isfinite(converted):
                    raise ValueError(f"{attribute} 必须是有限数值")
            normalized[attribute] = converted

        count = payload.get("default_count", current["default_count"])
        try:
            count = self._integer_setting(count, "循环次数")
        except ValueError as error:
            raise ValueError("循环次数必须是正整数") from error
        if count <= 0:
            raise ValueError("循环次数必须是正整数")
        normalized["default_count"] = count

        auto_switch_race = payload.get(
            "auto_switch_race", current["auto_switch_race"]
        )
        if type(auto_switch_race) is not bool:
            raise ValueError("auto_switch_race 必须是布尔值")
        normalized["auto_switch_race"] = auto_switch_race

        CollectionConfig(
            screenshot_region=normalized["region"] or (0, 0, 1, 1),
            copy_dna_button=normalized["copy_dna_button_pos"] or (0, 0),
            random_generate_button=(
                normalized["random_generate_button_pos"] or (0, 0)
            ),
            clipboard_settle_delay=normalized["clipboard_delay"],
            clipboard_timeout=normalized["clipboard_timeout"],
            ui_settle_delay=normalized["ui_update_delay"],
            stability_check_delay=normalized["stability_check_delay"],
            stability_timeout=normalized["stability_timeout"],
            screenshot_delay=normalized["screenshot_delay"],
            randomize_retries=normalized["randomize_retries"],
            sample_retries=normalized["sample_retries"],
            render_check_delay=normalized["render_check_delay"],
            render_stability_timeout=normalized["render_stability_timeout"],
            render_stability_threshold=normalized[
                "render_stability_threshold"
            ],
            render_min_contrast=normalized["render_min_contrast"],
            render_min_quality_ratio=normalized["render_min_quality_ratio"],
            auto_switch_race=normalized["auto_switch_race"],
            race_group_size=normalized["race_group_size"],
            race_count=normalized["race_count"],
            race_button=normalized["race_button_pos"],
            race_first_option=normalized["race_first_option_pos"],
            race_second_option=normalized["race_second_option_pos"],
            show_hair_beard_checkbox=normalized[
                "show_hair_beard_checkbox_pos"
            ],
            facial_structure_button=normalized["facial_structure_button_pos"],
        ).validate()
        return normalized

    def _apply_settings(self, settings):
        self.region = settings["region"]
        self.copy_dna_button_pos = settings["copy_dna_button_pos"]
        self.random_generate_button_pos = settings["random_generate_button_pos"]
        self.race_button_pos = settings["race_button_pos"]
        self.race_first_option_pos = settings["race_first_option_pos"]
        self.race_second_option_pos = settings["race_second_option_pos"]
        self.show_hair_beard_checkbox_pos = settings[
            "show_hair_beard_checkbox_pos"
        ]
        self.facial_structure_button_pos = settings[
            "facial_structure_button_pos"
        ]
        for _label, attribute, _converter in VALIDATION_SETTING_SPECS:
            setattr(self, attribute, settings[attribute])
        self.default_count = settings["default_count"]
        self.auto_switch_race = settings["auto_switch_race"]

    def _write_settings(self, settings):
        temporary = self.settings_path + ".partial"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(settings, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, self.settings_path)
        except Exception:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass
            raise

    def update_settings(self, **changes):
        """Validate and atomically persist settings before applying them."""
        payload = self._settings_payload()
        payload.update(changes)
        settings = self._normalize_settings(payload)
        self._write_settings(settings)
        self._apply_settings(settings)

    def save_settings(self):
        self.update_settings()

    def load_settings(self):
        """Load dataset-local settings, falling back safely on any error."""
        if not os.path.isfile(self.settings_path):
            return False
        try:
            with open(self.settings_path, "r", encoding="utf-8") as stream:
                settings = self._normalize_settings(json.load(stream))
            self._apply_settings(settings)
            self.settings_load_error = None
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            self.settings_load_error = str(error)
            return False

    @staticmethod
    def _require_gui_dependencies():
        if tk is None or messagebox is None or ttk is None:
            raise RuntimeError("当前 Python 缺少 tkinter，无法启动采集界面")
        if pyautogui is None or pyperclip is None:
            raise RuntimeError(
                "缺少 GUI 依赖，请运行: pip install pyautogui pillow pyperclip"
            )
        
    def setup_region(self):
        """设置截图区域"""
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("设置截图区域", "请将鼠标移动到截图区域的左上角，按空格键确认")
        
        # 获取左上角坐标
        left, top = pyautogui.position()
        
        messagebox.showinfo("设置截图区域", f"左上角坐标: ({left}, {top})\n请将鼠标移动到截图区域的右下角，按空格键确认")
        
        # 获取右下角坐标
        right, bottom = pyautogui.position()
        
        # 计算区域
        width = right - left
        height = bottom - top
        
        region = (left, top, width, height)
        
        messagebox.showinfo("设置完成", f"截图区域已设置为: 左上角({left}, {top}), 宽度:{width}, 高度:{height}")
        
        # 截取一张测试图片
        screenshot = pyautogui.screenshot(region=region)
        test_path = os.path.join(self.face_dir, "test_region.png")
        screenshot.save(test_path)
        
        result = messagebox.askyesno("确认区域", f"测试截图已保存到 {test_path}\n是否确认使用此区域？")
        if not result:
            root.destroy()
            self.setup_region()
            return
        try:
            self.update_settings(region=region)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("保存失败", f"截图区域未保存：{error}")
        root.destroy()
    
    def setup_buttons(self):
        """设置按钮位置"""
        root = tk.Tk()
        root.withdraw()
        
        # 设置复制DNA按钮位置
        messagebox.showinfo("设置按钮位置", "请将鼠标移动到'复制DNA'按钮上，按空格键确认")
        copy_dna_button_pos = tuple(pyautogui.position())
        
        # 设置随机生成外貌按钮位置
        messagebox.showinfo("设置按钮位置", "请将鼠标移动到'随机生成外貌'按钮上，按空格键确认")
        random_generate_button_pos = tuple(pyautogui.position())
        
        try:
            self.update_settings(
                copy_dna_button_pos=copy_dna_button_pos,
                random_generate_button_pos=random_generate_button_pos,
            )
            messagebox.showinfo("设置完成", "按钮位置已保存")
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("保存失败", f"按钮位置未保存：{error}")
        root.destroy()

    def setup_race_buttons(self):
        """Calibrate the race list and the post-selection cleanup controls."""
        root = tk.Tk()
        root.withdraw()

        messagebox.showinfo(
            "设置种族切换",
            "请将鼠标移动到左上角的“种族”按钮上，按空格键确认",
        )
        race_button_pos = tuple(pyautogui.position())

        # Open the menu so its first two rows can be calibrated.  Their vector
        # lets the collector address every race deterministically.
        pyautogui.click(race_button_pos)
        time.sleep(self.ui_update_delay)
        messagebox.showinfo(
            "设置种族切换",
            "种族列表已打开。请将鼠标移动到列表第一项的中央，按空格键确认",
        )
        race_first_option_pos = tuple(pyautogui.position())
        messagebox.showinfo(
            "设置种族切换",
            "请将鼠标移动到列表第二项的中央，按空格键确认（用于计算列表行距）",
        )
        race_second_option_pos = tuple(pyautogui.position())
        messagebox.showinfo(
            "设置种族切换",
            "请将鼠标移动到“显示头发与胡须”复选框上，按空格键确认",
        )
        show_hair_beard_checkbox_pos = tuple(pyautogui.position())
        messagebox.showinfo(
            "设置种族切换",
            "请将鼠标移动到“面部结构”按钮中央，按空格键确认",
        )
        facial_structure_button_pos = tuple(pyautogui.position())

        try:
            self.update_settings(
                race_button_pos=race_button_pos,
                race_first_option_pos=race_first_option_pos,
                race_second_option_pos=race_second_option_pos,
                show_hair_beard_checkbox_pos=show_hair_beard_checkbox_pos,
                facial_structure_button_pos=facial_structure_button_pos,
                auto_switch_race=True,
            )
            messagebox.showinfo(
                "设置完成",
                "种族切换位置已保存，并已启用自动切换种族",
            )
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("保存失败", f"种族切换位置未保存：{error}")
        root.destroy()
    
    def collection_config(self) -> CollectionConfig:
        if self.region is None:
            raise ValueError("截图区域未设置")
        if self.copy_dna_button_pos is None:
            raise ValueError("复制 DNA 按钮位置未设置")
        if self.random_generate_button_pos is None:
            raise ValueError("随机生成外貌按钮位置未设置")
        return CollectionConfig(
            screenshot_region=self.region,
            copy_dna_button=self.copy_dna_button_pos,
            random_generate_button=self.random_generate_button_pos,
            clipboard_settle_delay=self.clipboard_delay,
            clipboard_timeout=self.clipboard_timeout,
            ui_settle_delay=self.ui_update_delay,
            stability_check_delay=self.stability_check_delay,
            stability_timeout=self.stability_timeout,
            screenshot_delay=self.screenshot_delay,
            randomize_retries=self.randomize_retries,
            sample_retries=self.sample_retries,
            render_check_delay=self.render_check_delay,
            render_stability_timeout=self.render_stability_timeout,
            render_stability_threshold=self.render_stability_threshold,
            render_min_contrast=self.render_min_contrast,
            render_min_quality_ratio=self.render_min_quality_ratio,
            auto_switch_race=self.auto_switch_race,
            race_group_size=self.race_group_size,
            race_count=self.race_count,
            race_button=self.race_button_pos,
            race_first_option=self.race_first_option_pos,
            race_second_option=self.race_second_option_pos,
            show_hair_beard_checkbox=self.show_hair_beard_checkbox_pos,
            facial_structure_button=self.facial_structure_button_pos,
        )
    
    def open_settings(self):
        """Configure synchronization checks rather than blind fixed delays."""
        settings_window = tk.Toplevel()
        settings_window.title("采集校验设置")
        settings_window.geometry("520x690")
        settings_window.resizable(False, False)

        variables = {}
        form = tk.Frame(settings_window)
        form.pack(fill=tk.X, padx=24, pady=14)
        for row, (label, attribute, _converter) in enumerate(
            VALIDATION_SETTING_SPECS
        ):
            tk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            variable = tk.StringVar(value=str(getattr(self, attribute)))
            variables[attribute] = variable
            tk.Entry(form, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="e", pady=5
            )

        auto_switch_var = tk.BooleanVar(value=self.auto_switch_race)
        tk.Checkbutton(
            form,
            text="每到种族块边界自动切换种族",
            variable=auto_switch_var,
        ).grid(
            row=len(VALIDATION_SETTING_SPECS),
            column=0,
            columnspan=2,
            sticky="w",
            pady=8,
        )

        def save_settings():
            try:
                changes = {
                    attribute: converter(variables[attribute].get())
                    for _label, attribute, converter in VALIDATION_SETTING_SPECS
                }
                changes["auto_switch_race"] = auto_switch_var.get()
                self.update_settings(**changes)
                messagebox.showinfo("成功", f"设置已保存到 {self.settings_path}")
                settings_window.destroy()
            except (OSError, TypeError, ValueError) as error:
                messagebox.showerror("错误", str(error))

        buttons = tk.Frame(settings_window)
        buttons.pack(pady=8)
        tk.Button(buttons, text="保存", command=save_settings).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(buttons, text="取消", command=settings_window.destroy).pack(
            side=tk.LEFT, padx=6
        )
    
    def run_automation(self, count=1000):
        self._require_gui_dependencies()
        config = self.collection_config()
        try:
            state = discover_collection_state(self.base_dir)
        except Exception as error:
            messagebox.showerror("无法开始", str(error))
            return
        notification_config_error = None
        try:
            feishu_config = FeishuNotificationConfig.from_env()
        except ValueError as error:
            feishu_config = None
            notification_config_error = str(error)
        notifier = FeishuNotifier(feishu_config) if feishu_config else None

        progress_window = tk.Toplevel()
        progress_window.title("受校验采集进度")
        progress_window.geometry("560x190")
        progress_window.resizable(False, False)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=count)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_text_var = tk.StringVar()
        progress_text_var.set(
            f"准备从 face_{state.next_index:04d} 开始；"
            f"每种族 {config.race_group_size} 个样本；"
            f"飞书通知{'已启用' if notifier else '未配置'}"
        )
        progress_label = tk.Label(progress_window, textvariable=progress_text_var)
        progress_label.pack(pady=5)
        detail_var = tk.StringVar(
            value=(
                f"飞书通知未启用：{notification_config_error}"
                if notification_config_error
                else "等待工作线程启动"
            )
        )
        tk.Label(
            progress_window,
            textvariable=detail_var,
            wraplength=520,
            fg="#555555",
        ).pack(pady=3)
        cancel_button = tk.Button(progress_window, text="取消")
        cancel_button.pack(pady=10)

        stop_flag = threading.Event()
        events: queue.Queue[tuple] = queue.Queue()

        def write_progress(completed, result):
            path = os.path.join(self.base_dir, "collection_progress.json")
            temporary = path + ".partial"
            payload = {
                "completed": completed,
                "requested": count,
                "last_sample_id": result.sample_id,
                "last_dna_fingerprint": result.dna_fingerprint,
                "last_render_difference": round(result.render_difference, 4),
                "last_render_contrast": round(result.render_contrast, 4),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, path)

        def automation_thread():
            started_at = time.monotonic()
            last_notification_at = started_at
            last_notified_completed = 0
            completed_count = 0

            def elapsed_text(seconds):
                total = max(0, int(seconds))
                hours, remainder = divmod(total, 3600)
                minutes, secs = divmod(remainder, 60)
                return f"{hours:d}小时{minutes:02d}分{secs:02d}秒"

            def send_notification(message):
                if notifier is None:
                    return False
                try:
                    notifier.send_text(message)
                    return True
                except Exception as error:
                    events.put(("detail", f"飞书通知发送失败：{error}"))
                    return False

            try:
                collector = VerifiedCollector(
                    config,
                    pyautogui_module=pyautogui,
                    pyperclip_module=pyperclip,
                    cancelled=stop_flag.is_set,
                    on_event=lambda message: events.put(("detail", message)),
                )
                send_notification(
                    "[CK3 采集启动]\n"
                    f"数据集：{os.path.basename(self.base_dir)}\n"
                    f"起始样本：face_{state.next_index:04d}\n"
                    f"计划采集：{count} 个\n"
                    f"每种族：{config.race_group_size} 个"
                )
                previous_dna = state.previous_dna
                for completed in range(count):
                    sample_index = state.next_index + completed
                    switched_race = collector.prepare_race_for_sample(sample_index)
                    if switched_race:
                        race_number = (
                            (sample_index - 1) // config.race_group_size + 1
                        )
                        send_notification(
                            "[CK3 种族切换]\n"
                            f"即将采集：face_{sample_index:04d}\n"
                            f"当前种族序号：{race_number}/{config.race_count}"
                        )
                    result = collector.collect_sample(
                        self.base_dir,
                        sample_index,
                        previous_dna,
                    )
                    previous_dna = result.dna_text
                    completed_count = completed + 1
                    write_progress(completed_count, result)
                    events.put(("progress", completed_count, result))
                    finished = completed + 1
                    now = time.monotonic()
                    notify_by_count = (
                        finished - last_notified_completed
                        >= feishu_config.progress_every
                        if feishu_config
                        else False
                    )
                    notify_by_time = (
                        now - last_notification_at
                        >= feishu_config.progress_interval_seconds
                        if feishu_config
                        else False
                    )
                    if notifier and (notify_by_count or notify_by_time):
                        elapsed = now - started_at
                        speed = finished / elapsed * 60 if elapsed > 0 else 0.0
                        remaining = count - finished
                        eta = remaining / speed * 60 if speed > 0 else 0.0
                        last_notification_at = now
                        last_notified_completed = finished
                        send_notification(
                            "[CK3 采集进度]\n"
                            f"进度：{finished}/{count} "
                            f"({finished / count:.1%})\n"
                            f"最近样本：{result.sample_id}\n"
                            f"已运行：{elapsed_text(elapsed)}\n"
                            f"速度：{speed:.1f} 个/分钟\n"
                            f"预计剩余：{elapsed_text(eta)}\n"
                            f"渲染帧差：{result.render_difference:.2f}\n"
                            f"人物对比度：{result.render_contrast:.1f}"
                        )
                elapsed = time.monotonic() - started_at
                send_notification(
                    "[CK3 采集完成]\n"
                    f"本次完成：{count}/{count}\n"
                    f"最后样本：face_{state.next_index + count - 1:04d}\n"
                    f"总耗时：{elapsed_text(elapsed)}"
                )
                events.put(("done", count))
            except CollectionCancelled:
                elapsed = time.monotonic() - started_at
                send_notification(
                    "[CK3 采集已取消]\n"
                    f"已运行：{elapsed_text(elapsed)}\n"
                    "未完成事务不会写入数据集"
                )
                events.put(("cancelled",))
            except Exception as error:
                elapsed = time.monotonic() - started_at
                send_notification(
                    "[CK3 采集异常停止]\n"
                    f"运行时间：{elapsed_text(elapsed)}\n"
                    f"已完成：{completed_count}/{count}\n"
                    f"下一样本：face_{state.next_index + completed_count:04d}\n"
                    f"异常：{str(error)[:1200]}"
                )
                events.put(("error", str(error)))

        def cancel_operation():
            stop_flag.set()
            cancel_button.config(state=tk.DISABLED)
            detail_var.set("正在安全停止；未完成校验的截图不会写入")

        def poll_events():
            try:
                while True:
                    event = events.get_nowait()
                    if event[0] == "detail":
                        detail_var.set(event[1])
                    elif event[0] == "progress":
                        completed, result = event[1], event[2]
                        progress_var.set(completed)
                        progress_text_var.set(
                            f"进度: {completed}/{count}；已保存 {result.sample_id}"
                        )
                        detail_var.set(
                            "DNA 与渲染稳定校验通过；"
                            f"帧差 {result.render_difference:.2f}，"
                            f"人物对比度 {result.render_contrast:.1f}；"
                            f"随机尝试 {result.randomize_attempts} 次，"
                            f"事务尝试 {result.transaction_attempts} 次"
                        )
                    elif event[0] == "done":
                        messagebox.showinfo(
                            "完成", f"已完成 {event[1]} 个唯一且同步校验通过的样本"
                        )
                        progress_window.destroy()
                        return
                    elif event[0] == "cancelled":
                        progress_window.destroy()
                        return
                    elif event[0] == "error":
                        messagebox.showerror("采集停止", event[1])
                        progress_window.destroy()
                        return
            except queue.Empty:
                pass
            if progress_window.winfo_exists():
                progress_window.after(100, poll_events)

        cancel_button.config(command=cancel_operation)
        progress_window.protocol("WM_DELETE_WINDOW", cancel_operation)
        thread = threading.Thread(target=automation_thread, daemon=True)
        thread.start()
        progress_window.after(100, poll_events)
    
    def run(self):
        """运行工具"""
        self._require_gui_dependencies()
        root = tk.Tk()
        root.title("Face to CK3 数据收集工具")
        root.geometry("520x430")

        if self.settings_load_error:
            root.after_idle(
                lambda: messagebox.showwarning(
                    "设置加载失败",
                    f"无法读取 {self.settings_path}，已使用默认值：\n"
                    f"{self.settings_load_error}",
                )
            )
        
        # 欢迎信息
        welcome_label = tk.Label(root, text="欢迎使用Face to CK3数据收集工具", font=("Arial", 12))
        welcome_label.pack(pady=10)
        
        # 说明文本
        info_text = """每个样本执行受校验事务：
1. 随机生成并等待新 DNA
2. 连续两次复制结果一致后截图
3. 连续画面稳定且人物对比度正常后截图
4. 截图后再次验证 DNA 未变化
5. 图片和 DNA 原子配对写入
6. 渲染持续异常时自动停机，不写入当前样本
7. 每到种族块边界：选择下一种族、隐藏头发胡须、打开面部结构
失败会重试，不占用 sample ID"""
        
        info_label = tk.Label(root, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=10)
        
        # 本次采集数量
        count_frame = tk.Frame(root)
        count_frame.pack(pady=5)
        
        tk.Label(count_frame, text="本次采集样本数:").pack(side=tk.LEFT, padx=5)
        
        count_var = tk.StringVar(value=str(self.default_count))
        count_entry = tk.Entry(count_frame, textvariable=count_var, width=10)
        count_entry.pack(side=tk.LEFT, padx=5)

        # Keep the race-block size next to the run count so the two independent
        # values cannot be mistaken for one another.
        race_size_frame = tk.Frame(root)
        race_size_frame.pack(pady=5)
        tk.Label(race_size_frame, text="每个种族样本数:").pack(
            side=tk.LEFT, padx=5
        )
        race_size_var = tk.StringVar(value=str(self.race_group_size))
        tk.Entry(race_size_frame, textvariable=race_size_var, width=10).pack(
            side=tk.LEFT, padx=5
        )
        
        def validate_positive_integer(variable, label):
            try:
                value = int(variable.get())
                if value <= 0:
                    raise ValueError("次数必须大于0")
                return value
            except ValueError:
                messagebox.showerror("错误", f"{label}必须是正整数")
                return None
        
        # 设置按钮
        setup_frame = tk.Frame(root)
        setup_frame.pack(pady=10)
        
        region_button = tk.Button(setup_frame, text="设置截图区域", command=self.setup_region)
        region_button.pack(side=tk.LEFT, padx=5)
        
        buttons_button = tk.Button(setup_frame, text="设置按钮位置", command=self.setup_buttons)
        buttons_button.pack(side=tk.LEFT, padx=5)

        race_buttons_button = tk.Button(
            setup_frame,
            text="设置种族切换",
            command=self.setup_race_buttons,
        )
        race_buttons_button.pack(side=tk.LEFT, padx=5)
        
        settings_button = tk.Button(setup_frame, text="校验设置", command=self.open_settings)
        settings_button.pack(side=tk.LEFT, padx=5)
        
        # 运行按钮
        def start_automation():
            count = validate_positive_integer(count_var, "本次采集样本数")
            if count is None:
                return
            race_group_size = validate_positive_integer(
                race_size_var, "每个种族样本数"
            )
            if race_group_size is None:
                return
            try:
                self.update_settings(
                    default_count=count,
                    race_group_size=race_group_size,
                )
            except (OSError, TypeError, ValueError) as error:
                messagebox.showerror("保存失败", f"采集参数未保存：{error}")
                return
            self.run_automation(count)
        
        run_button = tk.Button(root, text="开始运行", command=start_automation, bg="green", fg="white")
        run_button.pack(pady=10)
        
        # 退出按钮
        quit_button = tk.Button(root, text="退出", command=root.quit)
        quit_button.pack(pady=5)
        
        root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect CK3 image/DNA pairs with synchronization checks."
    )
    parser.add_argument(
        "--base-dir",
        default="face_to_ck3_dataset_male_small",
        help="output dataset directory; use a new directory for recollection",
    )
    args = parser.parse_args()
    try:
        tool = FaceToCK3Tool(args.base_dir)
        tool.run()
    except RuntimeError as error:
        print(error)
        sys.exit(1)
