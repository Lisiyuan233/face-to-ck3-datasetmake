from __future__ import annotations

import argparse
import json
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
        
        # 延迟设置
        self.clipboard_delay = 0.10
        self.clipboard_timeout = 3.0
        self.ui_update_delay = 1.5
        self.stability_check_delay = 0.50
        self.stability_timeout = 8.0
        self.screenshot_delay = 0.50
        self.randomize_retries = 4
        self.sample_retries = 2

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
        
        self.region = (left, top, width, height)
        
        messagebox.showinfo("设置完成", f"截图区域已设置为: 左上角({left}, {top}), 宽度:{width}, 高度:{height}")
        
        # 截取一张测试图片
        screenshot = pyautogui.screenshot(region=self.region)
        test_path = os.path.join(self.face_dir, "test_region.png")
        screenshot.save(test_path)
        
        result = messagebox.askyesno("确认区域", f"测试截图已保存到 {test_path}\n是否确认使用此区域？")
        if not result:
            self.setup_region()
            
        root.destroy()
    
    def setup_buttons(self):
        """设置按钮位置"""
        root = tk.Tk()
        root.withdraw()
        
        # 设置复制DNA按钮位置
        messagebox.showinfo("设置按钮位置", "请将鼠标移动到'复制DNA'按钮上，按空格键确认")
        self.copy_dna_button_pos = pyautogui.position()
        
        # 设置随机生成外貌按钮位置
        messagebox.showinfo("设置按钮位置", "请将鼠标移动到'随机生成外貌'按钮上，按空格键确认")
        self.random_generate_button_pos = pyautogui.position()
        
        messagebox.showinfo("设置完成", "按钮位置设置完成")
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
        )
    
    def open_settings(self):
        """Configure synchronization checks rather than blind fixed delays."""
        settings_window = tk.Toplevel()
        settings_window.title("采集校验设置")
        settings_window.geometry("460x410")
        settings_window.resizable(False, False)

        specs = (
            ("剪贴板轮询间隔（秒）", "clipboard_delay", float),
            ("剪贴板更新超时（秒）", "clipboard_timeout", float),
            ("随机生成后初始等待（秒）", "ui_update_delay", float),
            ("DNA 稳定复核间隔（秒）", "stability_check_delay", float),
            ("单次稳定等待超时（秒）", "stability_timeout", float),
            ("DNA 稳定后截图等待（秒）", "screenshot_delay", float),
            ("随机生成最大尝试次数", "randomize_retries", int),
            ("样本事务额外重试次数", "sample_retries", int),
        )
        variables = {}
        form = tk.Frame(settings_window)
        form.pack(fill=tk.X, padx=24, pady=14)
        for row, (label, attribute, _converter) in enumerate(specs):
            tk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            variable = tk.StringVar(value=str(getattr(self, attribute)))
            variables[attribute] = variable
            tk.Entry(form, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="e", pady=5
            )

        def save_settings():
            try:
                for _label, attribute, converter in specs:
                    setattr(self, attribute, converter(variables[attribute].get()))
                self.collection_config().validate()
                messagebox.showinfo("成功", "设置已保存")
                settings_window.destroy()
            except (TypeError, ValueError) as error:
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

        progress_window = tk.Toplevel()
        progress_window.title("受校验采集进度")
        progress_window.geometry("560x190")
        progress_window.resizable(False, False)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=count)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_text_var = tk.StringVar()
        progress_text_var.set(
            f"准备从 face_{state.next_index:04d} 开始；只统计校验成功的样本"
        )
        progress_label = tk.Label(progress_window, textvariable=progress_text_var)
        progress_label.pack(pady=5)
        detail_var = tk.StringVar(value="等待工作线程启动")
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
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, path)

        def automation_thread():
            try:
                collector = VerifiedCollector(
                    config,
                    pyautogui_module=pyautogui,
                    pyperclip_module=pyperclip,
                    cancelled=stop_flag.is_set,
                    on_event=lambda message: events.put(("detail", message)),
                )
                previous_dna = state.previous_dna
                for completed in range(count):
                    result = collector.collect_sample(
                        self.base_dir,
                        state.next_index + completed,
                        previous_dna,
                    )
                    previous_dna = result.dna_text
                    write_progress(completed + 1, result)
                    events.put(("progress", completed + 1, result))
                events.put(("done", count))
            except CollectionCancelled:
                events.put(("cancelled",))
            except Exception as error:
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
                            "DNA 截图前后校验通过；"
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
        root.geometry("450x350")
        
        # 欢迎信息
        welcome_label = tk.Label(root, text="欢迎使用Face to CK3数据收集工具", font=("Arial", 12))
        welcome_label.pack(pady=10)
        
        # 说明文本
        info_text = """每个样本执行受校验事务：
1. 随机生成并等待新 DNA
2. 连续两次复制结果一致后截图
3. 截图后再次验证 DNA 未变化
4. 图片和 DNA 原子配对写入
失败会重试，不占用 sample ID"""
        
        info_label = tk.Label(root, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=10)
        
        # 循环次数设置
        count_frame = tk.Frame(root)
        count_frame.pack(pady=5)
        
        tk.Label(count_frame, text="循环次数:").pack(side=tk.LEFT, padx=5)
        
        count_var = tk.StringVar(value="1000")
        count_entry = tk.Entry(count_frame, textvariable=count_var, width=10)
        count_entry.pack(side=tk.LEFT, padx=5)
        
        def validate_count():
            try:
                count = int(count_var.get())
                if count <= 0:
                    raise ValueError("次数必须大于0")
                return count
            except ValueError:
                messagebox.showerror("错误", "请输入有效的正整数")
                return None
        
        # 设置按钮
        setup_frame = tk.Frame(root)
        setup_frame.pack(pady=10)
        
        region_button = tk.Button(setup_frame, text="设置截图区域", command=self.setup_region)
        region_button.pack(side=tk.LEFT, padx=5)
        
        buttons_button = tk.Button(setup_frame, text="设置按钮位置", command=self.setup_buttons)
        buttons_button.pack(side=tk.LEFT, padx=5)
        
        settings_button = tk.Button(setup_frame, text="校验设置", command=self.open_settings)
        settings_button.pack(side=tk.LEFT, padx=5)
        
        # 运行按钮
        def start_automation():
            count = validate_count()
            if count:
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
