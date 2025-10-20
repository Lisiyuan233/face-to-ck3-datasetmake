import pyautogui
import os
import time
import random
from PIL import Image
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import pyperclip
import sys
import threading

class FaceToCK3Tool:
    def __init__(self):
        self.base_dir = os.path.join(os.getcwd(), "face_to_ck3_dateset")
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
        self.clipboard_delay = 0.2  # 剪贴板等待时间
        self.ui_update_delay = 0.2  # UI更新等待时间
        self.random_delay_min = 0.1  # 随机延迟最小值
        self.random_delay_max = 0.2  # 随机延迟最大值
        
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
    
    def capture_and_save(self, index):
        """截取屏幕区域并保存"""
        if not self.region:
            raise ValueError("截图区域未设置")
            
        # 截图
        screenshot = pyautogui.screenshot(region=self.region)
        
        # 保存图片
        filename = f"face_{index:04d}.png"
        filepath = os.path.join(self.face_dir, filename)
        screenshot.save(filepath)
        
        return filename
    
    def copy_dna_to_file(self, filename):
        """复制DNA内容到文件"""
        if not self.copy_dna_button_pos:
            raise ValueError("复制DNA按钮位置未设置")
            
        # 点击复制DNA按钮
        pyautogui.click(self.copy_dna_button_pos)
        time.sleep(self.clipboard_delay)  # 使用可配置的剪贴板等待时间
        
        # 获取剪贴板内容
        try:
            dna_content = pyperclip.paste()
        except:
            # 如果pyperclip失败，尝试使用tkinter
            root = tk.Tk()
            root.withdraw()
            try:
                dna_content = root.clipboard_get()
            except:
                dna_content = ""
            root.destroy()
        
        # 保存到txt文件
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(self.dna_dir, txt_filename)
        
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(dna_content)
    
    def click_random_generate(self):
        """点击随机生成外貌按钮"""
        if not self.random_generate_button_pos:
            raise ValueError("随机生成外貌按钮位置未设置")
            
        pyautogui.click(self.random_generate_button_pos)
        time.sleep(self.ui_update_delay)  # 使用可配置的UI更新等待时间
    
    def open_settings(self):
        """打开设置窗口"""
        settings_window = tk.Toplevel()
        settings_window.title("延迟设置")
        settings_window.geometry("400x250")
        settings_window.resizable(False, False)
        
        # 剪贴板延迟设置
        clipboard_frame = tk.Frame(settings_window)
        clipboard_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(clipboard_frame, text="剪贴板等待时间(秒):").pack(side=tk.LEFT)
        clipboard_var = tk.StringVar(value=str(self.clipboard_delay))
        clipboard_entry = tk.Entry(clipboard_frame, textvariable=clipboard_var, width=10)
        clipboard_entry.pack(side=tk.RIGHT)
        
        # UI更新延迟设置
        ui_frame = tk.Frame(settings_window)
        ui_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(ui_frame, text="界面更新等待时间(秒):").pack(side=tk.LEFT)
        ui_var = tk.StringVar(value=str(self.ui_update_delay))
        ui_entry = tk.Entry(ui_frame, textvariable=ui_var, width=10)
        ui_entry.pack(side=tk.RIGHT)
        
        # 随机延迟最小值设置
        random_min_frame = tk.Frame(settings_window)
        random_min_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(random_min_frame, text="随机延迟最小值(秒):").pack(side=tk.LEFT)
        random_min_var = tk.StringVar(value=str(self.random_delay_min))
        random_min_entry = tk.Entry(random_min_frame, textvariable=random_min_var, width=10)
        random_min_entry.pack(side=tk.RIGHT)
        
        # 随机延迟最大值设置
        random_max_frame = tk.Frame(settings_window)
        random_max_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(random_max_frame, text="随机延迟最大值(秒):").pack(side=tk.LEFT)
        random_max_var = tk.StringVar(value=str(self.random_delay_max))
        random_max_entry = tk.Entry(random_max_frame, textvariable=random_max_var, width=10)
        random_max_entry.pack(side=tk.RIGHT)
        
        # 按钮区域
        button_frame = tk.Frame(settings_window)
        button_frame.pack(pady=10)
        
        def save_settings():
            try:
                self.clipboard_delay = float(clipboard_var.get())
                self.ui_update_delay = float(ui_var.get())
                self.random_delay_min = float(random_min_var.get())
                self.random_delay_max = float(random_max_var.get())
                
                # 验证随机延迟范围
                if self.random_delay_min >= self.random_delay_max:
                    messagebox.showerror("错误", "随机延迟最小值必须小于最大值")
                    return
                
                # 验证所有值为正数
                if any(val <= 0 for val in [self.clipboard_delay, self.ui_update_delay, 
                                           self.random_delay_min, self.random_delay_max]):
                    messagebox.showerror("错误", "所有延迟值必须大于0")
                    return
                
                messagebox.showinfo("成功", "设置已保存")
                settings_window.destroy()
                
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        save_button = tk.Button(button_frame, text="保存", command=save_settings)
        save_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="取消", command=settings_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # 预设按钮
        preset_frame = tk.Frame(settings_window)
        preset_frame.pack(pady=5)
        
        def set_fast_preset():
            clipboard_var.set("0.1")
            ui_var.set("0.2")
            random_min_var.set("0.05")
            random_max_var.set("0.15")
        
        def set_normal_preset():
            clipboard_var.set("0.2")
            ui_var.set("0.3")
            random_min_var.set("0.1")
            random_max_var.set("0.3")
        
        def set_slow_preset():
            clipboard_var.set("0.5")
            ui_var.set("1.0")
            random_min_var.set("0.5")
            random_max_var.set("1.5")
        
        tk.Label(preset_frame, text="预设:").pack(side=tk.LEFT, padx=5)
        fast_button = tk.Button(preset_frame, text="快速", command=set_fast_preset)
        fast_button.pack(side=tk.LEFT, padx=2)
        normal_button = tk.Button(preset_frame, text="正常", command=set_normal_preset)
        normal_button.pack(side=tk.LEFT, padx=2)
        slow_button = tk.Button(preset_frame, text="慢速", command=set_slow_preset)
        slow_button.pack(side=tk.LEFT, padx=2)
    
    def run_automation(self, count=1000):
        # 创建进度窗口
        progress_window = tk.Toplevel()
        progress_window.title("进度")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        
        # 进度条
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=count)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        # 进度文本
        progress_text_var = tk.StringVar()
        progress_text_var.set(f"进度: 0/{count} (0%)")
        progress_label = tk.Label(progress_window, textvariable=progress_text_var)
        progress_label.pack(pady=5)
        
        # 取消按钮
        cancel_button = tk.Button(progress_window, text="取消", command=progress_window.quit)
        cancel_button.pack(pady=10)
        
        # 停止标志
        stop_flag = threading.Event()
        
        def automation_thread():
            try:
                for i in range(1, count + 1):
                    if stop_flag.is_set():
                        break
                        
                    # 更新进度
                    progress_var.set(i)
                    percentage = (i / count) * 100
                    progress_text_var.set(f"进度: {i}/{count} ({percentage:.1f}%)")
                    
                    # 1. 截取屏幕区域
                    filename = self.capture_and_save(i)
                    
                    # 2. 复制DNA内容到文件
                    self.copy_dna_to_file(filename)
                    
                    # 3. 点击随机生成外貌
                    self.click_random_generate()
                    
                    # 添加随机延迟，模拟人工操作
                    time.sleep(random.uniform(self.random_delay_min, self.random_delay_max))
                    
                    # 每完成10次更新一次UI
                    if i % 10 == 0:
                        progress_window.update()
                    
                    # 每完成100次保存一次进度
                    if i % 100 == 0:
                        with open(os.path.join(self.base_dir, "progress.txt"), "w") as f:
                            f.write(f"已完成: {i}/{count}")
                
                # 完成后显示消息
                if not stop_flag.is_set():
                    progress_window.after(0, lambda: messagebox.showinfo("完成", f"已完成 {count} 次截图和数据收集"))
                    progress_window.after(0, lambda: progress_window.destroy())
                    
            except Exception as e:
                progress_window.after(0, lambda: messagebox.showerror("错误", f"执行过程中出错: {str(e)}"))
                progress_window.after(0, lambda: progress_window.destroy())
        
        # 启动线程
        thread = threading.Thread(target=automation_thread)
        thread.daemon = True
        thread.start()
        
        # 设置取消按钮功能
        def cancel_operation():
            stop_flag.set()
            progress_window.destroy()
        
        cancel_button.config(command=cancel_operation)
        
        # 确保窗口关闭时停止线程
        progress_window.protocol("WM_DELETE_WINDOW", cancel_operation)
        
        # 更新窗口
        progress_window.update()
    
    def run(self):
        """运行工具"""
        root = tk.Tk()
        root.title("Face to CK3 数据收集工具")
        root.geometry("450x350")
        
        # 欢迎信息
        welcome_label = tk.Label(root, text="欢迎使用Face to CK3数据收集工具", font=("Arial", 12))
        welcome_label.pack(pady=10)
        
        # 说明文本
        info_text = """此工具将自动执行以下操作：
1. 截取指定屏幕区域
2. 复制DNA内容到剪贴板
3. 保存DNA内容到txt文件
4. 点击"随机生成外貌"按钮
5. 重复以上步骤"""
        
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
        
        settings_button = tk.Button(setup_frame, text="延迟设置", command=self.open_settings)
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
    # 检查依赖
    try:
        import pyautogui
        import PIL
        import pyperclip
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请运行以下命令安装依赖:")
        print("pip install pyautogui pillow pyperclip")
        sys.exit(1)
    
    tool = FaceToCK3Tool()
    tool.run()