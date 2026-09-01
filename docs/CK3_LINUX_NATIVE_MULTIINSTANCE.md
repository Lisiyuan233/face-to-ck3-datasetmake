# 256ram Linux 原生多实例采集研究

研究时间：2026-09-01 19:50（北京时间）| 结论：**可行**，且资源效率优于 VM 路线；采集器零代码改动，最大不确定项是 CK3 在 Wine 下的渲染质量。

## 一、关键发现：采集器本来就是跨平台的

审查 `face_to_ck3_tool.py` + `ck3_collection.py`：
- 剪贴板：`pyperclip`（Linux 走 xclip/xsel 后端）——DNA 复制校验流程完整可用
- 鼠标/截图：`pyautogui`（click 绝对坐标 + screenshot(region)），X11 全支持
- UI：tkinter 进度窗（X 下正常）
- **全文件零 win32/pywin32 依赖**

"移植成本"实际 = 装 Linux 依赖（pip + apt），不改代码。

## 二、架构：每实例一个 Xorg（天然隔离）

- 实例 = 独立无头 Xorg（绑一张 GPU 硬件 GL）+ openbox + CK3@Wine + 采集器（DISPLAY 指向）
- **每个 X server 有独立剪贴板与坐标系** → 绝对坐标冲突、剪贴板串号两个核心问题自动消失
- x11vnc 挂各 DISPLAY 用于标定/监护；pulseaudio null sink 供音频占位
- Xvfb 方案不可行：软渲染（llvmpipe）带不动 CK3 的 3D

## 三、资源账（当前状态：GPU 0/1 空闲，83:00.0 在 VM 里）

| 项 | 预算 |
|---|---|
| GPU | 一卡一实例为保守起点；VRAM 每实例 2~3GB（22G 卡余量大，一卡多 Xorg 可再提密度） |
| CPU | 每实例约 2~3 线程（CK3+Wine+采集器），40 线程可容 4~6 实例 |
| 吞吐 | 2 实例 ≈ 18.6k/天；VM 试点结束释放第三张卡后 3 实例 ≈ 28k/天 |

## 四、落地步骤与工作量

1. `apt install wine xorg openbox x11vnc xdotool xclip` + pip 装 pyperclip/pyautogui（~1h）
2. Xorg 无头配置：Xwrapper.config `allowed_users=anybody` + nvidia `--allow-empty-initial-configuration`，每卡一个 DISPLAY（~1h，需 root）
3. CK3 P2P 包解压 + Wine 前缀，绕过 Paradox 启动器，验证 3D 渲染（**1~3h，未知项最多**）
4. 每实例经 x11vnc 标定按钮坐标（15 分钟/实例）
5. 试采 10~20 样本，对照 Windows 机的画质/速率基线

## 五、风险

1. **Wine 渲染质量与 Windows 的差异**：若画面对比度/清晰度有系统性偏差，`render_min_contrast` 等稳定性阈值需微调；极端情况整条路线否决——所以必须先做单实例 PoC
2. P2P 版启动器行为未知（大概率已绕过）
3. Wine 剪贴板与 X selection 的同步（winex11 原生支持，预期无问题，PoC 验证）

## 六、与 VM 路线对比

| | VM（ck3-collect1 已跑） | Linux 原生 |
|---|---|---|
| 代码改动 | 0 | 0（仅依赖） |
| GPU | 1 卡/实例独占 | 共享，可一卡多实例 |
| 每实例内存 | ~6-8GB（Win10 全栈） | ~3-4GB（Wine 栈） |
| Windows 许可 | 悬而未决 | 不需要 |
| 游戏来源 | 共享上的 P2P 包/本体 | 同一 P2P 包（Wine 跑） |
| 主要风险 | 安装交互繁琐 | Wine 渲染质量 |

## 七、建议

VM pilot 继续走完（验证端到端速率与采集质量基线）；**同时在 GPU 0 上做 Linux 原生单实例 PoC**——同机采 20 个样本与 Windows 基线对比画质和速率。渲染无差异 → 规模化全走 Linux 原生（GPU 1 也用上，VM 退役或留作备份）；有差异 → 回退 VM 路线扩到 3 实例。

## 八、PoC 执行记录（2026-09-01 20:40 北京时间，全部通过）

| 环节 | 结果 |
|---|---|
| 依赖安装 | wine64 + xorg + openbox + x11vnc + xdotool + xclip + scrot + python3-tk；pip 装 pyperclip/pyautogui（pyscreeze 需 XDG_SESSION_TYPE=x11） |
| 无头 Xorg | `:90` 绑 GPU 0（03:00.0），root 启动；nvidia_drv.so 在 Ubuntu 专路径，xorg.conf 加 ModulePath 即可；GL 验证 = RTX 2080 Ti / OpenGL 4.6 / 595.84 |
| CK3@Wine | P2P 包解压至 `~/ck3inst1/ck3game`（纯 ASCII 路径），`binaries/ck3.exe` 直启绕过启动器；主菜单正常渲染（GPU 显存 2.7G） |
| 截图链路 | pyautogui.screenshot ✓（scrot 后端） |
| 剪贴板链路 | pyperclip 写入 :90 的 X CLIPBOARD，xclip 读回 ✓（注意 xclip fork 占 selection 会让前台命令不退出，采集器轮询模式不受影响） |
| 远程操作 | x11vnc 挂 :90 → **端口 5902**（VM 在 5901） |

启动器脚本：`scripts/collect_linux_inst1.sh`（含全部环境变量）。

### 待用户经 VNC 5902 完成
1. 进游戏 → 统治者设计器/人物创建，摆到采集界面（性别按需）；
2. 运行采集器完成按钮标定（`bash scripts/collect_linux_inst1.sh 试采目录`）；
3. 试采 10~20 样本，对照 Windows 基线比对画质与单样本耗时。

### 唯一未验证环节
游戏内"复制 DNA"→ Wine 剪贴板 → X CLIPBOARD 的同步方向（winex11 原生支持，标定试采时自然验证）。
