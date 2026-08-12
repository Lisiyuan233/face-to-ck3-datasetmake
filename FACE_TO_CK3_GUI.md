# 照片到 CK3 DNA GUI

入口脚本：`face_to_ck3_gui.py`

该工具加载当前最优 checkpoint 的 EMA 权重，输入正面和右侧脸照片，输出可直接复制到 CK3 统治者设计器的完整 DNA。

## 首次准备

模型安装在 `.venv-linux` 中，但当前 WSL 缺少 Tk GUI 组件。只需安装一次：

```bash
sudo apt update
sudo apt install python3-tk
```

GUI 会在 WSL 中自动引用 Windows 已安装的微软雅黑/黑体等中文字体，正常情况下不需要另外安装字体。如果启动时仍提示没有中文字体，可执行：

```bash
sudo apt install fonts-noto-cjk
```

Windows 11 的 WSLg 可直接显示 Linux Tk 窗口。安装完成后启动：

```bash
cd /mnt/d/workspace/face-to-ck3-datasetmake
source .venv-linux/bin/activate
python face_to_ck3_gui.py
```

## 输入方式

GUI 支持两种方式：

1. 分别选择正面和右侧脸照片。普通照片会居中裁成训练输入的 2:3 宽高比。
2. 选择本工程 CK3 采集得到的 `1245×829` 组合截图。程序按照 v2 数据集 manifest 自动裁出正面和侧面；同宽高比的缩放图也支持。

最终模型是双视角模型。若只有正脸，可以勾选“正脸替代侧脸”，但侧面相关字段会明显不可靠。推荐拍摄条件：中性表情、无遮挡、均匀光线、头部大小和训练裁图接近。

## DNA 模板

完整 CK3 DNA 不只有模型训练的 83 个面部连续字段，因此必须从一份完整 DNA 模板开始。GUI 默认使用 v2 数据集的 `face_0001.txt`，也可换成自己的角色 DNA。

工具的覆盖规则：

- 默认只覆盖 test 上相对中位数改善至少 25% 的 68 个可靠字段。
- categorical class 没有在最终模型中训练，始终保留模板 allele；可靠的 categorical strength 可以由模型覆盖。
- 颜色、头发、身体、服装、表情及不可预测字段全部保留模板值。
- 模型覆盖的 gene 会同步写入两条 chromosome。

如果希望让模型覆盖更多弱信号字段，可以在 GUI 中选择 76 字段或全部 83 字段策略，但默认 68 字段更稳妥。

## 使用

1. 选择照片。
2. 可选：换成自己的完整 DNA 模板。
3. 点击“生成可粘贴 DNA”。首次加载约 500 MB checkpoint，需要等待几秒。
4. 点击“复制 DNA”，然后在 CK3 统治者设计器中使用粘贴 DNA 功能。
5. “保存 DNA”会同时保存 `.txt` DNA 和一个来源 `.json`。

在 WSLg 下，“复制 DNA”会直接调用 Windows 的 `clip.exe`，不会使用 Tk/X11 剪贴板桥接。这可以避免复制较长 DNA 时出现 `X connection to :0 broken` 并导致 GUI 退出。

## 限制

模型使用 CK3 渲染图训练，并不是在真实人像数据上训练。真实照片与游戏渲染存在域差异，因此 test 集的 MAE/改善率不能直接当作真实照片精度。生成结果适合用作 CK3 初始候选，再在游戏内人工微调或进行渲染闭环比较。
