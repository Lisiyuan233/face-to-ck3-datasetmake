# CK3 DNA 单字段扫描与截图工具

[`dna_field_sweep_tool.py`](dna_field_sweep_tool.py) 用于受控改变一份 CK3
DNA 中的单个 gene，把每个变体粘贴到游戏，并依次保存截图和精确对应的
DNA 文本。它适合生成 DNA 字段可辨识度实验数据，不会修改原有的
`face_to_ck3_tool.py` 数据采集流程。

## 启动

仅支持在能够控制 CK3 窗口的 Windows 桌面 Python 中运行：

```powershell
python -m pip install -r requirements.txt
python dna_field_sweep_tool.py
```

PyAutoGUI 启用了安全停止。自动化过程中把鼠标快速移到主屏幕左上角，会在
当前操作处触发紧急停止。

## 首次设置

1. 在文本框中粘贴 DNA，或加载现有 `face_*.txt`。
2. 点击“解析 DNA”，从下拉框选择要扫描的 gene。
3. 输入数值序列：
   - 离散列表：`0,32,64,128,192,255`；
   - 包含终点的范围：`0:255:32`。
4. Allele 留空或保留当前 allele，即只扫描强度；输入多个 allele，例如
   `chin_width_neg,chin_width_pos`，会对每个 allele 扫描完整数值序列。
5. 点击“记录粘贴 DNA 按钮”，关闭提示框后在 3 秒内把鼠标移动到游戏按钮
   中心。
6. 如果游戏还会弹出确认按钮，记录确认按钮；否则保持未设置。
7. 建议记录“复制 DNA 验证按钮”。启用后，每次粘贴后工具会让游戏重新复制
   DNA，并检查目标字段确实等于期望 allele 和数值，然后才截图。
8. 设置组合肖像截图的左上角和右下角。
9. 先分别运行“测试粘贴当前 DNA”和“测试截图”，人工确认位置、刷新等待时间
   与截图区域正确。

所有位置和延迟保存在 `%APPDATA%/CK3DNAFieldSweep/settings.json`，不会写入
仓库。游戏窗口位置或显示缩放改变后必须重新记录。

## 自动化顺序

每个变体严格执行：

```text
生成只改变目标字段的 DNA
→ 写入剪贴板
→ 点击游戏“粘贴 DNA”
→ 可选点击确认按钮
→ 等待脸部刷新
→ 可选复制并校验游戏内 DNA
→ 截图
→ 写入 manifest.jsonl
```

两个染色体槽始终被写成相同的 allele 和数值。生成器会重新解析输出，并确认
其他 gene 和颜色字段没有变化。

## 输出结构

每次新运行自动创建独立时间戳目录：

```text
experiments/dna_field_sweeps/
  20260802_120000_gene_chin_width/
    session.json
    base_dna.txt
    manifest.jsonl
    errors.jsonl                 # 仅发生错误时出现
    dna/
      00001_gene_chin_width_chin_width_neg_000.txt
    renders/
      00001_gene_chin_width_chin_width_neg_000.png
```

截图成功并落盘后才会向 `manifest.jsonl` 写入 `completed`。每行记录字段、
allele、原始 0～255 数值、DNA SHA-256、DNA 路径、截图路径和时间戳。

## 暂停、停止与恢复

- “暂停”和“停止”都在当前粘贴/截图步骤完成后生效，避免产生半条记录。
- 持久错误会写入 `errors.jsonl` 并停止，不会跳过后继续制造错位标签。
- 恢复时先加载相同的基础 DNA，选择相同字段、allele 和数值序列，再点击
  “选择恢复会话”指向原会话目录。
- 工具会校验计划 SHA-256，计划不同会拒绝混写；已存在有效截图和 completed
  manifest 的变体会自动跳过。

## 建议的可辨识度实验序列

第一轮可以为 signed 字段输入正负两个 allele，并使用：

```text
0,128,255
```

这样会得到负 allele 的 `0/128/255` 与正 allele 的 `0/128/255`，可以同时测试
零强度等价性、方向与极端视觉效果。categorical 字段则输入该字段的所有
allele，同样扫描 `0,128,255`。

正式批量前，至少完成以下人工门禁：

1. 连续测试 10 个变体并核对截图与 manifest；
2. 检查游戏是否处于固定相机、灯光和无动画干扰状态；
3. 启用复制 DNA 验证，或抽查游戏复制回来的 DNA；
4. 截图文件名、DNA 文件名和 manifest 的 `variant_id` 必须一致。
