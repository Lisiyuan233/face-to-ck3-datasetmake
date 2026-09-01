# 如何阅读 DNA 字段扫描结果

本文说明扫描目录、manifest 和综合指标分别代表什么，以及如何据此决定一个字段
能否进入训练。综合结论见 [`DNA_FIELD_SWEEP_RESULTS.md`](DNA_FIELD_SWEEP_RESULTS.md)。

## 1. 推荐阅读顺序

### 第一步：打开字段总表

先打开：

[`experiments/dna_field_sweeps/summary/field_summary.md`](experiments/dna_field_sweeps/summary/field_summary.md)

它每行对应一个 DNA 字段，适合快速寻找：

- 是否完整扫描；
- 完成/期望变体数；
- 有多少张唯一截图；
- 端点变化在全图、正面和侧面分别有多大。

### 第二步：查看字段原始会话

点击总表中的字段名会进入该字段的首选会话目录。重点查看：

```text
session.json       扫描计划、按钮坐标、延迟和期望变体
manifest.jsonl     真正 completed 的变体，一行一条
errors.jsonl       失败记录；没有失败时文件可能不存在
base_dna.txt       所有变体共同使用的基础 DNA
dna/*.txt          每个变体实际粘贴的 DNA
renders/*.png      与 DNA 一一对应的截图
```

### 第三步：比较端点和中点

对连续强度字段，至少打开 `0`、`128`、`255` 三张图：

1. 确认目标局部是否按预期改变；
2. 确认变化方向是否连续；
3. 检查是否同时引起尺度、姿态或其他部位变化；
4. 分别观察正面和侧面。

哈希不同只能证明图片不同，不能证明变化一定来自目标部位，所以最终仍需视觉抽查。

## 2. 字段总表各列含义

### 完成状态

- **完整**：该字段所有期望 `(allele, value)` 都存在有效 completed 记录。
- **部分**：至少有一个有效结果，但没有覆盖全部期望变体。
- **失败**：已尝试但没有可用 completed 结果。
- **未执行**：计划中存在，但没有运行记录。

只有“完整”可以直接进入标准字段分析。“部分”和“失败”必须先读错误信息。

### 变体

例如 `9/9` 表示计划 9 个变体且全部完成。`1/18` 表示计划两个 allele、每个 9
档，但只完成 1 个变体。

### 唯一截图

按 PNG 文件 SHA-256 去重后的数量：

- `9/9` 变体且唯一截图为 `9`：每档至少有一个像素不同；
- 唯一截图少于完成变体：存在游戏量化、饱和或条件门控；
- 多个完成变体但唯一截图为 `1`：当前渲染条件下完全不可见；
- 只有一个完成变体：不能判断视觉变化，需要更多有效档位。

### 视觉状态

- **各档不同**：所有 completed 档位截图哈希不同；
- **存在量化/饱和**：至少两个输入档位生成完全相同的截图；
- **无像素变化**：多个档位全部生成同一张图；
- **样本不足**：只有一个 completed 变体；
- **无图像**：没有 completed 截图。

### 端点 Δ

端点 Δ 的计算是：

```text
同一 allele 的最低已完成 value 截图
与最高已完成 value 截图
逐像素计算 RGB 绝对差
求所有通道和像素的平均值
除以 255，显示为百分比
```

- **全图**：整张正面+侧面组合图；
- **正面**：图片左半部分；
- **侧面**：图片右半部分。

它适合在相同截图条件下对字段做相对排序。它不等于分类准确率、回归误差、互信息
或人眼感知强度，也没有消除背景、衣服、颈部和构图变化。

## 3. 如何判断自动化结果可信

一个 completed 变体同时满足以下条件才进入统一汇总：

1. manifest 状态为 `completed`；
2. DNA 文件存在；
3. 截图文件存在；
4. 实际 DNA 文本 SHA-256 与计划一致；
5. 实际截图 SHA-256 与 manifest 一致；
6. 会话启用了复制 DNA 回读验证。

当前统一结果中，892 个 completed 变体全部通过游戏回读，DNA/截图完整性错误为 0。

需要注意，早期会话可能有重复截图或旧版点击错误。汇总脚本不会简单地按目录数量
相加，而是按 `(field, allele, value)` 合并并选择最新有效记录。

## 4. 如何读取 JSON 汇总

### `field_summary.json`

顶层常用字段：

```json
{
  "field_count": 102,
  "expected_variants": 936,
  "completed_variants": 892,
  "variant_coverage_percent": 95.2991,
  "round_trip_verified_variants": 892,
  "integrity_error_count": 0,
  "field_status_counts": {},
  "visual_status_counts": {},
  "fields": []
}
```

每个 `fields[]` 元素包含：

- `status`：完整性状态；
- `visual_status`：截图去重后的视觉状态；
- `expected_variants` / `completed_variants`；
- `verified_variants`；
- `unique_render_hashes`；
- `endpoint_metrics`：每个 allele 的端点差异；
- `strongest_endpoint_metric`：多个 allele 中全图差异最大的一个；
- `latest_error`：最近失败信息；
- `preferred_session`：字段总表链接使用的首选会话。

### `variant_summary.jsonl`

每行对应主计划中的一个期望变体。推荐用 `(field, allele, value)` 作为唯一键：

```json
{
  "field": "gene_chin_forward",
  "allele": "chin_forward_neg",
  "value": 128,
  "status": "completed",
  "session_path": "...",
  "dna_path": "...",
  "render_path": "...",
  "round_trip_verified": true,
  "dna_hash_ok": true,
  "render_hash_ok": true
}
```

`status: "missing"` 表示该期望变体没有有效 completed 记录。不要仅凭目录中存在 DNA
文件或 PNG 就认为该变体成功，必须以统一 JSONL 或 manifest 的 completed 为准。

## 5. 从结果决定训练策略

### 情况 A：各档不同，端点差异较大

优先纳入训练。先用 `0/128/255` 做小规模验证，再决定是否保留 9 档连续标签。

### 情况 B：各档不同，但端点差异很小

字段确实影响图片，但整图信号弱。可采用：

- 嘴、鼻、眼、耳、额头局部 ROI；
- 轮廓/边缘/关键点辅助输入；
- 提高局部分辨率；
- 按解剖区域设置独立预测头；
- 按字段可辨识度调整 loss 权重。

不要只通过加深整图 backbone 解决，因为信号可能在缩放时已经丢失。

### 情况 C：存在量化或饱和

先找出哈希重复的 value 分组。相同图像不应重复配不同连续标签。可以：

- 每个唯一画面只保留一个代表 value；
- 根据游戏阈值改成分段分类；
- 在真实产生变化的区间增加采样密度。

### 情况 D：无像素变化

当前图像无法监督该字段。先判断是不是条件缺失：

- 头发字段是否真的显示头发；
- 牙齿字段是否张嘴；
- 年龄字段是否处于可显示年龄变化的场景；
- accessory 是否装备并落在截图范围内。

如果改变渲染条件后仍无变化，再从视觉模型目标中排除。

### 情况 E：回读失败或双槽不对称

不要关闭验证强行生成数据。错误中的“期望/实际”说明游戏如何规范化该字段：

- 若只有一槽被恢复，应该固定该槽，只扫描另一槽；
- 若 allele 被替换，字段可能是类别组合而非普通连续强度；
- 若权重被钳制，需要先测出合法范围。

`gene_bs_body_shape`、`gene_bs_bust` 和 `clothes` 属于这一类。

## 6. 正面和侧面怎么读

比较同一字段的 `front_half_percent` 和 `side_half_percent`：

- 侧面明显更高：深度、前突、轮廓类字段依赖侧脸；
- 正面明显更高：宽度、距离、眼睑、鼻翼和嘴宽类字段依赖正脸；
- 两者接近：两个视角都提供有效信息。

半图指标只是快速诊断；左右人物在中线附近可能有少量重叠，正式模型仍应使用项目
现有的固定 front/side crop，而不是直接把组合图机械二等分。

## 7. 结果的适用边界

本轮结论受以下条件限制：

- 只有一份基础 DNA；
- 主要扫描基础 DNA 当前已有的 allele，不代表所有可选 allele；
- 固定人物性别、族群、年龄、光照和相机；
- 端点指标只比较最低/最高值，不证明中间变化单调；
- 表情、头发、accessory 和衣服字段存在场景条件；
- 身高、体型和颈部字段会改变构图，像素差可能高估脸部局部可辨识度。

因此，本报告适合筛选字段、设计模型分支和发现不可学习标签，不应单独作为最终训练
标签体系的唯一依据。

## 8. 重新生成汇总

扫描数据增加后运行：

```powershell
python tools/summarize_dna_field_sweeps.py experiments/dna_field_sweeps
```

默认覆盖更新 `experiments/dna_field_sweeps/summary` 下的三个派生文件，不修改任何
原始 session、DNA、manifest、errors 或截图。

