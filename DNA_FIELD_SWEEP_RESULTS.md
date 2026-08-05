# DNA 字段可辨识度扫描综合结果

本文整合 `experiments/dna_field_sweeps` 下的主扫描、补扫和早期重试会话，给出
字段级覆盖率、视觉可辨识度和后续训练建议。逐字段完整数值见生成的
[`field_summary.md`](experiments/dna_field_sweeps/summary/field_summary.md)，指标定义
和阅读方法见 [`DNA_FIELD_SWEEP_RESULT_GUIDE.md`](DNA_FIELD_SWEEP_RESULT_GUIDE.md)。

## 1. 数据范围与合并规则

本次以包含 102 个字段的最新完整主计划
`20260804_212019_batch.json` 作为字段顺序与期望变体集合，并合并：

- `20260804_212019_batch.json`：字段 1～90 的主扫描；
- `20260805_203040_batch.json`：从 `gene_bs_bust` 开始的补扫；
- `20260805_203124_batch.json`：`gene_age` 到 `clothes` 的补扫；
- 同字段的其他重试会话只在存在更新、完整且文件仍有效的 completed 记录时参与。

合并键为 `(field, allele, value)`。同一变体出现多份 completed 记录时，选择时间
最新、DNA 文件和截图文件均存在的一份。旧版重复截图不会覆盖新版有效记录。

## 2. 总体覆盖与完整性

| 指标 | 结果 |
|---|---:|
| 规划字段 | 102 |
| 规划变体 | 936 |
| 已完成变体 | 892 |
| 变体覆盖率 | 95.30% |
| 完整字段 | 99 |
| 部分完成字段 | 1 |
| 失败字段 | 2 |
| 未尝试字段 | 0 |
| 通过游戏 DNA 回读的 completed 变体 | 892/892 |
| DNA/截图完整性错误 | 0 |

因此，“所有字段都扫描过”是成立的；但其中 3 个特殊字段没有得到普通扫描意义上
的完整数据。可直接用于后续分析的是 99 个完整字段。

## 3. 三个特殊字段

| 字段 | 状态 | 原因 | 建议 |
|---|---|---|---|
| `gene_bs_body_shape` | 0/9，失败 | 写入双槽 `0/0` 后，游戏把第一槽恢复成 `25` | 作为非对称约束字段单独建模，不使用普通双槽同步扫描 |
| `gene_bs_bust` | 1/18，部分 | `32/32` 回读为 `25/32`；基础 DNA 的两个 allele 本来也不同 | 分别固定一槽、只扫描另一槽，不能强制两个槽相同 |
| `clothes` | 0/18，失败 | `western_bedchamber 0/0` 被游戏恢复为 `western_bedchamber 79 / most_clothes 0` | 按服装类别/权重组合处理，不作为连续面部 gene |

这三项的失败是游戏 round-trip 约束，不是剪贴板、确认按钮或截图自动化故障。

## 4. 视觉可辨识度分布

在 99 个完整字段中，以同一 allele 的最低值和最高值截图之间的全图 RGB 平均
绝对差作为相对效应量：

| 端点全图差异 | 字段数 | 建议解释 |
|---|---:|---|
| `>= 1.0%` | 10 | 极强视觉变化，容易辨识，但可能包含缩放/构图变化 |
| `0.5%～1.0%` | 10 | 强变化，适合作为第一阶段连续目标 |
| `0.1%～0.5%` | 41 | 中等变化，正侧双视图通常足够 |
| `0～0.1%` | 32 | 弱变化，需要局部 ROI、关键点或更高分辨率 |
| `0%` | 6 | 当前人物和渲染条件下完全不可见 |

这里的阈值是本项目内的工程分组，不是统计显著性或模型准确率。

### 4.1 视觉变化最大的字段

| 排名 | 字段 | 全图端点差异 |
|---:|---|---:|
| 1 | `gene_height` | 10.4654% |
| 2 | `gene_neck_length` | 5.9108% |
| 3 | `gene_head_height` | 5.5952% |
| 4 | `gene_bs_body_type` | 4.4421% |
| 5 | `gene_head_top_height` | 2.4891% |
| 6 | `gene_head_profile` | 2.1564% |
| 7 | `gene_head_width` | 1.9284% |
| 8 | `gene_forehead_angle` | 1.5503% |
| 9 | `gene_jaw_height` | 1.3939% |
| 10 | `gene_eye_height` | 1.0844% |

`gene_height`、`gene_neck_length` 和 `gene_bs_body_type` 的高差异包含人物尺度、
颈部和肩部变化，不应把它们的高像素差直接理解成“脸部参数更容易精确回归”。

### 4.2 当前画面完全不可见的完整字段

- `gene_age`
- `gene_hair_type`
- `gene_baldness`
- `eye_accessory`
- `teeth_accessory`
- `eyelashes_accessory`

这些结果表示“在当前基础人物、秃头外观、闭嘴状态和截图范围下不可见”，不表示
字段在所有人物上都无效。例如发型和秃顶字段需要显示头发，牙齿字段需要稳定的
张嘴视图，年龄字段可能需要不同年龄上下文或退出固定成年设计器预览。

### 4.3 存在量化或饱和的平台字段

- `gene_mouth_open`：9 个输入档位只有 6 张唯一截图；`0/32/64/96` 完全相同，
  从 `128` 才开始产生可见变化。
- `gene_body_hair`：9 个输入档位对应 7 张唯一截图。

训练时不应把完全相同的图像重复当作不同的连续标签。可以保留游戏真正产生不同
画面的档位，或者把该字段改成阈值/分段分类任务。

### 4.4 极弱字段

端点差异低于 `0.05%` 的代表字段主要集中在唇部细节、鼻部纹理和皱纹：

- `gene_bs_mouth_philtrum_shape`
- `gene_bs_mouth_philtrum_def`
- `gene_bs_mouth_upper_lip_def`
- `gene_bs_mouth_philtrum_width`
- `face_detail_chin_cleft`
- `expression_eye_wrinkles`
- `gene_bs_mouth_upper_lip_full`
- `gene_bs_mouth_upper_lip_profile`
- `gene_bs_mouth_lower_lip_full`
- `face_detail_nose_tip_def`
- `face_detail_nose_ridge_def`
- `expression_forehead_wrinkles`

这些字段是当前直接整图训练的主要瓶颈。建议增加嘴、鼻、眼和额头局部裁剪分支，
而不是继续单纯增大整图模型。

## 5. 正面与侧面的贡献

正侧双视图应继续保留。结果中侧面差异明显大于正面的代表字段包括：

- `gene_head_profile`
- `gene_jaw_forward`
- `gene_jaw_angle`
- `gene_chin_forward`
- `gene_eye_depth`
- `gene_bs_jaw_def`
- `face_detail_cheek_def`

其中 `gene_jaw_forward` 的侧面端点差异约为正面的 2.94 倍。若移除侧脸，这类
深度和轮廓字段会显著退化。

正面更重要的字段包括 `gene_head_width`、`gene_head_top_width`、
`gene_eye_distance`、`gene_mouth_width`、`gene_forehead_width` 和多数眼睑、鼻翼、
嘴唇宽度字段。因此不建议用单一完全侧脸替代当前“正面 + 固定侧面”组合。

## 6. 对训练方案的建议

1. **第一阶段目标**：优先使用端点差异 `>=0.1%` 的 61 个完整字段，先验证模型
   能否明显突破当前平台。
2. **第二阶段弱字段**：对 32 个 `0～0.1%` 字段增加局部 ROI、关键点/轮廓图、
   更高分辨率或字段分组专用头。
3. **暂时排除**：6 个当前无像素变化字段，以及 3 个特殊非对称字段，避免给
   模型提供不可学习或错误的监督。
4. **量化字段去重**：`gene_mouth_open`、`gene_body_hair` 只保留产生不同画面的
   档位，或改成分段分类。
5. **多人物复验**：本轮只基于一个基础 DNA。最终字段可辨识度应至少在不同
   性别、族群、年龄和头部形态的基础人物上重复，区分稳定效应与身份交互。
6. **视角规范**：保持当前固定正面和固定侧面；一致的角度、缩放和裁剪比追求
   名义上的精确 90° 更重要。

## 7. 生成的汇总文件

- [`field_summary.md`](experiments/dna_field_sweeps/summary/field_summary.md)：102 字段
  快速总表；
- [`field_summary.json`](experiments/dna_field_sweeps/summary/field_summary.json)：字段级
  完整机器可读指标；
- [`variant_summary.jsonl`](experiments/dna_field_sweeps/summary/variant_summary.jsonl)：
  936 个期望变体的规范记录，包含 completed/missing 状态和实际文件路径；
- [`summarize_dna_field_sweeps.py`](tools/summarize_dna_field_sweeps.py)：重新生成以上
  汇总的只读脚本。

