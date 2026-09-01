# CK3 DNA 字段学习能力与验证集泄漏审计

## 结论

本报告基于 `convnext_tiny_multiview_identifiability_v3` 的 epoch 15 验证结果，
并将每个连续输出与验证集上对 MAE 最有利的常数中位数基线比较。

- 83 个连续目标中，75 个相对常数基线改善至少 20%。
- 1 个目标与常数基线基本相同：`face_detail_chin_cleft` strength。
- 7 个目标仅有 5%～19% 改善，属于弱学习或上下文相关。
- categorical class 中只有 `face_detail_cheek_fat` 有可见验证样本；其余 9 个
  active class head 没有可靠监督，不应作为有效预测输出。
- 当前 sample-id split 存在严重的重复 DNA 泄漏。对第一个 30,000 样本块的
  审计发现，1,500 个验证样本中有 1,338 个在训练集存在完全相同的归一化
  DNA 目标，泄漏率为 89.2%。

因此，当前指标可以证明模型利用了图像与 DNA 的关联，但不能证明它能泛化到
未见过的 DNA。后续正式模型必须使用按归一化目标指纹分组的 split。

## 方法

连续目标的基线不是训练均值，而是直接使用完整验证集计算的逐字段中位数。这是
对常数预测器最有利、且带有验证集信息的乐观基线：

```text
relative_improvement
= (oracle_constant_mae - model_mae) / oracle_constant_mae
```

模型若无法超过该基线，可直接判定没有获得实用图像信号。模型超过该基线仍可能
受到重复 DNA 泄漏影响，因此还要与受控 identifiability tier 联合解释。

## 连续目标结果

| 分类 | 数量 | 解释 |
|---|---:|---|
| 不优于/几乎等于常数 | 1 | 当前不可训练 |
| 改善 0%～5% | 1 | 实用上等同常数 |
| 改善 5%～10% | 3 | 弱学习 |
| 改善 10%～20% | 4 | 边缘学习 |
| 改善至少 20% | 75 | 当前验证中有明确图像信号 |

### 当前不可训练

| 字段 | 输出 | 模型 MAE | 常数 MAE | 改善 |
|---|---|---:|---:|---:|
| `face_detail_chin_cleft` | strength | 0.05275 | 0.05276 | 0.03% |

### 弱学习与边缘学习

| 字段 | 输出 | 改善 | 建议 |
|---|---|---:|---|
| `gene_mouth_corner_depth` | scalar magnitude | 5.0% | 降权；保持 canonical allele |
| `gene_bs_mouth_upper_lip_def` | strength | 7.4% | strength-only；局部嘴唇特征 |
| `gene_mouth_open` | scalar magnitude | 8.0% | 检查动画/截图时序噪声 |
| `gene_bs_mouth_philtrum_def` | strength | 14.4% | strength-only；局部纹理特征 |
| `gene_jaw_forward` | scalar magnitude | 17.4% | 降权；保持 canonical allele |
| `gene_mouth_corner_height` | scalar magnitude | 17.4% | 局部嘴角特征 |
| `gene_bs_mouth_philtrum_width` | signed | 18.6% | conditioned local head |

### 受控实验与模型同时支持的字段

Tier A 共 24 个目标，常数基线改善范围为 22.1%～45.5%；Tier B 共 12 个目标，
改善范围为 23.2%～66.6%。这 36 个目标是下一版模型最可靠的监督信号。

Tier A signed：

```text
gene_bs_nose_size
gene_bs_nose_ridge_angle
gene_bs_cheek_height
gene_bs_nose_forward
gene_bs_forehead_brow_width
gene_bs_forehead_brow_inner_height
gene_bs_ear_size
gene_bs_forehead_brow_outer_height
gene_bs_ear_angle
gene_bs_eye_size
gene_bs_nose_length
gene_bs_nose_tip_angle
gene_bs_forehead_brow_curve
gene_bs_cheek_forward
gene_bs_forehead_brow_forward
gene_bs_nose_height
gene_forehead_brow_height
gene_bs_cheek_width
gene_bs_nose_nostril_width
gene_bs_nose_tip_forward
gene_bs_eye_corner_depth
```

Tier A strength：

```text
face_detail_nasolabial
gene_bs_nose_profile
face_detail_cheek_fat
```

Tier B local/side：

```text
gene_bs_nose_tip_width
gene_bs_ear_outward
gene_bs_mouth_upper_lip_width
face_detail_nose_ridge_def
gene_bs_nose_nostril_height
gene_bs_nose_ridge_width
gene_bs_jaw_def
gene_bs_eye_upper_lid_size
gene_bs_eye_fold_shape
gene_bs_ear_bend
face_detail_cheek_def
face_detail_nose_tip_def
```

## scalar 字段的解释限制

30 个 scalar 输出来自正负 allele 的视觉别名合并。模型只学习 magnitude：

```text
target = abs(original_signed_value)
```

即使 scalar MAE 显著优于常数，也不能恢复原始 allele 方向。推理阶段必须固定
canonical allele，仅写回预测 magnitude。

## categorical class

只有 `face_detail_cheek_fat` 有 65 个达到可见阈值的验证样本：

| 指标 | 常数多数类 | 模型 |
|---|---:|---:|
| Accuracy | 0.2923 | 0.6769 |
| Macro F1 | 0.0905 | 0.6787 |

该字段明显获得了 class 信号，但样本量不足以作为主 checkpoint 选择依据。

以下 9 个 active class head 没有可见验证样本，应视为未训练输出：

```text
gene_bs_ear_bend
gene_bs_eye_fold_shape
gene_bs_nose_profile
face_detail_cheek_def
face_detail_chin_cleft
face_detail_chin_def
face_detail_eye_lower_lid_def
face_detail_eye_socket
face_detail_nasolabial
```

这些字段可以保留 strength 输出；class 应使用模板、多候选或放弃预测。

## 泄漏根因与修复

现有预处理按 `sample_id` 在每个 30,000 样本块内做确定性排列。相同 DNA 的重复
截图有不同 sample ID，因此会跨 train/val/test。图片文件哈希虽然不同，但它们
共享完整 DNA，差异主要可能来自渲染时序、动画或截图压缩。

新 split 使用 schema v2 归一化后的全部训练目标生成 128-bit 指纹，并按指纹决定
split。指纹排除 sample ID、race group 和颜色，但包含：

- 合并后的 30 个 scalar magnitude；
- 37 个 signed 字段；
- 16 个 categorical class；
- 16 个 categorical strength。

因此任何具有相同模型目标的样本都只能出现在同一个 split。

## v4 执行顺序

当前 v3 可以继续跑完，但不能作为无泄漏泛化结论。建议在 v3 结束、磁盘空闲后：

```bash
python tools/build_dna_grouped_split.py \
  --labels face_to_ck3_dataset_male_small/labels.jsonl \
  --schema experiments/dna_identifiability/recommended_training_schema.json \
  --output face_to_ck3_dataset_male_small/grouped_split_v2

python tools/build_training_label_stats.py \
  --data-root face_to_ck3_dataset_male_small/processed_multiview \
  --schema experiments/dna_identifiability/recommended_training_schema.json \
  --split-index face_to_ck3_dataset_male_small/grouped_split_v2/manifest.json \
  --split train \
  --workers 4

python tools/validate_training_setup.py \
  --config configs/train_convnext_tiny_multiview_identifiability_v4_grouped.json

python train.py \
  --config configs/train_convnext_tiny_multiview_identifiability_v4_grouped.json
```

v4 必须从 ImageNet 预训练权重重新开始，不能从 v2/v3 checkpoint 恢复；旧模型已经
看过新验证 split 中的大量重复 DNA，续训无法消除评估污染。

