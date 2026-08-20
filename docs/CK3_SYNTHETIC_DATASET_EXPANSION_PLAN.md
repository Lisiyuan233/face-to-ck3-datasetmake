# CK3 合成数据集扩展方案

更新日期：2026-08-14

相关资料：[`DATA_COLLECTION.md`](../DATA_COLLECTION.md)、
[`DNA_IDENTIFIABILITY.md`](../DNA_IDENTIFIABILITY.md)、
[`FINAL_EVALUATION.md`](../runs/convnext_tiny_multiview_identifiability_v5_small_clean_finetune/FINAL_EVALUATION.md)。

## 1. 范围与结论

本方案只扩展 CK3 内生成并受校验采集的合成数据，当前阶段不包含真人照片采集、
真人 DNA 标注或真人域适配。

推荐采用两条并行数据线：

1. 将现有 12,000 个随机唯一 DNA 样本扩展到 24,000，门禁通过后再决定是否扩到
   48,000；
2. 先生成约 3,000～5,000 个弱字段单变量干预样本，验证有效后再扩大到
   7,000～10,000。

不建议一开始直接采集 96,000 个同分布随机样本。当前主要瓶颈不是所有字段都缺少
样本，而是少数字段取值范围窄、极值样本稀缺、局部视觉信号弱。随机扩容和字段定向
数据解决的是不同问题，必须分别评估。

## 2. 当前基线

当前有效数据集为 `face_to_ck3_dataset_male_v2`：

| 项目 | 当前值 |
|---|---:|
| 总样本 | 12,000 |
| train / val / test | 9,628 / 1,206 / 1,166 |
| 归一化目标指纹 | 12,000 个唯一值 |
| 连续目标 | 83 个 |
| 明确学到的字段 | 68 / 83 |
| test selection score | 0.0663645 |
| scalar MAE | 0.0337407 |
| signed MAE | 0.0913015 |
| categorical strength MAE | 0.0656139 |
| 相对训练集中位数基线改善 | 55.67% |

第二阶段精调只比第一阶段继续改善约 2.01%，当前数据上的继续低学习率训练已经接近
饱和。下一轮收益应主要来自新增独立目标、补齐字段尾部和改善局部监督，而不是继续
延长现有训练。

旧的 510,000 样本不能作为扩容来源直接混回。旧数据审计曾发现严重重复 DNA
跨集合泄漏；重复截图会增加训练时间，但不会提供等量的新目标信息。

## 3. 扩展目标与非目标

### 3.1 本轮目标

- 随机样本的归一化目标唯一率不低于 99.5%；
- 所有图片、DNA 和标签保持严格一一对应；
- 保持现有正面/右侧面构图、截图尺寸、CK3 版本、相机和灯光不变；
- 将 selection score 首阶段降到 `0.0630` 以下，后续候选降到 `0.0604` 以下；
- 将相对中位数改善至少 25% 的字段从 68 个提高到至少 72 个；
- 不以牺牲现有 68 个可靠字段为代价换取弱字段改善；
- 建立可重复的随机数据、定向数据和冻结测试集三套清单。

### 3.2 当前不做

- 真人照片数据；
- 女性、儿童或不同 CK3/模组版本；
- 颜色、发型、胡须、身体、服装和表情预测；
- 在没有可辨识度证据时恢复 categorical class loss；
- 为追求数量而接收错位、重复、未回读验证或构图变化的样本。

## 4. 数据版本与目录约定

原始 12,000 样本作为 `synthetic-random-v2-12k` 基线。继续采集前冻结以下信息：

- 当前 `manifest.json`；
- `labels.jsonl` SHA-256；
- 当前 schema SHA-256；
- 12,000 个 sample ID 清单；
- 当前 test 的 1,166 个 sample ID；
- 最终基线 checkpoint 和 `test-evaluation.json`。

原始 `face/` 与 `dna/` 可以继续追加，因为采集器拒绝覆盖已有 sample ID；所有
派生产物必须写入新路径，禁止覆盖当前 12k 版本：

```text
face_to_ck3_dataset_male_v2/
  face/                              # 原始图，可追加
  dna/                               # 原始 DNA，可追加
  snapshots/
    synthetic-random-v2-12k/
  dna_schema_full_24k.json
  labels_24k.jsonl
  recommended_training_schema_24k.json
  processed_multiview_24k/
  duplicate_audit_24k/

experiments/
  targeted_field_expansion_v1/
    bases.jsonl
    variants.jsonl
    render_manifest.jsonl
    renders/
    dna/
```

版本名必须包含数据类型和规模，例如：

```text
synthetic-random-v3-24k
synthetic-targeted-v1-4k
synthetic-mixed-v1-r24k-t4k
```

## 5. 扩展路线

### 5.1 R1：随机数据扩到 24k

继续使用受校验采集器新增约 12,000 个随机外貌。每个样本仍需执行：新 DNA 检查、
连续两次稳定回读、截图后 DNA 一致性检查，以及图片/DNA 原子提交。

```powershell
python face_to_ck3_tool.py --base-dir face_to_ck3_dataset_male_v2
```

采集节奏：

1. 先采 20 张，人工逐张检查；
2. 再采 500 张，完成一次完整质量审计；
3. 以 2,000 张为一个逻辑批次继续；
4. 每批结束保存起止 sample ID、设置文件、错误次数和人工抽检结果；
5. 达到 24,000 张后停止，不自动继续到 48,000。

R1 主要改善已有 68 个可学习字段的泛化，不能指望它自动解决极端取值几乎不存在的
字段。

### 5.2 T1：弱字段定向试验集

定向数据使用“固定基础 DNA，只修改一个字段”的干预设计。建议选择 32 个基础脸，
覆盖现有数据中的头型、五官比例、肤色和局部纹理变化。每个变体必须从原始基础 DNA
重新生成，不能在上一个变体上累积修改。

第一阶段预算：

```text
32 个 bases ×（140 个字段变体 + 5 个 baseline）
= 4,640 张截图，其中 4,480 张可训练定向样本
```

每个基础脸还应穿插至少 5 次未修改 baseline，用于估计动画、曝光和截图时序噪声。

#### 字段优先级

| 优先级 | 字段 | 当前问题 | 定向动作 |
|---|---|---|---|
| P0 | `gene_bs_mouth_philtrum_width` | 仅改善 3.9%，大强度极少 | 正负 allele × 全强度，嘴部局部输入 |
| P0 | `gene_bs_mouth_lower_lip_pad` | 仅改善 8.8%，尾部稀疏 | 正负 allele × 全强度，嘴部局部输入 |
| P0 | `gene_bs_mouth_upper_lip_def` | 仅改善 0.8%，12k 中仅约 2 个高强度样本 | strength sweep，局部高分辨率 |
| P0 | `gene_bs_mouth_philtrum_def` | 仅改善 2.7%，12k 中仅约 2 个高强度样本 | strength sweep，局部高分辨率 |
| P0 | `gene_mouth_corner_depth` | 仅改善 5.1%，多数值集中在约 0.45～0.55 | 两 allele 均衡扫描极值 |
| P0 | `gene_chin_width` | 仅改善 7.8%，随机范围偏窄 | 两 allele 均衡扫描极值 |
| P1 | `gene_mouth_open` | 改善 13.5%，易受嘴部动画/时序干扰 | 固定中性状态，延长稳定等待 |
| P1 | `face_detail_nasolabial` | 改善 13.5%，随机数据无高强度尾部 | 每个 class 扫描强度，主评 strength |
| P1 | `gene_bs_mouth_philtrum_shape` | 改善 19.3%，局部信号弱且极值少 | 正负 allele × 全强度 |
| P1 | `gene_bs_mouth_upper_lip_profile` | 改善 19.7%，局部轮廓信号 | 正负 allele × 全强度 |
| P1 | `gene_bs_mouth_upper_lip_full` | 改善 21.3%，极端值稀缺 | 正负 allele × 全强度 |
| P1 | `gene_bs_mouth_lower_lip_full` | 改善 23.5%，高强度样本缺失 | 正负 allele × 全强度 |
| P1 | `gene_eye_shut` | 改善 18.3%，随机值高度集中 | 扫描极值并排除眨眼帧 |
| P1 | `gene_eye_distance` | 改善 24.8%，随机值高度集中 | 扫描极值，保留完整正脸 |
| 诊断 | `face_detail_chin_cleft` | 仅改善 0.45%，受控实验也接近噪声 | 只做小规模 0/128/255 门禁 |

`face_detail_chin_cleft` 不进入正式扩容预算。若 32 个基础脸的极值试验仍不能稳定超过
baseline 噪声，继续保留模板值，不再追加数据。

#### 强度序列

第一阶段先使用 5 点 pilot sweep：

```text
0, 64, 128, 192, 255
```

它对应上面的 4,640 张预算。字段在 targeted validation 上确认有效后，第二阶段才为
受益字段加密到 `0,32,64,96,128,160,192,224,255`，避免在未证明可学习的字段上
先消耗约两倍采集时间。

- signed/scalar-alias 字段：负、正 allele 都扫描；
- 单类别 strength 字段：当前 allele 扫描全部强度；
- 多类别字段：每个 class 至少扫描 `0/128/192/255`，主任务仍为 strength；
- strength=0 时 sign/class 不可观察，分类损失必须屏蔽；
- `gene_mouth_open`、眼睛和嘴部字段需增加 baseline 重复，以分离动画噪声。

可复用 `dna_field_sweep_tool.py` 的桌面校准和 `run_identifiability_experiment.py`
的完整 DNA 回读机制。正式批量必须启用“复制 DNA 验证”，不能只依赖截图哈希。
实际生成和执行流程见 [`TARGETED_SAMPLE_COLLECTION.md`](../TARGETED_SAMPLE_COLLECTION.md)
及 `targeted_sample_collector.py`。

### 5.3 R2：有条件扩到 48k

只有 R1 训练达到以下任一条件时才进入 R2：

- selection score 相对 12k 基线改善至少 4%；
- 至少新增 2 个字段跨过 25% 基线改善门槛；
- 现有 68 个可靠字段的中位 MAE 继续稳定下降，且没有明显分组退化。

若 24k 相对 12k 改善不足 3%，应停止同分布随机扩容，优先处理定向数据、局部输入、
损失掩码或模型结构，不直接堆到 48k/96k。

## 6. 定向数据不能直接拼接

一个基础脸会生成数十到数百个单字段变体。如果把这些样本当普通随机样本训练，未修改
的 82 个目标会被同一基础脸重复监督，导致标签先验和身份背景严重偏置。

每个定向样本必须额外保存：

```json
{
  "source_type": "targeted_intervention",
  "base_dna_id": "base_001",
  "base_split": "train",
  "intervention_field": "gene_bs_mouth_philtrum_width",
  "target_family": "signed",
  "allele": "...",
  "strength": 192,
  "loss_mask": [{"family": "signed", "field": "gene_bs_mouth_philtrum_width"}],
  "dna_sha256": "...",
  "render_sha256": "..."
}
```

计划哈希保存在 `protocol.json` 的 `plan_sha256`，采集完成记录使用
`protocol_plan_sha256` 关联该计划。

训练时建议：

- 随机样本与定向样本的 batch 比例先设为 `4:1`；
- 定向样本的干预字段损失权重为 `1.0`；
- 非干预字段损失屏蔽或降到 `0～0.1`；
- 同一 base 的所有变体只能出现在同一个 split；
- 每轮报告随机验证集和定向验证集，不能只报告混合均值；
- T1 前需要给训练标签增加 `source_type`、`base_dna_id`、字段 loss mask，并让训练器
  支持按来源混合采样。这是把 sweep 数据用于训练的前置开发项。

## 7. 切分与防泄漏

### 7.1 冻结旧测试集

当前 1,166 个 test 样本保持不变，作为跨数据版本的主要可比基准。它们不得参与新模型
训练、早停、学习率选择、字段权重统计或定向 base 选择。

新增随机样本按固定 seed `20260718` 划分。旧 sample ID 的 split 必须保持稳定。最终
报告至少分开列出：

1. `legacy_random_test`：冻结的 1,166 个旧样本；
2. `new_random_test`：新增随机数据的独立测试子集；
3. `targeted_test`：按 base DNA 留出的定向测试集。

### 7.2 重复目标

随机数据在预处理前运行归一化目标指纹审计。目标唯一率低于 99.5% 时暂停训练并调查。
所有重复目标必须进入同一 split；不得因为图片哈希不同就视为独立标签。

现有 `tools/build_dna_grouped_split.py` 可以审计和隔离完全相同的归一化目标，但它不能
单独解决定向数据的 base 身份泄漏：同一 base 的不同字段变体拥有不同目标指纹。因此
定向数据还必须按 `base_dna_id` 分组切分。

### 7.3 基础脸分配

32 个试验 bases 建议固定为：

```text
train: 24 bases
val:    4 bases
test:   4 bases
```

扩大到 64 个 bases 时使用 48/8/8。不要把同一 base 的不同字段分到不同集合。

### 7.4 定向训练前的工具缺口

`targeted_sample_collector.py` 已能生成单字段干预 DNA、锁定 base、按
`base_dna_id` 分组切分、保存来源及 loss-mask 元数据，并复用完整 DNA 回读采集。
下列训练侧能力仍需在 T1 进入正式训练前补齐并测试：

- 把 sweep 的 DNA、截图和 manifest 转换为标准双视图训练分片；
- 在 DataLoader 中按来源控制随机/定向样本比例；
- 在损失函数中实际应用 manifest 已生成的字段级 loss mask；
- 让评估器读取冻结 ID 清单，分别输出 legacy、新随机和 targeted test 指标。

在这些能力完成前，定向数据只用于可辨识度分析，不能直接复制进随机训练目录。

## 8. 随机扩容处理流程

以下命令以扩展到 24k 为例。所有输出使用新文件名，确认无误前不覆盖 12k 产物。

### 8.1 重建完整 source schema

```powershell
python dna_normalizer.py schema face_to_ck3_dataset_male_v2/dna `
  --output face_to_ck3_dataset_male_v2/dna_schema_full_24k.json
```

比较新旧 schema：字段数、allele 集合和字段类型必须一致。若出现新 class 或类型变化，
停止流水线并重新做该字段可辨识度审核，不能静默改变输出 head。

### 8.2 重新生成全部标签

```powershell
python dna_normalizer.py normalize face_to_ck3_dataset_male_v2/dna `
  --schema face_to_ck3_dataset_male_v2/dna_schema_full_24k.json `
  --output face_to_ck3_dataset_male_v2/labels_24k.jsonl `
  --progress-every 2000
```

正式标签禁止使用 `--skip-invalid` 或 `--allow-pair-mismatch`。

### 8.3 绑定 identifiability schema

```powershell
python tools/adapt_identifiability_schema.py `
  --source face_to_ck3_dataset_male_v2/dna_schema_full_24k.json `
  --template experiments/dna_identifiability/recommended_training_schema.json `
  --output face_to_ck3_dataset_male_v2/recommended_training_schema_24k.json
```

### 8.4 构建重复目标审计

```powershell
python tools/build_dna_grouped_split.py `
  --labels face_to_ck3_dataset_male_v2/labels_24k.jsonl `
  --schema face_to_ck3_dataset_male_v2/recommended_training_schema_24k.json `
  --output face_to_ck3_dataset_male_v2/duplicate_audit_24k `
  --ratios 0.8,0.1,0.1 `
  --seed 20260718
```

检查 `duplicated_sample_count`、`maximum_group_size` 和
`cross_split_duplicate_groups`。随机扩容目标是前两项分别接近 `0/1`，后一项必须为 0。

### 8.5 预处理正面/侧面

当前 v2 截图规格为 `1245×829`，必须显式沿用当前 manifest 的裁剪框：

```powershell
python image_preprocessor.py `
  face_to_ck3_dataset_male_v2/face `
  face_to_ck3_dataset_male_v2/processed_multiview_24k `
  --labels face_to_ck3_dataset_male_v2/labels_24k.jsonl `
  --expected-size 1245,829 `
  --crop 80,19,620,829 `
  --side-crop 650,19,1190,829 `
  --size 256,384 `
  --splits 0.8,0.1,0.1 `
  --race-group-size 0 `
  --split-seed 20260718 `
  --jpeg-quality 95 `
  --jpeg-subsampling 0 `
  --shard-size 500 `
  --workers 4
```

不得加入 `--allow-size-mismatch` 或 `--skip-invalid`。正式 manifest 必须满足
`processed = matched_labels = 目标样本数` 且 `skipped = unmatched_labels = 0`。

### 8.6 重建训练标签统计

```powershell
python tools/build_training_label_stats.py `
  --data-root face_to_ck3_dataset_male_v2/processed_multiview_24k `
  --schema face_to_ck3_dataset_male_v2/recommended_training_schema_24k.json `
  --split train `
  --output face_to_ck3_dataset_male_v2/processed_multiview_24k/train_label_stats.json `
  --workers 4
```

字段均值、类别频数和强度分位数需与 12k 版本对比。任何字段均值变化超过 0.05、某个
class 占比变化超过 10 个百分点或强度范围缩窄，都要先解释采集分布变化。

## 9. 质量门禁

### 9.1 每个随机批次

- 图片和 DNA 编号集合完全相同；
- 无 `.partial` 残留；
- 图片尺寸全部为 `1245×829`；
- 相邻 DNA 指纹不同；
- 全量归一化目标唯一率不低于 99.5%；
- 随机人工回放至少 `max(20, 批次样本数×1%)` 个样本；
- 回放后正面、侧面轮廓与原截图一致；
- 曝光、相机、衣服、无头发遮挡和人物位置没有漂移；
- 记录失败/重试率，单批事务失败率超过 1% 时暂停采集。

### 9.2 每个定向批次

- 全部样本通过 CK3 DNA 完整回读；
- 除 `intervention_field` 外，所有训练字段和颜色均与 base 相同；
- 每个计划、DNA 和截图都有 SHA-256；
- 变体强度齐全，不允许跳过失败档位后继续；
- baseline 重复用于计算噪声，曝光漂移单独记录；
- 目标字段视觉变化应随强度基本单调；
- 相同 base、字段、allele 和强度不得重复计为新样本。

## 10. 训练实验矩阵

为了得到真实学习曲线，所有规模实验必须使用相同模型、增强、损失、初始化策略和冻结
测试集。不能用“12k 从 ImageNet 训练”和“24k 从现有 best.pt 精调”直接比较数据收益。

建议矩阵：

| 实验 | 随机数据 | 定向数据 | 用途 |
|---|---:|---:|---|
| E0 | 12k | 0 | 复现当前基线 |
| E1 | 18k | 0 | 学习曲线中间点 |
| E2 | 24k | 0 | 随机扩容收益 |
| E3 | 24k | 4.48k（另 160 baseline） | 定向数据收益 |
| E4 | 24k | 4.48k，非目标不掩码 | 验证损失掩码必要性 |
| E5 | 48k | 4.48k | 仅在 R2 门禁通过后运行 |

E0/E1/E2 至少使用相同 seed 完成主比较；最终 E2/E3 候选追加一个独立 seed 复验。
测试集只在配置和 checkpoint 选择冻结后运行一次。

每个实验必须报告：

- selection score；
- scalar/signed/strength MAE；
- 83 个字段相对训练中位数改善；
- 可靠、弱信号、接近基线字段数量；
- legacy/new-random/targeted 三套 test 指标；
- 每个定向字段的分强度区间 MAE；
- 现有 68 个可靠字段的回归情况；
- 至少 100 个固定样本的 CK3 回渲染或人工配对检查。

## 11. 预期收益与停止条件

仅根据当前单个干净数据规模点，随机扩容收益只能作为工程区间：

| 总随机样本 | 预计 selection score | 相对当前提升 |
|---:|---:|---:|
| 24k | 0.061～0.063 | 5%～8% |
| 48k | 0.057～0.060 | 9%～14% |
| 96k | 0.054～0.058 | 12%～19% |

这些区间不是已经验证的 scaling law。应使用 E0/E1/E2 拟合项目自己的学习曲线，再决定
是否继续扩容。

15 个弱字段目前约贡献总 selection score 的 28.7%。如果定向数据能把可辨识的弱字段
统一提升到相对基线改善 35%，纯算术上整体 score 可再改善约 7.2%；实际收益会受到
局部输入分辨率、字段多解性和渲染噪声限制。

停止条件：

- 24k 相对 12k 改善不足 3%；
- 连续两个规模点的 score 改善都不足 2%；
- 定向字段在 targeted validation 上改善不足 5%，或只记住训练 bases；
- 现有可靠字段中超过 5 个退化 5% 以上；
- 新数据唯一率、同步准确率或构图一致性未通过门禁；
- `face_detail_chin_cleft` 的小规模极值试验仍不超过 baseline 噪声。

满足停止条件后，应转向局部 crop/head、字段条件化、损失设计或输出多候选，不继续用
同一采集分布堆数量。

## 12. 首轮执行清单

按以下顺序推进：

1. 冻结 12k 基线的标签、schema、test IDs、manifest 和 checkpoint 哈希；
2. 验证新持久化采集设置，先采 20 张；
3. 完成 500 张试采和人工回放门禁；
4. 将随机数据扩到 24k；
5. 重建 schema、标签、重复审计、图像分片和 train stats；
6. 运行 E0/E1/E2，得到真实学习曲线；
7. 只从旧训练集身份中选择 32 个 bases，生成 4,640 张定向试验截图；
8. 完成 base 分组切分、来源采样和字段 loss mask 支持；
9. 运行 E3/E4，判断哪些弱字段真正受益；
10. 只有 R2 门禁通过时，继续随机扩到 48k；
11. 冻结最终候选后运行三套 test，并更新最终评估文档。

本轮默认推荐终点为“24k 随机 + 4.48k 可训练定向样本”，而不是预先承诺扩到 96k。是否继续由
学习曲线、字段改善数量和回渲染质量共同决定。
