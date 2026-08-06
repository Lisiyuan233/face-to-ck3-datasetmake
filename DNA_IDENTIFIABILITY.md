# CK3 DNA 字段可辨识度第一阶段自动化

第一阶段现在由两个脚本完成：

- [`build_identifiability_variants.py`](build_identifiability_variants.py)：自动选择 17 个基础 DNA，并按 schema 生成全部受控变体与穿插基线；
- [`run_identifiability_experiment.py`](run_identifiability_experiment.py)：逐项粘贴到 CK3、完整回读 DNA、截图、失败重试并断点续跑。

## 1. 计划定义

生成器直接读取训练使用的 `dna_schema.json`：

- signed 字段：负 allele 和正 allele 各自生成 `0/128/255`；
- categorical 字段：schema 中每个 class 各自生成 `0/128/255`；
- baseline：每个基础 DNA 原样重复 5 次，均匀放在该基础脸计划的开始、约 1/4、1/2、3/4 和结尾；
- 每个变体重新从对应基础 DNA 生成，并验证除目标字段外的 gene 和颜色均未改变。

当前 schema 为 67 个 signed 字段、16 个 categorical 字段、39 个 class，因此：

```text
signed:      67 × 2 × 3 = 402
categorical: 39 × 3     = 117
baseline:                  5
每个基础脸:              524
17 个基础脸: 17 × 524 = 8,908
```

## 2. 自动选择 17 个 bases

默认每 30,000 个样本为一个 `race_group`。安装 NumPy 时，选择器对 `labels.jsonl` 做一次流式读取和批量量化：

1. 对 signed、categorical strength 和颜色计算每组中位数，对 categorical class 计算每组众数；
2. 使用归一化 L1 距离和 class Hamming 距离，选择最接近该组混合中位数、且真实 DNA 文件存在的样本。

未安装 NumPy 时会自动退回只使用标准库的两遍流式算法，选择规则与结果保持一致，但处理 510,000 条标签会明显更慢。

平局时选择 sample ID 较小者。结果、距离、组样本数和 DNA SHA-256 都写入 `bases.jsonl`，选定 DNA 同时复制到实验目录，后续重跑不再依赖源数据路径。

直接选择并生成完整计划：

```powershell
python build_identifiability_variants.py prepare `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --labels face_to_ck3_dataset_male_small/labels.jsonl `
  --dna-dir face_to_ck3_dataset_male_small/dna `
  --output experiments/dna_identifiability
```

快速路径需要扫描约 510,000 条标签一遍，脚本每 50,000 条显示一次进度。

也可以先单独生成并人工审核 bases 清单：

```powershell
python build_identifiability_variants.py select-bases `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --labels face_to_ck3_dataset_male_small/labels.jsonl `
  --dna-dir face_to_ck3_dataset_male_small/dna `
  --output experiments/selected_bases.jsonl

python build_identifiability_variants.py prepare `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --bases-manifest experiments/selected_bases.jsonl `
  --output experiments/dna_identifiability
```

显式 bases manifest 必须恰好包含 `race_group=0..16`，每行至少包含：

```json
{"race_group":0,"sample_id":"face_0123","dna_path":"path/to/face_0123.txt"}
```

## 3. CK3 自动化准备

先启动旧的单字段工具完成一次桌面校准：

```powershell
python dna_field_sweep_tool.py
```

必须记录并保存：

1. “粘贴 DNA”按钮；
2. 粘贴后的“确定”按钮；
3. “复制 DNA 验证”按钮；
4. 同时包含正面和侧面的截图区域；
5. 刷新等待、截图等待和失败重试次数。

正式 runner 强制要求复制 DNA 验证按钮。每次应用后会比较游戏回读的全部 schema
脸部字段和颜色，而不是只比较当前目标字段；baseline 因而也能验证角色确实恢复到了
实验范围内的基础 DNA。身体、衣服、表情和配饰字段不属于训练 schema，且 CK3
可能在导入时自动规范化这些字段，因此不作为回读失败条件。

## 4. 运行、门禁与恢复

先做不控制游戏的完整计划/哈希检查：

```powershell
python run_identifiability_experiment.py experiments/dna_identifiability --dry-run
```

## 5. 分析与字段决策表

全部截图完成后运行：

```powershell
python analyze_identifiability_experiment.py experiments/dna_identifiability
```

分析器会先对每张图独立做亮度标准化，再计算正面、侧面和字段局部区域的梯度特征；
只有在重复 baseline 逐像素完全一致的 bases 上，正负 allele 的 `0/128/255` 三档
截图也全部一致时，才会自动建议合并 allele。输出包括：

- `field_identifiability.csv`：83 个字段的决策表；
- `field_class_identifiability.csv`：每个 sign/class 的强度指标；
- `allele_class_alias_matrix.csv`：class 变化向量相似度；
- `render_quality.csv`：baseline 噪声和曝光漂移；
- `field_groups.json`：训练分组和 A～E tier；
- `recommended_loss_weights.json`；
- `recommended_visibility_thresholds.json`；
- `recommended_training_schema.json`：schema v2 提案，不会覆盖当前训练 schema；
- `analysis_summary.json`。

`probe_accuracy` 在当前阶段留空；在冻结视觉特征 Probe 完成前，tier 会标记为
`provisional_no_probe_exposure_normalized`。

建议先对第一个基础脸运行 10 项门禁：

```powershell
python run_identifiability_experiment.py experiments/dna_identifiability `
  --base-id race_00_face_0123 `
  --limit 10
```

人工核对这 10 张图、DNA、manifest 和正侧面构图后，运行全量：

```powershell
python run_identifiability_experiment.py experiments/dna_identifiability
```

中断后执行同一命令即可恢复。只有 render 文件存在且 SHA-256 与 `render_manifest.jsonl` 一致的 completed 项才会跳过；失败会写入 `errors.jsonl` 并停止，不会越过失败项继续制造错配数据。

PyAutoGUI 安全停止仍然有效：把鼠标快速移动到主屏幕左上角可紧急终止。

## 5. 输出结构

```text
experiments/dna_identifiability/
  protocol.json
  bases.jsonl
  variants.jsonl
  bases/
  dna/
    race_00_face_0123/
  renders/
    race_00_face_0123/
  render_manifest.jsonl
  errors.jsonl
```

`protocol.json` 保存 schema 哈希、计划哈希和精确计数。若同一输出目录已经存在不同计划，生成器会拒绝覆盖，避免旧截图与新 DNA 混写。

`variants.jsonl` 每行包含：基础脸、race group、字段类型、sign/class、allele、强度、零强度参考、5 个 baseline 参考、DNA/截图路径及 DNA SHA-256。后续 SNR、单调性、Probe 和 alias 分析应以该文件与 `render_manifest.jsonl` 为事实源。
