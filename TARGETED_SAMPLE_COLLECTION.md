# CK3 定向样本采集器

[`targeted_sample_collector.py`](targeted_sample_collector.py) 用于从已有合成 DNA
中挑选基础脸，并围绕指定字段生成、采集“每次只改变一个字段”的训练样本。它复用
[`dna_field_sweep_tool.py`](dna_field_sweep_tool.py) 保存的桌面坐标、等待参数和完整
DNA 回读验证，不采集也不依赖真人数据。

采集流程分为四步：选择 bases、锁定计划、执行或恢复、检查状态。计划中的全部变体
都从锁定的原始 base DNA 重新生成，不会在前一个变体上叠加修改。

## 1. 前置条件

1. 现有数据集包含 `labels.jsonl` 和与 `sample_id` 同名的 `dna/*.txt`；
2. source schema 为 `schema_version=1`，训练 schema 为 `schema_version=2`；
3. 已在 `dna_field_sweep_tool.py` 中记录粘贴、确认、复制 DNA 验证按钮和截图区域；
4. 正式运行前，CK3 已固定相机、灯光、姿态和界面位置。

桌面参数默认从 `%APPDATA%/CK3DNAFieldSweep/settings.json` 读取。正式采集强制要求
已经设置“复制 DNA 验证按钮”；只配置截图坐标仍会被拒绝。

## 2. 选择 32 个基础 DNA

最安全的做法是只允许旧训练集 ID 参与 base 选择，避免把冻结的验证或测试身份带入
定向训练集：

```powershell
python targeted_sample_collector.py select-bases `
  --labels face_to_ck3_dataset_male_v2/labels.jsonl `
  --dna-dir face_to_ck3_dataset_male_v2/dna `
  --include-ids path/to/legacy_train_ids.txt `
  --count 32 `
  --output experiments/targeted_field_expansion_v1/selected_bases.jsonl
```

`--include-ids` 文件每行一个 `sample_id`。如果暂时只有冻结 test ID 清单，可以改为：

```powershell
python targeted_sample_collector.py select-bases `
  --labels face_to_ck3_dataset_male_v2/labels.jsonl `
  --dna-dir face_to_ck3_dataset_male_v2/dna `
  --exclude-ids path/to/frozen_test_ids.txt `
  --count 32 `
  --output experiments/targeted_field_expansion_v1/selected_bases.jsonl
```

`--include-ids` 和 `--exclude-ids` 都可重复指定；exclude 的优先级更高。筛选器会先取
接近各标签维度总体中位数/众数的锚点，再使用确定性的 farthest-point maximin
补齐形态分散的 bases。相同输入会得到相同顺序。只有同时存在标签和 DNA 的样本会
进入候选集。

也可以手写 base manifest，每行格式如下；`split` 必须全部填写或全部省略：

```json
{"base_dna_id":"base_001","sample_id":"face_0001","dna_path":"../../face_to_ck3_dataset_male_v2/dna/face_0001.txt","split":"train"}
```

省略 split 时，`prepare` 会按 `base_dna_id` 确定性分成 train/val/test。默认 32 个
bases 精确分成 24/4/4；同一 base 的全部变体永远属于同一个 split。

## 3. 生成并锁定采集计划

```powershell
python targeted_sample_collector.py prepare `
  --source-schema face_to_ck3_dataset_male_v2/dna_schema_full.json `
  --training-schema face_to_ck3_dataset_male_v2/recommended_training_schema.json `
  --bases-manifest experiments/targeted_field_expansion_v1/selected_bases.jsonl `
  --output experiments/targeted_field_expansion_v1
```

默认采集扩展方案中的 14 个弱字段，强度为 `0,64,128,192,255`，并在每个 base 的
序列中均匀穿插 5 个未修改 baseline。按当前 schema，单个 base 生成 140 个字段变体
和 5 个 baseline；32 个 bases 共 4,640 张，其中 4,480 张可用于定向训练，160 张
baseline 只用于漂移和时序噪声检查。

可用 `--field` 重复指定更小的字段集合，也可通过 `--strengths 0,128,255` 先做门禁：

```powershell
python targeted_sample_collector.py prepare `
  --source-schema face_to_ck3_dataset_male_v2/dna_schema_full.json `
  --training-schema face_to_ck3_dataset_male_v2/recommended_training_schema.json `
  --bases-manifest experiments/targeted_gate_v1/selected_bases.jsonl `
  --output experiments/targeted_gate_v1 `
  --field gene_bs_mouth_philtrum_width `
  --field gene_bs_mouth_lower_lip_pad `
  --strengths 0,128,255
```

`face_detail_chin_cleft` 默认不采。仅需诊断时增加
`--include-diagnostic-chin-cleft`，不要把它自动并入正式训练预算。

`prepare` 会复制并锁定 bases、生成所有变体 DNA、记录 schema 和 DNA 哈希，并生成
`plan_sha256`。同一计划可重复执行；如果输出目录已有不同计划，会拒绝混写。

## 4. 门禁、正式采集与恢复

先只校验计划和筛选范围，不控制 CK3：

```powershell
python targeted_sample_collector.py run `
  experiments/targeted_field_expansion_v1 `
  --dry-run
```

正式批量前，先运行一个 base 的前 10 个变体并人工核对截图、DNA 和 manifest：

```powershell
python targeted_sample_collector.py run `
  experiments/targeted_field_expansion_v1 `
  --base-id base_001_face_0001 `
  --limit 10
```

门禁通过后运行全部计划：

```powershell
python targeted_sample_collector.py run experiments/targeted_field_expansion_v1
```

也可以按 split 或字段分批：

```powershell
python targeted_sample_collector.py run `
  experiments/targeted_field_expansion_v1 `
  --split train `
  --field gene_bs_mouth_philtrum_width
```

重新执行相同命令即可断点恢复。只有截图存在且 SHA-256 与 completed manifest 一致的
变体才会跳过；失败记录写入 `errors.jsonl` 并立即停止，不会越过错误继续制造错位
标签。运行中仍可用 PyAutoGUI 的左上角 fail-safe 紧急停止。

## 5. 查看完成度

```powershell
python targeted_sample_collector.py status experiments/targeted_field_expansion_v1
```

状态按整体、split 和字段分别报告 total/completed，便于定位漏采批次。

## 6. 输出结构与训练约束

```text
experiments/targeted_field_expansion_v1/
  protocol.json
  selected_bases.jsonl
  bases.jsonl
  variants.jsonl
  render_manifest.jsonl
  errors.jsonl                 # 仅失败时出现
  bases/
  dna/<base_dna_id>/
  renders/<base_dna_id>/
```

`protocol.json`、`variants.jsonl` 和完成后的 `render_manifest.jsonl` 共同保留：

- `source_type`、`base_dna_id`、`base_split` 和 `source_sample_id`；
- `intervention_field`、`target_family`、`class_id`、allele 和 strength；
- `training_eligible` 与字段级 `loss_mask`；
- 计划、DNA、截图哈希及 baseline/零强度参考关系。

这些记录是训练转换器的输入，不表示截图可以直接复制到随机数据集目录。训练时仍应：

- 初始使用约 `4:1` 的随机/定向 batch 比例；
- 定向样本只给干预字段完整权重，非干预字段屏蔽或降到 `0～0.1`；
- baseline 不进入训练；
- 按 `base_dna_id` 分组切分，禁止身份跨 split；
- 分别报告随机验证集和 targeted validation，不只看混合均值。

将采集结果转换成标准双视图训练分片、按来源混合采样和在训练损失中真正应用 mask，
仍需在定向样本进入正式训练前实现并测试。
