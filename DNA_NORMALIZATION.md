# CK3 DNA 训练标签归一化

本文档描述 `face_to_ck3_dataset_male_small/dna` 中 CK3 男性 DNA 的解析、归一化和反归一化规则。配套脚本为 [`dna_normalizer.py`](dna_normalizer.py)。

## 1. 数据结构

跨数据集范围等距抽查的 10 份 DNA 均包含 102 个如下形式的字段：

```text
gene_chin_forward={ "chin_forward_neg" 97 "chin_forward_neg" 97 }
```

它由两套等位基因组成，每套均为：

```text
allele 名称 + 0～255 强度
```

脸型与面部细节字段的两套等位基因在抽样中相同。出现差异的是 `gene_bs_body_shape`、`gene_bs_bust` 和 `clothes`，它们不属于默认训练目标。

颜色字段不是 RGB，而是 CK3 调色板中的二维坐标：

```text
skin_color={ 212 109 212 109 }
```

前两个数字是第一套坐标，后两个数字是第二套坐标。采集数据中二者相同。

`hair_color` 紧跟在 `genes={` 后面，可能与 `genes={` 位于同一行。解析器必须搜索完整文本，不能假设每行只有一个字段。

## 2. 默认训练目标

脚本默认使用 83 个字段：

- 73 个 `gene_*` / `gene_bs_*` 脸部几何字段；
- 10 个 `face_detail_*` 面部细节字段；
- `hair_color`、`skin_color`、`eye_color` 三组二维颜色坐标。

以下字段不进入默认标签：

- `ruler_designer_*`、`type`、`id`、`random_seed`、`entity`；
- 身高、身体、胸部和衣服；
- 表情、年龄、肤质；
- 眉毛、头发、胡须和饰品；
- `portrait_modifier_overrides`。

生成完整 DNA 时，这些字段由输入模板保留。

## 3. Schema 必须由完整数据集生成

不同字段拥有不同的 allele 词表。不能给 allele 字符串设置一个全局编号，也不能只看少量样本建立最终词表。

运行：

```powershell
python dna_normalizer.py schema face_to_ck3_dataset_male_small/dna `
  --output face_to_ck3_dataset_male_small/dna_schema.json
```

Schema 会保存：

- 字段的固定顺序；
- 每个字段观察到的 allele 及频数；
- 观察到的强度最小值和最大值；
- 缺字段数量；
- 两套等位基因不一致的数量；
- 字段采用 `signed` 还是 `categorical + strength` 表示。

只有当一个字段的完整词表恰好是同词干的 `_neg` / `_pos` 两项时，脚本才将其判定为 `signed`。例如：

```text
chin_forward_neg
chin_forward_pos
```

若存在第三种形态、词干不同，或者只有单一形态，则使用类别与强度两个目标。这样不会错误压缩 `nose_profile_hawk`、`eye_socket_02` 等多形态基因。

## 4. 数值归一化

### 4.1 普通强度

CK3 强度的语义范围固定为 `[0, 255]`：

```text
strength = raw / 255
```

反归一化：

```text
raw = round(clamp(strength, 0, 1) × 255)
```

不使用样本的观测 min/max，也不默认进行 `(x - mean) / std`。固定除以 255 可逆、不会受数据集分布改变影响，也便于对所有字段使用同一损失尺度。

### 4.2 纯正负字段

纯正负字段编码到 `[-1, 1]`：

```text
negative allele: signed = -raw / 255
positive allele: signed = +raw / 255
```

例如：

```text
chin_forward_neg 97 -> -0.380392
chin_forward_pos 131 -> +0.513725
```

反归一化时由符号选择 allele，由绝对值恢复强度。强度恰好为 0 时 allele 没有视觉效果，脚本使用 schema 中频率最高的 allele。因此在 `raw=0` 时文本 token 不保证逐字还原，但渲染语义等价。

### 4.3 多类别字段

多类别字段保存两个目标：

```text
categorical_class: 字段内部的类别 ID
categorical_strength: raw / 255
```

类别 ID 只用于交叉熵，不具有连续数值或大小关系。每个字段有自己的类别词表。

当强度接近 0 时 allele 几乎不可从图像观察。训练分类头时建议按强度降低分类损失权重：

```python
weighted_ce = cross_entropy(logits, class_id) * max(strength, 0.05)
```

### 4.4 颜色

每个颜色字段保存两个坐标：

```text
[coordinate_1 / 255, coordinate_2 / 255]
```

例如：

```text
skin_color={ 212 109 212 109 }
-> [0.831373, 0.427451]
```

不要将这两个坐标解释或转换为 RGB。

## 5. JSONL 标签格式

Schema 已保存字段顺序，因此 51 万条标签不重复保存字段名。每行格式如下：

```json
{"sample_id":"face_0001","signed":[-0.380392],"categorical_class":[1],"categorical_strength":[0.341176],"colors":[0.262745,0.949020,0.831373,0.427451,0.164706,0.960784]}
```

数组含义：

- `signed`：依照 `schema.signed_fields` 排列；
- `categorical_class`：依照 `schema.categorical_fields` 排列；
- `categorical_strength`：与 `categorical_class` 一一对应；
- `colors`：依照 `schema.color_fields` 排列，每字段连续两个数。

生成全部标签：

```powershell
python dna_normalizer.py normalize face_to_ck3_dataset_male_small/dna `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --output face_to_ck3_dataset_male_small/labels.jsonl
```

默认遇到以下问题会立即停止：

- 缺少目标字段；
- allele 不在 schema 中；
- 数值不在 0～255；
- 两套脸型等位基因不一致；
- 两套颜色坐标不一致。

清洗阶段可以加 `--skip-invalid` 跳过并在标准错误中记录坏文件。不建议在正式标签生成时使用 `--allow-pair-mismatch`。

## 6. 反归一化为 CK3 DNA

推理程序输出与 JSONL 单行相同的数组结构后，可以把结果应用到一份男性 DNA 模板：

```powershell
python dna_normalizer.py denormalize `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --label prediction.json `
  --template face_to_ck3_dataset_male_small/dna/face_0001.txt `
  --output predicted_dna.txt
```

如果 `--label` 是包含多行的 JSONL，使用 `--sample-id` 选择记录：

```powershell
python dna_normalizer.py denormalize `
  --schema face_to_ck3_dataset_male_small/dna_schema.json `
  --label face_to_ck3_dataset_male_small/labels.jsonl `
  --sample-id face_0001 `
  --template face_to_ck3_dataset_male_small/dna/face_0001.txt `
  --output predicted_dna.txt
```

标签包含 `colors` 时脚本会替换颜色字段；几何模型的预测不包含 `colors`，此时模板原有的发色、肤色和眼色保持不变。模板中的性别、身体、衣服、发型、胡须和其他 CK3 内容同样保持不变。

## 7. 训练损失建议

推荐对字段先分别求平均，再合并，避免类别较多的字段天然获得更大权重：

```text
loss = mean(SmoothL1(signed_prediction, signed_target))
     + mean(strength_weighted_label_smoothed_CE(category_logits, category_target))
     + mean(SmoothL1(strength_prediction, strength_target))
     + 0.1 * dual_view_consistency
```

模型输出约束：

- `signed` 使用 `tanh`；
- `categorical_strength` 使用 `sigmoid`；
- `categorical_class` 使用每字段独立的 logits/softmax。

颜色不属于模型输出，由独立程序提取或保留 DNA 模板值。

验证指标至少包括：

- signed 参数 MAE；
- 多类别 allele accuracy；
- 强度 MAE（换算回 0～255）；
- 将预测 DNA 放回 CK3 后的渲染相似度。
