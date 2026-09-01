# CK3 正面/侧面图片预处理

本文档说明如何把 `face_to_ck3_dataset_male_small/face` 中的正面+侧面合成 PNG 转换为严格配对的正面/纯侧脸训练数据。配套脚本为 [`image_preprocessor.py`](image_preprocessor.py)。

## 1. 抽样观察结论

跨索引区间抽查的图片具有稳定布局：

- 原图均为 `1326 × 891` RGB PNG；
- 左侧为正脸，右侧为右侧面；
- 镜头、背景、人物位置和衣服基本固定；
- 人物没有头发遮住头型，但存在眉毛、胡茬和肤质差异；
- 部分人物嘴巴微张或眼睛半闭；
- 左上背景 RGB 均值约为 `(34.2, 36.5, 40.5)`，抽样变化不足 0.1；
- 正脸区域亮度差异明显，抽样灰度均值约为 `44.5～76.0`。

训练模型同时使用左侧正脸和右侧 90° 侧脸。正面负责纹理、宽度和类别细节，侧脸主要补充鼻、嘴、眼、下颌等深度与轮廓信息。

## 2. 固定裁剪

默认正面裁剪框：

```text
left=150, top=20, right=690, bottom=830
```

即：

```python
front = image.crop((150, 20, 690, 830))
```

裁剪结果为 `540 × 810`，恰好接近 `2:3`。缩放到 `256 × 384` 时基本不改变宽高比。

默认侧脸裁剪框：

```text
left=710, top=20, right=1250, bottom=830
```

```python
side = image.crop((710, 20, 1250, 830))
```

它同样是 `540 × 810`，覆盖后脑、耳朵、鼻尖、嘴部、下颌和颈部，并保持人物朝右；侧脸训练不做水平翻转。

这个区域在抽样中能够：

- 保留头顶、双耳、下巴和部分颈部；
- 排除右侧面；
- 去掉左侧大量固定背景；
- 去掉大部分与脸型无关的衣服区域。

脚本默认严格检查原图必须是 `1326 × 891`。任何不同尺寸都会停止处理，防止截图区域变化后仍悄悄产出错位标签。

## 3. 为什么输出 JPEG tar 分片

原始图片约 488 GiB，并且有 51 万个独立 PNG。直接随机读取会受到 PNG 解码、小文件访问和文件系统元数据开销限制。

默认输出：

- `256 × 384` RGB JPEG；
- JPEG quality 95；
- chroma subsampling 0，保留颜色细节；
- 每 2,000 个样本一个不压缩 tar；
- 图片与标签使用相同 basename。

一个分片内部类似：

```text
face_0001.front.jpg
face_0001.side.jpg
face_0001.json
face_0002.front.jpg
face_0002.side.jpg
face_0002.json
...
```

JPEG 已压缩，tar 不再做 gzip 压缩，这样训练时可以顺序读取并快速定位成员。

脚本先写 `*.tar.partial`，只有分片正常结束后才原子改名为 `*.tar`。中断产生的 partial 文件不会被误当成完整训练数据。

## 4. 种族块与数据划分

数据按种族顺序采集，每个种族连续 30,000 张。总计：

```text
510000 / 30000 = 17 个完整种族块
```

样本所属块：

```text
race_group = (图片编号 - 1) // 30000
```

因此 `face_0001～face_30000` 属于 `race_group=0`，`face_30001～face_60000` 属于 `race_group=1`，依此类推到 `race_group=16`。目前没有种族名称映射，所以使用稳定的数字 ID；后续可以另外维护 `race_group -> 种族名称` 表，不需要改训练标签。

默认划分为：

```text
train = 90%
val   = 5%
test  = 5%
```

脚本不会直接取每个种族块的末尾作为验证集，因为连续采集过程可能存在时间漂移。它会在每个 30,000 张的块内使用无需占用额外内存的确定性置换，再进行 90/5/5 切分。

因此完整数据中每个种族都精确得到：

```text
train = 27,000
val   =  1,500
test  =  1,500
```

总计应为：

```text
train = 459,000
val   =  25,500
test  =  25,500
```

划分不依赖文件枚举顺序，且不会为 51 万个样本在内存中生成随机排列。

默认 seed：

```text
20260718
```

改变 seed 会得到不同划分。模型实验期间应固定 seed，避免验证集变化。

每个 tar 中的 JSON 标签会增加：

```json
{"race_group":0}
```

它只用于分层采样和分组评估，不应作为模型输入。`manifest.json` 还会记录每个 `race_group` 的 train/val/test 数量。

如果将来处理一个没有“每种族连续 30,000 张”结构的数据集，可以使用：

```text
--race-group-size 0
```

此时退回普通的全局稳定哈希划分。

## 5. 先做小规模试运行

输出目录必须不存在或为空。先处理 100 张：

```powershell
python image_preprocessor.py `
  face_to_ck3_dataset_male_small/face `
  face_to_ck3_dataset_male_small/processed_multiview_preview `
  --labels face_to_ck3_dataset_male_small/labels.jsonl `
  --limit 100 `
  --shard-size 50 `
  --progress-every 10
```

检查输出：

```text
processed_multiview_preview/
  train/train-000000.tar
  val/val-000000.tar
  test/test-000000.tar
  manifest.json
```

由于哈希划分，小样本试运行中某个 split 可能为空，这是正常情况。

## 6. 处理完整数据集

确认预览无误后，使用一个新的输出目录：

```powershell
python image_preprocessor.py `
  face_to_ck3_dataset_male_small/face `
  face_to_ck3_dataset_male_small/processed_multiview `
  --labels face_to_ck3_dataset_male_small/labels.jsonl `
  --workers 4 `
  --shard-size 2000
```

完整运行会自动：

1. 只接受 `face_<数字>.png`；
2. 排除 `test_region.png`；
3. 给约 947 MB 的 `labels.jsonl` 建临时 SQLite 偏移索引；
4. 按 `sample_id` 精确关联图片与标签；
5. 使用多进程解码、裁剪和 JPEG 编码；
6. 稳定划分 train/val/test；
7. 写入原子 tar 分片；
8. 生成 `manifest.json`。

SQLite 只保存 `sample_id -> 文件偏移和长度`，标签正文仍从原始 JSONL 读取，所以不会把全部标签载入内存，也不会在数据库中复制近 1 GB 标签。

每个 tar 样本包含：

```text
face_0001.front.jpg
face_0001.side.jpg
face_0001.json
```

如果系统盘临时空间不足，可把临时索引放到数据盘上已有的空目录：

```powershell
python image_preprocessor.py `
  face_to_ck3_dataset_male_small/face `
  face_to_ck3_dataset_male_small/processed_multiview `
  --labels face_to_ck3_dataset_male_small/labels.jsonl `
  --temp-dir D:/temp
```

## 7. 输出校验

`manifest.json` 包含：

- 发现、选中、成功和跳过的图片数量；
- 自动排除的 PNG 文件名；
- 实际遇到的原图尺寸；
- 裁剪框和输出尺寸；
- JPEG 参数；
- train/val/test 数量；
- 每个 `race_group` 的 train/val/test 数量；
- 各集合分片数量；
- 标签总数、匹配数与未匹配数；
- 总耗时。

正式处理时应满足：

```text
processed = 510000
skipped = 0
excluded_pngs 包含 test_region.png
matched_labels = 510000
unmatched_labels = 0
source_sizes = [[1326, 891]]
race_groups.group_count = 17
每个 race_group = 27000/1500/1500
```

默认发现坏图片或缺失标签会立即停止。只有数据清洗调查时才使用：

```text
--skip-invalid
```

错误会写入 `errors.jsonl`。不要在正式生成最终训练集时忽略错误。

## 8. 离线与在线处理边界

离线脚本只做确定性处理：

- 固定裁剪；
- 固定缩放；
- JPEG 编码；
- 数据划分与分片。

以下增强应在训练 DataLoader 中在线随机执行：

- 水平翻转，概率 0.5；
- 旋转不超过 `±3°`；
- 平移不超过 3%；
- 缩放范围 `0.95～1.05`；
- 轻微高斯模糊、JPEG 退化和传感器噪声；
- 轻微随机遮挡；
- 模拟头发遮住额头/头顶的上半部遮挡。

不要使用大幅透视变换或激进随机裁剪，否则会改变头宽、耳距、下巴和五官比例。

## 9. 双视图几何输入

肤色与采集区间可能相关，几何模型容易把肤色当作脸型捷径。当前模型不再预测颜色，因此可以使用更强的色彩增强来压制这种伪相关。

训练时从同一张对齐图像生成两个几何一致的视图：

1. 主视图使用较强亮度、颜色、色相增强和随机灰度；
2. 参考视图使用轻度光度增强；
3. 对 signed、strength 和 categorical 输出施加跨视图一致性约束。

离线 JPEG 仍保留原始 RGB，不提前灰度化；肤色和眼色由独立程序化模块处理，发色在无毛发输入中不可观测。

使用预训练视觉骨干时，应在 DataLoader 中先把像素转换到 `[0,1]`，再使用对应预训练权重提供的 mean/std，而不是把标准化后的浮点图片保存到磁盘。

## 10. 真人照片推理

固定裁剪只适用于当前 CK3 训练截图。真人照片的构图不固定，推理时必须：

1. 检测人脸和关键点；
2. 根据双眼、鼻子和嘴巴做相似变换；
3. 将耳朵、头顶和下巴对齐到与训练裁剪相近的位置；
4. 输出 `256 × 384` RGB 输入。

不能直接对任意真人照片使用 `(150, 20, 690, 830)`。
