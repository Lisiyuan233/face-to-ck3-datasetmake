# Face to CK3 训练脚本

训练入口是 `train.py`。推荐配置读取 `processed_multiview/{train,val}/*.tar` 中严格配对的正面和纯侧脸，不依赖额外的 WebDataset 包；原 `processed_front` 配置仍可作为正面基线。

## 环境

当前仓库中的旧 `.venv` 指向已经不存在的 Python，请新建 Python 3.10+ 环境。根据显卡和 CUDA 版本安装匹配的 PyTorch 后，再安装训练依赖：

```powershell
python -m pip install -r requirements-train.txt
```

## 先运行测试

纯数据测试不需要 PyTorch：

```powershell
python -m unittest tests.test_schema_and_shards -v
```

首次训练前需生成只基于 train split、且仅统计可观察样本的类别权重；多视图命令见下文。训练入口会校验该文件的 split、样本数、可观察阈值和 schema SHA-256，避免用验证/测试分布计算损失权重。

安装 PyTorch 并按下文生成 `processed_multiview` 后，先跑两步冒烟训练：

```bash
python train.py \
  --config configs/train_convnext_tiny_multiview.json \
  --smoke-test \
  --device cpu
```

冒烟模式自动改用无预训练的 ResNet-18、`128×192` 输入、batch 2，并只执行 2 个训练 step 和 2 个验证 step；多视图配置仍会实际读取并融合侧脸。

## 正式单卡训练

先生成正面/侧面配对分片：

```bash
python image_preprocessor.py \
  face_to_ck3_dataset_male_small/face \
  face_to_ck3_dataset_male_small/processed_multiview \
  --labels face_to_ck3_dataset_male_small/labels.jsonl \
  --workers 4 \
  --shard-size 2000
```

按可观察阈值重新生成训练集类别权重：

```bash
python tools/build_training_label_stats.py \
  --data-root face_to_ck3_dataset_male_small/processed_multiview \
  --observable-threshold 0.1 \
  --workers 4
```

然后启动多视图训练：

```bash
python train.py \
  --config configs/train_convnext_tiny_multiview.json \
  --device cuda
```

多视图配置使用 ConvNeXt-Tiny、`256×384` 正面强/弱增强和一个纯侧脸，通过门控残差融合特征。micro-batch 16、梯度累积 2，保持有效 batch 32。侧脸不做水平翻转。

快速验证可使用确定性的缩小数据集。例如训练和验证约 10% 数据：

```bash
python train.py \
  --config configs/train_convnext_tiny_multiview.json \
  --data-fraction 0.1 \
  --device cuda
```

训练分片会在完整有序范围内等距选择，验证样本则从所有分片稳定抽样，因此不会像截取前 10% 那样丢失大部分 race group。实际比例和有效样本数会写入 `resolved_config.json` 并在启动日志中显示。正式训练不传该参数，默认使用 `1.0`。

## 几何支路消融

`train_convnext_tiny_multiview_geometry.json` 在原 RGB 多视图模型上增加一个轻量几何支路。数据加载器会在同一次仿射增强后、光度增强前生成两通道形状图：

- 前景通道根据裁剪图上方两角估计背景，保留头部、耳朵、颈部和侧脸轮廓；
- 边缘通道强调眼、鼻、嘴、下颌及侧面深度边界。

正脸和侧脸形状图由小型 CNN 编码，再通过初始偏置为负值的门控残差注入 RGB 特征。它不需要重建 tar 分片或安装额外的人脸模型；关闭 `model.geometry_branch.enabled` 后即回到原架构。

先用相同的 10% 子集做严格消融：

```bash
python train.py \
  --config configs/train_convnext_tiny_multiview_geometry.json \
  --data-fraction 0.1 \
  --device cuda
```

必须与同样 `--data-fraction 0.1` 的 `train_convnext_tiny_multiview.json` 比较，不能拿它和全量正脸实验直接比较。验证输出中的 `geometry_gate_mean` 表示模型平均使用几何支路的程度；若它长期接近零且综合分数没有改善，应停用该支路。通过小规模消融后，再去掉 `--data-fraction` 运行全量训练。

## 断点恢复

```bash
python train.py \
  --config configs/train_convnext_tiny_multiview.json \
  --resume runs/convnext_tiny_multiview_v1/last.pt \
  --device cuda
```

恢复时会校验 `dna_schema.json` 的 SHA-256；schema 不一致会拒绝加载。

## 多卡训练

在支持 `torchrun` 的环境中：

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/train_convnext_tiny_multiview.json \
  --device cuda
```

训练集按 rank 和 DataLoader worker 分配 tar shard。`world_size × num_workers` 不能超过训练分片数 230。

## 输出

多视图配置默认写入 `runs/convnext_tiny_multiview_v1/`：

- `best.pt`：验证综合分数最佳 checkpoint；
- `last.pt`：最近一个完整 epoch checkpoint；
- `metrics.jsonl`：step 与 epoch 指标；
- `validation-epoch-*.json`：完整验证指标、各字段混淆矩阵和 17 组误差；
- `resolved_config.json`：合并默认值后的实际配置；
- `schema_metadata.json`：字段顺序、类别词表和 schema 校验和。

模型选择分数由 signed MAE、strength MAE 和 `strength >= 0.1` 样本的 observable macro-F1 组成，权重分别为 0.40、0.25、0.35，越低越好。不可观察类别不参与分类 loss；6 个单类别字段不建立分类头，但仍训练其强度。

## 推理与写回 DNA

输入图像必须先按训练规范对齐；当前脚本不会自动做人脸检测：

```bash
python predict.py \
  --checkpoint runs/convnext_tiny_multiview_v1/best.pt \
  --image aligned_front.png \
  --side-image aligned_side.png \
  --output prediction.json \
  --template face_to_ck3_dataset_male_small/dna/face_0001.txt \
  --dna-output predicted_dna.txt \
  --device cuda
```

默认使用 checkpoint 中的 EMA 参数；添加 `--raw-weights` 可改用原始参数。输出前同样校验 schema SHA-256。预测结果不包含颜色；写回 DNA 模板时保留模板原有的发色、肤色和眼色，后续可由独立的程序化颜色提取模块覆盖。
