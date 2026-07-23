# Face to CK3 训练脚本

训练入口是 `train.py`。它直接读取 `processed_front/{train,val}/*.tar`，不依赖额外的 WebDataset 包。

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

首次训练前生成只基于 train split 的类别权重统计：

```powershell
python tools/build_training_label_stats.py --workers 4
```

训练入口会校验该文件的 split、样本数和 schema SHA-256，避免用验证/测试分布计算损失权重。

安装 PyTorch 后先跑两步冒烟训练：

```powershell
python train.py --config configs/train_convnext_tiny.json --smoke-test --device cpu
```

冒烟模式自动改用无预训练的 ResNet-18、`128×192` 输入、batch 2，并只执行 2 个训练 step 和 2 个验证 step。

## 正式单卡训练

```powershell
python train.py --config configs/train_convnext_tiny.json --device cuda
```

默认配置：ConvNeXt-Tiny、ImageNet 预训练、`256×384`、强/弱增强一致性双视图、全局 batch 32、BF16、30 epoch。颜色不再作为训练目标。显存不足时优先把 `batch_size` 调小并按比例增大 `gradient_accumulation`。

快速验证可使用确定性的缩小数据集。例如训练和验证约 10% 数据：

```powershell
python train.py `
  --config configs/train_convnext_tiny.json `
  --data-fraction 0.1 `
  --device cuda
```

训练分片会在完整有序范围内等距选择，验证样本则从所有分片稳定抽样，因此不会像截取前 10% 那样丢失大部分 race group。实际比例和有效样本数会写入 `resolved_config.json` 并在启动日志中显示。正式训练不传该参数，默认使用 `1.0`。

## 断点恢复

```powershell
python train.py `
  --config configs/train_convnext_tiny.json `
  --resume runs/convnext_tiny_geometry_v1/last.pt `
  --device cuda
```

恢复时会校验 `dna_schema.json` 的 SHA-256；schema 不一致会拒绝加载。

## 多卡训练

在支持 `torchrun` 的环境中：

```powershell
torchrun --standalone --nproc_per_node=2 train.py `
  --config configs/train_convnext_tiny.json `
  --device cuda
```

训练集按 rank 和 DataLoader worker 分配 tar shard。`world_size × num_workers` 不能超过训练分片数 230。

## 输出

默认写入 `runs/convnext_tiny_geometry_v1/`：

- `best.pt`：验证综合分数最佳 checkpoint；
- `last.pt`：最近一个完整 epoch checkpoint；
- `metrics.jsonl`：step 与 epoch 指标；
- `validation-epoch-*.json`：完整验证指标、各字段混淆矩阵和 17 组误差；
- `resolved_config.json`：合并默认值后的实际配置；
- `schema_metadata.json`：字段顺序、类别词表和 schema 校验和。

模型选择分数由 signed MAE、strength MAE 和 categorical macro-F1 组成，权重分别为 0.40、0.25、0.35，越低越好。6 个单类别字段不建立分类头，但仍训练其强度。

## 推理与写回 DNA

输入图像必须先按训练规范对齐；当前脚本不会自动做人脸检测：

```powershell
python predict.py `
  --checkpoint runs/convnext_tiny_geometry_v1/best.pt `
  --image aligned_face.png `
  --output prediction.json `
  --template face_to_ck3_dataset_male_small/dna/face_0001.txt `
  --dna-output predicted_dna.txt `
  --device cuda
```

默认使用 checkpoint 中的 EMA 参数；添加 `--raw-weights` 可改用原始参数。输出前同样校验 schema SHA-256。预测结果不包含颜色；写回 DNA 模板时保留模板原有的发色、肤色和眼色，后续可由独立的程序化颜色提取模块覆盖。
