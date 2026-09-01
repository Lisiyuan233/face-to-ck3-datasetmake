# 105865 第二轮热启动训练与 bf16 排障记录

记录时间：2026-09-01（服务器时钟 UTC，比北京时间慢 8 小时；本文所有时间均为 UTC）

## 一、背景

第一轮训练 `convnext_tiny_multiview_male_small_105865_v1` 已于 08-31 15:10–20:40 正常跑完
（30 epochs，无报错），最终指标：

| 指标 | epoch 0 | epoch 29（最终） |
|---|---|---|
| val_score（越小越好） | 0.5441 | 0.0782 |
| signed_mae | 0.478 | 0.137 |
| observable_macro_f1 | 0.207 | 0.967 |

收尾时 val_score 仍以约 0.002/epoch 的速度缓慢下降，未完全平台化，因此决定加练一轮。

## 二、第二轮任务定义（热启动，非 resume）

要求：从 v1 **最后的检查点**（`last.pt`，epoch 29）继续，但**不是恢复训练**：

| 项 | resume | 本轮采用的热启动 |
|---|---|---|
| 模型权重 | 恢复 | 恢复（取 EMA 影子权重） |
| 优化器状态 | 恢复 | **全新** |
| 学习率调度 | 接着走 | **从最初配置从头走完整调度**（warmup 5% → 峰值 → 余弦衰减） |
| epoch 计数 | 接续 | 从 0 重新计 |
| 早停/最优状态 | 恢复 | 重置 |

实现上用的是 train.py 自带的 `--finetune-from`（默认加载 EMA 权重，`load_finetune_weights()`
只取模型参数，不加载 optimizer/scheduler/epoch），完全符合上述语义。初始化审计信息见
run 目录 `initialization.json`（mode=finetune, source_epoch=29, weight_source=ema）。

超参与最初配置 `train_convnext_tiny_multiview_male_small_105865.json` 完全一致
（30 epochs / freeze_backbone_epochs=2 / batch 16 × 梯度累积 2 / backbone_lr 1e-4 /
head_lr 3e-4 / warmup 0.05 / min_lr_ratio 0.02 / ema_decay 0.9999 / 早停 patience 5），
唯一改动见下文 fp16。

## 三、bf16 崩溃与排障（重点）

### 现象

- 首次启动（08-31 23:55）：rank 2 在第一个 forward 崩，cuDNN
  `FIND was unable to find an engine to execute this computation`；
- 二次启动：三个 rank 全崩同样错误，**可稳定复现，非偶发**。

### 排查过程

1. 三卡各自跑 bf16 Conv2d 小测试 → 全部通过（排除单卡硬件问题）；
2. 单卡跑 convnext_tiny 完整模型 → fp32 通过，autocast bf16 / 显式 bf16 **全挂**
   （"GET was unable to find an engine"，不开 benchmark 也一样）；
3. 读 `ck3_training/engine.py` 的 `amp_settings()`：配置 `amp=bf16` 时本有
   bf16→fp16 运行时回退，判断依据是 `torch.cuda.is_bf16_supported()`；
4. 实测本机 `is_bf16_supported()` 返回 **True** —— 根因确认。

### 根因

2080Ti（sm_75 / Turing）**没有原生 bf16 支持**，cuDNN 也没有 Turing 的 bf16 卷积引擎，
bf16 前向必然失败。105865_v1 之所以能跑：其**启动配置实际是 fp16**
（run 目录 resolved_config.json 可证），磁盘上的配置文件仍是 bf16——本轮按磁盘文件复制配置
才踩坑。上月 34780 训练在 4060 Ti 16G（Ada 架构，原生 bf16）上进行，bf16 直接有效，从未触发该问题。

本轮失败是因为 venv 里的新 torch（2.13.0+cu130，08-31 14:56 安装）把
`is_bf16_supported()` 的默认参数改成了 `including_emulation=True`：sm_75 上能"模拟创建"
bf16 张量即返回 True，但这只代表存储/模拟可用，不代表 cuDNN 有计算引擎 →
`amp_settings()` 的回退被绕过 → autocast 进 bf16 → cuDNN 找不到引擎 → 崩溃。

### 修复

新配置显式钉死 `"amp": "fp16"`（sm_75 有 fp16 tensor core，cuDNN 支持完好）：
`configs/finetune_male_small_105865_v2.json`。实测 fp16 前向+反向正常后重启，一次通过。
吞吐 51–54 samples/s，与 v1 完全一致（v1 的 resolved config 也确认为 fp16）。

### 遗留建议（未改）

`ck3_training/engine.py` 中 `torch.cuda.is_bf16_supported()` 建议改为
`torch.cuda.is_bf16_supported(including_emulation=False)`，让本机 bf16→fp16
自动回退恢复生效，避免以后新配置再踩同一个坑。

## 四、产物清单

| 文件 | 说明 |
|---|---|
| `configs/finetune_male_small_105865_v2.json` | 第二轮配置（amp=fp16，其余同最初配置） |
| `scripts/start_finetune_male_small_105865_v2.sh` | 启动脚本（3 卡 torchrun + nohup + pid 管理，可重复执行） |
| `train_105865_v2.log` | 训练日志（repo 根目录） |
| `runs/convnext_tiny_multiview_male_small_105865_v2/` | 输出目录（checkpoints / metrics.jsonl / validation-epoch-*.json / initialization.json） |
| `runs/convnext_tiny_multiview_male_small_105865_v1/last.pt` | 权重来源（epoch 29，EMA） |

## 五、运行状态（终版，09-01 05:40 UTC）

- 00:05 启动，30/30 epoch 于 05:35 UTC 完成，全程无报错，吞吐稳定 51~54 samples/s/卡；
- **终值：val_score 0.051860（起点 0.078219，-33.7%）、signed_mae 0.08720、observable_macro_f1 0.9764**；
- 逐 epoch 单调改善无反弹，无过拟合迹象（val loss 30 轮连降，train−val gap 稳定为负）；
- best.pt / last.pt / best_continuous.pt / best_categorical.pt 均为 epoch 29；
- 与上月 34780（50 epoch）终值对比：总分 -32%，65/67 signed 字段、16/16 strength 字段反超。

## 六、运维速查（256ram 上）

```bash
# 实时日志
tail -f ~/face-to-ck3-datasetmake/train_105865_v2.log

# 每 epoch 验证指标
grep val_score ~/face-to-ck3-datasetmake/train_105865_v2.log

# 进程状态
kill -0 $(cat ~/face-to-ck3-datasetmake/runs/convnext_tiny_multiview_male_small_105865_v2/train.pid) && echo running

# 重启（脚本自带旧进程清理；仅必要时使用）
bash ~/face-to-ck3-datasetmake/scripts/start_finetune_male_small_105865_v2.sh
```


---
**时区记录**：本文撰写时服务器时钟为 UTC（比北京慢 8 小时）。2026-09-01 18:33（北京）起服务器显示时区已改为 Asia/Shanghai，此后新日志与文件时间均为北京时间；本文内历史时间戳保持 UTC 原值。
