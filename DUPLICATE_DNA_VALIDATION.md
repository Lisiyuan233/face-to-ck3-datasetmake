# 重复 DNA 重渲染对应关系验证

验证入口为 `validate_duplicate_dna_renders.py`，复用字段扫描工具已经保存的 CK3
粘贴、确认、复制回读和截图坐标。

## 已生成计划

仓库中的 `experiments/duplicate_dna_validation` 已包含 100 组确定性抽样计划：

- 前 15 个数值分块各 6 组，后 2 个分块各 5 组；
- 每组当前均为两张相邻历史图片共享逐字节相同的 DNA；
- 每组同时记录前一个和后一个不同 DNA 组，作为一帧错位的负样本；
- seed 为 `20260808`，plan SHA-256 记录在 `protocol.json`。

如需在空目录重新生成：

```powershell
python validate_duplicate_dna_renders.py prepare `
  experiments/duplicate_dna_validation_new `
  --dataset face_to_ck3_dataset_male_small `
  --groups 100 `
  --repeats 1
```

## CK3 重渲染

先打开 CK3 角色设计器并保持与字段扫描相同的界面、分辨率和缩放。建议先跑 3 组门禁：

```powershell
python validate_duplicate_dna_renders.py run `
  experiments/duplicate_dna_validation `
  --limit 3
```

确认三张截图正确后运行完整计划：

```powershell
python validate_duplicate_dna_renders.py run `
  experiments/duplicate_dna_validation
```

运行器会读取 `%APPDATA%/CK3DNAFieldSweep/settings.json`，执行：粘贴 DNA、点击确认、
回读全部脸部字段和颜色、等待稳定、截图。`render_manifest.jsonl` 带截图哈希，可安全续跑；
已完成且哈希一致的重渲染会跳过。

## 自动比较

100 张完成后运行：

```powershell
python validate_duplicate_dna_renders.py analyze `
  experiments/duplicate_dna_validation
```

分析会比较新重渲染与三类历史候选：

- `same`：计划中共享相同 DNA 的历史截图；
- `previous`：数值编号上前一个不同 DNA 组；
- `next`：数值编号上下一个不同 DNA 组。

使用正面、侧面、头部、鼻子、下颌和耳朵的曝光归一化梯度特征，输出：

- `render_comparison.csv`：逐张距离、最近候选和正负间隔；
- `group_comparison.csv`：逐 DNA 组结论；
- `analysis_summary.json`：总体 same-DNA Top-1 率和分类计数；
- `review/`：最可疑 20 组的重渲染/同组/前组/后组拼图。

分类含义：

- `aligned`：重渲染明显更接近同 DNA 历史图；
- `aligned_weak`：同 DNA 最近，但和邻组差距较小；
- `suspected_previous_dna_lag`：更接近前组，符合剪贴板 DNA 滞后一拍；
- `suspected_next_image_shift`：更接近后组，符合截图或编号前移；
- `ambiguous`：动画噪声或同范围脸过于相似，需要查看拼图。

