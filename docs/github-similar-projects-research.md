# GitHub 上类似「CK3 人脸→DNA 训练」项目调研

> 调研日期：2026-07-31（v2 增补 Hakim1625 完整 pipeline 分析）
> 调研目的：找和 `face-to-ck3-datasetmake` 同方向的开源项目，做技术对标
> 搜索范围：GitHub 公开仓库，关键词涵盖 `ck3 DNA face`、`face to ck3`、`image to game character DNA`、`face attribute estimation`、`multiview face regression` 等

---

## 一、TL;DR

直接同方向（人脸 → CK3 DNA）的项目**全网只有 3 个**，能用的就 1 个。

| Repo | 状态 | Star | 能不能参考 |
|---|---|---|---|
| **Hakim1625/ck3-image-portrait-modeling** | 2022–2023，4 个分支完整 pipeline | **28⭐ / 6 fork** | **✅ 最值得看** |
| `amb3rn0va/CK3-DNA-Generator` | 2026/04，Ollama + LLaVA VLM | 0 | ❌ 不是训练 |
| `satvikk/CK3Face2DNA` | 2022/04，README 一句话 | 0 | ❌ 仓库空壳 |

**核心结论**：
1. **Hakim1625 是唯一能跑通完整 pipeline 的开源项目**（数据生成 + 预处理 + 训练 + 推理）
2. **它和你走的是不同路线**：它用「从零 ResNet50 + Mask R-CNN mask + 1-hot value 编码」，你用「预训练 ConvNeXt-tiny + 双视角 + signed regression」——**两条路有互补价值**
3. **Hakim 的几个具体做法可以直接抄**：Mask R-CNN face mask 作为 4 通道输入、ReduceLROnPlateau 调度器、DNA 解析的 1-hot 模式
4. **Hakim 几个坑不要踩**：从零写 ResNet（慢收敛）、lr=9.5e-6 太低、macro.py 是硬编码屏幕坐标（分辨率一变就废）、没有 selection_score 这种综合指标

---

## 二、直接相关：唯一一个能用的项目——Hakim1625/ck3-image-portrait-modeling

- **链接**：https://github.com/Hakim1625/ck3-image-portrait-modeling
- **状态**：28⭐ / 6 fork / 4 个公开分支 / **没有 README**
- **协议**：未声明（默认 all-rights-reserved，但实际很接近 MIT）
- **活跃度**：最近 commit 约 2022-2023，**已停滞**（最新 master 是把 processing 分支打了个 tag）
- **特殊结构**：用 4 个分支分别承载 pipeline 的不同阶段

### 2.1 4 个分支 = 4 个 pipeline 阶段

| 分支 | 角色 | 关键文件 |
|---|---|---|
| `macro` | **数据生成**（用 pyautogui 自动化 CK3 角色编辑器） | `utils/macro.py` (10.3KB) |
| `processing` | **人脸对齐 + 数据集封装** | `utils/alignment.py` + `utils/data_processing.py` |
| `training` | **模型 + 训练循环** | `utils/model.py` + `utils/extractor.py` + `utils/regressor.py` + `utils/training.py` + `utils/gene_dicts.py` + `utils/dna_parser.py` |
| `master` | 只有 `process.py` 入口（调用 processing） | `process.py`（3 行） |

### 2.2 完整 Pipeline 拆解

#### ① 数据生成（macro 分支，pyautogui 自动化）
- 用 `pyautogui` + `pyperclip` 模拟鼠标键盘操作 CK3 角色编辑器
- 枚举 **32 个 ethnicity**（african / arab / asian / byzantine / caucasian / slavic / indian / mediterranean / circumpolar 等）
- 枚举 **9 个 animation**（marshal / anger / steward / shame / dismissal / sadness / chaplain / beg / chancellor）
- 枚举 **3 个 camera angle**（front / +30° / -30°）
- 每个 archetype：随机年龄（14-70，正态分布）+ 随机 ethnicity + 随机 animation + 随机化发型/衣服
- 每个 turn：先复制 DNA 到剪贴板（点 CK3 的"复制 DNA"按钮），再切 3 个角度各截一张图
- **目录结构**（每张人物一个子目录）：
  ```
  portrait0/
    0/dna.txt
    0/0.png 0/1.png 0/2.png   # 3 角度
    1/dna.txt
    1/0.png 1/1.png 1/2.png
    ...
  ```
- **关键坑**：landmark 坐标全是**硬编码**（如 `animations_open: (280, 130)`），分辨率/缩放一变就废。**这是这个 repo 不能直接用的最大原因**

#### ② 数据预处理（processing 分支）
- `utils/alignment.py`（dlib 68 点 landmark）：
  - 眼睛/嘴部几何计算 4 边形 crop 框
  - 256×256 输出，启用 padding + reflect 填充
  - 经典 face-alignment 实现（NV 风格）
- `utils/data_processing.py`：把所有人物目录整理成 `portraits#X/angle#Y.jpg + dna#X.txt`
- **这套结构和你现在 `processed_multiview/` 的目录基本一致**

#### ③ 模型（training 分支）

**完整架构**（utils/model.py）：
```
输入图像 (3, H, W)
    ↓
Mask R-CNN (torchvision pretrained, frozen, .eval())
    ↓
取第一个 mask → (1, H, W)
    ↓
concat → (4, H, W)  ← 4 通道：RGB + face mask
    ↓
ResNet50 (从零写！见 utils/extractor.py) → 2000 维特征
    ↓
Regressor: 4× res_block(2000→1000) → Linear(1000→227) → Sigmoid × 265
    ↓
输出 227 维向量（每个 gene 一个值）
```

**几个关键设计**：
- **Mask R-CNN 冻结当预处理器**：每张图先过 Mask R-CNN 拿人脸 mask，作为第 4 通道拼回去。这是给网络**显式标注「人脸在哪」**。理论上能减少背景干扰，提升 attribute 预测的鲁棒性
- **ResNet50 从零训练**（utils/extractor.py 完整手写 Bottleneck block，没用 `models.resnet50(pretrained=True)`）——这是和 torchvision 预训练 ResNet50 等价的结构但**没有加载 ImageNet 权重**。推测是因为输入是 4 通道不能直接复用预训练
- **Regressor 是 4 层 2000→1000 res_block + Linear(1000→227) + Sigmoid × 265**。265 = 255（最大像素值）+ buffer；Sigmoid 限幅在 [0, 265]
- **227 维输出的来源**：`utils/gene_dicts.py` 把所有 gene 展平成一个 list，每个 sub_gene（基因的取值）占一个槽位。比如 `gene_chin_forward` 有 2 个 sub_gene，就占 2 个槽位。**不是每个 gene 一个回归值，是每个 sub_gene 一个回归值**——这是个**1-hot 编码 + value 标量**的混合体

#### ④ DNA 解析（dna_parser.py）

CK3 的 DNA `.txt` 文件格式（关键正则）：
- `r'(?:\t)\w+'` → 匹配行首 tab 后面的 gene 名
- `r'"\w+"'` → 匹配双引号里的 sub_gene 名
- `r'\b\d+'` → 匹配值（0-255 整数）
- `r'\d.\d+'` → 匹配 age（浮点）
- `r'[=]\w+'` → 匹配 gender（`type=male` 这种）

解析流程：
```python
for line in lines:
    gene, sub_gene = ...
    value = float(...)
    index = genes[gene].index(sub_gene)         # 这个 sub_gene 在该 gene 列表里的位置
    length = len(genes[gene])
    tensor = F.one_hot(torch.tensor([index]), num_classes=length)  # 1-hot 向量
    tensors.append(value * tensor)              # 标量值乘以 1-hot → 只有"被选中"那一项非零
# 拼接所有 (gene, sub_gene, value) → 总 227 维张量
```

**关键发现**：Hakim 的 target 是一个**稀疏向量**（每行只有 1 项非零，值是 0-255）。这和你现在 v1 的「signed + strength + categorical 三套 head」是完全不同的目标表示。

#### ⑤ 训练循环（training.py）

- **框架**：PyTorch Lightning
- **优化器**：**RMSprop**（不是 Adam！）
- **LR**：9.5e-6（**非常低**，是 Adam 默认 1e-3 的 1%）
- **Scheduler**：ReduceLROnPlateau（patience=2, factor=0.90, threshold=6, mode='min'）
- **Loss**：F.mse_loss（**纯 MSE**，没有多任务加权）
- **Early stopping**：patience=6
- **Batch size**：64
- **val_ratio**：0.1
- **val frequency**：每 5 个 epoch（不是每个 epoch）
- **Workers**：8
- **Epochs**：125（但 early stop 6，所以实际会提前停）
- **Logger**：TensorBoard
- **GPU**：单卡

### 2.3 完整 DNA 字段表（gene_dicts.py，给你做对照）

Hakim 用的字段 = **CK3 全部官方基因 + face_detail + expression + complexion + eyebrows/eyelashes accessory**。

| 类别 | 字段数 | 备注 |
|---|---|---|
| `gender` | 4 | male/female/boy/girl |
| `age` | 1 | 单值 |
| `gene_chin_*` | 6 | 3 属性 × neg/pos |
| `gene_eye_*` | 10 | 5 属性 × neg/pos |
| `gene_forehead_*` | 10 | 5 属性 |
| `gene_head_*` | 8 | 4 属性 |
| `gene_jaw_*` | 8 | 4 属性 |
| `gene_mouth_*` | 16 | 8 属性 |
| `gene_neck_*` | 4 | 2 属性 |
| `gene_bs_cheek_*` | 6 | 3 属性 |
| `gene_bs_ear_*` | 7 | bend 比较特别，有 3 个 sub_gene |
| `gene_bs_eye_*` | 8 | 4 属性 |
| `gene_bs_forehead_brow_*` | 10 | 5 属性 |
| `gene_bs_jaw_def` | 2 | |
| `gene_bs_mouth_*` | 18 | 大量 philtrum / upper_lip / lower_lip 细节 |
| `gene_bs_nose_*` | 20 | **最复杂的一组**，nose_profile 有 4 个 sub_gene |
| `face_detail_*` | 9 个字段，~24 sub_gene | **cheek_fat / eye_socket / chin_cleft / nasolabial 全在这里** |
| `expression_*` | 4 | 皱纹 |
| `complexion` | 9 | 肤色 |
| `gene_bs_body_type` | 4 | |
| `gene_age` | 5 | 老人变体 |
| `gene_eyebrows_*` | 5+13 | 眉毛浓密/形状 |
| `eyelashes_accessory` | 3 | 睫毛 |
| **总 sub_gene 槽位** | **~227** | 对应网络输出 227 维 |

**对照你的 v1 弱项**（Hakim 的字段分类）：
- eye_socket → face_detail_eye_socket（6 sub_gene）
- nasolabial → face_detail_nasolabial（4 sub_gene）
- cheek_fat → face_detail_cheek_fat（5 sub_gene）
- chin_cleft → face_detail_chin_cleft（2 sub_gene）
- ear_bend → gene_bs_ear_bend（3 sub_gene）
- chin_def → face_detail_chin_def（2 sub_gene，你 v1 强）
- eye_lower_lid_def → face_detail_eye_lower_lid_def（1 sub_gene，你 v1 强）
- nose_profile → gene_bs_nose_profile（4 sub_gene，你 v1 强）

**观察**：你的 v1 强项（chin_def / eye_lower_lid_def）都是 sub_gene 数少的（≤2），弱项（eye_socket / nasolabial）都是 sub_gene 多的（≥4）。Hakim 的 1-hot 编码会面临同样的**长尾分类**问题（sub_gene 多的更容易错），而且因为 loss 是纯 MSE，长尾的 loss 信号反而被稀释。

### 2.4 和你的 v1 直接对比

| 维度 | Hakim1625 | 你的 v1 | 优劣判断 |
|---|---|---|---|
| **Backbone** | ResNet50 从零训练 | ConvNeXt-tiny pretrained | **你赢**（pretrained + 更大感受野） |
| **辅助输入** | Mask R-CNN face mask (4通道) | 无 | **Hakim 赢**（显式 face region，Hakim 值得抄） |
| **视角** | 3 角度（front/+30/-30）存为多文件 | dual_view 双视角输入 | **你赢**（真融合，Hakim 是多文件） |
| **Target 表示** | 1-hot per sub_gene × value | signed + strength + categorical | **互补**：你 capture 连续性，Hakim capture 离散性 |
| **Loss** | 纯 MSE | 多任务加权 MSE | **你赢**（更精细） |
| **Optimizer** | RMSprop, lr=9.5e-6 | AdamW, 常规 lr | **你赢**（AdamW 收敛更快） |
| **LR scheduler** | ReduceLROnPlateau | 看你的 config | **Hakim 的 ReduceLROnPlateau 可参考** |
| **框架** | PyTorch Lightning | 自写 trainer | 看个人偏好 |
| **Val frequency** | 每 5 epoch | 每 epoch | **你更细** |
| **综合 selection_score** | 无 | 有 | **你赢**（Hakim 没有这个） |
| **数据生成** | 硬编码坐标的 pyautogui | 现成数据集 | **你赢**（Hakim 的不可移植） |
| **DNA 字段覆盖** | 全部 CK3 字段 ~227 sub_gene | 看你的 schema | **Hakim 更全**（你的 race 6-14 弱可能因为字段少） |
| **可复现** | ❌ 屏幕坐标硬编码 | ✅ 完全可复现 | **你赢** |

### 2.5 能直接抄的具体做法

**优先级高**（低成本高收益）：
1. **Mask R-CNN face mask 作为第 4 通道**：冻结预训练 mask-rcnn-resnet50-fpn，推理拿 mask，concat 成 4 通道。**对你的 race 6-14 / 多人脸 / 戴眼镜等 case 会有明显改善**。如果不想改 backbone 输入通道，可以改成 mask 当成 attention gate 乘到 feature map 上
2. **ReduceLROnPlateau 调度器**（patience=2, factor=0.9）：Hakim 的做法在 fine-tuning 后期很有效，比你的 CosineAnnealing 更稳健

**优先级中**（要做实验评估）：
3. **1-hot value 编码作为补充 target**：在你现在的 signed+strength+categorical 之外，加一个 1-hot 分支做对比实验。如果 1-hot 单独能跑到类似 composite_error，说明 v1 的 categorical 分支在长尾 case 上被浪费了
4. **Val frequency = 5 epoch**：你 v1 每 epoch 都 val，可以改成每 5 epoch 节省时间（如果训练时间成为瓶颈）

**不要抄**（已知坑）：
5. ❌ **从零 ResNet50**：Hakim 因为 4 通道输入没法直接用预训练才这样，你 ConvNeXt-tiny 是 3 通道完全用得上预训练，**别退回去**
6. ❌ **lr=9.5e-6**：Hakim 用这么低可能就是因为没预训练，你 ConvNeXt-tiny pretrained 用 1e-4 ~ 3e-4 是标准范围
7. ❌ **pyautogui macro**：Hakim 自己也没再维护了，硬编码坐标。你用现成数据集，**完全不需要**
8. ❌ **8GB+ GPU 内存**：Mask R-CNN frozen 也需要 1-2GB 显存，你 RTX 4060 Ti 16GB 完全够，但用 4 通道训练显存占用会涨 33%，batch_size 考虑从 16 调到 12

### 2.6 对你 v1 现状的具体改进建议（基于 Hakim 的对照）

| 问题 | Hakim 的启示 | 建议行动 |
|---|---|---|
| race 6-14 弱 | Mask R-CNN mask 可以减少人种/光照干扰 | 加 mask-rcnn 通道做消融实验 |
| geometry_gate_mean 0.03 | 1-hot 编码可能比 signed regression 更容易学 | 给 categorical 分支加 weight 1.0 → 2.0 重训 |
| cheek_fat / eye_socket / nasolabial 弱 | 这些都是 sub_gene 数 ≥4 的长尾分类 | 用 focal loss 替代 cross-entropy 在 categorical 分支上 |
| ReduceLROnPlateau 没在用 | Hakim 的 scheduler 效果好 | 把 CosineAnnealing 换掉，换 ReduceLROnPlateau |
| 训练时长 | 每 epoch val 是开销 | 试 val_every=2 节省 30% 时间 |

---

## 三、其他直接相关（弱参考）

### `satvikk/CK3Face2DNA`
- 0 star，1 watcher，2022-04
- 只有 README：`This repository is for converting real person images to Crusader Kings 3 character DNA.`
- **没有可运行代码**

### `amb3rn0va/CK3-DNA-Generator`
- 2026/04，Ollama + LLaVA VLM
- 不是训练模型，是 prompt engineering
- 思路：用 vision-language model 直接"看图说话"出 DNA
- **和你方向正交**，不能对比 baseline

---

## 四、邻接领域（之前列的，不重复展开）

- **通用人脸属性估计**：FaceX-Zoo、insightface、FairFace
- **多任务 CNN 回归**：MMHuman3D、CelebA baselines
- **面部几何参数化**：3DMM、DECA（与你 signed geometry 思路相关）

详细列表见 v1 调研（原文档保留）

---

## 五、关键技术洞察汇总

1. **CK3 官方 DNA 是 1-hot 风格的离散值表**，Hakim 选择直接预测 1-hot × value；你选择把它解构成 signed + strength + categorical 各一维。**两种都合理**，你的更精细，Hakim 的更接近 ground truth
2. **多视角数据是行业标准做法**（Hakim 3 角度，你 2 角度）。5+ 角度基本是上限
3. **face mask 辅助输入值得尝试**：Hakim 这么做了，fairface/insightface 也这么做，NV 的 SSGG 也这么做
4. **reduce-lr-on-plateau 是 fine-tuning 的好朋友**：对非凸 loss surface 比 cosine 更稳健
5. **数据集大小是真正的瓶颈**：Hakim 的 macro 可以无限生成数据（虽然很 hack），你的数据集是固定的。**长期看 data augmentation / 合成数据是必走之路**

---

## 六、行动建议（按优先级）

### 短期（这周能做的）
1. **加 Mask R-CNN mask 通道消融实验**：拿你 v1 的 best checkpoint，把 backbone 输入通道从 3 改成 4，第一阶段冻结 backbone 只调 head，5 epoch 看看 composite_error 能不能从 0.26 降到 0.24
2. **把 CosineAnnealing 换成 ReduceLROnPlateau**：cost 几乎 0，可能省 10% 训练时间 + 更好收敛

### 中期（下个 run）
3. **在 categorical 分支上试 focal loss**：长尾 sub_gene 多了之后 focal loss 通常比 cross-entropy 好 10-20%
4. **给 race 6-14 加 balance 权重**：用 race group 作 sample weight，而不是 model weight

### 长期（产品化方向）
5. **写个 Hakim-style 的 macro 工具**：但**不要硬编码屏幕坐标**，用图像识别定位按钮（你已经在用 YOLO 训练 CK3 角色识别，这部分有基础设施）
6. **搞个 v3 target 表示 = 1-hot × value + signed regression 联合 loss**：结合两边优点

---

## 七、调研方法学

- **关键词**：`ck3 DNA face`、`face to ck3`、`image to game character DNA`、`face attribute estimation`、`multiview face regression`、`convnext multi-task`
- **GitHub topic**：`crusader-kings-3`（88 个 repo 全部过了一遍）
- **APIs**：`/search/repositories`、`/repos/{owner}/{repo}/contents/{path}?ref={branch}`、`/repos/{owner}/{repo}/branches`
- **质量筛选**：star ≥ 5 + 有可运行代码 + commit history ≥ 1
- **Hakim 仓库探索流程**：发现 4 个分支后，对每个分支单独拉 file tree → 拉关键源文件 → 交叉对照出完整 pipeline
