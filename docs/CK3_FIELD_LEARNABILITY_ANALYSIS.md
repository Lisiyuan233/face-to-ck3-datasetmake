# CK3 字段可学习性分析（105,865 数据集）

分析时间：2026-09-01（基于 v2 epoch16 验证指标）。方法：以无脑基线为对照——连续/强度字段用训练集均值预测器的 val MAE，分类字段用众数预测器的 val 准确率；模型显著优于基线 = 可学习。基线基于全部 5,293 个 val 样本实算（脚本 /tmp/learnability.py + /tmp/ratio.py 思路可复现）。

## 总览

| 类别 | 数量 | 结论 |
|---|---|---|
| signed 连续 | 67 | 全部可学习（比值 0.13~0.69） |
| categorical 分类 | 16 | 10 个可学习；**6 个单一类别，结构性不可学习** |
| color 颜色 | 3 | **标签完整但未接入训练（scalar 目标为空，loss=0）** |

## 结构性不可学习（全数据集单一类别）

gene_bs_ear_inner_shape（耳内形态）、gene_bs_mouth_lower_lip_def（下唇立体感）、gene_bs_mouth_philtrum_def（人中立体感）、gene_bs_mouth_upper_lip_def（上唇立体感）、face_detail_nose_tip_def（鼻尖立体感）、face_detail_temple_def（太阳穴立体感）。
这 6 个字段被管线主动剔除，不参与分类训练：model.py 的 categorical_heads 只为非恒定字段构建（无分类头），losses.py 的分类交叉熵只遍历 active_categorical_indices（无 loss），metrics.py 的验证同样只跟踪活跃字段。唯一参与训练的是强度回归——strength_head 覆盖全部 16 个字段，其强度（唯一类别的程度值）有变异且已学到（比值 0.09~0.10）。要学习它们的类别需 CK3 生成器先产生变异。

## 弱学习字段（可学习但信号弱）

- signed（模型/基线比值 0.49~0.69）：mouth_open、mouth_corner_depth、philtrum_width、philtrum_shape、nose_ridge_angle、eye_height、nose_size、mouth_lower_lip_size、upper_lip_profile、mouth_height
- categorical：chin_cleft 准确率 0.565 vs 众数基线 0.513（接近随机；observable 口径 0.958 是剔除不可观测样本后的值）
- strength：mouth_upper_lip_def 比值 0.891（≈未学到，亦为单类字段）

## 学习充分度参考（signed 比值两端）

最充分：jaw_def 0.13、chin_forward 0.14、eye_upper_lid_size 0.16、neck_length 0.17、head_profile 0.18、jaw_angle 0.18
最弱：见上。全部 67 个字段比值均 <0.7，无不可学习项。

## 颜色字段（最大遗漏）

eye/hair/skin_color：labels.jsonl 每样本 9 个浮点、missing=0、schema 已注册，但 train_label_stats.scalar_mean 为空、metrics scalar 分量恒 0——未作为目标训练。loss 配置 scalar_weight=1.0 现成，接入即可开训。

## 建议

1. 把 colors 接入 scalar 目标（改动在数据管线/schema→targets 映射处）；
2. 弱学习 signed 字段与扩充方案的定向干预样本线对齐；
3. 6 个单类字段在生成器侧制造变异前不值得投入。


---
**时区记录**：本文撰写时服务器时钟为 UTC（比北京慢 8 小时）。2026-09-01 18:33（北京）起服务器显示时区已改为 Asia/Shanghai，此后新日志与文件时间均为北京时间；本文内历史时间戳保持 UTC 原值。
