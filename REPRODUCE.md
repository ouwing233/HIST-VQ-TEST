# HiST-VQ 复现实现说明

基于 **SMQ** 代码库复现 **HiST-VQ**（*Unsupervised Skeleton-Based Action
Segmentation via Hierarchical Spatiotemporal Vector Quantization*）。按官方声明，
HiST-VQ 在 SMQ 之上构建；本复现严格依据提供的实现规格书逐项落地。

**目标**：先把整条流水线（训练 + 评估 + 匈牙利匹配指标）跑通，指标对齐为次要。
当前状态：✅ 在 LARa 上端到端跑通。

## SMQ → HiST-VQ 的改动落地

复用 SMQ 不动的部分：关节解耦 MS-TCN 编码器、1 秒不重叠 patch 分块、
inter-joint 距离重建损失、EMA 码本更新、全局匈牙利评估协议。

新增/改造（核心贡献）：

1. **两级串联量化**（`src/model/hist_quantizer.py`）
   - 第一级（子动作 `Z`）：`αK` 个码字，每个 `∈ ℝ^{P×V×D}`；patch `p_k → z_{j*}`（最近邻）。
   - 第二级（动作 `A`）：`K` 个码字；输入是**第一级码字** `q^Z_k`（不是原始 patch），`q^Z_k → a_{i*}`。
   - 串联结构：`p_k → z → a`。分割标签 = **第二级（动作）索引**。

2. **双 commitment 损失**（编码器侧，直通梯度）
   - `L_commitZ = ‖p − sg[q^Z]‖²`（拉 patch 向子动作）
   - `L_commitA = ‖q^Z_st − sg[q^A]‖²`（拉子动作向动作）

3. **时空双解码**（`src/model/histvq.py`）
   - **空间解码器**（编码器镜像 TCN）：输入 `Q_A` → 重建骨架 `Ŝ`。
   - **时间解码器**（2 隐层 MLP）：输入 per-patch `Q_Z` → 预测 patch 级时间戳 `T̂ ∈ ℝ^{N×M}`。
   - "难任务（空间）配高层 `Q_A`、易任务（时间）配低层 `Q_Z`" 的分流是论文关键 insight。

4. **两级 EMA + 死码复活**（`hist_quantizer.py`）
   - 两套 EMA（`β=0.5`）：`Z` 累加分配到的 patch，`A` 累加分配到的 `q^Z`。
   - 码本注册为 buffer（不进优化器，仅 EMA 更新）。
   - 死码复活：`Z` 计数 `<ν_z(=3)` 用随机 patch 重置，`A` 计数 `<ν_a(=1)` 用随机 `q^Z` 重置。

5. **patch 级时间戳构造**（`model.py:make_timestamps`，CTE 风格）
   - 每条序列的有效 patch 线性映射到 `[0,1]`：`tau[m] = m/(M_valid−1)`；padding patch 掩码掉。

6. **总损失**（`L_spat` 复用 SMQ 的 inter-joint 距离 MSE）
   ```
   L = λ_commit·(L_commitZ + L_commitA) + λ_spat·L_spat + λ_temp·L_temp
   ```

## 超参（默认值，见 `main.py`）

| 项 | 值 |
|---|---|
| `α`（第一级/第二级码本比） | 2（→ `Z=2K`, `A=K`） |
| EMA `β` | 0.5 |
| 死码阈值 `ν_z, ν_a` | 3, 1 |
| `λ_commit` | 1.0 |
| `λ_spat` | 0.001 |
| `λ_temp` | HuGaDB=0.2 / LARa=0.1 / BABEL=0.02（按数据集） |
| patch `P` | HuGaDB=60 / LARa=50 / BABEL=30 |
| latent `D` | 16 |
| 编码器/空间解码器 | 2-stage TCN，每 stage 3 dilated residual 层 |
| 时间解码器 | 2 隐层 MLP（hidden=128） |
| 优化器 | Adam, lr=5e-4 |
| 码本初始化 | 随机（kaiming） |

> ⚠ 规格书中标 ⚠[推断] 的项（时间戳归一化形式、死码复活采样粒度、TCN 通道数等）
> 均按规格书建议取 SMQ 默认值或 CTE 惯例实现；`λ_temp` 按数据集可调。
> LARa 的 `in_channels=6`（SMQ 的 `lara.py` 预处理输出 6 维/关节：3 静态 + 3 坐标）。

## 数据

真实数据集需自行下载/预处理（脚本见 `src/data/`，沿用 SMQ）：
- **LARa v3**（22 关节，3D）：Zenodo `records/8189341`
- **HuGaDB**、**BABEL** 见 SMQ README

预处理产出 `data/<dataset>/{features/*.npy, groundTruth/*.txt, mapping/mapping.txt}`，
特征张量形状 `(C, T, V, M)`。

## 跑通验证（Smoke Test，已执行 ✅）

环境：Python 3.11 + CPU PyTorch。在 3 条真实 LARa 序列上：

```bash
python main.py --action=train --dataset=lara --epoch=5
python main.py --action=eval  --dataset=lara --ckpt models/lara/epoch-5.model --epoch 5
```

实测：
- 训练正常前向/反向，四项损失（Spat / Temp / CommitZ / CommitA）均有限且在更新；
- checkpoint 里确认两级码本 `Z=(16,50,352)`、`A=(8,50,352)` 与时空双解码器齐全；
- 评估正常输出 Local / Global 匈牙利匹配的 MoF / Edit / F1@{0.10,0.25,0.50}。

> 注：仅 5 epoch、3 条序列，指标偏低（MoF ~31%）属正常——本阶段目标是"跑通"，
> 复现论文量级需用完整数据集与更长训练（论文 LARa 量级 MoF ~45.9）。

## 验收自检（规格书 §11，供后续对齐）

- 去掉 `L_spat` 应**大幅崩**（空间重建是地基）；
- 去掉 `L_temp`/commitment 应温和下降；
- 空间解码器误用 `Q_Z`、时间解码器误用 `Q_A` 应掉点（确认分流：本实现空间=`Q_A`、时间=`Q_Z`）。
