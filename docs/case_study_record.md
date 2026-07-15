# Case Study 实验记录

## 概览

为论文 Section 4.5 制作三个 Case Study 可视化图，基于真实模型输出数据，非示意图。

---

## 数据收集脚本

### 主脚本（Industrial_and_Scientific 专用）
- **路径**: `figures/case_collect.py`
- **功能**: 对 Industrial_and_Scientific 最优 checkpoint 推理，monkey-patch 三个模块截取中间值
- **输出**:
  - `figures/case_data.npz`：β权重、lens、targets、prototypes
  - `figures/case_assign.pkl`：x_before/x_after、cD_before/cD_after、seqs、assign

### 通用脚本（多数据集）
- **路径**: `figures/case_collect_general.py`
- **用法**:
```bash
python figures/case_collect_general.py \
  --dataset Video_Games \
  --ckpt ./myckpt/Video_Games/<ckpt_name>/best_model.pth \
  --device cuda:4 \
  --out_prefix ./figures/case_data_Video_Games
```
- **输出**: `<out_prefix>.npz` + `<out_prefix>_var.pkl`

### 截取的中间变量
| 变量 | 形状 | 说明 |
|------|------|------|
| `beta` | [N, wm+K] | DEBR 检索权重，前 wm_length 位是 working memory，后 K 位是语义原型 |
| `x_before` | list of [L, d] | WEBD 去噪前的序列 embedding |
| `x_after` | list of [L, d] | WEBD 去噪后的序列 embedding |
| `cD_before` | list of [L', d] | 小波细节系数去噪前绝对值 |
| `cD_after` | list of [L', d] | 小波细节系数去噪后绝对值 |
| `assign` | list of [n, K] | SMC 对每个长期历史 item 分配到各原型的概率 |
| `seqs` | list of [L] | 每个用户的实际历史 item id |
| `prototypes` | [K, d] | SMC 学到的 K 个原型 embedding |

---

## 各数据集 Checkpoint

| 数据集 | Checkpoint 目录 | wm_length | β 大小 |
|--------|----------------|-----------|--------|
| Industrial_and_Scientific | `Jun-29-2026_09-21-9e07b1_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05` | 10 | 26 |
| Baby_Products | `Jun-29-2026_14-38-b4b79e_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05` | 10 | 26 |
| Video_Games | `Jun-29-2026_14-37-785de4_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05` | 10 | 26 |
| ML-1M_TedRec | `Jun-29-2026_14-42-f80449_wm20_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.2_traceid_seqL2_trace_residual_debr_histneg0.05` | 20 | 36 |

---

## Case 1：DEBR 检索权重 β 可视化

### 图路径
- `figures/case1_retrieval.py` — 绘图脚本
- `figures/case1_retrieval.png / .pdf` — 输出图

### 数据来源
`figures/case_data.npz`（Industrial_and_Scientific）

### 图的内容
三个子图，每个对应一类用户，展示 26 维 β 权重分布：
- **User A（Semantic-Dominant）**：proto=0.98, wm=0.02，橙色大远高于蓝色，说明完全依赖语义原型
- **User B（Typical）**：proto=0.65, wm=0.35，蓝橙相对均衡
- **User C（Working-Memory-Anchored）**：proto=0.50, wm=0.50，最依赖 working memory 的用户

### 效果评价
✅ 能用。清晰说明不同用户的检索策略差异，User A 视觉冲击力强。

### 已知问题
- User C 命名为"Working-Memory-Anchored"但数据集中 wm 最大只到 0.50，不够典型
- 数据集里根本没有 wm 真正占主导（>0.6）的用户

---

## Case 2：WEBD 去噪效果

### 图路径
- `figures/case2_denoising.py` — 绘图脚本
- `figures/case2_denoising.png / .pdf` — 输出图

### 数据来源
`figures/case_assign.pkl`（Industrial_and_Scientific）中的 x_before / x_after

### 图的内容（当前版本）
两个用户，每人两行：
- **上行**：相邻时间步 embedding 差值 `||e_{t+1} - e_t||`（before/after WEBD）
- **下行**：每个时间步的 L2 norm（before/after WEBD）

### 效果评价
⚠️ 存在问题。WEBD 把几乎所有序列信息都去掉了（smoothing ratio ≈ 0.024），去噪后的绿线接近 0，看起来像"销毁信息"而非"选择性去噪"。

### 根本原因
阈值（~0.10）远大于 cD 能量（~0.001），导致所有高频细节系数被清零，序列从高波动压平到接近 0。

### 待解决
需要重新设计可视化角度，或者论证"压平到近似趋势"本身是 WEBD 的有效行为。详见下方讨论。

---

## Case 3：语义原型分析

### 图路径
- `figures/case3_prototypes.py` — 绘图脚本
- `figures/case3_prototypes.png / .pdf` — 输出图

### 数据来源
- t-SNE 数据：`figures/ml1m_proto_tsne_xy.npy`、`figures/ml1m_proto_dom.npy`
- checkpoint：ML-1M_TedRec（wm=20，K=16）

### 图的内容
- **左图（A）**：t-SNE 散点图，6040 个用户，每人一个 16 维原型权重向量，压缩到 2D，按主导原型着色
- **右图（B）**：用户×原型热图，按主导原型分组排列

### 制作逻辑
1. 取 β 向量后 16 维 → [N, 16] 矩阵
2. 每人 argmax → 主导原型（颜色）
3. t-SNE([N,16] → [N,2]) → 左图坐标
4. 按主导原型分组，取样 6 人/组 → 右图热图

### 各数据集原型使用分布（Entropy 越高越均匀）

| 数据集 | 主导原型 Top3 | Entropy |
|--------|-------------|---------|
| Industrial_and_Scientific | P12(6843), P13(2723) | 1.20 |
| Video_Games | P0(6468), P1(945), P10(853) | 1.29 |
| Baby_Products | P0(6101), P9(768), P1(697) | 1.48 |
| **ML-1M_TedRec** | **P5(2007), P7(1234), P13(870)** | **2.00** |

→ **ML-1M_TedRec 原型分布最分散，t-SNE 聚类视觉效果最好**，选用此数据集

### 效果评价
✅ 左图 t-SNE 聚类分散，可视化效果好，能说明"不同用户有不同的长期记忆检索偏好"。
⚠️ 不能直接证明"不同原型代表不同语义主题"（各原型的类目/类型分布几乎相同）。

### 各数据集原型语义分析结论
- **Industrial_and_Scientific**：P12=3D打印耗材，P13=实验室仪器，P3=工业工具，有一定区分，但仅 3 个原型有实际分配。
- **ML-1M_TedRec**：各原型均被 Drama/Comedy/Action 主导，无明显区分。
- **Video_Games**：仅 P9/P10 有数据，均为 Legacy 游戏，无区分。

---

## 辅助分析脚本

| 脚本 | 功能 |
|------|------|
| `figures/fig2_grad_conflict.py` | 梯度冲突直方图（Figure 2），已完成 |
| `figures/fig1a_heatmap.py` | 用户行为热图（已放弃） |
| `figures/fig1b_stats.py` | 高频行为统计图（已放弃，定义有歧义） |
| `figures/fig1_burst_noise.py` | 示意图版本（已放弃，不是真实数据） |

---

## 当前待解决问题

1. **Case 2 去噪图**：需要重新设计，找到能正面展示 WEBD 效果的角度
2. **Figure 1**：已全部放弃，还没有替代方案
3. **Case 1 命名**：User C 的"Working-Memory-Anchored"标签与数据不符，需修改

---

## 文件目录总览

```
figures/
├── case_collect.py              # Industrial 数据收集
├── case_collect_general.py      # 通用数据收集
├── case_data.npz                # Industrial 固定维度数据
├── case_assign.pkl              # Industrial 可变维度数据
├── case_data_Baby_Products.npz/.pkl
├── case_data_Video_Games.npz/.pkl
├── case_data_ML1M.npz/.pkl
├── ml1m_proto_tsne_xy.npy       # ML-1M t-SNE 坐标
├── ml1m_proto_dom.npy           # ML-1M 用户主导原型
├── case1_retrieval.py/.png/.pdf
├── case2_denoising.py/.png/.pdf
├── case3_prototypes.py/.png/.pdf
├── fig2_grad_conflict.py/.png/.pdf
└── grad_angles.npy              # 梯度冲突数据
```
