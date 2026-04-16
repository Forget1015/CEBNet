# CEB-Net 技术报告与设计方案

## 一、CCFRec 代码仓库审查

对 `/data0/yejinxuan/workspace/CCFRec` 全部代码逐行审查，包括模型、数据处理、训练、VQ 生成等完整流程。

---

### Bug 1（中等）：`get_item_embedding()` 中循环变量 `i` 被内层循环覆盖

**位置**: `model.py` → `CCFRec.get_item_embedding()`

```python
for i in range(n_batches):          # ← 外层循环变量 i
    ...
    text_embs = []
    for i in range(self.text_num):   # ← 内层循环也用 i，覆盖了外层！
        text_emb = self.item_text_embedding[i](batch_item)
        text_embs.append(text_emb)
```

**问题**: 内层 `for i in range(self.text_num)` 覆盖了外层的 batch 索引 `i`。当 `text_num=5`（run.sh 配置）时，外层循环第一次迭代后 `i` 被设为 4，第二次迭代 `start = 4*1024 = 4096`，直接跳过了 index 1~4095 的 item embedding 计算。

**CEB-Net 修复**: 内层循环改用 `j`。

---

### Bug 2（轻微）：`encode_emb.py` 中引用了不存在的变量

**位置**: `encode_emb.py`

```python
parser.add_argument('--text_types', ...)
attr_list = args.meta_types  # ← 应该是 args.text_types
```

加上 npy 保存路径缺少 `/` 分隔符：
```python
np.save(dataset_path + f'{dataset}.t5.{attr}.emb.npy', embs)
# 实际: ./dataset/Musical_InstrumentsMusical_Instruments.t5.title.emb.npy
```

**影响**: 文本嵌入编码脚本无法运行。但预处理数据已经存在（.npy 文件都在），不影响当前使用。CEB-Net 中复用时修复。

---

### Bug 3（轻微）：`args.FN` 未定义

**位置**: `model.py` → `self.false_neg = args.FN`

main.py 的 argparse 中没有 `--FN` 参数。该变量赋值后从未被使用。CEB-Net 中直接移除。

---

### 设计注意点 1：`forward()` 与 `get_item_embedding()` 的 item embedding 计算不一致

- `forward()`: `item_emb = qformer_output.mean(dim=1)` — 不加 query_emb 残差
- `get_item_embedding()` / `encode_item()`: `item_emb = qformer_output.mean(dim=1) + query_emb.mean(dim=1)` — 加了残差

训练时序列中的 item 表示和推理时全库 item 表示的计算方式不同。

**CEB-Net 决策**: 统一加上 query_emb 残差。理由：
1. 消除训练/推理的表示空间不对齐问题
2. query_code_embedding 编码了 VQ 码本的协作语义结构信息，残差连接保留了这条直连通路
3. 统一后模型不需要额外学习弥补这个 gap

---

### 设计注意点 2：负采样效率低

`encode_item()` 中每次调用都执行 `np.random.choice(list(all_items), ...)`，将 set 转 list 再采样。对于 24587 个 item、neg_num=24000 的配置，每次 forward 都要做一次大规模采样。

**CEB-Net 优化**: 预先将 item 集合转为 numpy array 缓存，采样时直接用 `np.random.choice(self.all_item_array, ...)`，避免重复的 set→list 转换。

---

### 数据处理流程审查

CCFRec 的完整数据处理链路：

```
原始数据 (csv.gz + meta.jsonl.gz)
    │
    ▼  preprocess.py
jsonl (交互序列) + item.json + emb_map.json + meta.json
    │
    ▼  encode_emb.py (有Bug但已有预处理结果)
.t5.{attr}.emb.npy (sentence-t5 文本嵌入, 每种属性一个文件)
    │
    ▼  vq/generate_faiss_multi_emb.py
.code.pq.{n_codebooks}_{code_num}.pca{pca_size}.{text_types}.json (VQ语义码)
    │
    ▼  main.py 中加载
.npy → PCA(128维, whiten) → model.item_text_embedding (冻结)
.code.json → model.index (VQ码索引)
```

**逐环节审查结果：**

| 环节 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 1. 数据预处理 | `preprocess.py` | ✅ 无问题 | csv.gz → jsonl，正确处理了 history 截断、item2id 映射、meta 文本清洗 |
| 2. 文本嵌入编码 | `encode_emb.py` | ⚠️ 有Bug | 变量名错误 + 路径缺分隔符，但预处理结果已存在，不影响使用 |
| 3. VQ码本生成 | `vq/generate_faiss_multi_emb.py` | ✅ 无问题 | FAISS PQ 量化，5种文本嵌入 concat 后 PCA→PQ，生成 20 层 ×256 码的语义码 |
| 4. 数据加载 | `data.py` | ✅ 无问题 | CCFSeqSplitDataset 正确处理了序列截断、VQ码映射、掩码生成、padding |
| 5. Collator | `data.py` | ✅ 无问题 | pad_sequence 正确处理了变长序列，code_inters reshape 正确 |
| 6. 文本嵌入加载 | `main.py` | ✅ 无问题 | PCA 降维到 embedding_size=128，whiten=True，加载到冻结的 Embedding 层 |

**数据格式确认（Musical_Instruments）：**
- 训练集: 339,519 条
- 验证集: 57,439 条
- 测试集: 57,439 条
- Item 数量: 24,587
- VQ 码: 20 层 × 256 码，值域 [0, 255]
- 文本嵌入: 5 种属性 (title, brand, features, categories, description)
- jsonl 中的 `session_ids` 字段是之前的脏数据，CCFRec 不使用，CEB-Net 也忽略

**CEB-Net 数据复用方案：**
- 直接软链接 CCFRec 的 `dataset/` 目录
- `data.py` 从 CCFRec 复用，只读取 `inter_history` 和 `target_id`，忽略 `session_ids`
- VQ 码、文本嵌入、item 映射全部复用，无需重新预处理

---

## 二、CEB-Net 方案可行性评估

### 2.1 复用 CCFRec 编码器的可行性：✅ 完全可行

CCFRec 的核心编码器：
1. **VQ Code Embedding** (`query_code_embedding`): 语义码 → 嵌入向量
2. **Q-Former** (`CrossAttTransformer`): 交叉注意力融合 VQ 查询和多视角文本嵌入

编码器输出是 target-free 的 item embedding，直接作为 CEB-Net 的输入 $\mathbf{X} \in \mathbb{R}^{n \times d}$。CEB-Net 用 WEBD + SMC + DEBR 替换 CCFRec 原版的因果 Transformer 序列建模器。

### 2.2 各模块效果预估

| 模块 | 预期效果 | 风险点 |
|------|---------|--------|
| WEBD（小波去噪） | +0.5~1.5% NDCG | 工作记忆区太短时频率分辨率有限 |
| SMC（原型巩固） | +1~2% NDCG | 短序列时长期历史区太短 |
| DEBR（解耦检索） | +0.5~1% NDCG | 额外投影层可能引入信息瓶颈 |
| 联合训练 | 整体 +2~4% NDCG | 损失权重需要调参 |

### 2.3 WEBD 和 SMC 编码器：保持独立

理由：
1. 因果注意力 vs 双向注意力，掩码模式不同，共享会互相干扰
2. 论文叙事中"复述编码"和"回放编码"是不同认知机制
3. 各用 1 层 Transformer，参数量仅约 260K，相比 Q-Former 微不足道

---

## 三、完整设计方案

### 3.1 架构总览

**关键设计决策：保留全序列因果 Transformer 作为性能保底**

CEB-Net 将 CCFRec 的全序列因果 Transformer 替换为分段处理（WEBD + SMC + DEBR）。
但如果直接移除全序列建模，工作记忆区（5个item）和长期历史区之间的信息交互
只通过 DEBR 的 anchor-memory 匹配来实现，可能丢失全序列上下文建模能力。

**保底策略**: 在分段处理之前，对全序列先做 1 层因果 Transformer 编码（对应认知
心理学中的"感知编码"阶段）。这确保：
1. 即使 WEBD/SMC/DEBR 全部消融，模型至少有全序列 Transformer 的建模能力
2. WEBD 和 SMC 接收到的 item embedding 已融入全序列上下文，去噪和原型分配质量更高
3. 消融实验中可以清晰地展示每个模块的增量贡献

```
输入: item_seq [B, L], item_seq_len [B]
  │
  ▼
┌─────────────────────────────────┐
│  CCFRec VQ Encoder (复用)       │
│  query_code_embedding           │
│  + item_text_embedding (冻结)   │
│  → Q-Former (CrossAttTransformer)│
│  输出: item_emb [B, L, d]      │  ← 统一使用 qformer.mean + query.mean
│        code_emb [B*L, C, d]    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  全序列感知编码 (Perceptual Enc)│
│  item_emb + PE → LN → Dropout  │
│  → CausalTransformer(1层)       │  ← 性能保底，保留全序列上下文
│  输出: ctx_emb [B, L, d]       │
└─────────────┬───────────────────┘
              │
     ┌────────┴────────┐
     │  向量化序列分割   │
     ├────────┬────────┤
     ▼                 ▼
┌──────────┐    ┌───────────┐
│ 工作记忆  │    │ 长期历史   │
│ [B, m, d]│    │ [B, n-m, d]│
└────┬─────┘    └─────┬─────┘
     │                │
     ▼                ▼
┌──────────┐    ┌───────────┐
│   WEBD   │    │    SMC    │
│ 因果TF   │    │ 双向TF    │
│ +DWT去噪 │    │ +原型巩固  │
│ +注意力   │    │           │
│  池化     │    │           │
└────┬─────┘    └─────┬─────┘
     │                │
     ▼                ▼
  anchor           memory
  [B, d]          [B, K, d]
     │                │
     └──────┬─────────┘
            ▼
     ┌────────────┐
     │    DEBR    │
     │ 解耦检索   │
     │ +门控融合  │
     └─────┬──────┘
           ▼
         z_u [B, d]
           │
           ▼
  ┌──────────────────┐
  │ L_rec (InfoNCE)  │
  │ + L_cl (对比)    │
  │ + L_mlm (掩码)   │
  │ + L_ortho (正交) │
  │ + L_freq (频域)  │
  └──────────────────┘
```

### 3.2 模块详细设计

#### 模块 0: VQ Encoder（从 CCFRec 复用）

编码流程：
```
item_seq [B, L] → flatten [B*L]
  → query_code_embedding(code_seq)     → [B*L, C, d]
  → item_text_embedding[j](items)      → stack → [B*L, text_num, d]
  → qformer(query, text)[-1]           → [B*L, C, d]
  → .mean(dim=1) + query_emb.mean(dim=1) → [B*L, d]  ← 统一加残差
  → reshape → [B, L, d]
```

**关键修复**: `forward()` 中也加上 `query_emb.mean(dim=1)` 残差，与 `get_item_embedding()` / `encode_item()` 保持一致。

#### 模块 1: WEBD — 小波增强的突发性保留去噪

输入: `x_wm [B, m, d]` + `wm_mask [B, m]`

1. **Rehearsal Encoding**: `x_wm + PE → LN → Dropout → CausalTransformer(1层) → x_rehearsed`
2. 保存 `x_before_dwt = x_rehearsed.clone()` （频域一致性损失用）
3. **DWT**: `x_rehearsed → conv1d(lo/hi filter, stride=2) → cA, cD`
   - 滤波器系数注册为 buffer，只创建一次
4. **软阈值**: `context = masked_mean(x_rehearsed) → threshold_net → τ`
   - `cD_clean = sign(cD) * max(|cD| - τ, 0)`
5. **IDWT**: `(cA, cD_clean) → upsample + conv1d → x_denoised`
6. **注意力池化**: `x_denoised → attn_pool → anchor [B, d]`

输出: `anchor`, `x_before_dwt`, `x_denoised`

#### 模块 2: SMC — 语义记忆巩固

输入: `x_long [B, n-m, d]` + `long_mask [B, n-m]`

1. **Replay Encoding**: `x_long + PE → LN → Dropout → BiTransformer(1层) → x_replayed`
2. **原型分配**: `sim = proj(x_replayed) @ proj(P).T / sqrt(d) → softmax → assign [B, n-m, K]`
   - 显式清零 padding: `assign = assign * long_mask.unsqueeze(-1)`
3. **加权聚合**: `memory = assign.T @ x_replayed / assign.T.sum() → [B, K, d]`

输出: `memory [B, K, d]`

辅助: `compute_ortho_loss()` — 最大余弦相似度惩罚

#### 模块 3: DEBR — 解耦情景缓冲器

输入: `anchor [B, d]`, `memory [B, K, d]`

1. **检索子空间**: `q = W_a(anchor), k = W_a(memory) → w = softmax(k@q/sqrt(d))`
2. **表示子空间**: `v = W_r(memory) → z_long = w @ v`
3. **门控融合**: `g = sigmoid(W_g[anchor; z_long]) → z_u = g*anchor + (1-g)*z_long`

输出: `z_u [B, d]`

### 3.3 损失函数

```
L = L_rec + γ_cl * L_cl + γ_mlm * L_mlm + γ_ortho * L_ortho + γ_freq * L_freq
```

| 损失 | 说明 | 来源 |
|------|------|------|
| L_rec | InfoNCE(z_u, item_emb) | CCFRec 复用 |
| L_cl | InfoNCE(z_u, z_u_masked) 序列对齐对比 | CCFRec 复用 |
| L_mlm | CE(code_output_masked, code_labels) 掩码编码建模 | CCFRec 复用 |
| L_ortho | (1/K) Σ_k max_{j≠k} cos(p_k, p_j) 原型正交 | CEB-Net 新增 |
| L_freq | MSE(\|FFT(x_before_dwt)\|, \|FFT(x_denoised)\|) 频域一致性 | CEB-Net 新增 |

### 3.4 超参数配置

| 参数 | 短序列 (his≤20) | 长序列 (his=50) |
|------|-----------------|-----------------|
| max_his_len | 20 | 50 |
| wm_length | 5 | 10 |
| n_prototypes | 16 | 32 |
| wavelet | haar | haar |
| n_layers_webd | 1 | 1 |
| n_layers_smc | 1 | 1 |
| ortho_weight | 0.1 | 0.1 |
| freq_weight | 0.01 | 0.01 |
| cl_weight | 0.4 | 0.3 |
| mlm_weight | 0.6 | 0.4 |
| embedding_size | 128 | 128 |
| hidden_size | 512 | 512 |
| n_heads | 2 | 2 |
| n_layers_cross | 2 | 2 |
| dropout_prob | 0.3 | 0.2 |
| dropout_prob_cross | 0.3 | 0.1 |
| lr | 1e-3 | 5e-4 |
| batch_size | 400 | 256 |
| neg_num | 24000 | 24000 |
| tau | 0.07 | 0.07 |
| proto_temperature | 1.0 | 1.0 |
| code_level | 20 | 20 |
| n_codes_per_lel | 256 | 256 |

### 3.5 项目文件结构

```
CEBNet/
├── CEB.md                  # 原始设计方案文档
├── TECHNICAL_REPORT.md     # 本技术报告
├── requirements.txt
├── model.py                # CEB-Net 模型 (VQ Encoder + WEBD + SMC + DEBR)
├── layers.py               # 基础层 (从CCFRec复用)
├── data.py                 # 数据加载 (从CCFRec复用，忽略session_ids)
├── trainer.py              # 训练器 (基于CCFRec适配)
├── main.py                 # 主入口
├── metrics.py              # 评估指标 (从CCFRec复用)
├── utils.py                # 工具函数 (从CCFRec复用)
├── preprocess.py           # 数据预处理 (从CCFRec复用)
├── encode_emb.py           # 文本嵌入编码 (从CCFRec复用，修复Bug)
├── run.sh                  # 训练脚本
├── run_preprocess.sh       # 预处理脚本 (从CCFRec复用)
├── dataset/                # → 软链接到 CCFRec/dataset/
└── vq/                     # VQ码本生成 (从CCFRec复用)
```

### 3.6 复用关系与修改清单

| 文件 | 策略 | 修改 |
|------|------|------|
| layers.py | 完全复用 | 无 |
| utils.py | 完全复用 | 无 |
| metrics.py | 完全复用 | 无 |
| preprocess.py | 完全复用 | 无 |
| encode_emb.py | 复用+修复 | meta_types→text_types, 路径加 os.path.join |
| vq/ | 完全复用 | 无 |
| data.py | 复用 | 与CCFRec格式兼容，忽略session_ids |
| model.py | 全新 | VQ Encoder移植(修Bug1+统一残差), 新增WEBD/SMC/DEBR |
| trainer.py | 基于CCFRec改写 | 适配多任务损失 |
| main.py | 基于CCFRec改写 | 增加CEB-Net参数 |

---

## 四、实施计划

1. **Phase 1**: 项目骨架 — 复制/链接 CCFRec 基础设施
2. **Phase 2**: model.py — VQ Encoder + WEBD + SMC + DEBR + 损失
3. **Phase 3**: trainer.py + main.py — 训练循环、评估、参数
4. **Phase 4**: run.sh — 各数据集训练配置
