# CEBNet 实验目标与模型改进分析

创建日期：2026-06-25  
最近更新：2026-06-28

## 1. 当前需要解决的核心目标

当前实验有两个主要目标：

1. **在 CCFRec 论文的四个数据集上进一步提升 CEBNet 效果**
   - 目前 CEBNet 在 CCFRec 原论文四个数据集上整体已经比 CCFRec 好一点。
   - 但提升幅度不大，论文结果说服力还不够强。
   - 希望在不牺牲现有验证集指标的前提下，进一步拉开和 CCFRec 的差距。

2. **在 TedRec 论文的 ML-1M_TedRec 长序列数据集上冲击 SOTA**
   - TedRec 在其论文的 ML-1M 数据集上效果很强。
   - 当前 CEBNet 在 `/data0/yejinxuan/workspace/CEBNet/dataset/ML-1M_TedRec` 上效果明显低于 TedRec。
   - 该数据集是长序列场景，和 CEBNet 论文核心故事“长程记忆 / 情景缓冲 / 长序列建模”高度相关，因此必须重点解决。

## 2. 已观察到的实验现象

### 2.1 CCFRec 四个短序列数据集

从已有日志看，CEBNet 在 CCFRec 原论文对应数据集上能取得小幅领先，但优势有限。

以 Musical_Instruments 为例，已有较好日志结果包括：

```text
Val NDCG@10 ≈ 0.0361
Val Recall@10 ≈ 0.0698
```

这和 CCFRec 论文中的 CCFRec(PQ) 结果接近并略高，但提升不大。

### 2.2 ML-1M_TedRec 旧强基线

当前最重要的旧版参照日志是：

```text
/data0/yejinxuan/workspace/CEBNet/logs/ML-1M_TedRec/基准对标.log
```

关键配置：

```text
neg_num = 3000
batch_size = 1000
max_his_len = 50
wm_length = 20
n_layers_smc = 1
```

该日志最佳验证结果：

```text
Epoch 40
Val Recall@5  = 0.128642
Val NDCG@5    = 0.079290
Val Recall@10 = 0.208113
Val NDCG@10   = 0.104900
```

TedRec 论文中 ML-1M 结果：

```text
TedRec Recall@10 = 0.2623
TedRec NDCG@10   = 0.1445
```

因此即使按旧版 CEBNet 最强结果计算，ML-1M_TedRec 上仍有明显差距：

```text
CEBNet Val NDCG@10: 0.1049
TedRec NDCG@10:    0.1445
差距:               0.0396
```

这个差距不是小范围调参可以轻易弥补的，需要从模型机制上补足 ID 协同信号和完整顺序轨迹。

## 3. 数据特性分析

### 3.1 CCFRec 原论文数据集偏短序列

已检查的数据长度示例：

```text
Musical_Instruments:
train mean = 7.8
valid mean = 6.89
test  mean = 7.89

Video_Games:
train mean = 7.68
valid mean = 6.46
test  mean = 7.45
```

这些数据集的平均历史长度大约只有 7 左右。

CEBNet 的核心模块包括：

- WEBD：工作记忆去噪
- SMC：长期语义记忆巩固
- DEBR：情景缓冲检索与融合

这些模块本质上是为了建模较长历史而设计的。短序列数据集中，长程记忆优势无法充分发挥，因此 CEBNet 相对 CCFRec 的提升有限是合理现象。

### 3.2 ML-1M_TedRec 是真正的长序列数据集

ML-1M_TedRec 的长度统计：

```text
train mean = 42.88
train median = 50
valid mean = 44.35
valid median = 50
test  mean = 44.66
test  median = 50
```

这里的 `median=50` 表示一半以上样本的历史长度已经达到最大截断长度 50。

这说明 ML-1M_TedRec 与 Amazon 短序列数据完全不同，是一个真实长序列任务。

### 3.3 ML-1M_TedRec 文本信息很少

ML-1M_TedRec 的 item meta 示例：

```json
{
  "title": "Money Pit, The (1986)",
  "genres": "Comedy",
  "meta": "Money Pit, The (1986) Comedy"
}
```

相比 Amazon 数据集：

```text
title + brand + features + categories + description
```

ML-1M_TedRec 只有：

```text
title + genres
```

因此文本语义信息明显更弱。很多电影之间的推荐关系不是由标题或类型直接决定，而是由用户协同行为决定。

## 4. 模型问题判断

### 4.1 当前 CEBNet 过度依赖文本语义码

CEBNet 继承了 CCFRec 的前端表示方式：

```text
T5 text embedding -> VQ semantic code -> Q-Former fusion -> item representation
```

这个设计在 Amazon 数据集上比较有效，因为 Amazon 文本字段丰富。

但在 ML-1M_TedRec 上，文本只有 title 和 genres，语义信息不足，导致模型难以学到强协同行为信号。

因此当前模型需要一个强的、可训练的 ID 协同分支。

### 4.2 SMC 对长序列的顺序信息保留不足

当前 CEBNet 长序列处理逻辑：

```text
最近 wm_length 个 item -> WEBD -> anchor
更早的 long history -> SMC -> K 个 prototype memory
anchor 检索 memory -> DEBR fusion
```

在 ML-1M_TedRec 当前配置中：

```text
max_his_len = 50
wm_length = 20
```

也就是：

```text
最近 20 个 item 进入 WEBD
前面 30 个 item 进入 SMC
```

SMC 会把长期历史压缩成 K 个原型槽。这适合表示长期兴趣主题，但可能丢失长序列中的细粒度顺序演化。

TedRec 在 ML-1M 上强的原因之一，是它做 sequence-level text-ID fusion，也就是在整段序列级别融合 ID 与文本，并保留全局上下文。

### 4.3 小范围调参已经不是主要矛盾

已经尝试过多组：

- `wm_length`
- `n_prototypes`
- `dropout`
- `ortho_weight`
- `freq_weight`

效果变化不大，说明瓶颈不是这些超参数，而是模型机制本身：

1. ID 协同信号不足。
2. 长序列顺序分支不足。
3. 文本少时，语义码无法承担主要推荐信号。

## 5. TedRec 对 ML-1M 有优势的原因

TedRec 的核心思想是：

```text
sequence-level semantic fusion
```

具体包括：

1. 对文本 embedding 和 ID embedding 沿序列维度做 FFT。
2. 在频域中进行 text-ID mutual filtering。
3. 再通过 IFFT 回到时域。
4. 将融合后的序列表示交给行为编码器进行下一物品预测。

这意味着 TedRec 的 item 表示不是孤立生成的，而是在整段序列上下文中融合得到的。

而当前 CEBNet / CCFRec 的 item 表示更偏向：

```text
单个 item 的文本语义 + VQ code 表示
```

然后再进入序列模块。

在短序列 Amazon 数据集上，这种方式足够有效；但在 ML-1M_TedRec 长序列、文本少的场景下，TedRec 的 sequence-level text-ID fusion 更适配。

## 6. 最近实验结论更新

### 6.1 full CE / `neg_num=0` 不适合作为当前主线

之前曾判断 ML-1M_TedRec item 数较少，因此可以尝试 full softmax CE：

```bash
--neg_num=0
```

但实际日志显示，这条路在当前 CEBNet 训练设置下明显变差。

#### full CE，无 ID residual

日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/ML-1M_TedRec/全负样本采样(现在已经得出结论很有问题).log
```

可见最佳验证结果大约为：

```text
Best visible Val NDCG@10 ≈ 0.07197
```

明显低于旧强基线：

```text
Old baseline Val NDCG@10 = 0.10490
```

#### full CE + ID residual

日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/ML-1M_TedRec/门控和全负采样.log
```

可见最佳验证结果大约为：

```text
Best visible Val NDCG@10 ≈ 0.08957
```

虽然比纯 full CE 好，但仍明显低于旧强基线 `0.10490`。

#### 结论

`neg_num=0` 在当前 CEBNet 中不是好的主线。原因可能不是“full CE 理论上不行”，而是当前 CEBNet 的辅助任务、温度系数、语义码建模和负采样训练动态是一起调出来的；直接切成 full CE 会改变梯度分布，使模型早期学习和辅助损失平衡变差。

后续 ML-1M_TedRec 主线应保持旧强配置：

```bash
--neg_num=3000
--batch_size=1000
```

### 6.2 trainable ID residual gate 不再作为后续主线

已实现可选 ID residual gate：

```text
semantic_emb = 原 CCFRec/CEBNet 语义 item 表示
id_emb       = trainable item ID embedding
gate         = sigmoid(W[semantic_emb; id_emb])
item_emb     = gate * semantic_emb + (1 - gate) * id_emb
```

该模块通过 `--use_id_residual` 开启，默认关闭，因此不会影响旧实验复现。

相关旧配置日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/ML-1M_TedRec/Jun-26-2026_02-37-4c8420_wm20_K16_wavhaar_mlm0.6_cl0.4_drop0.2_dpcross0.2_idres.log
```

该日志后续可见最佳验证结果为：

```text
Epoch 15
Val Recall@5  = 0.128808
Val NDCG@5    = 0.079127
Val Recall@10 = 0.211921
Val NDCG@10   = 0.105762
```

这个结果只比旧强基线 `0.10490` 略高，提升很小。Video_Games 上的 ID residual 日志也没有稳定提升，甚至可能压低原数据集效果。因此后续不再把 `--use_id_residual` 作为主线实验；代码保留只是为了历史消融和复现，不建议继续投入主实验资源。

## 7. 当前已实现的新模型改进：Sequential Trace Encoder

### 7.1 设计动机

CEBNet 原有主路径：

```text
item sequence -> WEBD / SMC / DEBR -> z_ceb
```

优势是符合认知记忆故事：

- WEBD 处理近期工作记忆；
- SMC 将长期历史巩固成语义原型；
- DEBR 用当前 anchor 检索长期记忆。

但在 ML-1M_TedRec 上，历史长度大量达到 50，SMC 原型压缩可能削弱完整左到右顺序轨迹。

因此新增一个可选 full-sequence causal Transformer 分支：

```text
full item_emb sequence -> causal Transformer -> z_seq
```

最终融合：

```text
z_final = fuse(z_ceb, z_seq)
```

这不是为了照抄 SASRec/TedRec，而是作为 CEBNet 的“显式情景轨迹保留”模块：在工作记忆和长期语义巩固之外，保留完整行为轨迹。

### 7.2 已加入的参数

`/data0/yejinxuan/workspace/CEBNet/main.py` 已加入：

```bash
--use_seq_branch
--n_layers_seq
--seq_fusion_mode {gate,add,seq_only}
--seq_gate_bias_init
--seq_add_weight
```

保存名中会追加：

```text
_seqL{n_layers_seq}_{seq_fusion_mode}
```

### 7.3 已加入的模型结构

`/data0/yejinxuan/workspace/CEBNet/model.py` 已加入：

- `self.seq_encoder`：复用现有 `Transformer`。
- `self.seq_layer_norm` / `self.seq_dropout` / `self.seq_output_norm`。
- `self.seq_fusion_gate`：仅 `seq_fusion_mode=gate` 时启用。
- `self.seq_fusion_norm`。

新增逻辑：

```text
item_emb -> add position embedding -> causal attention mask -> Transformer -> gather last valid hidden state -> z_seq
```

融合方式：

```text
gate:     z = norm((1 - gate) * z_ceb + gate * z_seq)
add:      z = norm(z_ceb + seq_add_weight * z_seq)
seq_only: z = norm(z_seq)
```

默认推荐使用：

```bash
--seq_fusion_mode=gate
--seq_gate_bias_init=-2.0
```

`seq_gate_bias_init=-2.0` 的目的是让模型初始更接近原 CEBNet 主路径，避免新分支一开始压过 WEBD/SMC/DEBR。

### 7.4 验证结果

已完成语法检查：

```bash
/data0/yejinxuan/miniconda3/envs/CCF/bin/python -m py_compile main.py model.py
```

已完成单 batch 冒烟测试，覆盖：

- `--use_id_residual`
- `--use_seq_branch`
- `seq_fusion_mode=gate`
- `calculate_loss`
- backward
- `full_sort_predict`

输出示例：

```text
smoke_ok {'loss': 9.940805, 'rec_loss': 1.286523, 'cl_loss': 1.383841, 'mlm_loss': 12.906382, 'ortho_loss': 0.183824, 'freq_loss': 33.853462} scores_shape (4, 3417)
```

这说明当前实现没有基础 shape 错误，loss 有限，反向传播和 full-sort 评测路径都能跑通。

## 8. 新增实验脚本

### 8.1 主实验：旧强配置 + ID residual + sequence branch

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_idres_seq_same_as_old.sh
```

用途：当前最重要的新主线实验。

关键差异：

```bash
--neg_num=3000
--batch_size=1000
--use_id_residual
--use_seq_branch
--n_layers_seq=2
--seq_fusion_mode=gate
--seq_gate_bias_init=-2.0
```

### 8.2 消融一：只有 sequence branch，无 ID residual

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_seq_same_as_old.sh
```

用途：判断 full-sequence 顺序分支本身是否有效。

### 8.3 消融二：ID residual + sequence-only

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_idres_seqonly_same_as_old.sh
```

用途：判断新顺序分支是否能单独承担用户表示；如果该结果很强，说明 CEB 主路径可能是瓶颈；如果弱于 gate fusion，说明 CEB 记忆路径仍有必要。

### 8.4 消融三：ID residual + additive fusion

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_idres_seqadd_same_as_old.sh
```

用途：判断简单加和是否比 gate 更稳。

## 9. 推荐实验顺序

ML-1M_TedRec 后续优先按以下顺序比较：

1. 旧强基线：`基准对标.log`
   - `neg_num=3000`
   - `batch_size=1000`
   - 无 ID residual
   - 无 sequence branch
   - 最佳 `Val NDCG@10 = 0.104900`

2. 旧配置 + ID residual：
   - `run_ml1m_tedrec_idres_same_as_old.sh`
   - 目标：确认 trainable ID 协同信号的收益。

3. 旧配置 + ID residual + sequence branch：
   - `run_ml1m_tedrec_idres_seq_same_as_old.sh`
   - 当前最重要的新主线。

4. 消融：
   - `run_ml1m_tedrec_seq_same_as_old.sh`
   - `run_ml1m_tedrec_idres_seqonly_same_as_old.sh`
   - `run_ml1m_tedrec_idres_seqadd_same_as_old.sh`

优先观察：

```text
Val NDCG@10
Val Recall@10
Test NDCG@10（只作为调试观察，不用于调参选择）
```

最终论文报告仍应以 validation 选 checkpoint，再在 test 上一次性报告为准；每 epoch test 只用于排查问题。

## 10. 2026-06-26 最新实验判断

### 10.1 sequence branch 是当前有效方向

`run_ml1m_tedrec_seq_same_as_old.sh` 只开启 sequence branch，不开启 ID residual。

日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/ML-1M_TedRec/Jun-26-2026_07-02-b20af9_wm20_K16_wavhaar_mlm0.6_cl0.4_drop0.2_dpcross0.2_seqL2_gate.log
```

当前可见结果：

```text
Epoch 8
Val Recall@5  = 0.130795
Val NDCG@5    = 0.081808
Val Recall@10 = 0.221358
Val NDCG@10   = 0.110753
Test NDCG@10  = 0.109778
```

相比旧强基线 `Val NDCG@10 = 0.104900` 有明确提升，说明完整顺序轨迹确实补上了 CEBNet 原路径的一部分短板。

### 10.2 ID residual 不再作为主线

结合 ML-1M_TedRec 和 Video_Games 的 ID residual 日志，ID residual gate 的收益很弱且有副作用风险。后续实验不再推荐：

```bash
--use_id_residual
```

它只保留为历史消融开关。

## 11. 已实现：Sequence-level Semantic Calibration

### 11.1 设计动机

当前 sequence branch 已经提升，但它主要是在最终用户表示层补充顺序轨迹：

```text
item_emb -> CEB path -> z_ceb
item_emb -> seq branch -> z_seq
```

问题是 `item_emb` 本身仍然是单个 item 独立生成的，没有先根据整段行为上下文进行校准。

因此新增一个可选的 sequence-level semantic calibration 层，在进入 WEBD / SMC / DEBR 和 seq branch 之前先处理整段 item 序列：

```text
item_emb sequence
-> FFT semantic calibration
-> calibrated item_emb sequence
-> WEBD / SMC / DEBR + seq branch
```

### 11.2 实现方式

新增模块：

```text
SemanticCalibrationLayer
```

输入：

```text
item_emb: [B, L, d]
item_seq_len: [B]
```

处理流程：

```text
1. 按 item_seq_len mask padding 位置
2. 沿序列维度做 rFFT
3. 使用可学习 freq_gate 过滤频率分量
4. irFFT 回到时域
5. 用 residual_gate 和 calibration_weight 残差加回原 item_emb
6. LayerNorm，并保持 padding 位置不变
```

公式近似为：

```text
F       = FFT(item_emb)
F_cal   = sigmoid(freq_gate) * F
x_cal   = IFFT(F_cal)
g       = sigmoid(W item_emb)
item'   = LayerNorm(item_emb + calibration_weight * g * x_cal)
```

为了降低对短序列和原数据集的副作用，模块默认关闭，且 gate bias 默认初始化为负数：

```bash
--calibration_gate_bias_init=-2.0
```

### 11.3 新增参数

```bash
--use_semantic_calibration
--calibration_mode=fft
--calibration_weight=0.2
--calibration_gate_bias_init=-2.0
```

不加 `--use_semantic_calibration` 时，原模型完全不受影响。

### 11.4 新增实验脚本

主线实验：sequence branch + semantic calibration，不使用 ID residual：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_seq_calib_same_as_old.sh
```

消融实验：只开 semantic calibration，不开 sequence branch，不使用 ID residual：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_calib_same_as_old.sh
```

### 11.5 验证结果

已完成语法检查：

```bash
/data0/yejinxuan/miniconda3/envs/CCF/bin/python -m py_compile main.py model.py
```

已完成单 batch 冒烟测试，覆盖：

- `--use_seq_branch`
- `--use_semantic_calibration`
- `calculate_loss`
- backward
- `full_sort_predict`

输出：

```text
calib_smoke_ok {'loss': 8.924476, 'rec_loss': 0.448229, 'cl_loss': 1.383007, 'mlm_loss': 12.599269, 'ortho_loss': 0.137763, 'freq_loss': 34.970604} scores_shape (4, 3417)
```

说明当前实现没有基础 shape 错误，loss 有限，反向传播和 full-sort 评测路径可运行。

## 12. 当前结论

1. `neg_num=0` / full CE 已被当前日志否定，不再作为主线。
2. `--use_id_residual` 不再作为主线，代码仅保留为历史消融。
3. 当前有效方向是 `--use_seq_branch`，其 ML-1M_TedRec 已明显超过旧强基线。
4. 下一步主线是旧强训练配置 + sequence branch + semantic calibration。
5. semantic calibration 默认关闭，主要用于 ML-1M_TedRec 长序列稀疏语义场景，不强制用于 CCFRec 四个短序列数据集。

## 13. 2026-06-27 新增：Trace-guided Residual DEBR

### 13.1 为什么需要这个版本

最近 ML-1M_TedRec 日志显示，`seq_only` 是当前最强方向之一，但如果直接把最终表示写成：

```text
z_user = z_seq
```

论文主线会被削弱，因为 WEBD / SMC / DEBR 没有参与最终用户表示。另一方面，普通 `seq_gate` 让 `z_seq` 和 `z_ceb` 在最后一层平等融合，实验上不如 `seq_only` 稳。

因此新增 `trace_residual_debr`，把顺序轨迹从“外接推荐主干”改成“情景记忆检索 query”：

```text
item_emb
  ├─ Trace Encoder -> q_trace
  ├─ WEBD -> working_memory
  └─ SMC  -> semantic_memory

memory = concat(working_memory, semantic_memory)
r_mem  = DEBR.retrieve(query=q_trace, memory=memory)
gate   = sigmoid(W[q_trace; r_mem])
z_user = LayerNorm(q_trace + gate * r_mem)
```

这样最终表示仍然是“被认知记忆增强后的情景表征”，而不是简单的 `seq_only`。

### 13.2 实现方式

新增 `seq_fusion_mode` 取值：

```bash
--seq_fusion_mode=trace_residual_debr
```

新增参数：

```bash
--trace_memory_gate_bias_init=-2.0
```

该 bias 初始化为负数，使训练初期：

```text
gate 较小
z_user ≈ q_trace
```

这样尽量保留当前 `seq_only` 的排序能力，同时允许模型学习什么时候注入 WEBD / SMC 产生的 cognitive memory。

### 13.3 论文解释

推荐命名：

```text
Trace-guided Residual DEBR
```

论文表述可以是：

```text
Sequential trace is not used as an independent recommendation backbone. Instead, it forms an episodic query to retrieve complementary cognitive memory from WEBD/SMC through DEBR. The final representation is a memory-augmented episodic representation.
```

中文解释：顺序轨迹不是外接的序列推荐主干，而是情景 query；CEBNet 的认知记忆路径仍然通过 DEBR 检索参与最终用户表示。

### 13.4 验证结果

已完成语法检查：

```bash
/data0/yejinxuan/miniconda3/envs/CCF/bin/python -m py_compile main.py model.py
```

已完成单 batch 冒烟测试，覆盖：

- `--use_seq_branch`
- `--seq_fusion_mode=trace_residual_debr`
- `calculate_loss`
- backward
- `full_sort_predict`

输出：

```text
trace_residual_debr_smoke_ok {'loss': 12.962907, 'rec_loss': 4.65524, 'cl_loss': 0.69603, 'mlm_loss': 12.80699, 'ortho_loss': 0.151275, 'freq_loss': 32.993401} scores_shape (4, 3417)
```

说明当前实现没有基础 shape 错误，loss 有限，反向传播和 full-sort 评测路径可运行。

### 13.5 推荐实验

优先跑不带 ID residual 的版本：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_trace_residual_debr_same_as_old.sh
```

核心配置：

```bash
--use_seq_branch
--seq_fusion_mode=trace_residual_debr
--trace_memory_gate_bias_init=-2.0
--neg_num=3000
--batch_size=1000
```

对比顺序建议：

```text
old CEBNet baseline
seq_gate
seq_only
trace_residual_debr
TedRec baseline
```

如果 `trace_residual_debr` 接近或超过 `seq_only`，它就更适合作为论文主线；如果略低但明显高于 `seq_gate`，也可以作为“性能和创新性折中”的主线版本。

## 14. 2026-06-27 新增：Decoupled Trace-only ID

### 14.1 为什么不继续使用全局 `use_id_residual`

`use_id_residual` 会把 trainable ID embedding 混入所有 item 表示：

```text
item_emb = semantic item representation + ID signal
```

这虽然增强了 ML-1M_TedRec 的协同信号，但也会让 WEBD / SMC / DEBR 的 cognitive memory 不再是纯语义记忆，论文解释上容易变成“到处加 ID”。

因此新增更符合主线的 decoupled trace ID：

```text
item_sem   = QFormer(text + semantic code)
item_trace = item_sem + gate * trace_id_embedding

item_sem   -> WEBD + SMC -> cognitive memory
item_trace -> Trace Encoder -> q_trace

r_mem  = DEBR.retrieve(q_trace, cognitive memory)
z_user = LayerNorm(q_trace + gate * r_mem)
```

也就是说：ID 协同信号只用于构造 episodic trace query 和候选 item 打分，CEB memory 仍然保持语义认知记忆。

### 14.2 新增参数

```bash
--use_decoupled_trace_id
--trace_id_gate_bias_init=-2.0
```

保存名中会追加：

```text
_traceid
```

### 14.3 推荐实验脚本

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_decoupled_trace_id_same_as_old.sh
```

该实验不要同时开启旧的 `--use_id_residual`，否则 memory 路径仍会被 ID 污染。

## 15. 2026-06-28 新增：Trace-preserving Dual Loss 与 300 维实验

### 15.1 动机

`decoupled_trace_id + trace_residual_debr` 已经接近 `idres + seq_only`，但仍没有稳定超过。当前瓶颈更像是 top-rank 排序能力不足：memory 注入提高了论文主线合理性，但可能干扰 trace 本身的排序信号。

因此新增 trace-preserving auxiliary recommendation loss：

```text
L = L_rec(z_user) + lambda * L_rec(q_trace) + other auxiliary losses
```

其中：

- `z_user` 是最终 memory-augmented 表示；
- `q_trace` 是 trace encoder 输出；
- `L_rec(q_trace)` 用来显式保留短期轨迹的排序能力。

### 15.2 新增参数

```bash
--trace_aux_rec_weight
```

默认值为 `0.0`，不开启时旧实验完全不受影响。开启时推荐先用：

```bash
--trace_aux_rec_weight=0.2
```

保存名会追加：

```text
_traceaux{weight}
```

### 15.3 新增脚本

调参版 decoupled trace ID：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_traceid_tuned.sh
```

trace auxiliary loss 版：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_traceid_traceaux.sh
```

300 维 + trace auxiliary loss 版：

```bash
/data0/yejinxuan/workspace/CEBNet/run_ml1m_tedrec_traceid_emb300_traceaux.sh
```

### 15.4 验证结果

已完成：

```bash
/data0/yejinxuan/miniconda3/envs/CCF/bin/python -m py_compile main.py model.py
```

并完成 300 维小 batch smoke test，覆盖 `trace_aux_rec_weight`、backward 和 full-sort prediction：

```text
trace_aux_emb300_smoke_ok {'loss': 9.91126, 'rec_loss': 3.477477, 'trace_aux_rec_loss': 3.461904, 'cl_loss': 0.004536, 'mlm_loss': 17.889837, 'ortho_loss': 0.088118, 'freq_loss': 36.473137} scores_shape (2, 3417)
```

## 16. ML-1M_TedRec 近期实验结果汇总

本节只记录已经看到的验证集结果，主要对比 `Val NDCG@10`。TedRec 论文 SOTA 仍以 `/data0/yejinxuan/workspace/CCFRec/TedRec.pdf` 表格为准；本地 `Ted基准效果.log` 只是复现实验参考线。

### 16.1 参考线

| 实验 | 日志 | 关键配置 | 最佳 Val NDCG@10 | 说明 |
|---|---|---|---:|---|
| CEBNet 旧强基线 | `基准对标.log` | old config, `neg_num=3000`, no seq branch | 0.10490 | 原始 CEBNet ML-1M_TedRec 参考线 |
| 本地 TedRec 复现 | `Ted基准效果.log` | TedRec, hidden size 300, full eval | 0.13940 | 本地复现参考，不等同论文最终 SOTA |
| `idres + seq_only` | `Jun-26-2026_16-01-eead00...idres_seqL2_seq_only.log` | `--use_id_residual --use_seq_branch --seq_fusion_mode=seq_only` | 0.13156 | 早期强结果，性能高但论文主线弱 |
| `idres + seq_only` 调参 | `Jun-27-2026_02-59-62f6d9...mlm0.3_cl0.2...idres_seqL2_seq_only.log` | `tau=0.05`, `cl=0.2`, `mlm=0.3` | 0.13190 | 当前已知最高 CEBNet 变体之一，但仍是 `seq_only` |

### 16.2 Trace-guided Residual DEBR 系列

| 实验 | 日志 | 关键配置 | 最佳 Val NDCG@10 | 判断 |
|---|---|---|---:|---|
| no-ID trace residual | `Jun-27-2026_05-32-c9d1c8...seqL2_trace_residual_debr.log` | no ID, no calibration, `trace_memory_gate_bias=-2.0` | 0.12753 | 比旧 CEBNet 强，接近 no-ID seq branch；论文主线更合理，但未超过 `seq_only` |
| global ID trace residual | `Jun-27-2026_12-15-f8ab60...idres_seqL2_trace_residual_debr.log` | `--use_id_residual`, gate bias `-2.0` | 0.12583 | 加全局 ID 后没有提升，反而低于 no-ID trace residual |
| global ID trace residual, conservative gate | `Jun-27-2026_12-16-f5a37c...idres_seqL2_trace_residual_debr.log` | `--use_id_residual`, gate bias `-3.0` | 0.12783 | 比 `-2.0` 稍好，但仍低于 `idres + seq_only` |
| global ID + calibration 0.2 | `Jun-27-2026_12-21-a5ca6b...calibfft_w0.2.log` | `--use_id_residual --use_semantic_calibration`, `w=0.2` | 0.12745 | calibration 没有带来明显收益 |
| global ID + calibration 0.1 | `Jun-27-2026_12-29-587e9b...calibfft_w0.1.log` | `--use_id_residual --use_semantic_calibration`, `w=0.1` | 0.12846 | 比 `w=0.2` 略好，但仍不足以追上 `seq_only` |

结论：`trace_residual_debr` 能把 WEBD / SMC / DEBR 留在最终主线里，论文解释更合理；但 memory 注入本身没有提供足够强的排序增益，继续调 gate 或 calibration 的收益有限。

### 16.3 Decoupled Trace-only ID 系列

| 实验 | 日志 | 关键配置 | 最佳 Val NDCG@10 | 判断 |
|---|---|---|---:|---|
| decoupled trace ID | `Jun-27-2026_17-05-2b3c57...traceid_seqL2_trace_residual_debr.log` | `--use_decoupled_trace_id`, no global ID, no calibration | 0.13085 | 当前最好的“论文主线合理版”，接近 `idres + seq_only`，但未超过 |

结论：把 ID 协同信号限制在 trace query 和 item scoring 中，比全局 `use_id_residual` 更合理，也更接近高性能；但差距说明瓶颈仍在 top-rank 排序能力。

### 16.4 待跑的新实验

以下脚本已经实现并通过 smoke test，但结果还未记录：

| 脚本 | 目的 |
|---|---|
| `run_ml1m_tedrec_traceid_tuned.sh` | 在 decoupled trace ID 上复用 `idres+seq_only` 的较优超参：`tau=0.05`, `cl=0.2`, `mlm=0.3` |
| `run_ml1m_tedrec_traceid_traceaux.sh` | 加 `--trace_aux_rec_weight=0.2`，显式保留 trace 表示的排序能力 |
| `run_ml1m_tedrec_traceid_emb300_traceaux.sh` | 300 维表示 + trace auxiliary loss，尝试对齐 TedRec 300 维设置 |

跑完后需要继续把最佳验证集指标追加到本节，避免后续实验混淆。

## 17. 2026-06-28 创新点与论文主线选择评估

本节总结目前已经尝试过的模型改动，从“实验效果”和“论文创新性”两个角度评估哪些适合写成主线，哪些只适合做消融或工程增强。

### 17.1 为什么不建议把 TedRec 作为必须超过的主基线

ML-1M_TedRec 对 item ID 协同转移非常敏感，而 TedRec 本身就是直接的 ID 序列推荐模型：

```text
item ID embedding + position embedding -> Transformer -> next-item ranking
```

本地 TedRec 复现日志显示其验证集 `NDCG@10` 前期上升很快：

```text
epoch 0: 0.0929
epoch 1: 0.1171
epoch 3: 0.1293
epoch 7: 0.1394
```

相比之下，CEBNet 的主线要经过语义编码、WEBD 去噪、SMC 记忆压缩和 DEBR 检索。这个结构更符合论文创新点，但在 ML-1M 这种 `title + genres` 语义较弱、ID 共现较强的数据上，会天然弱于直接 ID 序列排序模型。

因此如果论文主要想强调 CEBNet 的认知记忆机制，可以不把 TedRec 作为核心必须超越的主基线，而是将 ML-1M_TedRec 作为长序列挑战实验或补充实验。主结果更适合围绕 CCFRec 语义推荐数据集展开。

### 17.2 各方案效果与创新性对比

| 方案 | 当前效果表现 | 创新性 | 是否适合论文主线 | 判断 |
|---|---|---|---|---|
| 原始 WEBD + SMC + DEBR | ML-1M_TedRec 约 `0.10490`，效果偏弱 | 很强 | 高 | 必须作为 CEBNet 核心主线 |
| global ID residual | 和 `seq_only` 结合可到约 `0.1319`，但在 trace residual 中不稳 | 弱 | 低 | 不建议主推，只保留消融 |
| Sequential Trace Encoder | 明显提升 ML-1M_TedRec | 中等偏弱 | 中低 | 可作为轨迹建模组件，但不能直接写成主创新 |
| `seq_only` | 当前最强变体之一，约 `0.1315~0.1319` | 弱 | 很低 | 只能作为强性能对照，不适合作为主方法 |
| `gate/add` seq fusion | 有提升但不如 `seq_only` 稳 | 中等偏弱 | 中低 | 普通晚期融合，论文说服力一般 |
| `trace_residual_debr` | 约 `0.1275~0.1308`，低于 `seq_only` 但明显强于旧 CEBNet | 较强 | 高 | 推荐作为主线结构增强 |
| `decoupled_trace_id` | 当前最好的“论文主线合理版”，约 `0.13085` | 中高 | 高 | 推荐和 trace-guided DEBR 绑定使用 |
| semantic calibration | 约 `0.1274~0.1285`，收益有限 | 中等 | 中低 | 当前不建议主推，可放消融或 future work |
| trace auxiliary loss | 正在跑，早期趋势略优于无辅助 loss | 中等 | 中高 | 适合作为 trace-guided DEBR 的训练约束 |
| 300 维表示 | 正在跑，主要用于容量对齐 | 无 | 低 | 只能作为容量/公平性实验，不是创新点 |

### 17.3 每个方案的论文定位

#### 原始 WEBD + SMC + DEBR

这是 CEBNet 最干净的核心创新：

```text
item semantic embedding
-> WEBD: denoised working memory
-> SMC: consolidated semantic memory
-> DEBR: episodic memory retrieval
-> recommendation
```

优点是论文故事完整，缺点是 ML-1M_TedRec 上仅靠语义记忆难以捕获强 ID 序列转移。

#### global ID residual

全局 ID residual 会把 ID embedding 混入所有 item 表示：

```text
semantic item representation + gated ID embedding
```

这个方式能补一部分 ML-1M 的协同信号，但会污染 WEBD / SMC 的语义记忆路径，容易被理解成简单 ID trick，不建议作为论文主线。

#### `seq_only`

`seq_only` 的性能强，但最终表示基本绕过了 CEBNet 的 cognitive memory：

```text
z_user = z_seq
```

如果主结果依赖这个方案，论文容易被质疑为“核心其实是普通序列 Transformer，CEBNet 记忆模块只是附属”。因此它适合作为性能上界或消融，不适合作为主方法。

#### `trace_residual_debr`

这是当前最适合论文主线的增强结构。它不是把 sequence branch 当作最终推荐主干，而是把顺序轨迹当作 episodic cue：

```text
q_trace = TraceEncoder(sequence)
memory  = concat(working_memory, semantic_memory)
r_mem   = DEBR.retrieve(q_trace, memory)
z_user  = LayerNorm(q_trace + gate * r_mem)
```

这样可以解释为：当前行为轨迹触发对短期工作记忆和长期语义记忆的检索，最终表示仍然是 memory-augmented episodic representation，而不是简单 seq branch。

#### `decoupled_trace_id`

这是比 global ID residual 更合理的 ID 使用方式：

```text
semantic item -> WEBD / SMC cognitive memory
semantic item + gated ID -> trace query / item scoring
```

它保留 semantic memory path 的干净性，同时允许 trace cue 保留 item identity。论文中可以解释为：语义记忆负责抽象认知，轨迹线索负责具体行为身份，两者解耦后再通过 DEBR 交互。

#### trace auxiliary loss

trace auxiliary loss 不是单独创新点，而是配合 trace-guided memory retrieval 的训练约束：

```text
L = L_rec(z_user) + lambda * L_rec(q_trace) + auxiliary losses
```

它的作用是防止 memory fusion 后损伤 trace query 的排序能力。如果最终实验有效，可以写成 trace-preserving objective。

### 17.4 推荐的最终论文主线

不建议把方法描述成：

```text
CEBNet + seq branch
CEBNet + ID residual
```

更推荐写成：

```text
CEBNet with Trace-guided Cognitive Memory Retrieval
```

或中文表述：

```text
轨迹引导的认知记忆检索增强 CEBNet
```

推荐主线组件：

```text
1. WEBD: 构建去噪短期工作记忆
2. SMC: 将长历史压缩为语义长期记忆
3. Trace Encoder: 提取当前行为轨迹线索
4. Decoupled Trace ID: 只在轨迹线索中保留 item identity，避免污染语义记忆
5. Trace-guided DEBR: 用轨迹线索从认知记忆中检索相关记忆
6. Trace-preserving objective: 保证轨迹线索保持推荐排序能力
```

最终方法可以概括为：

```text
用户推荐不是只靠最近序列，而是由当前行为轨迹触发对短期工作记忆和长期语义记忆的检索。
```

这个表述能弱化 sequence branch 的“独立推荐器”色彩，同时保留 CEBNet 的核心创新。

### 17.5 当前最推荐保留的实验配置

如果后续实验结果不明显反转，最推荐作为主线候选的是：

```bash
--use_decoupled_trace_id
--trace_id_gate_bias_init=-2.0
--use_seq_branch
--seq_fusion_mode=trace_residual_debr
--trace_memory_gate_bias_init=-2.0
--trace_aux_rec_weight=0.2
```

其中：

- `trace_residual_debr` 是结构主创新；
- `decoupled_trace_id` 是 ML-1M 这类 ID 协同强数据集上的合理增强；
- `trace_aux_rec_weight` 是训练约束，若完整结果有效再纳入主方法；
- `seq_only` 和 global ID residual 只作为消融或强性能对照，不建议写成主方法。

### 17.6 当前结论

从论文创新点角度，最佳选择不是继续强化 `seq_only`，而是把主线收敛到：

```text
Trace-guided Cognitive Memory Retrieval
```

即：sequence trace 不是主角，memory retrieval 才是主角；sequence trace 只是触发 CEBNet cognitive memory 的检索线索。

## 18. 2026-06-29 CCFRec 四数据集新结构 NDCG 问题诊断

### 18.1 现象

将 ML-1M_TedRec 上较好的 `traceid + trace_residual_debr` 配置迁移到 CCFRec 四个 Amazon 数据集后，四个日志都出现相似现象：

```text
前 1~3 个 epoch 验证 NDCG 达到峰值，之后 training loss / rec_loss 持续下降，但 Val NDCG 持续下降。
```

代表日志：

| 数据集 | 日志 | 最佳 Val NDCG@10 | 峰值 epoch |
|---|---|---:|---:|
| Video_Games | `Jun-28-2026_17-06-ab8ede...traceid_seqL2_trace_residual_debr.log` | 0.05149 | 3 |
| Musical_Instruments | `Jun-28-2026_17-06-5e4575...traceid_seqL2_trace_residual_debr.log` | 0.03598 | 2 |
| Industrial_and_Scientific | `Jun-28-2026_17-11-593f99...traceid_seqL2_trace_residual_debr.log` | 0.02989 | 1 |
| Baby_Products | `Jun-28-2026_17-07-a361e7...traceid_seqL2_trace_residual_debr.log` | 0.02355 | 1 |

这说明新结构不是“训练不动”，而是很快学到可召回信号，随后继续优化训练 loss 时损伤了验证集 top-rank 排序。

### 18.2 NDCG 低但 Recall 尚可的直接原因

当前评估是单正样本 full-sort：

```text
Recall@K: target 是否进入 topK
NDCG@K: target 进入 topK 后排得越靠前越高
```

因此当模型能把 target 放进 top10，但经常排在第 3~10 位时，Recall 看起来还可以，但 NDCG 会明显偏低。

Video_Games 新旧模型小样本诊断显示：

| 模型 | hit@1 | hit@5 | hit@10 | top5 中历史 item 数/用户 | top10 中历史 item 数/用户 |
|---|---:|---:|---:|---:|---:|
| 旧结构最好日志 | 0.0142 | 0.0610 | 0.1016 | 0.878 | 1.277 |
| 新 `traceid + trace_residual_debr` | 0.0068 | 0.0628 | 0.1038 | 1.409 | 1.742 |

关键结论：新结构 `hit@10` 略高，但 `hit@1` 几乎腰斩，并且 top5/top10 中历史 item 明显更多。也就是说，新结构更容易把用户历史中的 item 或相似 item 排在 target 前面，导致 Recall 不差但 NDCG 下降。

### 18.3 为什么新结构更容易出现这个问题

1. **Trace 分支天然偏向复现近期历史**：`decoupled_trace_id` 和 causal trace encoder 强化了具体 item identity，在 full-sort 时容易把历史 item / 近邻 item 打到前排。
2. **当前评估没有屏蔽历史 item**：`trainer.py` 中直接对 `full_sort_predict` 的所有 item 分数做 `topk`，没有将 `item_inters` 中出现过的历史 item 置为 `-inf`。这会让 trace-heavy 模型的 NDCG 更吃亏。
3. **Amazon 四数据集平均历史较短**：大部分样本历史长度只有 6~8 左右，长期 memory 可发挥空间有限；新结构容量更大，容易在前几轮后过拟合训练负采样。
4. **ML-1M 调参不能直接迁移到 Amazon**：`tau=0.05, cl_weight=0.2, mlm_weight=0.3` 会强化排序尖锐度并削弱语义正则，对稀疏 Amazon 数据集容易让 ID/trace 信号过强。
5. **训练负采样与 full-sort 评估不完全一致**：训练只和随机负样本比较，后期 rec_loss 下降不代表 target 能在全库相似 item / 历史 item 前排第一。

### 18.4 当前判断

新结构在 CCFRec 四数据集上的问题不是 Recall 能力完全不足，而是 top-rank 精排能力下降：

```text
target 进 top10 的能力还可以；
target 被排到 top1/top3 的能力下降。
```

因此如果目标是提高 NDCG，应优先处理 top-rank 排序和历史 item 挤占问题，而不是继续单纯增加 trace 分支强度。

后续建议优先验证：

```text
1. 加历史 item mask 的评估对照，确认 NDCG 是否被历史 item 挤占显著压低。
2. 对 Amazon 数据集恢复旧语义正则：tau=0.07, cl_weight=0.4/0.5, mlm_weight=0.6 或原数据集最佳值。
3. 减弱 trace/ID：trace_id_gate_bias_init=-3.0 或 -4.0，trace_memory_gate_bias_init=-3.0。
4. 降低序列分支容量：n_layers_seq=1，或先跑 `traceid` 不加 `trace_residual_debr` 的消融。
5. 如果保留新结构，尝试 `trace_aux_rec_weight=0.2`，专门约束 trace query 的 top-rank 排序能力。
```

### 18.5 已实现的修复开关

为解决上述 top-rank NDCG 问题，新增两个默认关闭的可选机制：

```bash
--mask_history_in_eval
--history_neg_weight
--history_neg_num
```

含义：

- `--mask_history_in_eval`：评估时将输入历史中出现过的 item 分数置为 `-inf`，用于诊断/验证历史 item 是否挤占 topK。
- `--history_neg_weight`：训练时对历史 item 的高分进行惩罚，目标是减少 trace 分支把历史 item 排到 target 前面的倾向。
- `--history_neg_num`：每个用户最多取最近多少个历史 item 参与惩罚，默认 `20`。

历史负样本惩罚形式：

```text
history_neg_loss = mean(softplus(score(history_item) - score(pos_item).detach()))
```

总损失变为：

```text
L = L_rec + λ_hist * L_history_neg + λ_trace * L_trace_aux + other auxiliary losses
```

推荐先用：

```bash
--history_neg_weight=0.1
--history_neg_num=20
```

如果这个机制有效，预期现象不是 Recall@10 大幅上涨，而是：

```text
hit@1 / NDCG@5 / NDCG@10 上升；
top5/top10 中历史 item 数下降；
Recall@10 变化相对较小。
```

## 19. 2026-06-29 Industrial_and_Scientific history negative 调参记录

### 19.1 参考原配置

参考日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/Industrial_and_Scientific/Jun-28-2026_17-11-593f99_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr.log
```

关键配置：

```text
tau=0.05
lr=0.001
cl_weight=0.2
mlm_weight=0.3
wm_length=10
n_layers_webd=2
n_layers_smc=2
use_decoupled_trace_id=True
seq_fusion_mode=trace_residual_debr
history_neg_weight=0
```

原配置结果：

```text
Best Val NDCG@10 = 0.02989
Test NDCG@10     = 0.02477
```

### 19.2 当前最佳 history negative 配置

当前最佳日志：

```text
/data0/yejinxuan/workspace/CEBNet/logs/Industrial_and_Scientific/Jun-29-2026_09-11-9a548c_wm10_K16_wavhaar_mlm0.3_cl0.2_drop0.2_dpcross0.1_traceid_seqL2_trace_residual_debr_histneg0.05.log
```

关键变化：

```text
history_neg_weight=0.05
history_neg_num=10
tau=0.05
lr=0.001
```

最佳验证结果：

```text
Epoch 2
Val Recall@5  = 0.03868
Val NDCG@5    = 0.02440
Val Recall@10 = 0.05998
Val NDCG@10   = 0.03124
```

测试结果：

```text
Test Recall@5  = 0.03211
Test NDCG@5    = 0.02032
Test Recall@10 = 0.04996
Test NDCG@10   = 0.02604
```

相对原配置：

```text
Val NDCG@10:  0.02989 -> 0.03124
Test NDCG@10: 0.02477 -> 0.02604
```

### 19.3 调参结论

`history_neg_weight=0.05, history_neg_num=10` 比 `history_neg_weight=0.1, history_neg_num=20` 更稳，能够提升 NDCG，同时基本保住 Recall。说明 Industrial 上不适合强行压制太多历史 item；温和压制最近历史 item 更合理。

已测试对照：

| 日志 | 配置 | 最佳 Val NDCG@10 | 结论 |
|---|---|---:|---|
| `9a548c` | `tau=0.05, lr=0.001, histneg=0.05, num=10` | 0.03124 | 当前最佳 |
| `aa75f8` | `tau=0.05, lr=0.001, histneg=0.1, num=5` | 0.03075 | 稍弱 |
| `8c3bae` | `tau=0.07, lr=0.001, histneg=0.05, num=10` | 0.03088 | tau 变大没有明显收益 |
| `9e07b1` | `tau=0.05, lr=0.0005, histneg=0.05, num=10` | 0.03083 | 降 lr 没明显收益 |

后续在 Industrial_and_Scientific 上优先使用：

```bash
--tau=0.05 \
--lr=0.001 \
--history_neg_weight=0.05 \
--history_neg_num=10
```
