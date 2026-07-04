# CEBNet 论文核心问题与方法大纲

## 0. 建议论文主线

建议将论文主线从“长序列建模”或“单纯多模态语义增强”调整为：

```text
Context-Dependent Cognitive Memory Retrieval for Semantic Sequential Recommendation
面向语义序列推荐的上下文相关认知记忆检索
```

核心问题不是“序列不够长”，也不是“简单加入文本语义”，而是：

```text
现有序列推荐方法通常将短期意图、长期语义偏好、噪声行为和历史重复信号压缩到一个统一用户表示中，缺少一种能够根据当前行为上下文动态检索不同类型用户记忆的机制。
```

一句话版本：

> Existing sequential recommenders often compress heterogeneous user memories into a monolithic sequence representation, making it difficult to distinguish transient intent, stable semantic preference, and noisy/redundant historical behaviors for next-item ranking.

中文版本：

> 现有序列推荐方法通常将异质用户记忆压缩为单一序列表示，难以区分瞬时意图、长期语义偏好以及噪声/重复历史行为，从而限制了下一物品预测中的上下文相关记忆利用与 top-rank 排序能力。

---

## 1. 真实存在的研究问题

### 1.1 问题名称建议

建议在论文中命名为：

```text
Heterogeneous Memory Entanglement
异质记忆纠缠问题
```

或更保守地写成：

```text
context-agnostic memory entanglement in sequential recommendation
序列推荐中的上下文无关记忆纠缠
```

不要过度宣称这是一个全新理论概念，而是作为对现有问题的归纳：

```text
short-term intent / long-term preference / noisy interactions / semantic memory are entangled in a single representation.
```

### 1.2 为什么这是现有论文真实存在的问题

近年序列推荐和多模态序列推荐文献中，反复出现以下几个真实痛点：

#### A. 长短期兴趣难以平衡

大量工作专门建模 long-term / short-term preferences，说明单一序列表示不足以稳定捕获两类兴趣。例如 TLSTSRec、LSIDN、GLAD、DualCFGL 等都从不同角度处理长短期偏好、动态兴趣或 memory balance。

这说明：

```text
用户下一步行为不是单一兴趣向量能充分表达的，而是由短期意图和长期偏好共同决定。
```

#### B. 行为序列中存在噪声和兴趣漂移

Denoising long-/short-term interests、explicit intent denoising、robust sequential recommendation 等工作都说明用户历史并非干净信号。近期行为中既有真实突发意图，也有误触、探索、偶然浏览。

这说明：

```text
直接把近期交互编码为当前兴趣，容易把噪声也放大进最终排序。
```

#### C. 语义/多模态信息利用方式仍然粗糙

多模态序列推荐近年大量工作关注 modality imbalance、semantic alignment、feature concatenation 的不足、semantic representation degradation 等问题。现有模型常把文本/图像/语义 embedding 与 ID embedding 拼接或对齐，但没有形成可检索、可巩固的长期语义记忆。

这说明：

```text
item semantic information 不应只是输入特征，而应该被组织成长期语义记忆。
```

#### D. 检索和表示耦合会带来表示退化/坍塌风险

Semantic ID、generative recommendation、LLM-based sequential recommendation 方向中，已有工作讨论 embedding collapse、codebook collapse、semantic ID conflict、semantic degradation 等问题。虽然它们的技术对象不完全相同，但共同说明：当同一表示同时承担“相似性检索”和“预测区分”任务时，容易出现表达能力退化。

这为 CEBNet 的 DEBR 提供合理切入：

```text
retrieval space and representation space should be decoupled.
```

### 1.3 推荐最终问题定义

论文可以这样定义问题：

> Sequential recommendation requires identifying which parts of a user's past should be activated under the current behavioral context. However, existing methods often encode recent behaviors, long-term semantic preferences, and noisy/redundant historical interactions into a single representation. Such context-agnostic memory entanglement makes it difficult to selectively retrieve stable semantic memories while preserving transient intent, and may further blur the boundary between retrieval-oriented similarity and prediction-oriented representation.

中文：

> 序列推荐的关键并非简单建模 item 转移，而是在当前行为上下文下判断用户历史中的哪些记忆应该被激活。然而，现有方法通常将近期行为、长期语义偏好以及噪声/重复历史交互编码到单一表示中，导致上下文无关的记忆纠缠。这使模型难以在保留瞬时意图的同时选择性检索稳定语义记忆，并可能进一步混淆面向相似性检索的表示与面向预测区分的表示。

---

## 2. 论文不要主打什么

### 2.1 不建议主打“长序列”

原因：

- Amazon 四数据集平均历史并不长；
- ML-1M 虽然可以构造较长历史，但不是所有实验都依赖长序列；
- 如果主打 long sequence，审稿人会要求与更多长序列模型、效率模型比较。

更稳的说法是：

```text
CEBNet targets cognitive memory organization under sequential behaviors, which naturally supports both short and moderately long histories.
```

### 2.2 不建议主打“多模态 SOTA”

原因：

- 当前主要用文本语义和 VQ semantic codes；
- Taobao-MM 等强多模态数据集如果没有完整图像/视频处理，主打 multimodal 容易被质疑；
- 论文可以说 semantic-aware sequential recommendation，而不是 full multimodal recommendation。

### 2.3 不建议主打“history mask 后 SOTA”

`mask_history_in_eval` 是评估口径，不是模型贡献。除非所有 baseline 都用同样口径重跑，否则不能作为主结果。

---

## 3. 完整方法论框架：CEBNet 如何解决异质记忆纠缠

### 3.0 整体思想

CEBNet 的核心不是把用户历史编码成一个更强的统一向量，而是显式模拟推荐场景中的“记忆形成—记忆巩固—上下文检索—预测校准”过程：

```text
raw item semantics
  -> semantic item encoding
  -> working memory / long-term memory construction
  -> denoised working memory (WEBD)
  -> consolidated semantic memory (SMC)
  -> trajectory-conditioned retrieval query
  -> decoupled episodic buffer retrieval (DEBR)
  -> residual cognitive fusion
  -> history-aware next-item prediction
```

这条路径对应前文提出的核心问题：现有序列推荐把短期意图、长期偏好、语义信息、噪声行为和历史重复信号混入一个单一表示。CEBNet 则把这些异质信号拆成不同认知记忆，并根据当前行为上下文动态检索。

方法可以概括为：

```text
Cognitive Memory Construction
+ Trajectory-Conditioned Decoupled Retrieval
+ History-Aware Top-Rank Calibration
```

---

### 3.1 问题定义与符号

给定用户集合 $\mathcal{U}$ 和物品集合 $\mathcal{I}$。对于用户 $u$，其历史交互序列为：

$$
\mathcal{S}_u = [i_1, i_2, \ldots, i_t], \quad i_j \in \mathcal{I}.
$$

序列推荐目标是在给定历史 $\mathcal{S}_u$ 的情况下预测下一步交互物品 $i_{t+1}$。模型需要学习一个用户表示 $\mathbf{z}_u$，并对候选物品 $i$ 打分：

$$
s(u, i) = \mathbf{z}_u^\top \mathbf{e}_i.
$$

不同于普通序列模型直接学习：

$$
\mathbf{z}_u = f(i_1, \ldots, i_t),
$$

CEBNet 将用户历史拆解成三类信息：

```text
1. 近期工作记忆：反映短期即时意图，但包含噪声；
2. 长期语义记忆：反映稳定偏好，但冗余且分散；
3. 当前轨迹上下文：决定当前推荐场景下应该激活哪部分记忆。
```

因此，CEBNet 不是直接学习一个 monolithic user embedding，而是学习：

$$
\mathbf{z}_u = \text{Fuse}(\mathbf{q}_{trace}, \text{Retrieve}(\mathbf{q}_{trace}, \mathcal{M}_{cog})),
$$

其中 $\mathbf{q}_{trace}$ 是轨迹条件化检索查询，$\mathcal{M}_{cog}$ 是认知记忆库。

---

### 3.2 模块一：语义物品编码（Semantic Item Encoding）

#### 动机

现有 ID-based 序列模型擅长学习协同转移，但难以表达物品的语义相似性和跨物品知识迁移。语义序列推荐需要让每个物品不仅有 ID 属性，还具备文本/语义 code 表示。

#### 输入

对于每个物品 $i$，CEBNet 使用两类语义信息：

```text
1. text embeddings：title / brand / features / categories / description 等文本字段；
2. semantic codes：由 VQ / PQ 量化得到的离散语义 code 序列。
```

记物品 $i$ 的 semantic code 为：

$$
\mathbf{c}_i = [c_i^1, c_i^2, \ldots, c_i^L],
$$

对应 code embedding：

$$
\mathbf{Q}_i = \text{Emb}_{code}(\mathbf{c}_i) \in \mathbb{R}^{L \times d}.
$$

物品多字段文本 embedding 表示为：

$$
\mathbf{T}_i = [\mathbf{t}_i^1, \mathbf{t}_i^2, \ldots, \mathbf{t}_i^M] \in \mathbb{R}^{M \times d}.
$$

#### Q-Former 语义融合

CEBNet 使用 cross-attention Q-Former 让 semantic code query 从文本语义中抽取信息：

$$
\mathbf{H}_i = \text{QFormer}(\mathbf{Q}_i, \mathbf{T}_i).
$$

最终物品语义表示为：

$$
\mathbf{e}_i^{sem} = \text{Mean}(\mathbf{H}_i) + \text{Mean}(\mathbf{Q}_i).
$$

这里的 residual code term 保留离散语义 code 的结构信息，避免纯文本 embedding 过于平滑。

#### 可选 ID 信号

为了避免全局 ID embedding 污染语义记忆，主线不使用全局 `use_id_residual`。当前较优配置使用 decoupled trace ID：

$$
\mathbf{e}_i^{trace} = \text{LN}(\mathbf{e}_i^{sem} + g_i^{id} \odot \mathbf{e}_i^{id}),
$$

其中：

$$
g_i^{id} = \sigma(\mathbf{W}_{id}[\mathbf{e}_i^{sem}; \mathbf{e}_i^{id}]).
$$

解释：

```text
semantic item embedding 主要服务 WEBD/SMC 认知记忆构造；
trace ID embedding 主要增强轨迹查询和候选 item 区分能力。
```

这比把 ID 加到所有路径更安全，因为它避免让长期语义记忆退化成纯 ID 记忆。

---

### 3.3 模块二：认知记忆划分（Cognitive Memory Partition）

给定用户历史的语义表示：

$$
\mathbf{X}_u = [\mathbf{e}_{i_1}^{sem}, \ldots, \mathbf{e}_{i_t}^{sem}] \in \mathbb{R}^{t \times d},
$$

CEBNet 将其划分为：

```text
working memory segment：最近 m 个交互；
long-term memory segment：更早的历史交互。
```

即：

$$
\mathbf{X}_{wm} = [\mathbf{e}_{i_{t-m+1}}^{sem}, \ldots, \mathbf{e}_{i_t}^{sem}],
$$

$$
\mathbf{X}_{long} = [\mathbf{e}_{i_1}^{sem}, \ldots, \mathbf{e}_{i_{t-m}}^{sem}].
$$

如果用户历史长度不足，模型会使用有效 mask 避免 padding 影响记忆构造。

#### 为什么要划分？

这一步直接对应论文核心问题：

```text
短期意图和长期偏好不是同一种记忆。
```

近期行为更接近用户当前意图，但噪声更强；长期历史更稳定，但冗余且可能与当前意图冲突。因此 CEBNet 先划分记忆，再分别处理。

---

### 3.4 模块三：WEBD 去噪工作记忆编码

#### 3.4.1 动机

工作记忆包含用户最近的即时意图，但近期行为中同时存在：

```text
真实突发兴趣；
误触 / 随机浏览；
探索性点击；
短期重复行为。
```

传统低通滤波或简单序列 pooling 容易把所有高频变化视为噪声，导致真实 bursty intent 被抹掉。WEBD 的目标是：

```text
保留高幅值突发意图，抑制低幅值随机波动。
```

#### 3.4.2 Rehearsal Encoder

首先对工作记忆注入位置编码：

$$
\mathbf{E}_{wm} = \text{LN}(\mathbf{X}_{wm} + \mathbf{P}_{wm}).
$$

然后使用因果 Transformer 进行复述编码：

$$
\widetilde{\mathbf{X}}_{wm} = \text{CausalTransformer}(\mathbf{E}_{wm}).
$$

这里使用 causal attention，是因为工作记忆仍然服务于下一物品预测，不能让靠后位置反向泄露给靠前位置。

#### 3.4.3 Wavelet Denoising

对复述后的工作记忆序列做离散小波变换：

$$
\mathbf{A}, \mathbf{D} = \text{DWT}(\widetilde{\mathbf{X}}_{wm}),
$$

其中：

```text
A：低频近似系数，表示平稳趋势；
D：高频细节系数，表示局部波动。
```

CEBNet 学习一个动态阈值：

$$
\boldsymbol{\tau} = \alpha \cdot \sigma(\text{MLP}(\widetilde{\mathbf{X}}_{wm})),
$$

并对高频细节系数做 soft-thresholding：

$$
\mathbf{D}' = \text{sign}(\mathbf{D}) \cdot \max(|\mathbf{D}| - \boldsymbol{\tau}, 0).
$$

随后通过逆小波变换重构去噪工作记忆：

$$
\mathbf{W} = \text{IDWT}(\mathbf{A}, \mathbf{D}').
$$

#### 3.4.4 Working Memory Anchor

CEBNet 使用注意力池化从去噪工作记忆中提取即时意图 anchor：

$$
a_j = \text{softmax}(\mathbf{w}_a^\top \mathbf{W}_j),
$$

$$
\mathbf{z}_{wm} = \sum_j a_j \mathbf{W}_j.
$$

其中 $\mathbf{z}_{wm}$ 是去噪工作记忆 anchor。

#### 3.4.5 解决的问题

WEBD 对应解决：

```text
近期行为中真实突发意图和随机噪声纠缠的问题。
```

论文里可以强调：

> WEBD does not simply suppress high-frequency behavior signals. Instead, it preserves bursty high-amplitude intent while filtering out small random fluctuations.

---

### 3.5 模块四：SMC 语义记忆巩固

#### 3.5.1 动机

长期历史通常包含大量重复、稀疏且分散的行为。如果直接把长期历史作为序列输入，模型容易面临：

```text
计算冗余；
长期偏好分散；
语义主题不稳定；
历史噪声累积。
```

SMC 的目标是将长期历史巩固成有限个稳定的 semantic memory slots。

#### 3.5.2 Memory Replay Encoder

对长期历史注入位置编码：

$$
\mathbf{E}_{long} = \text{LN}(\mathbf{X}_{long} + \mathbf{P}_{long}).
$$

然后使用 replay Transformer 得到上下文化长期历史：

$$
\widetilde{\mathbf{X}}_{long} = \text{Transformer}(\mathbf{E}_{long}).
$$

这里可以解释为 memory replay：长期历史在巩固前需要先建立内部上下文关系。

#### 3.5.3 Prototype-based Consolidation

维护 $K$ 个可学习语义原型：

$$
\mathbf{P} = [\mathbf{p}_1, \ldots, \mathbf{p}_K] \in \mathbb{R}^{K \times d}.
$$

每个历史行为对原型的分配权重为：

$$
\alpha_{j,k} = \frac{\exp(\widetilde{\mathbf{x}}_j^\top \mathbf{p}_k / \tau_p)}{\sum_{l=1}^{K}\exp(\widetilde{\mathbf{x}}_j^\top \mathbf{p}_l / \tau_p)}.
$$

第 $k$ 个语义记忆槽为：

$$
\mathbf{m}_k = \frac{\sum_j \alpha_{j,k}\widetilde{\mathbf{x}}_j}{\sum_j \alpha_{j,k} + \epsilon}.
$$

最终长期语义记忆为：

$$
\mathbf{M}_{sem} = [\mathbf{m}_1, \ldots, \mathbf{m}_K] \in \mathbb{R}^{K \times d}.
$$

#### 3.5.4 解决的问题

SMC 对应解决：

```text
长期历史偏好冗余分散，缺少结构化语义存储的问题。
```

论文里可以强调：

> SMC turns raw historical interactions into compact and retrievable semantic memory slots, rather than forcing the model to repeatedly attend over the entire history.

---

### 3.6 模块五：轨迹条件化检索查询（Trajectory-Conditioned Retrieval Query）

#### 3.6.1 动机

工作记忆和语义记忆构造后，还需要回答一个问题：

```text
在当前推荐场景下，应该激活哪些记忆？
```

如果只用工作记忆 anchor 检索长期记忆，查询可能过于局部；如果只用长期偏好，模型又可能忽视当前意图。因此当前最优版本使用完整行为轨迹生成一个轨迹条件化检索查询。

注意论文中不要说：

```text
完整轨迹天然就是心理学情景线索。
```

更严谨的说法是：

```text
模型通过 next-item prediction 学习一个 trajectory-conditioned query。
```

#### 3.6.2 Decoupled Trace ID Input

为了增强协同转移信号，CEBNet 在 trace branch 中使用 decoupled ID：

$$
\mathbf{e}_i^{trace} = \text{LN}(\mathbf{e}_i^{sem} + g_i^{id}\odot \mathbf{e}_i^{id}).
$$

得到 trace 输入序列：

$$
\mathbf{X}_{trace} = [\mathbf{e}_{i_1}^{trace}, \ldots, \mathbf{e}_{i_t}^{trace}].
$$

#### 3.6.3 Causal Trace Encoder

加入位置编码后，通过因果序列编码器：

$$
\mathbf{H}_{trace} = \text{CausalTransformer}(\mathbf{X}_{trace} + \mathbf{P}_{trace}).
$$

取最后一个有效位置作为检索查询：

$$
\mathbf{q}_{trace} = \mathbf{H}_{trace}[t].
$$

#### 3.6.4 解决的问题

该模块解决：

```text
记忆检索需要依赖当前行为上下文，而不是静态地聚合所有历史记忆。
```

论文表述：

> The trajectory-conditioned query is optimized by the next-item prediction objective, and therefore acts as a dynamic selector that determines which cognitive memories are relevant under the current behavioral context.

---

### 3.7 模块六：解耦情景缓冲器检索（DEBR）

#### 3.7.1 认知记忆库构造

当前 `trace_residual_debr` 最优版本将去噪工作记忆和长期语义记忆共同作为认知记忆库：

$$
\mathbf{M}_{cog} = [\mathbf{W}; \mathbf{M}_{sem}].
$$

其中：

```text
W：WEBD 输出的去噪工作记忆；
M_sem：SMC 输出的长期语义记忆槽。
```

这样做的含义是：

```text
trajectory query 不直接替代 memory，而是从短期工作记忆和长期语义记忆中选择当前上下文相关的信息。
```

#### 3.7.2 为什么要解耦检索空间和表示空间

如果使用同一个 embedding 空间同时完成：

```text
1. similarity matching：哪个 memory 应该被检索；
2. predictive representation：检索出的 memory 如何用于预测；
```

模型会面临目标冲突。相似度检索偏向粗粒度聚类，下一物品预测需要细粒度区分。这就是前文的表示坍塌/表示退化风险。

#### 3.7.3 Retrieval Subspace

DEBR 首先将查询和记忆投影到检索子空间 $\mathcal{A}$：

$$
\mathbf{q}^{\mathcal{A}} = \mathbf{W}_{q}\mathbf{q}_{trace},
$$

$$
\mathbf{K}^{\mathcal{A}} = \mathbf{W}_{k}\mathbf{M}_{cog}.
$$

检索权重为：

$$
\boldsymbol{\beta} = \text{softmax}\left(\frac{\mathbf{K}^{\mathcal{A}}(\mathbf{q}^{\mathcal{A}})^\top}{\sqrt{d_a}}\right).
$$

#### 3.7.4 Representation Subspace

同时，将认知记忆映射到表示子空间 $\mathcal{R}$：

$$
\mathbf{V}^{\mathcal{R}} = \mathbf{W}_{v}\mathbf{M}_{cog}.
$$

检索出的认知记忆内容为：

$$
\mathbf{r}_{mem} = \boldsymbol{\beta}\mathbf{V}^{\mathcal{R}}.
$$

#### 3.7.5 解决的问题

DEBR 对应解决：

```text
检索匹配目标和预测表达目标混在同一空间导致的表示退化。
```

论文强调：

```text
Trace decides what to retrieve;
DEBR decides how to retrieve without collapsing retrieval and representation.
```

---

### 3.8 模块七：残差认知融合（Residual Cognitive Fusion）

检索到认知记忆后，CEBNet 使用门控残差融合：

$$
\mathbf{g}_{mem} = \sigma(\mathbf{W}_g[\mathbf{q}_{trace}; \mathbf{r}_{mem}]),
$$

$$
\mathbf{z}_u = \text{LN}(\mathbf{q}_{trace} + \mathbf{g}_{mem} \odot \mathbf{r}_{mem}).
$$

这里的设计含义是：

```text
q_trace：保留当前序列动态和协同转移信号；
r_mem：提供从认知记忆中检索出的语义/工作记忆补充；
g_mem：控制记忆注入强度。
```

为什么不用简单 concat 或 add？

```text
简单 concat 会增加预测层复杂度，不适合 full-sort；
简单 add 无法控制 memory 是否可靠；
残差门控既保留 trace 主干，又允许 memory 动态补充。
```

这也解释当前模型相比 `seq_only` 更有论文意义：最终表示不是纯序列向量，而是序列轨迹和认知记忆检索的结合。

---

### 3.9 预测层与全库排序

对最终用户表示做归一化：

$$
\hat{\mathbf{z}}_u = \frac{\mathbf{z}_u}{\|\mathbf{z}_u\|_2}.
$$

候选物品 embedding 为：

$$
\hat{\mathbf{e}}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|_2}.
$$

最终打分：

$$
s(u,i) = \frac{\hat{\mathbf{z}}_u^\top \hat{\mathbf{e}}_i}{\tau}.
$$

训练时使用 sampled softmax / sampled CE，评估时使用 full-sort ranking。

---

### 3.10 优化目标：论文中建议写成三类 loss

当前代码中会记录多个 loss，但论文主文不要写成 7 个独立目标。建议组织为三类：

$$
\mathcal{L} = \mathcal{L}_{rec} + \lambda_{sem}\mathcal{L}_{sem} + \lambda_{mem}\mathcal{L}_{mem}.
$$

#### 3.10.1 Recommendation Objective

主推荐损失：

$$
\mathcal{L}_{CE} = -\log \frac{\exp(s(u,i^+))}{\exp(s(u,i^+)) + \sum_{i^-\in\mathcal{N}}\exp(s(u,i^-))}.
$$

如果最终模型使用 history negative，则将其合并到 recommendation objective 中，而不是单独作为第四类 loss：

$$
\mathcal{L}_{HN} = \frac{1}{|\mathcal{H}_u|}\sum_{h\in\mathcal{H}_u}\text{softplus}(s(u,h) - s(u,i^+)_{detach}).
$$

因此：

$$
\mathcal{L}_{rec} = \mathcal{L}_{CE} + \lambda_{hn}\mathcal{L}_{HN}.
$$

解释：

```text
L_HN 不是为了刷指标单独增加的损失，而是为了让 sampled training 更贴近 full-sort top-rank ranking，避免近期历史重复项持续占据 target 前排。
```

#### 3.10.2 Semantic Consistency Objective

语义一致性目标合并 contrastive loss 和 masked semantic-code prediction：

$$
\mathcal{L}_{sem} = \mathcal{L}_{CL} + \lambda_{mlm}\mathcal{L}_{MLM}.
$$

其中：

```text
L_CL：增强用户表示与 masked view 的一致性；
L_MLM：通过 masked code prediction 保持 semantic code 表示能力。
```

论文中不用过度展开，可放附录。

#### 3.10.3 Memory Regularization Objective

记忆结构正则合并 SMC 原型多样性和 WEBD 频域一致性：

$$
\mathcal{L}_{mem} = \lambda_{ortho}\mathcal{L}_{ortho} + \lambda_{freq}\mathcal{L}_{freq}.
$$

原型多样性：

$$
\mathcal{L}_{ortho} = \frac{1}{K}\sum_k \max_{j\neq k}\cos(\mathbf{p}_k, \mathbf{p}_j).
$$

频域一致性：

$$
\mathcal{L}_{freq} = \left\||\mathcal{F}(\mathbf{X}_{before})| - |\mathcal{F}(\mathbf{X}_{after})|\right\|_2^2.
$$

解释：

```text
L_ortho 防止长期语义 memory slots 坍塌到同一主题；
L_freq 防止 WEBD 过度去噪破坏真实短期意图。
```

---

### 3.11 方法如何逐一对应核心问题

| 核心问题 | CEBNet 对应模块 | 解决方式 |
|---|---|---|
| 短期意图和噪声行为纠缠 | WEBD | 小波软阈值去噪，保留高幅值突发意图 |
| 长期偏好冗余且语义分散 | SMC | 巩固为有限 semantic memory slots |
| 记忆检索缺少当前上下文 | trajectory-conditioned query | 用当前轨迹生成动态检索查询 |
| 检索匹配和预测表示目标冲突 | DEBR | 检索子空间和表示子空间解耦 |
| 历史重复项挤占 top-rank | history-aware calibration | 轻量校准近期历史 item 分数 |

最终可以形成一个非常清楚的论文叙事：

```text
现有模型的问题：异质用户记忆被单一表示纠缠。
CEBNet 的解决：构造不同认知记忆，并在当前轨迹条件下解耦检索。
```

---

## 4. 推荐最终贡献点

建议只写 3 个贡献，避免显得模块过多。

### Contribution 1: Cognitive memory formulation

> We formulate semantic sequential recommendation as a context-dependent cognitive memory retrieval problem, where transient working memory and stable semantic memory should be separately constructed and dynamically activated.

中文：

> 我们将语义序列推荐建模为上下文相关的认知记忆检索问题，强调短期工作记忆与长期语义记忆应被分别构造并动态激活。

### Contribution 2: CEBNet architecture

> We propose CEBNet, which constructs denoised working memory via wavelet-enhanced burst-preserving denoising, consolidates long-term behaviors into semantic memory slots, and retrieves cognitive memories using a trajectory-conditioned query.

中文：

> 我们提出 CEBNet，通过小波增强的突发性保留去噪构造工作记忆，通过语义记忆巩固形成长期记忆槽，并利用轨迹条件化查询进行认知记忆检索。

### Contribution 3: Decoupled retrieval and ranking calibration

> We design a decoupled episodic buffer that separates retrieval matching from memory representation, and introduce a lightweight history-aware calibration objective to improve top-rank next-item prediction.

中文：

> 我们设计了解耦情景缓冲器，将检索匹配与记忆表达分离，并引入轻量级历史感知校准目标以提升 top-rank 下一物品预测。

---

## 5. 推荐论文标题

### 稳妥版本

```text
CEBNet: Context-Dependent Cognitive Memory Retrieval for Semantic Sequential Recommendation
```

### 更突出机制版本

```text
Decoupled Cognitive Memory Retrieval for Semantic Sequential Recommendation
```

### 更贴近当前最优结构版本

```text
Trace-Conditioned Cognitive Memory Retrieval for Sequential Recommendation
```

最推荐第一个：

```text
CEBNet: Context-Dependent Cognitive Memory Retrieval for Semantic Sequential Recommendation
```

原因：

- 不把论文限制为 long sequence；
- 不夸大 multimodal；
- 能包含 WEBD、SMC、DEBR、trace retrieval；
- 更像顶会推荐论文标题。

---

## 6. Introduction 写作骨架

### Paragraph 1: 序列推荐背景

序列推荐目标是根据用户历史交互预测下一步行为。Transformer/attention/LLM/Semantic ID 等方法提升了序列建模能力，但多数方法仍倾向于将用户历史压缩为单一表示。

### Paragraph 2: 现有问题

用户行为历史包含多种异质信号：

```text
transient intent
stable semantic preference
noisy exploratory behaviors
repeated historical interactions
```

现有统一序列表示难以区分这些信号，导致 memory entanglement。

### Paragraph 3: 为什么简单语义增强不够

文本/多模态/semantic ID 使 item representation 更丰富，但如果语义信息只是被拼接或作为 token 输入 Transformer，并没有解决：

```text
如何组织长期语义偏好；
如何根据当前上下文选择性检索；
如何避免检索空间和预测空间冲突。
```

### Paragraph 4: 我们的核心想法

CEBNet 受认知记忆机制启发，将用户历史组织为：

```text
denoised working memory
consolidated semantic memory
trajectory-conditioned retrieval query
```

然后通过 DEBR 进行解耦检索与残差融合。

### Paragraph 5: 贡献总结

列出三点 contribution。

---

## 7. Method 章节结构

```text
3. Method
  3.1 Problem Definition
  3.2 Semantic Item Encoding
  3.3 Denoised Working Memory Construction (WEBD)
  3.4 Semantic Memory Consolidation (SMC)
  3.5 Trajectory-Conditioned Decoupled Memory Retrieval (DEBR)
  3.6 Prediction and Optimization
```

### 3.6 中 loss 不要写太多

主文只写三类：

```text
L = L_rec + λ_sem L_sem + λ_mem L_mem
```

其中：

```text
L_rec = L_CE + λ_hn L_HN       # 如果最终主模型使用 history_neg
L_sem = L_CL + λ_mlm L_MLM
L_mem = L_ortho + λ_freq L_freq
```

这样看起来是三类目标，而不是七个 loss 堆料。

---

## 8. Experiment 章节设计

### 8.1 Main results

数据集：

```text
Musical_Instruments
Video_Games
Baby_Products
Industrial_and_Scientific
ML-1M_TedRec
```

主表报告：

```text
Recall@5 / NDCG@5 / Recall@10 / NDCG@10
```

### 8.2 Ablation studies

建议消融：

| Variant | 目的 |
|---|---|
| w/o WEBD | 验证去噪工作记忆 |
| w/o SMC | 验证语义记忆巩固 |
| w/o DEBR / use single-space retrieval | 验证解耦检索 |
| w/o trace-conditioned query | 验证轨迹条件化检索 |
| w/o history-aware calibration | 验证 top-rank calibration |

### 8.3 Analysis

建议分析：

1. 不同数据集上 history-aware calibration 对 NDCG@10 的影响；
2. early epoch 后 loss 下降但 NDCG 下降的现象，解释为 random-negative training 与 full-sort top-rank ranking 的 gap；
3. memory retrieval attention 的可视化或统计：不同用户检索 working/semantic memory 的权重不同；
4. SMC prototype diversity：验证 semantic memory slots 没有坍塌。

---

## 9. 和现有工作的关系

### 9.1 相比普通 Sequential Recommendation

普通 SR 关注：

```text
how to encode item transitions
```

CEBNet 关注：

```text
how to organize and retrieve heterogeneous user memories under the current behavioral context
```

### 9.2 相比 Long/Short-term Interest Models

长短期模型通常只是融合两个 preference vectors。CEBNet 的区别是：

```text
long-term preference is consolidated into semantic memory slots;
short-term preference is denoised as working memory;
retrieval is conditioned by current trajectory and decoupled from representation.
```

### 9.3 相比 Multimodal/Semantic SR

多模态/语义 SR 主要解决 item representation rich enough 的问题。CEBNet 进一步解决：

```text
how semantic representations are stored, denoised, consolidated, and retrieved as user memories.
```

### 9.4 相比 Retrieval-Augmented Recommendation

Retrieval-augmented methods通常检索外部历史、相似用户或相关 item。CEBNet 的区别是：

```text
retrieval is performed over internal cognitive memory constructed from the user's own history, with decoupled retrieval and representation spaces.
```

---

## 10. 推荐摘要草稿

英文草稿：

> Sequential recommendation requires identifying which parts of a user's past are relevant under the current behavioral context. However, existing methods often compress heterogeneous behavioral signals, including transient intent, stable semantic preference, and noisy/redundant historical interactions, into a monolithic sequence representation. This context-agnostic memory entanglement limits the model's ability to selectively retrieve semantic memories for top-rank next-item prediction. In this paper, we propose CEBNet, a context-dependent cognitive memory retrieval framework for semantic sequential recommendation. CEBNet constructs a denoised working memory to preserve bursty short-term intent, consolidates long-term behaviors into compact semantic memory slots, and learns a trajectory-conditioned query to retrieve context-relevant cognitive memories. To avoid coupling retrieval matching with predictive representation, we further design a decoupled episodic buffer that separates retrieval and representation subspaces. Extensive experiments on multiple benchmark datasets demonstrate that CEBNet consistently improves sequential recommendation performance, especially on NDCG-oriented top-rank ranking.

中文草稿：

> 序列推荐需要在当前行为上下文下判断用户历史中的哪些信息与下一步决策相关。然而，现有方法通常将瞬时意图、长期语义偏好以及噪声/重复历史行为压缩为单一序列表示，导致上下文无关的记忆纠缠，限制了模型对相关语义记忆的选择性检索能力。本文提出 CEBNet，一种面向语义序列推荐的上下文相关认知记忆检索框架。CEBNet 构造去噪工作记忆以保留突发短期意图，将长期行为巩固为紧凑的语义记忆槽，并学习轨迹条件化查询来检索当前上下文相关的认知记忆。进一步地，我们设计了解耦情景缓冲器，将检索匹配空间与预测表示空间分离，缓解单一空间中的目标冲突。多个数据集上的实验表明，CEBNet 能够稳定提升序列推荐性能，尤其改善 NDCG 导向的 top-rank 排序。

---

## 11. 文献依据与可引用方向

### Sequential recommendation surveys

- A survey on sequential recommendation: From classical models to LLM-powered systems. Frontiers of Computer Science, 2025/2026.
- Sequential recommender systems: A methodological taxonomy and research frontiers. Computer Science Review, 2025.
- Sequential Recommendation System Based on Deep Learning: A Survey. Electronics, 2025.

这些文献用于支持：

```text
sequential recommendation 已从传统序列模型发展到 Transformer/LLM，但仍面临动态兴趣、长期依赖、噪声行为和语义利用问题。
```

### Long/short-term preference and denoising

- TLSTSRec: Time-aware long short-term attention neural network for sequential recommendation.
- Denoising Long- and Short-term Interests for Sequential Recommendation.
- GLAD: Graph-based long-term attentive dynamic memory for sequential recommendation.
- DualCFGL: dual-channel fusion global and local features for sequential recommendation.

这些文献用于支持：

```text
长短期兴趣平衡、动态记忆和噪声行为建模是已有真实问题。
```

### Multimodal / semantic sequential recommendation

- DMESR: Dual-view MLLM-based Enhancing Framework for Multimodal Sequential Recommendation.
- DuAF-MAT: Capturing Dynamic User Interests Under Modality Imbalance for Multimodal Sequential Recommendation.
- Beyond feature concatenation: Mutual information-driven fusion for multimodal sequential recommendation.
- VLM4Rec: Multimodal Semantic Representation for Recommendation with Large Vision-Language Models.

这些文献用于支持：

```text
语义/多模态信息不是简单拼接即可，需要对齐、融合、去噪和动态兴趣建模。
```

### Semantic ID / representation collapse related

- Empowering LLM-based Sequential Recommendation via Multimodal Embeddings and Semantic IDs.
- Learnable Item Tokenization for Generative Recommendation.
- Unified Semantic and ID Representation Learning for Deep Recommenders.
- Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation.
- Differentiable Semantic ID for Generative Recommendation.
- PRISM: Purified Representation and Integrated Semantic Modeling for Generative Sequential Recommendation.

这些文献用于支持：

```text
语义表示、语义 ID、codebook/token 表示存在冲突、退化或坍塌风险，检索与预测目标需要更谨慎地解耦。
```

### Retrieval-augmented recommendation

- RaSeRec: Retrieval-Augmented Sequential Recommendation.
- RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning.
- Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation.
- User Long-Term Multi-Interest Retrieval Model for Recommendation.

这些文献用于支持：

```text
检索增强是推荐系统真实趋势，但 CEBNet 的区别是对用户自身历史构造内部认知记忆，并进行解耦检索。
```

---

## 12. 最终建议

论文核心切入点建议固定为：

```text
现有语义序列推荐存在异质记忆纠缠问题：短期意图、长期语义偏好、噪声/重复历史行为被压缩到单一表示中，缺少上下文相关的认知记忆检索机制。
```

CEBNet 的解决路径：

```text
1. WEBD 构造去噪工作记忆；
2. SMC 巩固长期语义记忆；
3. trajectory-conditioned query 根据当前行为上下文检索认知记忆；
4. DEBR 解耦检索空间和表示空间；
5. history-aware calibration 缓解 top-rank 历史重复挤占。
```

这样可以把你现在所有有效模块统一成一条线，而不是显得创新点零散堆叠。
