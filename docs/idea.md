# 一.论文主线
现有序列推荐方法通常将信息（包括序列信息、ID、文本）压缩成一个统一的用户表示中，但是一个序列其实是很丰富的，包括短期意图、长期语义偏好、噪声行为和历史重复信号等等,如何提取有效信息是一个问题.

或者说:

现有序列推荐大多把用户历史压成一个统一序列表示:

历史序列 -> Transformer/RNN/Attention -> 一个user embedding -> 推荐

这样的问题是:短期意图、长期偏好、噪声行为、语义信息都被混在同一个表示空间里。

## 这样子会反复出现以下几个痛点:

A. 长短期兴趣难以平衡:用户长期喜欢A，但最近正在找B，模型不知道该偏哪边。(用户下一步行为不是单一兴趣向量能充分表达的，而是由短期意图和长期偏好共同决定。)

B. 行为序列中存在噪声和兴趣漂移:用户户历史并非干净信号。近期行为中既有真实突发意图，也有误触、探索、偶然浏览。直接把近期交互编码为当前兴趣，容易把噪声也放大进最终排序。

C. 检索和表示耦合会带来表示退化/坍塌风险:同一个hidden state既要负责相似度匹配，又要负责表达真实语义内容，容易让 memory 表示被匹配目标污染.(当同一表示同时承担“相似性检索”和“预测区分”任务时，容易出现表达能力退化。)

# 二.方法论

CEBNet 的核心不是把用户历史编码成一个更强的统一向量，而是显式模拟推荐场景中的“记忆形成—记忆巩固—上下文检索—预测校准”过程：

```text
raw item semantics
  -> semantic item encoding
  -> full-sequence wavelet denoising
  -> working memory / long-term memory split
  -> working memory anchor extraction
  -> consolidated semantic memory (SMC)
  -> trajectory-conditioned retrieval query
  -> decoupled episodic buffer retrieval (DEBR)
  -> residual cognitive fusion
  -> history-aware next-item prediction
```
这条路径对应前文提出的核心问题：现有序列推荐把短期意图、长期偏好、语义信息、噪声行为和历史重复信号混入一个单一表示。CEBNet 则把这些异质信号拆成不同认知记忆，并根据当前行为上下文动态检索。

关键设计：噪声普遍存在于用户的完整历史行为中，而非仅限于近期交互。因此 CEBNet 先对全序列统一做小波去噪，再划分工作记忆与长期记忆，确保两类记忆的构建均基于干净的语义信号。

### 2.1 问题定义与符号

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
2. 当前轨迹上下文：决定当前推荐场景下应该激活哪部分记忆。
```

因此，CEBNet 不是直接学习一个 monolithic user embedding，而是学习：

$$
\mathbf{z}_u = \text{Fuse}(\mathbf{q}_{trace}, \text{Retrieve}(\mathbf{q}_{trace}, \mathcal{M}_{cog})),
$$

其中 $\mathbf{q}_{trace}$ 是轨迹条件化检索查询，$\mathcal{M}_{cog}$ 是认知记忆库。

---

### 2.2 物品语义表示

CEBNet 的输入为物品的文本特征(如标题、类别等)。每种文本特征经预训练语言模型编码为 embedding 后,通过 Q-Former 聚合模块融合为统一的语义表示:

$$
\mathbf{e}_i^{sem} = \text{SemanticEncoder}(\text{text}_i) \in \mathbb{R}^d.
$$

其中 $\text{SemanticEncoder}(\cdot)$ 由"预训练语言模型 + Q-Former 聚合"两部分组成:预训练语言模型把每种文本特征(标题、类别等)编码为 embedding,Q-Former 在多个文本特征之间做 cross-attention 聚合,输出固定维度的语义向量。

主路径表示直接采用纯语义形式:

$$
\mathbf{e}_i = \mathbf{e}_i^{sem},
$$

对应的用户历史序列表示为:

$$
\mathbf{X}_u = [\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_t}] \in \mathbb{R}^{t \times d}.
$$

主路径表示 $\mathbf{e}_i$ 作为后续所有模块(去噪、记忆划分、SMC 巩固)的统一输入,保持纯语义以利于 SMC 的 prototype 聚类。为检索查询分支单独引入协同信号的设计在 2.5 节展开。

---

### 2.3 全序列小波去噪与认知记忆划分

#### 2.3.1 动机

噪声行为（误触、随机浏览、探索性点击）并不只存在于近期交互，而是遍布用户的完整历史序列。若在划分记忆之后再分别处理，长期记忆中的噪声将直接影响 SMC 的语义原型质量。因此 CEBNet 采用"先去噪，再划分"的策略：对全序列统一做小波去噪预处理，再将干净的序列划分为工作记忆和长期记忆。

#### 2.3.2 全序列小波去噪

基于 2.2 节得到的物品表示 $\mathbf{X}_u$，CEBNet 直接对全序列做离散小波变换：

$$
\mathbf{A}, \mathbf{D} = \text{DWT}(\mathbf{X}_u),
$$

其中 $\mathbf{A}$ 为低频近似系数（平稳趋势），$\mathbf{D}$ 为高频细节系数（局部波动）。

CEBNet 从全序列的有效位置均值估计上下文，学习一个数据驱动的动态阈值：

$$
\boldsymbol{\tau} = \alpha \cdot \sigma\left(\text{MLP}\!\left(\frac{1}{t}\sum_{j=1}^{t} \mathbf{x}_j\right)\right),
$$

并对高频细节系数做 soft-thresholding：

$$
\mathbf{D}' = \text{sign}(\mathbf{D}) \cdot \max(|\mathbf{D}| - \boldsymbol{\tau},\ 0).
$$

通过逆小波变换重构去噪后的全序列：

$$
\widetilde{\mathbf{X}}_u = \text{IDWT}(\mathbf{A}, \mathbf{D}').
$$

#### 2.3.3 认知记忆划分

在去噪后的全序列上按固定长度 $m$ 切分：

```text
working memory segment：最近 m 个交互（去噪后）；
long-term memory segment：更早的历史交互（去噪后）。
```

$$
\mathbf{X}_{wm} = [\widetilde{\mathbf{e}}_{i_{t-m+1}}, \ldots, \widetilde{\mathbf{e}}_{i_t}],
$$

$$
\mathbf{X}_{long} = [\widetilde{\mathbf{e}}_{i_1}, \ldots, \widetilde{\mathbf{e}}_{i_{t-m}}].
$$

两类记忆均基于去噪后的表示，保证了后续 anchor 提取和语义原型巩固的信号质量。

#### 2.3.4 工作记忆 Anchor 提取

CEBNet 从去噪后的近期工作记忆中通过注意力池化提取即时意图 anchor，作为后续 DEBR 的检索起点：

$$
a_j = \text{softmax}(\mathbf{w}_a^\top [\mathbf{X}_{wm}]_j),
$$

$$
\mathbf{z}_{wm} = \sum_j a_j [\mathbf{X}_{wm}]_j.
$$

其中 $\mathbf{z}_{wm}$ 是去噪工作记忆 anchor，承担"当前意图"的角色。

---

### 2.4 SMC 语义记忆巩固

#### 2.4.1 动机

长期历史通常包含大量重复、稀疏且分散的行为。如果直接把长期历史作为序列输入，模型容易面临：

```text
计算冗余；
长期偏好分散；
语义主题不稳定；
历史噪声累积。
```

SMC 的目标是将长期历史巩固成有限个稳定的 semantic memory slots。

#### 2.4.2 Memory Replay Encoder

对长期历史注入位置编码：

$$
\mathbf{E}_{long} = \text{LN}(\mathbf{X}_{long} + \mathbf{P}_{long}).
$$

然后使用 replay Transformer 得到上下文化长期历史：

$$
\widetilde{\mathbf{X}}_{long} = \text{Transformer}(\mathbf{E}_{long}).
$$


#### 2.4.3 Prototype-based Consolidation

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


---

### 2.5 轨迹条件化检索查询（Trajectory-Conditioned Retrieval Query）

#### 2.5.1 动机

工作记忆和语义记忆构造后，还需要回答一个问题：

```text
在当前推荐场景下，应该激活哪些记忆？
```

如果只用工作记忆 anchor 检索长期记忆，查询可能过于局部；如果只用长期偏好，模型又可能忽视当前意图。因此当前最优版本使用完整行为轨迹生成一个轨迹条件化检索查询。

具体解释：

```text
模型通过 next-item prediction 学习一个 trajectory-conditioned query。
```

#### 2.5.2 解耦 Trace ID 表示

主路径表示 $\mathbf{e}_i = \mathbf{e}_i^{sem}$ 是纯语义的,没有协同信号,直接用于生成检索查询会让"下一步该检索哪些记忆"这个问题缺乏物品级别的区分度。因此 CEBNet 在检索分支上单独维护一个参数独立的 trace ID embedding 表:

$$
\mathbf{E}^{trace\_id} \in \mathbb{R}^{|\mathcal{I}| \times d},
$$

物品 $i$ 的 trace ID 表示通过查表得到:

$$
\mathbf{e}_i^{trace\_id} = \mathbf{E}^{trace\_id}[i] \in \mathbb{R}^d.
$$

该表与主路径完全独立(不与任何主路径模块共享参数),只在检索分支被调用。trace 表示通过门控残差机制把 trace ID 注入到语义表示上:

$$
\mathbf{e}_i^{trace} = \text{LN}(\mathbf{e}_i^{sem} + g_i^{trace} \odot \mathbf{e}_i^{trace\_id}),
$$

其中门控系数 $g_i^{trace}$ 由语义表示和 trace ID 表示拼接后过单层线性映射 + sigmoid 得到:

$$
g_i^{trace} = \sigma\!\left(\mathbf{W}_g^{trace}\,[\mathbf{e}_i^{sem};\, \mathbf{e}_i^{trace\_id}] + b_g^{trace}\right) \in \mathbb{R}^d,
$$

其中 $\mathbf{W}_g^{trace} \in \mathbb{R}^{d \times 2d}$、$b_g^{trace} \in \mathbb{R}^d$ 为可学习参数,$[\cdot;\cdot]$ 表示向量拼接,$\sigma$ 为 element-wise sigmoid,$\text{LN}$ 为 LayerNorm。$g_i^{trace}$ 初始化偏置为负值(默认 $-2.0$),让训练初期 trace ID 信号近乎关闭,避免冷启动阶段随机初始化的 ID 噪声主导查询。

#### 2.5.3 为什么要在检索分支单独引入 ID

主路径表示 $\mathbf{e}_i$ 服务"语义聚类友好"的目标(供 SMC 的 prototype 巩固使用),要求表示空间结构稳定、聚类清晰;trace 表示 $\mathbf{e}_i^{trace}$ 服务"协同区分性强"的目标(供轨迹查询生成使用),要求能区分用户下一步的细粒度物品选择。两者优化方向不同,若共享同一套 ID 信号(例如直接把主 ID 加到主路径),ID 梯度将密集地流经 SMC 聚类路径,在稀疏数据集上引起快速过拟合,反而降低性能。解耦设计让两类信号在 ID 层面彼此独立,主路径保留纯语义,trace 路径承载协同区分性。

#### 2.5.4 Trace 输入构造

由解耦 trace 表示得到 trace 输入序列:

$$
\mathbf{X}_{trace} = [\mathbf{e}_{i_1}^{trace}, \ldots, \mathbf{e}_{i_t}^{trace}].
$$

#### 2.5.5 Causal Trace Encoder

加入位置编码后，通过因果序列编码器：

$$
\mathbf{H}_{trace} = \text{CausalTransformer}(\mathbf{X}_{trace} + \mathbf{P}_{trace}).
$$

取最后一个有效位置作为检索查询：

$$
\mathbf{q}_{trace} = \mathbf{H}_{trace}[t].
$$

### 2.6 解耦情景缓冲器检索（DEBR）

#### 2.6.1 认知记忆库构造

当前 `trace_residual_debr` 最优版本将工作记忆和长期语义记忆共同作为认知记忆库：

$$
\mathbf{M}_{cog} = [\mathbf{W}; \mathbf{M}_{sem}].
$$

其中：

```text
W：工作记忆段（近期交互）；
M_sem：SMC 输出的长期语义记忆槽。
```

解释：

```text
trajectory query 不直接替代 memory，而是从短期工作记忆和长期语义记忆中选择当前上下文相关的信息。
```

#### 2.6.2 为什么要解耦检索空间和表示空间

如果使用同一个 embedding 空间同时完成：

```text
1. similarity matching：哪个 memory 应该被检索；
2. predictive representation：检索出的 memory 如何用于预测；
```

模型会面临目标冲突。相似度检索偏向粗粒度聚类，下一物品预测需要细粒度区分。这就是前文的表示坍塌/表示退化风险。

#### 2.6.3 Retrieval Subspace

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

#### 2.6.4 Representation Subspace

同时，将认知记忆映射到表示子空间 $\mathcal{R}$：

$$
\mathbf{V}^{\mathcal{R}} = \mathbf{W}_{v}\mathbf{M}_{cog}.
$$

检索出的认知记忆内容为：

$$
\mathbf{r}_{mem} = \boldsymbol{\beta}\mathbf{V}^{\mathcal{R}}.
$$

#### 2.6.5 解决的问题

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

### 2.7 残差认知融合（Residual Cognitive Fusion）

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

### 2.8 预测层与全库排序

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

### 2.9 优化目标

CEBNet 的训练目标由三项组成：推荐损失 $\mathcal{L}_{rec}$、对比损失 $\mathcal{L}_{CL}$、原型多样性正则 $\mathcal{L}_{ortho}$：

$$
\mathcal{L} = \mathcal{L}_{rec} + \lambda_{CL}\mathcal{L}_{CL} + \lambda_{ortho}\mathcal{L}_{ortho}.
$$

#### 2.9.1 Recommendation Objective

推荐损失即 sampled softmax 交叉熵,$\mathcal{L}_{rec} = \mathcal{L}_{CE}$。给定用户 $u$ 的正样本 $i^+$ 和采样负样本集合 $\mathcal{N}$:

$$
\mathcal{L}_{rec} = -\log \frac{\exp(s(u, i^+)/\tau)}{\exp(s(u, i^+)/\tau) + \sum_{i^- \in \mathcal{N}} \exp(s(u, i^-)/\tau)},
$$

其中 $s(u, i) = \mathbf{z}_u^\top \mathbf{e}_i$ 为用户表示与候选物品的内积打分,$\tau$ 为温度系数。

#### 2.9.2 Contrastive Objective

为增强用户表示对历史扰动的鲁棒性,CEBNet 对用户历史做随机 mask 得到 masked view $\widetilde{\mathbf{X}}_u$,经同一前向得到 masked 用户表示 $\widetilde{\mathbf{z}}_u$。在 batch 内做对称 InfoNCE,正样本为同一用户的两个视图,负样本为 batch 内其他用户:

$$
\mathcal{L}_{CL} = -\frac{1}{2}\left[
\log \frac{\exp(\langle \hat{\mathbf{z}}_u, \hat{\widetilde{\mathbf{z}}}_u \rangle / \tau)}{\sum_{j \in \mathcal{B}} \exp(\langle \hat{\mathbf{z}}_u, \hat{\widetilde{\mathbf{z}}}_j \rangle / \tau)}
+
\log \frac{\exp(\langle \hat{\widetilde{\mathbf{z}}}_u, \hat{\mathbf{z}}_u \rangle / \tau)}{\sum_{j \in \mathcal{B}} \exp(\langle \hat{\widetilde{\mathbf{z}}}_u, \hat{\mathbf{z}}_j \rangle / \tau)}
\right],
$$

其中 $\hat{\mathbf{z}} = \mathbf{z} / \|\mathbf{z}\|_2$ 为 $\ell_2$ 归一化后的表示,$\mathcal{B}$ 为当前 batch 的用户集合,$\langle \cdot, \cdot \rangle$ 为内积。两项分别对应 $\mathbf{z}_u \to \widetilde{\mathbf{z}}_u$ 和 $\widetilde{\mathbf{z}}_u \to \mathbf{z}_u$ 两个方向,对称求平均。

#### 2.9.3 Memory Regularization

为防止 SMC 的 $K$ 个语义原型坍塌到同一主题,CEBNet 对原型施加多样性正则:

$$
\mathcal{L}_{ortho} = \frac{1}{K}\sum_k \max_{j\neq k}\cos(\mathbf{p}_k, \mathbf{p}_j).
$$

$\mathcal{L}_{ortho}$ 取每个原型与其最相似原型之间的余弦相似度的均值,鼓励原型之间相互分散,覆盖不同的语义主题。

---

### 2.10 方法如何逐一对应核心问题

| 核心问题 | CEBNet 对应模块 | 解决方式 |
|---|---|---|
| 噪声行为遍布全序列，近期和长期记忆均受污染 | WEBD（全序列小波去噪）| 先对全序列做 DWT + 数据驱动动态阈值 soft-thresholding + IDWT，再划分记忆 |
| 近期工作记忆中即时意图难以提取 | 工作记忆 Anchor | 从去噪后的近期行为中注意力池化提取即时意图 anchor |
| 长期偏好冗余且语义分散 | SMC | 对去噪后的长期历史巩固为有限 semantic memory slots |
| 记忆检索缺少当前上下文 | trajectory-conditioned query | 用完整行为轨迹生成动态检索查询 |
| 检索匹配和预测表示目标冲突 | DEBR | 检索子空间和表示子空间解耦 |

---

# 三. 实验设置

## 3.1 数据集描述

如表 1 所示，我们在五个公开基准数据集上进行了实验。其中四个来自 Amazon Review 2023 [10]：Baby Products (Baby)、Musical Instruments (Instrument)、Video Games (Game) 和 Industrial and Scientific (Scientific)，它们均属于典型的高稀疏电商场景，用户平均交互量少（约 8–10 次），是评估模型在冷启动与稀疏条件下建模能力的标准基准。此外，我们纳入了具有挑战性的 MovieLens-1M (ML-1M) [12]，该数据集中用户平均历史交互多达约 165 次，序列长度远超 Amazon 数据集（中位数仅为 6），且文本语义特征相对有限（仅标题与类型），对模型的长序列兴趣建模与记忆检索能力提出了更严苛的要求。

对于 Amazon 数据集，我们将商品的标题、品牌、特征、类别与描述字段拼接为文本特征，通过预训练 T5 编码器获取语义表示；对于 ML-1M，则使用电影标题与类型作为文本特征。沿用现有研究 [36, 42, 43]，我们对 Amazon 数据集采用 5-core 策略过滤不活跃用户和不流行商品。每个用户的交互记录按时间顺序排列，并采用 leave-one-out [23, 42] 策略划分数据：最后一条交互用于测试，倒数第二条用于验证，其余用于训练。

**表 1：预处理后数据集统计信息**

| Dataset | #Users | #Items | #Actions | Avg. Seq. Len | Sparsity |
|---|---|---|---|---|---|
| Baby | 150,777 | 36,012 | 1,223,004 | 8.1 | 99.977% |
| Instrument | 57,439 | 24,587 | 510,507 | 8.9 | 99.964% |
| Game | 94,762 | 25,612 | 801,031 | 8.5 | 99.967% |
| Scientific | 50,985 | 25,848 | 412,947 | 8.1 | 99.969% |
| ML-1M | 6,040 | 3,416 | 999,611 | 165.5 | 95.155% |

## 3.2 基线方法

我们将 DREAMRec 与一系列代表性最先进方法进行对比，围绕本文核心问题——从复杂历史序列中精细建模用户兴趣——分为以下三类：

**(1) 基于 ID 的序列推荐方法**，仅依赖物品 ID 学习协同交互信号，包括基于 RNN 的 GRU4Rec [8]、基于 Transformer 的 SASRec [14] 和 BERT4Rec [27]，以及引入对比学习的 DuoRec [21] 和 MAERec [38]。这类方法缺乏语义信息，难以缓解数据稀疏性。

**(2) 语义增强序列推荐方法**，利用物品文本或多模态特征丰富表示学习，包括 FDSA [39]、S3Rec [42]、UniSRec [11]、VQRec [9]、MMSR [12] 和 CCFRec [18]。这类方法虽然改善了物品表示质量，但仍沿用统一用户表示范式，难以解决行为噪声与长短期兴趣动态协调问题。

**(3) 去噪序列推荐方法**，利用频域滤波或去噪机制提升序列表示鲁棒性，包括 FMLP-Rec [43] 和 TedRec [36]。这类方法通过全局低通滤波抑制高频噪声，但其统一的频域处理难以区分随机噪声与用户真实突发兴趣，且缺乏对长期兴趣的显式建模。

## 3.3 评估指标

我们在测试集上评估模型性能，采用 full-ranking 协议，将真实下一物品与候选池中所有其他物品进行全库排序，并报告所有用户的平均结果。我们采用 Recall@K 和 NDCG@K 两个指标，其中 K 取 5 和 10。

## 3.4 实现细节

我们基于 PyTorch 框架实现 DREAMRec，并参考各方法原始代码复现所有基线。为保证公平对比，各模型均使用 AdamW 优化器进行优化，物品嵌入维度统一固定为 128，隐层维度为 512。所有模型的学习率在 \{5\times10^{-4},\ 1\times10^{-3}\} 范围内调优，其余超参数按各方法原论文设置。

对于 DREAMRec，训练采用 sampled softmax 交叉熵作为主推荐损失，负采样数设为 24,000。在四个 Amazon 数据集上，最大历史截断长度设为 50，工作记忆长度 $m$ 设为 10，语义原型数 $K$ 设为 16，小波基函数选用 Haar 小波；在 ML-1M 上，为充分建模其长序列结构，工作记忆长度 $m$ 相应调整。温度系数 $\tau$ 设为 0.05，对比损失权重 $\lambda_{CL}$ 设为 0.2，原型多样性正则权重 $\lambda_{proto}$ 设为 0.1，批大小为 300，训练使用早停（patience = 10）。我们将提供完整代码、数据集与实验日志以保证可复现性。