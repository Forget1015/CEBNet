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
  -> working memory / long-term memory construction
  -> denoised working memory (WEBD)
  -> consolidated semantic memory (SMC)
  -> trajectory-conditioned retrieval query
  -> decoupled episodic buffer retrieval (DEBR)
  -> residual cognitive fusion
  -> history-aware next-item prediction
```
这条路径对应前文提出的核心问题：现有序列推荐把短期意图、长期偏好、语义信息、噪声行为和历史重复信号混入一个单一表示。CEBNet 则把这些异质信号拆成不同认知记忆，并根据当前行为上下文动态检索。

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

### 2.2 认知记忆划分

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



---

### 2.3 WEBD 去噪工作记忆编码

#### 2.3.1 动机

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

#### 2.3.2 Rehearsal Encoder

首先对工作记忆注入位置编码：

$$
\mathbf{E}_{wm} = \text{LN}(\mathbf{X}_{wm} + \mathbf{P}_{wm}).
$$

然后使用因果 Transformer 进行复述编码：

$$
\widetilde{\mathbf{X}}_{wm} = \text{CausalTransformer}(\mathbf{E}_{wm}).
$$

这里使用 causal attention，是因为工作记忆仍然服务于下一物品预测，不能让靠后位置反向泄露给靠前位置。

#### 2.3.3 Wavelet Denoising

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

#### 2.3.4 Working Memory Anchor

CEBNet 使用注意力池化从去噪工作记忆中提取即时意图 anchor：

$$
a_j = \text{softmax}(\mathbf{w}_a^\top \mathbf{W}_j),
$$

$$
\mathbf{z}_{wm} = \sum_j a_j \mathbf{W}_j.
$$

其中 $\mathbf{z}_{wm}$ 是去噪工作记忆 anchor。

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

#### 2.5.2 Decoupled Trace ID Input

为了增强协同转移信号，CEBNet 在 trace branch 中使用 decoupled ID：

$$
\mathbf{e}_i^{trace} = \text{LN}(\mathbf{e}_i^{sem} + g_i^{id}\odot \mathbf{e}_i^{id}).
$$

得到 trace 输入序列：

$$
\mathbf{X}_{trace} = [\mathbf{e}_{i_1}^{trace}, \ldots, \mathbf{e}_{i_t}^{trace}].
$$

#### 2.5.3 Causal Trace Encoder

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

当前 `trace_residual_debr` 最优版本将去噪工作记忆和长期语义记忆共同作为认知记忆库：

$$
\mathbf{M}_{cog} = [\mathbf{W}; \mathbf{M}_{sem}].
$$

其中：

```text
W：WEBD 输出的去噪工作记忆；
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

### 2.7：残差认知融合（Residual Cognitive Fusion）

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

### 2.9 优化目标：论文中建议写成三类 loss

当前代码中记录多个 loss，但论文主文不要写成 7 个独立目标。建议组织为三类：

$$
\mathcal{L} = \mathcal{L}_{rec} + \lambda_{sem}\mathcal{L}_{sem} + \lambda_{mem}\mathcal{L}_{mem}.
$$

#### 2.9.1 Recommendation Objective

主推荐损失：

$$
\mathcal{L}_{CE} = -\log \frac{\exp(s(u,i^+))}{\exp(s(u,i^+)) + \sum_{i^-\in\mathcal{N}}\exp(s(u,i^-))}.
$$




#### 2.9.2 Semantic Consistency Objective

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

#### 2.9.3 Memory Regularization Objective

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

### 2.10 方法如何逐一对应核心问题

| 核心问题 | CEBNet 对应模块 | 解决方式 |
|---|---|---|
| 短期意图和噪声行为纠缠 | WEBD | 小波软阈值去噪，保留高幅值突发意图 |
| 长期偏好冗余且语义分散 | SMC | 巩固为有限 semantic memory slots |
| 记忆检索缺少当前上下文 | trajectory-conditioned query | 用当前轨迹生成动态检索查询 |
| 检索匹配和预测表示目标冲突 | DEBR | 检索子空间和表示子空间解耦 |

---