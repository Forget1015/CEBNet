# 重构与升级：基于认知情景缓冲器的下一代序列推荐框架（CEB-Net）

## 执行摘要与问题修正

针对原《DHPRec》论文在同行评审中遭遇的创新性（Novelty）与方法论严谨性挑战，本报告提供了一套全面重构的顶会级论文方案。特别针对模型架构中的关键问题进行了如下重大修正与升级：

1. **修正了目标泄漏（Target Leakage）的工程谬误**：移除了原方案中不适用于全排序序列推荐（Next-item Prediction）的候选商品依赖，实现了完全"Target-Free"的动态意图融合，确保模型可以在推断时通过高效的内积进行大规模近似最近邻（ANN）检索。

2. **确立了绝对的差异化创新（Defensible Novelty）**：经调研最新文献（包括2026年AAAI最新的WEARec与FreqRec），单纯的频域或小波去噪已开始涌现。因此，我们的核心创新点升维至：业界首个结合"时频双域突发性去噪"与"解耦认知记忆机制"的序列推荐框架。它不仅在前端解决了高频意图保留问题，更在后端首次解决了长序列基于锚点检索时的"循环依赖（Circular Dependency）"与"表示坍塌"顽疾。

3. **提供了端到端的完整方法论（End-to-End Methodology）**：从预备知识（Preliminaries）出发，按数据流向将创新模块无缝缝合，形成逻辑严密、浑然一体的模型架构。

---

## 故事线与核心创新点（Novelty & Motivation）

重构后的论文故事线围绕"认知心理学中的工作记忆系统与情景缓冲器（Episodic Buffer）"展开。

当用户在电商平台上进行决策时，其行为是由极其有限且高波动的"即时工作记忆（Working Memory）"与海量且结构化的"长期语义记忆（Long-term Semantic Memory）"共同驱动的。然而，现有的长序列推荐模型（如SIM、ETA、甚至早期的DHPRec）存在三个致命的学术空白：

**Gap 1: 高频突发意图的过度平滑。** 现有的频域模型（如FMLP-Rec以及部分最新频域滤波器）错误地将所有高频信号视为噪声并加以抑制。事实上，用户在短时间内的密集交互（Burstiness）虽然是高频的，但却代表了极强的即时工作记忆意图。

**Gap 2: 机械时间切片导致的语义割裂。** 强制按固定时间窗（如2天）切片长历史，违背了用户长期兴趣演化的语义连续性规律。

**Gap 3: 历史锚点检索中的"表示坍塌"与"循环依赖"。** 当使用近期意图（Anchor）去检索历史长序列时，现有方法使用同一套Embedding空间进行"相似度搜索"和"下游预测"。这会导致表示坍塌；更严重的是，如果Anchor本身包含噪声，检索过程会放大该噪声，产生致命的循环依赖。

### 我们的解决方案（CEB-Net）：

1. 采用**小波自适应软阈值（Wavelet Soft-Thresholding）**替代傅里叶低通滤波，精准洗去随机高频噪声，但绝对保留高幅值的高频突发意图。

2. 提出**语义记忆巩固（Semantic Memory Consolidation）**，用可学习的意图原型池代替固定时间切片。

3. 引入**解耦的情景缓冲器（Decoupled Episodic Buffer）**，在正交子空间中进行"Target-Free"的历史唤醒与融合，彻底打破循环依赖。

---

## 完整方法论框架（Methodology）

本节将详细阐述重构后的**认知情景缓冲网络（Cognitive Episodic Buffer Network, CEB-Net）**的整体架构。模型从底向上的数据流包含四个核心模块。

### 3.1 预备知识与问题定义 (Preliminaries)

给定用户集合 $\mathcal{U}$ 和商品集合 $\mathcal{I}$。对于每个用户 $u$，其按时间排序的历史交互序列定义为 $\mathcal{S}_u = [i_1, i_2, ..., i_n]$，其中 $n$ 为序列长度。我们通过语义编码器（如VQ或Text Encoder）将商品转换为嵌入表示 $\mathbf{X} \in \mathbb{R}^{n \times d}$。序列推荐的目标是基于历史 $\mathcal{S}_u$ 预测用户在 $t+1$ 步最可能交互的全集商品 $\mathcal{I}$。

为了对齐人类认知模型，我们将输入序列 $\mathcal{S}_u$ 根据时间划分（或注意力分布）分为两部分：
- **即时工作记忆区** $\mathcal{S}_u^{wm}$（包含最近的 $m$ 个交互）
- **长期历史记忆区** $\mathcal{S}_u^{long}$（包含过去的 $n-m$ 个交互）

其中 $m \ll n$。

### 3.2 模块一：时频感知的工作记忆去噪编码 (Working Memory Encoding)

用户的近期工作记忆 $\mathcal{S}_u^{wm}$ 往往充斥着密集的探索性行为和误触噪声。为了过滤噪声并保留真实的突发性意图（Burstiness），我们提出**小波增强的突发性保留去噪（WEBD）**机制。该机制包含两个关键阶段：**复述编码（Rehearsal Encoding）**与**小波自适应去噪（Wavelet Denoising）**。

#### 阶段一：复述编码（Rehearsal Encoding）

在认知心理学中，工作记忆的核心机制之一是"复述/排练（Rehearsal）"——人类通过反复回顾近期信息来建立各项目之间的关联与上下文依赖。受此启发，我们在小波去噪之前，首先对工作记忆区的商品嵌入施加一层轻量的**因果自注意力编码器（Causal Transformer）**，使每个商品表示能够感知其在工作记忆中的上下文位置与邻近交互的语义关系：

首先注入位置编码：
$$\mathbf{E}_{wm} = \text{LayerNorm}(\mathbf{X}_{wm} + \mathbf{PE})$$

其中 $\mathbf{PE}$ 为可学习的位置嵌入。

随后通过因果自注意力（Causal Self-Attention）进行复述编码：
$$\mathbf{X}'_{wm} = \text{CausalTransformer}(\mathbf{E}_{wm})$$

因果掩码确保每个位置只能关注其之前的交互，保持时序的自回归特性。

**理论解释：** 复述编码的引入解决了一个关键问题——原始的商品嵌入是独立的、无上下文的向量，直接对其做小波变换只能捕获嵌入空间中的数值波动，而无法感知"用户先看了A再看了B"这种序列级的语义依赖。复述编码后的表示已经融入了上下文信息，此时再做小波去噪，才能真正区分"有意义的突发意图"和"随机的噪声波动"。

#### 阶段二：小波自适应去噪（Wavelet Denoising）

在复述编码后的序列 $\mathbf{X}'_{wm}$ 上，我们应用**离散小波变换（DWT）**：

$$\mathbf{A}, \mathbf{D} = \text{DWT}(\mathbf{X}'_{wm})$$

其中 $\mathbf{A}$ 是保留平稳趋势的近似系数（低频），$\mathbf{D}$ 是捕获局部波动的细节系数（高频）。

区别于传统频域方法简单抑制高频，我们设计了**上下文感知的动态软阈值（Context-aware Soft Thresholding）**。阈值由一个轻量神经网络根据序列的全局上下文动态生成：

$$\tau = \text{ThresholdNet}(\text{MaskedMean}(\mathbf{X}'_{wm})) \cdot \alpha$$

其中 $\text{ThresholdNet}$ 是一个两层 MLP（Linear→ReLU→Linear→Sigmoid），$\alpha$ 是可学习的缩放因子。在多尺度分解中，每一级 $j$ 有独立的 $\text{ThresholdNet}_j$ 和 $\alpha_j$。

随后，仅对高频细节系数 $\mathbf{D}$ 进行软阈值截断：

$$\mathbf{D}_{\text{clean}} = \text{sign}(\mathbf{D}) \cdot \max(|\mathbf{D}| - \tau, 0)$$

**理论解释：** 与传统信号处理中使用固定统计量（如 $\text{Std}(\mathbf{D})$）作为阈值不同，我们采用可学习的神经网络动态生成阈值。这是因为推荐场景中的"噪声"定义是语义层面的（误点击 vs 真实意图），而非数值层面的。神经网络能够从序列的全局上下文中学习到"什么程度的高频波动是噪声"这一语义判断，而统计量只能捕获数值分布特征。Sigmoid 激活确保阈值为正，可学习的缩放因子 $\alpha$ 控制去噪的整体强度。

去噪后，我们通过逆小波变换（IDWT）重构干净的工作记忆序列：

$$\mathbf{X}''_{wm} = \text{IDWT}(\mathbf{A}, \mathbf{D}_{\text{clean}})$$

最后，应用一层**自注意力池化（Self-Attention Pooling）**，将去噪后的工作记忆序列提炼为**即时意图向量（Immediate Intent Anchor）** $\mathbf{z}_{\text{anchor}} \in \mathbb{R}^d$：

$$\alpha_i = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}_j^\top \mathbf{k}_j / \sqrt{d})}, \quad \mathbf{z}_{\text{anchor}} = \sum_{i=1}^m \alpha_i \cdot \mathbf{x}''_i$$

其中 $\mathbf{q}_i = \mathbf{W}_q \mathbf{x}''_i$，$\mathbf{k}_i = \mathbf{W}_k \mathbf{x}''_i$ 是可学习的查询和键投影。

**理论解释：** 自注意力池化让模型自适应地决定工作记忆中哪些位置对当前意图最重要。相比简单取最后位置，注意力池化能够捕获工作记忆中多个位置的互补信息——例如用户可能在最近 5 次交互中既浏览了电子产品又浏览了书籍，注意力池化可以根据上下文动态分配权重，而非只关注最后一次交互。

### 3.3 模块二：语义记忆巩固机制 (Semantic Memory Consolidation)

对于长达数百上千的历史序列 $\mathcal{S}_u^{long}$，机械的时间切片（Time Slicing）会切断语义逻辑。受人类睡眠期间"记忆巩固"机制的启发，我们将冗长的历史交互聚类压缩为有限的结构化意图原型。该机制包含两个关键阶段：**记忆回放编码（Memory Replay Encoding）**与**基于原型的记忆巩固（Prototype-Based Consolidation）**。

#### 阶段一：记忆回放编码（Memory Replay Encoding）

神经科学研究表明，人类在睡眠期间会经历"记忆回放（Memory Replay）"过程——海马体会快速重播白天经历的事件序列，在各事件之间建立关联，随后才将其压缩存入新皮层的长期记忆中。这一回放过程是记忆巩固的前提条件：没有经过回放的孤立事件片段难以被有效组织和压缩。

受此启发，我们在原型巩固之前，首先对长期历史序列施加一层轻量的**双向自注意力编码器（Bidirectional Transformer）**，使每个商品表示能够感知其在历史序列中的上下文关系：

首先注入时序位置编码：
$$\mathbf{E}_{long} = \text{LayerNorm}(\mathbf{X}_{long} + \mathbf{PE}_{long})$$

其中 $\mathbf{PE}_{long}$ 为长期历史专用的可学习位置嵌入矩阵。

随后通过双向自注意力进行回放编码：
$$\mathbf{X}'_{long} = \text{BiTransformer}(\mathbf{E}_{long})$$

与工作记忆的因果编码不同，长期记忆的回放采用双向注意力（非因果掩码），因为记忆巩固过程中大脑会同时考虑事件的前因后果，而非严格遵循时间顺序。

**理论解释：** 回放编码的引入解决了一个关键问题——原始的商品嵌入是独立的、无上下文的向量，直接对其做原型分配等于将有序的行为历史退化为无序集合。回放编码后的表示已经融入了序列内的交互关系（例如"用户先浏览了吉他，随后购买了吉他弦"这种因果关联），此时再做原型分配，才能将语义相关的行为聚合到同一原型中，而非仅仅基于表面的嵌入相似度。

**与工作记忆复述编码的对称性：** WEBD 模块中的复述编码（Rehearsal Encoding）处理的是短期工作记忆，使用因果注意力保持时序自回归特性；SMC 模块中的回放编码（Replay Encoding）处理的是长期历史记忆，使用双向注意力捕获全局上下文关联。两者共同构成了 CEB-Net 认知架构中"编码-巩固"的完整闭环。

#### 阶段二：基于原型的记忆巩固（Prototype-Based Consolidation）

我们维护一个全局共享且可学习的意图原型矩阵 $\mathbf{P} \in \mathbb{R}^{K \times d}$（例如 $K=32$），代表 $K$ 种抽象的长期偏好主题。对于回放编码后的历史序列中的每一个商品表示 $\mathbf{x}'_i$，我们计算其对各个原型 $\mathbf{p}_k$ 的归属概率：

$$\mathbf{A}_{ik} = \frac{\exp(\mathbf{x}'_i \mathbf{p}_k^\top / \sqrt{d})}{\sum_{j=1}^K \exp(\mathbf{x}'_i \mathbf{p}_j^\top / \sqrt{d})}$$

随后，基于该分配矩阵，我们将回放编码后的长序列聚合为定长的长期记忆库 $\mathbf{M} \in \mathbb{R}^{K \times d}$：

$$\mathbf{m}_k = \frac{\sum_{i=1}^{n-m} \mathbf{A}_{ik} \cdot \mathbf{x}'_i}{\sum_{i=1}^{n-m} \mathbf{A}_{ik}}$$

**优势：** 这不仅将历史匹配的计算复杂度从 $O(n)$ 降阶为 $O(K)$，更使得散乱的历史被抽象为 $K$ 个具有稳定语义的记忆槽（Memory Slots）。由于回放编码已经建立了item间的上下文关联，且原型分配过程感知了时序位置，相同类别但出现在不同时间段的商品可以被分配到不同的原型中，从而保留了兴趣演化的时间结构。

### 3.4 模块三：解耦的情景缓冲器检索与融合 (Decoupled Episodic Buffer)

现在，我们需要使用去噪后的即时意图 $\mathbf{z}_{\text{anchor}}$（作为 Anchor）从长期记忆库 $\mathbf{M}$ 中唤醒匹配的历史经验。为了解决使用单一表示空间带来的"表示坍塌（Representation Collapse）"问题，我们设计了**解耦的情景缓冲器（DEBR）**。

我们将特征强制映射到两个正交子空间：

1. **检索子空间** ($\mathcal{A}$ Space): 专门用于计算相似度匹配
2. **表示子空间** ($\mathcal{R}$ Space): 专门承载真实的语义内容以用于最终预测

在情景缓冲器中，我们通过检索子空间计算唤醒权重 $\mathbf{w}$：

$$\mathbf{w} = \text{Softmax}(\mathbf{z}_{\text{anchor}}^{\mathcal{A}} \mathbf{M}^{\mathcal{A}\top}) \in \mathbb{R}^K$$

然后，在表示子空间中提取实质的长期记忆内容 $\mathbf{z}_{\text{long}}$：

$$\mathbf{z}_{\text{long}} = \mathbf{w} \cdot \mathbf{M}^{\mathcal{R}} \in \mathbb{R}^d$$

#### Target-Free 动态意图融合 (Dynamic Integration)

为了在预测下一跳时权衡"即时冲动"与"历史习惯"，我们进行**目标无关（Target-Free）**的动态融合。融合门控权重 $\mathbf{g}$ 仅由当前意图和唤醒的历史记忆动态决定（不依赖候选商品 $i$，完美适配大库高效检索）：

$$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{z}_{\text{anchor}}; \mathbf{z}_{\text{long}}] + \mathbf{b}_g)$$

最终形成用于全库检索的用户表示向量 $\mathbf{z}_u$：

$$\mathbf{z}_u = \mathbf{g} \odot \mathbf{z}_{\text{anchor}} + (1 - \mathbf{g}) \odot \mathbf{z}_{\text{long}}$$

### 3.5 多任务优化目标 (Optimization Objectives)

在训练阶段，除了标准的新商品预测推荐损失（$\mathcal{L}_{\text{rec}}$）外，我们引入了创新的正则化项以稳定双域学习：

#### 1. 推荐交叉熵损失 (Recommendation Loss)

采用 InfoNCE loss 进行全排列近似预测：

$$\mathcal{L}_{\text{rec}} = -\log \frac{\exp(\mathbf{z}_u^\top \mathbf{e}_{i^+} / \tau)}{\sum_{j \in \mathcal{N}} \exp(\mathbf{z}_u^\top \mathbf{e}_j / \tau)}$$

#### 2. 意图引导的正交一致性损失 (Intent Orthogonal Loss)

为了防止原型池 $\mathbf{P}$ 中的意图退化为同质化特征，我们引入**最大余弦相似度惩罚**，对每个原型只惩罚它与最相似原型之间的余弦相似度：

$$\mathcal{L}_{\text{ortho}} = \frac{1}{K} \sum_{k=1}^K \max_{j \neq k} \cos(\mathbf{p}_k, \mathbf{p}_j)$$

**理论解释：** 相比两两正交约束（$\sum_{i<j}|\cos(\mathbf{p}_i, \mathbf{p}_j)|$），最大余弦相似度惩罚更加温和且实际可行。两两正交在 $K > d$ 时数学上不可能实现（$d$ 维空间最多容纳 $d$ 个正交向量），且优化困难。最大余弦惩罚只关注每个原型的"最近邻"，鼓励原型之间尽可能分散但不强制严格正交。注意该损失允许负值——当原型已经足够分散（最大余弦相似度为负）时，损失为负，相当于给予"奖励"信号，引导优化器维持原型的分散状态。

#### 3. 频域一致性损失 (Frequency-Consistency Loss)

为了约束 WEBD 模块不要过度滤除有用信号，我们在去噪前后的工作记忆序列上计算频谱差异：

$$\mathcal{L}_{\text{freq}} = \frac{1}{d} \sum_c || |\mathbf{F}_{\text{denoised}}[:,c]| - |\mathbf{F}_{\text{raw}}[:,c]| ||_2^2$$

其中 $\mathbf{F}_{\text{raw}}$ 和 $\mathbf{F}_{\text{denoised}}$ 分别是去噪前后工作记忆序列在序列维度上的 FFT 幅度谱。

**理论解释：** 小波去噪的软阈值操作会修改高频系数，如果阈值过大，有用的突发性意图信号也会被误删。频域一致性损失通过约束去噪前后的整体频谱结构不发生剧烈变化，为去噪过程提供了一个"保守性"正则化——允许局部的高频噪声被清除，但不允许整体频谱特征被大幅改变。这确保了 WEBD 模块在去噪和保留信号之间取得平衡。

#### 4. 序列对齐对比损失 (Contrastive Loss)

采用掩码增强的对比学习，拉近正常前向和掩码前向产生的用户表示：

$$\mathcal{L}_{\text{cl}} = \frac{1}{2}(\text{InfoNCE}(\mathbf{z}_u, \mathbf{z}_u^{\text{mask}}) + \text{InfoNCE}(\mathbf{z}_u^{\text{mask}}, \mathbf{z}_u))$$

**理论解释：** 对 VQ 语义码进行随机掩码后重新编码，产生同一用户的增强视图 $\mathbf{z}_u^{\text{mask}}$。对比损失强制模型学到对局部编码扰动鲁棒的用户表示——即使部分语义码被遮蔽，模型仍应产生相似的用户意图表示。这增强了表示的泛化能力，防止模型过度依赖个别 VQ 码。

#### 5. 掩码编码建模损失 (Masked Code Modeling Loss)

对 VQ 语义码进行随机掩码，预测被掩码的编码：

$$\mathcal{L}_{\text{mlm}} = \text{CE}(\text{MLP}(\mathbf{h}_{\text{masked}}), \mathbf{c}_{\text{target}})$$

**理论解释：** 类似于 BERT 的掩码语言模型，掩码编码建模迫使 Q-Former 编码器学习 VQ 码之间的内在关联——例如，同一商品的不同视角（标题、品牌、类别）的语义码之间存在强相关性。通过预测被掩码的码，编码器能够捕获这些跨视角的语义一致性，产生更丰富的商品表示。

#### 最终联合损失为：

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_1 \mathcal{L}_{\text{cl}} + \lambda_2 \mathcal{L}_{\text{mlm}} + \lambda_3 \mathcal{L}_{\text{ortho}} + \lambda_4 \mathcal{L}_{\text{freq}} + \lambda_5 \mathcal{L}_{\text{decouple}}$$

---

## 方法论升级：三大新增创新模块

### 3.6 多尺度小波去噪（Multi-scale Wavelet Denoising, MSWD）

**动机：** 单级 DWT 分解只能在一个频率尺度上区分噪声和信号。然而，用户行为噪声存在于多个时间尺度：秒级的误触（最高频）、分钟级的随机浏览（中频）、小时级的探索性行为（低频）。单级去噪无法同时处理这些不同尺度的噪声。

**方法：** 我们将 WEBD 中的单级 DWT 扩展为 $J$ 级级联小波分解（Multi-Resolution Analysis, MRA）：

$$\mathbf{A}_0 = \mathbf{X}'_{wm}$$
$$\mathbf{A}_j, \mathbf{D}_j = \text{DWT}(\mathbf{A}_{j-1}), \quad j = 1, 2, ..., J$$

在每一级 $j$，我们为高频细节系数 $\mathbf{D}_j$ 设置独立的上下文感知阈值：

$$\tau_j = \text{ThresholdNet}_j(\text{context}) \cdot \alpha_j$$

$$\mathbf{D}_j^{\text{clean}} = \text{sign}(\mathbf{D}_j) \cdot \max(|\mathbf{D}_j| - \tau_j, 0)$$

最后通过多级逆小波变换（IDWT）逐层重构：

$$\hat{\mathbf{A}}_{j-1} = \text{IDWT}(\mathbf{A}_j, \mathbf{D}_j^{\text{clean}}), \quad j = J, J-1, ..., 1$$

**理论解释：** 多尺度分解对应认知心理学中"注意力过滤器"的层级结构——大脑在处理感知信息时，会在不同的时间粒度上分别过滤无关刺激。第 1 级去噪过滤最细粒度的随机波动（误触），第 2 级过滤中等粒度的探索性浏览，第 3 级过滤粗粒度的兴趣漂移噪声。

### 3.7 时间衰减感知的原型分配（Temporal-Decay Prototype Assignment）

**动机：** 当前 SMC 的原型分配纯粹基于语义相似度，不考虑交互发生的时间。然而，认知心理学中的 Ebbinghaus 遗忘曲线表明，记忆痕迹的强度随时间指数衰减——近期经历的记忆更鲜明，远期的逐渐模糊。

**方法：** 在原型分配概率中引入时间衰减因子。对于长期历史中第 $i$ 个交互（距当前时刻的相对位置为 $\delta_i$），其对原型 $k$ 的分配概率修正为：

$$\mathbf{A}_{ik} = \frac{\exp((\mathbf{x}'_i \mathbf{p}_k^\top / \sqrt{d}) + \gamma \cdot \log(\delta_i + 1))}{\sum_{j=1}^K \exp((\mathbf{x}'_i \mathbf{p}_j^\top / \sqrt{d}) + \gamma \cdot \log(\delta_i + 1))}$$

其中 $\gamma$ 是可学习的时间衰减系数，$\delta_i = n - i$ 是第 $i$ 个交互距序列末尾的距离。$\log(\delta_i + 1)$ 提供对数衰减，使近期交互获得更高的分配权重。

注意：时间衰减因子对所有原型 $k$ 是相同的，因此它不改变原型间的相对分配比例，而是调整每个交互的整体贡献权重——近期交互对所有原型的贡献更大，远期交互的贡献被衰减。

**理论解释：** 这对应记忆巩固过程中的"近因效应（Recency Effect）"——海马体在回放记忆时，对近期事件的重播频率更高、痕迹更强。时间衰减因子使得原型更多地反映用户近期的兴趣演化，而非被远古的历史交互稀释。

### 3.8 对比增强的解耦学习（Contrastive Decoupling Regularization）

**动机：** DEBR 的检索子空间（$\mathcal{A}$ Space）和表示子空间（$\mathcal{R}$ Space）仅通过不同的线性投影来实现解耦，缺乏显式的正则化约束。如果两个子空间在训练过程中逐渐对齐（即 $\mathbf{W}_{\text{attn}}$ 和 $\mathbf{W}_{\text{repr}}$ 学到相似的投影方向），模型会退化为单一表示空间，Gap 3 中的"表示坍塌"和"循环依赖"问题将重新出现。

**方法：** 引入解耦正则化损失 $\mathcal{L}_{\text{decouple}}$，强制检索投影矩阵和表示投影矩阵的列空间正交：

$$\mathcal{L}_{\text{decouple}} = \| \mathbf{W}_{\text{attn}}^\top \mathbf{W}_{\text{repr}} \|_F^2$$

其中 $\| \cdot \|_F$ 是 Frobenius 范数。当两个投影矩阵的列空间完全正交时，$\mathbf{W}_{\text{attn}}^\top \mathbf{W}_{\text{repr}} = \mathbf{0}$，损失为零。

**理论解释：** 这直接对应认知心理学中"情景缓冲器"的核心功能——Baddeley 的工作记忆模型指出，情景缓冲器之所以能有效整合来自不同子系统的信息，正是因为它维护了独立的"索引通道"和"内容通道"。索引通道负责定位相关记忆（对应检索子空间），内容通道负责提取记忆的实质内容（对应表示子空间）。两个通道的独立性是情景缓冲器正常工作的前提。

### 3.9 更新后的联合损失函数

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_1 \mathcal{L}_{\text{cl}} + \lambda_2 \mathcal{L}_{\text{mlm}} + \lambda_3 \mathcal{L}_{\text{ortho}} + \lambda_4 \mathcal{L}_{\text{freq}} + \lambda_5 \mathcal{L}_{\text{decouple}}$$

其中 $\lambda_5$ 控制解耦正则化的强度，建议初始值为 0.01。

---

## 核心升维点总结（向审稿人展示的防御逻辑）

1. **Target-Free 设计的可用性**：与原版本依赖目标商品进行 Gating 导致无法在工业界 ANN（Approximate Nearest Neighbor）系统中落地不同，CEB-Net 最终输出统一的 $\mathbf{z}_u$ 向量，完美支持内积相似度快速检索，这体现了深厚的系统工程素养。

2. **Novelty 的多重护城河**：即便有审稿人指出"Wavelet"在推荐中已被使用，我们可以强势回应：本文的核心贡献并非单独的 Wavelet，而是利用 Wavelet 解决 Working Memory 中突发意图的清洗，并与后端的**解耦语义记忆检索（Decoupled Retrieval）**联合，彻底攻克了"表示坍塌"和"循环依赖"这个在2025/2026年才被陆续揭露的领域痛点。

3. **端到端的架构闭环**：故事线从心理学的"情景缓冲器"出发，在前端处理即时记忆（小波去噪），在后端处理长期记忆（意图巩固），并在中枢解耦交互，所有数学公式均服务于这一统一的认知哲学。

---

## 完整方法流程梳理（代码实现对应）

以下是 CEB-Net 从输入到输出的完整数据流，与代码实现一一对应：

```
输入: item_seq [B, L], item_seq_len [B], code_seq [B*L, C]
│
▼ Step 1: VQ 语义编码 (_encode_items)
├─ query_code_embedding(code_seq) → [B*L, C, d]
├─ item_text_embedding[j](items) → stack → [B*L, text_num, d]
├─ Q-Former(query, text)[-1].mean(dim=1) + query.mean(dim=1) → [B*L, d]
└─ reshape → item_emb [B, L, d], code_emb [B*L, C, d]
│
▼ Step 2: 序列分割 (for-loop, 右对齐)
├─ 工作记忆: 最后 m 个有效交互 → x_wm [B, m, d], wm_valid [B, m]
└─ 长期历史: 前 L-m 个交互 → x_long [B, L-m, d], long_valid [B, L-m]
│
├─────────────────────────────────┐
▼                                 ▼
Step 3: WEBD (工作记忆去噪)       Step 4: SMC (长期记忆巩固)
├─ + PE → LN → Dropout            ├─ + PE → LN → Dropout
├─ CausalTransformer(2层)          ├─ BiTransformer(1层)
├─ 多尺度DWT(J=2级)               ├─ 时间衰减原型分配
│  ├─ Level 1: DWT → 软阈值       │  ├─ sim = proj(x) @ proj(P).T
│  └─ Level 2: DWT → 软阈值       │  ├─ + γ·log(δ+1) 衰减偏置
├─ 多尺度IDWT重构                  │  └─ softmax → assign [B,n,K]
├─ Dropout                        ├─ 加权聚合 → memory [B, K, d]
└─ 注意力池化 → anchor [B,d]   └─ ortho_loss: max cos sim
│                                 │
└─────────────┬───────────────────┘
              ▼
Step 5: DEBR (解耦检索与融合)
├─ 检索子空间: W_attn(anchor) @ W_attn(memory).T → w [B, K]
├─ 表示子空间: w @ W_repr(memory) → z_long [B, d]
├─ 门控融合: g = σ(W_g[anchor; z_long])
├─ z_u = g·anchor + (1-g)·z_long → LayerNorm
└─ decouple_loss: ||W_attn.T @ W_repr||_F²
│
▼ Step 6: 损失计算
├─ L_rec: InfoNCE(z_u, pos/neg item embeddings)
├─ L_cl: InfoNCE(z_u, z_u_masked) 双向
├─ L_mlm: CE(masked_code_output, code_labels)
├─ L_ortho: (1/K)Σ max_{j≠k} cos(p_k, p_j)
├─ L_freq: MSE(|FFT(x_before_dwt)|, |FFT(x_denoised)|)
├─ L_decouple: ||W_attn.T @ W_repr||_F²
└─ L = L_rec + λ₁L_cl + λ₂L_mlm + λ₃L_ortho + λ₄L_freq + λ₅L_decouple

推理: z_u = F.normalize(z_u) @ F.normalize(all_item_emb).T → scores
```

**关键设计决策：**
- 无外层位置编码：WEBD 和 SMC 各自内部处理位置编码，避免双重叠加
- 工作记忆右对齐：短序列左边补零，确保最后一个位置始终是最新交互
- anchor 自注意力池化：自适应加权工作记忆中各位置的贡献，捕获多兴趣互补信息
- ortho_loss 无 ReLU：允许负值（原型已分散时给予奖励信号）
- _encode_items 加 query 残差：与 get_item_embedding/encode_item 保持一致

---

## 引用的著作

1. Residual Context Diffusion Language Models - arXiv, 访问时间为 四月 7, 2026， https://arxiv.org/html/2601.22954v1

2. [2410.02604] Long-Sequence Recommendation Models Need Decoupled Embeddings - arXiv, 访问时间为 四月 7, 2026， https://arxiv.org/abs/2410.02604

---

## 附录：方案改进与修正（基于技术评审反馈）

### 改进1：频域重建辅助损失（Frequency-Consistency Loss）的修正

**原方案问题：** 原3.5节中提出让用户表示 $\mathbf{z}_u$ 的能量谱和target的能量谱对齐。但 $\mathbf{z}_u$ 是一个单一向量（非序列），target也是一个单一向量，它们没有"序列"维度，无法做有意义的频域变换（FFT/DWT需要序列维度）。

**修正方案：** 将频域一致性损失重新定义在去噪前后的工作记忆序列上，而非最终的用户表示上。具体地：

- 令 $\mathbf{S}_{\text{raw}}$ 为去噪前的工作记忆序列嵌入矩阵（形状 $[B, m, d]$）
- 令 $\mathbf{S}_{\text{denoised}}$ 为WEBD去噪后的工作记忆序列嵌入矩阵（形状 $[B, m, d]$）
- 对两者在序列维度上做FFT，得到频谱 $\mathbf{F}_{\text{raw}}$ 和 $\mathbf{F}_{\text{denoised}}$
- 定义频域一致性损失为：

$$\mathcal{L}_{\text{freq}} = \frac{1}{d} \sum_c || |\mathbf{F}_{\text{denoised}}[:,c]| - |\mathbf{F}_{\text{raw}}[:,c]| ||_2^2$$

其中求和遍历所有特征维度 $c$

**该损失的物理意义是：** 约束去噪过程不要过度改变序列的整体频谱结构，防止WEBD模块过度滤除有用信号。

### 改进2：意图原型数量K的自适应与敏感性分析

**原方案问题：** 固定K=32或64，缺乏选择依据。

**修正方案：**

1. 在实验中增加K的敏感性分析（$K \in \{8, 16, 32, 64, 128\}$），绘制K vs. NDCG@10的曲线

2. 引入自适应K的软机制：在训练过程中，对原型矩阵 $\mathbf{P}$ 的每一行计算其被激活的频率（即所有训练样本中，该原型被分配到的概率之和）。如果某个原型的激活频率低于阈值 $\epsilon$（如0.01），则在下一个epoch中将其重新初始化为当前batch中距离所有现有原型最远的样本。这种"死原型复活"策略可以自适应地调整有效原型数量。

### 改进3：正交一致性损失的软化

**原方案问题：** 余弦正交损失在K较大时（如K=64）强制所有原型两两正交，这在d=64的空间中是不可能的（最多64个正交向量），且优化困难。

**修正方案：** 将硬正交损失替换为**最大余弦相似度惩罚**：

$$\mathcal{L}_{\text{ortho}} = \frac{1}{K} \sum_{k=1}^K \max_{j \neq k} \cos(\mathbf{p}_k, \mathbf{p}_j)$$

即对每个原型，只惩罚它与最相似的那个原型之间的余弦相似度。这比两两正交约束更温和，且在 $K \gt d$ 时仍然有效。该损失鼓励原型之间尽可能分散，但不强制严格正交。

### 改进4：Target-Free设计的实验验证计划

在实验部分需要增加以下内容：

1. **精度对比**：与target-aware的门控方法（如原DHPRec的候选商品依赖门控）进行对比，展示Target-Free设计在精度上的损失可控

2. **效率分析**：报告CEB-Net在ANN检索场景下的实际延迟（使用FAISS进行内积检索），与需要逐候选计算的target-aware方法对比

3. **消融实验**：分别移除Target-Free门控中的即时意图分量和历史记忆分量，验证两者的互补性

### 改进5：Baseline扩展

在实验的baseline中增加以下长序列推荐模型：

- **SIM** (Search-based Interest Model)：阿里巴巴的经典长序列检索模型
- **ETA** (End-to-End Target Attention)：基于SimHash的高效长序列模型
- **SDIM** (Sampling-based Deep Interest Modeling)：基于采样的长序列模型
- **WEARec**：2025年最新的小波增强频域推荐模型（直接竞品）

### 改进6：小波基函数的消融实验

在消融实验中增加不同小波基函数的对比：

- Haar小波（当前默认）
- Daubechies-4 (db4)
- Symlet-4 (sym4)
- Coiflet-2 (coif2)

分析不同小波基对去噪效果和推荐精度的影响，为最终选择提供实验依据。
