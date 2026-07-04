# Claude Code 技能完整使用指南

本文档系统总结了当前可用的所有技能（Skills），按功能分类并说明使用场景。

**当前技能总数：173 个**

---

## 一、论文写作与学术研究（最核心）

### 1.0 深度研究引擎（★★★★★ 最重要）

**`deep-research`** - 通用深度研究 agent 团队

- **核心能力：** 13-agent 流水线，用于任何主题的严谨学术研究
- **8 种研究模式：**
  - `full research` - 完整深度研究
  - `quick brief` - 快速研究简报
  - `paper review` - 论文评审
  - `lit-review` - 文献综述
  - `fact-check` - 事实核查
  - `three-way-scan` - 三路文献扫描（WHY/HOW/WHAT 对比）
  - `paper comparison` - 论文对比
  - `cross-domain synthesis` - 跨领域综合
- **使用场景：** 需要对研究主题进行深入调研时的首选工具

**`academic-pipeline`** - 完整学术研究流程编排器

- **完整流程：** research → write → integrity check → review → revise → re-review → re-revise → final integrity check → finalize
- **协调三大系统：** deep-research + academic-paper + academic-paper-reviewer
- **使用场景：** 从调研到最终发表的全流程自动化

**`academic-paper`** - 12-agent 学术论文写作流水线

- **11 种写作模式：** full/plan/outline/revision/revision-coach/abstract/lit-review/format-convert/citation-check/disclosure/rebuttal-audit
- **6 种论文类型支持**
- **5 种引用风格：** APA, MLA, Chicago, IEEE, Vancouver

**`academic-paper-reviewer`** - 多视角学术论文评审

- **5 位独立评审人：** 主编 + 3 位同行评审 + 魔鬼代言人
- **领域特定专业知识**
- **支持完整评审和快速扫描**

### 1.1 ARS 系列斜杠命令

| 技能命令 | 用途 | 使用场景 |
|---------|------|---------|
| `/ars-full` | 完整论文写作流程 | research → write → review → revise → finalize |
| `/ars-plan` | 论文计划模式 | Socratic 逐章节规划 |
| `/ars-outline` | 大纲模式 | 详细大纲 + 证据地图 |
| `/ars-abstract` | 摘要模式 | 双语摘要 + 关键词 |
| `/ars-lit-review` | 文献综述模式 | 带注释的文献综述 |
| `/ars-revision` | 修订模式 | 修改论文 + R&R 回复 |
| `/ars-revision-coach` | 修订指导 | 修订路线图 + 回复信骨架 |
| `/ars-reviewer` | 同行评审模拟 | 模拟 5 位评审人 |
| `/ars-citation-check` | 引用检查 | 引用错误报告 |
| `/ars-format-convert` | 格式转换 | LaTeX/DOCX/PDF/Markdown |
| `/ars-disclosure` | AI 使用声明 | 期刊要求的 AI 使用声明 |
| `/ars-3w` | 三路文献对比 | WHY/HOW/WHAT 深度对比 |
| `/ars-rebuttal-audit` | 反驳审计 | QA 现有反驳草稿 |
| `/ars-mark-read` | 标记已读 | 记录人工阅读的引用 |
| `/ars-unmark-read` | 取消标记 | 撤销已读标记 |
| `/ars-cache-invalidate` | 缓存失效 | 删除引用缓存条目 |

### 1.2 其他学术写作技能

- `paper-spine`：端到端论文写作（期刊/会议/报告/综述/竞赛），输出 LaTeX/PDF/Word
- `scientific-writing`：科学论文核心写作（完整段落，两阶段流程，从不用项目符号）
- `venue-templates`：访问主流期刊会议 LaTeX 模板（Nature, Science, IEEE, ACM, NeurIPS, ICML, ICLR 等）
- `paperspine`：自动启动 PaperSpine UI（配置缺失时）
- `markdown-mermaid-writing`：Markdown 和 Mermaid 图表写作（科学文档、报告、分析、可视化）

### 1.3 文献检索与管理

- `paper-lookup`：搜索 10 个学术数据库（PubMed, PMC, arXiv, bioRxiv, medRxiv, OpenAlex, Crossref, Semantic Scholar 等）
- `literature-review`：系统性文献综述（多数据库综合检索）
- `citation-management`：引用管理（搜索 Google Scholar/PubMed，提取元数据，生成 BibTeX）
- `pyzotero`：与 Zotero 文献库交互（通过 Python API v3）
- `bgpt-paper-search`：搜索论文并提取结构化实验数据（25+ 字段：方法、结果、样本量等）
- `paperzilla`：与 Paperzilla agent 聊天（项目、推荐、经典论文）

### 1.4 研究辅助工具

- `research-lookup`：当前研究信息查询（parallel-cli 快速搜索、Parallel Chat 深度研究、Perplexity 学术论文搜索）
- `hypothesis-generation`：从观察生成可测试假设（预测、机制、实验设计）
- `scientific-brainstorming`：创意研究构思（跨学科连接、挑战假设、识别研究空白）
- `scientific-critical-thinking`：评估科学声明和证据质量（实验设计、偏差、混淆因素、GRADE 框架）
- `scholar-evaluation`：系统性评估学术工作（ScholarEval 框架：问题阐述、方法论、分析、贡献）
- `open-notebook`：开源 NotebookLM 替代方案（AI-powered 研究和文档分析）
- `dhdna-profiler`：从文本提取认知模式和思维指纹（认知风格、写作模式分析）
- `consciousness-council`：多视角思维委员会审议（决策、创意挑战）
- `hypogenic`：自动化 LLM 驱动假设生成和测试（表格数据模式探索）

### 1.5 同行评审

- `peer-review`：结构化论文/基金评审（清单式评估：方法论、统计有效性、报告标准）
- `ars-reviewer`：模拟多角色同行评审（5 位评审人）

### 1.6 演示与可视化

- `scientific-slides`：研究演讲幻灯片（PowerPoint、会议演讲、研讨会、论文答辩）
- `scientific-schematics`：发表级科学示意图（Nano Banana 2 AI + Gemini 3.1 Pro 质量审查 + 迭代优化）
- `infographics`：专业信息图表（Nano Banana Pro AI + Gemini 3 Pro + 研究数据整合）
- `latex-posters`：LaTeX 研究海报（beamerposter, tikzposter, baposter）
- `pptx-posters`：PowerPoint 研究海报（HTML/CSS，可导出 PDF/PPTX）
- `generate-image`：AI 图像生成（FLUX, Nano Banana 2：照片、插图、艺术、概念图）

### 1.7 其他学术文档类型

- `research-grants`：研究提案写作（NSF, NIH, DOE, DARPA, Taiwan NSTC）
- `clinical-reports`：临床报告（病例报告 CARE 指南、诊断报告、临床试验报告）
- `treatment-plans`：医疗治疗计划（3-4 页，LaTeX/PDF）
- `market-research-reports`：市场研究报告（50+ 页，McKinsey/BCG/Gartner 风格）
- `iso-13485-certification`：ISO 13485 医疗器械质量管理体系认证文档

---

## 二、数据分析与统计

### 2.1 探索性数据分析

- `exploratory-data-analysis`：综合 EDA（支持 200+ 文件格式，理解结构、内容、统计、分布）
- `statistical-analysis`：统计分析指导（检验选择、假设检查、功效分析、APA 格式结果）
- `what-if-oracle`：结构化情景分析（4-6 个分支：最佳、可能、最坏、野卡、反向、二阶）

### 2.2 机器学习

- `scikit-learn`：通用机器学习（监督学习、无监督学习、模型评估、网格搜索）
- `pytorch-lightning`：深度学习框架（LightningModule、多 GPU/TPU 训练、回调、日志）
- `transformers`：Hugging Face Transformers（AutoModel、pipeline 推理、文本生成、Trainer 微调）
- `shap`：模型可解释性（SHAP 值、特征重要性、力图、摘要图）
- `hugging-science`：科学领域 AI/ML（生物、化学、物理、天文、气候、基因组学、材料）

### 2.3 统计建模

- `statsmodels`：统计模型（OLS, GLM, 混合模型, ARIMA，详细诊断、残差、推断）
- `pymc`：贝叶斯建模（层次模型、MCMC NUTS、变分推断、LOO/WAIC 比较）
- `sympy`：精确符号数学（代数、微积分、方程求解、符号线性代数、代码生成）

### 2.4 数据处理框架

- `polars`：快速内存 DataFrame（1-100GB，惰性评估、并行执行、Arrow 后端）
- `polars-bio`：高性能基因组区间操作（Polars DataFrames，BED/VCF/BAM/GFF）
- `vaex`：超大表格数据处理（数十亿行，超出内存，惰性计算）
- `dask`：分布式计算（扩展 pandas/NumPy，跨集群，并行文件处理）
- `zarr-python`：分块 N-D 数组（云存储，Zarr-Python 3，压缩、并行 I/O、S3/GCS）

---

## 三、生物信息学与计算生物学（超过 40 个技能）

### 3.1 基因组学

- `biopython`：分子生物学工具包（序列操作、文件解析 FASTA/GenBank/PDB、系统发育、NCBI Entrez）
- `pysam`：基因组文件工具（读写 SAM/BAM/CRAM、VCF/BCF、FASTA/FASTQ、区域提取、覆盖度）
- `pydeseq2`：差异基因表达分析（Python DESeq2，bulk RNA-seq 计数、Wald 检验、FDR 校正）
- `bulk-rnaseq`：端到端 bulk RNA-seq 编排（FASTQ → QC/trim → 比对/定量 → 基因级计数矩阵）
- `nextflow`：构建/运行 Nextflow 数据流水线（nf-core 工作流、DSL2、进程/通道/操作符）
- `deeptools`：NGS 分析工具（BAM to bigWig、QC、热图/轮廓 TSS/peaks，ChIP-seq/RNA-seq/ATAC-seq）
- `gtars`：高性能基因组区间分析（Rust + Python，BED 文件、覆盖度、重叠检测、ML 标记化）
- `geniml`：基因组区间机器学习（Region2Vec、BEDspace、单细胞 ATAC-seq、scEmbed、scBasset）
- `tiledbvcf`：高效基因组变异数据存储（TileDB、可扩展 VCF/BCF 摄取、并行查询）
- `pacsomatic`：nf-core/pacsomatic 配对肿瘤-正常工作流操作工具（验证输入、生成样本表）

### 3.2 单细胞分析

- `scanpy`：标准单细胞 RNA-seq 分析（QC、归一化、PCA/UMAP/t-SNE、聚类、差异表达）
- `anndata`：单细胞分析注释矩阵数据结构（.h5ad 文件、scverse 生态系统集成）
- `scvi-tools`：单细胞深度生成模型（scVI 概率批次校正、迁移学习、TOTALVI 多模态）
- `scvelo`：RNA 速度分析（未剪接/剪接 mRNA 动态、细胞状态转换、潜在时间、驱动基因）
- `cellxgene-census`：查询 CELLxGENE Census（6100 万+细胞，跨组织/疾病/细胞类型表达数据）

### 3.3 蛋白质与结构生物学

- `esm`：进化尺度蛋白质语言模型（ESM3 生成式多模态设计：序列/结构/功能，ESM C 高效嵌入）
- `pyopenms`：质谱分析平台（蛋白组学工作流、特征检测、肽段鉴定、蛋白定量、LC-MS/MS）
- `molecular-dynamics`：分子动力学模拟（OpenMM, MDAnalysis，蛋白/小分子系统、力场、能量最小化）
- `diffdock`：基于扩散的分子对接（预测蛋白-配体结合姿态、置信度得分、虚拟筛选）
- `glycoengineering`：蛋白糖基化分析（扫描 N-glycosylation sequons、预测 O-glycosylation、糖工程工具）

### 3.4 通路与功能富集

- `pathway-enrichment`：通路和基因集富集分析（运行并解释结果）
- `bioservices`：统一接口访问 40+ 生物信息学服务（UniProt, KEGG, ChEMBL, Reactome，跨数据库分析）
- `gget`：快速 CLI/Python 查询 20+ 生物信息学数据库（基因信息、BLAST、AlphaFold 结构、富集）
- `primekg`：查询精准医学知识图谱（PrimeKG，多尺度生物数据：基因、药物、疾病、表型）

### 3.5 代谢与系统生物学

- `cobrapy`：约束优化代谢建模（COBRA，FBA、FVA、基因敲除、通量采样、SBML 模型）
- `arboreto`：基因调控网络推断（GRNBoost2、GENIE3，转录组数据、bulk/单细胞 RNA-seq）

### 3.6 系统发育

- `phylogenetics`：构建和分析系统发育树（MAFFT 多序列比对、IQ-TREE 2 最大似然、FastTree、ETE3/FigTree 可视化）
- `etetoolkit`：系统发育树工具包（ETE，树操作 Newick/NHX、进化事件检测、同源/旁系同源、NCBI 分类）
- `scikit-bio`：生物数据工具包（序列分析、比对、系统发育树、多样性指标 alpha/beta/UniFrac、PCoA、PERMANOVA）

### 3.7 流式细胞术

- `flowio`：解析 FCS 文件（v2.0-3.1，提取事件为 NumPy 数组、读取元数据/通道、转 CSV/DataFrame）

### 3.8 代谢组学

- `matchms`：光谱相似度和化合物鉴定（代谢组学，比较质谱、余弦相似度、鉴定未知化合物）

### 3.9 显微镜与成像

- `histolab`：轻量级 WSI 切片提取和预处理（基础切片处理、组织检测、切片提取、染色归一化 H&E）
- `pathml`：全功能计算病理学工具包（高级 WSI 分析、多重免疫荧光 CODEX/Vectra、细胞核分割、组织图构建、ML 模型）
- `omero-integration`：显微镜数据管理平台（通过 Python 访问图像、检索数据集、分析像素、管理 ROI/注释、批处理）

### 3.10 医学健康
