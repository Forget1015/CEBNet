# Claude Code 技能完整使用指南

**技能总数：173 个** | 更新时间：2026-07-01

---

## 核心学术研究技能（论文写作必备）

### ★★★★★ 最重要的 4 个编排器

1. **`deep-research`** - 13-agent 深度研究流水线
   - 8 种模式：full/quick-brief/paper-review/lit-review/fact-check/three-way-scan/paper-comparison/cross-domain-synthesis
   
2. **`academic-pipeline`** - 完整学术流程编排
   - research → write → integrity-check → review → revise → finalize
   
3. **`academic-paper`** - 12-agent 论文写作流水线
   - 11 种模式：full/plan/outline/revision/abstract/lit-review/format-convert/citation-check 等
   
4. **`academic-paper-reviewer`** - 多视角同行评审
   - 5 位评审人：主编 + 3 评审 + 魔鬼代言人

### ARS 系列命令（15 个）

- `/ars-full` - 完整论文流程
- `/ars-plan` - Socratic 章节规划
- `/ars-outline` - 大纲+证据地图
- `/ars-abstract` - 双语摘要
- `/ars-lit-review` - 文献综述
- `/ars-revision` - 修订+R&R回复
- `/ars-revision-coach` - 修订指导
- `/ars-reviewer` - 同行评审模拟
- `/ars-citation-check` - 引用检查
- `/ars-format-convert` - 格式转换
- `/ars-disclosure` - AI使用声明
- `/ars-3w` - WHY/HOW/WHAT对比
- `/ars-rebuttal-audit` - 反驳审计
- `/ars-mark-read` / `/ars-unmark-read` - 标记引用
- `/ars-cache-invalidate` - 缓存失效

### 其他学术写作（10 个）

- `paper-spine` - 端到端论文写作
- `scientific-writing` - 科学论文核心写作
- `venue-templates` - 期刊会议模板
- `paperspine` - PaperSpine UI
- `markdown-mermaid-writing` - Markdown图表
- `research-grants` - 研究提案（NSF/NIH）
- `clinical-reports` - 临床报告
- `treatment-plans` - 治疗计划
- `market-research-reports` - 市场研究报告
- `iso-13485-certification` - ISO认证文档

### 文献检索（7 个）

- `paper-lookup` - 搜索10个学术数据库
- `literature-review` - 系统性文献综述
- `citation-management` - 引用管理+BibTeX
- `pyzotero` - Zotero API
- `bgpt-paper-search` - 结构化实验数据提取
- `paperzilla` - Paperzilla聊天
- `database-lookup` - 78个科学数据库

### 研究辅助（9 个）

- `research-lookup` - 研究信息查询
- `hypothesis-generation` - 假设生成
- `scientific-brainstorming` - 科研头脑风暴
- `scientific-critical-thinking` - 批判性思维
- `scholar-evaluation` - ScholarEval评估
- `open-notebook` - NotebookLM替代
- `dhdna-profiler` - 认知模式提取
- `consciousness-council` - 多视角审议
- `hypogenic` - 自动假设生成测试

### 同行评审（2 个）

- `peer-review` - 结构化评审
- `ars-reviewer` - 多角色模拟

### 演示可视化（6 个）

- `scientific-slides` - 研究PPT
- `scientific-schematics` - 科学示意图
- `infographics` - 信息图表
- `latex-posters` - LaTeX海报
- `pptx-posters` - PowerPoint海报
- `generate-image` - AI图像生成

---

## 数据分析与统计（13 个）

### 探索性分析
- `exploratory-data-analysis` - 综合EDA（200+格式）
- `statistical-analysis` - 统计分析指导
- `what-if-oracle` - 情景分析

### 机器学习
- `scikit-learn` - 通用ML
- `pytorch-lightning` - 深度学习
- `transformers` - Hugging Face
- `shap` - 模型可解释性
- `hugging-science` - 科学领域AI

### 统计建模
- `statsmodels` - 统计模型
- `pymc` - 贝叶斯建模
- `sympy` - 符号数学

### 数据处理
- `polars` - 快速DataFrame
- `vaex` - 超大数据

---

## 生物信息学（45+ 个）

### 基因组学（10个）
- `biopython`, `pysam`, `pydeseq2`, `bulk-rnaseq`
- `nextflow`, `deeptools`, `gtars`, `geniml`
- `tiledbvcf`, `pacsomatic`

### 单细胞（5个）
- `scanpy`, `anndata`, `scvi-tools`
- `scvelo`, `cellxgene-census`

### 蛋白质（5个）
- `esm`, `pyopenms`, `molecular-dynamics`
- `diffdock`, `glycoengineering`

### 通路富集（4个）
- `pathway-enrichment`, `bioservices`
- `gget`, `primekg`

### 代谢系统（2个）
- `cobrapy`, `arboreto`

### 系统发育（3个）
- `phylogenetics`, `etetoolkit`, `scikit-bio`

### 流式细胞术（1个）
- `flowio`

### 代谢组学（1个）
- `matchms`

### 显微镜（3个）
- `histolab`, `pathml`, `omero-integration`

### 医学健康（3个）
- `pyhealth`, `scikit-survival`, `neurokit2`

### 神经科学（2个）
- `neuropixels-analysis`, `bids`

### 医学影像（3个）
- `pydicom`, `imaging-data-commons`, `pathml`

### 实验室自动化（7个）
- `opentrons-integration`, `pylabrobot`
- `ginkgo-cloud-lab`, `benchling-integration`
- `protocolsio-integration`, `labarchive-integration`
- `lamindb`

### 平台集成（3个）
- `latchbio-integration`, `dnanexus-integration`
- `benchling-integration`

---

## 化学与药物发现（13 个）

### 化学信息学
- `rdkit`, `datamol`, `molfeat`

### 药物发现
- `deepchem`, `pytdc`, `torchdrug`, `medchem`

### 分子建模
- `rowan`, `diffdock`, `depmap`

### 其他
- `adaptyv` - Adaptyv Bio Foundry API

---

## 文档处理（6 个）

- `pdf` - PDF处理
- `docx` - Word文档
- `xlsx` - Excel表格
- `pptx` - PowerPoint
- `markitdown` - 转Markdown
- `liteparse` - 本地解析

---

## 可视化（5 个）

- `matplotlib` - 底层绘图
- `seaborn` - 统计可视化
- `networkx` - 网络图
- `scientific-visualization` - 发表级图表
- `generate-image` - AI图像

---

## Web搜索（4 个）

- `research-lookup` - 研究查询
- `parallel-web` - 学术web工具
- `exa-search` - 科技内容搜索
- `database-lookup` - 78个数据库

---

## 时间序列（2 个）

- `aeon` - 时间序列ML
- `timesfm-forecasting` - 零样本预测

---

## 强化学习（2 个）

- `stable-baselines3` - 生产级RL
- `pufferlib` - 高性能RL

---

## 量子计算（4 个）

- `qiskit` - IBM
- `cirq` - Google
- `pennylane` - 硬件无关
- `qutip` - 开放量子系统

---

## 地理空间（2 个）

- `geopandas` - 矢量数据
- `geomaster` - 遥感GIS

---

## 材料科学（1 个）

- `pymatgen` - 材料工具包

---

## 仿真（2 个）

- `simpy` - 离散事件仿真
- `fluidsim` - 流体力学

---

## 其他（6 个）

- `matlab` - MATLAB/Octave
- `usfiscaldata` - 美国财政数据
- `torch-geometric` - 图神经网络
- `dask` - 分布式计算
- `zarr-python` - 分块数组
- `modal` - 无服务器云平台

---

## 系统配置（6 个）

- `update-config` - 配置Claude Code
- `keybindings-help` - 键盘快捷键
- `simplify` - 代码重构
- `fewer-permission-prompts` - 减少权限提示
- `get-available-resources` - 检测系统资源
- `optimize-for-gpu` - GPU加速

---

## 其他工具技能（7 个）

- `loop` - 定期循环任务
- `claude-api` - Claude API开发
- `init` - 初始化CLAUDE.md
- `review` - PR审查
- `security-review` - 安全审查
- `autoskill` - 观察工作流自动创建技能
- `paper-lookup` - 论文查找

---

## 快速查找表

### 我要写论文 →
1. `/ars-plan` 规划结构
2. `deep-research` 深度调研
3. `/ars-full` 完整写作
4. `/ars-reviewer` 模拟评审
5. `/ars-revision` 修订

### 我要做数据分析 →
1. `exploratory-data-analysis` 先EDA
2. `statistical-analysis` 选检验
3. `scikit-learn` 建模
4. `matplotlib`/`seaborn` 可视化

### 我要做生物信息 →
- RNA-seq: `bulk-rnaseq`
- 单细胞: `scanpy`
- 富集: `pathway-enrichment`
- 序列: `biopython`

### 我要处理文档 →
- PDF: `pdf`
- Word: `docx`
- Excel: `xlsx`
- PPT: `pptx`

### 我要搜文献 →
- `paper-lookup` 搜索
- `literature-review` 综述
- `citation-management` 引用管理

---

## 使用方式

```bash
# 方式1：斜杠命令
/ars-full

# 方式2：对话触发
"用 deep-research 帮我调研 transformer 的最新进展"
"用 paper-lookup 搜索关于推荐系统的论文"
"用 scientific-slides 生成一个组会PPT"
```

---

**总结：173 个技能，涵盖学术研究全流程**
