# 任务计划书：Base Model 正确预测 API 的逐层涌现机制分析

> 状态：**Step 0/1/2 全部完成**（starcoder2-7b，376 正确样本，逐层 top-k 轨迹 + 跨样本统计 + 4张图 + 案例叙事文档）；产出见 `output/starcoder2_7b/` 和 `docs/CASE_STUDIES.md`。Step 3（组件归因+因果验证）按 §4.4 视情况决定是否纳入。
> 创建日期：2026-06-08

## 0. 背景与定位

之前 `06_mechanism/` 下的 OV 电路 / Logit Lens / Tuned Lens 研究，关注点是"**deprecated API ↔ replacement API 决策对**"——即模型在两个候选之间如何竞争、SFT+DPO 如何改变这个竞争结果。

本任务是机制解释层面的一个**新切入点**：不再预设"两个候选谁赢"，而是只看 **base model 自己就能预测正确的样本**（"正确"不限定于 replacement/deprecated，只要 base model 自己 predict 对了即可），逐层拉出 **top-5 / top-10** 候选，回答"**模型是怎么一步一步把正确的 API 预测出来的**"。

先在 **StarCoder2-7B** 单模型上把方法论跑通、得到初步结论，再推广到其余 6 个 backbone。

---

## 1. 研究问题

> 说明：以下问题是**为本实验新拟定的**，与项目里已有的 `rq1_effectiveness`～`rq4_attention_paths` 没有继承关系——本实验的核心判据只有一条：**模型预测出来的 API 是否就是正确答案（与 ground-truth 一致）**，不再区分"是不是弃用 API"或"是不是最新 API"。下面用描述性标题代替编号，避免与旧 RQ 体系混淆。

- **【涌现层定位】**：在 base model 预测正确的样本中，正确 API token 的 rank / prob 在哪些层开始进入 top-k？在哪一层开始稳定占据 top-1（"饱和"）？是否存在普遍的"涌现层"模式（如集中在中后层）？
- **【候选竞争画像】**：在"涌现层"之前，top-k 候选集合里都有谁？这些"竞争者"是同库的其他 API、形近 token、还是纯语法 token？竞争模式随层数如何演化？
- **【组件归因，扩展项】**：涌现是否可以归因到具体模型组件（特定层的 attention head / MLP）？可否用因果干预（knockout/patch）验证"该层确实是决定性的"？
- **【跨样本模式】**：涌现层 / 竞争画像是否与样本特征（library、API 类别、task_family、数据来源、API 是否已出现在 prompt 中）相关？

---

## 2. 文献调研总结（方法论提炼）

> 调研范围：在 `my_paper/paper2/ref/style_index.md` 与 `cer_dpo_references.md` 已整理的 28 篇摘录基础上（用途是论文写作摘录，不够"方法论精读"），针对本任务主题额外检索并精读了机制可解释性方向的经典与前沿工作。按方法论类型分组如下：

### 2.1 Logit Lens / Tuned Lens —— "逐层读出"预测分布的地基方法

- **nostalgebraist (2020), "interpreting GPT: the logit lens"**（博客，logit lens 的提出）：把每一层残差流通过最终 LayerNorm + unembedding 矩阵投影到词表空间，"偷看"模型在该层当前的预测分布，提出 **iterative inference（迭代推理）** 假说——每一层只对预测做小幅修正，预测是逐层精炼出来的。这正是"逐层 top-k"分析的理论基础。
- **Belrose et al. (2023), "Eliciting Latent Predictions from Transformers with the Tuned Lens"**（arXiv:2303.08112）：在 logit lens 基础上为每层学习一个 affine probe 校正 representational drift（"基变换"），比直接复用 unembedding 更准确地反映"模型在该层真正知道什么"。
  → **`06_mechanism` 已经实现了 `LowRankTunedLens` 并跑过 starcoder2_7b 的 tuned lens 训练，可直接复用**，把本任务的逐层 top-k 同时跑 logit lens 和 tuned lens 两个版本对照。
- **Chen et al. (2026), "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"**（引用 [19]，arXiv:2602.13517）：用逐层预测分布与最终层分布的 **JSD** 来量化"分布何时收敛"，提出 high-JSD token（分布晚收敛）代表更多有效推理步骤。这给"涌现层"提供了一个可操作的量化定义——**收敛速度**而不只是"何时排名第一"。`06_mechanism/src/lens_analysis.py` 里的 `jsd_from_logits` / `decision_jsd_to_final` 已经实现了这个量，可直接复用。

### 2.2 Direct Logit Attribution / 组件归因 —— "是谁在贡献正确预测"

- **Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits"**（Anthropic）：提出 **DLA（Direct Logit Attribution）**——把最终 logits 分解为各层 / 各 attention head / 各 MLP 输出在 unembedding 方向上的投影贡献之和，逐组件排序找到"贡献者"。
  → `06_mechanism/scripts/ov_dla_flip.py`、`ov_dla_multimodel.py` 已实现按 head 分解贡献（目标方向是 `W_U[replacement] - W_U[deprecated]`），本任务可以直接把目标方向换成 **`W_U[correct_token]`**（或 `W_U[correct] - W_U[runner_up]`），复用同一套 DLA 代码做"涌现层"的组件归因验证。
- **Geva et al. (2022), "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space"**（arXiv:2203.14680）：把每个 FFN 子更新单独投影到词表空间，观察它"促进"了哪些概念（即看它的 top-k token 是什么），逐层叠加看预测是如何被"构建"出来的。
  → 这篇论文的方法论与本任务"逐层 top-k"**几乎同构**，是最值得模仿的核心参照：不仅看残差流累积后的 top-k，也可以看单个子层（attention-out / MLP-out）独立贡献了哪些候选，从而把"涌现"细化到子层粒度。

### 2.3 知识定位 / 电路解释 —— "在哪一层、靠什么机制"

- **Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories"** + **Dai et al., "Knowledge Neurons in Pretrained Transformers"**：FFN 是 key-value 记忆，特定神经元对特定概念有选择性触发模式——为"正确 API 是从哪个/哪些神经元被'召回'出来的"提供了下钻方向。
- **Geva et al. (2023), "Dissecting Recall of Factual Associations in Auto-Regressive LMs"**（引用 [13]，EMNLP 2023）：通过对注意力边的干预，揭示了一个三段式机制——早期 MLP 富集主语表征 → 关系位置的注意力传播 → 主语位置的注意力传播 → 最终预测属性 token，并用因果实验逐段验证。
  → 这是"用因果干预把一个逐层观察到的现象，转化成一个可信的多步骤机制故事"的范式样板，本任务"组件归因"环节可以模仿这个"观察 → 假设 → 因果验证"三段式。
- **Meng et al. (2022), "Locating and Editing Factual Associations in GPT"**（ROME）：提出 **causal tracing**——用 corrupt-then-restore（往干净的隐藏状态中"注入"被破坏的版本，再逐层逐位置地恢复，观察哪里恢复后预测就对了）来定位"关键层 / 关键位置"，是定位"决定性层"的经典因果方法，可与本任务的"涌现层"统计观察互相印证。
- **Hanna et al. (2023), "How does GPT-2 compute greater-than?"**：案例研究式电路叙事——把"模型为什么能正确完成某类任务"拆解成一个可读的故事（早期层识别数值表征 → 后期 MLP 完成比较运算）。这是本任务"为什么能一步步预测对"在写作/叙事层面最直接的范式参照：既要有统计聚合，也要有具体样本的"故事"。

### 2.4 ICL / Induction / Function Vectors —— "如何从上下文里复制/泛化出正确答案"

- **Olsson et al. (2022), "In-context Learning and Induction Heads"**：induction head 完成 `[A][B] ... [A] → [B]` 的模式补全。这对代码场景里"从 `import torch` / 函数签名中复制出 API 名"这类样本有直接解释力——很多"正确预测"可能根本不是"知识召回"，而是"从上下文复制"，两者的涌现层模式应该不同，值得在"跨样本模式"分析里区分。
- **Todd et al. (2024), "Function Vectors in Large Language Models"** / **Ghandeharioun et al. (2024), "Patchscopes: A Unifying Framework for Inspecting Hidden Representations"**（arXiv:2401.06102）：用 activation patching 把中间层表示"翻译"成自然语言，或迁移到另一个上下文中，验证"这一层到底已经编码了什么"。比单纯 logit lens 更进一步——可以在"涌现层"附近用 patchscope 式的探测，验证模型在该层是否已经"知道"了正确答案，而不只是该层的 unembedding 投影恰好命中。

### 2.5 Code LLM 专门的可解释性工作

- **Yin et al. (2026), "Neuron-Guided Interpretation of Code LLMs: Where, Why, and How?"**（引用 [25]，FSE 2026）：在神经元层面定位 Code LLM 的 "concept layers"——发现浅层编码语法、深层编码语义，且语义性的 "concept layers" 集中在中后层。
  → 与 `06_mechanism` 现有发现"replacement API 偏好在 layer 17+ 形成"相互印证，本任务若发现"涌现层集中在中后层"，可以与这篇论文的神经元级证据汇聚，互为佐证。

### 2.6 方法论小结（对本任务实验设计的直接启示）

1. **逐层读出（logit lens / tuned lens）是地基**：先把"涌现层"定位出来——正确 token 何时进入 top-k、何时稳定在 top-1，并用 JSD-to-final 曲线刻画"收敛速度"（[19] 的方法）。
2. **top-k 轨迹比单点 rank 更有信息量**：要回答"为什么"，必须看"谁在竞争"——候选集合的演化比单一 win-rate / margin 指标更能讲清楚机制故事（这也是 Geva 2022 [FFN promote concepts] 的核心方法论）。
3. **组件归因（DLA / causal tracing / attention knockout）是验证环节**：统计观察到"涌现层"之后，应该用因果干预（knockout 该层/head，或 patch 该层表示）验证它确实是"决定性"的，而不仅仅是相关的（Geva 2023、Meng 2022 的范式）。
4. **案例研究式叙事 + 跨样本统计并重**：既要有"为什么这条样本在 layer X 突然预测对了"的具体故事（Hanna 2023 范式），也要有跨样本的统计模式（按 library / category / task_family 分组，呼应"跨样本模式"问题）。
5. **"复制" vs "召回"要分开看**：induction-head 式的"从上下文复制"和 knowledge-neuron 式的"参数化知识召回"可能呈现完全不同的涌现层模式，分析时要先区分样本属于哪一类（API 是否已出现在 prompt 上文中）。

---

## 3. 已有基础设施盘点（复用清单）

### 3.1 可直接复用

| 模块/文件 | 用途 | 复用方式 |
|---|---|---|
| `06_mechanism/src/lens_analysis.py::build_model/get_decoder_layers/get_final_norm/get_output_projection/project_hidden_to_logits` | 模型加载、定位各架构的层/输出头/最终归一化、隐藏态→logits 投影 | 直接 import 复用，已验证适配 7 个 backbone |
| `06_mechanism/src/lens_analysis.py::run_hidden_forward` | teacher-forcing 前向 + `output_hidden_states=True` | 直接复用，是逐层分析的统一入口 |
| `06_mechanism/src/lens_analysis.py::gather_token_metrics/gather_single_token_metric` | 给定 logits 和 token id，算 logprob/prob/rank | 直接复用；**本任务需要在此基础上新增一个 `topk_at_layer`，返回 top-k token 列表（id/text/logit/prob/rank）** —— 这是核心新增点 |
| `06_mechanism/src/lens_analysis.py::jsd_from_logits` | 逐层分布与最终层分布的 JSD | 直接复用，用于刻画"收敛速度"（呼应文献 [19]） |
| `06_mechanism/src/lens_analysis.py::MODEL_REGISTRY` | 7 个 backbone 的路径与 max_length 映射 | 直接复用，`starcoder2_7b` 已注册 |
| `06_mechanism/src/lens_analysis.py::LowRankTunedLens/load_tuned_lens` | 已训练好的 tuned lens | 直接复用（如 `output/tuned_lens/starcoder2_7b_official_base.pt` 若存在则直接加载，否则按 README 流程现训） |
| `06_mechanism/README.md` 的输出命名规范 | `*_focus_examples.jsonl`、`*_logit_lens_summary.json`、`run_summary.json` 等 | 本任务延续该命名习惯，便于横向对照和复用 `summarize_*.py` |
| `rq3_token_competition/batch_logit_lens.py` + `OUTPUT_FORMAT.md` | 已实现"逐层 top-500 中匹配特定 API 列表"的 ranking 输出（`api_ranking_by_layer` 字段，含 token_text/token_id/rank/probability/logit） | **是"逐层 top-k"最直接的历史先例**，本任务的 `*_topk_trajectory.jsonl` schema 可参照其设计，但要泛化成"任意正确 token"而不限定于预先给定的 API 列表 |
| `06_mechanism/scripts/ov_dla_flip.py`、`ov_dla_multimodel.py` | DLA：按 head 分解对 logit 方向的贡献 | "组件归因"阶段复用，目标方向从 `W_U[replacement]-W_U[deprecated]` 换成 `W_U[correct_token]` |
| `rq4_attention_paths/attn_knockout.py` | attention head/layer knockout | "组件归因"阶段复用，做"涌现层是否决定性"的因果验证 |

### 3.2 需要新写/扩展

- **"base model 预测正确"的判定与筛选逻辑**（目前没有现成实现——已有的 `compare_focus_example` 是围绕 replacement vs deprecated 的二元对比，本任务需要的是更通用的"模型 teacher-forcing / greedy 解码是否命中 ground truth API"判定）
- **逐层 top-k 提取与轨迹聚合**（`layerwise_topk_trace`，结构上类似 `layerwise_candidate_trace`，但输出的是每层的 top-k 候选列表，而不是单一 token 的 metric）
- **涌现层定位指标的计算与聚合**：`emergence_layer`（首次进入 top-k 的层）、`saturation_layer`（首次成为 top-1 且此后保持 top-1 的层）、收敛曲线（JSD-to-final）
- **跨样本模式归纳与案例选择脚本**：按 emergence_layer 分桶、按 library/category/task_family 分组统计，并自动挑选有代表性的样本做案例叙事

---

## 4. 实验设计

### 4.1 Step 0：数据与"正确"样本筛选

**候选样本来源（两路并行，都在第一阶段做）**：
1. **repair/consistency 测试集**：复用 `05_positive_engineering/data/processed_clean/{repair,consistency}_sft_test.jsonl`——与 `06_mechanism` 现有分析口径一致，便于横向对照。
2. **EDAPIBench（现在就构造 starcoder2-7b 子集）**：`adalora/edapibench-src/EDAPI-Bench-main/data/EditDeprecatedAPI/*/all.json` 含 `probing input` / `reference` / `expected call` / `category`（已区分 outdated / up-to-dated）/ `deprecated api` / `replacement api` 等字段。
   - EDAPIBench 现成的 `probing predictions` 是别的模型（codegemma-2b / qwencoder-3b / deepseek-1.3b）跑出来的，starcoder2-7b 没有现成子集——但**不需要依赖它**：直接拿 EDAPIBench 的 `probing input`（prompt）+ `reference`/`expected call`（ground truth）这一对"prompt+gt"，喂给 starcoder2-7b 做 teacher-forcing/greedy 解码，跑一遍与 repair/consistency 完全相同的"正确性判定"流程（见下），即可自行构造出"starcoder2-7b 在 EDAPIBench 上的正确预测子集"——**这一步本质上就是 Step 0 选样流程本身，并不是一个独立的高成本"重新构造数据集"任务**，只是多了一路输入源。
   - 这样 EDAPIBench 子集和 repair/consistency 子集的构造方式完全统一（同一套"模型预测是否等于 ground truth"判据），且 EDAPIBench 的 `category` 天然覆盖 outdated / up-to-dated 两类，能直接验证"正确不区分弃用或最新"这一设定。
   - 输出文件按来源区分前缀：`output/starcoder2_7b/{source}_correct_examples.jsonl`，其中 `source ∈ {repair, consistency, edapibench}`，最终聚合分析时再合并/对比。

**"正确"判定标准**（不限定 replacement/deprecated，只要 base model 自己预测对就算）：
- 用 base model 在 `version_prompt`（或 `probing_input`）上做 teacher-forcing，对照 `target`/`reference` 中的 API 调用span（含别名形式，复用 `alias_forms`/`first_alias_hit`）
- 判定粒度建议两级都做：
  1. **首 token 正确**（最严格——直接定位到"决策 token"的涌现，与现有 `decision_*` 指标口径一致）
  2. **完整 API span 正确**（与现有 `sequence_logprob`/`replacement_wins_sequence` 口径对齐，覆盖多 token API 名）
- 输出：
  - `output/starcoder2_7b/correct_examples.jsonl`（命中样本子集，附带正确判定的具体证据：matched token/span、rank、logprob）
  - `output/starcoder2_7b/selection_summary.json`（命中率统计：整体 + 按 library / category / task_family 分组，呼应"跨样本模式"问题）

### 4.2 Step 1（核心）：逐层 top-k 轨迹提取

对每个"正确"样本，在决策位置做 teacher-forcing 前向，对每一层：
- 投影到词表空间（复用 `project_hidden_to_logits`，同时跑 logit lens 和 tuned lens 两个变体）
- 提取 **top-k**（k=5、10，并保留一份 k=50 用于更细粒度的竞争分析）候选：`token_id`/`token_text`/`logit`/`prob`/`rank`
- 记录正确 token 自身的 rank/prob/logprob 轨迹，以及该层分布与最终层分布的 JSD（呼应 [19]）

**输出**：
- `output/starcoder2_7b/topk_trajectory.jsonl`（每条样本 × 每层的 top-k 列表 + 正确 token 轨迹，schema 参照 `rq3_token_competition/OUTPUT_FORMAT.md` 的 `api_ranking_by_layer` 设计但泛化）
- `output/starcoder2_7b/emergence_summary.json`（聚合统计）

**关键聚合指标**：
- **emergence_layer**：正确 token 首次进入 top-k 的层
- **saturation_layer**：正确 token 首次成为 top-1 且之后保持 top-1 的层（用 JSD-to-final 收敛曲线辅助定义/交叉验证）
- **competitor profile**：在 emergence_layer 之前，top-k 中稳定出现的"竞争 token"是谁——是同库的其他 API？是形近 token（共享前缀）？还是纯语法 token（如 `.`/`(`等）？

### 4.3 Step 2：案例研究 + 跨样本模式归纳

- 按 `emergence_layer` 分桶（浅层涌现 vs 中层涌现 vs 深层涌现），从每个桶中挑选若干代表性样本做案例叙事（模仿 Hanna et al. 2023 的电路叙事范式：具体讲清楚"这条样本在哪一层、因为什么候选退出/进入，预测开始变对"）
- 跨样本统计：`emergence_layer`/`saturation_layer` 的分布是否与下列因素相关——
  - library（不同库的 API 命名风格/长度是否影响涌现速度）
  - category（outdated vs up-to-dated：模型对"已弃用"和"未弃用"API 的涌现模式是否不同）
  - task_family（repair vs consistency）
  - **API 是否已经出现在 prompt 上文中**（用于区分"复制"型涌现 vs "召回"型涌现，呼应 induction head 文献）

### 4.4 Step 3（扩展，视第一阶段结论决定是否纳入）：组件归因与因果验证

- 用 DLA（复用 `ov_dla_flip.py` 思路，目标方向换成 `W_U[correct_token]` 或 `W_U[correct] - W_U[runner_up]`）定位涌现层中贡献最大的 attention head / MLP
- 用 attention knockout（复用 `rq4_attention_paths/attn_knockout.py`）做因果验证：敲除候选层/head 后，"正确预测"的命中率是否显著下降，从而把"涌现层"从"观察到的相关性"升级为"验证过的因果性"

### 4.5 实验范围与推广路径

- **第一阶段**：仅 StarCoder2-7B，候选样本来自 repair/consistency 测试集 + 自行构造的 EDAPIBench-starcoder2-7b 子集（两路并行，§4.1），先把"筛选 → top-k 轨迹 → 聚合统计 → 案例叙事"全流程跑通，得到初步结论（重点回答"涌现层定位"与"候选竞争画像"两个问题）
- **第二阶段**（依赖第一阶段结论）：推广到其余 6 个 backbone，验证涌现模式的跨模型一致性（同 series 内 vs 跨 series），并视情况展开"组件归因"与"跨样本模式"的完整跨模型验证

---

## 5. 输出产物与目录/命名规范

延续 `06_mechanism` 的目录与命名习惯：

```
07_correct_emergence/
  PLAN.md                                <- 本计划书
  src/
    correct_selection.py                 <- "正确预测"判定与筛选逻辑
    topk_trace.py                        <- 逐层 top-k 提取、聚合（扩展 lens_analysis.py 的能力）
  scripts/
    select_correct_examples.py           <- Step 0 入口
    run_topk_trajectory.py               <- Step 1 入口
    summarize_emergence.py               <- Step 2 聚合统计
    plot_emergence_layers.py             <- 可视化（涌现层分布直方图、竞争画像热力图等）
  output/
    smoke_starcoder2_7b/                 <- 先用 ≤20 条样本跑通全流程（smoke test）
    starcoder2_7b/
      correct_examples.jsonl
      selection_summary.json
      topk_trajectory.jsonl
      emergence_summary.json
      run_summary.json
```

---

## 6. 风险与注意事项

- **GPU 占用**：写本计划书时（2026-06-08）`nvidia-smi` 显示卡 1/2/3/7 空闲（0% 利用率，显存几乎全空），卡 0 占用约 8.7GB，卡 4/5 各占约 64GB，**卡 6 利用率 100%**——开跑前务必再次用 `nvidia-smi` 确认，优先用 1/2/3/7 号空闲卡，避免重蹈"和别人共用卡训练崩溃"的覆辙。
- **批量脚本健壮性**：涉及按库分组统计/批跑时，库名硬编码为 `LIBRARIES=(numpy pandas pytorch scipy seaborn sklearn tensorflow transformers)`，不要依赖运行时 `ls`/`find` 的返回值（历史上因 exit code 255 反复崩溃过）。
- **先 smoke test 再全量跑**：先用 ≤20 条样本跑通"筛选 → top-k 轨迹 → 聚合 → 可视化"整条流水线（参照 `06_mechanism/output/smoke_starcoder2_3b` 的产出形态），确认输出 schema、聚合逻辑、可视化都没问题后再全量跑、再推广到其他模型。
- **输出体量**：top-k=50、逐层、全量样本会产生较大的 jsonl，建议默认只持久化 top-10（分析用）+ 正确 token 自身轨迹，top-50 仅在按需做"竞争分析"时临时跑小样本子集。

---

## 7. 与用户确认后的执行口径（已拍板，正式生效）

> 以下 4 项在汇报后与用户逐一确认，结论已写回上面对应章节，作为开始写代码前的最终口径：

1. **"正确"判定粒度**——✅两种都做：首 token 正确（精确定位"决策 token"涌现层）+ 完整 API span 正确（与现有 `sequence_logprob` 口径对齐），分别筛出子集、分别分析（见 §4.1）。
2. **EDAPIBench**——✅现在就构造 starcoder2-7b 子集：不依赖其现成的 `probing predictions`（那是别的模型跑的），而是直接用它的 `probing input`（prompt）+ `reference`/`expected call`（ground truth）作为"prompt+gt"输入，复用 Step 0 的同一套"模型预测是否等于 ground truth"判定流程自行构造——成本等同于多跑一路输入源，而不是独立的高成本数据集构造任务（见 §4.1 已更新）。EDAPIBench 的 `category` 字段天然覆盖 outdated/up-to-dated 两类，正好印证"正确不区分弃用或最新"这一核心设定。
3. **top-k 的 k 值**——✅5/10 为主、全量持久化用于涌现层定位与聚合统计；k=50 仅在需要做"候选竞争"细粒度分析时按需对感兴趣的子集/案例小批量跑（见 §4.2、§6 输出体量提示）。
4. **Step 3（组件归因 / 因果验证）**——✅第一阶段不纳入，先把"涌现层定位 + 候选竞争画像"（统计观察 + 案例叙事）跑完、出初步结论，再决定是否值得投入做 DLA / attention knockout 因果验证（见 §4.4、§4.5）。

补充说明（用户在确认时强调，已落实到全文措辞中）：本实验是**全新切入点**，与项目里 `rq1_effectiveness`～`rq4_attention_paths` 的既有研究问题没有继承关系；唯一判据是"模型预测的 API 是否等于 ground truth"，不再以"是否弃用/是否最新"来框定样本范围——这也是 EDAPIBench 可以被直接拿来用、且需要自行按 starcoder2-7b 重新构造子集的原因。
