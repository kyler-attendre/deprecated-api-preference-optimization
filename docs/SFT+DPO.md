# 面向弃用API的DPO与机制分析，

## 1. 摘要

**Plan A（SFT LoRA）**通过参数高效微调强化模型对显式版本信息的响应能力，使模型在版本上下文中更稳定地抑制已弃用 API，并提高替换率。

**Plan  B（DPO / SFT+DPO）**进一步使用 preference optimization 检验是否可以在保持通用代码能力的同时控制 API 选择。DPO 能显著压低已弃用 API，但在 StarCoder2 上往往降低替换率，表现为“避免旧 API”而不是“生成新 API”。加入 replacement API CE anchor 后，SFT+DPO 同时实现已弃用抑制和替换 API 诱导。

## 2. 数据集构建

数据源 1 为 `bad2good`，对应模型在无版本信息输入下生成已弃用 API、在加入版本信息后生成新版本 API 的样本。

数据源 2 为 `version-control-results-version` 中面向新版本语境的 `up-to-dated` 样本。

`bad2good` 共包含 513 条样本，其中 `outdated` 260 条、`up-to-dated` 253 条。

`version_results` 中原始 `up-to-dated` 样本为 73,360 条，清洗后保留 **72,444 条**。

### 2.1 样本类别

- `repair`：来自 `outdated bad2good`，用于训练模型将已弃用 API 修正为替换 API，共 260 条。
- `consistency`：来自 `up-to-dated bad2good`，用于训练模型在应使用新 API 的语境下保持版本一致性，共 253 条。
- `reference`：来自 `version_results up-to-dated`，提供大规模、稳定的新版本参考答案，共 72,444 条。

### 2.2 `mixed_sft_v1`：

去重后，`mixed_sft_v1` 由以下三部分构成：

1. 全部 `repair` 样本：248 条。
2. 全部 `consistency` 样本：249 条。
3. 从 `reference` 中采样：2,500 条。

由于 `reference_sft` 规模远大于前两类样本，若全量加入，训练过程可能被 reference 样本主导，从而削弱模型对关键修复行为的学习。

最终 `mixed_sft_v1` 的规模为**训练 2,408、验证 304、测试 285**，其中测试集由 `repair` 14 条、`consistency` 21 条和 `reference` 250 条组成。

## 3. 方法

### 3.1 评价指标

- **弃用率**：预测中命中任一已弃用 API 的样本比例（↓）
- **替换率**：预测中命中任一替换 API 的样本比例（↑）

### 3.2 Plan A：SFT LoRA

每条训练记录将带有显式版本前缀的 `version_prompt` 与目标版本一致的 `target` 补全配对，模型仅在响应 token 上计算交叉熵损失，版本前缀不参与损失计算。LoRA Adapter作用于注意力投影矩阵 `q_proj`、`k_proj`、`v_proj` 和 `o_proj`。

| 主干模型 | 训练/验证 | epoch | lr | max_length | LoRA Rank | Alpha | Dropout |
|---|---:|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 2,408 / 304 | 3 | 1e-4 | 384 | 16 | 32 | 0.05 |
| StarCoder2-7B | 2,408 / 304 | 3 | 1e-4 | 384 | 16 | 32 | 0.05 |
| StarCoder2-15B | 2,408 / 304 | 3 | 1e-4 | 256 | 16 | 32 | 0.05 |

### 3.3 Plan B：DPO 与 SFT+DPO（DPO-CE）

DPO的preference pair 来自目标版本一致 completion 与合成 deprecated completion；只使用 pairwise preference loss；

DPO 设置为：

- `chosen`：包含 replacement API 的目标版本一致补全；
- `rejected`：将 `chosen` 中 replacement API 替换为 deprecated API 后得到的合成负样本；
- DPO `beta = 0.1`，sequence log-prob reduction 为 `sum`；
- LoRA rank `8`，alpha `16`，dropout `0.05`；
- 训练 1 epoch，learning rate `5e-5`。

**但实验发现仅使用 pairwise preference loss 容易产生 deprecated avoidance，而非 replacement induction。**



为补充正向 replacement 信号，在 DPO loss 上加入轻量 CE anchor：

**SFT+DPO：在 DPO 的偏好损失基础上，以 replacement API token 的交叉熵损失作为正则化项，抑制模型只学回避而不学生成**

$$\mathcal{L} = \mathcal{L}_{\text{DPO}}(y_w, y_l) + 0.1\cdot\mathcal{L}_{\text{CE}}(\text{replacement API tokens})$$​

## 4. 实验结果

### 4.1 SFT LoRA 

#### 4.1.1 SFT LoRA 在StarCoder2 上的详细结果

| 模型           | 样本数 | Baseline 弃用率 | LoRA 弃用率 | Δ弃用率（绝对 / 相对） | Baseline 替换率 | LoRA 替换率 | Δ替换率（绝对 / 相对） |
| -------------- | -----: | --------------: | ----------: | ---------------------: | --------------: | ----------: | ---------------------: |
| StarCoder2-3B  |    285 |           8.42% |       1.40% |       -7.02% / ↓83.37% |          11.58% |      56.84% |     +45.26% / ↑390.85% |
| StarCoder2-7B  |    285 |          11.23% |       1.40% |       -9.83% / ↓87.53% |          12.63% |      58.25% |     +45.62% / ↑361.20% |
| StarCoder2-15B |    285 |          10.53% |       1.05% |       -9.48% / ↓90.03% |          14.39% |      52.63% |     +38.24% / ↑265.74% |

| 模型           | 子集        | 样本数 | Baseline 弃用率 | LoRA 弃用率 | Δ弃用率（绝对 / 相对） | Baseline 替换率 | LoRA 替换率 | Δ替换率（绝对 / 相对） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B  | repair      |     14 |          35.71% |      14.29% |      -21.43% / ↓59.98% |           7.14% |     100.00% |    +92.86% / ↑1300.56% |
| StarCoder2-3B  | consistency |     21 |          52.38% |       0.00% |     -52.38% / ↓100.00% |          28.57% |      95.24% |     +66.67% / ↑233.36% |
| StarCoder2-3B  | reference   |    250 |           3.20% |       0.80% |       -2.40% / ↓75.00% |          10.40% |      51.20% |     +40.80% / ↑392.31% |
| StarCoder2-7B  | repair      |     14 |          57.14% |      14.29% |      -42.86% / ↓74.99% |          28.57% |      92.86% |     +64.29% / ↑225.03% |
| StarCoder2-7B  | consistency |     21 |          57.14% |       0.00% |     -57.14% / ↓100.00% |          23.81% |      95.24% |     +71.43% / ↑300.00% |
| StarCoder2-7B  | reference   |    250 |           4.80% |       0.80% |       -4.00% / ↓83.33% |          10.80% |      53.20% |     +42.40% / ↑392.59% |
| StarCoder2-15B | repair      |     14 |          35.71% |      14.29% |      -21.43% / ↓59.98% |          21.43% |     100.00% |     +78.57% / ↑366.64% |
| StarCoder2-15B | consistency |     21 |          47.62% |       0.00% |     -47.62% / ↓100.00% |          33.33% |      85.71% |     +52.38% / ↑157.16% |
| StarCoder2-15B | reference   |    250 |           6.00% |       0.40% |       -5.60% / ↓93.33% |          12.40% |      47.20% |     +34.80% / ↑280.65% |

子集结果表明，改进并非主要由大规模 `reference` 子集驱动；在 `repair` 和 `consistency` 子集上，LoRA 同样显著降低弃用率并提高替换率。

####  4.1.2  SFT LoRA 结果

| 模型                         | Baseline 弃用率 | LoRA 弃用率 | Δ弃用率（绝对 / 相对） | Baseline 替换率 | LoRA 替换率 | Δ替换率（绝对 / 相对） |
| ---------------------------- | --------------: | ----------: | ---------------------: | --------------: | ----------: | ---------------------: |
| StarCoder2-3B                |           8.42% |       1.40% |       -7.02% / ↓83.37% |          11.58% |      56.84% |     +45.26% / ↑390.85% |
| StarCoder2-7B                |          11.23% |       1.40% |       -9.83% / ↓87.53% |          12.63% |      58.25% |     +45.62% / ↑361.20% |
| StarCoder2-15B               |          10.53% |       1.05% |       -9.48% / ↓90.03% |          14.39% |      52.63% |     +38.24% / ↑265.74% |
| DeepSeek-Coder-6.7B-Instruct |           6.67% |       0.70% |       -5.97% / ↓89.51% |          24.91% |      54.74% |     +29.83% / ↑119.75% |
| Qwen2.5-Coder-3B-Instruct    |           8.07% |       0.70% |       -7.37% / ↓91.33% |          19.30% |      55.09% |     +35.79% / ↑185.44% |
| Qwen2.5-Coder-7B-Instruct    |           5.96% |       1.05% |       -4.91% / ↓82.38% |          24.21% |      52.28% |     +28.07% / ↑115.94% |
| Qwen2.5-Coder-14B-Instruct   |           5.96% |       0.70% |       -5.26% / ↓88.26% |          27.37% |      46.67% |      +19.30% / ↑70.52% |

版本感知微调在全部模型上均实现了已弃用抑制和替换率提升。

### 4.2 DPO/SFT+DPO

#### 4.2.1 DPO （仅pairwise preference loss）

| 模型                         | Baseline 弃用率 | DPO 弃用率 | Δ弃用率（绝对 / 相对） | Baseline 替换率 | DPO 替换率 | Δ替换率（绝对 / 相对） |
| ---------------------------- | --------------: | ---------: | ---------------------: | --------------: | ---------: | ---------------------: |
| StarCoder2-3B                |           8.42% |      0.00% |      -8.42% / ↓100.00% |          11.58% |      3.86% |       -7.72% / ↓66.67% |
| StarCoder2-7B                |          11.23% |      0.00% |     -11.23% / ↓100.00% |          12.63% |      8.42% |       -4.21% / ↓33.33% |
| StarCoder2-15B               |          10.53% |      0.00% |     -10.53% / ↓100.00% |          14.39% |      8.07% |       -6.32% / ↓43.92% |
| DeepSeek-Coder-6.7B-Instruct |           6.67% |      1.05% |       -5.62% / ↓84.26% |          24.91% |     28.42% |       +3.51% / ↑14.09% |
| Qwen2.5-Coder-7B-Instruct   |           5.96% |      0.35% |       -5.61% / ↓94.12% |          24.21% |     14.74% |       -9.47% / ↓39.13% |

仅 DPO 的结论很明确：它能有效抑制弃用率，但在 StarCoder2 系列上替换率反而下降。因此，该DPO 更是接近deprecated-avoidance ，而不是可靠的 replacement-induction 。

#### 4.2.2 SFT+DPO（添加CE Anchor）（例子）

为修复这一问题，后续在 StarCoder2-7B 上筛选了 API-span DPO、API-span DPO + anchor、Full DPO + anchor 等变体。

> 1. DPO
>    - dpo_scope=full；api_anchor_weight=0；整段补全做 DPO，不加额外监督
> 2. API-span DPO
>    - dpo_scope=api_span；api_anchor_weight=0；只在 replacement / deprecated API 那个局部 span 上做 DPO
> 3. API-span DPO + anchor
>    - dpo_scope=api_span；api_anchor_weight=0.1；局部 API span 做 DPO，同时显式监督模型生成replacement API token
> 4. Full DPO + anchor
>    - dpo_scope=full；api_anchor_weight=0.1；整段 completion 做 DPO，同时加 replacement API token 的 anchor 监督

筛选结果表明，`Full DPO + replacement API CE anchor, api_anchor_weight = 0.1` 在 7B 上最好：弃用率从 11.23% 降到 0.00%，替换率从 12.63% 升到 63.51%，同时 HumanEval 仅下降 0.61%，MBPP 上升 2.00%。

SFT+DPO 随后扩展到 StarCoder2、DeepSeek 和 Qwen。结果如下。

| 模型                         | Baseline 弃用率 | SFT+DPO 弃用率 | Δ弃用率（绝对 / 相对） | Baseline 替换率 | SFT+DPO 替换率 | Δ替换率（绝对 / 相对） |
| ---------------------------- | --------------: | ------------------: | ---------------------: | --------------: | ------------------: | ---------------------: |
| StarCoder2-3B                |           8.42% |               0.35% |       -8.07% / ↓95.84% |          11.58% |              57.54% |     +45.96% / ↑396.89% |
| StarCoder2-7B                |          11.23% |               0.00% |     -11.23% / ↓100.00% |          12.63% |              63.51% |     +50.88% / ↑403.01% |
| StarCoder2-15B               |          10.53% |               0.70% |       -9.82% / ↓93.33% |          14.39% |              62.46% |     +48.07% / ↑334.15% |
| DeepSeek-Coder-6.7B-Instruct |           6.67% |               0.70% |       -5.97% / ↓89.51% |          24.91% |              48.42% |      +23.51% / ↑94.38% |
| Qwen2.5-Coder-3B-Instruct    |           8.07% |               0.35% |       -7.72% / ↓95.66% |          19.30% |              56.14% |     +36.84% / ↑190.88% |
| Qwen2.5-Coder-7B-Instruct    |           5.96% |               0.00% |      -5.96% / ↓100.00% |          24.21% |              59.30% |     +35.09% / ↑144.94% |
| Qwen2.5-Coder-14B-Instruct   |           5.96% |               0.70% |       -5.26% / ↓88.26% |          27.37% |              54.39% |      +27.02% / ↑98.72% |

结果，SFT+DPO 将弃用率从 8.12% 降至 0.40%，并将替换率从 19.20% 提升至 57.39%。这说明 CE anchor 成功解决了仅DPO 的失败模式：模型不再只是回避 deprecated API，而是稳定转向 replacement API。

- SFT LoRA：证明版本一致行为可以通过监督微调强力注入，但存在通用能力代价；
- DPO：证明 preference tuning 能保持能力并抑制 deprecated API，但正向 replacement 信号不足；
- SFT+DPO：

#### 4.2.3 替换率总体结果

| 模型                         | 方法         | 弃用率 | 替换率 | 其他 |
| ---------------------------- | ------------ | ---------: | ----------: | -----: |
| StarCoder2-3B                | Baseline     |      8.42% |      11.58% | 80.00% |
| StarCoder2-3B                | DPO          |      0.00% |       3.86% | 96.14% |
| StarCoder2-3B                | SFT+DPO |      0.35% |      57.54% | 42.11% |
| StarCoder2-7B                | Baseline     |     11.23% |      12.63% | 76.14% |
| StarCoder2-7B                | DPO          |      0.00% |       8.42% | 91.58% |
| StarCoder2-7B                | SFT+DPO |      0.00% |      63.51% | 36.49% |
| StarCoder2-15B               | Baseline     |     10.53% |      14.39% | 75.08% |
| StarCoder2-15B               | DPO          |      0.00% |       8.07% | 91.93% |
| StarCoder2-15B               | SFT+DPO |      0.70% |      62.46% | 36.84% |
| DeepSeek-Coder-6.7B-Instruct | Baseline     |      6.67% |      24.91% | 68.42% |
| DeepSeek-Coder-6.7B-Instruct | DPO          |      1.05% |      28.42% | 70.53% |
| DeepSeek-Coder-6.7B-Instruct | SFT+DPO |      0.70% |      48.42% | 50.88% |

#### 4.2.4 替换率按库分类

以 StarCoder2（3B / 7B / 15B）为例，按库统计测试集中各变体的替换率，以面板雷达图呈现（图中红线为 SFT+DPO，蓝色虚线为 DPO，灰线为 base）。

![fig_library_radar_panel_replacement_20260511.png](pic/fig_library_radar_panel_replacement_20260511.png)

各库样本数差异显著：pytorch（120）、tensorflow（76）、numpy（38）、scipy（20）、sklearn（15）、seaborn（14）、pandas（2）。

pytorch 与 tensorflow 效果最为显著

pytorch 的 base model 替换率约 8–12%，SFT+DPO 提升至 72–83%；

ensorflow 的 base model 替换率约 20–29%，SFT+DPO 提升至 80–84%。

### 4.3 通用代码能力评估：HumanEval 与 MBPP（SFT LoRA  & DPO）

**SFT LoRA**

| 模型                         | HumanEval Baseline | HumanEval LoRA | ΔHumanEval（绝对 / 相对） | MBPP Baseline | MBPP LoRA | ΔMBPP（绝对 / 相对） |
| ---------------------------- | -----------------: | -------------: | ------------------------: | ------------: | --------: | -------------------: |
| StarCoder2-3B                |             31.71% |         28.66% |           -3.05% / ↓9.62% |        42.20% |    21.60% |    -20.60% / ↓48.82% |
| StarCoder2-7B                |             32.32% |         35.37% |           +3.05% / ↑9.44% |        44.40% |    32.60% |    -11.80% / ↓26.58% |
| StarCoder2-15B               |             40.85% |         42.68% |           +1.83% / ↑4.48% |        52.20% |    34.00% |    -18.20% / ↓34.87% |
| DeepSeek-Coder-6.7B-Instruct |             73.17% |         48.17% |         -25.00% / ↓34.17% |        57.80% |    21.40% |    -36.40% / ↓62.97% |
| Qwen2.5-Coder-3B-Instruct    |             20.73% |          2.44% |         -18.29% / ↓88.23% |        55.80% |    15.20% |    -40.60% / ↓72.76% |

**SFT+DPO**

| 模型                         | HumanEval Baseline | HumanEval SFT+DPO | ΔHumanEval（绝对 / 相对） | MBPP Baseline | MBPP SFT+DPO | ΔMBPP（绝对 / 相对） |
| ---------------------------- | -----------------: | ---------------------: | ------------------------: | ------------: | ----------------: | -------------------: |
| StarCoder2-3B                |             31.71% |                 28.05% |          -3.66% / ↓11.54% |        42.20% |            42.20% |        0.00% / 0.00% |
| StarCoder2-7B                |             32.32% |                 31.71% |           -0.61% / ↓1.89% |        44.40% |            46.40% |      +2.00% / ↑4.50% |
| StarCoder2-15B               |             40.85% |                 45.73% |          +4.88% / ↑11.95% |        52.20% |            56.20% |      +4.00% / ↑7.66% |
| DeepSeek-Coder-6.7B-Instruct |             73.17% |                 72.56% |           -0.61% / ↓0.83% |        57.80% |            59.20% |      +1.40% / ↑2.42% |

![fig2_humaneval_mbpp.png](pic/fig2_humaneval_mbpp.png)

### 4.4 Retention：常用 API 的补全保留率

#### 4.4.1 SFT+DPO

为了验证SFT+DPO是否会破坏模型对其他常用 API 的既有补全能力，构造了一个API retention candidate：包含 **1596** 条样本、**39** 种 API，覆盖 `numpy`、`pytorch`、`pandas`、`sklearn`、`seaborn`、`tensorflow` 和 `scipy` 等库。

该数据集首先从公开代码中抽取常用 API 的函数级补全样本，并显式排除训练集中出现过的 API；**随后使用base model进行初筛，仅保留其本来就能够正确补全的样本**。对于每个模型，按每种 API 最多选取 20 条样本，形成该模型对应的 retention set，并在同一批样本上比较 base model 与 SFT+DPO 的

由于 retention set 本身由 base model 正确样本构成，因此 base model 在该集合上的准确率按定义为 **100%**；SFT+DPO 的结果则对这部分既有能力的保留程度。

| 模型                         | API Match on Candidate | Retention APIs | Retention Set Samples | Base Model | SFT+DPO on Retention Set |
| ---------------------------- | ---------------------: | -------------: | --------------------: | ---------: | ----------------------------: |
| StarCoder2-3B                |                 67.86% |             26 |                   520 |    100.00% |                        72.88% |
| StarCoder2-7B                |                 70.05% |             27 |                   540 |    100.00% |                        72.22% |
| StarCoder2-15B               |                 71.74% |             27 |                   540 |    100.00% |                        85.37% |
| DeepSeek-Coder-6.7B-Instruct |                 63.85% |             25 |                   500 |    100.00% |                        90.80% |

结果表明，对于 StarCoder2 与 DeepSeek，SFT+DPO 在 retention set 上仍然保留了较高的 exact API match，其中 StarCoder2-15B 与 DeepSeek 分别达到 **85.37%** 和 **90.80%**。

综上，**SFT+DPO 在能够学习到版本信息的同时，依然保留了较大部分常用 API 的补全能力**。



(注：

Qwen 系列也进行了相同的 retention 测试，但由于其 base model 在 API 初筛集上的正确率较低，最终只能构造较小的 retention set：

Qwen2.5-Coder-3B 为 10 APIs / 200 samples / 81.00%，

Qwen2.5-Coder-7B 为 4 APIs / 80 samples / 77.50%，

Qwen2.5-Coder-14B 为 7 APIs / 140 samples / 82.86%)

#### 4.4.2 DPO与SFT+DPO 对比

本节在同一 retention set（StarCoder2-7B，540 条，覆盖 numpy / pandas / pytorch / sklearn，base model 准确率按定义为 100%）上补充 DPO 的对比数据：

| 库       | DPO 保留率 | SFT+DPO 保留率 |       差距 |
| -------- | ---------: | ------------------: | ---------: |
| numpy    |      45.0% |               66.8% |     −21.8% |
| pandas   |      95.0% |               97.5% |      −2.5% |
| pytorch  |      55.0% |               80.0% |     −25.0% |
| sklearn  |      15.0% |               20.0% |      −5.0% |
| **整体** |  **51.3%** |           **72.2%** | **−20.9%** |

DPO 对非目标 API 的扰动**强于** SFT+DPO，整体保留率低约 21 %，其中 numpy 和 pytorch 差距尤为显著。

这也再次说明了SFT+DPO 的代码语境敏感性使其扰动相对更局限，从而保留了更多既有能力。

## 5. SFT+DPO 机制分析：Logit Lens 与 Tuned Lens

本节讨论 SFT+DPO这一改进在模型内部的形成机制。

### 5.1 实验设置

#### 5.1.1 样本

取自 `repair_sft_test` 与 `consistency_sft_test` 的并集，共 **35** 条。观测点为 replacement API 与 deprecated API 首次分叉的共享决策前缀。

#### 5.1.2 方法

 **（一）Logit Lens**

直接将输出层的 unembedding 矩阵 $W_U$ 作用于第 $l$ 层的隐藏状态 $h_l$，得到该层的词表分布 $p_l = \text{softmax}(W_U h_l)$。其优势是零参数、无需训练，但中间层的表征空间与输出层存在系统性偏移，直接套用 $W_U$​ 会引入较大噪声，在浅层尤为明显。

---

**（二）Tuned Lens**

为每一层 $l$ 单独训练一个 low-rank 仿射变换（translator）$T_l$，将中间层隐藏状态先映射到与输出层表征空间更对齐的位置，再施加 unembedding：$$p_l^{\text{tuned}} = \text{softmax}\!\left(W_U \cdot T_l(h_l)\right)$$

$T_l$​​ 的训练目标是最小化其读出分布与输出层分布之间的 KL 散度，使各层读出在量纲和方向上与最终决策分布更可比。为聚焦于 API 补全场景，使用 `repair_sft_train/val + consistency_sft_train/val` 中能够规约到 API 决策位置的 **403/58** 条样本训练 Tuned Lens，使 Tuned Lens 在中间层更能反映模型对版本一致性的实际偏好。

#### 5.1.3 指标

**（一）Replacement Margin**：衡量第 $l$ 层读出对 replacement continuation 相对于 deprecated continuation 的偏好强度，正值表示该层更偏向 replacement API：

$$M_{\text{seq}}^{(l)} = \log P_l(y_{\text{rep}} \mid x) - \log P_l(y_{\text{dep}} \mid x)$$

$$M_{\text{first}}^{(l)} = \log P_l\!\left(t_0^{\text{rep}} \mid x\right) - \log P_l\!\left(t_0^{\text{dep}} \mid x\right)$$

其中 $P_l(\cdot)$ 为第 $l$ 层经 lens 读出的分布，$y_{\text{rep}}$/$y_{\text{dep}}$ 为完整 API suffix，$t_0^{\text{rep}}$/$t_0^{\text{dep}}$ 为首个分叉 token。

**（二）Pairwise Accuracy**：衡量 replacement 在样本上优于 deprecated 的比例：

$$\text{WR}^{(l)} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[M_{\text{seq}}^{(l,\,i)} > 0\right]$$

**（三）JSD(p_l, p_L)**：衡量第 $l$ 层全词表读出分布与输出层分布的距离，值越小表示该层越接近最终决策分布：

$$\text{JSD}(p_l,\, p_L) = \frac{1}{2} D_{\text{KL}}(p_l \| m) + \frac{1}{2} D_{\text{KL}}(p_L \| m), \quad m = \frac{p_l + p_L}{2}$$

---

### 5.2 输出层 Replacement Margin

如图展示了输出层的 replacement margin（$M_{\text{seq}}^{(L)}$ 与 $M_{\text{first}}^{(L)}$）：空心点为 base model，实心点为 SFT+DPO，连线方向与长度直接反映偏好的移动方向与幅度；竖直虚线为零分界，越过零线表示决策方向发生翻转。

![fig2_mechanism_margin.png](pic/fig2_mechanism_margin.png)

在 Logit Lens 下，sequence margin 均值从 base model 的 **−0.05** 提升至 **32.88**，first-token margin 均值从 **−2.43** 提升至 **21.13**。

在 Tuned Lens 下，变化更为显著：sequence margin 均值从 **−0.62** 提升至 **60.76**，first-token margin 从 **−3.30** 提升至 **43.91**。

**结果表明，SFT+DPO 在输出层显著增强了 replacement margin，故可以假设在模型中间层就已经出现了偏向增强。**

---

### 5.3 中间层 Replacement Margin 与 样本 Pairwise Accuracy

StarCoder2-3B、StarCoder2-7B 和 StarCoder2-15B 用于**纵向比较同一模型系列在不同规模下的表现**；

StarCoder2-7B、Qwen2.5-Coder-7B-Instruct 和 DeepSeek-Coder-6.7B-Instruct 用于**横向比较相近规模下不同模型系列的表现**。

![fig3_mechanism_trajectory.png](pic/fig3_mechanism_trajectory.png)

在 base model 中，replacement preference 往往要到深层才出现且幅度有限；而 SFT+DPO 在中间层就已拉开两者差距，并在后续层中持续放大。

纵向比较：StarCoder2 三个规模的模型呈现出相似的趋势，说明这一改进在同一系列模型内具有一致性；

横向比较：Starcoder7B、Qwen7B 与 DeepSeek6.7B 在相近规模下展现出不同的偏向起点，但 SFT+DPO 最终均为正值。

---



<img src="pic/fig3b_winrate.png" alt="fig3b_winrate.png" style="zoom: 67%;" />

在 Logit Lens 下从 base model 的 **0.395** 提升至 **0.977**（+0.582），在 Tuned Lens 下从 **0.399** 提升至 **0.894**（+0.495）。也使绝大多数样本在输出层能够正确预测出最新 API。

---

### 5.4  JSD

本节通过 JSD(p_l, p_L) 考察第 $l$ 层全词表分布与输出层分布之间的距离，从而判断版本一致性决策是否在更早的层级上开始成形。JSD 越低表示该层分布越接近输出层分布。

![fig5_mechanism_jsd.png](pic/fig5_mechanism_jsd.png)

在 Logit Lens 下，整体下降比较缓和，JSD均值（除最后一层外的其他所有中间层）仅从 **0.601** 降至 **0.598**；

在 Tuned Lens 下，整体下降则较为显著，从 **0.214** 降至 **0.099**。

这表明，在未校准中间层读出失配的情况下，JSD 容易受到原始 unembedding 噪声的干扰；而在校准后的读出空间中，SFT+DPO 确实使中间层分布更早、更稳定地向最新 API 决策分布收敛。

并且与 base model 相比，SFT+DPO 在 Tuned Lens 下更早地降低了中后层 JSD，这说明偏向最新api的决策分布在更浅层已开始形成。这一表现在 Tuned Lens 下更为显著。

### 5.5 PPL

对每条样本计算 `deprecated / replacement` 的 token-level perplexity ratio，以 $\log_{10}[\mathrm{PPL}(y_{\mathrm{dep}}) / \mathrm{PPL}(y_{\mathrm{rep}})]$ 报告：正值表示模型对 deprecated api 的预测置信度低于 replacement api，负值则相反。

![fig6_mechanism_ppl_depth.png](pic/fig6_mechanism_ppl_depth.png)

base model 的平均 log-ratio 在 Logit Lens 下仅为 **−0.199**，说明其对两种 api补全并无显著的方向性差异。

**SFT+DPO** 将该值分别提升至 **5.861**（Logit Lens）与 **13.210**（Tuned Lens），表明模型对 deprecated api补全的 每个 token 所赋予的平均概率低于 replacement api，两种方法都展示了显著且一致的概率差距。

---

### 5.6 base / DPO / SFT+DPO 三方机制对比

前面 replacement margin 与 JSD 的分析仅覆盖 base 与 SFT+DPO。然而，仅 dpo 的原因是"未学到 replacement api的偏好"还是"学到了但未能在生成中稳定体现"。

为此，重新在同样条件下对三者进行对比，以明确 仅 DPO 的失败原因，并在此基础上明确 SFT+DPO 所发挥的具体作用。

左侧 **Logit Lens**，replacement api 相对 deprecated api 的单位 token 对数概率优势，正值表示 replacement 更被偏好）；

右侧 **Tuned Lens**，**stable decision depth**——即逐层读出分布时，最早出现且此后不再翻转的 replacement api所在的层编号，值越小说明偏向 replacement 的决策越早形成

![fig_triplet_combined_20260507.png](pic/fig_triplet_combined_20260507.png)

**结论一（左侧）：dpo 已学到较强的局部 replacement preference。** 

StarCoder2 同系列上，dpo 的平均 token margin 达到 **23.494**，甚至略高于 SFT+DPO（**18.238**）。这说明 dpo 的问题并不在于没有区分 replacement 与 deprecated，在给定的 decision prefix 上，它已经能明确地偏向 replacement。

真正的问题在于：模型在进行正常的补全时，往往在尚未到达这个 API 决策位置之前，就已经生成了绕开该位置的补全路径（例如不断重复或提前终止），从而表现为 deprecated avoidance。

---

**结论二（右侧）：SFT+DPO 则是把这种偏好写进更浅层。** 

base model 的 stable decision depth 在 StarCoder2 中平均为 **24.271**，说明版本一致性偏好直到深层才出现。

SFT+DPO 将其提前至 **0.969**（跨系列：**20.667 → 6.574**），这说明 SFT+DPO 在浅层就已经能理解对 replacement api 的明确偏好。anchor 的作用不只是进一步放大局部偏好差距，而是把版本一致性偏好写入模型整个中间层的生成轨迹中，从而更可靠地支持最终的 replacement api 的补全。

---

### 5.7 激活差异分析：中性提示上的静态偏置检验

#### 5.7.1 方法来源

Minder et al.（ICLR 2026） *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences*。

**在无法访问微调数据的黑盒条件下，能否仅凭对比微调前后模型的内部表示，逆向推断出窄域微调（narrow finetuning）的训练目标？** 

该论文发现：对 LLM 进行少量微调（只训练少量特定任务数据）之后，即便在与训练任务毫不相关的随机文本上，微调前后模型的中间层激活之差也会形成可读的、任务相关的表征偏移——可以用输出头将这一差向量投影回词表，并读出与微调目标对应的 token 倾向。

该论文将这套读方法称为 **Activation Difference Lens（ADL）**，其操作流程为：
1. 对同一输入分别前向 base 模型和微调模型，在每一层 $l$ 的最后一个 token 位置取出隐藏状态 $h_l^{\text{base}}$ 和 $h_l^{\text{ft}}$；

2. 计算各层激活差向量 $\Delta h_l = h_l^{\text{ft}} - h_l^{\text{base}}$；

3. 将 $\Delta h_l$​ 投影到输出词表分布，读出被"放大"的 token 集合及对应分数。

   （steering vector + logit-lens）

这种做法的优势在于无需任何专门的测试集——在随机中性文本上就能暴露微调的表征痕迹，从而区分"全局静态重写"（对任意输入均施加均匀偏置）与"条件化偏移"（只在特定语境下激活）。

#### 5.7.2 实验设置

本节在完全不含目标 API 的两类输入上，分别检验 dpo 和 anchored_dpo 相对于 base 的激活差是否可读、且在多大程度上方向指向 replacement api 而非 deprecated api：

- **random_neutral_text**：与代码无关的纯中性prompt，不含任何目标库名称、API 名称或版本声明。例如：

  ```
  Describe a straightforward way to handle an empty list input.
  ```

- **library_neutral_code**：从现有 `probing_input` 字段派生的弱代码prompt，已过滤掉目标 replacement API、deprecated API 及显式版本前缀，保留了库相关的语法结构但不直接触发 API 决策。例如（PyTorch 库语境，但不含目标版本敏感 API）：

  ```python
  def getExtrinsics_(x_c, p_w):
      c_center = torch.mean(x_c, dim=1)
      w_center = torch.mean(p_w, dim=1)
      x_c_centered = x_c - c_center.unsqueeze(1)
      p_w_centered = p_w - w_center.unsqueeze(1)
      # create matrix to solve
  ```

对每类输入和每个模型，该论文两个指标。设 $h_l^{\text{ft}}$、$h_l^{\text{base}}$ 分别为微调模型和 base 模型在第 $l$ 层最后一个 token 的隐藏状态，$\Delta h_l = h_l^{\text{ft}} - h_l^{\text{base}}$，$W_U$ 为输出投影矩阵，$\mathcal{T}_{\text{rep}}$、$\mathcal{T}_{\text{dep}}$ 分别为 replacement / deprecated API 的 token 集合：

- **Final-layer ADL Norm**：最终层激活差向量的 L2 范数，反映微调引入的表征变化幅度：$\text{ADL Norm} = \|\Delta h_L\|_2$

- **Replacement $-$ Deprecated Readout Score**：将 $\Delta h_L$ 投影到词表后，replacement 与 deprecated token 家族的平均 logit 差，正值表示激活差方向偏向 replacement：$\text{Score} = \frac{1}{|\mathcal{T}_{\text{rep}}|}\sum_{t \in \mathcal{T}_{\text{rep}}} (W_U \Delta h_L)_t \;-\; \frac{1}{|\mathcal{T}_{\text{dep}}|}\sum_{t \in \mathcal{T}_{\text{dep}}} (W_U \Delta h_L)_t$

#### 5.7.3 实验结果与结论

![fig_adl_neutral_starcoder.png](pic/fig_adl_neutral_starcoder.png)

两个结论：

**第一**，dpo 在中性 prompt 上的 ADL Norm（**28.69**）略高于 anchored_dpo（**27.90**），说明仅 DPO 的影响不局限于 API 补全，而是在大部分输入上都留下了可读的全局偏置，这也印证了之前其在he和mbpp上指标大幅下降。

**第二**，切换到弱代码 prompt 上，anchored_dpo 的 Readout Score 上升幅度（0.416 → 0.543，+0.127）大于 dpo（0.408 → 0.518，+0.110），这说明anchored dpo 的额外收益集中出现在代码语境下，而非随机文本上，说明它强化的是代码语境相关的条件化偏置，而非单纯放大全局偏置强度。

综合来看，dpo 在激活层面更多体现为均匀的全局表征偏移；而 anchored_dpo 表现出更强的代码语境敏感性，正是这种条件化偏置，使其在API 补全中更可靠地命中 replacement API，同时对非目标 API 的干扰相对有限。

---

## 6. 有效参数的稀疏性与集中性分析

以 StarCoder2-7B 为例，从三个递进层面考察 DPO 与 SFT+DPO 的参数更新集中在哪里，以及保留这部分参数能否复现主要效果。

### 6.1 delta weight 分布

投影矩阵 $W$ 的有效更新为：$\Delta W = \frac{\alpha}{r}BA,\qquad W' = W + \Delta W$

其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$、$B \in \mathbb{R}^{d_{\text{out}} \times r}$，$r$ 为 rank，$\alpha$ 为 scaling factor。为衡量更新的层间分布，对每个 attention 模块计算有效更新矩阵的 Frobenius 范数：$\|\Delta W_l\|_F = \sqrt{\sum_{i,j} (\Delta W_{l,ij})^2}$

DPO 与 SFT+DPO 均作用于 StarCoder2-7B 的 32 层 attention（`q_proj/k_proj/v_proj/o_proj`），共 **128** 个模块，总参数量 **1.51B**

统计结果显示，DPO 的平均 Frobenius 范数为 **0.225**，SFT+DPO 为 **0.228**，整体幅度相近；但变化高度集中于中后层（层 17–31）的 `o_proj`，其 Frobenius 范数通常为全体均值的 **2–3 倍**，远高于浅层。

<img src="pic/fig_delta_fro_heatmap.png" alt="fig_delta_fro_heatmap.png" style="zoom: 33%;" />

### 6.2 稀疏化实验

稀疏化实验不重新训练模型，而是**对已训练的 effective delta 做裁剪**：将每个模块的 $\Delta W$ 按绝对值排序，保留 top-k% 的元素、其余置零，再将裁剪后的更新叠加到 base model 上评测。

“有效 delta 参数量”指裁剪后保留的 $\Delta W$​ 元素数。这里的“full”仅指**所有 LoRA 注入位点均开放或完整保留**

完整 LoRA adapter 的参数量为**1.51B**，keep_50% / keep_30% / keep_20% / keep_15% / keep_10% / keep_5% 分别对应约 **755M / 453M / 302M / 226M / 151M / 75M** 个元素。

| 配置 | 有效 delta 参数量 | 弃用率 | 替换率 |
|---|---:|---:|---:|
| <u>Base model</u> | 0 | 11.2% | **12.6%** |
| <u>DPO full LoRA adapter</u> | 1.51B | 0.0% | 8.4% |
| DPO keep_50% | 755M | 0.0% | 8.8% |
| DPO keep_30% | 453M | 0.0% | 14.4% |
| DPO keep_20% | 302M | 0.0% | 18.2% |
| DPO keep_15% | 226M | 0.4% | 23.5% |
| DPO keep_10% | 151M | 1.1% | <u>26.3%</u> |
| DPO keep_5% | 75M | 1.1% | **28.1%** |
| <u>SFT+DPO full LoRA adapter</u> | 1.51B | 0.0% | 63.5% |
| SFT+DPO keep_50% | 755M | 0.0% | 64.2% |
| SFT+DPO keep_30% | 453M | 0.7% | **64.6%** |
| SFT+DPO keep_20% | 302M | 1.4% | 62.1% |
| SFT+DPO keep_15% | 226M | 1.4% | 56.5% |
| SFT+DPO keep_10% | 151M | 1.4% | 50.5% |
| SFT+DPO keep_5% | 75M | 1.4% | 40.0% |

对 SFT+DPO，保留 30%–50% 参数时替换率约为 **64–65%**，与完整 adapter（63.5%）几乎持平，说明这一区间内稀疏化无损。在 keep_20%（302M）时，替换率仍高达 **62.1%**，相当于完整 SFT+DPO 的 97.8%。替换率的明显拐点出现在 **keep_20%–keep_15%** 之间：keep_30% → keep_20% 的降幅仅 2.5%，而 keep_20% → keep_15% 的降幅扩大至 5.6%，此后每减少 5% 的保留比例，降幅进一步加速（keep_15% → keep_10% 降 6.0%；keep_10% → keep_5% 降 10.5%）。这说明 SFT+DPO 的 replacement-induction 信号主体集中于约 **20%–30%** 的 top-magnitude 参数，低于这一阈值后效果出现非线性衰减。

DPO 则呈现相反趋势：完整 LoRA adapter 的替换率低于 base model（8.4% vs 12.6%），keep_5% 后反而升至 **28.1%**。结合之前 §4.2.1 中”DPO 更接近 deprecated-avoidance”的结论，一种合理解释是：DPO 全参更新中包含抑制 replacement 的分量；裁剪到最高幅度的参数后，这些抑制成分被优先保留下来，从而比完整更新更有利于 replacement 的出现。

### 6.3 冻结层与跨规模验证实验

#### 6.3.1 StarCoder2-7B 模块级受限重训练

冻结层实验**从 base model 重新训练新的 LoRA**，但仅允许层 17–31 的 `q_proj` 与 `o_proj` 参与更新，其余层全部冻结。这一设置检验的是：若微调从一开始就被限制在该子空间内，能否仍然学到主要行为。

受限训练共覆盖约 **637M** 个有效 delta 元素，占全空间的 **42.2%**。

在此基础上，进一步补充一个模块级受限重训练 probe：固定 `StarCoder2-7B`，仅开放 `layers 17–31` 的 `o_proj`，分别训练 plain DPO 与 SFT+DPO，并与完整 LoRA adapter 及 `q_proj + o_proj` 冻结层结果比较。该补充实验沿用主实验数据 `mixed_sft_v1`，LoRA 设置为 `r=8, alpha=16, dropout=0.05`，训练 `1 epoch`。相比 `layers 17–31, q_proj + o_proj`，`o_proj-only` 设置不再开放 `q_proj/k_proj/v_proj`，因此是在继续追问：**仅靠中后层的输出投影，能否单独承载主要偏好信号？**

| 配置 | 允许更新的有效 delta 空间 | 弃用率 | 替换率 |
|---|---:|---:|---:|
| Base model | 0 | 11.2% | 12.6% |
| SFT+DPO full（用于对比） | 1.51B | 0.0% | 63.5% |
| SFT+DPO （layers 17–31，q+o） | 637M | 0.0% | **49.8%** |
| SFT+DPO （layers 17–31，o_proj only） | ~430M | 0.0% | **44.6%** |
| DPO full（用于对比） | 1.51B | 0.0% | 8.4% |
| DPO （layers 17–31，q+o） | 637M | 0.0% | **12.3%** |
| DPO （layers 17–31，o_proj only） | ~430M | 0.0% | **11.2%** |

SFT+DPO 冻结层（q+o）获得 **49.8%** 的替换率，与 SFT+DPO keep_10% 稀疏化（**50.5%**）高度接近。进一步将可训练模块收窄到仅 `o_proj` 后，替换率降至 **44.6%**，约为完整 SFT+DPO 的 70.2%。两个冻结层变体相差 **5.2%**，说明高层 `q_proj` 提供了非忽略性的补充收益，但主要偏好信号已高度集中于中后层 `o_proj`。

对 plain DPO，`o_proj-only` 子空间下弃用率仍压至 **0.0%**，替换率为 **11.2%**，略高于完整 DPO 的 **8.4%**——这进一步印证了 §6.2 中的观察：DPO 全参更新中包含抑制 replacement 的分量，受限子空间反而减轻了这种干扰。

两条路径（稀疏化 vs 冻结层）收敛到相近结果，表明 SFT+DPO 的主要收益集中编码于中后层的有限参数子集，而非依赖全网络范围的分散微调。当前最合理的判断是：关键子空间主要落在 `layers 17–31` 的 `o_proj`，并由少量高层 `q_proj` 补全。

需要注意的是，这组实验仍是**模块级限制**，不是元素级或更细粒度的 retraining，因此还不能回答“最小关键参数集合到底是哪几个矩阵元素”。它目前能支持的边界结论是：对 plain DPO，`deprecated-avoidance` 行为已经可以在很小的模块子空间中复现；对 SFT+DPO，中后层 `o_proj` 已经承载了大部分有效信号；但若要逼近完整 SFT+DPO 的 **63.5%** 替换率，仅靠 `o_proj-only` 还不够，下一步更合理的是继续测试更小的 `q_proj + o_proj` 子集，而不是重新回到全模块训练。

#### 6.3.2 跨规模验证：StarCoder2-3B / 15B

为验证"中后层集中"这一结论是否具有规模一致性，对 StarCoder2-3B（30 层）和 StarCoder2-15B（40 层）分别进行相同的受限重训练：每个模型按比例取深层约 **47%** 的层段（3B：layers 16–29，15B：layers 21–39），仅开放 `q_proj` 与 `o_proj`。受限训练的有效 delta 参数量分别为约 **264M**（3B）和 **1,434M**（15B）。

| 模型 | 受限层范围 | 弃用率 | 替换率 | 占完整 SFT+DPO 的比例 |
|---|---|---:|---:|---:|
| StarCoder2-3B，完整 SFT+DPO | — | 0.4% | 57.5% | — |
| StarCoder2-3B，restricted（q+o，layers 16–29） | ~264M | 0.0% | **44.2%** | **76.8%** |
| StarCoder2-7B，完整 SFT+DPO | — | 0.0% | 63.5% | — |
| StarCoder2-7B，restricted（q+o，layers 17–31） | ~637M | 0.0% | **49.8%** | **78.4%** |
| StarCoder2-15B，完整 SFT+DPO | — | 0.7% | 62.5% | — |
| StarCoder2-15B，restricted（q+o，layers 21–39） | ~1,434M | 0.7% | **50.2%** | **80.4%** |

三个规模的受限重训练均将弃用率压至 **0%–0.7%**，替换率分别达到完整 SFT+DPO 的 **76.8% / 78.4% / 80.4%**，这表明"中后层的 q+o 子空间可以承载 SFT+DPO 主要偏好信号"，具有较好的规模一致性。

### 6.4 模块类型消融：MLP、投影类型与层范围细化

前几节的分析均以 attention 的 `q/k/v/o_proj` 作为训练目标，且冻结层实验进一步将范围收窄至中后层。本节在 StarCoder2-7B 上补充三组消融，系统考察 MLP 模块、不同投影类型以及更细粒度层范围对 replacement 行为的贡献。所有变体均采用相同超参（`r=8, alpha=16, lr=5e-5, 1 epoch`），以完整 SFT+DPO（弃用率=0.0%，替换率=**63.5%**）为参照。

#### 6.4.1 MLP 与 Attention 的对比

| 目标模块 | 弃用率 | 替换率 | 备注 |
|---|---:|---:|---|
| `q/k/v/o_proj`（当前 SFT+DPO） | 0.0% | **63.5%** | 参照 |
| `q/k/v/o_proj` + `c_fc/c_proj` | 0.0% | **66.0%** | Attention + MLP |
| `c_fc/c_proj` | 0.4% | **61.1%** | 仅 MLP |

MLP-only 变体的替换率达到 **61.1%**，仅比attention-only 低 **2.4%**。这说明 MLP 模块同样具有承载 replacement API 偏好知识的能力，

将 Attention 与 MLP 同时训练后，替换率提升至 **66.0%**，但幅度有限，表明两类模块在编码 replacement 偏好方面存在高度信息重叠，叠加后的边际收益因此趋于饱和。

#### 6.4.2 Attention 内部投影类型消融

在全部 32 层开放的前提下，按投影类型对 SFT+DPO 做消融：

| 目标投影 | 弃用率 | 替换率 | 备注 |
|---|---:|---:|---|
| `q/k/v/o_proj`（完整） | 0.0% | **63.5%** | 参照 |
| `o_proj`（全层） | 0.0% | **58.6%** | 仅 o_proj |
| `k_proj + v_proj` | 1.1% | 40.7% | 仅 kv |
| `q/k/v`（排除 o） | 1.1% | 40.7% | qkv 不含 o |

`o_proj`（全 32 层）单独训练可达 **58.6%**，远高于任何不含 `o_proj` 的组合（40.7%）。

排除 `o_proj` 后，替换率均下降约 23%；相应地，弃用率也因偏好信号减弱而回升至 1.1%。这与 §6.1 的 Frobenius 范数热图和 §6.3 的冻结层结论一致：**`o_proj` 是 attention 模块内部承载 replacement API 内化信号最为集中的投影矩阵**。

全层 `o_proj` 的 **58.6%** 高于仅开放中后层 `o_proj`（layers 17–31）的 **44.6%**（§6.3），差距 14%，说明浅层 `o_proj` 也提供了可观的补充贡献，但中后层仍是替换偏好内化最为集中的层段。

#### 6.4.3 层范围细化

在 §6.3 的基础上进一步收窄中后层的 `q+o` 范围，观察替换率如何随可训练层减少而衰减：

| 可训练层范围 | 弃用率 | 替换率 | 与 L17–31 的差距 |
|---|---:|---:|---:|
| layers 17–31（q+o，§6.3 参照） | 0.0% | **49.8%** | — |
| layers 20–31（q+o） | 0.0% | 40.4% | −9.4% |
| layers 25–31（q+o） | 0.4% | 33.3% | −16.5% |
| layers 28–31（q+o） | 0.4% | 29.1% | −20.7% |

从层 17 缩至层 20 时，替换率下降 **9.4%**，此后每进一步缩减 5–6 层，衰减幅度在 7–11% 之间，整体呈近线性递减。层 28–31（最末 4 层）单独训练时，替换率仅为 **29.1%**，接近 base model 水平（12.6%）的两倍多，但已显著弱于 layers 17–31 的结果。

以上结果共同表明，SFT+DPO 的有效信号在 **层 17 附近存在明确的收益临界点**：17–31 这一范围基本覆盖了版本 API 知识内化的核心层段，继续向浅层扩展的边际收益有限，而从层 17 向深层收窄则会导致 replacement 召回率的持续衰减。

这一集中特性从参数局部化视角进一步印证：SFT+DPO 将版本信息与最新 API 偏好内化于模型的中后层紧凑子空间中，而非依赖全网络的均匀微调。

---

## 7. 反向版本前缀实验

本节讨论：SFT+DPO 在新版本前缀下能够正确命中 replacement API 的样本，若将版本前缀替换为**旧版本前缀**（对应 deprecated API 仍普遍使用的早期版本号，如 `# pytorch 1.13.1`），模型是否会回到 deprecated API。

实验以 StarCoder2-7B 为例，从 SFT+DPO 在新版本前缀下的预测中，筛选同时满足“包含 replacement API 且不包含 deprecated API”的样本，共得到 **181** 条 clean subset，随后统一替换为旧版本前缀，重新推理。例如，将 `# pytorch 2.x` 改写为 `# pytorch 1.13.1`。

| 模型 / 设定 | 样本数 | 旧版本前缀下弃用率 | 旧版本前缀下替换率 |
|---|---:|---:|---:|
| StarCoder2-7B base model | 181 | 14.9% | 17.1% |
| StarCoder2-7B SFT+DPO | 181 | **0.0%** | **97.2%** |

base model 在加入旧版本前缀后，替换率降至 17.1%，且有 14.9% 的样本出现 deprecated 命中——说明旧版本前缀对 base model 确实有引导作用，能将其拉向 deprecated API。

SFT+DPO 在旧版本前缀下没有被拉回 deprecated API：弃用率依然为 **0.0%**，替换率也维持在 **97.2%**，未出现任何翻回行为。

这一结果说明，当前 SFT+DPO 训练更像是在模型中建立一个强的向前更新偏好：当训练目标主要是“新版本语境下使用 replacement API”时，模型会稳定偏向最新 API，而不会因为一个旧版本前缀就轻易回退到 deprecated API。这并不意味着版本前缀没有作用；相反，后续 §8 会进一步显示，新版本前缀仍能为替换率提供额外增益。这里的限制是：当前训练数据并没有显式覆盖“旧版本前缀 → deprecated API”的反向偏好，因此它还不能被解释为一个可随任意版本号双向切换的版本条件生成器。

当然，这仍只是一个单模型、单子集的 sanity check，不应直接外推到所有样本或所有模型。

---

## 8. 版本上下文消融：版本前缀的作用

之前的 SFT+DPO 主实验采用的是一个**带版本上下文的 preference triple**。具体来说，

训练输入 $x$ 是 `version_prompt`，其中包含显式的 `# library version` 前缀；

正样本 $y^+$ 是目标版本一致的 completion，包含 replacement API；

负样本 $y^-$ 是将 $y^+$ 中 replacement API 替换为 deprecated API 后得到的合成 completion。

SFT+DPO 在这个三元组上同时优化 DPO loss 和 replacement API token 的 CE anchor。评测时，主实验也使用同一测试集中的 `version_prompt`，因此训练和测试都处在“有版本前缀”的设置下。

这个设置解释了 §4.2 的强结果，但也留下一个软件工程场景中很关键的问题：SFT+DPO 的提升中，显式版本前缀和 preference pair 各自承担了什么角色？一种可能是，版本前缀主要作为上下文信号，帮助模型在测试时识别“当前应面向新版本 API”；另一种可能是，preference pair 与 CE anchor 会把 replacement API 偏好进一步内化到模型参数中，使模型即使在版本信息不完整的提示里，也更倾向于生成最新 API。换句话说，本节不是要否定版本信息的作用，而是要区分：**版本前缀带来的显式上下文增益**，与**微调后模型对最新 API 的参数化内化**。

为回答这个问题，本节补充主实验的反事实设置。我们使用 `probing_input` 字段作为无版本上下文的 prompt；它仅保留纯代码上下文，不含任何 `# library version` 前缀行和 `import` 声明。这样可以构造两类对立面：

| 设置 | 训练 prompt | preference pair / anchor | 评测 prompt |
|---|---|---|---|
| SFT+DPO主实验 | `version_prompt` | replacement chosen vs deprecated rejected；replacement-token CE anchor | `version_prompt` |
| 无版本评测 | `version_prompt` | 同主实验 | `probing_input` |
| 无版本训练 | `probing_input` | 同样的 chosen/rejected 与 CE anchor | `version_prompt` / `probing_input` |

### 8.1 评测时去掉版本前缀

本实验保持模型权重不变，只在评测时用 `probing_input` 替换测试集的 `version_prompt`。这相当于问：在已经通过带版本上下文完成 SFT+DPO 训练后，测试时显式版本前缀还能贡献多少额外收益？如果版本前缀只是无关装饰，那么有/无版本评测应当几乎一致；如果版本前缀是有效上下文信号，那么去掉它后替换率应当下降，但若微调已经内化了最新 API 偏好，下降不应回到 base model 水平。

| 模型 | 替换率（有版本） | 替换率（无版本） | 替换率变化 | 保留率 |
|---|---:|---:|---:|---:|
| StarCoder2-7B SFT+DPO | 63.5% | **49.8%** | −13.7% | 78% |
| StarCoder2-7B DPO | 8.4% | **6.7%** | −1.7% | 79% |
| StarCoder2-3B SFT+DPO | 57.5% | **51.2%** | −6.3% | 89% |
| StarCoder2-15B SFT+DPO | 62.5% | **50.5%** | −12.0% | 81% |
| DeepSeek-Coder-6.7B SFT+DPO | 48.4% | **40.7%** | −7.7% | 84% |
| Qwen2.5-Coder-3B SFT+DPO | 56.1% | **43.5%** | −12.6% | 78% |
| Qwen2.5-Coder-7B SFT+DPO | 59.3% | **48.4%** | −10.9% | 82% |
| Qwen2.5-Coder-14B SFT+DPO | 54.4% | **54.4%** | 0% | **100%** |

去掉版本前缀后，所有模型的弃用率均维持在 **0%–0.4%**（与有版本时相同），说明 SFT+DPO 对 deprecated API 的抑制已通过微调稳定写入模型参数，不依赖推理时的版本上下文。更关键的是替换率：去掉版本前缀后，大多数模型下降 **6–14%**，说明版本前缀并非无关信息，而是确实提供了额外的上下文激活增益；同时，保留率普遍在 **78%–89%** 之间，说明 SFT+DPO 已将相当一部分最新 API 偏好内化到模型参数中，这部分内化偏好不随版本前缀的有无而消失。

因此，本实验呈现的是一种”双层机制”：参数化内化层持续降低弃用率并保障基础替换能力，显式版本前缀则作为上下文激活信号进一步提高替换率。

Qwen2.5-Coder-14B 是唯一在去除版本前缀后替换率无下降的模型（保留率 100%），说明在部分大型模型上 replacement API 偏好可能已被更充分地内化进模型中。

### 8.2 训练时去掉版本前缀（StarCoder2-7B）

以 StarCoder2-7B 为例，我们用 `probing_input` 作为训练 prompt，重新训练一个无版本上下文的 SFT+DPO 变体；chosen/rejected completion 与 replacement-token CE anchor 保持不变，唯一变化是三元组中的输入 $x$ 不再携带版本声明。

然后分别用有版本和无版本 prompt 评测，形成完整的 2×2 矩阵：

| 训练 prompt | 评测 prompt | 弃用率 | 替换率 |
|---|---|---:|---:|
| **有版本前缀（主实验）** | **有版本前缀** | 0.0% | **63.5%** |
| 有版本前缀 | 无版本前缀 | 0.4% | 49.8% |
| 无版本前缀 | 有版本前缀 | 0.4% | **64.9%** |
| 无版本前缀 | 无版本前缀 | 0.4% | 55.8% |

训练时完全去掉版本前缀后，以有版本前缀评测的替换率为 **64.9%**，与标准 SFT+DPO 的 **63.5%** 几乎持平。这说明在训练阶段，chosen/rejected 序列之间的 API 对比信号与 replacement-token CE anchor 已足以驱动模型将 replacement API 偏好内化到参数中。

但版本前缀在评测时仍有一定的激活作用：无版本训练模型在有版本前缀下达到 **64.9%**，在无版本前缀下为 **55.8%**，相差 **9.1%**。也就是说，即使训练阶段不提供版本声明，测试阶段加入版本前缀仍能进一步释提高替换率 。无版本训练 + 无版本评测是最弱条件：训练时没有版本声明，测试时也没有版本声明。即便如此，替换率仍为 **55.8%**，约为 base model（无版本时 11.6%）的 **4.8 倍**。这说明 SFT+DPO 并非仅在识别到 `# library version` 标记时触发局部生成模式，而是将最新 API 偏好系统地写入了模型的默认补全分布，实现了版本信息与 API 知识在参数层面的持久内化；显式版本前缀在此基础上仍是一个有效的上下文触发信号，可进一步激活这一内化偏好。

### 8.3 小结

这组补充实验将 §4.2 的 SFT+DPO 主结果置于更清晰的因果分析框架中。主实验在”训练有版本、测试有版本”的版本感知设置下完成；§7 和 §8 则通过逐步移除或反转版本上下文，系统考察版本信息的实际贡献。三组证据共同指向一个更精确的结论：**SFT+DPO 通过 preference pair 与 CE anchor 的联合优化，将版本信息和最新 API 偏好内化进模型参数，从而降低弃用率、提高替换率；显式版本前缀作为有效的上下文激活信号，可在参数化内化的基础上进一步提升替换率。**

- §7 的旧版本前缀实验：将新版本前缀替换为旧版本前缀后，SFT+DPO 没有翻回 deprecated API，说明训练已在模型内部形成了稳定的向新版本 API 内化的单向偏好。
- §8.1 的无版本评测：训练仍有版本前缀，但测试时去掉版本上下文，replacement 偏好仍保留 78%–100%，说明最新 API 知识已被大量内化至模型参数；大多数模型仍下降 6–14%，说明版本前缀持续提供额外的上下文激活增益。
- §8.2 的无版本训练：训练阶段不提供版本前缀也能将 replacement 偏好充分内化，测试阶段加入版本前缀仍带来 9.1% 的激活增益，两阶段均缺少版本前缀时替换率仍达 base model 的 4.8 倍。

从软件工程实践的角度看，这一结论准确刻画了 SFT+DPO 的作用机制：它并非将版本信息降格为无关变量，而是将版本信息的作用分解为**参数化内化**与**上下文激活**两个互补层次。参数化内化通过 preference 训练将版本信息和最新 API 的关联知识稳定写入模型参数，使模型在无显式版本声明时也能持续降低弃用率；上下文激活则通过显式版本前缀在推理时进一步释放这一内化偏好，为替换率提供额外增益。前者确立了 SFT+DPO 相对于单纯提示工程的泛化优势，后者说明了显式版本信息的持续价值。当前方法的适用边界是：由于训练目标主要面向”新版本前缀 → replacement API”的单向更新，它更适合维护向前兼容的默认生成行为，而不是构建可按任意历史版本号双向切换的版本条件生成器。这一结论也与 §5.7 的 ADL 分析相互印证：SFT+DPO 在模型参数层面留下了可检测的版本偏好迹象，说明版本信息与最新 API 知识已被真正内化进模型，而非依赖上下文临时激活。
是是是