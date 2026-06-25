# 第一版 DPO LoRA 结果记录

Date: 2026-04-23

Scope: 记录第一版 `version_aware_dpo_lora` 在 `mixed_sft_v1` 上的目标任务结果，以及在 HumanEval / MBPP 上的通用代码能力结果。

注意：本文档只记录 2026-04-22 已完成的第一版纯 DPO LoRA 结果。后续 DPO + API-anchor 的 StarCoder2-7B 初筛结果见 `output/technical_report_dpo7b_screen_20260423.md`，不纳入本文档的第一版 DPO 统计。

## 1. 方法定义

第一版 DPO LoRA 的核心设置：

- `chosen`: mixed SFT 样本中的 replacement completion，即包含推荐替代 API 的补全。
- `rejected`: 将 `chosen` 中命中的 replacement API 反替换为 deprecated API 后得到的合成负样本。
- Reference policy: 同一个 base model，在 PEFT `disable_adapter()` 状态下作为 reference。
- Preference objective: DPO，`beta = 0.1`，sequence log-prob reduction 为 `sum`。
- LoRA: `r = 8`，`alpha = 16`，`dropout = 0.05`。
- Training: 1 epoch，learning rate `5e-5`，gradient checkpointing enabled。
- DPO pairs: train 1970，val 260。

训练输出目录：

- `output/dpo_lora_mixed_sft_v1_20260422/starcoder2_3b`
- `output/dpo_lora_mixed_sft_v1_20260422/starcoder2_7b`
- `output/dpo_lora_mixed_sft_v1_20260422/starcoder2_15b`
- `output/dpo_lora_mixed_sft_v1_20260422/deepseek_coder_6_7b_instruct`

## 2. Benchmark

目标任务评测使用：

- Test file: `data/mixed_sft_v1/mixed_sft_test.jsonl`
- Test size: 285
- Split composition: `repair_sft = 14`，`consistency_sft = 21`，`reference_sft = 250`
- Leakage check: train / val / test semantic overlap 均为 0
- Prompt field: `version_prompt`
- Metrics:
  - Deprecated usage rate，越低越好
  - Replacement hit rate，越高越好
  - Exact match target rate，越高越好

目标任务结果源文件：

- `output/dpo_lora_mixed_sft_v1_20260422_eval/starcoder2_3b/comparison_summary.json`
- `output/dpo_lora_mixed_sft_v1_20260422_eval/starcoder2_7b/comparison_summary.json`
- `output/dpo_lora_mixed_sft_v1_20260422_eval/starcoder2_15b/comparison_summary.json`
- `output/dpo_lora_mixed_sft_v1_20260422_eval/deepseek_coder_6_7b_instruct/comparison_summary.json`

通用能力评测使用 BigCode evaluation harness：

- HumanEval pass@1
- MBPP pass@1

通用能力结果源文件：

- DPO: `output/bigcode_code_eval_dpo_20260422/*/dpo/*/metrics.json`
- StarCoder2 base / SFT LoRA: `output/bigcode_code_eval_starcoder2_20260422/*/*/*/metrics.json`
- DeepSeek base / SFT LoRA: `output/bigcode_code_eval_deepseek_6_7b_fg_20260422/deepseek_coder_6_7b_instruct/*/*/metrics.json`

## 3. 目标任务结果

单位为百分比。`Delta` 为 DPO 相对 base 的变化，负数表示下降。

| Model | Base Dep. | DPO Dep. | Dep. Delta | Base Repl. | DPO Repl. | Repl. Delta | DPO Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 8.42 | 0.00 | -8.42 pp | 11.58 | 3.86 | -7.72 pp | 0.00 |
| StarCoder2-7B | 11.23 | 0.00 | -11.23 pp | 12.63 | 8.42 | -4.21 pp | 0.00 |
| StarCoder2-15B | 10.53 | 0.00 | -10.53 pp | 14.39 | 8.07 | -6.32 pp | 0.00 |
| DeepSeek-Coder-6.7B-Instruct | 6.67 | 1.05 | -5.61 pp | 24.91 | 28.42 | +3.51 pp | 0.00 |

### 3.1 目标任务观察

第一版 DPO 的 deprecated suppression 很强：

- StarCoder2-3B / 7B / 15B 的 deprecated usage 均降到 0.00%。
- DeepSeek-Coder-6.7B-Instruct 从 6.67% 降到 1.05%。
- 四个模型平均 deprecated usage 从 9.21% 降到 0.26%。

但是 replacement induction 不稳定：

- StarCoder2-3B replacement hit 从 11.58% 降到 3.86%。
- StarCoder2-7B replacement hit 从 12.63% 降到 8.42%。
- StarCoder2-15B replacement hit 从 14.39% 降到 8.07%。
- 只有 DeepSeek-Coder-6.7B-Instruct 的 replacement hit 从 24.91% 提升到 28.42%。
- 四个模型平均 replacement hit 从 15.88% 降到 12.19%。

因此，第一版 DPO 的目标任务结论不是“学会替换 API”，而是“学会避免 deprecated API”。在 StarCoder2 系列上，模型更可能采取 avoidance 策略：不生成 deprecated API，但也没有稳定生成目标 replacement API。

## 4. 通用代码能力结果

单位为 pass@1 百分比。`Delta` 为 DPO 相对 base 的变化。

| Model | HumanEval Base | HumanEval DPO | HumanEval Delta | MBPP Base | MBPP DPO | MBPP Delta |
|---|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 31.71 | 29.88 | -1.83 pp | 42.20 | 43.40 | +1.20 pp |
| StarCoder2-7B | 32.32 | 32.93 | +0.61 pp | 44.40 | 44.20 | -0.20 pp |
| StarCoder2-15B | 40.85 | 42.68 | +1.83 pp | 52.20 | 54.40 | +2.20 pp |
| DeepSeek-Coder-6.7B-Instruct | 73.17 | 69.51 | -3.66 pp | 57.80 | 60.20 | +2.40 pp |

### 4.1 通用能力观察

第一版 DPO 没有出现 SFT LoRA 那种明显通用能力崩坏：

- HumanEval 平均变化约为 -0.76 pp。
- MBPP 平均变化约为 +1.40 pp。
- StarCoder2-7B / 15B 的 HumanEval 不降反升。
- MBPP 除 StarCoder2-7B 小降 0.20 pp 外，其余模型均上升。

这说明第一版 DPO 的能力保持效果很好。问题主要不在通用能力，而在目标任务 replacement hit 不足。

## 5. 与 SFT LoRA 的能力保持对比

SFT LoRA 在目标任务上显著强于 DPO，但通用能力损失明显，尤其 MBPP。

| Model | SFT HumanEval Delta | DPO HumanEval Delta | SFT MBPP Delta | DPO MBPP Delta |
|---|---:|---:|---:|---:|
| StarCoder2-3B | -3.05 pp | -1.83 pp | -20.60 pp | +1.20 pp |
| StarCoder2-7B | +3.05 pp | +0.61 pp | -11.80 pp | -0.20 pp |
| StarCoder2-15B | +1.83 pp | +1.83 pp | -18.20 pp | +2.20 pp |
| DeepSeek-Coder-6.7B-Instruct | -25.00 pp | -3.66 pp | -36.40 pp | +2.40 pp |

这个对比显示：

- SFT LoRA 更强地诱导 replacement API，但会显著改变模型的通用代码生成分布。
- DPO 保持了 base model 的通用能力，但 preference signal 不足以稳定诱导 replacement API。

## 6. 解释与定位

第一版 DPO 的结果支持一个清晰 trade-off：

> Preference tuning preserves general coding ability much better than SFT, but pairwise preference alone is insufficient to induce correct replacement API usage.

更具体地说：

- DPO 对 deprecated API 有很强的 suppression 效果。
- DPO 对 replacement API 的正向诱导不足，尤其在 StarCoder2 系列上 replacement hit 反而下降。
- Exact match 全部为 0.00%，说明第一版 DPO 没有学习到稳定的目标补全模式。
- 通用能力保持很好，因此 DPO 是一个重要的 capability-preserving 对照，而不是当前最强的正向工程主方法。

论文定位建议：

- Plan A SFT LoRA: 主结果，目标任务最强，但需要承认通用能力损失。
- First-version DPO LoRA: 能力保持对照，说明单纯 preference tuning 可以避免能力崩坏，但目标行为控制不足。
- 后续改进方向应围绕 API 选择点增强 DPO 信号，例如 API-span DPO、replacement API CE anchor、hard negative avoidance pairs，而不是简单增加 epoch。

## 7. Paper-ready Summary

On the unified `mixed_sft_v1` benchmark, first-version DPO LoRA almost eliminates deprecated API usage for StarCoder2 models and substantially reduces it for DeepSeek-Coder. However, unlike SFT LoRA, it does not reliably increase replacement API usage: replacement hit rates decrease for all three StarCoder2 models and increase only for DeepSeek-Coder. In contrast, HumanEval and MBPP show little to no degradation, and MBPP often improves. These results indicate that DPO is effective as a capability-preserving deprecated-avoidance method, but pairwise preference tuning alone is insufficient for robust replacement API induction.
