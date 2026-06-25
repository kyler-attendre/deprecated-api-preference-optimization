# DPO Anchor Cross-Model Results

Date: 2026-04-23

Scope: cross-model expansion of the StarCoder2-7B screened DPO variant:

> Full-sequence DPO + replacement API CE anchor, `api_anchor_weight = 0.1`.

This report records the results after applying the selected 7B method to StarCoder2-3B, StarCoder2-15B, and DeepSeek-Coder-6.7B-Instruct, together with the StarCoder2-7B screening result.

## 1. Method

The selected method was chosen from the StarCoder2-7B screening in `output/technical_report_dpo7b_screen_20260423.md`.

Training objective:

- DPO preference loss on the full chosen/rejected completion.
- A supervised CE anchor on replacement API tokens.
- Anchor weight: `0.1`.

The reason for adding the CE anchor is that first-version pure DPO strongly reduced deprecated API usage but often reduced replacement API hit rate on StarCoder2 models. The anchor adds direct positive pressure toward the intended replacement API while keeping the DPO objective as the main preference-tuning signal.

## 2. Outputs

Adapters:

- `output/dpo_anchor_full01_20260423/starcoder2_3b`
- `output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b`
- `output/dpo_anchor_full01_20260423/starcoder2_15b`
- `output/dpo_anchor_full01_20260423/deepseek_coder_6_7b_instruct`

Target-task evaluation:

- `output/dpo_anchor_full01_20260423_eval/starcoder2_3b/comparison_summary.json`
- `output/dpo7b_screen_eval_20260423/full_anchor01/comparison_summary.json`
- `output/dpo_anchor_full01_20260423_eval/starcoder2_15b/comparison_summary.json`
- `output/dpo_anchor_full01_20260423_eval/deepseek_coder_6_7b_instruct/comparison_summary.json`

General-code evaluation:

- `output/bigcode_code_eval_dpo_anchor_full01_20260423/starcoder2_3b/full_anchor01/*/metrics.json`
- `output/bigcode_code_eval_dpo7b_screen_20260423/starcoder2_7b/full_anchor01/*/metrics.json`
- `output/bigcode_code_eval_dpo_anchor_full01_20260423/starcoder2_15b/full_anchor01/*/metrics.json`
- `output/bigcode_code_eval_dpo_anchor_full01_20260423/deepseek_coder_6_7b_instruct/full_anchor01/*/metrics.json`

## 3. StarCoder2-7B Screening Decision

Unit: percentage. Delta is relative to base.

| Method | Dep. Usage | Dep. Delta | Repl. Hit | Repl. Delta | HumanEval | HE Delta | MBPP | MBPP Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 11.23 | - | 12.63 | - | 32.32 | - | 44.40 | - |
| Pure DPO | 0.00 | -11.23 pp | 8.42 | -4.21 pp | 32.93 | +0.61 pp | 44.20 | -0.20 pp |
| API-span DPO | 0.00 | -11.23 pp | 2.81 | -9.82 pp | n/a | n/a | n/a | n/a |
| API-span DPO + anchor 0.1 | 0.35 | -10.88 pp | 62.46 | +49.82 pp | 30.49 | -1.83 pp | 45.80 | +1.40 pp |
| Full DPO + anchor 0.1 | 0.00 | -11.23 pp | 63.51 | +50.88 pp | 31.71 | -0.61 pp | 46.40 | +2.00 pp |

Decision: `Full DPO + anchor 0.1` is the best screened variant on StarCoder2-7B because it fixes the pure-DPO replacement-rate drop while keeping HumanEval / MBPP close to or above base.

## 4. Target-Task Cross-Model Results

Benchmark: `data/mixed_sft_v1/mixed_sft_test.jsonl`, 285 samples.

Unit: percentage. Delta is relative to each model's base result in the same evaluation run.

| Model | Base Dep. | Anchor DPO Dep. | Dep. Delta | Base Repl. | Anchor DPO Repl. | Repl. Delta | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 8.42 | 0.35 | -8.07 pp | 11.58 | 57.54 | +45.96 pp | 0.00 |
| StarCoder2-7B | 11.23 | 0.00 | -11.23 pp | 12.63 | 63.51 | +50.88 pp | 0.00 |
| StarCoder2-15B | 12.63 | 0.70 | -11.93 pp | 16.49 | 62.46 | +45.96 pp | 0.00 |
| DeepSeek-Coder-6.7B-Instruct | 6.67 | 0.70 | -5.96 pp | 24.91 | 48.42 | +23.51 pp | 0.00 |

Average over the four models:

- Deprecated usage: `9.74% -> 0.44%`, delta `-9.30 pp`.
- Replacement hit: `16.40% -> 57.98%`, delta `+41.58 pp`.

The anchored DPO variant solves the main failure mode of first-version pure DPO. It no longer merely avoids deprecated APIs; it also substantially increases replacement API hits on all evaluated models.

## 5. General-Code Ability

Benchmarks: HumanEval pass@1 and MBPP pass@1 through BigCode evaluation harness.

Unit: pass@1 percentage. Delta is relative to base.

| Model | Base HE | Anchor DPO HE | HE Delta | Base MBPP | Anchor DPO MBPP | MBPP Delta |
|---|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 31.71 | 28.05 | -3.66 pp | 42.20 | 42.20 | +0.00 pp |
| StarCoder2-7B | 32.32 | 31.71 | -0.61 pp | 44.40 | 46.40 | +2.00 pp |
| StarCoder2-15B | 40.85 | 45.73 | +4.88 pp | 52.20 | 56.20 | +4.00 pp |
| DeepSeek-Coder-6.7B-Instruct | 73.17 | 72.56 | -0.61 pp | 57.80 | 59.20 | +1.40 pp |

Average over the four models:

- HumanEval: `44.51% -> 44.51%`, average delta approximately `+0.00 pp`.
- MBPP: `49.15% -> 51.00%`, average delta `+1.85 pp`.

This is materially different from the earlier SFT LoRA pattern, where the target-task improvement came with large MBPP degradation, especially on StarCoder2-3B, StarCoder2-15B, and DeepSeek-Coder-6.7B-Instruct.

## 6. Comparison With First-Version Pure DPO

First-version pure DPO was useful as a capability-preserving deprecated-avoidance method, but it did not reliably induce replacement API usage.

Target-task comparison:

| Model | Pure DPO Dep. | Anchor DPO Dep. | Pure DPO Repl. | Anchor DPO Repl. |
|---|---:|---:|---:|---:|
| StarCoder2-3B | 0.00 | 0.35 | 3.86 | 57.54 |
| StarCoder2-7B | 0.00 | 0.00 | 8.42 | 63.51 |
| StarCoder2-15B | 0.00 | 0.70 | 8.07 | 62.46 |
| DeepSeek-Coder-6.7B-Instruct | 1.05 | 0.70 | 28.42 | 48.42 |

The CE anchor trades a very small amount of deprecated suppression on 3B / 15B for a large replacement-hit gain. For 7B it keeps deprecated usage at 0.00%. For DeepSeek it improves both deprecated suppression and replacement hit compared with pure DPO.

## 7. Comparison With SFT LoRA

SFT LoRA remains the only method with non-zero exact-match target rate, because it directly imitates full target completions. However, for the core API-choice metrics, anchored DPO is competitive or stronger while preserving general code ability better.

Target-task comparison on StarCoder2:

| Model | SFT Dep. | Anchor DPO Dep. | SFT Repl. | Anchor DPO Repl. | SFT Exact | Anchor DPO Exact |
|---|---:|---:|---:|---:|---:|---:|
| StarCoder2-3B | 1.40 | 0.35 | 56.84 | 57.54 | 7.72 | 0.00 |
| StarCoder2-7B | 1.40 | 0.00 | 58.25 | 63.51 | 8.42 | 0.00 |
| StarCoder2-15B | 1.05 | 0.70 | 52.63 | 62.46 | 10.53 | 0.00 |

General-code ability comparison:

| Model | SFT HE Delta | Anchor DPO HE Delta | SFT MBPP Delta | Anchor DPO MBPP Delta |
|---|---:|---:|---:|---:|
| StarCoder2-3B | -3.05 pp | -3.66 pp | -20.60 pp | +0.00 pp |
| StarCoder2-7B | +3.05 pp | -0.61 pp | -11.80 pp | +2.00 pp |
| StarCoder2-15B | +1.83 pp | +4.88 pp | -18.20 pp | +4.00 pp |
| DeepSeek-Coder-6.7B-Instruct | -25.00 pp | -0.61 pp | -36.40 pp | +1.40 pp |

The clearest advantage of anchored DPO is MBPP preservation. SFT LoRA substantially damages MBPP, while anchored DPO keeps MBPP stable or improves it in all four evaluated models.

## 8. Interpretation

The results support a sharper method-level conclusion than the first DPO report:

> DPO alone is capability-preserving but under-specifies the positive replacement behavior. Adding a small replacement API CE anchor supplies the missing positive signal and yields both deprecated suppression and replacement induction without large general-code degradation.

This method has a different profile from SFT LoRA:

- SFT LoRA learns target completions and can produce exact matches, but it substantially shifts the model's general code distribution.
- Pure DPO preserves general ability but can learn avoidance rather than replacement.
- Anchored DPO keeps the capability-preservation benefit of DPO while recovering strong replacement API induction.

## 9. Paper-Ready Conclusion

On the unified `mixed_sft_v1` benchmark, full-sequence DPO with a lightweight replacement API CE anchor reduces deprecated API usage from `9.74%` to `0.44%` on average across StarCoder2-3B, StarCoder2-7B, StarCoder2-15B, and DeepSeek-Coder-6.7B-Instruct. At the same time, it increases replacement API hit rate from `16.40%` to `57.98%`. Unlike SFT LoRA, this improvement does not come with a large loss in general coding ability: average HumanEval is unchanged and average MBPP improves by `1.85 pp`.

This makes anchored DPO the current strongest capability-preserving positive-engineering method in `05_positive_engineering`. It should be presented as an improved Plan A variant or as a DPO-based Plan A+ method, distinct from the original full-target SFT LoRA.

## 10. Caveats

- Exact match remains `0.00%` for anchored DPO. The method optimizes API choice rather than full target imitation.
- The StarCoder2-15B target base rate differs from the earlier SFT report because the evaluation run here used `max_length = 384`; comparison should rely on each method's in-run base for deltas.
- These results use one anchor weight, `0.1`, selected from the StarCoder2-7B screen. No broader hyperparameter sweep is claimed here.
