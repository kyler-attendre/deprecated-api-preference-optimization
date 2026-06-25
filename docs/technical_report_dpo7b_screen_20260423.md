# StarCoder2-7B DPO Variant Screening

Date: 2026-04-23

Scope: preliminary screening on `StarCoder2-7B` only, used to decide which DPO variant should be expanded to other models.

## 1. Motivation

The first-version DPO LoRA showed a useful but incomplete behavior:

- It strongly suppresses deprecated API usage.
- It preserves HumanEval / MBPP much better than SFT LoRA.
- But on StarCoder2 models it often reduces replacement API hit rate, meaning it learns deprecated avoidance rather than correct replacement induction.

This screening tests whether adding an explicit replacement API anchor can fix the replacement-rate failure while preserving the general-code ability advantage of DPO.

## 2. Compared Methods

All methods use the same model and benchmark:

- Model: `/data/models/StarCoder/starcoder2-7b`
- Target benchmark: `data/mixed_sft_v1/mixed_sft_test.jsonl`
- Target samples: 285
- Prompt field: `version_prompt`
- General benchmarks: HumanEval pass@1 and MBPP pass@1 through BigCode evaluation harness

Compared methods:

| Method | Definition |
|---|---|
| Base | No adapter |
| SFT LoRA | Existing full-target SFT LoRA on `mixed_sft_v1` |
| Pure DPO | First-version DPO LoRA, full chosen/rejected preference only |
| API-span DPO | DPO preference computed only on replacement/deprecated API span |
| API-span DPO + anchor 0.1 | API-span DPO plus replacement API CE anchor |
| Full DPO + anchor 0.1 | Full-sequence DPO plus replacement API CE anchor |

Implementation notes:

- `--dpo-scope full`: preference loss uses full chosen/rejected completion.
- `--dpo-scope api_span`: preference loss uses replacement/deprecated API anchor span.
- `--api-anchor-weight 0.1`: adds supervised CE on replacement API anchor tokens.

## 3. Target-Task Results

Unit: percentage. Delta is relative to base.

| Method | Dep. Usage | Dep. Delta | Repl. Hit | Repl. Delta | Exact |
|---|---:|---:|---:|---:|---:|
| Base | 11.23 | - | 12.63 | - | 0.00 |
| SFT LoRA | 1.40 | -9.82 pp | 58.25 | +45.61 pp | 8.42 |
| Pure DPO | 0.00 | -11.23 pp | 8.42 | -4.21 pp | 0.00 |
| API-span DPO | 0.00 | -11.23 pp | 2.81 | -9.82 pp | 0.00 |
| API-span DPO + anchor 0.1 | 0.35 | -10.88 pp | 62.46 | +49.82 pp | 0.00 |
| Full DPO + anchor 0.1 | 0.00 | -11.23 pp | 63.51 | +50.88 pp | 0.00 |

Source files:

- `output/mixed_sft_v1_starcoder2_7b_card0_e3_compare/comparison_summary.json`
- `output/dpo_lora_mixed_sft_v1_20260422_eval/starcoder2_7b/comparison_summary.json`
- `output/dpo7b_screen_eval_20260423/api_span/comparison_summary.json`
- `output/dpo7b_screen_eval_20260423/api_span_anchor01/comparison_summary.json`
- `output/dpo7b_screen_eval_20260423/full_anchor01/comparison_summary.json`

## 4. General-Code Ability Results

Unit: pass@1 percentage. Delta is relative to base.

| Method | HumanEval | HumanEval Delta | MBPP | MBPP Delta |
|---|---:|---:|---:|---:|
| Base | 32.32 | - | 44.40 | - |
| SFT LoRA | 35.37 | +3.05 pp | 32.60 | -11.80 pp |
| Pure DPO | 32.93 | +0.61 pp | 44.20 | -0.20 pp |
| API-span DPO + anchor 0.1 | 30.49 | -1.83 pp | 45.80 | +1.40 pp |
| Full DPO + anchor 0.1 | 31.71 | -0.61 pp | 46.40 | +2.00 pp |

Source files:

- Base / SFT LoRA: `output/bigcode_code_eval_starcoder2_20260422/starcoder2_7b/*/*/metrics.json`
- Pure DPO: `output/bigcode_code_eval_dpo_20260422/starcoder2_7b/dpo/*/metrics.json`
- DPO screening: `output/bigcode_code_eval_dpo7b_screen_20260423/starcoder2_7b/*/*/metrics.json`

## 5. Key Observations

First, the replacement-rate problem of pure DPO is real on StarCoder2-7B:

- Deprecated usage: `11.23% -> 0.00%`
- Replacement hit: `12.63% -> 8.42%`

This confirms that pure DPO mainly learns to avoid deprecated APIs.

Second, API-span DPO alone is worse than pure DPO:

- Deprecated usage is still `0.00%`.
- Replacement hit falls to `2.81%`.

This suggests that narrowing the preference loss to only the API span removes too much sequence context and does not provide enough positive generation pressure.

Third, adding a replacement API CE anchor changes the behavior substantially:

- API-span DPO + anchor 0.1: replacement hit reaches `62.46%`.
- Full DPO + anchor 0.1: replacement hit reaches `63.51%`.

The anchor supplies the missing positive signal: DPO still discourages deprecated completions, while CE directly teaches the replacement API tokens.

Fourth, the full-sequence anchored variant is the best current candidate:

- It has the lowest deprecated usage: `0.00%`.
- It has the highest replacement hit: `63.51%`.
- Its HumanEval drop is small: `32.32% -> 31.71%`, only `-0.61 pp`.
- Its MBPP improves: `44.40% -> 46.40%`, `+2.00 pp`.

Compared with SFT LoRA, Full DPO + anchor 0.1 has stronger target-task metrics on deprecated usage and replacement hit, and avoids the large MBPP degradation seen in SFT LoRA.

## 6. Current Decision

For StarCoder2-7B, the best preliminary method is:

> Full-sequence DPO + replacement API CE anchor, with `api_anchor_weight = 0.1`.

This is the first DPO variant that simultaneously satisfies the three desired properties:

- reduce deprecated API usage;
- increase replacement API hit rate;
- preserve general code ability on HumanEval / MBPP.

## 7. Caveats

- This section records the original single-model screening result on StarCoder2-7B; cross-model claims should use the follow-up report below.
- Exact match remains `0.00%` for DPO variants, while SFT LoRA has `8.42%`. The DPO-anchor variants are better at API choice than exact completion imitation.
- The StarCoder2-7B screening by itself should be treated as model-selection evidence, not final cross-model evidence.

Update: the selected `Full DPO + anchor 0.1` method has now been expanded to StarCoder2-3B, StarCoder2-15B, and DeepSeek-Coder-6.7B-Instruct. See `output/technical_report_dpo_anchor_full01_cross_model_20260423.md` for the cross-model results.
