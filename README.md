# Replication Package: Deprecated API Recommendation in Code LLMs via Preference Optimization

This is the replication package for a submission titled *Mitigating and
Demystifying Deprecated API Recommendation in Code LLMs via Preference
Optimization*.

This package covers **RQ1–RQ3** of the paper. RQ4 (the version-prefix
robustness study) is tracked separately and is intentionally excluded here.
All files are copies; nothing here is a symlink back into the authors'
working repository.

This repository is intended for double-anonymous review (e.g. via
anonymous.4open.science). It contains no author names, institution names, or
personal file-path identifiers — all absolute paths have been rewritten to
generic placeholders (`/workspace`, `/data/models`, `/opt/conda`).
Intermediate training checkpoints (optimizer/scheduler/RNG state) and
out-of-scope intermediate data (raw scraped candidate pools, unused
experiment families such as `rerank_*`/`repair_sft_*`/`consistency_*`) have
been removed; only the final adapter weights and the data actually consumed
by the in-scope RQ1–RQ3 pipelines are kept. `code/` has been pruned to only
the scripts that are actually invoked by the experiments reported below;
no exploratory/abandoned code paths are included.

**Terminology note.** The proposed method is referred to as **SFT+DPO**
throughout this README (plain DPO augmented with a cross-entropy anchor
loss, `api_anchor_weight = 0.1`, over the replacement-API token range — see
"Method" below). Script, config, and directory names in `code/` and
`results/` still use the internal identifiers `cerdpo`/`anchor` for this
same method; this is a naming artifact of the original codebase, not a
different method.

## Method

Three fine-tuning recipes are compared, all injecting LoRA adapters into
the four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of
every transformer layer:

- **SFT** — supervised fine-tuning on version-consistent completions only
  (no contrastive signal). 3 epochs, lr `1e-4`.
- **DPO** — standard preference optimization (Rafailov et al.) on
  (chosen, rejected) pairs that differ only at the API token span.
  1 epoch, lr `5e-5`, LoRA rank 8, β = 0.1.
- **SFT+DPO** (paper/internal name `CER-DPO`/`anchor`) — DPO plus a
  cross-entropy anchor term restricted to the replacement-API token range,
  weight `λ = 0.1`. Same hyperparameters as DPO otherwise. `λ` was selected
  once on StarCoder2-7B and reused unchanged on every other model.

Training/eval data is **DepPref**: 2,408 train / 304 val / 285 test
preference triples built from a 145-pair deprecated→replacement API mapping
(Wang et al.), at `code/positive_engineering/data/mixed_sft_v1/`.

## Models

| Model | Layers | Used for |
|---|---|---|
| StarCoder2-3B | 30 | RQ1, RQ3 |
| StarCoder2-7B | 32 | RQ1–RQ3 (primary) |
| StarCoder2-15B | 40 | RQ1, RQ3 |
| DeepSeek-Coder-6.7B-Instruct | 32 | RQ1, RQ2 |
| Qwen2.5-Coder-3B-Instruct | — | RQ1 |
| Qwen2.5-Coder-7B-Instruct | — | RQ1, RQ2 |
| Qwen2.5-Coder-14B-Instruct | — | RQ1 |

Base model weights are **not** redistributed in this repository; only the
trained LoRA adapters (`adapter_model.safetensors` + `adapter_config.json`)
are included under `results/`.

## Layout

```
code/
  positive_engineering/   training (SFT/DPO/SFT+DPO) + behavioral eval + RQ1/RQ3 analysis scripts
    src/                  dataset_utils.py, dpo_training.py — shared library code
    scripts/              entry points, see "Reproduction" below
    configs/              example LoRA config
    data/mixed_sft_v1/    DepPref dataset (train/val/test + per-library splits)
  mechanism/              RQ2: Logit Lens / Tuned Lens / Activation-Difference Lens / OV-circuit
    src/                  lens_analysis.py, adl_compare.py, variant_compare.py
    scripts/              entry points, see "Reproduction" below
    tests/
  correct_emergence/      supplementary: base-model decision-emergence-layer analysis (Discussion section)
    src/, scripts/, PLAN.md
results/
  rq1_effectiveness/      Table 1 (DUR/RHR), Table 2 (HumanEval/MBPP Δ), Table 3 (retention), library radar
  rq2_mechanism/          Table 4 (final-layer margin), stable decision depth, Table 5 (ADL), Table 6 (OV-circuit)
  rq3_parameter_localization/  Frobenius-norm heatmap, magnitude-pruning curve, Table 7 (restricted retraining), module ablations
  emergence_analysis/     two dated variants of the base-model emergence-layer analysis, see note below
requirements.txt
```

## Setup

```bash
conda create -n sftdpo python=3.10 -y
conda activate sftdpo
pip install -r requirements.txt
```

Base model weights (StarCoder2-{3B,7B,15B}, Qwen2.5-Coder-{3B,7B,14B}-Instruct,
DeepSeek-Coder-6.7B-Instruct) must be downloaded separately from their
respective official sources and are not included here.

---

## Detailed results

All numbers below are exactly as reported in the paper, with a pointer to
the script that produced them and the raw output directory under `results/`.

### RQ1 — Effectiveness across code LLMs

**Approach.** Train SFT, DPO, SFT+DPO on DepPref; run greedy line-level
completion on the 285-sample test set; score against the 145-pair mapping.

Training: `code/positive_engineering/scripts/train_lora.py` (SFT),
`train_dpo_lora.py` (DPO / SFT+DPO, launched via `run_dpo_lora_nohup.sh`,
`run_qwen_mixed_sft_v1_nohup.sh`, `run_deepseek_mixed_sft_v1_nohup.sh`).
Eval: `eval_compare_lora.py`.

**Table 1 — DUR / RHR (%) per model.** Bold = best RHR per model.
DPO was not evaluated on Qwen2.5-Coder-3B/14B (marked `—`).

| Model | Base DUR | Base RHR | SFT DUR | SFT RHR | DPO DUR | DPO RHR | SFT+DPO DUR | SFT+DPO RHR |
|---|---|---|---|---|---|---|---|---|
| StarCoder2-3B | 8.42 | 11.58 | 1.40 | 56.84 | 0.00 | 3.86 | 0.35 | **57.54** |
| StarCoder2-7B | 11.23 | 12.63 | 1.40 | 58.25 | 0.00 | 8.42 | 0.00 | **63.51** |
| StarCoder2-15B | 10.53 | 14.39 | 1.05 | 52.63 | 0.00 | 8.07 | 0.70 | **62.46** |
| DeepSeek-Coder-6.7B | 6.67 | 24.91 | 0.70 | 54.74 | 1.05 | 28.42 | 0.70 | **48.42** |
| Qwen2.5-Coder-3B | 8.07 | 19.30 | 0.70 | 55.09 | — | — | 0.35 | **56.14** |
| Qwen2.5-Coder-7B | 5.96 | 24.21 | 1.05 | 52.28 | 0.35 | 14.74 | 0.00 | **59.30** |
| Qwen2.5-Coder-14B | 5.96 | 27.37 | 0.70 | 46.67 | — | — | 0.70 | **54.39** |
| **Overall** (7 models for Base/SFT/SFT+DPO, 5 for DPO) | 8.12 | 19.20 | 1.00 | 53.79 | 0.28 | 12.70 | **0.40** | **57.39** |
| **Overall (DPO subset, 5 models)** | 8.56 | 17.54 | 1.12 | 54.95 | 0.28 | 12.70 | 0.35 | 58.25 |

Raw run directories: `results/rq1_effectiveness/mixed_sft_v1_starcoder2_{3b,7b,15b}_*` (SFT),
`dpo_lora_mixed_sft_v1_20260422(_eval)`, `dpo_lora_mixed_sft_v1_qwen_20260511(_eval)` (DPO),
`dpo_anchor_full01_20260423(_eval)`, `dpo_anchor_full01_qwen_20260423(_eval)`,
`dpo7b_screen_full_anchor01_20260423`, `dpo7b_screen_eval_20260423` (SFT+DPO / `anchor`).
`dpo_baseline_refill_20260612(_eval)` contains a later, supplementary DPO run on
Qwen2.5-Coder-3B/7B/14B that fills in the `—` cells above for exploratory reference;
**it is not part of the published Table 1**, which reports DPO as not evaluated on
Qwen-3B/14B.

**Failure mode of DPO.** On StarCoder2-3B/7B/15B and Qwen2.5-Coder-7B, DPO drives
DUR to ≤0.35% but *also* cuts RHR by 33–67% relative to base — "deprecated-avoidance
without replacement-induction", consistent with likelihood-displacement theory
(Razin et al.): DPO only enforces a *relative* preference, not an absolute one.

**Library-level breakdown** (Figure: per-library RHR radar, StarCoder2 series).
Base RHR on `pytorch`/`tensorflow` is 8–29%; SFT+DPO raises this to 72–84%.
Script: `plot_library_radar.py` / `plot_library_radar_panel.py`.
Raw data: `results/rq1_effectiveness/library_radar_20260509`.

**Table 2 — HumanEval / MBPP pass@1 Δ (%) vs. base, greedy decoding.**

| Model | SFT HumanEval Δ | SFT+DPO HumanEval Δ | SFT MBPP Δ | SFT+DPO MBPP Δ |
|---|---|---|---|---|
| StarCoder2-3B | −3.05 | −3.66 | −20.60 | 0.00 |
| StarCoder2-7B | +3.05 | −0.61 | −11.80 | +2.00 |
| StarCoder2-15B | +1.83 | +4.88 | −18.20 | +4.00 |
| DeepSeek-Coder-6.7B | −25.00 | −0.61 | −36.40 | +1.40 |

Script: `evaluate_mbpp_official.py`, `generate_official_code_samples.py`,
`run_official_code_eval_nohup.sh` (wraps the BigCode evaluation harness).
Raw data: `results/rq1_effectiveness/official_code_eval_20260421_full_g*`,
`bigcode_code_eval_*`.

**Table 3 — Retention on a 1,596-sample, 39-API benchmark disjoint from the
training mapping** (every retained sample has 100% base-model accuracy by
construction).

| Model | Retention APIs | Samples | SFT+DPO Acc. (%) |
|---|---|---|---|
| StarCoder2-3B | 26 | 520 | 72.88 |
| StarCoder2-7B | 27 | 540 | 72.22 |
| StarCoder2-15B | 27 | 540 | 85.37 |
| DeepSeek-Coder-6.7B | 25 | 500 | **90.80** |

On the StarCoder2-7B set, DPO retains only 51.3% vs. SFT+DPO's 72.2% (Δ = 20.9%),
heterogeneous across libraries (numpy Δ = 21.8%, pytorch Δ = 25.0%, sklearn
Δ = 5.0%, pandas Δ = 2.5%).

Scripts: `build_multilib_retention_candidates.py`, `select_retention_set.py`,
`split_by_library.py`, `analyze_retention_shift.py`.
Raw data: `results/rq1_effectiveness/multilib_retention_20260427`,
`results/rq1_effectiveness/retention_shift_20260511`.

**Answer to RQ1.** SFT+DPO is the only recipe that simultaneously suppresses
deprecated usage, induces replacement usage, preserves general code ability,
and retains non-target completions. DPO fails on the replacement-induction
axis on the StarCoder2 series; SFT matches SFT+DPO behaviorally but degrades
general code ability (e.g. DeepSeek-Coder loses 36.4% on MBPP under SFT).

### RQ2 — Layer-wise mechanism (StarCoder2-7B primary; cross-model replication)

**Approach.** Logit Lens / Tuned Lens / Activation-Difference Lens (ADL) /
OV-circuit attribution on the 35-sample mechanism set (union of repair +
consistency test splits).

Training (Tuned Lens translator): `code/mechanism/scripts/train_tuned_lens.py`.
Lens comparison: `run_lens_compare.py`, `run_lens_compare_variants.py`
(launched via `run_mechanism_model.sh` / `run_mechanism_compare_only.sh` /
`launch_full_mechanism_20260427.sh` / `launch_compare_only_jsd_20260427.sh` /
`run_triplet_batch.sh`). ADL: `run_activation_difference_lens.py`
(via `run_adl_batch.sh`). OV-circuit: `ov_circuit_flip.py`, `ov_dla_flip.py`,
`ov_dla_multimodel.py`, `ov_dla_prefix.py`.

**Table 4 — Final-layer replacement margin on StarCoder2-7B** (35-sample
mechanism set average).

| Lens | Base M_seq | SFT+DPO M_seq | Base M_first | SFT+DPO M_first |
|---|---|---|---|---|
| Logit | −0.05 | +32.88 | −2.43 | +21.13 |
| Tuned | −0.62 | +60.76 | −3.30 | +43.91 |

**Stable decision depth.** StarCoder2 series: base commits at layer 24.27
on average; SFT+DPO reduces this to 0.97. Cross-series (StarCoder2-7B,
DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B): 20.67 → 6.57. DPO attains an even
larger local margin (23.49 vs. SFT+DPO's 18.24 at the final layer) but fails
to commit early — it avoids the deprecated token without ever reaching the
replacement, consistent with likelihood-displacement (Razin et al.).

**Table 5 — Activation-Difference Lens on neutral prompts** (no API/version
tokens). `rand` = NeutralText, `code` = NeutralCode, Δ = Readout_code − Readout_rand.

| Setting | Norm_rand | Readout_rand | Readout_code | Δ |
|---|---|---|---|---|
| DPO | 28.69 | 0.408 | 0.518 | 0.110 |
| SFT+DPO | 27.90 | 0.416 | 0.543 | **0.127** |

DPO leaves a stronger global trace (higher Norm on prose); SFT+DPO leaves a
sharper code-context-conditioned trace (larger Δ).

**Table 6 — OV-circuit direct logit attribution**, final-layer
Δℓ = ℓ(replacement) − ℓ(deprecated), two linalg migration cases.

| Model | svd→linalg.svd Base | svd→linalg.svd SFT+DPO | qr→linalg.qr Base | qr→linalg.qr SFT+DPO |
|---|---|---|---|---|
| StarCoder2-3B | −3.22 | **+33.25** | −3.13 | **+31.19** |
| StarCoder2-7B | −2.06 | **+35.03** | −3.25 | **+32.75** |
| StarCoder2-15B | −5.52 | **+36.31** | −4.94 | **+32.81** |
| DeepSeek-Coder-6.7B | −6.77 | **+39.28** | −0.27 | **+39.63** |

Dominant heads: L31H2, L31H1 (largest causal flip deltas on both linalg
cases), L25H33 third.

Raw data: `results/rq2_mechanism/full_mechanism_20260427` (Logit/Tuned Lens,
7 models), `triplet_mechanism_20260506` (DPO/SFT+DPO/base triplet),
`adl_neutral_20260506` (Table 5), `ov_dla_results.json` + per-model variants
and `ov_flip_results.json` (Table 6).

**Answer to RQ2.** SFT+DPO stores the replacement preference in
mid-to-late attention parameters, resolved in the earliest layers of the
forward pass. DPO encodes an even larger local preference but writes a
uniformly distributed global bias that pulls the generation trajectory away
from the decision point before the preference can be expressed; the
cross-entropy anchor breaks this coupling.

### RQ3 — Parameter localization (StarCoder2-7B primary; 3B/15B replication)

**Approach.** (i) Post-hoc magnitude pruning of the LoRA effective delta,
no retraining. (ii) Restricted retraining: re-run SFT+DPO with LoRA
injection limited to a layer/module subset. (iii) Cross-scale replication
on StarCoder2-3B/15B. All evaluated on the 285-sample test set.

Scripts: `analyze_lora_delta.py`, `compute_gradient_attribution.py`,
`plot_delta_heatmap.py` (Frobenius-norm heatmap); `eval_sparse_lora_delta.py`
+ `run_sparse_extended_eval.sh` (magnitude pruning); `train_dpo_lora_restricted.py`
+ `run_layer_range_ablation.sh` (restricted retraining); `train_dpo_lora.py`
+ `run_module_ablation.sh` (module ablations); `run_ablation_eval.sh`,
`run_per_library_cerdpo.sh` / `launch_per_library_cerdpo.sh` (eval).

**Frobenius-norm distribution.** The SFT+DPO LoRA adapter on StarCoder2-7B
has 128 injected modules (32 layers × 4 projections, 1.51B effective delta
parameters). DPO and SFT+DPO have nearly identical mean norms (0.225 vs.
0.228), both concentrated on layers 17–31 of the O projection (2–3× the
network-wide mean). Raw data: `results/rq3_parameter_localization/lora_delta_20260509`.

**Post-hoc magnitude pruning** (retained weight fraction k%, RHR on
StarCoder2-7B):

| k% | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| DPO RHR | 28.1 | 26.3 | 23.5 | 18.2 | 14.4 | 8.8 | 8.4 |
| SFT+DPO RHR | 40.0 | 50.5 | 56.5 | 62.1 | 64.6 | 64.2 | 63.5 |

SFT+DPO is essentially insensitive to pruning down to k=30% and retains
97.8% of full performance at k=20%; DPO's RHR *rises* under pruning, peaking
at k=5% (≈4× its complete-adapter value) — evidence of a sizeable
replacement-suppressing component that pruning strips away first.
Raw data: `results/rq3_parameter_localization/sparse_delta_eval_20260509`,
`sparse_delta_extended_20260511`.

**Table 7 — Restricted retraining** (layers 17–31, `{q_proj, o_proj}` only,
on StarCoder2-7B; proportional layer band on 3B/15B).

| Model | Restricted params | DUR | RHR | Rel. RHR (vs. full SFT+DPO) |
|---|---|---|---|---|
| StarCoder2-3B | ≈264M | 0.0% | 44.2% | 76.8% |
| StarCoder2-7B | ≈637M | 0.0% | 49.8% | 78.4% |
| StarCoder2-15B | ≈1,434M | 0.7% | 50.2% | **80.4%** |

Narrowing further to `{o_proj}` alone on StarCoder2-7B (≈318M params):
DUR = 0.0%, RHR = 44.6% (70.2% of full performance).
Raw data: `results/rq3_parameter_localization/cerdpo_restricted_{3b,15b}_20260511`,
`anchor_restricted_layers_20260509`, `dpo_restricted_layers_20260509`,
`cerdpo_restricted_L{20,25,28}_31_20260511`, `cerdpo_o17_31_20260511`,
`cerdpo_top11_oprojs_20260511`, `cerdpo_lora_o_proj_toplayers_20260514_tty(_eval)`,
`dpo_lora_o_proj_toplayers_20260514(_manual,_tty,_tty_eval)`,
`restricted_layer_eval_20260511`, `ablation_eval_20260511`.

**Module-level ablations** on StarCoder2-7B (full layer range, restricted
module subsets): MLP-only reaches RHR = 61.1% (−2.4% vs. attention-only
default); attention+MLP gains a further +2.5%. `{o_proj}` alone (all 32
layers) reaches RHR = 58.6%; removing `o_proj` (`{q_proj,k_proj,v_proj}`
only) drops RHR to 40.7% (−22.8%) — O projection is the single most
concentrated carrier of the effect.
Raw data: `results/rq3_parameter_localization/cerdpo_{attn_mlp,mlp_only,kv_only,o_only,qkv_no_o}_20260511`.

**Answer to RQ3.** Frobenius-norm distribution, magnitude pruning, and
restricted retraining converge on the same localized subspace:
approximately the upper 47% of layers on the O projection, with a
quantifiable but non-dominant complement from Q. This recovers ~4/5 of
complete-adapter performance across all three StarCoder2 scales.

### Supplementary: base-model decision-emergence-layer analysis

`results/emergence_analysis/` and `code/correct_emergence/` support a
Discussion-section observation about where in the network the base model's
deprecated-vs-replacement decision becomes legible (independent of any
fine-tuning recipe), via top-k trajectory tracing at the post-API decision
token. Two dated variants are included:

- `starcoder2_7b_PAPER_CURRENT_prefix_uncorrected/` — the run matching the
  numbers currently cited in the paper text (Logit Lens mean emergence
  layer 19.1, median 18, P25–P75 18–20; mean saturation layer 24.8, median
  24, P25–P75 21–29).
- `starcoder2_7b_CORRECTED_decision_after_dot/` — a corrected re-run fixing
  a decision-token-alignment bug (selecting the token immediately after the
  attribute-access dot rather than the dot itself): mean emergence layer
  19.2, median 20, P25–P75 18–21; mean saturation layer 27.9, median 28,
  P25–P75 27–31.

Both are included for transparency; the corrected variant is the
methodologically accurate one.

---

## Reproduction

Example end-to-end commands (StarCoder2-7B, SFT+DPO):

```bash
# 1. Train
python code/positive_engineering/scripts/train_dpo_lora.py \
  --model-name-or-path /path/to/starcoder2-7b \
  --train-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_val.jsonl \
  --api-anchor-weight 0.1 --output-dir runs/sftdpo_starcoder2_7b

# 2. Evaluate (DUR/RHR vs. base)
python code/positive_engineering/scripts/eval_compare_lora.py \
  --model-name-or-path /path/to/starcoder2-7b \
  --adapter-dir runs/sftdpo_starcoder2_7b \
  --test-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_test.jsonl \
  --output-dir runs/sftdpo_starcoder2_7b_eval
```

See each script's `--help` for the full argument list; `run_*_nohup.sh` /
`launch_*.sh` files under `code/*/scripts/` show the exact arguments used to
produce the included `results/` artifacts.

## Known reproduction variance

Re-running DPO on Qwen2.5-Coder-7B (`dpo_baseline_refill_20260612`) reproduced
the DUR exactly (0.35%) but obtained RHR = 17.19% versus the paper's reported
14.74%. Both values are consistent with the paper's qualitative claim (DPO
RHR well below SFT+DPO's 59.30%); the discrepancy reflects ordinary
run-to-run variance in DPO training, not a methodological difference.
