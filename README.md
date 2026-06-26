# Replication Package: Deprecated API Recommendation in Code LLMs via Preference Optimization

This is the replication package for a submission titled *Mitigating and
Demystifying Deprecated API Recommendation in Code LLMs via Preference
Optimization*.

This package covers all three research questions of the paper (RQ1–RQ3).
All files are copies; nothing here is a symlink back into the authors'
working repository.

## Anonymity note

This repository is prepared for **ICSE 2027 double-anonymous review**. It
contains no author names, institution names, acknowledgements, funding
information, or personal file-path identifiers — all absolute paths have
been rewritten to generic placeholders (`/workspace`, `/data/models`,
`/opt/conda`). `LICENSE` and `CITATION.cff` are placeholders for the review
period; real author/citation/DOI information will be added after the
review concludes. Intermediate training checkpoints (optimizer/scheduler/RNG
state) and out-of-scope intermediate data (raw scraped candidate pools,
unused experiment families such as `rerank_*`/`repair_sft_*`/`consistency_*`)
have been removed; only the final adapter weights and the data actually
consumed by the in-scope RQ1–RQ3 pipelines are kept. `code/` has been pruned
to only the scripts that are actually invoked by the experiments reported
below; no exploratory/abandoned code paths are included.

**Terminology note.** The proposed method is referred to as **SFT+DPO**
throughout this README and throughout the paper (plain DPO augmented with a
cross-entropy regularization term, weight `λ = 0.1`, over the replacement-API
token positions — see "Method" below). All `results/` directories use this
name. Two script filenames in `code/positive_engineering/scripts/`
(`run_per_library_cerdpo.sh`, `launch_per_library_cerdpo.sh`) still carry the
internal codename `cerdpo` for the same method; this is a naming artifact of
the original codebase, not a different method.

## Method

Three fine-tuning methods are compared, all using LoRA injected into the
four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of every
transformer layer:

- **SFT** — supervised fine-tuning on version-consistent completions only.
  3 epochs, lr `1e-4`, LoRA rank 16.
- **DPO** — standard preference optimization (Rafailov et al.) on
  (chosen, rejected) completion pairs that differ only at the API call.
  1 epoch, lr `5e-5`, LoRA rank 8, β = 0.1.
- **SFT+DPO** — keeps all DPO hyperparameters and adds a cross-entropy regularization
  term restricted to the replacement-API token positions, weight `λ = 0.1`.
  `λ` was selected once on StarCoder2-7B and reused unchanged on every other
  model.

Training/eval data is **DepPref**: 2,408 train / 304 val / 285 test
prompt-completion pairs built from a 145-pair deprecated→replacement API
mapping (Wang et al.), at `code/positive_engineering/data/mixed_sft_v1/`.

## Models

Seven code LLMs, three model series, four parameter scales: StarCoder2-3B/7B/15B,
DeepSeek-Coder-6.7B-Instruct, Qwen2.5-Coder-3B/7B/14B-Instruct. All seven are
used for the RQ1 effectiveness evaluation. StarCoder2-7B is the primary model
for the RQ2 parameter-localization and RQ3 layer-wise analyses, with
StarCoder2-3B/15B providing cross-scale replication of the RQ2 restricted-
retraining result and appearing in the RQ3 OV-circuit table; DeepSeek-Coder-6.7B
and Qwen2.5-Coder-7B additionally appear in the RQ3 cross-series stable-
decision-depth and OV-circuit results. Base model weights are **not**
redistributed here; only the trained LoRA adapters
(`adapter_model.safetensors` + `adapter_config.json`) are included under
`results/`.

## Layout

```
.
├── code/
│   ├── positive_engineering/        training (SFT/DPO/SFT+DPO) + behavioral eval + RQ2 parameter-localization scripts
│   │   ├── src/                     dataset_utils.py, dpo_training.py — shared library code
│   │   ├── scripts/                 entry points, see "Reproduction" below
│   │   └── data/mixed_sft_v1/       DepPref dataset (train/val/test + per-library splits, see its README.md)
│   ├── mechanism/                   RQ3: Logit Lens / Tuned Lens / Activation-Difference Lens / OV-circuit
│   │   ├── src/                     lens_analysis.py, adl_compare.py, variant_compare.py
│   │   ├── scripts/                 entry points, see "Reproduction" below
│   │   └── tests/
│   └── correct_emergence/           supplementary: base-model decision-emergence-layer analysis (Discussion section)
│       ├── src/
│       └── scripts/
├── results/
│   ├── rq1_effectiveness/           Table 1 (DUR/RHR), Table 2 (HumanEval/MBPP Δ), Table 3 (retention), library radar
│   ├── rq2_parameter_localization/  module-norm heatmap, magnitude-pruning curve, restricted-retraining table, module ablations
│   ├── rq3_layerwise_analysis/      final-layer margin table, stable decision depth, ADL table, OV-circuit table
│   └── emergence_analysis/          two dated variants of the base-model emergence-layer analysis, see note below
├── requirements.txt
├── reproduce.sh                      smoke test, see "Quick start" below
├── LICENSE
├── CITATION.cff
└── README.md
```

## Environment

- OS: Linux (tested on Ubuntu); any OS that runs PyTorch + CUDA should work.
- Python 3.10.
- GPU: required for training and for any LoRA adapter evaluation; not
  required for the smoke test below or for inspecting `results/` directly.
- VRAM: ≈8GB for the 3B models up to ≈32GB for the 14B/15B models, fp16/bf16.
- Disk: this repository is ≈1.6GB; add ≈15-60GB per base model you download.

**Minimal environment** (browse `results/`, run the smoke test): just
`pip install -r requirements.txt`, no GPU needed.
**Full reproduction environment**: the above, plus a downloaded base model
and a GPU, for any training/eval step in "Reproduction" below.

## Installation

```bash
conda create -n sftdpo python=3.10 -y
conda activate sftdpo
pip install -r requirements.txt
```

Base model weights (StarCoder2-{3B,7B,15B}, Qwen2.5-Coder-{3B,7B,14B}-Instruct,
DeepSeek-Coder-6.7B-Instruct) must be downloaded separately from their
respective official sources and are not included here.

## Quick start / smoke test

This does **not** require downloading a base model — it only checks that the
dataset and adapter files in this repository are well-formed and that the
evaluation code path runs end-to-end on a tiny slice. It does require a GPU
and the actual base model weights to produce real numbers; without those,
use `--model-name-or-path <any small local causal LM>` just to exercise the
code path (the printed metrics will be meaningless in that case, but a clean
run with no stack trace confirms the environment/paths are correct).

```bash
bash reproduce.sh --quick --model-name-or-path <MODEL>
```

Expected: completes in well under a minute on 5 samples (excluding model
load time) and prints a `comparison_summary.json` with `base`/`lora` blocks
containing `deprecated_usage_rate`/`replacement_hit_rate`.
On 5 samples these rates will be noisy (each sample moves the rate by 20
percentage points) — this step verifies plumbing, not paper numbers; see
"Reproduction" below for the full 285-sample runs that match Table 1.

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

**Table 1 — DUR / RHR (%) per model, all four methods, all seven models.**

| Model | Base DUR | Base RHR | SFT DUR | SFT RHR | DPO DUR | DPO RHR | SFT+DPO DUR | SFT+DPO RHR |
|---|---|---|---|---|---|---|---|---|
| StarCoder2-3B | 8.42 | 11.58 | 1.40 | 56.84 | 0.00 | 3.86 | 0.35 | 57.54 |
| StarCoder2-7B | 11.23 | 12.63 | 1.40 | 58.25 | 0.00 | 8.42 | 0.00 | 63.51 |
| StarCoder2-15B | 10.53 | 14.39 | 1.05 | 52.63 | 0.00 | 8.07 | 0.70 | 62.46 |
| DeepSeek-Coder-6.7B | 6.67 | 24.91 | 0.70 | 54.74 | 1.05 | 28.42 | 0.70 | 48.42 |
| Qwen2.5-Coder-3B | 8.07 | 19.30 | 0.70 | 55.09 | 0.00 | 14.39 | 0.35 | 56.14 |
| Qwen2.5-Coder-7B | 5.96 | 24.21 | 1.05 | 52.28 | 0.35 | 14.74 | 0.00 | 59.30 |
| Qwen2.5-Coder-14B | 6.32 | 31.58 | 0.70 | 46.67 | 0.35 | 17.89 | 0.70 | 54.39 |
| **Overall** (mean over all 7 models) | 8.17 | 19.80 | 1.00 | **53.79** | **0.25** | 13.68 | 0.40 | 57.39 |

Raw run directories (under `results/rq1_effectiveness/`):
SFT — `sft_starcoder2-{3b,7b,15b}` (+ `_eval` siblings); Qwen/DeepSeek SFT
weights are not separately kept, only their HumanEval/MBPP eval (Table 2).
DPO (StarCoder2/DeepSeek) — `dpo_starcoder2_deepseek(_eval)`.
DPO (Qwen2.5-Coder-7B, official) — `dpo_qwen2.5-coder-7b(_eval)`.
DPO (Qwen2.5-Coder-3B/14B, official; 7B re-run for the reproduction-variance
check below) — `dpo_qwen2.5-coder-3b-14b(_eval)`.
SFT+DPO — `sftdpo_starcoder2-3b-15b_deepseek(_eval)`,
`sftdpo_qwen2.5-coder-3b-7b-14b(_eval)`, and `sftdpo_starcoder2-7b`
+ `sftdpo_starcoder2-7b_eval` (the StarCoder2-7B screening run that fixed `λ=0.1`).

**Library-level breakdown** (Figure: per-library RHR radar, StarCoder2 series).
Base RHR on `pytorch`/`tensorflow` is 8–29%; SFT+DPO raises this to 72–84%.
Script: `plot_library_radar.py` / `plot_library_radar_panel.py`.
Raw data: `results/rq1_effectiveness/library_radar_data`.

**Table 2 — HumanEval / MBPP pass@1 Δ (%) vs. base, greedy decoding, all 7 models.**

| Model | SFT HE | DPO HE | SFT+DPO HE | SFT MBPP | DPO MBPP | SFT+DPO MBPP |
|---|---|---|---|---|---|---|
| StarCoder2-3B | −3.05 | −1.83 | −3.66 | −20.60 | +1.20 | 0.00 |
| StarCoder2-7B | +3.05 | +0.61 | −0.61 | −11.80 | −0.20 | +2.00 |
| StarCoder2-15B | +1.83 | +1.83 | +4.88 | −18.20 | +2.20 | +4.00 |
| DeepSeek-Coder-6.7B | −25.00 | −3.66 | −0.61 | −36.40 | +2.40 | +1.40 |
| Qwen2.5-Coder-3B | −18.29 | +8.54 | −2.44 | −40.60 | +0.40 | +0.60 |
| Qwen2.5-Coder-7B | −18.90 | 0.00 | 0.00 | −37.00 | +4.40 | +0.40 |
| Qwen2.5-Coder-14B | −43.29 | −4.27 | −16.46 | −4.40 | −1.40 | −1.60 |
| **Overall** | −14.81 | +0.17 | −2.70 | −24.14 | +1.29 | +0.97 |

SFT causes the largest degradation on average; DPO leaves the base
distribution largely intact; SFT+DPO is much closer to DPO than to SFT on
six of seven models. Qwen2.5-Coder-14B is the one exception where SFT+DPO
only partially recovers SFT's HumanEval collapse (0% → 26.83%, vs. a 43.29%
base) — reported as-is rather than omitted.

Scripts: `evaluate_mbpp_official.py`, `generate_official_code_samples.py`,
`run_official_code_eval_nohup.sh` (wraps the BigCode evaluation harness).
Raw data, by model/method (each directory below is under
`results/rq1_effectiveness/` and contains `<model>/<method>/{humaneval,mbpp}/metrics.json`):
- StarCoder2-3B/7B/15B SFT+base: `humaneval_mbpp_starcoder2_sft`
- StarCoder2-3B/7B/15B DPO: `humaneval_mbpp_starcoder2_deepseek_dpo`
- StarCoder2-3B/15B SFT+DPO: `humaneval_mbpp_starcoder2-3b-15b_deepseek_sftdpo`; StarCoder2-7B SFT+DPO: `humaneval_mbpp_starcoder2-7b_sftdpo`
- DeepSeek-Coder-6.7B SFT+base: `humaneval_mbpp_deepseek_sft`; DPO: `humaneval_mbpp_starcoder2_deepseek_dpo`; SFT+DPO: `humaneval_mbpp_starcoder2-3b-15b_deepseek_sftdpo`
- Qwen2.5-Coder-3B SFT+base, Qwen2.5-Coder-7B base: `humaneval_mbpp_qwen2.5-coder-3b_sft_7b_base`
- Qwen2.5-Coder-7B SFT: `humaneval_mbpp_qwen2.5-coder-7b_sft`
- Qwen2.5-Coder-14B base+SFT(HumanEval): `humaneval_mbpp_qwen2.5-coder-14b_sft`; SFT(MBPP): `humaneval_mbpp_qwen2.5-coder-14b_sft_mbpp`
- Qwen2.5-Coder-3B/7B/14B DPO: `humaneval_mbpp_qwen2.5-coder_dpo`
- Qwen2.5-Coder-3B/7B/14B SFT+DPO: `humaneval_mbpp_qwen2.5-coder_sftdpo`

**Table 3 — Retention on a 1,596-candidate, 39-non-target-API benchmark**
disjoint from the training mapping (every retained sample has 100%
base-model accuracy by construction; retention-set size varies by model
because the Qwen2.5-Coder models have lower zero-shot coverage of the 39
candidate APIs to begin with).

| Model | Samples | DPO Acc. (%) | SFT+DPO Acc. (%) |
|---|---|---|---|
| StarCoder2-3B | 520 | 38.85 | 72.88 |
| StarCoder2-7B | 540 | 51.30 | 72.22 |
| StarCoder2-15B | 540 | 57.96 | 85.37 |
| DeepSeek-Coder-6.7B | 500 | 92.00 | 90.80 |
| Qwen2.5-Coder-3B | 200 | 64.50 | 81.00 |
| Qwen2.5-Coder-7B | 80 | 90.00 | 77.50 |
| Qwen2.5-Coder-14B | 140 | 83.57 | 82.86 |
| **Overall** (weighted by sample count) | 2,520 | 62.30 | 80.32 |

On StarCoder2-3B/7B, DPO retains markedly less than SFT+DPO (38.9% vs.
72.9%; 51.3% vs. 72.2%); the StarCoder2-7B gap is heterogeneous across
libraries (numpy Δ=21.8%, pytorch Δ=25.0%, sklearn Δ=5.0%, pandas Δ=2.5%).
The pattern is not universal: on DeepSeek-Coder-6.7B and Qwen2.5-Coder-7B,
DPO retains slightly *more* than SFT+DPO (92.0% vs. 90.8%; 90.0% vs. 77.5%).

Scripts: `build_multilib_retention_candidates.py`, `select_retention_set.py`,
`split_by_library.py`, `analyze_retention_shift.py`.
Raw data: `results/rq1_effectiveness/retention_candidates_and_sftdpo` (candidate
set + SFT+DPO retention for all 7 models), `retention_starcoder2-7b_dpo`
(StarCoder2-7B DPO retention), `retention_dpo_remaining_models` (DPO retention
for the remaining 6 models), `retention_per_library_breakdown` (per-library Δ breakdown).

**Answer to RQ1.** Among the three methods, only SFT+DPO meets all four
outcomes: it drives DUR to near zero, raises RHR to several times the base
level, preserves general-purpose code ability on six of seven models, and
retains most of the base model's correct completions on non-target APIs.
DPO reduces RHR below the base model on the StarCoder2 series (consistent
with likelihood-displacement theory, Razin et al.) and retains far fewer
non-target completions than SFT+DPO on most models. SFT improves DUR/RHR
but degrades general code ability substantially.

### RQ2 — Which layers and modules account for the effect of SFT+DPO (parameter localization)

**Approach.** Treat the LoRA adapter as a set of effective updates
`ΔW_l = (α/r) B_l A_l` per attention projection per layer. (i) Magnitude
pruning, no retraining, sweep `k ∈ {5,10,15,20,30,50,100}`. (ii) Retraining
with LoRA injection restricted to a selected layer/module subset.
(iii) Cross-scale replication on StarCoder2-3B/15B. All evaluated on the
285-sample test set, primary model StarCoder2-7B.

Scripts: `analyze_lora_delta.py`, `compute_gradient_attribution.py`,
`plot_delta_heatmap.py` (module-norm heatmap); `eval_sparse_lora_delta.py`
+ `run_sparse_extended_eval.sh` (magnitude pruning); `train_dpo_lora_restricted.py`
+ `run_layer_range_ablation.sh` (restricted retraining); `train_dpo_lora.py`
+ `run_module_ablation.sh` (module ablations); `run_ablation_eval.sh`,
`run_per_library_cerdpo.sh` / `launch_per_library_cerdpo.sh` (eval).

**Module-norm distribution.** The SFT+DPO LoRA adapter on StarCoder2-7B has
128 injected modules (32 layers × 4 projections, 1.51B effective update
parameters). DPO and SFT+DPO have nearly identical mean norms (0.225 vs.
0.228), both concentrated on the O projection of layers 17–31 (2–3× the
network-wide mean). Raw data: `results/rq2_parameter_localization/module_norm_starcoder2-7b`.

**Magnitude pruning** (retained weight fraction k%, RHR on StarCoder2-7B):

| k% | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| DPO RHR | 28.1 | 26.3 | 23.5 | 18.2 | 14.4 | 8.8 | 8.4 |
| SFT+DPO RHR | 40.0 | 50.5 | 56.5 | 62.1 | 64.6 | 64.2 | 63.5 |

SFT+DPO is nearly insensitive to pruning down to k=30% (retains 97.8% of
full performance at k=20%); DPO's RHR *increases* under pruning, peaking at
k=5%, suggesting the DPO adapter also carries a replacement-suppressing
component that magnitude pruning removes first.
Raw data: `results/rq2_parameter_localization/magnitude_pruning_eval`,
`magnitude_pruning_eval_extended`.

**Restricted retraining** (layers 17–31, `{q_proj, o_proj}` only, on
StarCoder2-7B; proportional layer band on 3B/15B).

| Model | Restricted params | DUR | RHR | Rel. RHR (vs. full SFT+DPO) |
|---|---|---|---|---|
| StarCoder2-3B | ≈264M | 0.0% | 44.2% | 76.8% |
| StarCoder2-7B | ≈637M | 0.0% | 49.8% | 78.4% |
| StarCoder2-15B | ≈1,434M | 0.7% | 50.2% | **80.4%** |

Narrowing further to `{o_proj}` alone on StarCoder2-7B (≈318M params):
DUR = 0.0%, RHR = 44.6% (70.2% of full performance).
Raw data: `results/rq2_parameter_localization/restricted_retraining_starcoder2-{3b,15b}`,
`restricted_retraining_sftdpo_L17-31`, `restricted_retraining_dpo_L17-31`,
`restricted_retraining_L{20,25,28}-31`, `restricted_retraining_o_proj_L17-31`,
`restricted_retraining_o_proj_top11`, `module_ablation_o_proj_toplayers_sftdpo(_eval)`,
`module_ablation_o_proj_toplayers_dpo(_partial,_eval)`,
`restricted_retraining_eval`, `restricted_retraining_layerband_eval`.

**Module-level ablations** on StarCoder2-7B (full layer range, restricted
module subsets): MLP-only reaches RHR = 61.1% (−2.4% vs. attention-only
default); attention+MLP gains a further +2.5%. `{o_proj}` alone (all 32
layers) reaches RHR = 58.6%; removing `o_proj` (`{q_proj,k_proj,v_proj}`
only) drops RHR to 40.7% (−22.8%) — the O projection is the main carrier of
the SFT+DPO effect.
Raw data: `results/rq2_parameter_localization/module_ablation_{attn_mlp,mlp_only,kv_only,o_proj_only,qkv_no_o}`.

**Answer to RQ2.** Module-norm distribution, magnitude pruning, and
restricted retraining all point to the same modules: the O projection of
roughly the upper 47% of layers, with a smaller but non-negligible
contribution from Q. Restricting SFT+DPO training to these modules recovers
about four-fifths of full-adapter performance across the 3B/7B/15B scales.

### RQ3 — Layer-wise analysis of SFT+DPO

**Approach.** Logit Lens / Tuned Lens / Activation-Difference Lens (ADL) /
OV-circuit attribution on a 35-sample calibration set (union of repair +
consistency test splits), primary model StarCoder2-7B, with DeepSeek-Coder-6.7B
and Qwen2.5-Coder-7B as cross-series replicates.

Training (Tuned Lens translator): `code/mechanism/scripts/train_tuned_lens.py`.
Lens comparison: `run_lens_compare.py`, `run_lens_compare_variants.py`
(launched via `run_mechanism_model.sh` / `run_mechanism_compare_only.sh` /
`launch_full_mechanism_20260427.sh` / `launch_compare_only_jsd_20260427.sh` /
`run_triplet_batch.sh`). ADL: `run_activation_difference_lens.py`
(via `run_adl_batch.sh`). OV-circuit: `ov_circuit_flip.py`, `ov_dla_flip.py`,
`ov_dla_multimodel.py`.

**Final-layer replacement margin on StarCoder2-7B** (35-sample calibration
set average).

| Lens | Base m_seq | SFT+DPO m_seq | Base m_first | SFT+DPO m_first |
|---|---|---|---|---|
| Logit | −0.05 | +32.88 | −2.43 | +21.13 |
| Tuned | −0.62 | +60.76 | −3.30 | +43.91 |

**Stable decision depth.** StarCoder2 series: base commits at layer 24.27
on average; SFT+DPO reduces this to 0.97. Cross-series (StarCoder2-7B,
DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B): 20.67 → 6.57. DPO attains a similar
local margin (23.49 vs. SFT+DPO's 18.24 at the final layer) but its stable
decision depth stays high — it favors the replacement token locally but
fails to commit early, consistent with the lower RHR of DPO in RQ1.

**Activation-Difference Lens on neutral prompts** (no API/version tokens).
`rand` = NeutralText, `code` = NeutralCode, Δ = Readout_code − Readout_rand.

| Setting | Norm_rand | Readout_rand | Readout_code | Δ |
|---|---|---|---|---|
| DPO | 28.69 | 0.408 | 0.518 | 0.110 |
| SFT+DPO | 27.90 | 0.416 | 0.543 | **0.127** |

DPO produces a larger activation-difference signal on natural-language
inputs (consistent with its larger RQ1 retention/HumanEval/MBPP drops);
SFT+DPO's signal is more concentrated in code contexts (larger Δ).

**OV-circuit direct logit attribution**, final-layer
Δℓ = ℓ(replacement) − ℓ(deprecated), two linalg migration cases, including
Qwen2.5-Coder-7B as the cross-series ≈7B-scale representative.

| Model | svd→linalg.svd Base | svd→linalg.svd SFT+DPO | qr→linalg.qr Base | qr→linalg.qr SFT+DPO |
|---|---|---|---|---|
| StarCoder2-3B | −3.22 | **+33.25** | −3.13 | **+31.19** |
| StarCoder2-7B | −2.06 | **+35.03** | −3.25 | **+32.75** |
| StarCoder2-15B | −5.52 | **+36.31** | −4.94 | **+32.81** |
| DeepSeek-Coder-6.7B | −6.77 | **+39.28** | −0.27 | **+39.63** |
| Qwen2.5-Coder-7B | −11.05 | **+12.52** | −1.25 | **+19.03** |

Dominant heads on StarCoder2-7B: L31H2, L31H1 (largest flip deltas on both
linalg cases), L25H33 third. The flip is smaller on Qwen2.5-Coder-7B than
on the StarCoder2/DeepSeek-Coder models but still clearly positive.

Raw data: `results/rq3_layerwise_analysis/logit_tuned_lens_all_models` (Logit/Tuned
Lens, 7 models), `logit_lens_triplet_comparison` (DPO/SFT+DPO/base triplet),
`activation_difference_lens` (ADL table), `ov_circuit_attribution_<model>.json`
(one per model, OV-circuit table) and `ov_circuit_attribution_flip_check.json`.

**Answer to RQ3.** SFT+DPO concentrates the change in mid-to-late attention
layers; the replacement decision becomes stable in the earliest layers and
surfaces at the output layer as a clear positive margin. DPO reaches a
similar local margin but does not act on it reliably, because it also makes
a broad, context-independent change in the output; the cross-entropy
regularization reduces this broad change and makes the update more specific
to code contexts.

### Supplementary: base-model decision-emergence-layer analysis

`results/emergence_analysis/` and `code/correct_emergence/` support a
Discussion-section observation about where in the network the base model's
deprecated-vs-replacement decision becomes legible (independent of any
fine-tuning method), via top-k trajectory tracing at the post-API decision
token. Two dated variants are included:

- `starcoder2-7b_as_cited_in_paper/` — the run matching the numbers currently
  cited in the paper text (Logit Lens mean emergence layer 19.1, median 18,
  P25–P75 18–20; mean saturation layer 24.8, median 24, P25–P75 21–29).
- `starcoder2-7b_corrected/` — a corrected re-run fixing a decision-token-
  alignment bug (selecting the token immediately after the attribute-access
  dot rather than the dot itself): mean emergence layer 19.2, median 20,
  P25–P75 18–21; mean saturation layer 27.9, median 28, P25–P75 27–31.

Both are included for transparency; the corrected variant is the
methodologically accurate one.

---

## Reproduction

This section walks through the experiments in the same order they are
reported in the paper (RQ1 → RQ2 → RQ3), naming the exact script for each
step and what it consumes/produces. Every flag below was read directly off
the script's `argparse` definition — run any script with `--help` for the
complete list. `<MODEL>` is a local path to a downloaded base model (see
"Setup"); `<OUT>` is any output directory you choose.

### Step 0 — data

`code/positive_engineering/data/mixed_sft_v1/` already contains the built
DepPref dataset (`mixed_sft_train/val/test.jsonl`, plus a `by_library/`
breakdown). You do not need to rebuild it to reproduce RQ1–RQ3; it is only
documented here for completeness. It was built by
`code/positive_engineering/scripts/build_mixed_sft.py` from upstream
repair/consistency/reference buckets that are out of scope for this package.

### Step 1 — RQ1: train and evaluate all three methods

For each of the 7 models (StarCoder2-3B/7B/15B, DeepSeek-Coder-6.7B-Instruct,
Qwen2.5-Coder-3B/7B/14B-Instruct):

```bash
# SFT
python code/positive_engineering/scripts/train_lora.py \
  --model-name-or-path <MODEL> \
  --train-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_val.jsonl \
  --output-dir <OUT>/sft

# DPO (api-anchor-weight defaults to 0, i.e. plain DPO)
python code/positive_engineering/scripts/train_dpo_lora.py \
  --model-name-or-path <MODEL> \
  --train-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_val.jsonl \
  --output-dir <OUT>/dpo

# SFT+DPO (cross-entropy anchor over the replacement-API span, λ=0.1)
python code/positive_engineering/scripts/train_dpo_lora.py \
  --model-name-or-path <MODEL> \
  --train-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_val.jsonl \
  --api-anchor-weight 0.1 --output-dir <OUT>/sftdpo

# Evaluate each adapter: DUR/RHR on the 285-sample test set (Table 1)
python code/positive_engineering/scripts/eval_compare_lora.py \
  --model-name-or-path <MODEL> --adapter-dir <OUT>/sftdpo \
  --test-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_test.jsonl \
  --output-dir <OUT>/sftdpo_eval
```

`run_dpo_lora_nohup.sh` / `run_qwen_mixed_sft_v1_nohup.sh` /
`run_deepseek_mixed_sft_v1_nohup.sh` show the exact per-model arguments used
for the included `results/rq1_effectiveness/` artifacts; `run_module_ablation.sh`
and `run_layer_range_ablation.sh` belong to RQ2 (Step 2), not here.

**General code ability (Table 2).** Generate completions with
`generate_official_code_samples.py --benchmark {humaneval,mbpp} --model-name-or-path <MODEL> [--adapter-dir <OUT>/sftdpo] --output-jsonl <OUT>/samples.jsonl`,
then score with `evaluate_mbpp_official.py --samples-jsonl <OUT>/samples.jsonl --output-json <OUT>/metrics.json`
for MBPP, or the BigCode evaluation harness (`run_official_code_eval_nohup.sh`
wraps it) for HumanEval pass@1.

**Retention on the disjoint-API benchmark (Table 3).**
```bash
python code/positive_engineering/scripts/build_multilib_retention_candidates.py \
  --source-root <some-code-corpus> --training-jsonl code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --output-file <OUT>/candidates.jsonl
python code/positive_engineering/scripts/select_retention_set.py \
  --candidate-file <OUT>/candidates.jsonl --base-predictions <OUT>/base_predictions.jsonl \
  --output-file <OUT>/retention_set.jsonl
# then eval_compare_lora.py on the retention set, same as the DUR/RHR step above
```
`split_by_library.py` / `analyze_retention_shift.py` produce the per-library
Δ breakdown; `plot_library_radar.py` / `plot_library_radar_panel.py` produce
the radar figure.

### Step 2 — RQ2: parameter localization

```bash
# Module-norm distribution (Frobenius norm per layer/projection)
python code/positive_engineering/scripts/analyze_lora_delta.py \
  --adapter DPO=<OUT>/dpo --adapter SFT+DPO=<OUT>/sftdpo --output-dir <OUT>/delta_analysis
# plot_delta_heatmap.py reads {dpo,anchored_dpo}_module_stats.csv from a
# hardcoded DATA_DIR constant at the top of the script — point it at the
# directory analyze_lora_delta.py just wrote, or edit the constant directly.

# Magnitude pruning sweep (no retraining)
python code/positive_engineering/scripts/eval_sparse_lora_delta.py \
  --model-name-or-path <MODEL> --adapter-dir <OUT>/sftdpo \
  --test-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_test.jsonl \
  --keep-fraction 5 --keep-fraction 10 --keep-fraction 20 --keep-fraction 30 \
  --keep-fraction 50 --keep-fraction 100 --output-dir <OUT>/pruning_sweep

# Restricted retraining (layers 17-31, {q_proj,o_proj} only, on StarCoder2-7B)
python code/positive_engineering/scripts/train_dpo_lora_restricted.py \
  --model-name-or-path <MODEL> \
  --train-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_val.jsonl \
  --layers-to-transform 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --target-modules q_proj o_proj --api-anchor-weight 0.1 \
  --output-dir <OUT>/sftdpo_restricted_L17_31
# then eval_compare_lora.py on this adapter, same as Step 1

# Module-level ablations (e.g. train on {q_proj,k_proj,v_proj} only, all 32 layers)
python code/positive_engineering/scripts/train_dpo_lora.py \
  --model-name-or-path <MODEL> --train-file ... --val-file ... \
  --target-modules q_proj k_proj v_proj --api-anchor-weight 0.1 --output-dir <OUT>/sftdpo_qkv_no_o
```
`run_sparse_extended_eval.sh`, `run_layer_range_ablation.sh`,
`run_module_ablation.sh`, `run_ablation_eval.sh` show the exact per-config
arguments used for the included `results/rq2_parameter_localization/` artifacts.

### Step 3 — RQ3: layer-wise analysis

```bash
# Train the Tuned Lens translator on the repair+consistency splits
python code/mechanism/scripts/train_tuned_lens.py \
  --model-key starcoder2_7b --output-file <OUT>/tuned_lens_starcoder2_7b.pt

# Logit Lens / Tuned Lens margins + stable decision depth (Table 4, depth figure)
python code/mechanism/scripts/run_lens_compare.py \
  --model-key starcoder2_7b --adapter-dir <OUT>/sftdpo \
  --base-tuned-lens <OUT>/tuned_lens_starcoder2_7b.pt --output-dir <OUT>/lens_compare

# Activation-Difference Lens on neutral prompts (Table 5)
python code/mechanism/scripts/run_activation_difference_lens.py \
  --model-key starcoder2_7b --adapter-dir <OUT>/sftdpo --output-dir <OUT>/adl

# OV-circuit direct logit attribution (Table 6)
python code/mechanism/scripts/ov_dla_multimodel.py --model starcoder2_7b
```
`run_mechanism_model.sh` / `run_mechanism_compare_only.sh` /
`launch_full_mechanism_20260427.sh` / `run_triplet_batch.sh` / `run_adl_batch.sh`
show the exact per-model arguments used for the included
`results/rq3_layerwise_analysis/` artifacts.

### Plotting / table scripts

Several scripts under `code/*/scripts/` (`plot_*.py`, `summarize_*.py`) turn
the raw per-run JSON/CSV above into the paper's figures and aggregate tables.
Most take `--output-dir`/`--output-file` flags; a few (`plot_delta_heatmap.py`)
use hardcoded path constants near the top of the file instead — open the
script and adjust those constants if your output layout differs from the
original `output/<run_name>/...` convention used when producing the included
`results/`.

## Known reproduction variance

The paper's reported Qwen2.5-Coder-7B DPO numbers (DUR=0.35%, RHR=14.74%)
come from `results/rq1_effectiveness/dpo_qwen2.5-coder-7b_eval`. A later,
independent re-run of the same recipe on Qwen2.5-Coder-7B, included in
`results/rq1_effectiveness/dpo_qwen2.5-coder-3b-14b_eval` (built primarily to
cover Qwen2.5-Coder-3B/14B, which had no earlier DPO run), reproduced the DUR
exactly (0.35%) but obtained RHR = 17.19%. Both values are consistent with
the paper's qualitative claim (DPO RHR well below SFT+DPO's 59.30%); the
discrepancy reflects ordinary run-to-run variance in DPO training, not a
methodological difference.

## Known limitations

- **Base model weights are not redistributed.** Training/eval scripts require
  separately downloading StarCoder2-{3B,7B,15B}, Qwen2.5-Coder-{3B,7B,14B}-Instruct,
  and DeepSeek-Coder-6.7B-Instruct from their official sources. This does not
  affect verification of the included `results/`, only end-to-end re-running.
- **Qwen2.5-Coder-14B is the one model where SFT+DPO does not fully recover
  general code ability** after SFT's collapse (HumanEval 0% → 26.83%, vs. a
  43.29% base) — reported in the paper as-is, not a gap hidden by this package.
- **Full-scale training is GPU-hour-intensive** across 7 models × 3 methods ×
  RQ2/RQ3 ablations; `results/` already contains the completed outputs for
  every reported number so reviewers do not need to re-run training to verify
  the paper's claims, only to inspect or partially re-run them.
- **DPO has ordinary run-to-run variance** (see "Known reproduction variance"
  above) — this is a property of the baseline method's training stability,
  not of this package's completeness.
- None of the above limitations affect the ability to verify the paper's
  core claims (Tables 1–3, RQ2/RQ3 answers) from the included `results/`.
