# Replication Package: Deprecated API Recommendation in Code LLMs via Preference Optimization

This is the replication package for a submission titled *Mitigating and
Demystifying Deprecated API Recommendation in Code LLMs via Preference
Optimization*.

This directory packages the code and result artifacts behind RQ1–RQ3 of the
paper (RQ4, the version-prefix robustness study, is intentionally excluded —
tracked separately). All artifacts are copies; nothing here is a symlink back
into the working repo.

This repository is intended for double-anonymous review (e.g. via
anonymous.4open.science). It contains no author names, institution names, or
personal file-path identifiers — all absolute paths have been rewritten to
generic placeholders (`/workspace`, `/data/models`, `/opt/conda`).
Intermediate training checkpoints (optimizer/scheduler/rng state) and
out-of-scope intermediate data (raw scraped candidate pools, unused
experiment-family datasets such as `rerank_*`/`repair_sft_*`/`consistency_*`)
have been removed; only the final adapter weights and the data actually
consumed by the in-scope RQ1–RQ3 pipelines are kept.

Note on terminology: the method is referred to as **SFT+DPO** throughout
this documentation (DPO with an `api_anchor_weight=0.1` cross-entropy anchor
term on the replacement API token). Script, config, and directory names
below still use internal identifiers such as `cerdpo`/`anchor` for this same
method.

## Layout

```
code/
  positive_engineering/   SFT / DPO / SFT+DPO training + eval code
                           (src/, scripts/, configs/, data/ — DepPref dataset)
  mechanism/               Logit Lens / Tuned Lens / OV-circuit mechanism analysis (RQ2)
  correct_emergence/       Base-model decision-token emergence-layer analysis
                           (Discussion section, "pre-existing latent knowledge")
docs/
  SFT+DPO.md, 0605.md, 0612.md, 0619.md   methodology notes behind RQ1-RQ3 numbers
  CASE_STUDIES.md                          worked examples for the emergence analysis
  technical_report_*.md                    per-run technical reports
results/
  rq1_effectiveness/          Base/SFT/DPO/SFT+DPO main table, HumanEval/MBPP,
                               retention, library radar — across 7 models
  rq2_mechanism/               margin/depth, ADL, OV-circuit dest-loosing analysis
  rq3_parameter_localization/  Frobenius-norm heatmap, pruning curve,
                               restricted-layer retraining, module ablations
                               (includes fine-grained layer-range ablations that
                               did not make the main paper table — kept as
                               supplementary material)
  emergence_analysis/
    starcoder2_7b_PAPER_CURRENT_prefix_uncorrected/   matches the numbers
        currently cited in the paper's Discussion paragraph (mean emergence
        layer 19.1, mean saturation layer 24.8, Logit Lens)
    starcoder2_7b_CORRECTED_decision_after_dot/        corrected run after fixing
        a decision-token-position bug (the original analysis treated the
        namespace-alias token, e.g. `tf`/`torch`/`F`, as the decision token
        in ~89% of samples instead of the API name itself). Corrected Logit
        Lens numbers: mean emergence layer 19.2 (median 20, P25-P75 18-21),
        mean saturation layer 27.9 (median 28, P25-P75 27-31). See docs/0619.md
        for full derivation. **The paper text has not yet been updated to
        the corrected numbers** — treat starcoder2_7b_PAPER_CURRENT_prefix_uncorrected
        as the version backing the current paper draft until that is fixed.
```

## Reproduction

LoRA adapter weights are included inside each `results/.../<run_name>/`
directory (`adapter_model.safetensors` + `adapter_config.json`). Training and
eval entry points are in `code/positive_engineering/scripts/`:
- `train_dpo_lora.py` / `run_dpo_lora_nohup.sh` — DPO / SFT+DPO training
- `eval_compare_lora.py` — base-vs-LoRA DUR/RHR evaluation
- `train_lora.py` — plain SFT training

Each run directory contains a `run_manifest.json` (or equivalent `*_eval.log`)
recording the exact hyperparameters and source checkpoint used.

## Setup

```bash
conda create -n sftdpo python=3.10 && conda activate sftdpo
pip install torch transformers peft trl  # see code/positive_engineering/scripts for exact versions used
```

Base model weights (StarCoder2-{3b,7b,15b}, Qwen2.5-Coder-{3b,7b,14b}-Instruct,
DeepSeek-Coder-6.7b-Instruct) are not redistributed here; download them from
their original public sources and point `--model-name-or-path` at the local
copy. LoRA adapters in `results/` apply on top of these base models.

## Known reproduction variance

A self-check re-run of the Qwen2.5-Coder-7B DPO baseline (see
`results/rq1_effectiveness/dpo_baseline_refill_20260612_eval/`) reproduced
DUR exactly (0.35%) but RHR came out at 17.19% vs. the paper's originally
reported 14.74% (~2.45pp difference). This is disclosed for transparency;
the newer run is treated as authoritative going forward.
