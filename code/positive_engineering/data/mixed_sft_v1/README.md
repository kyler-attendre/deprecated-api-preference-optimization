# DepPref dataset

2,408 train / 304 val / 285 test prompt-completion pairs, built from a
145-pair deprecated→replacement API mapping (Wang et al.) across 7 libraries
(pytorch, tensorflow, numpy, scipy, sklearn, seaborn, pandas). The test set
comprises 14 repair, 21 consistency, and 250 reference samples (see the
paper's "Preference Data" section for the construction procedure).

`by_library/<library>/` contains the same train/val/test split filtered to
a single library, used for the per-library breakdown (Table 1 radar figure,
Table 3 per-library Δ).

## Files

- `mixed_sft_train.jsonl`, `mixed_sft_val.jsonl`, `mixed_sft_test.jsonl`
- `summary.json` — build-time provenance (source bucket sizes, split caps)

## Fields (one JSON object per line)

| Field | Meaning |
|---|---|
| `id` | unique sample id |
| `model` | which base model's completion was used to construct this sample |
| `library` | target library (pytorch, numpy, ...) |
| `version_prompt` | prompt with explicit target-version context |
| `probing_input` | prompt without version context (used by the layer-wise analyses, RQ3) |
| `target` | the replacement-consistent completion ($y_w$ for DPO/SFT+DPO) |
| `reference` | ground-truth reference completion |
| `deprecated_api` | the deprecated API call this sample is keyed on |
| `replacement_api` | the corresponding replacement API call |
| `category` | `repair` / `consistency` / `reference` (see paper Section "Preference Data") |
| `sample_type`, `task_family`, `label_axis` | internal bucketing metadata from dataset construction |
| `mixed_source_bucket`, `mixed_original_split`, `mixed_assigned_split` | provenance of which upstream bucket and split this sample was drawn from |
| `semantic_group_id` | groups samples derived from the same underlying code snippet, used to prevent train/test leakage across splits |

For DPO/SFT+DPO, the rejected completion $y_l$ is derived at training time
from `target` by substituting the replacement API span with the
corresponding `deprecated_api` form (see `code/positive_engineering/src/dataset_utils.py`),
not stored as a separate field.

## License / provenance

Code snippets are derived from public GitHub repositories used to construct
the deprecated-API mapping of Wang et al.; this package redistributes only
the short function-level excerpts needed to reproduce the paper's results,
not full repositories.
