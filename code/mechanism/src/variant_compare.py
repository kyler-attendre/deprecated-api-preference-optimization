from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MODEL_LABELS = {
    "starcoder2_3b": "StarCoder2-3B",
    "starcoder2_7b": "StarCoder2-7B",
    "starcoder2_15b": "StarCoder2-15B",
    "deepseek_coder_6_7b_instruct": "DeepSeek-Coder-6.7B-Instruct",
    "qwen2_5_coder_3b_instruct": "Qwen2.5-Coder-3B-Instruct",
    "qwen2_5_coder_7b_instruct": "Qwen2.5-Coder-7B-Instruct",
    "qwen2_5_coder_14b_instruct": "Qwen2.5-Coder-14B-Instruct",
}


PLAIN_DPO_ROOTS = {
    "starcoder2_3b": "dpo_lora_mixed_sft_v1_20260422",
    "starcoder2_7b": "dpo_lora_mixed_sft_v1_20260422",
    "starcoder2_15b": "dpo_lora_mixed_sft_v1_20260422",
    "deepseek_coder_6_7b_instruct": "dpo_lora_mixed_sft_v1_20260422",
}

ANCHORED_DPO_ROOTS = {
    "starcoder2_3b": "dpo_anchor_full01_20260423",
    "starcoder2_7b": "dpo7b_screen_full_anchor01_20260423",
    "starcoder2_15b": "dpo_anchor_full01_20260423",
    "deepseek_coder_6_7b_instruct": "dpo_anchor_full01_20260423",
    "qwen2_5_coder_3b_instruct": "dpo_anchor_full01_qwen_20260423",
    "qwen2_5_coder_7b_instruct": "dpo_anchor_full01_qwen_20260423",
    "qwen2_5_coder_14b_instruct": "dpo_anchor_full01_qwen_20260423",
}


@dataclass(frozen=True)
class VariantSpec:
    label: str
    adapter_dir: Optional[str] = None
    tuned_lens_path: Optional[str] = None


def normalize_variant_specs(raw_specs: Iterable[Dict]) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    for raw in raw_specs:
        specs.append(
            VariantSpec(
                label=str(raw["label"]),
                adapter_dir=str(raw["adapter_dir"]) if raw.get("adapter_dir") else None,
                tuned_lens_path=str(raw["tuned_lens_path"]) if raw.get("tuned_lens_path") else None,
            )
        )
    return specs


def build_default_variant_specs(
    *,
    model_key: str,
    mechanism_root: Path,
    positive_engineering_root: Path,
) -> List[VariantSpec]:
    specs: List[VariantSpec] = []

    official_tuned = mechanism_root / model_key / "tuned_lens" / f"{model_key}_official_base.pt"
    specs.append(
        VariantSpec(
            label="official_base",
            adapter_dir=None,
            tuned_lens_path=str(official_tuned) if official_tuned.exists() else None,
        )
    )

    plain_root_name = PLAIN_DPO_ROOTS.get(model_key)
    if plain_root_name is not None:
        plain_adapter = positive_engineering_root / "output" / plain_root_name / model_key
        if plain_adapter.exists():
            plain_tuned = mechanism_root / model_key / "tuned_lens" / f"{model_key}_plain_dpo.pt"
            specs.append(
                VariantSpec(
                    label="plain_dpo",
                    adapter_dir=str(plain_adapter),
                    tuned_lens_path=str(plain_tuned) if plain_tuned.exists() else None,
                )
            )

    anchor_root_name = ANCHORED_DPO_ROOTS.get(model_key)
    if anchor_root_name is not None:
        anchor_adapter = positive_engineering_root / "output" / anchor_root_name / model_key
        if anchor_adapter.exists():
            anchor_tuned = mechanism_root / model_key / "tuned_lens" / f"{model_key}_anchored_dpo.pt"
            specs.append(
                VariantSpec(
                    label="anchored_dpo",
                    adapter_dir=str(anchor_adapter),
                    tuned_lens_path=str(anchor_tuned) if anchor_tuned.exists() else None,
                )
            )

    return specs


def build_group_labels(model_key: str) -> Dict[str, bool]:
    return {
        "model_group_starcoder_scale": model_key in {"starcoder2_3b", "starcoder2_7b", "starcoder2_15b"},
        "model_group_cross_family_same_scale": model_key
        in {"starcoder2_7b", "qwen2_5_coder_7b_instruct", "deepseek_coder_6_7b_instruct"},
    }


def build_variant_rows(model_key: str, model_label: str, run_summary: Dict) -> List[Dict]:
    rows: List[Dict] = []
    group_labels = build_group_labels(model_key)
    for variant, lens_payloads in (run_summary.get("variants") or {}).items():
        for lens, payload in lens_payloads.items():
            layer_summary = payload.get("layer_summary") or []
            if not layer_summary:
                continue
            last = layer_summary[-1]
            depth_stats = payload.get("depth_stats") or {}
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "variant": variant,
                    "lens": lens,
                    "final_layer": int(last["layer"]),
                    "samples": int(last["samples"]),
                    "sequence_win_rate": float(last["replacement_sequence_win_rate"]),
                    "first_token_win_rate": float(last["replacement_first_token_win_rate"]),
                    "sequence_margin": float(last["mean_sequence_logprob_margin"]),
                    "avg_token_margin": float(last["mean_avg_token_logprob_margin"]),
                    "geometric_mean_perplexity_ratio": float(
                        last["geometric_mean_perplexity_ratio_deprecated_over_replacement"]
                    ),
                    "first_token_margin": float(last["mean_first_token_logprob_margin"]),
                    "stable_depth_reach_rate": float(depth_stats.get("stable_depth_reach_rate", 0.0)),
                    "mean_stable_depth_with_fallback": float(
                        depth_stats.get("mean_stable_depth_with_fallback", 0.0)
                    ),
                    **group_labels,
                }
            )
    return rows


def model_label_for(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)
