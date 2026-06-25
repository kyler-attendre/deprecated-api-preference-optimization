import sys
import tempfile
import unittest
from pathlib import Path


MECH_ROOT = Path(__file__).resolve().parents[1]
if str(MECH_ROOT) not in sys.path:
    sys.path.insert(0, str(MECH_ROOT))

from src.variant_compare import (  # noqa: E402
    VariantSpec,
    build_default_variant_specs,
    build_group_labels,
    build_variant_rows,
    normalize_variant_specs,
)


class VariantCompareTests(unittest.TestCase):
    def test_normalize_variant_specs_preserves_optional_fields(self):
        raw_specs = [
            {"label": "official_base"},
            {
                "label": "plain_dpo",
                "adapter_dir": "/tmp/plain",
            },
            {
                "label": "anchored_dpo",
                "adapter_dir": "/tmp/anchor",
                "tuned_lens_path": "/tmp/anchor.pt",
            },
        ]
        specs = normalize_variant_specs(raw_specs)
        self.assertEqual(
            specs,
            [
                VariantSpec(label="official_base", adapter_dir=None, tuned_lens_path=None),
                VariantSpec(label="plain_dpo", adapter_dir="/tmp/plain", tuned_lens_path=None),
                VariantSpec(label="anchored_dpo", adapter_dir="/tmp/anchor", tuned_lens_path="/tmp/anchor.pt"),
            ],
        )

    def test_build_variant_rows_handles_sparse_variants(self):
        run_summary = {
            "variants": {
                "official_base": {
                    "logit_lens": {
                        "layer_summary": [
                            {"layer": 0, "samples": 2},
                            {
                                "layer": 1,
                                "samples": 2,
                                "replacement_sequence_win_rate": 0.25,
                                "replacement_first_token_win_rate": 0.5,
                                "mean_sequence_logprob_margin": 1.5,
                                "mean_avg_token_logprob_margin": 0.4,
                                "geometric_mean_perplexity_ratio_deprecated_over_replacement": 1.4918,
                                "mean_first_token_logprob_margin": 0.3,
                            },
                        ],
                        "depth_stats": {
                            "stable_depth_reach_rate": 0.5,
                            "mean_stable_depth_with_fallback": 2.0,
                        },
                    }
                },
                "anchored_dpo": {
                    "logit_lens": {
                        "layer_summary": [
                            {"layer": 0, "samples": 2},
                            {
                                "layer": 1,
                                "samples": 2,
                                "replacement_sequence_win_rate": 1.0,
                                "replacement_first_token_win_rate": 1.0,
                                "mean_sequence_logprob_margin": 4.0,
                                "mean_avg_token_logprob_margin": 1.2,
                                "geometric_mean_perplexity_ratio_deprecated_over_replacement": 3.3201,
                                "mean_first_token_logprob_margin": 1.0,
                            },
                        ],
                        "depth_stats": {
                            "stable_depth_reach_rate": 1.0,
                            "mean_stable_depth_with_fallback": 1.0,
                        },
                    }
                },
            }
        }
        rows = build_variant_rows(
            model_key="qwen2_5_coder_7b_instruct",
            model_label="Qwen2.5-Coder-7B-Instruct",
            run_summary=run_summary,
        )
        self.assertEqual(len(rows), 2)
        labels = {row["variant"] for row in rows}
        self.assertEqual(labels, {"official_base", "anchored_dpo"})
        self.assertTrue(all(row["model_group_cross_family_same_scale"] for row in rows))
        self.assertTrue(all(not row["model_group_starcoder_scale"] for row in rows))

    def test_build_group_labels_marks_target_buckets(self):
        starcoder = build_group_labels("starcoder2_7b")
        qwen = build_group_labels("qwen2_5_coder_7b_instruct")
        self.assertEqual(
            starcoder,
            {
                "model_group_starcoder_scale": True,
                "model_group_cross_family_same_scale": True,
            },
        )
        self.assertEqual(
            qwen,
            {
                "model_group_starcoder_scale": False,
                "model_group_cross_family_same_scale": True,
            },
        )

    def test_build_default_variant_specs_discovers_available_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mechanism_root = root / "mech"
            pe_root = root / "pe"
            model_key = "starcoder2_7b"

            official_tuned = mechanism_root / model_key / "tuned_lens" / f"{model_key}_official_base.pt"
            anchor_tuned = mechanism_root / model_key / "tuned_lens" / f"{model_key}_anchored_dpo.pt"
            official_tuned.parent.mkdir(parents=True, exist_ok=True)
            official_tuned.write_text("x")
            anchor_tuned.write_text("x")

            plain_adapter = pe_root / "output" / "dpo_lora_mixed_sft_v1_20260422" / model_key
            anchor_adapter = pe_root / "output" / "dpo7b_screen_full_anchor01_20260423" / model_key
            plain_adapter.mkdir(parents=True, exist_ok=True)
            anchor_adapter.mkdir(parents=True, exist_ok=True)

            specs = build_default_variant_specs(
                model_key=model_key,
                mechanism_root=mechanism_root,
                positive_engineering_root=pe_root,
            )

            self.assertEqual(
                specs,
                [
                    VariantSpec(label="official_base", adapter_dir=None, tuned_lens_path=str(official_tuned)),
                    VariantSpec(label="plain_dpo", adapter_dir=str(plain_adapter), tuned_lens_path=None),
                    VariantSpec(label="anchored_dpo", adapter_dir=str(anchor_adapter), tuned_lens_path=str(anchor_tuned)),
                ],
            )


if __name__ == "__main__":
    unittest.main()
