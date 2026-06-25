import unittest
import json
import tempfile
from pathlib import Path

from scripts.compile_followup_artifacts import (
    build_reverse_scale_table,
    build_sparse_restricted_table,
    summarize_retention_gap,
)


class CompileFollowupArtifactsTest(unittest.TestCase):
    def test_summarize_retention_gap_computes_overall_and_library_deltas(self):
        payload = {
            "triplet_library_table": [
                {
                    "library": "numpy",
                    "samples": 280,
                    "base_exact_match_rate": 1.0,
                    "dpo_exact_match_rate": 0.45,
                    "dpo_prediction_changed_rate": 0.55,
                    "anchored_dpo_exact_match_rate": 0.667857,
                    "anchored_dpo_prediction_changed_rate": 0.332143,
                },
                {
                    "library": "pytorch",
                    "samples": 200,
                    "base_exact_match_rate": 1.0,
                    "dpo_exact_match_rate": 0.55,
                    "dpo_prediction_changed_rate": 0.45,
                    "anchored_dpo_exact_match_rate": 0.8,
                    "anchored_dpo_prediction_changed_rate": 0.2,
                },
            ],
            "variants": {
                "dpo": {"summary": {"overall": {"variant_exact_match_rate": 0.513}}},
                "anchored_dpo": {"summary": {"overall": {"variant_exact_match_rate": 0.722}}},
            },
        }

        summary = summarize_retention_gap(payload)

        self.assertAlmostEqual(summary["overall_exact_match_gap"], 0.209)
        self.assertEqual(summary["largest_exact_match_gaps"][0]["library"], "pytorch")
        self.assertAlmostEqual(summary["largest_exact_match_gaps"][0]["exact_match_gap"], 0.25)

    def test_build_reverse_scale_table_uses_same_subset_for_new_prefix_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            subset_file = tmpdir / "subset.jsonl"
            predictions_file = tmpdir / "predictions.jsonl"
            subset_file.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a"}),
                        json.dumps({"id": "c"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_file.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a", "has_replacement": True}),
                        json.dumps({"id": "b", "has_replacement": False}),
                        json.dumps({"id": "c", "has_replacement": False}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            reverse_payloads = {
                "3b": {
                    "reverse_subset_file": str(subset_file),
                    "variants": {
                        "base": {"replacement_hit_rate": 0.848},
                        "dpo": {"replacement_hit_rate": 0.091},
                        "anchored_dpo": {"replacement_hit_rate": 0.879},
                    },
                }
            }
            anchored_eval_payloads = {
                "3b": {"lora": {"predictions_file": str(predictions_file)}}
            }

            rows = build_reverse_scale_table(reverse_payloads, anchored_eval_payloads, model_order=["3b"])

            self.assertEqual(rows[0]["model"], "3b")
            self.assertAlmostEqual(rows[0]["anchored_dpo_new_prefix_shared_subset_replacement_hit_rate"], 0.5)
            self.assertAlmostEqual(rows[0]["dpo_old_prefix_replacement_hit_rate"], 0.091)

    def test_build_sparse_restricted_table_orders_variants_for_anchor_story(self):
        full_dpo = {"lora": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.084}}
        full_anchor = {"lora": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.635}}
        dpo_sparse_payload = {
            "variants": {
                "keep_50": {"summary": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.088}},
                "keep_20": {"summary": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.182}},
                "keep_10": {"summary": {"deprecated_usage_rate": 0.011, "replacement_hit_rate": 0.263}},
            }
        }
        sparse_payload = {
            "variants": {
                "keep_50": {"summary": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.642}},
                "keep_20": {"summary": {"deprecated_usage_rate": 0.014, "replacement_hit_rate": 0.621}},
                "keep_10": {"summary": {"deprecated_usage_rate": 0.014, "replacement_hit_rate": 0.505}},
            }
        }
        restricted_dpo_payload = {"lora": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.123}}
        restricted_payload = {"lora": {"deprecated_usage_rate": 0.0, "replacement_hit_rate": 0.498}}

        rows = build_sparse_restricted_table(
            full_dpo,
            full_anchor,
            dpo_sparse_payload,
            sparse_payload,
            restricted_dpo_payload,
            restricted_payload,
        )

        self.assertEqual(
            [row["config"] for row in rows],
            [
                "dpo_full",
                "dpo_keep_50",
                "dpo_keep_20",
                "dpo_keep_10",
                "dpo_restricted_layers",
                "anchored_dpo_full",
                "anchored_dpo_keep_50",
                "anchored_dpo_keep_20",
                "anchored_dpo_keep_10",
                "anchored_dpo_restricted_layers",
            ],
        )
        self.assertAlmostEqual(rows[-1]["replacement_hit_rate"], 0.498)


if __name__ == "__main__":
    unittest.main()
