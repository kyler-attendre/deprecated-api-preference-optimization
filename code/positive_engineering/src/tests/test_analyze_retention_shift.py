import unittest

from scripts.analyze_retention_shift import (
    build_triplet_library_table,
    summarize_shift_against_base,
)


class AnalyzeRetentionShiftTest(unittest.TestCase):
    def test_summarize_shift_against_base_tracks_changes_recoveries_and_degradations(self):
        candidate_rows = [
            {"id": "s1", "library": "numpy", "target_api": "array"},
            {"id": "s2", "library": "numpy", "target_api": "zeros"},
            {"id": "s3", "library": "pytorch", "target_api": "randn"},
            {"id": "s4", "library": "pytorch", "target_api": "tensor"},
        ]
        base_rows = [
            {"id": "s1", "predicted_api": "array", "exact_api_match": True},
            {"id": "s2", "predicted_api": "ones", "exact_api_match": False},
            {"id": "s3", "predicted_api": "randn", "exact_api_match": True},
            {"id": "s4", "predicted_api": "tensor", "exact_api_match": True},
        ]
        variant_rows = [
            {"id": "s1", "predicted_api": "array", "exact_api_match": True},
            {"id": "s2", "predicted_api": "zeros", "exact_api_match": True},
            {"id": "s3", "predicted_api": "tensor", "exact_api_match": False},
            {"id": "s4", "predicted_api": "tensor", "exact_api_match": True},
        ]

        summary = summarize_shift_against_base(
            candidate_rows=candidate_rows,
            base_rows=base_rows,
            variant_rows=variant_rows,
            libraries=["numpy", "pytorch"],
        )

        self.assertEqual(summary["overall"]["samples"], 4)
        self.assertAlmostEqual(summary["overall"]["base_exact_match_rate"], 0.75)
        self.assertAlmostEqual(summary["overall"]["variant_exact_match_rate"], 0.75)
        self.assertAlmostEqual(summary["overall"]["prediction_changed_rate"], 0.5)
        self.assertAlmostEqual(summary["overall"]["recovered_from_base_error_rate"], 0.25)
        self.assertAlmostEqual(summary["overall"]["degraded_from_base_correct_rate"], 0.25)

        numpy_stats = summary["by_library"]["numpy"]
        self.assertEqual(numpy_stats["samples"], 2)
        self.assertAlmostEqual(numpy_stats["prediction_changed_rate"], 0.5)
        self.assertAlmostEqual(numpy_stats["recovered_from_base_error_rate"], 0.5)

        pytorch_stats = summary["by_library"]["pytorch"]
        self.assertEqual(pytorch_stats["samples"], 2)
        self.assertAlmostEqual(pytorch_stats["prediction_changed_rate"], 0.5)
        self.assertAlmostEqual(pytorch_stats["degraded_from_base_correct_rate"], 0.5)

    def test_build_triplet_library_table_emits_one_row_per_library(self):
        payload = {
            "libraries": ["numpy", "pytorch"],
            "variants": {
                "dpo": {
                    "summary": {
                        "by_library": {
                            "numpy": {
                                "samples": 2,
                                "base_exact_match_rate": 1.0,
                                "variant_exact_match_rate": 0.5,
                                "prediction_changed_rate": 0.5,
                            },
                            "pytorch": {
                                "samples": 3,
                                "base_exact_match_rate": 1.0,
                                "variant_exact_match_rate": 2 / 3,
                                "prediction_changed_rate": 1 / 3,
                            },
                        }
                    }
                },
                "anchored_dpo": {
                    "summary": {
                        "by_library": {
                            "numpy": {
                                "samples": 2,
                                "base_exact_match_rate": 1.0,
                                "variant_exact_match_rate": 1.0,
                                "prediction_changed_rate": 0.0,
                            },
                            "pytorch": {
                                "samples": 3,
                                "base_exact_match_rate": 1.0,
                                "variant_exact_match_rate": 1 / 3,
                                "prediction_changed_rate": 2 / 3,
                            },
                        }
                    }
                },
            },
        }

        rows = build_triplet_library_table(payload)

        self.assertEqual(
            rows,
            [
                {
                    "library": "numpy",
                    "samples": 2,
                    "base_exact_match_rate": 1.0,
                    "dpo_exact_match_rate": 0.5,
                    "dpo_prediction_changed_rate": 0.5,
                    "anchored_dpo_exact_match_rate": 1.0,
                    "anchored_dpo_prediction_changed_rate": 0.0,
                },
                {
                    "library": "pytorch",
                    "samples": 3,
                    "base_exact_match_rate": 1.0,
                    "dpo_exact_match_rate": 2 / 3,
                    "dpo_prediction_changed_rate": 1 / 3,
                    "anchored_dpo_exact_match_rate": 1 / 3,
                    "anchored_dpo_prediction_changed_rate": 2 / 3,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
