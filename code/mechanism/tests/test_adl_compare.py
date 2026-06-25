import sys
import unittest
from pathlib import Path


MECH_ROOT = Path(__file__).resolve().parents[1]
if str(MECH_ROOT) not in sys.path:
    sys.path.insert(0, str(MECH_ROOT))

from src.adl_compare import (  # noqa: E402
    build_adl_rows,
    build_library_neutral_code_prompts,
    build_prompt_sets,
    build_random_neutral_text_prompts,
    collect_blocklist_fragments,
    contains_banned_fragment,
)


class AdlCompareTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "id": "row-1",
                "library": "pytorch",
                "version_prompt": "# pytorch 2.3.0\nimport torch\nx = 1\n",
                "probing_input": "def helper(values):\n    total = 0\n    for value in values:\n        total += int(value)\n    return total\n",
                "replacement_api": "torch.linalg.svd",
                "deprecated_api": ["torch.svd"],
            },
            {
                "id": "row-2",
                "library": "numpy",
                "version_prompt": "# numpy 1.26.4\nimport numpy as np\nx = 1\n",
                "probing_input": "def build_pairs(items):\n    pairs = []\n    for item in items:\n        pairs.append((item, len(str(item))))\n    return pairs\n",
                "replacement_api": "numpy.prod",
                "deprecated_api": ["numpy.product"],
            },
            {
                "id": "bad",
                "library": "pytorch",
                "version_prompt": "# pytorch 2.3.0\n",
                "probing_input": "def direct_api(x):\n    return torch.linalg.svd(x)\n",
                "replacement_api": "torch.linalg.svd",
                "deprecated_api": ["torch.svd"],
            },
        ]

    def test_collect_blocklist_fragments_contains_expected_items(self):
        fragments = collect_blocklist_fragments(self.rows)
        self.assertIn("torch.linalg.svd", fragments)
        self.assertIn("torch.svd", fragments)
        self.assertIn("pytorch", fragments)
        self.assertTrue(contains_banned_fragment("use torch.linalg.svd here", fragments))

    def test_build_library_neutral_code_prompts_filters_direct_api_mentions(self):
        prompts = build_library_neutral_code_prompts(self.rows, max_prompts=4)
        self.assertGreaterEqual(len(prompts), 2)
        prompt_ids = {prompt.source_row_id for prompt in prompts if prompt.source_row_id}
        self.assertIn("row-1", prompt_ids)
        self.assertIn("row-2", prompt_ids)
        self.assertNotIn("bad", prompt_ids)
        self.assertTrue(all(prompt.prompt_type == "library_neutral_code" for prompt in prompts))

    def test_build_random_neutral_text_prompts_uses_expected_label(self):
        prompts = build_random_neutral_text_prompts(self.rows, max_prompts=3)
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all(prompt.prompt_type == "random_neutral_text" for prompt in prompts))

    def test_build_prompt_sets_combines_both_types(self):
        prompts = build_prompt_sets(self.rows, max_prompts_per_type=2)
        prompt_types = {prompt.prompt_type for prompt in prompts}
        self.assertEqual(prompt_types, {"random_neutral_text", "library_neutral_code"})

    def test_build_adl_rows_injects_base_rows(self):
        run_summary = {
            "prompt_counts": {"random_neutral_text": 2, "library_neutral_code": 2},
            "variants": {
                "dpo": {
                    "prompt_type_summary": {
                        "random_neutral_text": {
                            "count": 2,
                            "final_layer_mean_diff_norm": 3.0,
                            "mean_family_scores": {
                                "replacement": 0.4,
                                "deprecated": -0.2,
                                "library_version": 0.1,
                            },
                            "replacement_minus_deprecated_score": 0.6,
                        }
                    }
                }
            },
        }
        rows = build_adl_rows("starcoder2_7b", run_summary)
        self.assertEqual(len(rows), 3)
        base_rows = [row for row in rows if row["variant"] == "base"]
        self.assertEqual(len(base_rows), 2)
        dpo_rows = [row for row in rows if row["variant"] == "dpo"]
        self.assertEqual(len(dpo_rows), 1)


if __name__ == "__main__":
    unittest.main()
