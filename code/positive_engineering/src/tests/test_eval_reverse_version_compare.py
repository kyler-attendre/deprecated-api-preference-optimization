import unittest

from scripts.eval_reverse_version_compare import (
    COARSE_OLD_VERSION_HEADERS,
    build_reverse_subset,
    rewrite_row_with_coarse_old_version,
)


class ReverseVersionCompareTest(unittest.TestCase):
    def test_rewrite_row_with_coarse_old_version_replaces_prefix_header(self):
        row = {
            "id": "p1",
            "library": "pytorch",
            "version_prompt": "# pytorch 2.3.0\nimport torch\nx = torch.randn(3, 3)\n",
            "probing_input": "x = torch.randn(3, 3)\n",
        }

        rewritten = rewrite_row_with_coarse_old_version(row)

        self.assertTrue(rewritten["version_prompt"].startswith(COARSE_OLD_VERSION_HEADERS["pytorch"]))
        self.assertIn("x = torch.randn(3, 3)", rewritten["version_prompt"])
        self.assertEqual(rewritten["original_version_prompt"], row["version_prompt"])
        self.assertEqual(rewritten["reverse_version_kind"], "coarse_old_version")

    def test_build_reverse_subset_keeps_only_base_replacement_hits(self):
        rows = [
            {"id": "a", "library": "pytorch", "version_prompt": "# pytorch 2.3.0\nimport torch\n", "probing_input": ""},
            {"id": "b", "library": "numpy", "version_prompt": "# numpy 1.26.4\nimport numpy\n", "probing_input": ""},
        ]
        base_predictions = [
            {"id": "a", "has_replacement": True},
            {"id": "b", "has_replacement": False},
        ]

        subset = build_reverse_subset(rows, base_predictions)

        self.assertEqual([row["id"] for row in subset], ["a"])
        self.assertEqual(subset[0]["reverse_version_kind"], "coarse_old_version")


if __name__ == "__main__":
    unittest.main()
