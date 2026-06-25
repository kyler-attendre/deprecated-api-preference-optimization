import json
import sys
import tempfile
import unittest
from pathlib import Path


MECH_ROOT = Path(__file__).resolve().parents[1]
if str(MECH_ROOT) not in sys.path:
    sys.path.insert(0, str(MECH_ROOT))

from src.lens_analysis import build_focus_example, build_focus_examples, focus_example_to_dict  # noqa: E402


class FocusExampleTests(unittest.TestCase):
    def test_build_focus_example_uses_shared_api_prefix(self):
        row = {
            "id": "row-1",
            "library": "pytorch",
            "category": "repair",
            "task_family": "repair",
            "sample_type": "test",
            "source_file": "demo.py",
            "version_prompt": "# pytorch 2.3.0\nimport torch\n",
            "target": "    U, S, V = torch.linalg.svd(F)\n",
            "replacement_api": "torch.linalg.svd",
            "deprecated_api": ["torch.svd"],
        }
        example = build_focus_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.shared_api_prefix, "torch.")
        self.assertTrue(example.decision_prefix.endswith("torch."))
        self.assertEqual(example.replacement_suffix, "linalg.svd")
        self.assertEqual(example.deprecated_suffix, "svd")

    def test_build_focus_example_respects_alias_form(self):
        row = {
            "id": "row-2",
            "library": "pytorch",
            "category": "consistency",
            "task_family": "consistency",
            "sample_type": "test",
            "source_file": "demo.py",
            "version_prompt": "# pytorch 2.3.0\nimport torch.nn.functional as F\n",
            "target": "    edge = F.interpolate(edge, size=(h, w), mode='bilinear')\n",
            "replacement_api": "torch.nn.functional.interpolate",
            "deprecated_api": ["torch.nn.functional.upsample"],
        }
        example = build_focus_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.shared_api_prefix, "F.")
        self.assertEqual(example.replacement_suffix, "interpolate")
        self.assertEqual(example.deprecated_suffix, "upsample")

    def test_build_focus_examples_filters_unusable_rows(self):
        rows = [
            {
                "id": "ok",
                "library": "numpy",
                "category": "consistency",
                "task_family": "consistency",
                "sample_type": "test",
                "source_file": "demo.py",
                "version_prompt": "# numpy 1.26.4\nimport numpy as np\n",
                "target": "np.prod(x)\n",
                "replacement_api": "numpy.prod",
                "deprecated_api": ["numpy.product"],
            },
            {
                "id": "bad",
                "library": "numpy",
                "version_prompt": "# numpy 1.26.4\n",
                "target": "np.sum(x)\n",
                "replacement_api": "numpy.prod",
                "deprecated_api": ["numpy.product"],
            },
        ]
        examples = build_focus_examples(rows)
        self.assertEqual(len(examples), 1)
        payload = focus_example_to_dict(examples[0])
        self.assertEqual(payload["row_id"], "ok")


if __name__ == "__main__":
    unittest.main()
