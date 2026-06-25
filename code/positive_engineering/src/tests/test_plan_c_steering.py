import json
from pathlib import Path
import unittest

from src.plan_c_steering import (
    build_contrast_pair,
    iter_official_mbpp_test_rows,
    parse_layer_spec,
    safe_model_label,
)


class PlanCSteeringTest(unittest.TestCase):
    def test_parse_layer_spec_accepts_ranges_and_lists(self):
        self.assertEqual(parse_layer_spec("2:5"), [2, 3, 4, 5])
        self.assertEqual(parse_layer_spec("0,2,4"), [0, 2, 4])
        self.assertEqual(parse_layer_spec("1,3:5"), [1, 3, 4, 5])

    def test_parse_layer_spec_rejects_empty_specs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_layer_spec("")

    def test_build_contrast_pair_uses_library_api_mapping(self):
        row = {
            "library": "pytorch",
            "version_prompt": "# pytorch 2.3.0\nimport torch\nx = torch.randn(3, 3)\n",
            "target": "q, r = torch.linalg.qr(x)\nreturn q",
            "deprecated_api": ["torch.qr"],
            "replacement_api": "torch.linalg.qr",
        }

        pair = build_contrast_pair(row)

        self.assertIsNotNone(pair)
        self.assertEqual(pair.library, "pytorch")
        self.assertTrue(pair.positive_text.endswith("torch.linalg.qr(x)\nreturn q"))
        self.assertTrue(pair.negative_text.endswith("torch.qr(x)\nreturn q"))

    def test_build_contrast_pair_handles_tensorflow_aliases(self):
        row = {
            "library": "tensorflow",
            "version_prompt": "# tensorflow 2.16.1\nimport tensorflow as tf\n",
            "target": "x = tf.random.normal(shape)",
            "deprecated_api": ["tensorflow.random_normal"],
            "replacement_api": "tensorflow.random.normal",
        }

        pair = build_contrast_pair(row)

        self.assertIsNotNone(pair)
        self.assertIn("tf.random.normal", pair.positive_text)
        self.assertIn("tf.random_normal", pair.negative_text)

    def test_build_contrast_pair_skips_rows_without_replacement_hit(self):
        row = {
            "library": "numpy",
            "version_prompt": "# numpy 1.26.4\nimport numpy\n",
            "target": "return value",
            "deprecated_api": ["numpy.old"],
            "replacement_api": "numpy.new",
        }

        self.assertIsNone(build_contrast_pair(row))

    def test_safe_model_label_normalizes_paths(self):
        self.assertEqual(
            safe_model_label("/data/models/Qwen/Qwen2.5-Coder-7B-Instruct"),
            "qwen2_5_coder_7b_instruct",
        )
        self.assertEqual(safe_model_label("bigcode/starcoder2-3b"), "starcoder2_3b")

    def test_iter_official_mbpp_test_rows_uses_google_split_and_prompt(self):
        data = [
            {
                "task_id": 2,
                "prompt": "Write a function a.",
                "code": "def a():\n    return 1",
                "test_list": ["assert a() == 1"],
            },
            {
                "task_id": 3,
                "prompt": "Write a function b.",
                "code": "def b():\n    return 2",
                "test_list": ["assert b() == 2"],
            },
            {
                "task_id": 4,
                "prompt": "Write a function c.",
                "code": "def c():\n    return 3",
                "test_list": ["assert c() == 3"],
            },
            {
                "task_id": 11,
                "prompt": "Write a function target.",
                "code": "def target():\n    return 4",
                "test_list": ["assert target() == 4", "assert target() != 5"],
            },
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sanitized-mbpp.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            rows = list(iter_official_mbpp_test_rows(path))

        self.assertEqual([row["task_id"] for row in rows], ["mbpp/11"])
        self.assertEqual(rows[0]["prompt"].count("[BEGIN]"), 4)
        self.assertEqual(rows[0]["prompt"].count("[DONE]"), 3)
        self.assertEqual(rows[0]["tests"], ["assert target() == 4", "assert target() != 5"])


if __name__ == "__main__":
    unittest.main()
