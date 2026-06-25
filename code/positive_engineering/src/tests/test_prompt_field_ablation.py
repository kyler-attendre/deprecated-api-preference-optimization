import unittest

from scripts.train_lora import VersionAwareSFTDataset, limit_rows
from scripts.eval_compare_lora import select_prompt


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


class PromptFieldAblationTest(unittest.TestCase):
    def test_train_dataset_can_use_original_prompt_without_version_prefix(self):
        rows = [
            {
                "version_prompt": "# pytorch 2.3.0\nimport torch\nx = torch.randn(3, 3)\n",
                "probing_input": "x = torch.randn(3, 3)\n",
                "target": "q, r = torch.linalg.qr(x)\n",
            }
        ]
        dataset = VersionAwareSFTDataset(
            rows,
            FakeTokenizer(),
            max_length=256,
            prompt_field="probing_input",
        )

        item = dataset[0]
        input_text = "".join(chr(token_id) for token_id in item["input_ids"].tolist() if token_id)

        self.assertIn("x = torch.randn", input_text)
        self.assertNotIn("# pytorch 2.3.0", input_text)

    def test_eval_prompt_selection_supports_original_and_version_prompt(self):
        row = {
            "version_prompt": "# numpy 1.26.4\nimport numpy\nnumpy.array([1])",
            "probing_input": "numpy.array([1])",
        }

        self.assertEqual(select_prompt(row, "version_prompt"), row["version_prompt"])
        self.assertEqual(select_prompt(row, "probing_input"), row["probing_input"])

    def test_limit_rows_keeps_all_rows_when_limit_is_zero(self):
        rows = [{"id": 1}, {"id": 2}, {"id": 3}]

        self.assertEqual(limit_rows(rows, 0), rows)
        self.assertEqual(limit_rows(rows, 2), rows[:2])


if __name__ == "__main__":
    unittest.main()
