import unittest

import torch

from src.dpo_training import (
    DPOPair,
    DPOCollator,
    VersionAwareDPODataset,
    VersionAwareDPOTrainer,
    api_anchor_cross_entropy_loss,
    build_dpo_pair,
    disable_dropout_in_model,
    dpo_loss,
    encode_api_anchor,
    temporary_eval_mode,
    sequence_log_probs,
)


class DPOTrainingTest(unittest.TestCase):
    def test_build_dpo_pair_constructs_chosen_and_rejected_completion(self):
        row = {
            "id": "row-1",
            "library": "pytorch",
            "version_prompt": "# pytorch 2.3.0\nimport torch\nx = torch.randn(3, 3)\n",
            "target": "q, r = torch.linalg.qr(x)\nreturn q",
            "deprecated_api": ["torch.qr"],
            "replacement_api": "torch.linalg.qr",
        }

        pair = build_dpo_pair(row)

        self.assertEqual(
            pair,
            DPOPair(
                row_id="row-1",
                library="pytorch",
                prompt="# pytorch 2.3.0\nimport torch\nx = torch.randn(3, 3)\n",
                chosen="q, r = torch.linalg.qr(x)\nreturn q",
                rejected="q, r = torch.qr(x)\nreturn q",
                replacement_form="torch.linalg.qr",
                deprecated_form="torch.qr",
            ),
        )

    def test_build_dpo_pair_skips_rows_without_replacement_in_target(self):
        row = {
            "version_prompt": "# numpy 1.26.4\n",
            "target": "return x",
            "deprecated_api": ["numpy.old"],
            "replacement_api": "numpy.new",
        }

        self.assertIsNone(build_dpo_pair(row))

    def test_sequence_log_probs_uses_only_unmasked_label_positions(self):
        logits = torch.full((1, 4, 5), -10.0)
        labels = torch.tensor([[-100, 2, 3, -100]])
        logits[0, 0, 2] = 10.0
        logits[0, 1, 3] = 10.0

        result = sequence_log_probs(logits, labels)

        self.assertAlmostEqual(float(result.item()), 0.0, places=4)

    def test_encode_api_anchor_masks_everything_except_replacement_api(self):
        tokenizer = WhitespaceTokenizer()

        encoded = encode_api_anchor(
            tokenizer,
            prompt="# pytorch 2.3.0\n",
            completion="q r = torch.linalg.qr x",
            api_form="torch.linalg.qr",
            max_length=32,
        )

        label_tokens = [
            tokenizer.id_to_token[int(token_id)]
            for token_id in encoded["labels"].tolist()
            if int(token_id) != -100
        ]
        self.assertEqual(label_tokens, ["torch.linalg.qr"])
        self.assertNotIn("torch.qr", label_tokens)

    def test_dataset_includes_rejected_api_anchor_for_api_span_dpo(self):
        tokenizer = WhitespaceTokenizer()
        dataset = VersionAwareDPODataset(
            [
                {
                    "id": "row-1",
                    "library": "pytorch",
                    "version_prompt": "# pytorch 2.3.0\n",
                    "target": "q r = torch.linalg.qr x",
                    "deprecated_api": ["torch.qr"],
                    "replacement_api": "torch.linalg.qr",
                }
            ],
            tokenizer,
            max_length=32,
        )

        item = dataset[0]

        chosen_label_tokens = [
            tokenizer.id_to_token[int(token_id)]
            for token_id in item["api_anchor_labels"].tolist()
            if int(token_id) != -100
        ]
        rejected_label_tokens = [
            tokenizer.id_to_token[int(token_id)]
            for token_id in item["rejected_api_anchor_labels"].tolist()
            if int(token_id) != -100
        ]
        self.assertEqual(chosen_label_tokens, ["torch.linalg.qr"])
        self.assertEqual(rejected_label_tokens, ["torch.qr"])

    def test_collator_pads_rejected_api_anchor_fields(self):
        tokenizer = WhitespaceTokenizer()
        dataset = VersionAwareDPODataset(
            [
                {
                    "id": "row-1",
                    "library": "pytorch",
                    "version_prompt": "# pytorch 2.3.0\n",
                    "target": "q r = torch.linalg.qr x",
                    "deprecated_api": ["torch.qr"],
                    "replacement_api": "torch.linalg.qr",
                }
            ],
            tokenizer,
            max_length=32,
        )

        batch = DPOCollator(tokenizer)([dataset[0]])

        self.assertIn("rejected_api_anchor_input_ids", batch)
        self.assertIn("rejected_api_anchor_attention_mask", batch)
        self.assertIn("rejected_api_anchor_labels", batch)

    def test_dpo_trainer_rejects_unknown_dpo_scope(self):
        with self.assertRaises(ValueError):
            VersionAwareDPOTrainer(dpo_scope="unknown")

    def test_api_anchor_cross_entropy_prefers_replacement_api_token(self):
        labels = torch.tensor([[-100, 1, -100]])
        good_logits = torch.full((1, 3, 4), -5.0)
        bad_logits = torch.full((1, 3, 4), -5.0)
        good_logits[0, 0, 1] = 5.0
        bad_logits[0, 0, 2] = 5.0

        good_loss = api_anchor_cross_entropy_loss(good_logits, labels)
        bad_loss = api_anchor_cross_entropy_loss(bad_logits, labels)

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_dpo_loss_decreases_when_policy_prefers_chosen_more_than_reference(self):
        policy_chosen = torch.tensor([5.0])
        policy_rejected = torch.tensor([1.0])
        reference_chosen = torch.tensor([2.0])
        reference_rejected = torch.tensor([1.0])

        good_loss = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1,
        )
        bad_loss = dpo_loss(
            policy_rejected,
            policy_chosen,
            reference_chosen,
            reference_rejected,
            beta=0.1,
        )

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_disable_dropout_in_model_sets_dropout_probability_to_zero(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Dropout(p=0.5),
            torch.nn.Sequential(torch.nn.Dropout(p=0.2)),
        )

        changed = disable_dropout_in_model(model)

        self.assertEqual(changed, 2)
        self.assertEqual(model[1].p, 0.0)
        self.assertEqual(model[2][0].p, 0.0)

    def test_temporary_eval_mode_restores_original_training_state(self):
        model = torch.nn.Linear(2, 2)
        model.train()

        with temporary_eval_mode(model):
            self.assertFalse(model.training)

        self.assertTrue(model.training)

        model.eval()
        with temporary_eval_mode(model):
            self.assertFalse(model.training)

        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()


class WhitespaceTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def __init__(self):
        self.token_to_id = {"<pad>": 0}
        self.id_to_token = {0: "<pad>"}

    def __call__(self, text, add_special_tokens=False):
        tokens = str(text).split()
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return {"input_ids": ids}
