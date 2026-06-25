import unittest

import torch

from scripts.eval_sparse_lora_delta import prune_delta_keep_fraction


class EvalSparseLoraDeltaTest(unittest.TestCase):
    def test_prune_delta_keep_fraction_keeps_largest_magnitudes(self):
        delta = torch.tensor([[1.0, -4.0], [0.5, 3.0]])

        pruned = prune_delta_keep_fraction(delta, keep_fraction=0.5)

        expected = torch.tensor([[0.0, -4.0], [0.0, 3.0]])
        self.assertTrue(torch.equal(pruned, expected))

    def test_keep_fraction_one_returns_original_delta(self):
        delta = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

        pruned = prune_delta_keep_fraction(delta, keep_fraction=1.0)

        self.assertTrue(torch.equal(pruned, delta))


if __name__ == "__main__":
    unittest.main()
