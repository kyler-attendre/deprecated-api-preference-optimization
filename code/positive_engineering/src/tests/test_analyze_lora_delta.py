import unittest

import torch

from scripts.analyze_lora_delta import (
    compute_effective_delta,
    compute_tensor_stats,
    parse_lora_module_key,
)


class AnalyzeLoraDeltaTest(unittest.TestCase):
    def test_parse_lora_module_key_extracts_layer_and_module(self):
        key = "base_model.model.model.layers.17.self_attn.q_proj.lora_A.weight"

        parsed = parse_lora_module_key(key)

        self.assertEqual(parsed, (17, "q_proj", "base_model.model.model.layers.17.self_attn.q_proj"))

    def test_compute_effective_delta_matches_scaled_ba_product(self):
        lora_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        lora_b = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

        delta = compute_effective_delta(lora_a=lora_a, lora_b=lora_b, lora_alpha=6.0, lora_r=2)

        expected = (lora_b @ lora_a) * 3.0
        self.assertTrue(torch.equal(delta, expected))

    def test_compute_tensor_stats_reports_basic_norm_quantiles_and_sparsity(self):
        tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

        stats = compute_tensor_stats(tensor, epsilon=0.5)

        self.assertAlmostEqual(stats["fro_norm"], torch.linalg.norm(tensor).item())
        self.assertAlmostEqual(stats["mean_abs"], 1.5)
        self.assertAlmostEqual(stats["sparsity_below_epsilon"], 0.25)
        self.assertGreaterEqual(stats["q90_abs"], stats["q50_abs"])
        self.assertGreaterEqual(stats["q99_abs"], stats["q95_abs"])


if __name__ == "__main__":
    unittest.main()
