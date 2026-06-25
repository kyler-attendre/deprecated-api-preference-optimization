import unittest

from scripts.train_dpo_lora import build_lora_config_kwargs, parse_int_list


class TrainDpoLoraTest(unittest.TestCase):
    def test_parse_int_list_accepts_comma_and_space_separated_values(self):
        values = ["17,18, 19", "21", "19", "30,31"]

        parsed = parse_int_list(values)

        self.assertEqual(parsed, [17, 18, 19, 21, 30, 31])

    def test_build_lora_config_kwargs_includes_layers_when_provided(self):
        kwargs = build_lora_config_kwargs(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["o_proj"],
            layers_to_transform=[17, 18, 19],
            task_type="causal_lm",
        )

        self.assertEqual(kwargs["target_modules"], ["o_proj"])
        self.assertEqual(kwargs["layers_to_transform"], [17, 18, 19])
        self.assertEqual(kwargs["bias"], "none")

    def test_build_lora_config_kwargs_omits_layers_when_not_provided(self):
        kwargs = build_lora_config_kwargs(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "o_proj"],
            layers_to_transform=None,
            task_type="causal_lm",
        )

        self.assertEqual(kwargs["target_modules"], ["q_proj", "o_proj"])
        self.assertNotIn("layers_to_transform", kwargs)


if __name__ == "__main__":
    unittest.main()
