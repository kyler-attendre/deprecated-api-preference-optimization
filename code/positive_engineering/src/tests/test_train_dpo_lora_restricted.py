import unittest

from scripts.train_dpo_lora_restricted import parse_int_list


class TrainDpoLoraRestrictedTest(unittest.TestCase):
    def test_parse_int_list_accepts_comma_and_space_separated_values(self):
        values = ["17,18, 19", "21", "19", "30,31"]

        parsed = parse_int_list(values)

        self.assertEqual(parsed, [17, 18, 19, 21, 30, 31])


if __name__ == "__main__":
    unittest.main()
