import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_lora_prompt_ablation import build_rows


def write_summary(path: Path, *, base_dep, base_repl, lora_dep, lora_repl, lora_exact=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "base": {
                    "deprecated_usage_rate": base_dep,
                    "replacement_hit_rate": base_repl,
                },
                "lora": {
                    "deprecated_usage_rate": lora_dep,
                    "replacement_hit_rate": lora_repl,
                    "exact_match_target_rate": lora_exact,
                },
            }
        ),
        encoding="utf-8",
    )


class LoraPromptAblationTest(unittest.TestCase):
    def test_summary_combines_original_and_version_prompt_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "starcoder2_3b"
            write_summary(
                model_dir / "original_prompt" / "lora_version_compare" / "comparison_summary.json",
                base_dep=0.2,
                base_repl=0.1,
                lora_dep=0.12,
                lora_repl=0.3,
            )
            write_summary(
                model_dir / "original_prompt" / "lora_no_version_compare" / "comparison_summary.json",
                base_dep=0.2,
                base_repl=0.1,
                lora_dep=0.18,
                lora_repl=0.2,
            )
            write_summary(
                model_dir / "version_prompt" / "lora_version_compare" / "comparison_summary.json",
                base_dep=0.09,
                base_repl=0.15,
                lora_dep=0.01,
                lora_repl=0.55,
            )
            write_summary(
                model_dir / "version_prompt" / "lora_no_version_compare" / "comparison_summary.json",
                base_dep=0.09,
                base_repl=0.15,
                lora_dep=0.05,
                lora_repl=0.25,
            )

            rows = build_rows(root, ["starcoder2_3b"])

        self.assertEqual(len(rows), 2)
        version_row = next(row for row in rows if row["test_prompt"] == "version_prompt")
        self.assertEqual(version_row["baseline_deprecated"], 0.09)
        self.assertEqual(version_row["lora_with_version_replacement"], 0.55)
        self.assertEqual(version_row["lora_without_version_deprecated"], 0.05)

    def test_launcher_declares_prompt_field_grid(self):
        script = Path("scripts/run_lora_prompt_ablation_nohup.sh").read_text(encoding="utf-8")

        self.assertIn("--prompt-field probing_input", script)
        self.assertIn("--prompt-field version_prompt", script)
        self.assertIn("--max-train-samples", script)
        self.assertIn("--max-val-samples", script)
        self.assertIn("lora_no_version_compare", script)
        self.assertIn("lora_version_compare", script)

    def test_gpu_04_execution_script_activates_lkl_llm(self):
        script = Path("scripts/launch_lora_prompt_ablation_gpu04.sh").read_text(encoding="utf-8")

        self.assertIn("conda activate lkl_llm", script)
        self.assertIn("--gpu 0", script)
        self.assertIn("--gpu 4", script)
        self.assertIn("lora_prompt_ablation_20260422_full", script)


if __name__ == "__main__":
    unittest.main()
