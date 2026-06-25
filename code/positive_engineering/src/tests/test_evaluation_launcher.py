import unittest
from pathlib import Path


class EvaluationLauncherTest(unittest.TestCase):
    def test_bigcode_eval_outputs_inside_project_output(self):
        script = Path("scripts/run_official_code_eval_nohup.sh").read_text(encoding="utf-8")

        self.assertIn('RUN_ROOT="${PROJECT_ROOT}/output/${TAG}"', script)
        self.assertNotIn('EVALUATION_ROOT="${EVALUATION_ROOT:-/workspace/evaluation}"', script)
        self.assertNotIn('RUN_ROOT="${EVALUATION_ROOT}/${TAG}"', script)

    def test_launcher_uses_bigcode_harness_instead_of_custom_generator(self):
        script = Path("scripts/run_official_code_eval_nohup.sh").read_text(encoding="utf-8")

        self.assertIn('BIGCODE_ROOT="${BIGCODE_ROOT:-/workspace/bigcode-evaluation-harness}"', script)
        self.assertIn('BIGCODE_LOCAL_HUMANEVAL_FILE="${BIGCODE_LOCAL_HUMANEVAL_FILE:-/workspace/evaluation/human-eval/data/HumanEval.jsonl.gz}"', script)
        self.assertIn('BIGCODE_LOCAL_MBPP_FILE="${BIGCODE_LOCAL_MBPP_FILE:-/workspace/evaluation/google-research/mbpp/mbpp.jsonl}"', script)
        self.assertIn('export BIGCODE_LOCAL_HUMANEVAL_FILE', script)
        self.assertIn('export BIGCODE_LOCAL_MBPP_FILE', script)
        self.assertIn('"${PYTHON_BIN}" "${BIGCODE_ROOT}/main.py"', script)
        self.assertIn('--tasks "${bigcode_task}"', script)
        self.assertIn('--peft_model "${adapter_dir}"', script)
        self.assertIn('--n_samples 1', script)
        self.assertIn('--allow_code_execution', script)
        self.assertNotIn('generate_official_code_samples.py', script)
        self.assertNotIn('evaluate_mbpp_official.py', script)
        self.assertNotIn('from human_eval.evaluation import evaluate_functional_correctness', script)


if __name__ == "__main__":
    unittest.main()
