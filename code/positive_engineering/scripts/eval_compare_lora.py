#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_existing_path(path: Path, *, label: str) -> Path:
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"{label} not found: {path}")

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    tried = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"{label} not found: {path}\nTried these locations:\n{tried}")


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    value = str(value).strip()
    return [value] if value else []


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def select_prompt(row: Dict, prompt_field: str) -> str:
    if prompt_field not in row:
        raise KeyError(f"Prompt field {prompt_field!r} not found in row")
    return row[prompt_field]


def alias_forms(api_name: str) -> List[str]:
    aliases = {api_name}

    if api_name.startswith("torch.nn.functional."):
        aliases.add("F." + api_name.split(".")[-1])
    if api_name.startswith("tensorflow."):
        aliases.add("tf." + api_name.split(".", 1)[1])

    return sorted(x for x in aliases if x)


def any_api_mentioned(prediction: str, api_names: Iterable[str]) -> bool:
    for api_name in api_names:
        for alias in alias_forms(api_name):
            if alias and alias in prediction:
                return True
    return False


def build_model(model_name_or_path: str, adapter_dir: Optional[Path] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def evaluate_model(
    *,
    label: str,
    model,
    tokenizer,
    rows: List[Dict],
    output_path: Path,
    max_length: int,
    max_new_tokens: int,
    prompt_field: str,
) -> Dict:
    deprecated_hits = 0
    replacement_hits = 0
    exact_match_hits = 0
    nonempty_hits = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"Evaluating {label}", total=len(rows)):
            prompt = select_prompt(row, prompt_field)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            deprecated_apis = ensure_list(row.get("deprecated_api"))
            replacement_apis = ensure_list(row.get("replacement_api"))
            target = row.get("target", "")

            has_deprecated = any_api_mentioned(prediction, deprecated_apis)
            has_replacement = any_api_mentioned(prediction, replacement_apis)
            exact_match = normalize_text(prediction) == normalize_text(target)
            nonempty = bool(prediction.strip())

            deprecated_hits += int(has_deprecated)
            replacement_hits += int(has_replacement)
            exact_match_hits += int(exact_match)
            nonempty_hits += int(nonempty)

            fout.write(
                json.dumps(
                    {
                        "run_label": label,
                        "id": row.get("id"),
                        "model": row.get("model"),
                        "library": row.get("library"),
                        "category": row.get("category"),
                        "sample_type": row.get("sample_type"),
                        "deprecated_api": deprecated_apis,
                        "replacement_api": replacement_apis,
                        "target": target,
                        "prediction": prediction,
                        "has_deprecated": has_deprecated,
                        "has_replacement": has_replacement,
                        "exact_match_target": exact_match,
                        "nonempty_prediction": nonempty,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(rows)
    return {
        "label": label,
        "samples": total,
        "deprecated_usage_rate": deprecated_hits / total if total else 0.0,
        "replacement_hit_rate": replacement_hits / total if total else 0.0,
        "exact_match_target_rate": exact_match_hits / total if total else 0.0,
        "nonempty_prediction_rate": nonempty_hits / total if total else 0.0,
        "predictions_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare base model and LoRA adapter on a held-out dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="version_prompt",
        choices=["version_prompt", "probing_input"],
        help="Input field used for generation. Default keeps version-prompt evaluation.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    args = parser.parse_args()

    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
    test_file = resolve_existing_path(args.test_file, label="test file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    base_model, base_tokenizer = build_model(args.model_name_or_path)
    base_summary = evaluate_model(
        label="base",
        model=base_model,
        tokenizer=base_tokenizer,
        rows=rows,
        output_path=output_dir / "base_predictions.jsonl",
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        prompt_field=args.prompt_field,
    )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_model, lora_tokenizer = build_model(args.model_name_or_path, adapter_dir=adapter_dir)
    lora_summary = evaluate_model(
        label="lora",
        model=lora_model,
        tokenizer=lora_tokenizer,
        rows=rows,
        output_path=output_dir / "lora_predictions.jsonl",
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        prompt_field=args.prompt_field,
    )

    comparison = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(adapter_dir),
        "test_file": str(test_file),
        "prompt_field": args.prompt_field,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "base": base_summary,
        "lora": lora_summary,
        "delta": {
            "deprecated_usage_rate": lora_summary["deprecated_usage_rate"] - base_summary["deprecated_usage_rate"],
            "replacement_hit_rate": lora_summary["replacement_hit_rate"] - base_summary["replacement_hit_rate"],
            "exact_match_target_rate": lora_summary["exact_match_target_rate"] - base_summary["exact_match_target_rate"],
            "nonempty_prediction_rate": lora_summary["nonempty_prediction_rate"] - base_summary["nonempty_prediction_rate"],
        },
    }

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
