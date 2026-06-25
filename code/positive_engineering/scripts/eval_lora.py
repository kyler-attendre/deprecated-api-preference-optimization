#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List
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
    raise FileNotFoundError(
        f"{label} not found: {path}\nTried these locations:\n{tried}"
    )


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a version-aware LoRA adapter on held-out data")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    args = parser.parse_args()

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Please install transformers and peft in the evaluation environment."
        ) from exc

    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
    test_file = resolve_existing_path(args.test_file, label="test file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    deprecated_hits = 0
    replacement_hits = 0
    exact_matches = 0
    total = 0

    with predictions_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Evaluating LoRA", total=len(rows)):
            prompt = row["version_prompt"]
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            deprecated_api = row.get("deprecated_api", "")
            replacement_api = row.get("replacement_api", "")
            reference = row.get("reference", "")
            target = row.get("target", "")

            has_deprecated = bool(deprecated_api) and deprecated_api in prediction
            has_replacement = bool(replacement_api) and replacement_api in prediction
            exact_match = normalize_text(prediction) == normalize_text(target)

            deprecated_hits += int(has_deprecated)
            replacement_hits += int(has_replacement)
            exact_matches += int(exact_match)
            total += 1

            fout.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "model": row.get("model"),
                        "library": row.get("library"),
                        "category": row.get("category"),
                        "sample_type": row.get("sample_type"),
                        "deprecated_api": deprecated_api,
                        "replacement_api": replacement_api,
                        "reference": reference,
                        "target": target,
                        "prediction": prediction,
                        "has_deprecated": has_deprecated,
                        "has_replacement": has_replacement,
                        "exact_match_target": exact_match,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(adapter_dir),
        "test_file": str(test_file),
        "samples": total,
        "max_new_tokens": args.max_new_tokens,
        "deprecated_usage_rate": deprecated_hits / total if total else 0.0,
        "replacement_hit_rate": replacement_hits / total if total else 0.0,
        "exact_match_target_rate": exact_matches / total if total else 0.0,
        "predictions_file": str(predictions_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
