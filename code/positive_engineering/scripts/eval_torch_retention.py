#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plan_c_steering import MODEL_REGISTRY  # noqa: E402


PREDICT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


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
    candidates = [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    tried = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"{label} not found: {path}\nTried these locations:\n{tried}")


def build_model(model_name_or_path: str, adapter_dir: Optional[Path] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def extract_predicted_api(text: str) -> str:
    match = PREDICT_RE.match(text)
    return match.group(1) if match else ""


def evaluate(
    *,
    label: str,
    model,
    tokenizer,
    rows: List[Dict],
    output_path: Path,
    max_length: int,
    max_new_tokens: int,
) -> Dict:
    exact = 0
    nonempty = 0
    per_api_correct: Counter[str] = Counter()
    per_api_total: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_api = extract_predicted_api(prediction)
            target_api = row["target_api"]
            is_exact = predicted_api == target_api

            exact += int(is_exact)
            nonempty += int(bool(predicted_api))
            per_api_total[target_api] += 1
            per_api_correct[target_api] += int(is_exact)

            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "run_label": label,
                        "target_api": target_api,
                        "predicted_text": prediction,
                        "predicted_api": predicted_api,
                        "exact_api_match": is_exact,
                        "source_file": row["source_file"],
                        "function_name": row["function_name"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    per_api = []
    for api in sorted(per_api_total):
        per_api.append(
            {
                "api": api,
                "samples": per_api_total[api],
                "exact_api_match_rate": per_api_correct[api] / per_api_total[api],
            }
        )

    total = len(rows)
    return {
        "label": label,
        "samples": total,
        "exact_api_match_rate": exact / total if total else 0.0,
        "nonempty_api_prediction_rate": nonempty / total if total else 0.0,
        "per_api": per_api,
        "predictions_file": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate function-level torch API retention.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    rows = load_jsonl(resolve_existing_path(args.candidate_file, label="candidate file"))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = None
    label = "base"
    if args.adapter_dir is not None:
        adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
        label = "adapter"

    model, tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
    summary = evaluate(
        label=label,
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        output_path=output_dir / f"{label}_predictions.jsonl",
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    payload = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "candidate_file": str(args.candidate_file),
        "summary": summary,
    }
    with (output_dir / f"{label}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
