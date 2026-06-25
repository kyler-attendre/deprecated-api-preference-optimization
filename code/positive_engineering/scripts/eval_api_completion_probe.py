#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
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


def evaluate_rows(
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
    intrusion = 0
    nonempty = 0
    predicted_counter: Counter[str] = Counter()
    intrusion_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = row["prompt"]
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
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_api = extract_predicted_api(prediction)
            target_api = row["target_api"]
            trained_prefixes = set(row.get("trained_prefixes") or [])

            is_exact = predicted_api == target_api
            is_intrusion = bool(predicted_api) and predicted_api in trained_prefixes and target_api not in trained_prefixes

            exact += int(is_exact)
            intrusion += int(is_intrusion)
            nonempty += int(bool(predicted_api))
            if predicted_api:
                predicted_counter[predicted_api] += 1
            if is_intrusion:
                intrusion_counter[predicted_api] += 1

            payload = {
                "id": row["id"],
                "run_label": label,
                "target_api": target_api,
                "predicted_text": prediction,
                "predicted_api": predicted_api,
                "exact_api_match": is_exact,
                "trained_api_intrusion": is_intrusion,
                "source_file": row["source_file"],
                "prompt": prompt,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    total = len(rows)
    return {
        "label": label,
        "samples": total,
        "exact_api_match_rate": exact / total if total else 0.0,
        "trained_api_intrusion_rate": intrusion / total if total else 0.0,
        "nonempty_api_prediction_rate": nonempty / total if total else 0.0,
        "top_predictions": predicted_counter.most_common(20),
        "top_intrusions": intrusion_counter.most_common(20),
        "predictions_file": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate base vs anchored DPO on a normal-API completion probe."
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--probe-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
    probe_file = resolve_existing_path(args.probe_file, label="probe file")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(probe_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    base_model, base_tokenizer = build_model(model_name_or_path)
    base_summary = evaluate_rows(
        label="base",
        model=base_model,
        tokenizer=base_tokenizer,
        rows=rows,
        output_path=output_dir / "base_predictions.jsonl",
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_model, lora_tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
    lora_summary = evaluate_rows(
        label="anchored_dpo",
        model=lora_model,
        tokenizer=lora_tokenizer,
        rows=rows,
        output_path=output_dir / "anchored_dpo_predictions.jsonl",
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir),
        "probe_file": str(probe_file),
        "base": base_summary,
        "anchored_dpo": lora_summary,
        "delta": {
            "exact_api_match_rate": lora_summary["exact_api_match_rate"] - base_summary["exact_api_match_rate"],
            "trained_api_intrusion_rate": lora_summary["trained_api_intrusion_rate"] - base_summary["trained_api_intrusion_rate"],
            "nonempty_api_prediction_rate": lora_summary["nonempty_api_prediction_rate"] - base_summary["nonempty_api_prediction_rate"],
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
