#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plan_c_steering import (  # noqa: E402
    ActivationSteering,
    MODEL_REGISTRY,
    alias_forms,
    ensure_list,
    load_jsonl,
    load_vector_file,
    parse_layer_spec,
)


def resolve_path(path: Path, *, label: str) -> Path:
    if path.is_absolute():
        candidate = path
    else:
        candidate = (PROJECT_ROOT / path).resolve()
        if not candidate.exists():
            candidate = (Path.cwd() / path).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return candidate


def any_api_mentioned(prediction: str, api_names: Iterable[str]) -> bool:
    for api_name in api_names:
        for alias in alias_forms(api_name):
            if alias and alias in prediction:
                return True
    return False


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_model(model_name_or_path: str, precision: str, adapter_dir: Optional[Path] = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[precision],
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def generate_prediction(
    *,
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    if torch.cuda.is_available():
        encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_rows(
    *,
    label: str,
    rows: List[Dict],
    model,
    tokenizer,
    output_path: Path,
    max_length: int,
    max_new_tokens: int,
    steerer: Optional[ActivationSteering] = None,
    fail_on_missing_vector: bool = True,
) -> Dict:
    deprecated_hits = 0
    replacement_hits = 0
    exact_hits = 0
    nonempty_hits = 0
    missing_vectors = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"Evaluating {label}", total=len(rows)):
            library = row.get("library")
            if steerer is not None and library not in steerer.vectors:
                missing_vectors += 1
                if fail_on_missing_vector:
                    raise ValueError(f"No steering vector for library {library!r}")

            if steerer is None:
                prediction = generate_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=row["version_prompt"],
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )
            else:
                with steerer.use(library):
                    prediction = generate_prediction(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=row["version_prompt"],
                        max_length=max_length,
                        max_new_tokens=max_new_tokens,
                    )

            deprecated_apis = ensure_list(row.get("deprecated_api"))
            replacement_apis = ensure_list(row.get("replacement_api"))
            target = row.get("target") or row.get("reference") or ""

            has_deprecated = any_api_mentioned(prediction, deprecated_apis)
            has_replacement = any_api_mentioned(prediction, replacement_apis)
            exact_match = normalize_text(prediction) == normalize_text(target)
            nonempty = bool(prediction.strip())

            deprecated_hits += int(has_deprecated)
            replacement_hits += int(has_replacement)
            exact_hits += int(exact_match)
            nonempty_hits += int(nonempty)

            fout.write(
                json.dumps(
                    {
                        "run_label": label,
                        "id": row.get("id"),
                        "model": row.get("model"),
                        "library": library,
                        "category": row.get("category"),
                        "sample_type": row.get("sample_type"),
                        "mixed_source_bucket": row.get("mixed_source_bucket"),
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
        "exact_match_target_rate": exact_hits / total if total else 0.0,
        "nonempty_prediction_rate": nonempty_hits / total if total else 0.0,
        "missing_vector_rows": missing_vectors,
        "predictions_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct steering or KTS+steering on mixed_sft_v1.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None, help="KTS adapter dir. Omit for direct base steering.")
    parser.add_argument("--test-file", type=Path, default=Path("data/mixed_sft_v1/mixed_sft_test.jsonl"))
    parser.add_argument("--vector-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="2:22")
    parser.add_argument("--coefficient", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow-missing-vector", action="store_true")
    args = parser.parse_args()

    if args.model_key:
        model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
        max_length = args.max_length or int(MODEL_REGISTRY[args.model_key]["max_length"])
        model_label = args.model_key
    elif args.model_name_or_path:
        model_name_or_path = args.model_name_or_path
        max_length = args.max_length or 384
        model_label = Path(model_name_or_path).name
    else:
        raise SystemExit("Either --model-key or --model-name-or-path is required.")

    test_file = resolve_path(args.test_file, label="test file")
    vector_file = resolve_path(args.vector_file, label="vector file")
    adapter_dir = resolve_path(args.adapter_dir, label="adapter dir") if args.adapter_dir else None
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model(model_name_or_path, args.precision, adapter_dir)
    vector_payload = load_vector_file(vector_file)
    layers = parse_layer_spec(args.layers)
    steerer = ActivationSteering(
        model=model,
        vectors=vector_payload["vectors"],
        layers=layers,
        coefficient=args.coefficient,
    ).install()

    base_label = "kts_no_steering" if adapter_dir else "base"
    steered_label = "kts_steering" if adapter_dir else "direct_steering"

    base_summary = evaluate_rows(
        label=base_label,
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        output_path=output_dir / f"{base_label}_predictions.jsonl",
        max_length=max_length,
        max_new_tokens=args.max_new_tokens,
        steerer=None,
    )
    steered_summary = evaluate_rows(
        label=steered_label,
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        output_path=output_dir / f"{steered_label}_predictions.jsonl",
        max_length=max_length,
        max_new_tokens=args.max_new_tokens,
        steerer=steerer,
        fail_on_missing_vector=not args.allow_missing_vector,
    )
    steerer.remove()

    comparison = {
        "model_label": model_label,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "test_file": str(test_file),
        "vector_file": str(vector_file),
        "layers": layers,
        "coefficient": args.coefficient,
        "max_length": max_length,
        "max_new_tokens": args.max_new_tokens,
        base_label: base_summary,
        steered_label: steered_summary,
        "delta": {
            "deprecated_usage_rate": steered_summary["deprecated_usage_rate"] - base_summary["deprecated_usage_rate"],
            "replacement_hit_rate": steered_summary["replacement_hit_rate"] - base_summary["replacement_hit_rate"],
            "exact_match_target_rate": steered_summary["exact_match_target_rate"] - base_summary["exact_match_target_rate"],
            "nonempty_prediction_rate": steered_summary["nonempty_prediction_rate"] - base_summary["nonempty_prediction_rate"],
        },
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
