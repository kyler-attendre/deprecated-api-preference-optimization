#!/usr/bin/env python3
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

from src.version_aware_rerank import build_reranker_from_args, ensure_list


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_prediction_rows(label: str, rows: List[Dict], predictions_file: str) -> Dict:
    total = len(rows)
    deprecated_hits = sum(int(row.get("has_deprecated", False)) for row in rows)
    replacement_hits = sum(int(row.get("has_replacement", False)) for row in rows)
    exact_match_hits = sum(int(row.get("exact_match_target", False)) for row in rows)
    nonempty_hits = sum(int(row.get("nonempty_prediction", False)) for row in rows)
    rerank_applied_total = sum(int(row.get("rerank_applied_steps", 0)) for row in rows)
    rerank_changed_total = sum(int(row.get("rerank_changed_steps", 0)) for row in rows)

    return {
        "label": label,
        "samples": total,
        "deprecated_usage_rate": deprecated_hits / total if total else 0.0,
        "replacement_hit_rate": replacement_hits / total if total else 0.0,
        "exact_match_target_rate": exact_match_hits / total if total else 0.0,
        "nonempty_prediction_rate": nonempty_hits / total if total else 0.0,
        "avg_rerank_applied_steps": rerank_applied_total / total if total else 0.0,
        "avg_rerank_changed_steps": rerank_changed_total / total if total else 0.0,
        "predictions_file": predictions_file,
    }


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


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def alias_forms(api_name: str) -> List[str]:
    aliases = {api_name.strip()}
    api_name = api_name.strip()
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


def build_model(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def greedy_generate(
    *,
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    reranker=None,
    deprecated_apis: Optional[List[str]] = None,
    replacement_apis: Optional[List[str]] = None,
):
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    generated_ids: List[int] = []
    generated_text = ""
    rerank_applied_steps = 0
    rerank_changed_steps = 0
    step_traces: List[Dict] = []

    for step_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :][0]

        if reranker is None:
            next_token_id = int(torch.argmax(next_token_logits).item())
            next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
            step_traces.append(
                {
                    "step": step_idx,
                    "mode": "base",
                    "chosen_token_id": next_token_id,
                    "chosen_token_text": next_token_text,
                }
            )
        else:
            decision = reranker.rerank_next_token(
                logits=next_token_logits,
                tokenizer=tokenizer,
                generated_text=generated_text,
                deprecated_apis=deprecated_apis or [],
                replacement_apis=replacement_apis or [],
            )
            next_token_id = decision.token_id
            next_token_text = decision.token_text
            rerank_applied_steps += int(decision.applied)
            rerank_changed_steps += int(decision.changed)
            step_traces.append(
                {
                    "step": step_idx,
                    "mode": "rerank",
                    "chosen_token_id": decision.token_id,
                    "chosen_token_text": decision.token_text,
                    "base_token_text": decision.base_token_text,
                    "fragment": decision.fragment,
                    "match_type": decision.match_type,
                    "rerank_applied": decision.applied,
                    "rerank_changed": decision.changed,
                    "top_candidates": [
                        {
                            "token_id": c.token_id,
                            "token_text": c.token_text,
                            "base_score": c.base_score,
                            "reranked_score": c.reranked_score,
                            "fragment": c.fragment,
                            "adjustment": c.adjustment,
                            "match_type": c.match_type,
                        }
                        for c in decision.top_candidates
                    ],
                }
            )

        generated_ids.append(next_token_id)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        next_tensor = torch.tensor([[next_token_id]], dtype=input_ids.dtype, device=input_ids.device)
        next_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

    return {
        "prediction": generated_text.strip(),
        "rerank_applied_steps": rerank_applied_steps,
        "rerank_changed_steps": rerank_changed_steps,
        "step_traces": step_traces,
    }


def evaluate_model(
    *,
    label: str,
    model,
    tokenizer,
    rows: List[Dict],
    output_path: Path,
    max_length: int,
    max_new_tokens: int,
    reranker=None,
) -> Dict:
    deprecated_hits = 0
    replacement_hits = 0
    exact_match_hits = 0
    nonempty_hits = 0
    rerank_applied_total = 0
    rerank_changed_total = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"Evaluating {label}", total=len(rows)):
            prompt = row["version_prompt"]
            deprecated_apis = ensure_list(row.get("deprecated_api"))
            replacement_apis = ensure_list(row.get("replacement_api"))
            target = row.get("target") or row.get("reference") or ""

            generation = greedy_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                reranker=reranker,
                deprecated_apis=deprecated_apis,
                replacement_apis=replacement_apis,
            )

            prediction = generation["prediction"]
            has_deprecated = any_api_mentioned(prediction, deprecated_apis)
            has_replacement = any_api_mentioned(prediction, replacement_apis)
            exact_match = normalize_text(prediction) == normalize_text(target)
            nonempty = bool(prediction.strip())

            deprecated_hits += int(has_deprecated)
            replacement_hits += int(has_replacement)
            exact_match_hits += int(exact_match)
            nonempty_hits += int(nonempty)
            rerank_applied_total += generation["rerank_applied_steps"]
            rerank_changed_total += generation["rerank_changed_steps"]

            fout.write(
                json.dumps(
                    {
                        "run_label": label,
                        "id": row.get("id"),
                        "model": row.get("model"),
                        "library": row.get("library"),
                        "category": row.get("category"),
                        "sample_type": row.get("sample_type"),
                        "task_family": row.get("task_family"),
                        "deprecated_api": deprecated_apis,
                        "replacement_api": replacement_apis,
                        "target": target,
                        "prediction": prediction,
                        "has_deprecated": has_deprecated,
                        "has_replacement": has_replacement,
                        "exact_match_target": exact_match,
                        "nonempty_prediction": nonempty,
                        "rerank_applied_steps": generation["rerank_applied_steps"],
                        "rerank_changed_steps": generation["rerank_changed_steps"],
                        "step_traces": generation["step_traces"] if reranker is not None else [],
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
        "avg_rerank_applied_steps": rerank_applied_total / total if total else 0.0,
        "avg_rerank_changed_steps": rerank_changed_total / total if total else 0.0,
        "predictions_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare prompt-only decoding and version-aware reranking on rerank_eval data"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed_clean/rerank_eval_test.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument(
        "--run-mode",
        type=str,
        default="both",
        choices=["both", "prompt_only", "rerank_only"],
        help="Run both phases, only prompt-only decoding, or only reranking",
    )
    parser.add_argument(
        "--prompt-only-predictions-file",
        type=Path,
        default=None,
        help="Existing prompt_only_predictions.jsonl used to build comparison_summary when running rerank_only",
    )
    args = parser.parse_args()

    test_file = resolve_existing_path(args.test_file, label="test file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    import torch

    model, tokenizer = build_model(args.model_name_or_path)
    base_summary = None
    rerank_summary = None

    if args.run_mode in {"both", "prompt_only"}:
        base_summary = evaluate_model(
            label="prompt_only",
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_path=output_dir / "prompt_only_predictions.jsonl",
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            reranker=None,
        )
        with (output_dir / "prompt_only_summary.json").open("w", encoding="utf-8") as f:
            json.dump(base_summary, f, indent=2, ensure_ascii=False)

    if args.run_mode in {"both", "rerank_only"}:
        reranker = build_reranker_from_args(args)
        rerank_summary = evaluate_model(
            label=f"prompt_{args.rerank_mode}_rerank",
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_path=output_dir / f"prompt_{args.rerank_mode}_rerank_predictions.jsonl",
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            reranker=reranker,
        )
        with (output_dir / "rerank_summary.json").open("w", encoding="utf-8") as f:
            json.dump(rerank_summary, f, indent=2, ensure_ascii=False)

    if base_summary is None and args.prompt_only_predictions_file is not None:
        prompt_only_predictions_file = resolve_existing_path(
            args.prompt_only_predictions_file, label="prompt-only predictions file"
        )
        prompt_only_rows = load_jsonl(prompt_only_predictions_file)
        base_summary = summarize_prediction_rows(
            "prompt_only",
            prompt_only_rows,
            str(prompt_only_predictions_file),
        )

    if base_summary is not None and rerank_summary is not None:
        comparison = {
            "model_name_or_path": args.model_name_or_path,
            "test_file": str(test_file),
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "alpha": args.alpha,
            "beta": args.beta,
            "top_k": args.top_k,
            "rerank_mode": args.rerank_mode,
            "run_mode": args.run_mode,
            "prompt_only": base_summary,
            "rerank": rerank_summary,
            "delta": {
                "deprecated_usage_rate": rerank_summary["deprecated_usage_rate"] - base_summary["deprecated_usage_rate"],
                "replacement_hit_rate": rerank_summary["replacement_hit_rate"] - base_summary["replacement_hit_rate"],
                "exact_match_target_rate": rerank_summary["exact_match_target_rate"] - base_summary["exact_match_target_rate"],
                "nonempty_prediction_rate": rerank_summary["nonempty_prediction_rate"] - base_summary["nonempty_prediction_rate"],
                "avg_rerank_applied_steps": rerank_summary["avg_rerank_applied_steps"] - base_summary["avg_rerank_applied_steps"],
                "avg_rerank_changed_steps": rerank_summary["avg_rerank_changed_steps"] - base_summary["avg_rerank_changed_steps"],
            },
        }

        with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        return

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "run_mode": args.run_mode,
        "prompt_only": base_summary,
        "rerank": rerank_summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
