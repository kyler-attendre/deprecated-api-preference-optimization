#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

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
    MODEL_REGISTRY,
    iter_official_mbpp_test_rows,
    read_humaneval_jsonl,
    write_jsonl,
)


HUMANEVAL_STOP = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
MBPP_STOP = ["[DONE]", "\n[DONE]", "\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n```"]


def resolve_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    if path.is_absolute():
        candidate = path
    else:
        candidate = (PROJECT_ROOT / path).resolve()
        if not candidate.exists():
            candidate = (Path.cwd() / path).resolve()
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return candidate


def trim_at_stop(text: str, stop_words: Iterable[str]) -> str:
    cut = len(text)
    for stop in stop_words:
        idx = text.find(stop)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].rstrip()


def load_humaneval_rows(problem_file: Path = None) -> List[Dict]:
    if problem_file is None:
        from human_eval.data import read_problems

        problems = read_problems()
    else:
        problems = read_humaneval_jsonl(problem_file)
    return [
        {
            "task_id": task_id,
            "prompt": problem["prompt"],
            "tests": problem.get("test", ""),
            "entry_point": problem.get("entry_point", ""),
        }
        for task_id, problem in sorted(problems.items())
    ]


def load_model(model_name_or_path: str, precision: str, adapter_dir: Path = None):
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
        model = model.merge_and_unload()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def generate_completion(
    *,
    model,
    tokenizer,
    prompt: str,
    max_length_generation: int,
    max_new_tokens: int,
    stop_words: List[str],
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    import torch

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length_generation)
    if torch.cuda.is_available():
        encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][encoded["input_ids"].shape[1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return trim_at_stop(completion, stop_words)


def main():
    parser = argparse.ArgumentParser(
        description="Generate official-format HumanEval or Google MBPP samples for pass@1 evaluation."
    )
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--humaneval-problem-file", type=Path, default=None)
    parser.add_argument("--mbpp-file", type=Path, default=None, help="Official google-research sanitized-mbpp.json")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-length-generation", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = parser.parse_args()

    if args.model_key:
        model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    elif args.model_name_or_path:
        model_name_or_path = args.model_name_or_path
    else:
        raise SystemExit("Either --model-key or --model-name-or-path is required.")

    adapter_dir = resolve_path(args.adapter_dir, label="adapter dir") if args.adapter_dir else None
    output_jsonl = args.output_jsonl if args.output_jsonl.is_absolute() else (PROJECT_ROOT / args.output_jsonl)
    output_jsonl = output_jsonl.resolve()

    if args.benchmark == "humaneval":
        problem_file = resolve_path(args.humaneval_problem_file, label="HumanEval problem file") if args.humaneval_problem_file else None
        rows = load_humaneval_rows(problem_file)
        stop_words = HUMANEVAL_STOP
    else:
        if args.mbpp_file is None:
            raise SystemExit("--mbpp-file is required for MBPP because the official Google file is not bundled here.")
        rows = list(iter_official_mbpp_test_rows(resolve_path(args.mbpp_file, label="MBPP file")))
        stop_words = MBPP_STOP

    if args.limit > 0:
        rows = rows[: args.limit]

    model, tokenizer = load_model(model_name_or_path, args.precision, adapter_dir)
    samples = []
    for row in tqdm(rows, desc=f"Generating {args.benchmark}", total=len(rows)):
        for sample_idx in range(args.n_samples):
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                max_length_generation=args.max_length_generation,
                max_new_tokens=args.max_new_tokens,
                stop_words=stop_words,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + sample_idx,
            )
            payload = {
                "task_id": row["task_id"],
                "completion": completion,
            }
            if args.benchmark == "mbpp":
                payload["tests"] = row["tests"]
            samples.append(payload)

    write_jsonl(output_jsonl, samples)
    manifest = {
        "benchmark": args.benchmark,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "output_jsonl": str(output_jsonl),
        "tasks": len(rows),
        "n_samples": args.n_samples,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "max_length_generation": args.max_length_generation,
        "max_new_tokens": args.max_new_tokens,
    }
    with output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
