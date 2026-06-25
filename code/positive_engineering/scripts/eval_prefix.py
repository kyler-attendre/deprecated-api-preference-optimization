#!/usr/bin/env python3
"""
Evaluate a CER-PT prefix-tuned model on the replacement API test set.

Generation strategy:
  Pass 1 (prefill): PrefixCausalLM.forward() injects prefix KV and returns
    an updated DynamicCache containing [prefix_kv + prompt_kv].
  Pass 2+ (decode): base_model.forward() called directly with the extended
    cache and attention mask (prefix_len + prompt_len + decoded_so_far).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dpo_prefix import PrefixCausalLM  # noqa: E402


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
    for candidate in [Path.cwd() / path, PROJECT_ROOT / path]:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} not found: {path}")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def generate_with_prefix(
    prefix_model: PrefixCausalLM,
    input_ids: torch.Tensor,        # [1, seq_len]
    attention_mask: torch.Tensor,   # [1, seq_len]
    max_new_tokens: int,
    eos_token_id: int,
) -> List[int]:
    """
    Greedy decode with prefix injection.

    Position IDs: prompt tokens use [0, ..., seq_len-1] (matching the override
    in PrefixCausalLM.forward during training). Decode tokens continue at
    [seq_len, seq_len+1, ...] so RoPE is continuous across the seam.

    Attention mask: extended to cover [prefix | prompt | generated] positions.
    """
    device = input_ids.device
    bsz = 1
    prompt_len = input_ids.shape[1]

    # --- Prefill ---
    with torch.no_grad():
        prefill_out = prefix_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    past_kv = prefill_out.past_key_values
    # Extended mask covers [prefix | prompt] positions
    ext_mask = torch.cat(
        [
            torch.ones(bsz, prefix_model.prefix_length, dtype=attention_mask.dtype, device=device),
            attention_mask,
        ],
        dim=1,
    )

    # First generated token from prefill logits
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated_ids: List[int] = [next_token[0, 0].item()]

    if generated_ids[-1] == eos_token_id:
        return generated_ids

    base_model = prefix_model.base_model
    # Next position continues from end of prompt (training used 0..prompt_len-1)
    current_pos = prompt_len

    # --- Decode ---
    for _ in range(max_new_tokens - 1):
        ext_mask = torch.cat(
            [ext_mask, torch.ones(bsz, 1, dtype=ext_mask.dtype, device=device)],
            dim=1,
        )
        position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        with torch.no_grad():
            decode_out = base_model(
                input_ids=next_token,
                attention_mask=ext_mask,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
        past_kv = decode_out.past_key_values
        next_token = decode_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token[0, 0].item()
        generated_ids.append(tok_id)
        current_pos += 1
        if tok_id == eos_token_id:
            break

    return generated_ids


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a CER-PT prefix-tuned model on the test set."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--prefix-params", type=Path, required=True,
                        help="Path to prefix_params.pt saved by train_dpo_prefix.py")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prefix_params_path = resolve_existing_path(args.prefix_params, label="prefix params")
    test_file = resolve_existing_path(args.test_file, label="test file")
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model from {args.model_name_or_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    base_model.eval()

    print(f"Loading prefix from {prefix_params_path}...")
    state = torch.load(prefix_params_path, map_location="cpu")
    prefix_length = state["prefix_length"]
    prefix_model = PrefixCausalLM(base_model, prefix_length=prefix_length)
    prefix_model.prefix_k.data.copy_(
        state["prefix_k"].to(dtype=prefix_model.prefix_k.dtype)
    )
    prefix_model.prefix_v.data.copy_(
        state["prefix_v"].to(dtype=prefix_model.prefix_v.dtype)
    )
    prefix_model.eval()
    print(f"  prefix_length={prefix_length}, n_layers={state['n_layers']}, "
          f"n_kv_heads={state['n_kv_heads']}, d_head={state['d_head']}")

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    deprecated_hits = 0
    replacement_hits = 0
    exact_matches = 0
    total = 0

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    with predictions_path.open("w", encoding="utf-8") as fout:
        for row in _tqdm(rows, desc="Evaluating prefix model", total=len(rows)):
            prompt = row.get("version_prompt", "")
            if not prompt:
                continue

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            generated_ids = generate_with_prefix(
                prefix_model=prefix_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            deprecated_api = row.get("deprecated_api", "")
            replacement_api = row.get("replacement_api", "")
            reference = row.get("reference", "")
            target = row.get("target", "")

            # deprecated_api may be a list; check any element
            if isinstance(deprecated_api, list):
                has_deprecated = any(bool(d) and d in prediction for d in deprecated_api)
                deprecated_api_str = deprecated_api[0] if deprecated_api else ""
            else:
                has_deprecated = bool(deprecated_api) and deprecated_api in prediction
                deprecated_api_str = deprecated_api
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
                        "deprecated_api": deprecated_api_str,
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
        "prefix_params": str(prefix_params_path),
        "prefix_length": prefix_length,
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
