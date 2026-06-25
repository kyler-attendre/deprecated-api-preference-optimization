#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List

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
    build_contrast_pairs,
    get_decoder_layers,
    load_jsonl,
    parse_layer_spec,
    safe_model_label,
)


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


def load_model_and_tokenizer(model_name_or_path: str, precision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[precision],
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def final_token_hidden_states(model, tokenizer, text: str, max_length: int, layers: List[int]) -> Dict[int, torch.Tensor]:
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    if torch.cuda.is_available():
        encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    # hidden_states[0] is embedding output; decoder layer i output is hidden_states[i + 1].
    return {
        layer: outputs.hidden_states[layer + 1][0, -1].detach().float().cpu()
        for layer in layers
        if layer + 1 < len(outputs.hidden_states)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute per-library Plan C steering vectors from mixed_sft_v1.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--train-file", type=Path, default=Path("data/mixed_sft_v1/mixed_sft_train.jsonl"))
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="2:22")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-pairs-per-library", type=int, default=0, help="0 means use all contrast pairs")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = parser.parse_args()

    if args.model_key:
        model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
        max_length = args.max_length or int(MODEL_REGISTRY[args.model_key]["max_length"])
    elif args.model_name_or_path:
        model_name_or_path = args.model_name_or_path
        max_length = args.max_length or 384
    else:
        raise SystemExit("Either --model-key or --model-name-or-path is required.")

    train_file = resolve_path(args.train_file, label="train file")
    output_file = args.output_file if args.output_file.is_absolute() else (PROJECT_ROOT / args.output_file)
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    layers = parse_layer_spec(args.layers)
    rows = load_jsonl(train_file)
    pairs = build_contrast_pairs(rows)
    by_library = defaultdict(list)
    for pair in pairs:
        by_library[pair.library].append(pair)

    if args.max_pairs_per_library > 0:
        for library in list(by_library):
            by_library[library] = by_library[library][: args.max_pairs_per_library]

    if not by_library:
        raise SystemExit("No contrast pairs could be built from the training file.")

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, args.precision)
    num_layers = len(get_decoder_layers(model))
    active_layers = [layer for layer in layers if 0 <= layer < num_layers]
    if not active_layers:
        raise SystemExit(f"No requested layers exist in model with {num_layers} layers.")

    sums = {library: {layer: None for layer in active_layers} for library in by_library}
    counts = Counter()
    skipped = Counter()

    for library, library_pairs in sorted(by_library.items()):
        for pair in tqdm(library_pairs, desc=f"Vectors {library}", total=len(library_pairs)):
            try:
                positive = final_token_hidden_states(model, tokenizer, pair.positive_text, max_length, active_layers)
                negative = final_token_hidden_states(model, tokenizer, pair.negative_text, max_length, active_layers)
            except RuntimeError as exc:
                skipped[library] += 1
                print(f"Skipping pair {pair.row_id} for {library}: {exc}", file=sys.stderr)
                continue
            for layer in active_layers:
                diff = positive[layer] - negative[layer]
                sums[library][layer] = diff if sums[library][layer] is None else sums[library][layer] + diff
            counts[library] += 1

    vectors = {}
    for library, layer_sums in sums.items():
        if counts[library] == 0:
            continue
        vectors[library] = {str(layer): tensor / counts[library] for layer, tensor in layer_sums.items() if tensor is not None}

    payload = {
        "model_name_or_path": model_name_or_path,
        "model_label": args.model_key or safe_model_label(model_name_or_path),
        "train_file": str(train_file),
        "layers": active_layers,
        "max_length": max_length,
        "vector_definition": "mean(final-token hidden state of version-consistent completion - synthetic deprecated completion)",
        "counts_by_library": dict(counts),
        "skipped_by_library": dict(skipped),
        "vectors": vectors,
    }
    import torch

    torch.save(payload, output_file)

    manifest = dict(payload)
    manifest["vectors"] = {library: list(layer_map.keys()) for library, layer_map in vectors.items()}
    with output_file.with_suffix(output_file.suffix + ".manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
