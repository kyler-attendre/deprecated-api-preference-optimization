#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MECH_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MECH_ROOT.parent
SRC_ROOT = MECH_ROOT
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.lens_analysis import (  # noqa: E402
    MODEL_REGISTRY,
    LowRankTunedLens,
    build_focus_examples,
    build_model,
    build_training_batch,
    get_decoder_layers,
    get_final_norm,
    get_hidden_size,
    get_output_projection,
    load_many_jsonl,
    resolve_existing_path,
    save_tuned_lens,
    tuned_lens_loss,
    write_json,
)


DEFAULT_TRAIN_FILES = [
    "05_positive_engineering/data/processed_clean/repair_sft_train.jsonl",
    "05_positive_engineering/data/processed_clean/consistency_sft_train.jsonl",
]
DEFAULT_VAL_FILES = [
    "05_positive_engineering/data/processed_clean/repair_sft_val.jsonl",
    "05_positive_engineering/data/processed_clean/consistency_sft_val.jsonl",
]


def resolve_many(paths: List[str], label: str) -> List[Path]:
    return [resolve_existing_path(Path(path), label=label) for path in paths]


def chunk_indices(size: int, batch_size: int):
    indices = list(range(size))
    random.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def evaluate_loss(*, model, tuned_lens, final_norm, output_projection, tokenizer, examples, batch_size, max_length):
    tuned_lens.eval()
    losses = []
    with torch.no_grad():
        for indices in chunk_indices(len(examples), batch_size):
            batch = build_training_batch(
                tokenizer=tokenizer,
                examples=examples,
                indices=indices,
                max_length=max_length,
            )
            loss = tuned_lens_loss(
                model=model,
                tuned_lens=tuned_lens,
                final_norm=final_norm,
                output_projection=output_projection,
                batch=batch,
            )
            losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train a low-rank tuned lens for API-decision positions.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--prompt-field", type=str, default="version_prompt", choices=["version_prompt", "probing_input"])
    parser.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--val-files", nargs="+", default=DEFAULT_VAL_FILES)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_files = resolve_many(args.train_files, "train file")
    val_files = resolve_many(args.val_files, "val file")
    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir") if args.adapter_dir else None

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    model, tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
    hidden_size = get_hidden_size(model)
    num_layers = len(get_decoder_layers(model))
    final_norm = get_final_norm(model)
    output_projection = get_output_projection(model)
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]

    train_examples = build_focus_examples(load_many_jsonl(train_files), prompt_field=args.prompt_field)
    val_examples = build_focus_examples(load_many_jsonl(val_files), prompt_field=args.prompt_field)

    tuned_lens = LowRankTunedLens(num_layers=num_layers, hidden_size=hidden_size, rank=args.rank)
    tuned_lens.train()
    if torch.cuda.is_available():
        tuned_lens = tuned_lens.cuda()

    optimizer = torch.optim.AdamW(tuned_lens.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_val = None
    best_state = None

    for epoch in range(args.epochs):
        tuned_lens.train()
        epoch_losses = []
        for indices in chunk_indices(len(train_examples), args.batch_size):
            batch = build_training_batch(
                tokenizer=tokenizer,
                examples=train_examples,
                indices=indices,
                max_length=max_length,
            )
            loss = tuned_lens_loss(
                model=model,
                tuned_lens=tuned_lens,
                final_norm=final_norm,
                output_projection=output_projection,
                batch=batch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        val_loss = evaluate_loss(
            model=model,
            tuned_lens=tuned_lens,
            final_norm=final_norm,
            output_projection=output_projection,
            tokenizer=tokenizer,
            examples=val_examples,
            batch_size=args.batch_size,
            max_length=max_length,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in tuned_lens.state_dict().items()}

    if best_state is not None:
        tuned_lens.load_state_dict(best_state)

    metadata = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "prompt_field": args.prompt_field,
        "train_files": [str(path) for path in train_files],
        "val_files": [str(path) for path in val_files],
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": max_length,
        "seed": args.seed,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "history": history,
    }
    output_file = args.output_file if args.output_file.is_absolute() else (Path.cwd() / args.output_file)
    output_file = output_file.resolve()
    save_tuned_lens(output_file, tuned_lens, metadata)
    write_json(output_file.with_suffix(".json"), metadata)
    print(metadata)


if __name__ == "__main__":
    main()
