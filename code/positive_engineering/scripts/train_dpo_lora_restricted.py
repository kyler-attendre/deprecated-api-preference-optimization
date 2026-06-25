#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_lora import limit_rows, load_jsonl, resolve_existing_path  # noqa: E402
from src.dpo_training import (  # noqa: E402
    DPOCollator,
    VersionAwareDPODataset,
    VersionAwareDPOTrainer,
    disable_dropout_in_model,
)
from scripts.train_dpo_lora import (  # noqa: E402
    HFVersionAwareDPOTrainer,
    build_lora_config_kwargs,
    parse_int_list,
)


def main():
    parser = argparse.ArgumentParser(description="Train a restricted-layer version-aware DPO LoRA adapter.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers-to-transform", nargs="+", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--eval-strategy", type=str, default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logprob-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--dpo-scope", choices=["full", "api_span"], default="full")
    parser.add_argument("--api-anchor-weight", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "o_proj"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-dropout", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Please install transformers and peft.") from exc

    train_file = resolve_existing_path(args.train_file, label="train file")
    val_file = resolve_existing_path(args.val_file, label="val file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_to_transform = parse_int_list(args.layers_to_transform)
    train_rows = limit_rows(load_jsonl(train_file), args.max_train_samples)
    val_rows = limit_rows(load_jsonl(val_file), args.max_val_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        **build_lora_config_kwargs(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            layers_to_transform=layers_to_transform,
            task_type=TaskType.CAUSAL_LM,
        )
    )
    model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    disabled_dropout_modules = 0 if args.keep_dropout else disable_dropout_in_model(model)

    train_dataset = VersionAwareDPODataset(train_rows, tokenizer, args.max_length)
    val_dataset = VersionAwareDPODataset(val_rows, tokenizer, args.max_length)
    collator = DPOCollator(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "version_aware_dpo_lora_restricted",
                "model_name_or_path": args.model_name_or_path,
                "train_file": str(train_file),
                "val_file": str(val_file),
                "layers_to_transform": layers_to_transform,
                "target_modules": args.target_modules,
                "raw_train_rows": len(train_rows),
                "raw_val_rows": len(val_rows),
                "train_dpo_pairs": len(train_dataset),
                "val_dpo_pairs": len(val_dataset),
                "dpo_scope": args.dpo_scope,
                "api_anchor_weight": args.api_anchor_weight,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "disabled_dropout_modules": disabled_dropout_modules,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    trainer_cls = HFVersionAwareDPOTrainer.build(
        Trainer,
        beta=args.beta,
        logprob_reduction=args.logprob_reduction,
        api_anchor_weight=args.api_anchor_weight,
        dpo_scope=args.dpo_scope,
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
