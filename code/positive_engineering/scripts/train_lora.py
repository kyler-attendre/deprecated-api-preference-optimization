#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

import torch
from torch.utils.data import Dataset

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


def limit_rows(rows: List[Dict], max_samples: int) -> List[Dict]:
    if max_samples <= 0:
        return rows
    return rows[:max_samples]


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


def select_prompt(row: Dict, prompt_field: str) -> str:
    if prompt_field not in row:
        raise KeyError(f"Prompt field {prompt_field!r} not found in row")
    return row[prompt_field]


class VersionAwareSFTDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int, prompt_field: str = "version_prompt"):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_field = prompt_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = select_prompt(row, self.prompt_field)
        target = row["target"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        target_with_eos = target_ids + eos
        max_prompt_len = max(0, self.max_length - len(target_with_eos))
        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

        input_ids = prompt_ids + target_with_eos
        labels = ([-100] * len(prompt_ids)) + target_with_eos
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class SFTCollator:
    tokenizer: object

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(item["input_ids"].shape[0] for item in features)

        input_ids = []
        labels = []
        attention_mask = []
        for item in features:
            cur_len = item["input_ids"].shape[0]
            pad_len = max_len - cur_len
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            labels.append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


def main():
    parser = argparse.ArgumentParser(description="Train a version-aware LoRA model")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="version_prompt",
        choices=["version_prompt", "probing_input"],
        help="Input field used as the SFT prompt. Default keeps version-aware training.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all training samples")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0 means use all validation samples")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--eval-strategy", type=str, default="epoch")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Please install transformers and peft in the training environment."
        ) from exc

    train_file = resolve_existing_path(args.train_file, label="train file")
    val_file = resolve_existing_path(args.val_file, label="val file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()

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

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    train_dataset = VersionAwareSFTDataset(train_rows, tokenizer, args.max_length, prompt_field=args.prompt_field)
    val_dataset = VersionAwareSFTDataset(val_rows, tokenizer, args.max_length, prompt_field=args.prompt_field)
    collator = SFTCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": args.model_name_or_path,
                "train_file": str(train_file),
                "val_file": str(val_file),
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "prompt_field": args.prompt_field,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "target_modules": args.target_modules,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    trainer = Trainer(
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
