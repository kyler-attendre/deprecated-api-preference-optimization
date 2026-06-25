#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    F = None
    Dataset = object

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plan_c_steering import (  # noqa: E402
    ActivationSteering,
    MODEL_REGISTRY,
    load_jsonl,
    load_vector_file,
    parse_layer_spec,
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


class KTSDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        prompt = row["version_prompt"]
        target = row.get("target") or row.get("reference") or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        target_with_eos = target_ids + eos
        max_prompt_len = max(0, self.max_length - len(target_with_eos))
        prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        input_ids = prompt_ids + target_with_eos
        labels = ([-100] * len(prompt_ids)) + target_with_eos
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "library": row.get("library", ""),
        }


@dataclass
class KTSCollator:
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(item["input_ids"].shape[0] for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        libraries = []
        for item in features:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            libraries.append(item["library"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "libraries": libraries,
        }


class KTSTrainer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        steerer: ActivationSteering,
        train_loader,
        val_loader,
        output_dir: Path,
        learning_rate: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        max_steering_multiplier: float,
        no_steer_prob: float,
        temperature: float,
        log_steps: int,
        resume: bool,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steerer = steerer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steering_multiplier = max_steering_multiplier
        self.no_steer_prob = no_steer_prob
        self.temperature = temperature
        self.log_steps = log_steps
        self.resume = resume
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.start_epoch = 0
        self.global_step = 0

    def maybe_resume(self):
        checkpoint = self.output_dir / "kts_trainer_state.pt"
        if self.resume and checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu")
            self.optimizer.load_state_dict(payload["optimizer"])
            self.start_epoch = int(payload["epoch"])
            self.global_step = int(payload["global_step"])
            adapter_dir = self.output_dir / "resume_adapter"
            if adapter_dir.exists():
                # PEFT weights are loaded by the launcher before training when this path exists.
                pass

    def save_state(self, epoch: int):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "optimizer": self.optimizer.state_dict(),
            },
            self.output_dir / "kts_trainer_state.pt",
        )

    def batch_to_device(self, batch: Dict) -> Dict:
        if not torch.cuda.is_available():
            return batch
        return {
            key: value.cuda() if hasattr(value, "cuda") else value
            for key, value in batch.items()
        }

    def kl_loss_for_batch(self, batch: Dict) -> torch.Tensor:
        libraries = batch.pop("libraries")
        if len(set(libraries)) != 1:
            raise ValueError("KTS training currently requires per-device train batch size 1 or same-library batches.")
        library = libraries[0]
        labels = batch["labels"]
        steering_multiplier = 0.0 if random.random() < self.no_steer_prob else random.random() * self.max_steering_multiplier

        with torch.no_grad():
            with self.steerer.disabled():
                with self.model.disable_adapter():
                    teacher_logits = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits.detach()

        with self.steerer.use(library, multiplier=steering_multiplier):
            student_logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits

        teacher = teacher_logits[:, :-1, :] / self.temperature
        student = student_logits[:, :-1, :] / self.temperature
        target_mask = labels[:, 1:] != -100
        token_kl = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        if target_mask.any():
            return token_kl[target_mask].mean() * (self.temperature ** 2)
        return token_kl.mean() * (self.temperature ** 2)

    def evaluate_loss(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self.batch_to_device(batch)
                loss = self.kl_loss_for_batch(batch)
                losses.append(float(loss.detach().cpu()))
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def train(self):
        self.maybe_resume()
        self.model.train()
        for epoch in range(self.start_epoch, self.num_epochs):
            running = []
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.train_loader):
                batch = self.batch_to_device(batch)
                loss = self.kl_loss_for_batch(batch) / self.gradient_accumulation_steps
                loss.backward()
                running.append(float(loss.detach().cpu()) * self.gradient_accumulation_steps)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.global_step % self.log_steps == 0:
                        avg = sum(running[-self.log_steps :]) / min(len(running), self.log_steps)
                        print(json.dumps({"epoch": epoch + 1, "global_step": self.global_step, "train_kl": avg}))
            val_loss = self.evaluate_loss()
            print(json.dumps({"epoch": epoch + 1, "global_step": self.global_step, "val_kl": val_loss}))
            self.save_state(epoch + 1)


def main():
    parser = argparse.ArgumentParser(description="Train a KL-then-steer LoRA adapter on mixed_sft_v1.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--train-file", type=Path, default=Path("data/mixed_sft_v1/mixed_sft_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/mixed_sft_v1/mixed_sft_val.jsonl"))
    parser.add_argument("--vector-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="2:22")
    parser.add_argument("--coefficient", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steering-multiplier", type=float, default=1.0)
    parser.add_argument("--no-steer-prob", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("Missing dependency: torch. Run this script in the training environment.")

    if args.per_device_train_batch_size != 1 or args.per_device_eval_batch_size != 1:
        raise SystemExit("Use batch size 1 for KTS so each batch has a single library-specific vector.")

    if args.model_key:
        model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
        max_length = args.max_length or int(MODEL_REGISTRY[args.model_key]["max_length"])
    elif args.model_name_or_path:
        model_name_or_path = args.model_name_or_path
        max_length = args.max_length or 384
    else:
        raise SystemExit("Either --model-key or --model-name-or-path is required.")

    try:
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Install transformers, peft, and torch in the training environment.") from exc

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    train_file = resolve_path(args.train_file, label="train file")
    val_file = resolve_path(args.val_file, label="val file")
    vector_file = resolve_path(args.vector_file, label="vector file")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.precision],
    )

    if args.resume and (output_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(output_dir), is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)

    if torch.cuda.is_available():
        model = model.cuda()
    if hasattr(model, "config"):
        model.config.use_cache = False

    vector_payload = load_vector_file(vector_file)
    steerer = ActivationSteering(
        model=model,
        vectors=vector_payload["vectors"],
        layers=parse_layer_spec(args.layers),
        coefficient=args.coefficient,
    ).install()

    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)
    collator = KTSCollator(tokenizer=tokenizer)
    train_loader = DataLoader(KTSDataset(train_rows, tokenizer, max_length), batch_size=1, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(KTSDataset(val_rows, tokenizer, max_length), batch_size=1, shuffle=False, collate_fn=collator)

    manifest = {
        "method": "KL-then-steer LoRA",
        "model_name_or_path": model_name_or_path,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "vector_file": str(vector_file),
        "layers": parse_layer_spec(args.layers),
        "coefficient": args.coefficient,
        "max_length": max_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_steering_multiplier": args.max_steering_multiplier,
        "no_steer_prob": args.no_steer_prob,
        "temperature": args.temperature,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    trainer = KTSTrainer(
        model=model,
        tokenizer=tokenizer,
        steerer=steerer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steering_multiplier=args.max_steering_multiplier,
        no_steer_prob=args.no_steer_prob,
        temperature=args.temperature,
        log_steps=args.logging_steps,
        resume=args.resume,
    )
    trainer.train()
    steerer.remove()


if __name__ == "__main__":
    main()
