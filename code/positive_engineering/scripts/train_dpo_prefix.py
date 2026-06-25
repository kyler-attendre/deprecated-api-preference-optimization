#!/usr/bin/env python3
"""
CER-PT: DPO + CE anchor with Prefix Tuning instead of LoRA.
Backbone is fully frozen; only learnable KV prefix parameters are trained.

Reference model = same backbone with prefix disabled (context manager mirrors
LoRA's disable_adapter_if_available pattern used in VersionAwareDPOTrainer).
"""
import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from transformers import DynamicCache

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


class PrefixCausalLM(nn.Module):
    """
    Wraps a frozen causal LM and injects learnable KV prefix at every
    attention layer via past_key_values.

    Parameters (prefix_k, prefix_v) live in float32 for optimizer stability;
    they are cast to the backbone dtype during the forward pass.

    disable_adapter() context manager disables prefix injection, turning
    the model into its frozen backbone — used for DPO reference logprobs.
    """

    def __init__(self, base_model, prefix_length: int):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self._prefix_enabled = True

        cfg = base_model.config
        n_layers = cfg.num_hidden_layers
        n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        d_model = cfg.hidden_size
        d_head = d_model // cfg.num_attention_heads

        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        # Float32 parameters for optimizer numerical stability; cast to model
        # dtype inside forward.
        self.prefix_k = nn.Parameter(
            torch.randn(n_layers, prefix_length, n_kv_heads, d_head) * 0.02
        )
        self.prefix_v = nn.Parameter(
            torch.randn(n_layers, prefix_length, n_kv_heads, d_head) * 0.02
        )

        # Freeze all backbone weights
        for p in base_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Properties / compatibility shims
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.base_model.config

    # ------------------------------------------------------------------
    # Prefix toggle (mirrors PeftModel.disable_adapter interface)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def disable_adapter(self):
        old = self._prefix_enabled
        self._prefix_enabled = False
        try:
            yield
        finally:
            self._prefix_enabled = old

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        model_dtype = next(self.base_model.parameters()).dtype

        if not self._prefix_enabled or self.prefix_length == 0:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        # Build a DynamicCache with prefix KV for every layer.
        # DynamicCache.update() expects [bsz, n_kv_heads, seq_len, d_head].
        past_kvs = DynamicCache()
        for l in range(self.n_layers):
            pk = self.prefix_k[l].to(dtype=model_dtype, device=device)  # [m, kv, d]
            pv = self.prefix_v[l].to(dtype=model_dtype, device=device)
            # [m, kv, d] → [1, kv, m, d] → [bsz, kv, m, d]
            pk = pk.unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2).contiguous()
            pv = pv.unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2).contiguous()
            past_kvs.update(pk, pv, layer_idx=l)

        # Extend attention mask so prefix positions are attended to.
        if attention_mask is not None:
            prefix_mask = torch.ones(
                bsz, self.prefix_length, dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Explicit position_ids starting at 0 so RoPE for the actual tokens
        # is not shifted by prefix_length (prefix is treated as virtual context).
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kvs,
            position_ids=position_ids,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Gradient checkpointing support
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        def _require_grad(module, inp, out):
            out.requires_grad_(True)

        self.base_model.get_input_embeddings().register_forward_hook(_require_grad)


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------


class HFVersionAwareDPOTrainer:
    @staticmethod
    def build(
        base_trainer_cls,
        *,
        beta: float,
        logprob_reduction: str,
        api_anchor_weight: float,
        dpo_scope: str,
    ):
        helper = VersionAwareDPOTrainer(
            beta=beta,
            logprob_reduction=logprob_reduction,
            api_anchor_weight=api_anchor_weight,
            dpo_scope=dpo_scope,
        )

        class _Trainer(base_trainer_cls):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss = helper.compute_loss(model, inputs)
                if return_outputs:
                    return loss, {"loss": loss.detach()}
                return loss

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                with torch.no_grad():
                    loss = helper.compute_loss(model, inputs)
                return loss.detach(), None, None

            def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
                save_dir = Path(output_dir or self.args.output_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                # Unwrap model from any DDP/DataParallel wrapper
                raw = self.model
                if hasattr(raw, "module"):
                    raw = raw.module
                torch.save(
                    {
                        "prefix_k": raw.prefix_k.detach().cpu().float(),
                        "prefix_v": raw.prefix_v.detach().cpu().float(),
                        "prefix_length": raw.prefix_length,
                        "n_layers": raw.n_layers,
                        "n_kv_heads": raw.n_kv_heads,
                        "d_head": raw.d_head,
                    },
                    save_dir / "prefix_params.pt",
                )

        return _Trainer


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train CER-PT: DPO + CE anchor using Prefix Tuning (frozen backbone)."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
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
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=64,
        help="Number of learnable KV prefix tokens per layer (m=16/64/256).",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--keep-dropout",
        action="store_true",
        help="Keep dropout. Default disables for deterministic DPO policy/reference comparisons.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt-field", default="version_prompt")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Missing dependencies: transformers") from exc

    train_file = resolve_existing_path(args.train_file, label="train file")
    val_file = resolve_existing_path(args.val_file, label="val file")
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = limit_rows(load_jsonl(train_file), args.max_train_samples)
    val_rows = limit_rows(load_jsonl(val_file), args.max_val_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = PrefixCausalLM(base_model, prefix_length=args.prefix_length)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    disabled_dropout_modules = 0 if args.keep_dropout else disable_dropout_in_model(model)

    n_prefix_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Prefix params: {n_prefix_params:,}  "
        f"(m={args.prefix_length}, layers={model.n_layers}, "
        f"kv_heads={model.n_kv_heads}, d_head={model.d_head})"
    )

    train_dataset = VersionAwareDPODataset(
        train_rows, tokenizer, args.max_length, prompt_field=args.prompt_field
    )
    val_dataset = VersionAwareDPODataset(
        val_rows, tokenizer, args.max_length, prompt_field=args.prompt_field
    )
    if len(train_dataset) == 0:
        raise SystemExit("No DPO pairs could be built from the training file.")
    if len(val_dataset) == 0:
        raise SystemExit("No DPO pairs could be built from the validation file.")

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
                "method": "version_aware_dpo_prefix_tuning",
                "model_name_or_path": args.model_name_or_path,
                "train_file": str(train_file),
                "val_file": str(val_file),
                "raw_train_rows": len(train_rows),
                "raw_val_rows": len(val_rows),
                "train_dpo_pairs": len(train_dataset),
                "val_dpo_pairs": len(val_dataset),
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "beta": args.beta,
                "logprob_reduction": args.logprob_reduction,
                "dpo_scope": args.dpo_scope,
                "api_anchor_weight": args.api_anchor_weight,
                "prefix_length": args.prefix_length,
                "n_prefix_params": n_prefix_params,
                "n_layers": model.n_layers,
                "n_kv_heads": model.n_kv_heads,
                "d_head": model.d_head,
                "prompt_field": args.prompt_field,
                "keep_dropout": args.keep_dropout,
                "disabled_dropout_modules": disabled_dropout_modules,
                "gradient_checkpointing": args.gradient_checkpointing,
                "reference_policy": "same backbone with prefix disabled",
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
