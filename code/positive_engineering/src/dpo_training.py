import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.plan_c_steering import (
    deprecated_alias_matching_replacement,
    ensure_list,
    first_alias_hit,
)


@dataclass(frozen=True)
class DPOPair:
    row_id: str
    library: str
    prompt: str
    chosen: str
    rejected: str
    replacement_form: str
    deprecated_form: str


def build_dpo_pair(row: Dict, prompt_field: str = "version_prompt") -> Optional[DPOPair]:
    prompt = str(row.get(prompt_field) or "")
    chosen = str(row.get("target") or row.get("reference") or "").strip()
    replacement_api = str(row.get("replacement_api") or "").strip()
    deprecated_apis = ensure_list(row.get("deprecated_api"))

    if not prompt or not chosen or not replacement_api or not deprecated_apis:
        return None

    replacement_form = first_alias_hit(chosen, replacement_api)
    if not replacement_form:
        return None

    deprecated_form = deprecated_alias_matching_replacement(
        replacement_api=replacement_api,
        replacement_form=replacement_form,
        deprecated_api=deprecated_apis[0],
    )
    rejected = chosen.replace(replacement_form, deprecated_form, 1)
    if rejected == chosen:
        return None

    return DPOPair(
        row_id=str(row.get("id") or ""),
        library=str(row.get("library") or "unknown"),
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        replacement_form=replacement_form,
        deprecated_form=deprecated_form,
    )


def build_dpo_pairs(rows: Iterable[Dict], prompt_field: str = "version_prompt") -> List[DPOPair]:
    pairs: List[DPOPair] = []
    for row in rows:
        pair = build_dpo_pair(row, prompt_field=prompt_field)
        if pair is not None:
            pairs.append(pair)
    return pairs


def _encode_prompt_completion(tokenizer, prompt: str, completion: str, max_length: int) -> Dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    completion_ids = completion_ids + eos

    max_prompt_len = max(0, max_length - len(completion_ids))
    prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + completion_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def encode_api_anchor(
    tokenizer,
    prompt: str,
    completion: str,
    api_form: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    api_start = completion.find(api_form)
    if api_start < 0:
        raise ValueError(f"API form not found in completion: {api_form}")

    prefix = completion[:api_start]
    context_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context_ids += tokenizer(prefix, add_special_tokens=False)["input_ids"]
    api_ids = tokenizer(api_form, add_special_tokens=False)["input_ids"]
    if not api_ids:
        raise ValueError(f"API form encoded to no tokens: {api_form}")

    max_context_len = max(0, max_length - len(api_ids))
    context_ids = context_ids[-max_context_len:] if max_context_len > 0 else []

    input_ids = context_ids + api_ids
    labels = ([-100] * len(context_ids)) + api_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class VersionAwareDPODataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int, prompt_field: str = "version_prompt"):
        self.pairs = build_dpo_pairs(rows, prompt_field=prompt_field)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[index]
        chosen = _encode_prompt_completion(
            self.tokenizer,
            pair.prompt,
            pair.chosen,
            self.max_length,
        )
        rejected = _encode_prompt_completion(
            self.tokenizer,
            pair.prompt,
            pair.rejected,
            self.max_length,
        )
        anchor = encode_api_anchor(
            self.tokenizer,
            pair.prompt,
            pair.chosen,
            pair.replacement_form,
            self.max_length,
        )
        rejected_anchor = encode_api_anchor(
            self.tokenizer,
            pair.prompt,
            pair.rejected,
            pair.deprecated_form,
            self.max_length,
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "api_anchor_input_ids": anchor["input_ids"],
            "api_anchor_attention_mask": anchor["attention_mask"],
            "api_anchor_labels": anchor["labels"],
            "rejected_api_anchor_input_ids": rejected_anchor["input_ids"],
            "rejected_api_anchor_attention_mask": rejected_anchor["attention_mask"],
            "rejected_api_anchor_labels": rejected_anchor["labels"],
        }


@dataclass
class DPOCollator:
    tokenizer: object

    def _pad(self, tensors: List[torch.Tensor], *, value: int) -> torch.Tensor:
        max_len = max(tensor.shape[0] for tensor in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_len - tensor.shape[0]
            padded.append(torch.cat([tensor, torch.full((pad_len,), value, dtype=tensor.dtype)]))
        return torch.stack(padded)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        return {
            "chosen_input_ids": self._pad([item["chosen_input_ids"] for item in features], value=pad_id),
            "chosen_attention_mask": self._pad(
                [item["chosen_attention_mask"] for item in features],
                value=0,
            ),
            "chosen_labels": self._pad([item["chosen_labels"] for item in features], value=-100),
            "rejected_input_ids": self._pad([item["rejected_input_ids"] for item in features], value=pad_id),
            "rejected_attention_mask": self._pad(
                [item["rejected_attention_mask"] for item in features],
                value=0,
            ),
            "rejected_labels": self._pad([item["rejected_labels"] for item in features], value=-100),
            "api_anchor_input_ids": self._pad(
                [item["api_anchor_input_ids"] for item in features],
                value=pad_id,
            ),
            "api_anchor_attention_mask": self._pad(
                [item["api_anchor_attention_mask"] for item in features],
                value=0,
            ),
            "api_anchor_labels": self._pad(
                [item["api_anchor_labels"] for item in features],
                value=-100,
            ),
            "rejected_api_anchor_input_ids": self._pad(
                [item["rejected_api_anchor_input_ids"] for item in features],
                value=pad_id,
            ),
            "rejected_api_anchor_attention_mask": self._pad(
                [item["rejected_api_anchor_attention_mask"] for item in features],
                value=0,
            ),
            "rejected_api_anchor_labels": self._pad(
                [item["rejected_api_anchor_labels"] for item in features],
                value=-100,
            ),
        }


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor, *, reduction: str = "sum") -> torch.Tensor:
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = torch.gather(
        F.log_softmax(shifted_logits.float(), dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = token_log_probs * mask
    summed = token_log_probs.sum(dim=-1)
    if reduction == "sum":
        return summed
    counts = mask.sum(dim=-1).clamp_min(1)
    return summed / counts


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = float(beta) * (policy_logratios - reference_logratios)
    return -F.logsigmoid(logits).mean()


def api_anchor_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]).float(),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def disable_dropout_in_model(model) -> int:
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            if module.p != 0.0:
                module.p = 0.0
                changed += 1
    return changed


@contextlib.contextmanager
def temporary_eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


@contextlib.contextmanager
def disable_adapter_if_available(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        yield
    else:
        with disable_adapter():
            yield


class VersionAwareDPOTrainer:
    def __init__(
        self,
        *,
        beta: float = 0.1,
        logprob_reduction: str = "sum",
        api_anchor_weight: float = 0.0,
        dpo_scope: str = "full",
    ):
        if dpo_scope not in {"full", "api_span"}:
            raise ValueError("dpo_scope must be 'full' or 'api_span'")
        self.beta = float(beta)
        self.logprob_reduction = logprob_reduction
        self.api_anchor_weight = float(api_anchor_weight)
        self.dpo_scope = dpo_scope

    def batch_logps(self, model, inputs: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        outputs = model(
            input_ids=inputs[f"{prefix}_input_ids"],
            attention_mask=inputs[f"{prefix}_attention_mask"],
        )
        return sequence_log_probs(
            outputs.logits,
            inputs[f"{prefix}_labels"],
            reduction=self.logprob_reduction,
        )

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        chosen_prefix = "chosen"
        rejected_prefix = "rejected"
        if self.dpo_scope == "api_span":
            chosen_prefix = "api_anchor"
            rejected_prefix = "rejected_api_anchor"

        with temporary_eval_mode(model):
            policy_chosen = self.batch_logps(model, inputs, chosen_prefix)
            policy_rejected = self.batch_logps(model, inputs, rejected_prefix)
            anchor_loss = None
            if self.api_anchor_weight > 0.0:
                anchor_outputs = model(
                    input_ids=inputs["api_anchor_input_ids"],
                    attention_mask=inputs["api_anchor_attention_mask"],
                )
                anchor_loss = api_anchor_cross_entropy_loss(
                    anchor_outputs.logits,
                    inputs["api_anchor_labels"],
                )

            with torch.no_grad():
                with disable_adapter_if_available(model):
                    reference_chosen = self.batch_logps(model, inputs, chosen_prefix)
                    reference_rejected = self.batch_logps(model, inputs, rejected_prefix)

        preference_loss = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=self.beta,
        )
        if anchor_loss is None:
            return preference_loss
        return preference_loss + self.api_anchor_weight * anchor_loss
