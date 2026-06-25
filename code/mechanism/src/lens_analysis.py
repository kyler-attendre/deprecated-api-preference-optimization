import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/data/models"))


def model_path_for_env(model_key: str, *relative_parts: str) -> str:
    env_key = f"MODEL_PATH_{model_key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    return str(MODEL_ROOT.joinpath(*relative_parts))


MODEL_REGISTRY = {
    "starcoder2_3b": {
        "path": model_path_for_env("starcoder2_3b", "StarCoder", "starcoder2-3b"),
        "max_length": 384,
    },
    "starcoder2_7b": {
        "path": model_path_for_env("starcoder2_7b", "StarCoder", "starcoder2-7b"),
        "max_length": 384,
    },
    "starcoder2_15b": {
        "path": model_path_for_env("starcoder2_15b", "StarCoder", "starcoder2-15b"),
        "max_length": 256,
    },
    "deepseek_coder_6_7b_instruct": {
        "path": model_path_for_env("deepseek_coder_6_7b_instruct", "deepseek-ai", "deepseek-coder-6.7b-instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_3b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_3b_instruct", "Qwen", "Qwen2.5-Coder-3B-Instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_7b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_7b_instruct", "Qwen", "Qwen2.5-Coder-7B-Instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_14b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_14b_instruct", "Qwen", "Qwen2.5-Coder-14B-Instruct"),
        "max_length": 256,
    },
}


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_many_jsonl(paths: Sequence[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


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


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def alias_forms(api_name: str) -> List[str]:
    api_name = str(api_name).strip()
    aliases = {api_name}
    if api_name.startswith("torch.nn.functional."):
        aliases.add("F." + api_name.split(".")[-1])
    if api_name.startswith("tensorflow."):
        aliases.add("tf." + api_name.split(".", 1)[1])
    if api_name.startswith("numpy."):
        aliases.add("np." + api_name.split(".", 1)[1])
    if api_name.startswith("pandas."):
        aliases.add("pd." + api_name.split(".", 1)[1])
    return sorted(alias for alias in aliases if alias)


def first_alias_hit(text: str, api_name: str) -> Optional[str]:
    for alias in alias_forms(api_name):
        if alias and alias in text:
            return alias
    return None


def deprecated_alias_matching_replacement(
    replacement_api: str,
    replacement_form: str,
    deprecated_api: str,
) -> str:
    if replacement_form == replacement_api:
        return deprecated_api
    if replacement_api.startswith("tensorflow.") and replacement_form.startswith("tf."):
        return "tf." + deprecated_api.split(".", 1)[1]
    if replacement_api.startswith("torch.nn.functional.") and replacement_form.startswith("F."):
        return "F." + deprecated_api.split(".")[-1]
    if replacement_api.startswith("numpy.") and replacement_form.startswith("np."):
        return "np." + deprecated_api.split(".", 1)[1]
    if replacement_api.startswith("pandas.") and replacement_form.startswith("pd."):
        return "pd." + deprecated_api.split(".", 1)[1]
    return deprecated_api


def common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return left[:idx]


def shared_api_decision_prefix(replacement_form: str, deprecated_form: str) -> str:
    shared = common_prefix(replacement_form, deprecated_form)
    if not shared:
        return ""
    if shared == replacement_form or shared == deprecated_form or shared[-1].isalnum() or shared[-1] == "_":
        last_dot = shared.rfind(".")
        if last_dot >= 0:
            return shared[: last_dot + 1]
        return ""
    return shared


@dataclass(frozen=True)
class FocusExample:
    row_id: str
    library: str
    category: str
    task_family: str
    sample_type: str
    prompt_field: str
    prompt_text: str
    completion_prefix: str
    decision_prefix: str
    replacement_api: str
    deprecated_api: str
    replacement_form: str
    deprecated_form: str
    shared_api_prefix: str
    replacement_suffix: str
    deprecated_suffix: str
    source_file: str


def build_focus_example(row: Dict, prompt_field: str = "version_prompt") -> Optional[FocusExample]:
    prompt_text = str(row.get(prompt_field) or "")
    target = str(row.get("target") or row.get("reference") or "")
    replacement_api = str(row.get("replacement_api") or "").strip()
    deprecated_apis = ensure_list(row.get("deprecated_api"))
    if not prompt_text or not target or not replacement_api or not deprecated_apis:
        return None

    replacement_form = first_alias_hit(target, replacement_api)
    if not replacement_form:
        return None

    deprecated_form = deprecated_alias_matching_replacement(
        replacement_api=replacement_api,
        replacement_form=replacement_form,
        deprecated_api=deprecated_apis[0],
    )
    replacement_start = target.find(replacement_form)
    if replacement_start < 0:
        return None

    shared_api_prefix = shared_api_decision_prefix(replacement_form, deprecated_form)
    replacement_suffix = replacement_form[len(shared_api_prefix):]
    deprecated_suffix = deprecated_form[len(shared_api_prefix):]
    if not replacement_suffix or not deprecated_suffix:
        return None

    completion_prefix = target[:replacement_start]
    decision_prefix = prompt_text + completion_prefix + shared_api_prefix

    return FocusExample(
        row_id=str(row.get("id") or ""),
        library=str(row.get("library") or "unknown"),
        category=str(row.get("category") or "unknown"),
        task_family=str(row.get("task_family") or "unknown"),
        sample_type=str(row.get("sample_type") or "unknown"),
        prompt_field=prompt_field,
        prompt_text=prompt_text,
        completion_prefix=completion_prefix,
        decision_prefix=decision_prefix,
        replacement_api=replacement_api,
        deprecated_api=deprecated_apis[0],
        replacement_form=replacement_form,
        deprecated_form=deprecated_form,
        shared_api_prefix=shared_api_prefix,
        replacement_suffix=replacement_suffix,
        deprecated_suffix=deprecated_suffix,
        source_file=str(row.get("source_file") or ""),
    )


def build_focus_examples(rows: Iterable[Dict], prompt_field: str = "version_prompt") -> List[FocusExample]:
    examples: List[FocusExample] = []
    for row in rows:
        example = build_focus_example(row, prompt_field=prompt_field)
        if example is not None:
            examples.append(example)
    return examples


def focus_example_to_dict(example: FocusExample) -> Dict:
    return asdict(example)


def build_model(model_name_or_path: str, adapter_dir: Optional[Path] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def get_decoder_layers(model):
    base = getattr(model, "base_model", model)
    candidate_roots = [
        getattr(base, "model", None),
        getattr(getattr(base, "model", None), "model", None),
        getattr(base, "transformer", None),
        base,
    ]
    for root in candidate_roots:
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        if hasattr(root, "h"):
            return root.h
        if hasattr(root, "gpt_neox") and hasattr(root.gpt_neox, "layers"):
            return root.gpt_neox.layers
    raise ValueError(f"Cannot locate decoder layers for model type {type(model).__name__}")


def get_hidden_size(model) -> int:
    embeddings = model.get_input_embeddings()
    return int(embeddings.weight.shape[1])


def get_output_projection(model):
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        return model.get_output_embeddings()
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    raise ValueError("Cannot locate model output projection")


def get_final_norm(model):
    base = getattr(model, "base_model", model)
    candidate_modules = [
        getattr(getattr(base, "model", None), "norm", None),
        getattr(getattr(base, "model", None), "final_layernorm", None),
        getattr(getattr(base, "transformer", None), "ln_f", None),
        getattr(getattr(base, "gpt_neox", None), "final_layer_norm", None),
        getattr(base, "norm", None),
    ]
    for module in candidate_modules:
        if module is not None:
            return module
    return None


def apply_final_norm(hidden: torch.Tensor, final_norm) -> torch.Tensor:
    if final_norm is None:
        return hidden
    return final_norm(hidden)


def project_hidden_to_logits(hidden: torch.Tensor, final_norm, output_projection) -> torch.Tensor:
    normalized = apply_final_norm(hidden, final_norm)
    return output_projection(normalized)


def encode_focus_input(tokenizer, prefix_text: str, suffix_text: str, max_length: int) -> Dict[str, List[int]]:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    if not prefix_ids or not suffix_ids:
        raise ValueError("Focus input encoded to empty prefix or suffix")

    max_prefix_len = max(1, max_length - len(suffix_ids))
    prefix_ids = prefix_ids[-max_prefix_len:]
    input_ids = prefix_ids + suffix_ids
    predict_positions = [len(prefix_ids) - 1 + offset for offset in range(len(suffix_ids))]
    return {
        "input_ids": input_ids,
        "suffix_ids": suffix_ids,
        "predict_positions": predict_positions,
    }


def encode_prefix_only(tokenizer, prefix_text: str, max_length: int) -> List[int]:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError("Prefix-only focus input encoded to empty sequence")
    return prefix_ids[-max_length:]


def run_hidden_forward(model, input_ids: Sequence[int]) -> Dict:
    tensor = torch.tensor([list(input_ids)], dtype=torch.long)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        outputs = model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            output_hidden_states=True,
            use_cache=False,
        )
    return {
        "hidden_states": outputs.hidden_states,
        "logits": outputs.logits,
    }


class LowRankTranslator(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        projected = hidden.to(dtype=self.down.weight.dtype)
        translated = self.up(self.down(projected))
        return hidden + translated.to(dtype=input_dtype)


class LowRankTunedLens(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, rank: int):
        super().__init__()
        self.rank = int(rank)
        self.hidden_size = int(hidden_size)
        self.translators = nn.ModuleList(
            [LowRankTranslator(hidden_size=hidden_size, rank=rank) for _ in range(num_layers)]
        )

    def forward_layer(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.translators[layer_idx](hidden)


def save_tuned_lens(path: Path, tuned_lens: LowRankTunedLens, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": tuned_lens.state_dict(),
        "rank": tuned_lens.rank,
        "hidden_size": tuned_lens.hidden_size,
        "num_layers": len(tuned_lens.translators),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_tuned_lens(path: Path, *, device: torch.device) -> Tuple[LowRankTunedLens, Dict]:
    payload = torch.load(path, map_location=device)
    tuned_lens = LowRankTunedLens(
        num_layers=int(payload["num_layers"]),
        hidden_size=int(payload["hidden_size"]),
        rank=int(payload["rank"]),
    )
    tuned_lens.load_state_dict(payload["state_dict"])
    tuned_lens.to(device)
    tuned_lens.eval()
    return tuned_lens, dict(payload.get("metadata") or {})


def gather_token_metrics(logits: torch.Tensor, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(dim=-1, index=token_ids.view(-1, 1)).squeeze(-1)
    target_probs = probs.gather(dim=-1, index=token_ids.view(-1, 1)).squeeze(-1)
    target_logits = logits.gather(dim=-1, index=token_ids.view(-1, 1)).squeeze(-1)
    ranks = 1 + (logits > target_logits.unsqueeze(-1)).sum(dim=-1)
    return {
        "log_probs": target_log_probs,
        "probs": target_probs,
        "ranks": ranks.to(dtype=torch.float32),
    }


def gather_single_token_metric(logits: torch.Tensor, token_id: int) -> Dict[str, float]:
    token_tensor = torch.tensor([int(token_id)], dtype=torch.long, device=logits.device)
    metrics = gather_token_metrics(logits.unsqueeze(0), token_tensor)
    return {
        "logprob": float(metrics["log_probs"][0].item()),
        "prob": float(metrics["probs"][0].item()),
        "rank": float(metrics["ranks"][0].item()),
    }


def jsd_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-12))
    kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
    kl_qm = torch.sum(q * (log_q - log_m), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def layerwise_candidate_trace(
    *,
    model,
    final_norm,
    output_projection,
    input_ids: Sequence[int],
    suffix_ids: Sequence[int],
    predict_positions: Sequence[int],
    tuned_lens: Optional[LowRankTunedLens] = None,
) -> List[Dict]:
    forward = run_hidden_forward(model, input_ids)
    hidden_states = forward["hidden_states"]
    token_tensor = torch.tensor(list(suffix_ids), dtype=torch.long, device=hidden_states[-1].device)
    positions = torch.tensor(list(predict_positions), dtype=torch.long, device=hidden_states[-1].device)

    traces: List[Dict] = []
    num_layers = len(hidden_states) - 1
    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx + 1][0, positions, :]
        if tuned_lens is not None:
            hidden = tuned_lens.forward_layer(layer_idx, hidden)
        logits = project_hidden_to_logits(hidden, final_norm, output_projection)
        metrics = gather_token_metrics(logits, token_tensor)
        traces.append(
            {
                "layer": layer_idx,
                "sequence_logprob": float(metrics["log_probs"].sum().item()),
                "avg_token_logprob": float(metrics["log_probs"].mean().item()),
                "first_token_logprob": float(metrics["log_probs"][0].item()),
                "first_token_prob": float(metrics["probs"][0].item()),
                "first_token_rank": float(metrics["ranks"][0].item()),
                "token_count": len(suffix_ids),
            }
        )
    return traces


def layerwise_decision_jsd_trace(
    *,
    model,
    final_norm,
    output_projection,
    input_ids: Sequence[int],
    replacement_first_token_id: int,
    deprecated_first_token_id: int,
    tuned_lens: Optional[LowRankTunedLens] = None,
) -> List[Dict]:
    forward = run_hidden_forward(model, input_ids)
    hidden_states = forward["hidden_states"]

    final_hidden = hidden_states[-1][0, -1, :]
    final_logits = project_hidden_to_logits(final_hidden.unsqueeze(0), final_norm, output_projection).squeeze(0)
    final_replacement = gather_single_token_metric(final_logits, replacement_first_token_id)
    final_deprecated = gather_single_token_metric(final_logits, deprecated_first_token_id)

    traces: List[Dict] = []
    num_layers = len(hidden_states) - 1
    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx + 1][0, -1, :]
        if tuned_lens is not None:
            hidden = tuned_lens.forward_layer(layer_idx, hidden.unsqueeze(0)).squeeze(0)
        logits = project_hidden_to_logits(hidden.unsqueeze(0), final_norm, output_projection).squeeze(0)
        replacement = gather_single_token_metric(logits, replacement_first_token_id)
        deprecated = gather_single_token_metric(logits, deprecated_first_token_id)
        traces.append(
            {
                "layer": layer_idx,
                "jsd_to_final": float(jsd_from_logits(logits, final_logits).item()),
                "replacement_first_token_logprob": replacement["logprob"],
                "deprecated_first_token_logprob": deprecated["logprob"],
                "first_token_logprob_margin": replacement["logprob"] - deprecated["logprob"],
                "replacement_first_token_prob": replacement["prob"],
                "deprecated_first_token_prob": deprecated["prob"],
                "replacement_first_token_rank": replacement["rank"],
                "deprecated_first_token_rank": deprecated["rank"],
                "replacement_wins_first_token": replacement["logprob"] > deprecated["logprob"],
                "final_replacement_first_token_logprob": final_replacement["logprob"],
                "final_deprecated_first_token_logprob": final_deprecated["logprob"],
                "final_first_token_logprob_margin": final_replacement["logprob"] - final_deprecated["logprob"],
            }
        )
    return traces


def compare_focus_example(
    *,
    model,
    tokenizer,
    final_norm,
    output_projection,
    example: FocusExample,
    max_length: int,
    tuned_lens: Optional[LowRankTunedLens] = None,
) -> Dict:
    replacement_encoded = encode_focus_input(
        tokenizer=tokenizer,
        prefix_text=example.decision_prefix,
        suffix_text=example.replacement_suffix,
        max_length=max_length,
    )
    deprecated_encoded = encode_focus_input(
        tokenizer=tokenizer,
        prefix_text=example.decision_prefix,
        suffix_text=example.deprecated_suffix,
        max_length=max_length,
    )
    prefix_input_ids = encode_prefix_only(
        tokenizer=tokenizer,
        prefix_text=example.decision_prefix,
        max_length=max_length,
    )

    replacement_trace = layerwise_candidate_trace(
        model=model,
        final_norm=final_norm,
        output_projection=output_projection,
        input_ids=replacement_encoded["input_ids"],
        suffix_ids=replacement_encoded["suffix_ids"],
        predict_positions=replacement_encoded["predict_positions"],
        tuned_lens=tuned_lens,
    )
    deprecated_trace = layerwise_candidate_trace(
        model=model,
        final_norm=final_norm,
        output_projection=output_projection,
        input_ids=deprecated_encoded["input_ids"],
        suffix_ids=deprecated_encoded["suffix_ids"],
        predict_positions=deprecated_encoded["predict_positions"],
        tuned_lens=tuned_lens,
    )
    decision_trace = layerwise_decision_jsd_trace(
        model=model,
        final_norm=final_norm,
        output_projection=output_projection,
        input_ids=prefix_input_ids,
        replacement_first_token_id=replacement_encoded["suffix_ids"][0],
        deprecated_first_token_id=deprecated_encoded["suffix_ids"][0],
        tuned_lens=tuned_lens,
    )

    layerwise = []
    for rep, dep, dec in zip(replacement_trace, deprecated_trace, decision_trace):
        replacement_perplexity = float(torch.exp(torch.tensor(-rep["avg_token_logprob"])).item())
        deprecated_perplexity = float(torch.exp(torch.tensor(-dep["avg_token_logprob"])).item())
        layerwise.append(
            {
                "layer": rep["layer"],
                "replacement_sequence_logprob": rep["sequence_logprob"],
                "deprecated_sequence_logprob": dep["sequence_logprob"],
                "sequence_logprob_margin": rep["sequence_logprob"] - dep["sequence_logprob"],
                "replacement_avg_token_logprob": rep["avg_token_logprob"],
                "deprecated_avg_token_logprob": dep["avg_token_logprob"],
                "avg_token_logprob_margin": rep["avg_token_logprob"] - dep["avg_token_logprob"],
                "replacement_perplexity": replacement_perplexity,
                "deprecated_perplexity": deprecated_perplexity,
                "perplexity_ratio_deprecated_over_replacement": deprecated_perplexity / max(replacement_perplexity, 1e-12),
                "replacement_first_token_logprob": rep["first_token_logprob"],
                "deprecated_first_token_logprob": dep["first_token_logprob"],
                "first_token_logprob_margin": rep["first_token_logprob"] - dep["first_token_logprob"],
                "replacement_first_token_rank": rep["first_token_rank"],
                "deprecated_first_token_rank": dep["first_token_rank"],
                "replacement_first_token_prob": rep["first_token_prob"],
                "deprecated_first_token_prob": dep["first_token_prob"],
                "replacement_wins_sequence": rep["sequence_logprob"] > dep["sequence_logprob"],
                "replacement_wins_first_token": rep["first_token_logprob"] > dep["first_token_logprob"],
                "decision_jsd_to_final": dec["jsd_to_final"],
                "decision_replacement_first_token_logprob": dec["replacement_first_token_logprob"],
                "decision_deprecated_first_token_logprob": dec["deprecated_first_token_logprob"],
                "decision_first_token_logprob_margin": dec["first_token_logprob_margin"],
                "decision_replacement_first_token_prob": dec["replacement_first_token_prob"],
                "decision_deprecated_first_token_prob": dec["deprecated_first_token_prob"],
                "decision_replacement_first_token_rank": dec["replacement_first_token_rank"],
                "decision_deprecated_first_token_rank": dec["deprecated_first_token_rank"],
                "decision_replacement_wins_first_token": dec["replacement_wins_first_token"],
                "decision_final_first_token_logprob_margin": dec["final_first_token_logprob_margin"],
            }
        )

    payload = focus_example_to_dict(example)
    payload["layerwise"] = layerwise
    return payload


def aggregate_layerwise(results: Sequence[Dict]) -> List[Dict]:
    if not results:
        return []
    num_layers = len(results[0]["layerwise"])
    summary: List[Dict] = []
    for layer_idx in range(num_layers):
        rows = [result["layerwise"][layer_idx] for result in results]
        count = len(rows)
        summary.append(
            {
                "layer": layer_idx,
                "samples": count,
                "replacement_sequence_win_rate": sum(row["replacement_wins_sequence"] for row in rows) / count,
                "replacement_first_token_win_rate": sum(row["replacement_wins_first_token"] for row in rows) / count,
                "mean_sequence_logprob_margin": sum(row["sequence_logprob_margin"] for row in rows) / count,
                "mean_avg_token_logprob_margin": sum(row["avg_token_logprob_margin"] for row in rows) / count,
                "geometric_mean_perplexity_ratio_deprecated_over_replacement": float(
                    torch.exp(
                        torch.tensor(sum(row["avg_token_logprob_margin"] for row in rows) / count, dtype=torch.float32)
                    ).item()
                ),
                "mean_first_token_logprob_margin": sum(row["first_token_logprob_margin"] for row in rows) / count,
                "mean_replacement_first_token_rank": sum(row["replacement_first_token_rank"] for row in rows) / count,
                "mean_deprecated_first_token_rank": sum(row["deprecated_first_token_rank"] for row in rows) / count,
                "mean_decision_jsd_to_final": sum(row["decision_jsd_to_final"] for row in rows) / count,
                "decision_replacement_first_token_win_rate": sum(
                    row["decision_replacement_wins_first_token"] for row in rows
                )
                / count,
                "mean_decision_first_token_logprob_margin": sum(
                    row["decision_first_token_logprob_margin"] for row in rows
                )
                / count,
                "mean_decision_replacement_first_token_rank": sum(
                    row["decision_replacement_first_token_rank"] for row in rows
                )
                / count,
                "mean_decision_deprecated_first_token_rank": sum(
                    row["decision_deprecated_first_token_rank"] for row in rows
                )
                / count,
            }
        )
    return summary


def build_training_batch(
    *,
    tokenizer,
    examples: Sequence[FocusExample],
    indices: Sequence[int],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    encoded_rows = []
    for idx in indices:
        encoded = encode_focus_input(
            tokenizer=tokenizer,
            prefix_text=examples[idx].decision_prefix,
            suffix_text=examples[idx].replacement_suffix,
            max_length=max_length,
        )
        encoded_rows.append(encoded)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_seq_len = max(len(row["input_ids"]) for row in encoded_rows)
    input_ids = []
    attention_masks = []
    all_positions = []
    all_targets = []
    row_ids = []
    for encoded in encoded_rows:
        ids = list(encoded["input_ids"])
        pad_len = max_seq_len - len(ids)
        input_ids.append(ids + ([pad_id] * pad_len))
        attention_masks.append(([1] * len(ids)) + ([0] * pad_len))
        all_positions.append(list(encoded["predict_positions"]))
        all_targets.append(list(encoded["suffix_ids"]))
        row_ids.append(len(encoded["suffix_ids"]))

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "positions": all_positions,
        "targets": all_targets,
        "token_counts": row_ids,
    }
    if torch.cuda.is_available():
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()
    return batch


def tuned_lens_loss(
    *,
    model,
    tuned_lens: LowRankTunedLens,
    final_norm,
    output_projection,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    final_hidden = hidden_states[-1]
    total_loss = torch.tensor(0.0, device=final_hidden.device)
    total_positions = 0

    final_vectors = []
    layer_vectors: List[List[torch.Tensor]] = [[] for _ in range(len(hidden_states) - 1)]
    target_ids = []
    for batch_idx, (positions, targets) in enumerate(zip(batch["positions"], batch["targets"])):
        for position, token_id in zip(positions, targets):
            final_vectors.append(final_hidden[batch_idx, position, :])
            target_ids.append(token_id)
            for layer_idx in range(len(hidden_states) - 1):
                layer_vectors[layer_idx].append(hidden_states[layer_idx + 1][batch_idx, position, :])

    if not target_ids:
        return total_loss

    final_vectors_tensor = torch.stack(final_vectors, dim=0)
    final_logits = project_hidden_to_logits(final_vectors_tensor, final_norm, output_projection).detach()
    target_log_probs = F.log_softmax(final_logits.float(), dim=-1)

    for layer_idx, vectors in enumerate(layer_vectors):
        layer_tensor = torch.stack(vectors, dim=0)
        translated = tuned_lens.forward_layer(layer_idx, layer_tensor)
        pred_logits = project_hidden_to_logits(translated, final_norm, output_projection)
        pred_log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        layer_loss = F.kl_div(pred_log_probs, target_log_probs, reduction="batchmean", log_target=True)
        total_loss = total_loss + layer_loss
        total_positions += 1

    return total_loss / max(total_positions, 1)


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
