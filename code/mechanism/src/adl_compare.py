from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from .lens_analysis import project_hidden_to_logits, run_hidden_forward
from .variant_compare import build_group_labels, model_label_for


DEFAULT_RANDOM_NEUTRAL_TEXT_PROMPTS = [
    "Write a short note explaining what this helper function should return.\n",
    "Summarize the purpose of this code block in one sentence.\n",
    "Describe a simple validation step for a generic input value.\n",
    "Explain how to rename a local variable without changing behavior.\n",
    "Outline a minimal unit test for a small utility function.\n",
    "Give a neutral comment that could appear above a data-processing helper.\n",
    "Describe a straightforward way to handle an empty list input.\n",
    "Write a brief explanation of why a temporary variable is useful here.\n",
    "State a generic requirement for formatting a return value.\n",
    "Describe a basic refactor that improves readability without changing logic.\n",
    "Explain how to guard against missing keys in a dictionary.\n",
    "Describe a simple loop that collects values into a new list.\n",
]

DEFAULT_LIBRARY_NEUTRAL_CODE_PROMPTS = [
    "def normalize_values(items):\n    cleaned = []\n    for item in items:\n        if item is None:\n            continue\n",
    "def format_record(record):\n    result = {}\n    for key, value in record.items():\n        if value is None:\n            continue\n",
    "class Counter:\n    def update(self, values):\n        total = 0\n        for value in values:\n            total += int(value)\n",
    "def chunk_pairs(values, size):\n    chunks = []\n    current = []\n    for value in values:\n        current.append(value)\n",
    "def build_mapping(rows):\n    mapping = {}\n    for row in rows:\n        name = row.get('name')\n        if not name:\n            continue\n",
    "def merge_lists(left, right):\n    merged = []\n    for item in left:\n        merged.append(item)\n    for item in right:\n        merged.append(item)\n",
]


@dataclass(frozen=True)
class AdlPrompt:
    prompt_id: str
    prompt_type: str
    prompt_text: str
    source: str
    source_row_id: str = ""
    library: str = ""


def prompt_to_dict(prompt: AdlPrompt) -> Dict:
    return asdict(prompt)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _version_header(line: str) -> bool:
    line = line.strip().lower()
    return line.startswith("# ") and any(ch.isdigit() for ch in line)


def collect_blocklist_fragments(rows: Iterable[Dict]) -> Set[str]:
    fragments: Set[str] = set()
    for row in rows:
        replacement = str(row.get("replacement_api") or "").strip().lower()
        if replacement:
            fragments.add(replacement)
        deprecated = row.get("deprecated_api") or []
        if isinstance(deprecated, str):
            deprecated = [deprecated]
        for item in deprecated:
            item = str(item or "").strip().lower()
            if item:
                fragments.add(item)
        library = str(row.get("library") or "").strip().lower()
        if library:
            fragments.add(library)
        prompt = str(row.get("version_prompt") or "")
        first_line = prompt.splitlines()[0].strip().lower() if prompt else ""
        if _version_header(first_line):
            fragments.add(first_line)
    fragments.update({"deprecated", "replacement", "version_prompt"})
    return {frag for frag in fragments if frag}


def contains_banned_fragment(text: str, banned_fragments: Set[str]) -> bool:
    lowered = text.lower()
    return any(fragment in lowered for fragment in banned_fragments)


def _strip_version_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _version_header(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def build_library_neutral_code_prompts(
    rows: Sequence[Dict],
    *,
    max_prompts: int,
    min_chars: int = 48,
    max_chars: int = 320,
) -> List[AdlPrompt]:
    banned = collect_blocklist_fragments(rows)
    prompts: List[AdlPrompt] = []
    seen: Set[str] = set()
    for row in rows:
        prompt = _strip_version_lines(str(row.get("probing_input") or ""))
        prompt = _normalize_text(prompt)
        if len(prompt) < min_chars:
            continue
        if contains_banned_fragment(prompt, banned):
            continue
        prompt = prompt[:max_chars].rstrip() + ("\n" if not prompt.endswith("\n") else "")
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(
            AdlPrompt(
                prompt_id=f"code-{len(prompts):04d}",
                prompt_type="library_neutral_code",
                prompt_text=prompt,
                source="processed_clean.probing_input",
                source_row_id=str(row.get("id") or ""),
                library=str(row.get("library") or ""),
            )
        )
        if len(prompts) >= max_prompts:
            break
    if len(prompts) < max_prompts:
        for text in DEFAULT_LIBRARY_NEUTRAL_CODE_PROMPTS:
            prompt = _normalize_text(text)
            if prompt in seen:
                continue
            prompts.append(
                AdlPrompt(
                    prompt_id=f"code-fallback-{len(prompts):04d}",
                    prompt_type="library_neutral_code",
                    prompt_text=prompt if prompt.endswith("\n") else prompt + "\n",
                    source="builtin_fallback",
                )
            )
            if len(prompts) >= max_prompts:
                break
    return prompts


def build_random_neutral_text_prompts(
    rows: Sequence[Dict],
    *,
    max_prompts: int,
) -> List[AdlPrompt]:
    banned = collect_blocklist_fragments(rows)
    prompts: List[AdlPrompt] = []
    for idx, text in enumerate(DEFAULT_RANDOM_NEUTRAL_TEXT_PROMPTS):
        prompt = _normalize_text(text)
        if contains_banned_fragment(prompt, banned):
            continue
        prompts.append(
            AdlPrompt(
                prompt_id=f"text-{idx:04d}",
                prompt_type="random_neutral_text",
                prompt_text=prompt if prompt.endswith("\n") else prompt + "\n",
                source="builtin_templates",
            )
        )
        if len(prompts) >= max_prompts:
            break
    return prompts


def build_prompt_sets(rows: Sequence[Dict], *, max_prompts_per_type: int) -> List[AdlPrompt]:
    prompts = []
    prompts.extend(build_random_neutral_text_prompts(rows, max_prompts=max_prompts_per_type))
    prompts.extend(build_library_neutral_code_prompts(rows, max_prompts=max_prompts_per_type))
    return prompts


def encode_prompt(tokenizer, prompt_text: str, *, max_length: int) -> List[int]:
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Prompt encoded to an empty sequence")
    return ids[-max_length:]


def trace_last_token_hidden_and_logits(
    *,
    model,
    tokenizer,
    final_norm,
    output_projection,
    prompt_text: str,
    max_length: int,
) -> Dict:
    input_ids = encode_prompt(tokenizer, prompt_text, max_length=max_length)
    forward = run_hidden_forward(model, input_ids)
    hidden_states = forward["hidden_states"]
    last_token_hiddens: List[torch.Tensor] = []
    for layer_idx in range(len(hidden_states) - 1):
        hidden = hidden_states[layer_idx + 1][0, -1, :].detach().float().cpu()
        last_token_hiddens.append(hidden)
    final_hidden = hidden_states[-1][0, -1, :]
    final_logits = project_hidden_to_logits(final_hidden.unsqueeze(0), final_norm, output_projection).squeeze(0)
    return {
        "input_ids": input_ids,
        "last_token_hiddens": last_token_hiddens,
        "final_logits": final_logits.detach().float().cpu(),
    }


def build_token_family_token_ids(tokenizer, rows: Sequence[Dict]) -> Dict[str, List[int]]:
    def tokenize_strings(items: Iterable[str]) -> List[int]:
        token_ids: Set[int] = set()
        for item in items:
            item = str(item or "").strip()
            if not item:
                continue
            encoded = tokenizer(item, add_special_tokens=False)["input_ids"]
            token_ids.update(int(token_id) for token_id in encoded)
        return sorted(token_ids)

    replacements: Set[str] = set()
    deprecated: Set[str] = set()
    library_and_version: Set[str] = set()
    for row in rows:
        replacement = str(row.get("replacement_api") or "").strip()
        if replacement:
            replacements.add(replacement)
        dep = row.get("deprecated_api") or []
        if isinstance(dep, str):
            dep = [dep]
        for item in dep:
            item = str(item or "").strip()
            if item:
                deprecated.add(item)
        library = str(row.get("library") or "").strip()
        if library:
            library_and_version.add(library)
        version_prompt = str(row.get("version_prompt") or "")
        if version_prompt:
            first_line = version_prompt.splitlines()[0].strip()
            if first_line:
                library_and_version.add(first_line)
    return {
        "replacement": tokenize_strings(replacements),
        "deprecated": tokenize_strings(deprecated),
        "library_version": tokenize_strings(library_and_version),
    }


def _top_tokens(tokenizer, logits_diff: torch.Tensor, *, k: int, largest: bool) -> List[Dict]:
    if logits_diff.numel() == 0:
        return []
    values, indices = torch.topk(logits_diff, k=min(k, logits_diff.shape[0]), largest=largest)
    rows = []
    for value, index in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "token_id": int(index),
                "token_text": tokenizer.decode([int(index)]),
                "logit_diff": float(value),
            }
        )
    return rows


def compare_variant_against_base(
    *,
    tokenizer,
    base_prompt_traces: Dict[str, Dict],
    variant_prompt_traces: Dict[str, Dict],
    prompts: Sequence[AdlPrompt],
    family_token_ids: Dict[str, List[int]],
    top_k: int = 25,
) -> Dict:
    if not prompts:
        raise ValueError("No prompts provided for ADL comparison")

    prompt_rows: List[Dict] = []
    per_type: Dict[str, Dict] = {}
    num_layers = len(next(iter(base_prompt_traces.values()))["last_token_hiddens"])

    for prompt in prompts:
        base_trace = base_prompt_traces[prompt.prompt_id]
        variant_trace = variant_prompt_traces[prompt.prompt_id]
        layer_norms = []
        for base_hidden, variant_hidden in zip(base_trace["last_token_hiddens"], variant_trace["last_token_hiddens"]):
            diff = variant_hidden - base_hidden
            layer_norms.append(float(torch.linalg.norm(diff).item()))
        final_logits_diff = variant_trace["final_logits"] - base_trace["final_logits"]
        family_scores = {}
        for family, token_ids in family_token_ids.items():
            if not token_ids:
                family_scores[family] = 0.0
                continue
            family_scores[family] = float(final_logits_diff[token_ids].mean().item())

        row = {
            "prompt_id": prompt.prompt_id,
            "prompt_type": prompt.prompt_type,
            "source": prompt.source,
            "layer_diff_norms": layer_norms,
            "final_layer_diff_norm": layer_norms[-1],
            "family_scores": family_scores,
        }
        prompt_rows.append(row)

        bucket = per_type.setdefault(
            prompt.prompt_type,
            {
                "count": 0,
                "layer_norm_sums": [0.0] * num_layers,
                "final_logits_diff_sum": None,
                "family_score_sums": {key: 0.0 for key in family_token_ids},
            },
        )
        bucket["count"] += 1
        bucket["layer_norm_sums"] = [
            acc + value for acc, value in zip(bucket["layer_norm_sums"], layer_norms)
        ]
        bucket["final_logits_diff_sum"] = (
            final_logits_diff.clone()
            if bucket["final_logits_diff_sum"] is None
            else bucket["final_logits_diff_sum"] + final_logits_diff
        )
        for key, value in family_scores.items():
            bucket["family_score_sums"][key] += value

    prompt_type_summary = {}
    for prompt_type, bucket in per_type.items():
        count = max(bucket["count"], 1)
        mean_logits_diff = bucket["final_logits_diff_sum"] / count
        mean_family_scores = {
            key: value / count for key, value in bucket["family_score_sums"].items()
        }
        prompt_type_summary[prompt_type] = {
            "count": bucket["count"],
            "layer_summary": [
                {
                    "layer": layer_idx,
                    "mean_diff_norm": bucket["layer_norm_sums"][layer_idx] / count,
                }
                for layer_idx in range(num_layers)
            ],
            "final_layer_mean_diff_norm": bucket["layer_norm_sums"][-1] / count,
            "mean_family_scores": mean_family_scores,
            "replacement_minus_deprecated_score": mean_family_scores.get("replacement", 0.0)
            - mean_family_scores.get("deprecated", 0.0),
            "top_amplified_tokens": _top_tokens(tokenizer, mean_logits_diff, k=top_k, largest=True),
            "top_suppressed_tokens": _top_tokens(tokenizer, mean_logits_diff, k=top_k, largest=False),
        }
    return {
        "prompts": [prompt_to_dict(prompt) for prompt in prompts],
        "prompt_rows": prompt_rows,
        "prompt_type_summary": prompt_type_summary,
    }


def build_adl_rows(model_key: str, run_summary: Dict) -> List[Dict]:
    rows: List[Dict] = []
    model_label = model_label_for(model_key)
    group_labels = build_group_labels(model_key)
    prompt_counts = run_summary.get("prompt_counts") or {}

    for prompt_type, count in prompt_counts.items():
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "variant": "base",
                "prompt_type": prompt_type,
                "samples": int(count),
                "final_layer_mean_diff_norm": 0.0,
                "replacement_family_score": 0.0,
                "deprecated_family_score": 0.0,
                "library_version_family_score": 0.0,
                "replacement_minus_deprecated_score": 0.0,
                **group_labels,
            }
        )

    for variant, payload in (run_summary.get("variants") or {}).items():
        if variant == "base":
            continue
        summaries = payload.get("prompt_type_summary") or {}
        for prompt_type, summary in summaries.items():
            family_scores = summary.get("mean_family_scores") or {}
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "variant": variant,
                    "prompt_type": prompt_type,
                    "samples": int(summary.get("count", 0)),
                    "final_layer_mean_diff_norm": float(summary.get("final_layer_mean_diff_norm", 0.0)),
                    "replacement_family_score": float(family_scores.get("replacement", 0.0)),
                    "deprecated_family_score": float(family_scores.get("deprecated", 0.0)),
                    "library_version_family_score": float(family_scores.get("library_version", 0.0)),
                    "replacement_minus_deprecated_score": float(
                        summary.get("replacement_minus_deprecated_score", 0.0)
                    ),
                    **group_labels,
                }
            )
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    positives = [max(float(value), 1e-12) for value in values]
    return math.exp(sum(math.log(value) for value in positives) / len(positives))
