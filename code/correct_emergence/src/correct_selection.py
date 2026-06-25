"""Step 0: locate samples where the base model itself predicts the ground-truth
API call correctly (first-token and/or full-span), across two input sources:

1. repair/consistency test sets (`05_positive_engineering/data/processed_clean`)
2. EDAPIBench (`adalora/edapibench-src/EDAPI-Bench-main/data/EditDeprecatedAPI`)

Unlike `06_mechanism/src/lens_analysis.py`'s `FocusExample` (which is built around
the deprecated-vs-replacement *fork point*), `CorrectExample` only needs a single
ground-truth API span: "correct" here means "matches the dataset's expected
answer", regardless of whether that answer happens to be the deprecated or the
replacement form. We therefore reuse the alias-matching primitives from
`lens_analysis` but define a leaner example/record shape.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
# NOTE: both `06_mechanism` and `07_correct_emergence` ship a package literally
# named `src`; adding `06_mechanism` (the parent) to sys.path would shadow this
# module's own `src` package. Add `06_mechanism/src` itself instead and import
# `lens_analysis` as a top-level module to sidestep the name collision.
MECH_SRC_DIR = SRC_DIR.parent.parent / "06_mechanism" / "src"
if str(MECH_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MECH_SRC_DIR))

from lens_analysis import first_alias_hit  # noqa: E402


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectExample:
    row_id: str
    source: str  # repair | consistency | edapibench
    library: str
    category: str
    task_family: str
    sample_type: str
    prompt_field: str
    prompt_text: str
    completion_prefix: str
    decision_prefix: str
    ground_truth_api: str
    ground_truth_form: str
    source_file: str


def example_to_dict(example: CorrectExample) -> Dict:
    return asdict(example)


def _build_from_generic_row(
    row: Dict,
    *,
    source: str,
    prompt_field: str,
    prompt_key: str,
    target_keys: Sequence[str],
    api_key: str,
    expected_form_key: Optional[str] = None,
    id_key: str = "id",
    library_key: str = "library",
    category_key: str = "category",
    task_family_key: Optional[str] = "task_family",
    sample_type_key: Optional[str] = "sample_type",
    source_file_key: Optional[str] = "source_file",
) -> Optional[CorrectExample]:
    def optional_field(key: Optional[str], default: str) -> str:
        if key is None:
            return default
        return str(row.get(key) or default)

    prompt_text = str(row.get(prompt_key) or "")
    target = ""
    for key in target_keys:
        target = str(row.get(key) or "")
        if target:
            break
    ground_truth_api = str(row.get(api_key) or "").strip()
    if not prompt_text or not target or not ground_truth_api:
        return None

    ground_truth_form = None
    if expected_form_key:
        candidate = str(row.get(expected_form_key) or "").strip()
        if candidate and candidate in target:
            ground_truth_form = candidate
    if ground_truth_form is None:
        ground_truth_form = first_alias_hit(target, ground_truth_api)
    if not ground_truth_form:
        return None

    span_start = target.find(ground_truth_form)
    if span_start < 0:
        return None

    completion_prefix = target[:span_start]
    decision_prefix = prompt_text + completion_prefix

    return CorrectExample(
        row_id=str(row.get(id_key) or ""),
        source=source,
        library=str(row.get(library_key) or "unknown"),
        category=str(row.get(category_key) or "unknown"),
        task_family=optional_field(task_family_key, source),
        sample_type=optional_field(sample_type_key, "unknown"),
        prompt_field=prompt_field,
        prompt_text=prompt_text,
        completion_prefix=completion_prefix,
        decision_prefix=decision_prefix,
        ground_truth_api=ground_truth_api,
        ground_truth_form=ground_truth_form,
        source_file=optional_field(source_file_key, ""),
    )


def build_repair_consistency_example(
    row: Dict, *, source: str, prompt_field: str = "version_prompt"
) -> Optional[CorrectExample]:
    return _build_from_generic_row(
        row,
        source=source,
        prompt_field=prompt_field,
        prompt_key=prompt_field,
        target_keys=("target", "reference"),
        api_key="replacement_api",
        expected_form_key=None,
    )


def build_edapibench_example(row: Dict, *, prompt_field: str = "probing_input") -> Optional[CorrectExample]:
    return _build_from_generic_row(
        row,
        source="edapibench",
        prompt_field=prompt_field,
        prompt_key="probing input",
        target_keys=("reference", "expected call"),
        api_key="replacement api",
        expected_form_key="expected call",
        id_key="case-id",
        library_key="library",
        category_key="category",
        task_family_key=None,  # not present; falls back to `source`
        sample_type_key=None,
        source_file_key=None,
    )


def build_repair_consistency_examples(
    rows: Iterable[Dict], *, source: str, prompt_field: str = "version_prompt"
) -> List[CorrectExample]:
    examples = []
    for row in rows:
        example = build_repair_consistency_example(row, source=source, prompt_field=prompt_field)
        if example is not None:
            examples.append(example)
    return examples


def build_edapibench_examples(rows: Iterable[Dict]) -> List[CorrectExample]:
    examples = []
    for row in rows:
        example = build_edapibench_example(row)
        if example is not None:
            examples.append(example)
    return examples


# ---------------------------------------------------------------------------
# Loading raw rows
# ---------------------------------------------------------------------------


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_edapibench_rows(paths: Sequence[Path]) -> List[Dict]:
    """Merge `all.json` files from EDAPIBench, de-duplicating by `case-id`.

    The benchmark ships one `all.json` per probed model (codegemma-2b /
    qwencoder-3b / deepseek-1.3b); the underlying (probing input, reference,
    expected call) triples overlap heavily across these files (same source
    cases, different probed models), so we de-dup on `case-id`/`probing input`
    to avoid triple-counting the same code context.
    """
    seen_keys = set()
    rows: List[Dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            key = (row.get("library"), row.get("probing input"), row.get("expected call"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Correctness evaluation (single teacher-forced forward pass per example)
# ---------------------------------------------------------------------------


def encode_decision_input(tokenizer, prefix_text: str, suffix_text: str, max_length: int) -> Optional[Dict]:
    """Tokenize `prefix_text + suffix_text` jointly and split it back into a
    (prefix, suffix) token pair at the boundary the *joint* tokenization implies.

    Tokenizing `prefix_text` and `suffix_text` separately and concatenating the
    ids is unsafe: BPE merges characters across the boundary (e.g. a trailing
    space on the prefix gets absorbed into the suffix's first token, producing
    `' torch'` instead of separate `' '` + `'torch'`). Comparing the model's
    prediction against the standalone-suffix tokenization would then flag a
    correct prediction (`' torch'`) as wrong (expected bare `'torch'`). We avoid
    this by locating the longest common-prefix run between the standalone-prefix
    token ids and the joint token ids — BPE is a deterministic left-to-right
    greedy process, so the two agree up to the first boundary-spanning merge —
    and treat everything from there on as the suffix to be predicted.
    """
    prefix_ids_standalone = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix_text + suffix_text, add_special_tokens=False)["input_ids"]
    if not prefix_ids_standalone or len(full_ids) <= len(prefix_ids_standalone):
        return None

    boundary = 0
    limit = min(len(prefix_ids_standalone), len(full_ids) - 1)
    while boundary < limit and prefix_ids_standalone[boundary] == full_ids[boundary]:
        boundary += 1

    prefix_ids = full_ids[:boundary]
    suffix_ids = full_ids[boundary:]
    if not prefix_ids or not suffix_ids:
        return None

    max_prefix_len = max(1, max_length - len(suffix_ids))
    prefix_ids = prefix_ids[-max_prefix_len:]
    input_ids = prefix_ids + suffix_ids
    predict_positions = [len(prefix_ids) - 1 + offset for offset in range(len(suffix_ids))]
    return {
        "input_ids": input_ids,
        "suffix_ids": suffix_ids,
        "predict_positions": predict_positions,
    }


def split_at_namespace_dot(ground_truth_form: str) -> tuple[str, str]:
    """Split `ground_truth_form` at its first '.' into (namespace_prefix, remainder).

    Dotted forms such as `"tf.random.categorical"` or `"torch.linalg.lstsq"`
    start with a library import alias (`tf`/`torch`/`F`/...) that is largely
    interchangeable across an entire codebase and not itself an "API choice" --
    the model commits to which *API* to call at the first token of the
    remainder (`"random.categorical"` / `"linalg.lstsq"`). This mirrors
    `06_mechanism/src/lens_analysis.py`'s `shared_api_decision_prefix`, which
    strips the shared namespace prefix before identifying the decision token.
    Forms without a '.' (e.g. `"GaussianMixture"`) are returned unchanged with
    an empty namespace prefix -- there is nothing to strip.
    """
    dot_idx = ground_truth_form.find(".")
    if dot_idx < 0:
        return "", ground_truth_form
    return ground_truth_form[: dot_idx + 1], ground_truth_form[dot_idx + 1 :]


def run_logits_forward(model, input_ids: Sequence[int]) -> torch.Tensor:
    tensor = torch.tensor([list(input_ids)], dtype=torch.long)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        outputs = model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            use_cache=False,
        )
    return outputs.logits[0]


def evaluate_example_correctness(
    *,
    model,
    tokenizer,
    example: CorrectExample,
    max_length: int,
) -> Optional[Dict]:
    """Teacher-force `decision_prefix + ground_truth_form` through the model and
    check, token by token, whether the model's own argmax prediction matches the
    ground-truth token at each position.

    Returns a dict with `first_token_correct` / `full_span_correct` booleans plus
    supporting evidence (matched tokens, ranks, logprobs), or None if the example
    cannot be encoded (e.g. empty prefix/suffix after tokenization).
    """
    encoded = encode_decision_input(
        tokenizer=tokenizer,
        prefix_text=example.decision_prefix,
        suffix_text=example.ground_truth_form,
        max_length=max_length,
    )
    if encoded is None:
        return None

    logits = run_logits_forward(model, encoded["input_ids"])
    suffix_ids = encoded["suffix_ids"]
    predict_positions = encoded["predict_positions"]

    predict_logits = logits[predict_positions, :]
    target_ids = torch.tensor(suffix_ids, dtype=torch.long, device=predict_logits.device)

    log_probs = F.log_softmax(predict_logits.float(), dim=-1)
    target_logprobs = log_probs.gather(dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
    target_logits = predict_logits.gather(dim=-1, index=target_ids.view(-1, 1)).squeeze(-1)
    ranks = 1 + (predict_logits > target_logits.unsqueeze(-1)).sum(dim=-1)
    argmax_ids = predict_logits.argmax(dim=-1)

    token_matches = (argmax_ids == target_ids)
    first_token_correct = bool(token_matches[0].item())
    full_span_correct = bool(token_matches.all().item())

    token_evidence = []
    for idx in range(len(suffix_ids)):
        token_evidence.append(
            {
                "position_in_span": idx,
                "ground_truth_token_id": int(target_ids[idx].item()),
                "ground_truth_token_text": tokenizer.decode([int(target_ids[idx].item())]),
                "predicted_token_id": int(argmax_ids[idx].item()),
                "predicted_token_text": tokenizer.decode([int(argmax_ids[idx].item())]),
                "match": bool(token_matches[idx].item()),
                "ground_truth_rank": float(ranks[idx].item()),
                "ground_truth_logprob": float(target_logprobs[idx].item()),
            }
        )

    return {
        "row_id": example.row_id,
        "source": example.source,
        "library": example.library,
        "category": example.category,
        "task_family": example.task_family,
        "sample_type": example.sample_type,
        "ground_truth_api": example.ground_truth_api,
        "ground_truth_form": example.ground_truth_form,
        "span_token_count": len(suffix_ids),
        "first_token_correct": first_token_correct,
        "full_span_correct": full_span_correct,
        "first_token_rank": token_evidence[0]["ground_truth_rank"],
        "first_token_logprob": token_evidence[0]["ground_truth_logprob"],
        "mean_span_logprob": float(sum(t["ground_truth_logprob"] for t in token_evidence) / len(token_evidence)),
        "token_evidence": token_evidence,
        "decision_prefix_token_count": len(encoded["input_ids"]) - len(suffix_ids),
    }


__all__ = [
    "CorrectExample",
    "example_to_dict",
    "build_repair_consistency_example",
    "build_edapibench_example",
    "build_repair_consistency_examples",
    "build_edapibench_examples",
    "load_jsonl_rows",
    "load_edapibench_rows",
    "encode_decision_input",
    "split_at_namespace_dot",
    "run_logits_forward",
    "evaluate_example_correctness",
]
