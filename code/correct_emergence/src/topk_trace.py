"""Step 1: layer-wise top-k trajectory extraction for "correct" examples
(the subset Step 0 selected — base model already predicts the ground-truth
API span correctly).

For each example we run ONE teacher-forced forward pass through
`decision_prefix + ground_truth_form`, then for every decoder layer project the
hidden state at each span position into vocabulary space (logit lens, and
optionally tuned lens) and record:
  - the correct token's own rank/prob/logprob ("does the answer climb the
    rankings layer by layer")
  - the top-k candidates at that layer/position (the "competition" the correct
    answer has to win)
  - JSD between this layer's distribution and the final layer's distribution
    (a convergence-speed proxy, cf. lens_analysis' decision_jsd trace)

From the per-layer rank trajectory we derive two headline indicators per
position: `emergence_layer` (first layer where the correct token enters top-k)
and `saturation_layer` (first layer from which the correct token stays rank-1
through to the final layer).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
MECH_SRC_DIR = SRC_DIR.parent.parent / "06_mechanism" / "src"
if str(MECH_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MECH_SRC_DIR))

from lens_analysis import (  # noqa: E402
    LowRankTunedLens,
    gather_token_metrics,
    jsd_from_logits,
    project_hidden_to_logits,
    run_hidden_forward,
)


# ---------------------------------------------------------------------------
# Per-example layer-wise top-k trace
# ---------------------------------------------------------------------------


def layerwise_topk_trace(
    *,
    model,
    tokenizer,
    final_norm,
    output_projection,
    input_ids: Sequence[int],
    suffix_ids: Sequence[int],
    predict_positions: Sequence[int],
    top_k: int,
    tuned_lens: Optional[LowRankTunedLens] = None,
) -> List[List[Dict]]:
    """Run one forward pass and return, for each span position, a list of
    per-layer dicts: `{layer, ground_truth_rank/prob/logprob, jsd_to_final,
    top_candidates: [{rank, token_id, token_text, logit, prob, logprob}, ...]}`.

    Returned as `position_traces[position_index][layer_index]`.
    """
    forward = run_hidden_forward(model, input_ids)
    hidden_states = forward["hidden_states"]
    device = hidden_states[-1].device

    token_tensor = torch.tensor(list(suffix_ids), dtype=torch.long, device=device)
    positions = torch.tensor(list(predict_positions), dtype=torch.long, device=device)
    num_positions = len(predict_positions)
    num_layers = len(hidden_states) - 1

    final_hidden = hidden_states[-1][0, positions, :]
    final_logits = project_hidden_to_logits(final_hidden, final_norm, output_projection)

    position_traces: List[List[Dict]] = [[] for _ in range(num_positions)]
    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx + 1][0, positions, :]
        if tuned_lens is not None:
            hidden = tuned_lens.forward_layer(layer_idx, hidden)
        logits = project_hidden_to_logits(hidden, final_norm, output_projection)

        metrics = gather_token_metrics(logits, token_tensor)
        jsd = jsd_from_logits(logits, final_logits)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        top_logits, top_ids = logits.topk(top_k, dim=-1)
        top_logprobs = log_probs.gather(dim=-1, index=top_ids)
        top_probs = top_logprobs.exp()

        for pos_idx in range(num_positions):
            candidates = []
            for rank_idx in range(top_k):
                token_id = int(top_ids[pos_idx, rank_idx].item())
                candidates.append(
                    {
                        "rank": rank_idx + 1,
                        "token_id": token_id,
                        "token_text": tokenizer.decode([token_id]),
                        "logit": float(top_logits[pos_idx, rank_idx].item()),
                        "logprob": float(top_logprobs[pos_idx, rank_idx].item()),
                        "prob": float(top_probs[pos_idx, rank_idx].item()),
                    }
                )
            position_traces[pos_idx].append(
                {
                    "layer": layer_idx,
                    "ground_truth_rank": float(metrics["ranks"][pos_idx].item()),
                    "ground_truth_prob": float(metrics["probs"][pos_idx].item()),
                    "ground_truth_logprob": float(metrics["log_probs"][pos_idx].item()),
                    "jsd_to_final": float(jsd[pos_idx].item()),
                    "top_candidates": candidates,
                }
            )
    return position_traces


# ---------------------------------------------------------------------------
# Emergence / saturation indicators
# ---------------------------------------------------------------------------


def emergence_layer(layerwise: Sequence[Dict], k: int) -> Optional[int]:
    """First layer at which the ground-truth token's rank is <= k, or None if
    it never enters the top-k within the traced layers."""
    for entry in layerwise:
        if entry["ground_truth_rank"] <= k:
            return entry["layer"]
    return None


def saturation_layer(layerwise: Sequence[Dict]) -> Optional[int]:
    """First layer from which the ground-truth token stays rank-1 all the way
    to the final traced layer, or None if it is never rank-1 at the end."""
    if not layerwise or layerwise[-1]["ground_truth_rank"] != 1:
        return None
    candidate = layerwise[-1]["layer"]
    for entry in reversed(layerwise):
        if entry["ground_truth_rank"] != 1:
            break
        candidate = entry["layer"]
    return candidate


# ---------------------------------------------------------------------------
# Lightweight competitor-token classification (for the "competitor profile"
# aggregate metric — a coarse first cut: punctuation/syntax vs. identifier-like
# vs. numeric vs. whitespace, not a full API-name lookup).
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"^[\s]*[^\w\s]+[\s]*$")
_NUMERIC_RE = re.compile(r"^[\s]*[0-9]+[\s]*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def classify_token_text(token_text: str) -> str:
    stripped = token_text.strip()
    if not stripped:
        return "whitespace"
    if _NUMERIC_RE.match(token_text):
        return "numeric"
    if _PUNCT_RE.match(token_text):
        return "punctuation"
    if _IDENTIFIER_RE.match(stripped):
        return "identifier_fragment"
    return "other"


def collect_pre_emergence_competitors(
    layerwise: Sequence[Dict],
    *,
    ground_truth_token_id: int,
    emergence_layer_idx: Optional[int],
) -> List[Dict]:
    """Tally non-ground-truth tokens that appear in the top-k at any layer
    strictly before `emergence_layer_idx` (or across all traced layers if the
    token never emerges). Returns entries sorted by appearance frequency."""
    cutoff = emergence_layer_idx if emergence_layer_idx is not None else (layerwise[-1]["layer"] + 1 if layerwise else 0)
    tally: Dict[int, Dict] = {}
    for entry in layerwise:
        if entry["layer"] >= cutoff:
            continue
        for cand in entry["top_candidates"]:
            if cand["token_id"] == ground_truth_token_id:
                continue
            bucket = tally.setdefault(
                cand["token_id"],
                {
                    "token_id": cand["token_id"],
                    "token_text": cand["token_text"],
                    "token_class": classify_token_text(cand["token_text"]),
                    "appearances": 0,
                    "best_rank": cand["rank"],
                },
            )
            bucket["appearances"] += 1
            bucket["best_rank"] = min(bucket["best_rank"], cand["rank"])
    return sorted(tally.values(), key=lambda item: (-item["appearances"], item["best_rank"]))


__all__ = [
    "layerwise_topk_trace",
    "emergence_layer",
    "saturation_layer",
    "classify_token_text",
    "collect_pre_emergence_competitors",
]
