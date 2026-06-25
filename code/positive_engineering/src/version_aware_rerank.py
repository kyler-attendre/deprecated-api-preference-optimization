from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence


IDENTIFIER_TRAIL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\.]*)$")


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    value = str(value).strip()
    return [value] if value else []


def alias_forms(api_name: str) -> List[str]:
    aliases = {api_name}
    api_name = api_name.strip()

    if api_name.startswith("torch.nn.functional."):
        aliases.add("F." + api_name.split(".")[-1])
    if api_name.startswith("tensorflow."):
        aliases.add("tf." + api_name.split(".", 1)[1])

    return sorted(x for x in aliases if x)


def expand_aliases(api_names: Iterable[str]) -> List[str]:
    aliases = set()
    for api_name in api_names:
        aliases.update(alias_forms(api_name))
    return sorted(aliases)


def extract_identifier_fragment(text: str) -> str:
    match = IDENTIFIER_TRAIL_RE.search(text)
    if not match:
        return ""
    return match.group(1)


@dataclass
class StepCandidate:
    token_id: int
    token_text: str
    base_score: float
    reranked_score: float
    fragment: str
    adjustment: float
    match_type: str


@dataclass
class RerankDecision:
    token_id: int
    token_text: str
    base_score: float
    reranked_score: float
    changed: bool
    applied: bool
    fragment: str
    base_token_text: str
    match_type: str
    top_candidates: List[StepCandidate]


class PrefixAwareVersionReranker:
    def __init__(self, *, alpha: float = 0.5, beta: float = 1.0, mode: str = "soft", top_k: int = 50):
        if mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported rerank mode: {mode}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.top_k = top_k

    def _match_type(
        self,
        *,
        fragment: str,
        deprecated_aliases: Sequence[str],
        replacement_aliases: Sequence[str],
    ) -> str:
        if not fragment:
            return "none"

        dep_matches = [alias for alias in deprecated_aliases if alias.startswith(fragment)]
        rep_matches = [alias for alias in replacement_aliases if alias.startswith(fragment)]

        if rep_matches and not dep_matches:
            return "replacement_prefix"
        if dep_matches and not rep_matches:
            return "deprecated_prefix"
        return "none"

    def _adjustment_from_match(self, match_type: str) -> float:
        if match_type == "replacement_prefix":
            return self.alpha
        if match_type == "deprecated_prefix":
            return -1e9 if self.mode == "hard" else -self.beta
        return 0.0

    def rerank_next_token(
        self,
        *,
        logits,
        tokenizer,
        generated_text: str,
        deprecated_apis: Iterable[str],
        replacement_apis: Iterable[str],
    ) -> RerankDecision:
        deprecated_aliases = expand_aliases(deprecated_apis)
        replacement_aliases = expand_aliases(replacement_apis)

        top_scores, top_ids = logits.topk(min(self.top_k, logits.shape[-1]))

        candidates: List[StepCandidate] = []
        for score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            fragment = extract_identifier_fragment(generated_text + token_text)
            match_type = self._match_type(
                fragment=fragment,
                deprecated_aliases=deprecated_aliases,
                replacement_aliases=replacement_aliases,
            )
            adjustment = self._adjustment_from_match(match_type)
            candidates.append(
                StepCandidate(
                    token_id=int(token_id),
                    token_text=token_text,
                    base_score=float(score),
                    reranked_score=float(score + adjustment),
                    fragment=fragment,
                    adjustment=float(adjustment),
                    match_type=match_type,
                )
            )

        base_best = candidates[0]
        reranked_best = max(candidates, key=lambda c: c.reranked_score)
        applied = any(c.adjustment != 0.0 for c in candidates)

        return RerankDecision(
            token_id=reranked_best.token_id,
            token_text=reranked_best.token_text,
            base_score=base_best.base_score,
            reranked_score=reranked_best.reranked_score,
            changed=reranked_best.token_id != base_best.token_id,
            applied=applied,
            fragment=reranked_best.fragment,
            base_token_text=base_best.token_text,
            match_type=reranked_best.match_type,
            top_candidates=candidates,
        )


def build_reranker_from_args(args) -> PrefixAwareVersionReranker:
    return PrefixAwareVersionReranker(
        alpha=args.alpha,
        beta=args.beta,
        mode=args.rerank_mode,
        top_k=args.top_k,
    )
