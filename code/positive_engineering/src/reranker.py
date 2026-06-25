from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Candidate:
    text: str
    score: float
    api_name: Optional[str] = None


class VersionAwareReranker:
    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def rescore_candidate(
        self,
        candidate: Candidate,
        deprecated_apis: Iterable[str],
        replacement_api: str,
    ) -> Candidate:
        new_score = candidate.score
        api_name = candidate.api_name or candidate.text.strip()
        if replacement_api and api_name == replacement_api:
            new_score += self.alpha
        if api_name in set(deprecated_apis):
            new_score -= self.beta
        return Candidate(text=candidate.text, score=new_score, api_name=api_name)

    def rerank(
        self,
        candidates: List[Candidate],
        deprecated_apis: Iterable[str],
        replacement_api: str,
    ) -> List[Candidate]:
        rescored = [
            self.rescore_candidate(c, deprecated_apis=deprecated_apis, replacement_api=replacement_api)
            for c in candidates
        ]
        return sorted(rescored, key=lambda c: c.score, reverse=True)


def build_candidates_from_topk(topk: List[Dict]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for item in topk:
        candidates.append(
            Candidate(
                text=item.get("token_text", ""),
                score=float(item.get("logit", 0.0)),
                api_name=item.get("api_name"),
            )
        )
    return candidates
