"""Reciprocal Rank Fusion (PREREGISTRATION.md §2.7).

score(d) = sum over rankings containing d (within their top ``pool``) of 1 / (c + rank(d)),
rank starting at 1; ties broken by ``doc_id`` in lexicographic order. Scores from
different retrievers are never added directly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

C = 60
POOL = 100


def rrf(rankings: Sequence[Sequence[str]], c: int = C, pool: int = POOL, depth: int = 100) -> List[str]:
    score: Dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking[:pool], 1):
            score[d] = score.get(d, 0.0) + 1.0 / (c + r)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:depth]
