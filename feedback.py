"""BM25-weighted RM3-style feedback, candidate C3 (PREREGISTRATION.md §2.3).

It follows RM3's structure, but the document weights are normalized first-round BM25
scores instead of query likelihoods:

    pi(d)   = s(d) / sum_{d' in F} s(d')                    F = base top-n
    P_R(t)  = sum_{d in F} pi(d) * tf(t, d) / |d|           top m terms, renormalized
    P_Q(t)  = c(t, Q) / |Q|                                 in-vocabulary question tokens
    w(t)    = lam * P_Q(t) + (1 - lam) * P_R(t)
    score2(d) = sum_t w(t) * BM25_B(t, d)

tf and |d| come from the document's indexed representation in the base index. The
second-round ranking replaces the first. If the question has no in-vocabulary term, F is
empty, or the feedback scores sum to zero, the base ranking is returned unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FeedbackConfig:
    n_docs: int
    n_terms: int
    lam: float = 0.5

    @property
    def name(self) -> str:
        return f"rm3style_n{self.n_docs}_m{self.n_terms}_lam{self.lam}"


C3A = FeedbackConfig(3, 10)
C3B = FeedbackConfig(5, 20)


def query_weights(retriever, question: str, cfg: FeedbackConfig) -> Tuple[Dict[str, float], Dict]:
    """Second-round term weights, and diagnostics. Returns ({}, info) on a fallback."""
    terms = retriever.query_terms(question)
    info = {"fallback": None, "expansion_terms": []}
    if not terms:
        info["fallback"] = "no_query_terms"
        return {}, info
    first = retriever.search(question, k=cfg.n_docs)
    total = sum(s for _, s in first)
    if not first or total <= 0:
        info["fallback"] = "no_feedback_scores"
        return {}, info
    p_r: Dict[str, float] = {}
    for doc_id, s in first:
        toks = retriever.doc_tokens(doc_id)
        if not toks:
            continue
        pi = s / total
        for t, c in Counter(toks).items():
            p_r[t] = p_r.get(t, 0.0) + pi * c / len(toks)
    top = sorted(p_r.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg.n_terms]
    z = sum(v for _, v in top)
    p_r = {t: v / z for t, v in top}
    q_counts = Counter(terms)
    p_q = {t: c / len(terms) for t, c in q_counts.items()}
    weights = {t: cfg.lam * p_q.get(t, 0.0) + (1 - cfg.lam) * p_r.get(t, 0.0) for t in set(p_q) | set(p_r)}
    info["expansion_terms"] = [t for t, _ in top]
    return weights, info


def rank(retriever, question: str, cfg: FeedbackConfig, depth: int = 100) -> Tuple[List[str], Dict]:
    """C3's depth-``depth`` ranking and diagnostics."""
    weights, info = query_weights(retriever, question, cfg)
    if not weights:
        return [d for d, _ in retriever.search(question, k=depth)], info
    scores = np.zeros(len(retriever.doc_ids), dtype=np.float64)
    for t in sorted(weights):
        scores += weights[t] * retriever.term_scores([t])
    return [d for d, _ in retriever.rank(scores, depth)], info
