"""Cross-encoder reranking, candidate D4 (PREREGISTRATION.md §2.7).

``cross-encoder/ms-marco-MiniLM-L6-v2`` at a pinned revision reads each (question,
``title. text``) pair, truncated to 512 tokens, in batches of 32. The top ``pool`` of a
base ranking (D2's top 50) are ordered by logit, ties by ``doc_id``; the reranked top 10
is followed by the base ranking's order, skipping placed documents, to depth 100.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

import data

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_LENGTH = 512
BATCH = 32
POOL = 50


def load_model(device: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL, revision=REVISION, max_length=MAX_LENGTH, device=device)


def logits(model, question: str, docs: Sequence[data.Doc]) -> np.ndarray:
    pairs = [(question, f"{d.title}. {d.text}") for d in docs]
    return np.asarray(model.predict(pairs, batch_size=BATCH, show_progress_bar=False,
                                    activation_fn=None), dtype=np.float64)


def order(doc_ids: Sequence[str], scores: Sequence[float]) -> List[str]:
    """Documents by descending logit, ties by doc_id."""
    return [d for _, d in sorted(zip((-float(s) for s in scores), doc_ids))]


def rerank(model, question: str, base: Sequence[str], pool: int = POOL, top: int = 10,
           depth: int = 100) -> Tuple[List[str], List[float]]:
    corpus = data.corpus()
    cand = list(base[:pool])
    s = logits(model, question, [corpus[d] for d in cand])
    head = order(cand, s)[:top]
    placed = set(head)
    tail = [d for d in base if d not in placed]
    return (head + tail)[:depth], [float(x) for x in s]
