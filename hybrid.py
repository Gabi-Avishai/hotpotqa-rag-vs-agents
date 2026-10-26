"""The overall Phase 2 retriever R = D3 as a drop-in retriever for the RAG path.

D3 = G*(H), H = RRF(BM25, TAS-B), with G* the frozen C5@5 graph operator and BM25's
first-round scores ranking backward candidates (PREREGISTRATION.md §2.7). ``search``
returns the depth-100 ranking; the score is 1/rank, since D3 is rank-based.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import dense
import fusion
import graph
import retrieval


class D3Retriever:
    def __init__(self, bm25: "retrieval.BM25Retriever", dense_retriever: "dense.DenseRetriever",
                 g: "graph.TitleGraph", operator: "graph.GraphConfig" = graph.C5_5):
        self.bm25, self.dense, self.g, self.operator = bm25, dense_retriever, g, operator
        self.last_added: List[Dict] = []

    def search(self, question: str, k: int = 100) -> List[Tuple[str, float]]:
        b_rank = [d for d, _ in self.bm25.search(question, k=100)]
        lex = self.bm25.score_of(self.bm25.scores(question))
        d1 = [d for d, _ in self.dense.search(question, k=100)]
        ranking, self.last_added = graph.expand(fusion.rrf([b_rank, d1]), self.g, self.operator, lex)
        return [(d, 1.0 / r) for r, d in enumerate(ranking[:k], 1)]

    def describe(self) -> Dict[str, object]:
        return {"retriever": "D3 = G*(RRF(BM25, TAS-B))", "bm25": self.bm25.describe(),
                "dense": {"model": dense.MODEL, "revision": dense.REVISION},
                "rrf": {"c": fusion.C, "pool": fusion.POOL}, "operator": asdict(self.operator),
                "graph_fingerprint": self.g.fingerprint()}
