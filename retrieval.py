"""BM25 over the shared corpus, with deterministic ranking.

Rankings order by descending score and break ties by ``doc_id`` in lexicographic order,
so a tie at any cutoff always resolves the same way. Only documents that match at least
one query term are returned. ``deterministic_top_k`` and the id tie-break are adapted
from the FiQA project.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import data
from normalize import norm_v1


def id_tiebreak_rank(doc_ids: Sequence[str]) -> np.ndarray:
    """Position of each document id in string order, used to break score ties."""
    ids = np.asarray(doc_ids, dtype=str)
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(ids, kind="stable")] = np.arange(len(ids))
    return rank


def deterministic_top_k(scores: np.ndarray, k: int, tiebreak: np.ndarray,
                        candidates: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices of the ``k`` best entries by (descending score, ascending id rank)."""
    scores = np.asarray(scores, dtype=np.float64)
    pool = np.arange(len(scores)) if candidates is None else np.asarray(candidates)
    if len(pool) == 0:
        return pool
    values = scores[pool]
    if len(pool) > k:
        # keep everything at least as good as the k-th score (ties included), then order
        threshold = -np.partition(-values, k - 1)[k - 1]
        keep = values >= threshold
        pool, values = pool[keep], values[keep]
    order = np.lexsort((tiebreak[pool], -values))
    return pool[order[:k]]


@dataclass(frozen=True)
class BM25Config:
    stem: bool = True
    remove_stopwords: bool = True
    k1: float = 1.5
    b: float = 0.75
    normalize: bool = False      # norm_v1 on corpus text and queries (Phase 2a, C1)
    title_repeats: int = 1       # copies of the title in the index text (Phase 2a, C2: 2)

    @property
    def name(self) -> str:
        base = (f"bm25_{'stem' if self.stem else 'nostem'}_{'sw' if self.remove_stopwords else 'nosw'}"
                f"_k{self.k1}_b{self.b}")
        return base + ("_norm" if self.normalize else "") + (f"_title{self.title_repeats}" if self.title_repeats != 1 else "")

    def index_text(self, doc: data.Doc) -> str:
        """The text that is tokenized and indexed for ``doc``."""
        text = ". ".join([doc.title] * self.title_repeats + [doc.text])
        return norm_v1(text) if self.normalize else text

    def query_text(self, question: str) -> str:
        return norm_v1(question) if self.normalize else question


# The preregistered candidates in preference order (PREREGISTRATION.md §1.3).
CANDIDATES: Tuple[BM25Config, ...] = tuple(
    BM25Config(stem, sw, k1, b)
    for stem, sw in ((True, True), (True, False), (False, True), (False, False))
    for k1, b in ((1.5, 0.75), (1.2, 0.75), (0.9, 0.4)))


# Pinned from the tune grid (``run_eval.py grid --split tune``, 2026-09-29): best score
# 0.4807, plateau of 4 within 0.02; the first plateau member in preference order.
SELECTED: Optional[BM25Config] = BM25Config(stem=True, remove_stopwords=True, k1=1.5, b=0.75)


class BM25Retriever:
    """``bm25s`` (method "lucene") over ``Doc.index_text`` of the whole corpus."""

    def __init__(self, config: BM25Config = BM25Config(), docs: Optional[Sequence[data.Doc]] = None):
        import bm25s
        self.config = config
        docs = list(docs) if docs is not None else list(data.corpus().values())
        self.doc_ids = np.asarray([d.doc_id for d in docs])
        self._tiebreak = id_tiebreak_rank(self.doc_ids)
        self._stemmer = self._make_stemmer(config.stem)
        t0 = time.perf_counter()
        self._docs = {d.doc_id: d for d in docs}
        self._position = {d: i for i, d in enumerate(self.doc_ids)}
        tokens = bm25s.tokenize([config.index_text(d) for d in docs], stopwords=self._stopwords,
                                stemmer=self._stemmer, show_progress=False)
        self.index = bm25s.BM25(k1=config.k1, b=config.b, method="lucene")
        self.index.index(tokens, show_progress=False)
        self.build_seconds = time.perf_counter() - t0
        self.vocabulary_size = len(tokens.vocab)

    @property
    def _stopwords(self):
        return "en" if self.config.remove_stopwords else None

    @staticmethod
    def _make_stemmer(stem: bool):
        if not stem:
            return None
        import Stemmer
        return Stemmer.Stemmer("english")

    def describe(self) -> Dict[str, object]:
        from importlib.metadata import version
        return {"retriever": "bm25", "library": f"bm25s {version('bm25s')}", "method": "lucene",
                "stemmer": f"PyStemmer {version('PyStemmer')} english" if self.config.stem else None,
                "index_text": ". ".join(["title"] * self.config.title_repeats + ["text"])
                + (" (norm_v1)" if self.config.normalize else ""), "tie_break": "doc_id lexicographic",
                "only_matched_documents": True, **asdict(self.config)}

    def _tokens(self, text: str) -> List[str]:
        import bm25s
        return bm25s.tokenize([text], stopwords=self._stopwords, stemmer=self._stemmer,
                              return_ids=False, show_progress=False)[0]

    def query_terms(self, text: str) -> List[str]:
        """The question's tokens under this index's tokenizer that are in its vocabulary
        (duplicates kept)."""
        vocab = self.index.vocab_dict
        return [t for t in self._tokens(self.config.query_text(text)) if t in vocab]

    def doc_tokens(self, doc_id: str) -> List[str]:
        """``doc_id``'s tokens exactly as indexed (index text, normalization, stop words,
        stemming); used for feedback term counts and document lengths."""
        return self._tokens(self.config.index_text(self._docs[doc_id]))

    def term_scores(self, terms: Sequence[str]) -> np.ndarray:
        """BM25 score of every document for the given in-vocabulary query terms."""
        if not terms:
            return np.zeros(len(self.doc_ids), dtype=np.float64)
        return np.asarray(self.index.get_scores(list(terms)), dtype=np.float64)

    def scores(self, text: str) -> np.ndarray:
        """First-round BM25 score of every document for a question."""
        return self.term_scores(self.query_terms(text))

    def score_of(self, scores: np.ndarray):
        """doc_id -> score lookup over a score vector from :meth:`scores`."""
        return lambda doc_id: float(scores[self._position[doc_id]])

    def rank(self, scores: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        """Top ``k`` documents with a positive score, deterministic under ties."""
        matched = np.flatnonzero(scores > 0)
        top = deterministic_top_k(scores, k, self._tiebreak, candidates=matched)
        return [(str(self.doc_ids[i]), float(scores[i])) for i in top]

    def search(self, text: str, k: int = 100) -> List[Tuple[str, float]]:
        """Top ``k`` matched documents for one query, as ``(doc_id, score)``."""
        terms = self.query_terms(text)
        if not terms:
            return []
        return self.rank(self.term_scores(terms), k)
