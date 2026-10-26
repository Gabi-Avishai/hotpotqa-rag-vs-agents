"""BM25 wrapper and deterministic ranking."""
import numpy as np
import pytest

import data
import retrieval


def test_deterministic_top_k_breaks_ties_by_id():
    ids = np.array(["c", "a", "b", "d"])
    tb = retrieval.id_tiebreak_rank(ids)
    scores = np.array([1.0, 1.0, 1.0, 2.0])
    assert list(ids[retrieval.deterministic_top_k(scores, 3, tb)]) == ["d", "a", "b"]
    assert list(ids[retrieval.deterministic_top_k(scores, 2, tb, candidates=np.array([0, 2]))]) == ["b", "c"]
    assert len(retrieval.deterministic_top_k(scores, 2, tb, candidates=np.array([], dtype=int))) == 0


def test_candidates_are_the_preregistered_twelve_in_order():
    c = retrieval.CANDIDATES
    assert len(c) == 12 and len(set(c)) == 12
    assert c[0] == retrieval.BM25Config(True, True, 1.5, 0.75)
    assert c[2] == retrieval.BM25Config(True, True, 0.9, 0.4)
    assert c[3] == retrieval.BM25Config(True, False, 1.5, 0.75)
    assert c[6] == retrieval.BM25Config(False, True, 1.5, 0.75)
    assert c[11] == retrieval.BM25Config(False, False, 0.9, 0.4)


def _docs():
    mk = lambda i, t, s: data.Doc(i, t, data.text_hash(s), tuple(s))
    return [mk("Alpha", "Alpha", ["Alpha is a river in France."]),
            mk("Beta", "Beta", ["Beta was directed by David Cronenberg."]),
            mk("Gamma", "Gamma", ["Gamma is a small town."]),
            mk("Delta", "Delta", ["The river flows north."])]


def test_bm25_ranks_matched_documents_only_and_deterministically():
    r = retrieval.BM25Retriever(retrieval.BM25Config(), docs=_docs())
    hits = r.search("Which river is in France?", k=10)
    assert [d for d, _ in hits][0] == "Alpha"
    assert "Gamma" not in [d for d, _ in hits]            # no shared term: not returned
    assert r.search("Which river is in France?", k=10) == hits
    assert r.search("zzz qqq", k=5) == []


def test_stemming_and_stopwords_change_query_terms():
    stem = retrieval.BM25Retriever(retrieval.BM25Config(True, True), docs=_docs())
    plain = retrieval.BM25Retriever(retrieval.BM25Config(False, False), docs=_docs())
    assert "direct" in stem.query_terms("who directed it")
    assert "directed" in plain.query_terms("who directed it")
    assert "is" not in stem.query_terms("what is Alpha") and "is" in plain.query_terms("what is Alpha")


def test_describe_records_configuration():
    d = retrieval.BM25Retriever(retrieval.BM25Config(True, False, 1.2, 0.75), docs=_docs()).describe()
    assert d["method"] == "lucene" and d["k1"] == 1.2 and d["remove_stopwords"] is False
