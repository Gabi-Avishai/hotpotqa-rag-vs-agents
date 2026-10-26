"""Post hoc P1: X1 = cross-encoder reranking of D3."""
import numpy as np
import pytest

import data
import rerank
import run_eval


class RecordingCE:
    """Stand-in cross-encoder: scores by a fixed map and records what it was asked."""

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def predict(self, pairs, **kw):
        self.seen = [p[1].split(".")[0] for p in pairs]
        return np.array([self.scores.get(t, 0.0) for t in self.seen])


@pytest.fixture
def corpus(monkeypatch):
    docs = {f"d{i:03d}": data.Doc(f"d{i:03d}", f"d{i:03d}", "h", ("x.",)) for i in range(120)}
    monkeypatch.setattr(data, "corpus", lambda: docs)
    return docs


def test_x1_reranks_exactly_the_first_50_then_keeps_d3_order(corpus):
    d3 = [f"d{i:03d}" for i in range(100)]
    ce = RecordingCE({"d049": 9.0, "d030": 8.0, "d007": 7.0})
    ranking, _ = run_eval.x1_rerank(ce, "q", d3)
    assert ce.seen == d3[:50]                                   # exactly the first 50, nothing beyond
    head = ranking[:10]
    assert head[:3] == ["d049", "d030", "d007"]
    assert head[3:] == ["d000", "d001", "d002", "d003", "d004", "d005", "d006"]   # zero-logit ties by doc_id
    assert ranking[10:] == [d for d in d3 if d not in set(head)]                   # D3's order, no duplicates
    assert len(ranking) == len(set(ranking)) == 100
    assert "d050" not in head                                    # beyond the pool never enters the top 10


def test_x1_config_is_marked_posthoc_and_pins_its_inputs():
    cfg = run_eval.x1_config("f" * 64)
    assert cfg["posthoc"] is True and cfg["d3_rankings_fingerprint"] == "f" * 64
    assert cfg["rerank"]["revision"] == rerank.REVISION and cfg["rerank"]["pool"] == 50 == rerank.POOL
    assert cfg["rerank"]["max_length"] == 512 and cfg["rerank"]["batch"] == 32


def test_rankings_fingerprint_is_canonical():
    a = run_eval.rankings_fingerprint({"q1": ["a", "b"], "q2": ["c"]})
    assert a == run_eval.rankings_fingerprint({"q2": ["c"], "q1": ["a", "b"]})
    assert a != run_eval.rankings_fingerprint({"q1": ["b", "a"], "q2": ["c"]})


def test_posthoc_refuses_other_splits():
    for split in ("dev", "test_retrieval"):
        with pytest.raises(SystemExit):
            run_eval.main(["posthoc-x1", "--split", split])
