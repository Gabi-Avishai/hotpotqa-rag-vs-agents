"""Phase 2b: RRF, TAS-B dense retrieval, cross-encoder reranking, D3 composition, refusal
and manifests."""
import json

import numpy as np
import pytest

import data
import dense
import fusion
import graph
import rerank
import run_eval


def test_rrf_hand_computed_and_deterministic():
    # a: 1/61; b: 1/62 + 1/61; c: 1/63; d: 1/62
    assert fusion.rrf([["a", "b", "c"], ["b", "d"]]) == ["b", "a", "d", "c"]
    assert fusion.rrf([["y"], ["x"]]) == ["x", "y"]                      # equal scores: doc_id order
    assert fusion.rrf([["a", "b"], ["c"]], pool=1) == ["a", "c"]         # only each ranking's top `pool`
    assert fusion.rrf([["a", "b", "c"], ["b", "d"]]) == fusion.rrf([["a", "b", "c"], ["b", "d"]])


@pytest.fixture(scope="module")
def tasb():
    return dense.load_model("cpu")


def test_tasb_vectors_are_768_float32_and_unnormalized(tasb):
    v = dense.encode(tasb, ["What is the capital of France?", "Paris. Paris is the capital of France."])
    assert v.shape == (2, 768) and v.dtype == np.float32
    assert not np.allclose(np.linalg.norm(v, axis=1), 1.0)           # no normalization
    assert dense.paragraph_text(data.Doc("Paris", "Paris", "h", (" Capital of France.",))) == "Paris. Capital of France."


class StubModel:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def encode(self, texts, **kw):
        return np.stack([self.vec for _ in texts])


def test_dense_search_is_dot_product_with_id_tiebreak():
    m = np.array([[1, 0], [2, 0], [0, 5], [2, 0]], dtype=np.float32)       # "b" and "d" tie
    r = dense.DenseRetriever(["a", "b", "c", "d"], m, query_model=StubModel([1, 0]))
    assert r.search("q", k=4) == [("b", 2.0), ("d", 2.0), ("a", 1.0), ("c", 0.0)]
    r2 = dense.DenseRetriever(["a", "b", "c", "d"], m, query_model=StubModel([0, 1]))
    assert r2.search("q", k=1) == [("c", 5.0)]                            # dot product, not cosine


def test_dense_cache_key_covers_every_setting():
    key = dense.cache_key("cuda")
    assert {"model", "revision", "corpus_fingerprint", "text_template", "max_length", "dtype", "device",
            "normalize", "pooling"} <= set(key)
    assert key["revision"] == "b12d9352e776979147078a8975a4885042984fd1" and key["normalize"] is False


class StubCE:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, **kw):
        return np.array([self.scores[p[1].split(".")[0]] for p in pairs])


def test_cross_encoder_ordering_ties_and_tail(monkeypatch):
    corpus = {t: data.Doc(t, t, "h", ("x.",)) for t in "abcdefghijkl"}
    monkeypatch.setattr(data, "corpus", lambda: corpus)
    base = list("abcdefghijkl")
    scores = {c: 0.0 for c in base} | {"c": 3.0, "e": 5.0, "b": 3.0}
    ranking, _ = rerank.rerank(StubCE(scores), "q", base, pool=6, top=3)
    assert ranking[:3] == ["e", "b", "c"]                                  # logit, then doc_id on the tie
    assert ranking[3:] == ["a", "d", "f", "g", "h", "i", "j", "k", "l"]    # base order, placed skipped
    assert rerank.order(["x", "w"], [1.0, 1.0]) == ["w", "x"]


def test_cross_encoder_scores_relevance(tmp_path):
    m = rerank.load_model("cpu")
    docs = [data.Doc("Oslo", "Oslo", "h", ("Oslo is in Norway.",)), data.Doc("Paris", "Paris", "h", ("Paris is the capital of France.",))]
    s = rerank.logits(m, "What is the capital of France?", docs)
    assert rerank.order(["Oslo", "Paris"], s) == ["Paris", "Oslo"]


def test_d3_is_the_frozen_operator_on_rrf_of_b_and_dense():
    class G:
        def targets(self, d):
            return [(0, 0, "x1")] if d == "b1" else []

        def sources(self, d):
            return ["s1", "s2"] if d == "d1" else []

        def in_degree(self, d):
            return 0
    b_rank = ["b1", "b2", "b3"]
    d1_rank = ["d1", "d2", "d3"]
    lex = {"s1": 1.0, "s2": 4.0}
    ranking, added, h = run_eval.dense_seeded_graph(b_rank, d1_rank, G(), lambda d: lex.get(d, 0.0))
    assert h == fusion.rrf([b_rank, d1_rank]) and ranking[:5] == h[:5]
    assert [a["doc"] for a in added] == ["x1", "s2", "s1"]                  # forward from b1; backward by B score
    assert run_eval.G_STAR == graph.C5_5


def test_phase2b_refuses_other_splits():
    for split in ("test_retrieval", "test_agents", "dev"):
        with pytest.raises(SystemExit):
            run_eval.main(["phase2b", "--split", split])


def test_phase2b_manifest_config_carries_models_and_fingerprints():
    g = graph.build([data.Doc("A", "A", "h", ("x.",))])
    info = {"key": dense.cache_key("cuda"), "fingerprint": "f" * 64, "shape": [66635, 768]}
    cfg = run_eval.phase2b_config(info, g)
    assert cfg["dense"]["revision"] == dense.REVISION and cfg["dense"]["embedding_fingerprint"] == "f" * 64
    assert cfg["rerank"]["revision"] == rerank.REVISION and cfg["rerank"]["pool"] == 50
    assert cfg["rrf"] == {"c": 60, "pool": 100} and cfg["incumbent"]["graph_fingerprint"] == g.fingerprint()
    assert "build_s" not in json.dumps(cfg) and "encode_s" not in json.dumps(cfg)


def test_d3_retriever_matches_the_operator_composition():
    import hybrid

    class B:
        def search(self, q, k=100):
            return [("b1", 3.0), ("b2", 2.0), ("b3", 1.0)][:k]

        def scores(self, q):
            return {"s": 1.0}

        def score_of(self, s):
            return lambda d: s.get(d, 0.0)

        def describe(self):
            return {}

    class D:
        def search(self, q, k=100):
            return [("d1", 9.0), ("b2", 8.0)][:k]

    class G:
        def targets(self, d):
            return [(0, 0, "x")] if d == "d1" else []

        def sources(self, d):
            return ["s"] if d == "b2" else []

        def in_degree(self, d):
            return 1

        def fingerprint(self):
            return "fp"

    r = hybrid.D3Retriever(B(), D(), G())
    got = [d for d, _ in r.search("q", k=10)]
    want, _, _ = run_eval.dense_seeded_graph(["b1", "b2", "b3"], ["d1", "b2"], G(), lambda d: {"s": 1.0}.get(d, 0.0))
    assert got == want[:10]
