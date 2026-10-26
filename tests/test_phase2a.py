"""Phase 2a: normalization, title repetition, feedback, the title-mention graph (with
amendment A2), the expansion operator, baseline replay and manifests."""
import inspect
import json
import os

import numpy as np
import pytest

import data
import feedback
import graph
import retrieval
import run_eval
from normalize import norm_v1, tok_v1


def doc(title, *sentences, doc_id=None):
    return data.Doc(doc_id or title, title, data.text_hash(sentences), tuple(sentences))


# ---------------------------------------------------------------- normalization and title repetition

@pytest.mark.parametrize("raw,want", [
    ("Björk", "Bjork"), ("Ødegaard", "Odegaard"), ("Straße", "Strasse"), ("İstanbul", "Istanbul"),
    ("Kilikya’s bar", "Kilikya bar"), ("O'Brien", "OBrien"), ("1992–1994", "1992-1994"),
    ("U.S. and U.S.A.", "US and USA"), ("a   b\tc", "a b c"),
])
def test_norm_v1_steps(raw, want):
    assert norm_v1(raw) == want


def test_tok_v1_splits_like_paragraph_matching():
    assert tok_v1("Spider-Man") == ["spider", "man"] and tok_v1("The U.S. Open") == ["the", "us", "open"]


def test_normalization_is_symmetric_between_corpus_and_query():
    docs = [doc("Björk", "Björk is a singer from Iceland."), doc("Other", "An unrelated text.")]
    plain = retrieval.BM25Retriever(retrieval.BM25Config(), docs=docs)
    norm = retrieval.BM25Retriever(retrieval.BM25Config(normalize=True), docs=docs)
    assert plain.search("Bjork", 5) == []
    assert norm.search("Bjork", 5)[0][0] == "Björk" and norm.search("Björk", 5)[0][0] == "Björk"


def test_title_repetition_and_names():
    d = doc("Alpha Beta", "Gamma text.")
    c2 = retrieval.BM25Config(normalize=True, title_repeats=2)
    assert c2.index_text(d) == "Alpha Beta. Alpha Beta. Gamma text."
    assert retrieval.BM25Config().index_text(d) == d.index_text        # C0 unchanged
    assert retrieval.BM25Config().name == "bm25_stem_sw_k1.5_b0.75"
    r = retrieval.BM25Retriever(c2, docs=[d, doc("Other", "Delta.")])
    assert r.doc_tokens("Alpha Beta").count("alpha") == 2


# ---------------------------------------------------------------- feedback

class FakeRetriever:
    doc_ids = np.array(["d1", "d2", "d3"])

    def __init__(self, terms=("a", "b"), first=(("d1", 3.0), ("d2", 1.0))):
        self.terms, self.first = list(terms), list(first)

    def query_terms(self, q):
        return list(self.terms)

    def search(self, q, k=100):
        return self.first[:k]

    def doc_tokens(self, d):
        return {"d1": ["a", "c", "c", "d"], "d2": ["b", "c"], "d3": ["e"]}[d]

    def term_scores(self, terms):
        return np.array([{"a": 1.0, "b": 0.0, "c": 2.0}.get(terms[0], 0.0),
                         {"b": 1.0, "c": 1.0}.get(terms[0], 0.0), 0.0])

    def rank(self, scores, k=100):
        order = sorted((-s, d) for s, d in zip(scores, self.doc_ids) if s > 0)
        return [(d, -s) for s, d in order][:k]


def test_feedback_formula_hand_computed():
    # pi = (0.75, 0.25); P_R: c .5, a .1875, d .1875, b .125 -> top 2 = c, a (a < d on the tie)
    w, info = feedback.query_weights(FakeRetriever(), "q", feedback.FeedbackConfig(2, 2, 0.5))
    assert info["expansion_terms"] == ["c", "a"] and info["fallback"] is None
    assert w == pytest.approx({"a": 0.5 * 0.5 + 0.5 * (0.1875 / 0.6875), "b": 0.25, "c": 0.5 * (0.5 / 0.6875)})
    ranking, _ = feedback.rank(FakeRetriever(), "q", feedback.FeedbackConfig(2, 2, 0.5))
    # d1: a .386 + c .727 ; d2: b .25 + c .364 -> d1 first
    assert ranking == ["d1", "d2"]
    assert feedback.query_weights(FakeRetriever(), "q", feedback.FeedbackConfig(2, 2, 0.5)) == (w, info)


def test_feedback_fallbacks_return_the_base_ranking():
    no_terms = FakeRetriever(terms=())
    ranking, info = feedback.rank(no_terms, "q", feedback.C3A)
    assert info["fallback"] == "no_query_terms" and ranking == ["d1", "d2"]
    zero = FakeRetriever(first=(("d1", 0.0),))
    assert feedback.rank(zero, "q", feedback.C3A)[1]["fallback"] == "no_feedback_scores"


# ---------------------------------------------------------------- aliases and matching

def test_aliases_full_and_short():
    assert graph.title_aliases("Naked Lunch (film)") == [("naked", "lunch", "film"), ("naked", "lunch")]
    assert graph.title_aliases("Spider-Man") == [("spider", "man")]


@pytest.mark.parametrize("alias,df,why", [
    (("up",), {}, "short"), (("the", "it"), {}, "stopwords_only"), (("1999",), {}, "digits"),
    (("home",), {"home": 101}, "frequent_single_token"), (("home",), {"home": 100}, None),
    (("david", "cronenberg"), {}, None),
])
def test_alias_rejection(alias, df, why):
    assert graph.alias_rejection(alias, df) == why


def filler(n, text, prefix):
    return [doc(f"{prefix} {i:03d}", text, doc_id=f"{prefix}{i:03d}") for i in range(n)]


@pytest.fixture(scope="module")
def a2_docs():
    docs = [doc("Grand Hotel (film)", "A film."), doc("Blue Lagoon (novel)", "A novel."),
            doc("Red Planet", "A place."), doc("New York (band)", "A band."), doc("New York City", "A city."),
            doc("Rare Title", "A rare thing."), doc("Linker", "It mentions Rare Title once.")]
    docs += filler(101, "They stayed at the grand hotel.", "gh")                 # short alias, 101 paragraphs
    docs += filler(100, "The blue lagoon, and again the blue lagoon.", "bl")    # exactly 100; repeats count once
    docs += filler(101, "The red planet rises.", "rp")                          # full alias, 101 paragraphs
    docs += filler(60, "She moved to New York City.", "nyc")                    # longer alias contains "new york"
    docs += filler(41, "A trip to New York.", "ny")
    return docs


@pytest.fixture(scope="module")
def a2_graph(a2_docs):
    return graph.build(a2_docs)


def test_a2_independent_counts(a2_docs):
    sents = {d.doc_id: [tok_v1(s) for s in d.sentences] for d in a2_docs}
    counts = graph.alias_paragraph_counts(sents, [("grand", "hotel"), ("blue", "lagoon"), ("red", "planet"),
                                                  ("new", "york"), ("new", "york", "city")])
    assert counts == {("grand", "hotel"): 101, ("blue", "lagoon"): 100, ("red", "planet"): 101,
                      ("new", "york"): 101, ("new", "york", "city"): 60}


def test_a2_multi_token_alias_in_101_paragraphs_is_removed(a2_graph):
    assert ("grand", "hotel") not in a2_graph.alias_title


def test_a2_alias_in_exactly_100_paragraphs_is_kept_and_repeats_count_once(a2_graph):
    counts = graph.alias_paragraph_counts({d: [tok_v1(s)] for d, s in [("x", "the blue lagoon the blue lagoon")]},
                                          [("blue", "lagoon")])
    assert counts[("blue", "lagoon")] == 1
    assert ("blue", "lagoon") in a2_graph.alias_title


def test_a2_applies_to_full_titles(a2_graph):
    assert ("red", "planet") not in a2_graph.alias_title


def test_a2_counts_are_independent_of_longer_aliases(a2_graph):
    # "new york" occurs in 60 + 41 = 101 paragraphs; greedy matching would see only 41
    assert ("new", "york") not in a2_graph.alias_title
    assert ("new", "york", "city") in a2_graph.alias_title


def test_rare_useful_link_remains(a2_graph):
    assert [d for _, _, d in a2_graph.targets("Linker")] == ["Rare Title"]


def test_graph_rebuild_and_fingerprint_are_deterministic(a2_graph):
    docs = [doc("Rare Title", "A rare thing."), doc("Linker", "It mentions Rare Title once.")]
    assert graph.build(docs).fingerprint() == graph.build(list(reversed(docs))).fingerprint()


def test_matching_longest_non_overlapping_within_sentences():
    at = {("new", "york"): "NY", ("new", "york", "city"): "NYC", ("city", "hall"): "CH"}
    lb = {"new": [3, 2], "city": [2]}
    assert graph.match_sentence(["new", "york", "city", "hall"], at, lb) == [(0, "NYC")]
    g = graph.build([doc("New York City", "x."), doc("City Hall", "y."),
                     doc("Paper", "He left New York", "City Hall was closed.")])
    assert [t for t, _, _ in g.out["Paper"]] == ["City Hall"]     # no match across the sentence break


def test_self_edges_and_duplicate_title_variants():
    v1 = doc("Twin", "First text.", doc_id="Twin #aaaa")
    v2 = doc("Twin", "Second text mentions Twin.", doc_id="Twin #bbbb")
    src = doc("Source", "About Twin and more.")
    g = graph.build([v1, v2, src])
    assert [d for _, _, d in g.targets("Source")] == ["Twin #aaaa", "Twin #bbbb"]   # both variants
    assert "Twin #bbbb" not in g.out                                                # no edge to own title


def test_ambiguous_short_alias_is_dropped():
    g = graph.build([doc("Innocence (2005 film)", "x."), doc("Innocence (song)", "y."),
                     doc("Writer", "Innocence was praised.")])
    assert ("innocence",) not in g.alias_title and "Writer" not in g.out


# ---------------------------------------------------------------- the expansion operator

class G:
    """Minimal graph for operator tests: explicit targets and sources per document."""

    def __init__(self, targets=None, sources=None):
        self.t, self.s = targets or {}, sources or {}

    def targets(self, d):
        return [(0, i, x) for i, x in enumerate(self.t.get(d, []))]

    def sources(self, d):
        return self.s.get(d, [])

    def in_degree(self, d):
        return len(self.s.get(d, []))


BASE = [f"b{i:02d}" for i in range(1, 31)]


def test_forward_cap_order_and_exactly_ten_unique():
    g = G({"b01": ["x1", "x2", "x3", "x4"], "b02": ["x5", "x6"]})
    ranking, added = graph.expand(BASE, g, graph.C4_5, lambda d: 0.0)
    assert ranking[:10] == BASE[:5] + ["x1", "x2", "x3", "x5", "x6"]
    assert len(ranking[:10]) == len(set(ranking[:10])) == 10
    assert len(ranking) == len(set(ranking)) == 35          # 5 retained + 5 added + the 25 remaining base documents
    assert [a["direction"] for a in added] == ["forward"] * 5


def test_seed_ranked_6_to_10_is_eligible_when_another_seed_links_to_it():
    g = G({"b02": ["b07"]})
    ranking, added = graph.expand(BASE, g, graph.C4_10, lambda d: 0.0)
    assert ranking[5] == "b07" and added[0]["doc"] == "b07" and added[0]["seed_rank"] == 2


def test_unlinked_seed_ranked_6_to_10_gets_no_boost():
    g = G({"b01": ["x1", "x2", "x3"], "b02": ["x4", "x5"]})
    ranking, _ = graph.expand(BASE, g, graph.C4_10, lambda d: 0.0)
    assert "b08" not in ranking[:10] and ranking.index("b08") > 9


def test_retained_documents_are_never_candidates():
    g = G({"b01": ["b02", "x1"]})
    ranking, added = graph.expand(BASE, g, graph.C4_5, lambda d: 0.0)
    assert [a["doc"] for a in added] == ["x1"] and ranking[:6] == BASE[:5] + ["x1"]


def test_backward_cap_score_order_and_zero_score_exclusion_on_a_hub():
    hub_sources = [f"s{i:03d}" for i in range(500)]
    score = {"s010": 5.0, "s020": 9.0, "s030": 1.0}
    g = G(sources={"b01": hub_sources})
    ranking, added = graph.expand(BASE, g, graph.C5_5, lambda d: score.get(d, 0.0))
    assert [a["doc"] for a in added] == ["s020", "s010"]          # top 2 by lexical score; zero scores excluded
    assert all(a["direction"] == "backward" and a["seed_in_degree"] == 500 for a in added)


def test_d3_retains_and_fills_from_H_not_B():
    B = BASE
    H = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11"]
    ranking, _ = graph.expand(H, G(), graph.C4_5, lambda d: 0.0)
    assert ranking[:10] == H[:10] and not set(ranking) & set(B)


def test_d3_ranks_backward_candidates_by_B_scores():
    H = ["h1", "h2", "h3", "h4", "h5", "h6"]
    b_score = {"s1": 1.0, "s2": 3.0, "s3": 2.0}          # B's lexical scores, not H's order
    ranking, added = graph.expand(H, G(sources={"h1": ["s1", "s2", "s3"]}), graph.C5_5, lambda d: b_score.get(d, 0.0))
    assert [a["doc"] for a in added] == ["s2", "s3"]


def test_candidate_functions_take_no_gold_information():
    assert list(inspect.signature(graph.expand).parameters) == ["base", "g", "cfg", "lexical_score", "depth"]
    assert list(inspect.signature(feedback.rank).parameters) == ["retriever", "question", "cfg", "depth"]


# ---------------------------------------------------------------- selection, replay, refusal, manifests

def test_plateau_select():
    r = run_eval.plateau_select({"B": 0.60, "C4@5": 0.63, "C4@10": 0.64}, ["B", "C4@5", "C4@10"])
    assert r["plateau"] == ["C4@5", "C4@10"] and r["selected"] == "C4@5"


def test_phase2a_refuses_other_splits():
    with pytest.raises(SystemExit):
        run_eval.main(["phase2a", "--split", "test_retrieval"])
    with pytest.raises(SystemExit):
        run_eval.main(["phase2a", "--split", "dev"])


def test_manifest_config_carries_all_settings():
    g = graph.build([doc("Rare Title", "x."), doc("Linker", "Mentions Rare Title.")])
    cfg = run_eval.phase2a_config(g)
    assert cfg["graph"]["fingerprint"] == g.fingerprint() and cfg["graph"]["max_alias_df"] == 100
    assert set(cfg["step1"]) == {"C0", "C1", "C2"} and set(cfg["feedback"]) == {"C3a", "C3b"}
    assert set(cfg["graph"]["configs"]) == {"C4@5", "C4@10", "C5@5", "C5@10"}
    json.dumps(cfg)


class MissingOne:
    """Offline-client stand-in: every generation is a cache hit except one."""

    options = {"num_ctx": 8192, "num_predict": 512}

    def __init__(self, real, missing_qid_prompt_fragment):
        self.real, self.fragment = real, missing_qid_prompt_fragment

    def count_tokens(self, text):
        return self.real.count_tokens(text)

    def generate(self, user, schema):
        if self.fragment in user:
            raise KeyError("cache miss in offline mode")
        return self.real.generate(user, schema)


PHASE1 = os.path.join(data.HERE, "results", "dev", "bm25_k10.jsonl")
needs_phase1 = pytest.mark.skipif(not os.path.exists(PHASE1), reason="Phase 1 dev results are local only")


def _offline_client():
    import llm
    with open(PHASE1 + ".manifest.json") as f:
        base = json.load(f)["comparison_invariants"]["base_llm"]
    return llm.OllamaClient(offline=True, offline_digest=base["digest"], offline_quantization=base["quantization"])


@needs_phase1
def test_replay_aborts_on_one_missing_cache_entry():
    q = data.load_split("dev")[37]
    with pytest.raises(KeyError):
        run_eval.replay_baseline(client=MissingOne(_offline_client(), q.question))[0]


@needs_phase1
def test_replay_reproduces_all_100_phase1_top10_lists():
    rows, client = run_eval.replay_baseline(client=_offline_client())
    old = {r["qid"]: r for r in run_eval.read_jsonl(PHASE1)}
    with open(PHASE1 + ".manifest.json") as f:
        phase1_llm = json.load(f)["comparison_invariants"]["base_llm"]
    assert {k: client.describe()[k] for k in phase1_llm} == phase1_llm       # the replay manifest names the LLM
    assert len(rows) == 100 and all(r["llm"]["cached"] for r in rows)
    assert all(r["ranked"][:10] == old[r["qid"]]["ranked"][:10] for r in rows)


def test_phase2a_config_is_deterministic_and_carries_no_timing():
    docs = [doc("Rare Title", "x."), doc("Linker", "Mentions Rare Title.")]
    a, b = run_eval.phase2a_config(graph.build(docs)), run_eval.phase2a_config(graph.build(docs))
    assert a == b
    assert "build_s" not in json.dumps(a) and "_s\"" not in json.dumps(a["graph"]["stats"])


def test_phase2a_selection_is_written_atomically_with_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    g = graph.build([doc("Rare Title", "x."), doc("Linker", "Mentions Rare Title.")])
    qids = data.load_split_ids("tune")
    path = run_eval.save_phase2a_selection({"C": "C5@5", "graph_stats": g.stats}, qids, run_eval.phase2a_config(g), 1.5)
    assert json.load(open(path)) == {"C": "C5@5", "graph_stats": json.loads(json.dumps(g.stats))}
    m = json.load(open(path + ".manifest.json"))
    assert m["comparison_invariants"]["split_hash"] == data.EXPECTED_SPLIT_HASHES["tune"]
    assert "build_s" not in json.dumps(m["system_config"]) and m["run_metadata"]["runtime_s"] == 1.5
    assert not [p for p in os.listdir(os.path.dirname(path)) if p.endswith(".tmp")]
