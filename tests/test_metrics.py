"""Metric tests, including a cross-check against the official HotpotQA scorer.

The official ``hotpot_evaluate_v1.py`` has no licence file in its repository, so it is not
copied into this project: it is downloaded into the ignored ``cache/`` at a pinned commit
and verified by SHA-256. Gold labels for the cross-check come from open splits only.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
import types
import urllib.request

import numpy as np
import pytest

import data
import metrics as M

OFFICIAL_URL = ("https://raw.githubusercontent.com/hotpotqa/hotpot/"
                "3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py")
OFFICIAL_SHA256 = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
OFFICIAL_PATH = os.path.join(data.HERE, "cache", "official", "hotpot_evaluate_v1.py")


@pytest.fixture(scope="module")
def official():
    if not os.path.exists(OFFICIAL_PATH):
        os.makedirs(os.path.dirname(OFFICIAL_PATH), exist_ok=True)
        urllib.request.urlretrieve(OFFICIAL_URL, OFFICIAL_PATH)
    with open(OFFICIAL_PATH, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == OFFICIAL_SHA256
    sys.modules.setdefault("ujson", json)  # the scorer imports ujson only for file loading
    spec = importlib.util.spec_from_file_location("hotpot_official", OFFICIAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- answers

@pytest.mark.parametrize("pred,gold,em,f1", [
    ("The Beatles", "beatles", 1.0, 1.0),
    ("the  Beatles!", "Beatles", 1.0, 1.0),
    ("yes", "no", 0.0, 0.0),
    ("yes", "Yes.", 1.0, 1.0),
    ("no", "no way", 0.0, 0.0),            # yes/no rule: no partial credit
    ("Walter Coy", "Walter Darwin Coy", 0.0, 0.8),
    ("", "Paris", 0.0, 0.0),
    ("an apple", "a", 0.0, 0.0),           # both normalise to "apple" vs ""
])
def test_answer_scores_hand_cases(pred, gold, em, f1):
    s = M.answer_scores(pred, gold)
    assert s["em"] == em and s["f1"] == pytest.approx(f1)


def _synthetic_predictions(questions, seed=0):
    """Answers and supporting facts spanning exact, partial, wrong, yes/no, punctuation and
    empty cases, and evidence sets with missing, extra and out-of-range facts."""
    rng = random.Random(seed)
    answers = [q.answer for q in questions]
    pred_ans, pred_sp = {}, {}
    for i, q in enumerate(questions):
        mode = i % 7
        a = q.answer
        pred_ans[q.qid] = [a, a.upper() + ".", " ".join(a.split()[:1]), rng.choice(answers),
                           rng.choice(["yes", "no"]), f"the {a} and more", ""][mode]
        sf = [list(f) for f in q.supporting_facts]
        extra = [[q.context[rng.randrange(len(q.context))][0], rng.randrange(6)]]
        pred_sp[q.qid] = [sf, sf[:1], sf + extra, extra, [], sf[1:] + extra, sf][i % 7 - (i // 7) % 2]
    return pred_ans, pred_sp


def test_cross_check_with_official_scorer(official, tmp_path):
    questions = data.load_split("tune") + data.load_split("dev")
    pred_ans, pred_sp = _synthetic_predictions(questions)
    gold = [{"_id": q.qid, "answer": q.answer, "supporting_facts": [list(f) for f in q.supporting_facts]}
            for q in questions]
    (tmp_path / "pred.json").write_text(json.dumps({"answer": pred_ans, "sp": pred_sp}))
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        official.eval(str(tmp_path / "pred.json"), str(tmp_path / "gold.json"))
    want = ast.literal_eval(buf.getvalue().strip().splitlines()[-1])

    ours = {k: [] for k in want}
    for q in questions:
        a = M.answer_scores(pred_ans[q.qid], q.answer)
        sp = M.set_scores(map(tuple, pred_sp[q.qid]), q.supporting_facts)
        j = M.joint_scores(a, sp)
        for name, src in (("", a), ("sp_", sp), ("joint_", j)):
            for k in ("em", "f1", "prec", "recall"):
                ours[name + k].append(src[k])
    for k, v in want.items():
        assert np.mean(ours[k]) == pytest.approx(v, abs=1e-12), k
    # the synthetic set exercises every regime
    assert 0 < want["em"] < 1 and 0 < want["sp_em"] < 1 and 0 < want["joint_f1"] < 1


def test_per_item_agreement_with_official(official):
    rng = random.Random(1)
    words = ["yes", "no", "noanswer", "the", "a", "Paris", "France", "U.S.", "1990", "", "an"]
    for _ in range(2000):
        p = " ".join(rng.choice(words) for _ in range(rng.randint(0, 3)))
        g = " ".join(rng.choice(words) for _ in range(rng.randint(0, 3)))
        f1, prec, rec = official.f1_score(p, g)
        s = M.answer_scores(p, g)
        assert (s["f1"], s["prec"], s["recall"]) == pytest.approx((f1, prec, rec))
        assert s["em"] == float(official.exact_match_score(p, g))


# ---------------------------------------------------------------- evidence variants

@pytest.fixture
def dup_case():
    doc_key = {"T #aaaa": ("T", "h1"), "T #bbbb": ("T", "h2"), "U": ("U", "h3")}
    gold_facts = [("T", 0), ("U", 1)]
    gold_hash = {"T": "h1", "U": "h3"}
    return doc_key, gold_facts, gold_hash


def test_strict_rejects_wrong_variant_official_accepts(dup_case):
    doc_key, gold_facts, gold_hash = dup_case
    s = M.evidence_scores([("T #bbbb", 0), ("U", 1)], gold_facts, gold_hash, doc_key)
    assert s["official"]["f1"] == 1.0 and s["official"]["em"] == 1.0
    assert s["strict"]["prec"] == 0.5 and s["strict"]["recall"] == 0.5 and s["strict"]["em"] == 0.0


def test_strict_equals_official_on_right_variant(dup_case):
    doc_key, gold_facts, gold_hash = dup_case
    s = M.evidence_scores([("T #aaaa", 0), ("U", 1), ("U", 2)], gold_facts, gold_hash, doc_key)
    assert s["strict"] == s["official"]
    assert s["strict"]["prec"] == pytest.approx(2 / 3)


def test_unknown_doc_counts_against_precision(dup_case):
    doc_key, gold_facts, gold_hash = dup_case
    s = M.evidence_scores([("nope", 0), ("U", 1)], gold_facts, gold_hash, doc_key)
    assert s["strict"]["prec"] == 0.5 and s["official"]["prec"] == 0.5


def test_citation_validity():
    ctx = {"A": 3, "B": 1}
    assert M.citation_validity([("A", 2), ("B", 0)], ctx) == {
        "n": 2, "valid": 2, "has_citations": 1.0, "all_valid": 1.0, "valid_fraction": 1.0}
    v = M.citation_validity([("A", 3), ("C", 0), ("B", 0)], ctx)  # out of range; not in context
    assert v["valid"] == 1 and v["all_valid"] == 0.0 and v["valid_fraction"] == pytest.approx(1 / 3)


def test_no_citations_is_not_valid_citation():
    v = M.citation_validity([], {})
    assert v == {"n": 0, "valid": 0, "has_citations": 0.0, "all_valid": 0.0, "valid_fraction": None}
    v = M.citation_validity([], {"A": 3})
    assert v["all_valid"] == 0.0 and v["valid_fraction"] is None


def test_aggregate_citations_separates_coverage_from_validity():
    ctx = {"A": 3}
    outs = [M.citation_validity([("A", 0), ("A", 1)], ctx),     # 2 valid
            M.citation_validity([("A", 0), ("A", 9)], ctx),     # 1 of 2 valid
            M.citation_validity([], ctx)]                        # no citations
    agg = M.aggregate_citations(outs)
    assert agg["coverage"] == pytest.approx(2 / 3)
    assert agg["per_citation_validity"] == pytest.approx(3 / 4)
    assert agg["all_valid_rate"] == pytest.approx(1 / 3)
    none = M.aggregate_citations([M.citation_validity([], ctx)] * 2)
    assert none["coverage"] == 0.0 and none["per_citation_validity"] is None and none["all_valid_rate"] == 0.0
    with pytest.raises(ValueError):
        M.aggregate_citations([])


# ---------------------------------------------------------------- retrieval

def test_exact_variant_versus_title_only_retrieval():
    key = {"T #aaaa": "T", "T #bbbb": "T", "U": "U", "V": "V"}
    gold = ["T #aaaa", "U"]
    ranked = ["V", "T #bbbb", "U", "T #aaaa"]  # wrong variant ranked above the right one
    exact = M.recall_at_k(ranked, gold, ks=(3, 4))
    title = M.recall_at_k(ranked, gold, ks=(3, 4), key=key)
    assert exact["recall@3"] == 0.5 and exact["joint_recall@3"] == 0.0
    assert title["recall@3"] == 1.0 and title["joint_recall@3"] == 1.0
    assert exact["joint_recall@4"] == 1.0


def test_final_context_and_cumulative_joint_recall():
    gold = ["A", "B"]
    searched = [["A", "C"], ["D", "B"]]           # both found across the searches
    final_context = ["A", "C", "D"]               # B dropped before the answer step
    assert M.joint_recall([d for s in searched for d in s], gold) == 1.0
    assert M.joint_recall(final_context, gold) == 0.0
    assert M.recall(final_context, gold) == 0.5


# ---------------------------------------------------------------- aggregation

def test_answered_f1():
    f1 = {"a": 1.0, "b": 0.0, "c": 0.5}
    assert M.answered_f1(f1, {"a": True, "b": False, "c": True}) == {"f1_answered": 0.75, "n_answered": 2}
    assert M.answered_f1(f1, {q: False for q in f1}) == {"f1_answered": None, "n_answered": 0}


def test_answered_f1_requires_identical_ids():
    f1 = {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError, match="different questions"):
        M.answered_f1(f1, {"a": True, "b": True, "extra": True})
    with pytest.raises(ValueError, match="different questions"):
        M.answered_f1(f1, {"a": True})
    with pytest.raises(ValueError):
        M.answered_f1({}, {})


def test_weighted_mean_hand_computed():
    scores = {"b1": 1.0, "b2": 0.0, "c1": 1.0, "c2": 1.0}
    strata = {"b1": "bridge", "b2": "bridge", "c1": "comparison", "c2": "comparison"}
    assert M.weighted_mean(scores, strata) == 0.75
    assert M.weighted_mean(scores, strata, {"bridge": 0.8, "comparison": 0.2}) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        M.weighted_mean(scores, strata, {"bridge": 0.8, "comparison": 0.3})
    with pytest.raises(ValueError):
        M.weighted_mean({"b1": 1.0}, {"b1": "bridge"}, {"bridge": 0.5, "comparison": 0.5})


def test_weighted_mean_fails_clearly():
    w = {"bridge": 0.8, "comparison": 0.2}
    with pytest.raises(ValueError, match="no scores"):
        M.weighted_mean({}, {}, w)
    with pytest.raises(ValueError, match="no scores"):
        M.weighted_mean({})
    with pytest.raises(ValueError, match="no stratum"):
        M.weighted_mean({"b1": 1.0, "c1": 0.0}, {"b1": "bridge"}, w)
    with pytest.raises(ValueError, match="without a weight"):     # would otherwise be dropped silently
        M.weighted_mean({"b1": 1.0, "c1": 0.0, "x": 5.0},
                        {"b1": "bridge", "c1": "comparison", "x": "other"}, w)
    with pytest.raises(ValueError, match="need strata"):
        M.weighted_mean({"b1": 1.0}, None, w)


def test_bootstrap_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="different questions"):
        M.paired_bootstrap({"a": 1.0, "b": 0.0}, {"a": 1.0, "c": 0.0})
    with pytest.raises(ValueError):
        M.paired_bootstrap({}, {})


def test_bootstrap_constant_difference_and_add_one_p():
    a = {str(i): 1.0 for i in range(50)}
    b = {str(i): 0.5 for i in range(50)}
    r = M.paired_bootstrap(a, b, n_samples=2000)
    assert r["delta"] == 0.5 and r["ci_low"] == 0.5 and r["ci_high"] == 0.5
    assert r["p_value"] == pytest.approx(2 / 2001)  # never exactly zero
    assert r["n"] == 50


def test_bootstrap_is_antisymmetric_and_seeded():
    rng = np.random.default_rng(3)
    a = {str(i): float(v) for i, v in enumerate(rng.random(80))}
    b = {str(i): float(v) for i, v in enumerate(rng.random(80))}
    ab, ba = M.paired_bootstrap(a, b, seed=7), M.paired_bootstrap(b, a, seed=7)
    assert ab["delta"] == pytest.approx(-ba["delta"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"]) and ab["p_value"] == pytest.approx(ba["p_value"])
    assert M.paired_bootstrap(a, b, seed=7) == ab


def test_stratified_weighted_bootstrap():
    # bridge questions improve by 0.2, comparison by 0.0: the weighted delta is 0.8 * 0.2
    strata = {f"b{i}": "bridge" for i in range(40)} | {f"c{i}": "comparison" for i in range(40)}
    a = {q: (0.7 if s == "bridge" else 0.5) for q, s in strata.items()}
    b = {q: 0.5 for q in strata}
    w = {"bridge": 0.8, "comparison": 0.2}
    r = M.paired_bootstrap(a, b, strata=strata, weights=w)
    assert r["delta"] == pytest.approx(0.16)
    assert r["ci_low"] == pytest.approx(0.16) and r["ci_high"] == pytest.approx(0.16)
    assert M.paired_bootstrap(a, b)["delta"] == pytest.approx(0.10)  # unweighted 50/50 mean
    with pytest.raises(ValueError):
        M.paired_bootstrap(a, b, weights=w)
    with pytest.raises(ValueError, match="without a weight"):
        M.paired_bootstrap(a, b, strata={**strata, "b0": "other"}, weights=w)


def test_holm():
    adj = M.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj == pytest.approx({"x": 0.03, "z": 0.06, "y": 0.06})


def test_latency_stats():
    s = M.latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["p50_s"] == 3.0 and s["mean_s"] == 22.0 and s["p95_s"] == pytest.approx(80.8)
    with pytest.raises(ValueError):
        M.latency_stats([])


def test_mean_ci():
    r = M.mean_ci([1.0] * 10)
    assert r == {"mean": 1.0, "ci_low": 1.0, "ci_high": 1.0, "n": 10}
    r = M.mean_ci([0.0, 1.0] * 50)
    assert r["mean"] == 0.5 and r["ci_low"] < 0.5 < r["ci_high"]
    with pytest.raises(ValueError):
        M.mean_ci([])
