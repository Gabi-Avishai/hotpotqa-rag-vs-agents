"""Scoring of pipeline records, taxonomy buckets, and the evaluation-code hash guard."""
import ast
import os

import pytest

import data
import hops
import manifest
import scoring


@pytest.fixture(scope="module")
def q():
    return data.load_split("dev_smoke")[0]


def rec_for(q, displayed, citations, answer, supported=True, schema_error=False):
    corpus = data.corpus()
    return {"displayed": displayed, "displayed_sentences": [len(corpus[d].sentences) for d in displayed],
            "citations": citations, "schema_error": schema_error,
            "output": {"answer": answer, "supported": supported, "rationale": "", "citations": []}}


def test_perfect_output_scores_one(q):
    ids = data.doc_id_of()
    gold_docs = [ids[g] for g in q.gold]
    cites = [[ids[(t, dict(q.gold)[t])], s] for t, s in q.supporting_facts]
    s = scoring.score(q, rec_for(q, gold_docs, cites, q.answer))
    assert s["em"] == 1 and s["sp_strict_f1"] == 1 and s["sp_official_f1"] == 1 and s["joint_official_f1"] == 1
    assert s["ctx_joint_recall"] == 1 and s["incomplete_gold_evidence"] == 0 and s["all_valid_citations"] == 1
    assert s["outcome"] == "correct"


def test_missing_gold_and_no_citations(q):
    ids = data.doc_id_of()
    first = ids[q.gold[0]]
    s = scoring.score(q, rec_for(q, [first], [], "something else", supported=False))
    assert s["ctx_recall"] == 0.5 and s["ctx_joint_recall"] == 0
    assert s["has_citations"] == 0 and s["all_valid_citations"] == 0 and s["incomplete_gold_evidence"] == 1
    assert s["gold_in_context"] == {q.gold[0][0]: True, q.gold[1][0]: False}


def test_closed_book_has_no_evidence_metrics(q):
    s = scoring.score(q, rec_for(q, [], [], q.answer), evidence=False)
    assert s["em"] == 1 and s["sp_strict_f1"] is None and s["ctx_joint_recall"] is None


def test_answer_outcome():
    assert scoring.answer_outcome(1.0, 1.0) == "correct"
    assert scoring.answer_outcome(0.0, 0.5) == "partial"
    assert scoring.answer_outcome(0.0, 0.49) == "wrong"


def test_summarize_skips_none():
    out = scoring.summarize([{"f1": 1.0, "em": None}, {"f1": 0.0, "em": None}], names=("f1", "em"))
    assert out["f1"]["mean"] == 0.5 and "em" not in out


def test_hop_rule_reproduces_phase0_counts_on_tune():
    qs = [q for q in data.load_split("tune") if q.type == "bridge"]
    ctx = data.load_contexts("tune")
    rules = [hops.chain(q, ctx[q.qid]).rule for q in qs]
    counts = {r: rules.count(r) for r in set(rules)}
    assert counts == {"supporting": 305, "paragraph": 32, "question": 29, "answer": 26, "undetermined": 8}


def test_evidence_buckets():
    class Q:
        type = "bridge"
    ch = hops.Chain("A", "B", "supporting")
    und = hops.Chain("A", "B", "undetermined")
    assert hops.evidence_bucket(Q, ch, {"A": True, "B": True}) == "both"
    assert hops.evidence_bucket(Q, ch, {"A": True, "B": False}) == "only_hop1"
    assert hops.evidence_bucket(Q, ch, {"A": False, "B": True}) == "only_hop2"
    assert hops.evidence_bucket(Q, und, {"A": False, "B": True}) == "only_one_hop_undetermined"
    assert hops.evidence_bucket(Q, ch, {"A": False, "B": False}) == "neither"
    Q.type = "comparison"
    assert hops.evidence_bucket(Q, hops.Chain("A", "B", None), {"A": True, "B": False}) == "only_one_entity"


SCORING_FUNCS = {"normalize_answer", "answer_scores", "set_scores", "joint_scores", "evidence_scores",
                 "citation_validity", "aggregate_citations", "recall", "joint_recall", "recall_at_k",
                 "answered_f1", "weighted_mean", "mean_ci", "paired_bootstrap", "latency_stats"}


def test_every_module_that_scores_is_in_the_evaluation_hash():
    users = set()
    for name in os.listdir(data.HERE):
        if not name.endswith(".py") or name == "metrics.py":
            continue
        tree = ast.parse(open(os.path.join(data.HERE, name)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "metrics" and node.attr in SCORING_FUNCS:
                users.add(name)
            if isinstance(node, ast.ImportFrom) and node.module == "metrics" \
                    and any(a.name in SCORING_FUNCS for a in node.names):
                users.add(name)
    assert users, "the scan found nothing"
    assert users <= set(manifest.EVAL_CODE), users - set(manifest.EVAL_CODE)


@pytest.mark.parametrize("answer,flags", [
    ("", ["empty"]), ("   ", ["empty"]),
    ("strconv(1887)", ["encoded"]), ("int(5)", ["encoded"]), ("math.floor(3.5)", ["encoded"]),
    ("math:2001-2016", ["encoded"]), ("answer:Paris", ["encoded"]),
    ("Eric (musician)", []), ("Star Wars: A New Hope", []), ("1:30", []), ("Paris", []),
    ("March 15, 1943", []), ("yes", []), ("U.S. (state)", []),
    ("one two three four five six seven eight nine ten eleven", ["long"]),
])
def test_answer_format_flags(answer, flags):
    assert scoring.answer_format_flags(answer) == flags


def test_flags_are_labels_only(q):
    s = scoring.score(q, rec_for(q, [], [], "strconv(" + q.answer + ")"), evidence=False)
    assert s["answer_flags"] == ["encoded"] and s["em"] == 0.0   # scored as written, not repaired
