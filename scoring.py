"""Per-question scores from pipeline records, and their summaries.

This is the only path from a system's output to a reported number (with ``metrics.py``
underneath), so it is part of the evaluation-code hash in run manifests.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import data
import metrics

# metric name -> None when it does not apply to a record (e.g. evidence for closed-book)
QUALITY = ("em", "f1", "sp_strict_f1", "sp_strict_prec", "sp_strict_recall", "sp_official_f1",
           "joint_official_f1", "ctx_recall", "ctx_joint_recall", "ctx_joint_recall_title",
           "incomplete_gold_evidence", "supported", "has_citations", "all_valid_citations",
           "schema_error")


# Malformed-answer diagnostic (PREREGISTRATION.md amendment A1). It only labels answers;
# it never rewrites them and never changes F1 or EM.
_ENCODED_CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\(.*\)", re.S)   # whole answer, e.g. strconv(1887)
_ENCODED_LABEL = re.compile(r"[a-z_][a-z0-9_]*:\S")                  # prefix, e.g. math:2001
MAX_ANSWER_WORDS = 10


def answer_format_flags(answer: str) -> List[str]:
    """``empty``: no non-whitespace character. ``encoded``: the whole answer is a function
    call with no space before "(", or it starts with a lowercase identifier joined to a
    colon with no space after it. ``long``: more than 10 whitespace-separated words."""
    flags = []
    a = answer.strip()
    if not a:
        flags.append("empty")
    elif _ENCODED_CALL.fullmatch(a) or _ENCODED_LABEL.match(a):
        flags.append("encoded")
    if len(answer.split()) > MAX_ANSWER_WORDS:
        flags.append("long")
    return flags


def answer_outcome(em: float, f1: float) -> str:
    """Taxonomy outcome: correct (EM 1), partial (EM 0, F1 >= 0.5), wrong (F1 < 0.5)."""
    return "correct" if em == 1.0 else ("partial" if f1 >= 0.5 else "wrong")


def score(q: data.Question, rec: Mapping[str, Any], evidence: bool = True,
          doc_key: Optional[Mapping[str, tuple]] = None) -> Dict[str, Any]:
    """Scores for one output. ``evidence=False`` (closed-book) leaves evidence, citation
    and context metrics as None. ``doc_key`` maps doc_id -> (title, text_hash); the corpus
    mapping by default."""
    out = rec["output"]
    ans = metrics.answer_scores(out["answer"], q.answer)
    s: Dict[str, Any] = {"qid": q.qid, "type": q.type, "em": ans["em"], "f1": ans["f1"],
                         "supported": float(bool(out["supported"])),
                         "schema_error": float(rec["schema_error"]),
                         "answer_flags": answer_format_flags(out["answer"]),
                         "outcome": answer_outcome(ans["em"], ans["f1"])}
    if not evidence:
        for k in QUALITY:
            s.setdefault(k, None)
        return s
    if doc_key is None:
        corpus = data.corpus()
        doc_key = {d: (corpus[d].title, corpus[d].text_hash) for d in rec["displayed"]}
        for d, _ in rec["citations"]:
            if d in corpus:
                doc_key[d] = (corpus[d].title, corpus[d].text_hash)
    gold_hash = dict(q.gold)
    ev = metrics.evidence_scores([tuple(c) for c in rec["citations"]], q.supporting_facts, gold_hash, doc_key)
    joint = metrics.joint_scores(ans, ev["official"])
    cv = metrics.citation_validity([tuple(c) for c in rec["citations"]],
                                   dict(zip(rec["displayed"], rec["displayed_sentences"])))
    shown_keys = {doc_key[d] for d in rec["displayed"]}
    shown_titles = {doc_key[d][0] for d in rec["displayed"]}
    cited_keys = {doc_key[d] for d, _ in rec["citations"] if d in doc_key}
    s.update({
        "sp_strict_f1": ev["strict"]["f1"], "sp_strict_prec": ev["strict"]["prec"],
        "sp_strict_recall": ev["strict"]["recall"], "sp_official_f1": ev["official"]["f1"],
        "joint_official_f1": joint["f1"],
        "ctx_recall": metrics.recall(shown_keys, q.gold),
        "ctx_joint_recall": metrics.joint_recall(shown_keys, q.gold),
        "ctx_joint_recall_title": metrics.joint_recall(shown_titles, [t for t, _ in q.gold]),
        "gold_in_context": {t: (t, h) in shown_keys for t, h in q.gold},
        "incomplete_gold_evidence": float(not set(q.gold) <= cited_keys),
        "has_citations": cv["has_citations"], "all_valid_citations": cv["all_valid"],
        "n_citations": cv["n"], "valid_citations": cv["valid"],
    })
    return s


def summarize(scores: Sequence[Mapping[str, Any]], names: Iterable[str] = QUALITY) -> Dict[str, Dict[str, float]]:
    """Mean and 95% bootstrap interval of each applicable question-level metric."""
    out = {}
    for n in names:
        vals = [s[n] for s in scores if s.get(n) is not None]
        if vals:
            out[n] = metrics.mean_ci(vals)
    return out


def retrieval_scores(q: data.Question, ranked: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    """Recall@K and Joint Recall@K for a ranked list of corpus doc ids, exact variant and
    title-only (prefixed ``title_``)."""
    ids = data.doc_id_of()
    corpus = data.corpus()
    gold = [ids[g] for g in q.gold]
    exact = metrics.recall_at_k(ranked, gold, ks)
    title = metrics.recall_at_k(ranked, gold, ks, key={d: corpus[d].title for d in list(ranked) + gold})
    return {**exact, **{f"title_{k}": v for k, v in title.items()}}
