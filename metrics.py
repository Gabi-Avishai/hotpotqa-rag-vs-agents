"""Evaluation metrics for the HotpotQA closed-corpus benchmark.

Answer EM/F1 and supporting-fact scoring reproduce the official ``hotpot_evaluate_v1.py``
(commit 3635853) exactly; ``tests/test_metrics.py`` cross-checks them against it. Two
evidence variants share one set-scoring function:

- **official-compatible**: facts are ``(title, sentence_id)``, as in the official scorer;
- **strict** (primary for this corpus): facts are ``(title, text_hash, sentence_id)``, so a
  citation of the other text variant of a duplicated title does not count.

Retrieval metrics compare ranked document ids with the question's gold documents, either
by exact identity (doc id) or by title only (secondary, compatibility).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

METRICS_VERSION = "1"   # bump on any change to a metric definition; part of the run manifest
MPI = 0.03              # minimum practically meaningful difference, 0-1 scale (PLAN.md §5)
_PUNCT = frozenset(string.punctuation)

# ---------------------------------------------------------------- answers (official)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_scores(prediction: str, gold: str) -> Dict[str, float]:
    """Official answer EM, F1, precision and recall, including the yes/no/noanswer rule:
    if either side normalises to one of those words, only an exact match scores."""
    p, g = normalize_answer(prediction), normalize_answer(gold)
    em = float(p == g)
    zero = {"em": em, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    special = ("yes", "no", "noanswer")
    if (p in special or g in special) and p != g:
        return zero
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return zero
    prec, rec = same / len(pt), same / len(gt)
    return {"em": em, "f1": 2 * prec * rec / (prec + rec), "prec": prec, "recall": rec}


# ---------------------------------------------------------------- supporting facts


def set_scores(predicted: Iterable[Hashable], gold: Iterable[Hashable]) -> Dict[str, float]:
    """Official supporting-fact scoring over sets of facts (any hashable fact form)."""
    pred, gold = set(predicted), set(gold)
    tp = len(pred & gold)
    fp, fn = len(pred - gold), len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"em": float(fp + fn == 0), "f1": f1, "prec": prec, "recall": rec}


def joint_scores(ans: Mapping[str, float], sp: Mapping[str, float]) -> Dict[str, float]:
    """Official joint metric: products of answer and evidence precision/recall."""
    jp, jr = ans["prec"] * sp["prec"], ans["recall"] * sp["recall"]
    return {"em": ans["em"] * sp["em"], "f1": 2 * jp * jr / (jp + jr) if jp + jr else 0.0,
            "prec": jp, "recall": jr}


Citation = Tuple[str, int]  # (doc_id, sentence_id)


def evidence_scores(citations: Sequence[Citation], gold_facts: Sequence[Tuple[str, int]],
                    gold_hash: Mapping[str, str], doc_key: Mapping[str, Tuple[str, str]]
                    ) -> Dict[str, Dict[str, float]]:
    """Strict and official-compatible supporting-fact scores for one output.

    ``citations`` are ``(doc_id, sentence_id)``; ``doc_key`` maps doc_id -> (title, text_hash).
    ``gold_facts`` are the official ``(title, sentence_id)``; ``gold_hash`` maps each gold
    title to the text variant of the question's own context. Unknown doc ids are kept as
    facts that match nothing (they count against precision)."""
    official_pred, strict_pred = set(), set()
    for doc_id, sid in citations:
        title, h = doc_key.get(doc_id, (f"<unknown:{doc_id}>", ""))
        official_pred.add((title, int(sid)))
        strict_pred.add((title, h, int(sid)))
    strict_gold = {(t, gold_hash[t], int(s)) for t, s in gold_facts}
    return {"strict": set_scores(strict_pred, strict_gold),
            "official": set_scores(official_pred, {(t, int(s)) for t, s in gold_facts})}


def citation_validity(citations: Sequence[Citation], context: Mapping[str, int]) -> Dict[str, object]:
    """A citation is valid when its doc_id was in the answer context and the sentence id
    exists in that exact variant. ``context`` maps doc_id -> number of sentences.

    An output without citations has ``has_citations = 0``, ``all_valid = 0`` and
    ``valid_fraction = None``: citing nothing never counts as citing validly."""
    n = len(citations)
    valid = sum(1 for d, s in citations if d in context and 0 <= int(s) < context[d])
    return {"n": n, "valid": valid, "has_citations": float(n > 0),
            "all_valid": float(n > 0 and valid == n),
            "valid_fraction": valid / n if n else None}


def aggregate_citations(per_output: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Citation coverage and validity over outputs, kept separate: ``coverage`` is the share
    of outputs with at least one citation; ``per_citation_validity`` pools all citations
    (None if there are none); ``all_valid_rate`` is over all outputs, so an output without
    citations counts as not all-valid."""
    if not per_output:
        raise ValueError("no outputs to aggregate")
    n_cit = sum(int(o["n"]) for o in per_output)
    return {"n_outputs": len(per_output),
            "coverage": float(np.mean([o["has_citations"] for o in per_output])),
            "n_citations": n_cit,
            "per_citation_validity": sum(int(o["valid"]) for o in per_output) / n_cit if n_cit else None,
            "all_valid_rate": float(np.mean([o["all_valid"] for o in per_output]))}


# ---------------------------------------------------------------- retrieval


def recall(retrieved: Iterable[str], gold: Iterable[str]) -> float:
    """Fraction of gold items present in ``retrieved``."""
    got, gold = set(retrieved), set(gold)
    return len(got & gold) / len(gold)


def joint_recall(retrieved: Iterable[str], gold: Iterable[str]) -> float:
    """1.0 when every gold item is present in ``retrieved``, else 0.0."""
    return float(set(gold) <= set(retrieved))


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], ks: Sequence[int],
                key: Optional[Mapping[str, str]] = None) -> Dict[str, float]:
    """Recall@K and Joint Recall@K for ranked doc ids. With ``key`` (doc_id -> title) both
    sides are compared by title: the secondary, title-only compatibility metric."""
    gold = list(gold)
    if key is not None:
        ranked = [key[d] for d in ranked]
        gold = [key[d] for d in gold]
    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"recall@{k}"] = recall(top, gold)
        out[f"joint_recall@{k}"] = joint_recall(top, gold)
    return out


# ---------------------------------------------------------------- aggregation


def _check_same_keys(a: Mapping, b: Mapping) -> None:
    if set(a) != set(b):
        raise ValueError(f"score sets cover different questions ({len(set(b) - set(a))} absent "
                         f"from a, {len(set(a) - set(b))} absent from b)")
    if not a:
        raise ValueError("no questions to compare")


def answered_f1(f1: Mapping[str, float], supported: Mapping[str, bool]) -> Dict[str, object]:
    """Mean F1 over outputs marked ``supported = true``; None when no output is. Both
    mappings must cover exactly the same questions."""
    _check_same_keys(f1, supported)
    vals = [f1[q] for q in f1 if supported[q]]
    return {"f1_answered": float(np.mean(vals)) if vals else None, "n_answered": len(vals)}


def _check_strata(scores: Mapping[str, float], strata: Mapping[str, str], weights: Mapping[str, float]) -> None:
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    missing = [q for q in scores if q not in strata]
    if missing:
        raise ValueError(f"{len(missing)} question(s) have no stratum")
    unweighted = sorted({strata[q] for q in scores} - set(weights))
    if unweighted:
        raise ValueError(f"strata without a weight: {unweighted}")


def weighted_mean(scores: Mapping[str, float], strata: Optional[Mapping[str, str]] = None,
                  weights: Optional[Mapping[str, float]] = None) -> float:
    """Mean of per-stratum means, weighted by ``weights`` (e.g. validation prevalence).
    Without weights, the plain mean over questions. Every question needs a stratum that
    has a weight; nothing is dropped silently."""
    if not scores:
        raise ValueError("no scores to average")
    if weights is None:
        return float(np.mean(list(scores.values())))
    if strata is None:
        raise ValueError("weights need strata")
    _check_strata(scores, strata, weights)
    by = {s: [v for q, v in scores.items() if strata[q] == s] for s in weights}
    empty = [s for s, v in by.items() if not v]
    if empty:
        raise ValueError(f"no questions in strata {empty}")
    return float(sum(w * np.mean(by[s]) for s, w in weights.items()))


def paired_bootstrap(scores_a: Mapping[str, float], scores_b: Mapping[str, float],
                     n_samples: int = 10_000, seed: int = 0,
                     strata: Optional[Mapping[str, str]] = None,
                     weights: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """Paired bootstrap of ``a - b`` over identical question sets (mismatches raise).

    With ``strata`` and ``weights``, questions are resampled within each stratum and the
    statistic is the weighted mean of per-stratum mean differences. Returns the observed
    difference, a two-sided 95% percentile interval, and a two-sided p-value with an
    add-one correction. The p-value is valid for one pre-declared comparison."""
    _check_same_keys(scores_a, scores_b)
    rng = np.random.default_rng(seed)
    qs = sorted(scores_a)
    diffs = {q: scores_a[q] - scores_b[q] for q in qs}
    if weights is None:
        d = np.array([diffs[q] for q in qs])
        means = d[rng.integers(0, len(d), size=(n_samples, len(d)))].mean(axis=1)
        observed = float(d.mean())
    else:
        observed = weighted_mean(diffs, strata, weights)  # validates strata and weights
        means = np.zeros(n_samples)
        for s, w in sorted(weights.items()):
            d = np.array([diffs[q] for q in qs if strata[q] == s])
            means += w * d[rng.integers(0, len(d), size=(n_samples, len(d)))].mean(axis=1)
    if observed > 0:
        beyond = int(np.sum(means <= 0.0))
    elif observed < 0:
        beyond = int(np.sum(means >= 0.0))
    else:
        beyond = n_samples
    return {"delta": observed,
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "p_value": min(1.0, 2.0 * (beyond + 1) / (n_samples + 1)),
            "n": len(qs)}


def mean_ci(values: Sequence[float], n_samples: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Mean with a 95% percentile bootstrap interval over questions."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no values")
    means = v[np.random.default_rng(seed).integers(0, v.size, size=(n_samples, v.size))].mean(axis=1)
    return {"mean": float(v.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(v.size)}


def holm(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Holm-adjusted p-values for a pre-declared family of comparisons."""
    order = sorted(p_values, key=p_values.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        out[name] = running
    return out


def latency_stats(seconds: Sequence[float]) -> Dict[str, float]:
    """Per-question wall-clock latency: mean, median and p95, in seconds."""
    s = np.asarray(seconds, dtype=float)
    if s.size == 0:
        raise ValueError("no latencies")
    return {"mean_s": float(s.mean()), "p50_s": float(np.median(s)), "p95_s": float(np.percentile(s, 95))}
