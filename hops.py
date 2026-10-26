"""Evidence-chain structure of a question: which gold paragraph is the first hop.

The rule is the one defined and reported in ``00_eda.ipynb`` (Phase 0). For a bridge
question, the first rule that separates the two gold paragraphs decides; ``rule`` records
which one did:

1. ``supporting``: the paragraph whose supporting sentences name the other's title is hop 1;
2. ``paragraph``: the same test over the full paragraph text;
3. ``question``: the paragraph whose title the question names is hop 1;
4. ``answer``: the paragraph containing the answer is hop 2;
otherwise ``undetermined`` (the hops are then never labelled from paragraph order).

Comparison questions have two entity paragraphs and no hop order (``rule`` is None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from metrics import normalize_answer


def mentions(text: str, title: str) -> bool:
    """Case-insensitive mention of ``title`` in ``text``, ignoring a trailing
    disambiguator such as "(1926 film)"."""
    return title.split(" (")[0].lower() in text.lower()


def is_yes_no(answer: str) -> bool:
    return normalize_answer(answer) in ("yes", "no")


def contains_answer(answer: str, text: str) -> bool:
    """Whole-token containment of the normalised answer."""
    a, t = normalize_answer(answer).split(), normalize_answer(text).split()
    return any(t[i:i + len(a)] == a for i in range(len(t) - len(a) + 1)) if a else False


@dataclass(frozen=True)
class Chain:
    first: str                 # gold title; for comparison, the first in sorted order
    second: str
    rule: Optional[str]        # decision rule for bridge questions; None for comparison

    @property
    def determined(self) -> bool:
        return self.rule not in (None, "undetermined")


def chain(question, paragraphs: Mapping[str, Sequence[str]]) -> Chain:
    """Hop order for ``question`` (a ``data.Question``) given its own paragraphs
    ``{title: sentences}``."""
    a, b = sorted(t for t, _ in question.gold)
    if question.type != "bridge":
        return Chain(a, b, None)
    sup = {t: " ".join(paragraphs[t][i] for tt, i in question.supporting_facts
                       if tt == t and i < len(paragraphs[t])) for t in (a, b)}
    full = {t: " ".join(paragraphs[t]) for t in (a, b)}
    has_ans = {t: (not is_yes_no(question.answer)) and contains_answer(question.answer, full[t]) for t in (a, b)}
    tests = (("supporting", lambda x, y: mentions(sup[x], y)),
             ("paragraph", lambda x, y: mentions(full[x], y)),
             ("question", lambda x, y: mentions(question.question, x)),
             ("answer", lambda x, y: has_ans[y]))
    for rule, first in tests:
        fa, fb = first(a, b), first(b, a)
        if fa != fb:
            return Chain(a, b, rule) if fa else Chain(b, a, rule)
    return Chain(a, b, "undetermined")


def evidence_bucket(question, chain_: Chain, gold_in_context: Mapping[str, bool]) -> str:
    """Taxonomy bucket for which gold paragraphs (by title) reached the context."""
    got = [t for t in (chain_.first, chain_.second) if gold_in_context[t]]
    if len(got) == 2:
        return "both"
    if not got:
        return "neither"
    if question.type != "bridge":
        return "only_one_entity"
    if not chain_.determined:
        return "only_one_hop_undetermined"
    return "only_hop1" if got[0] == chain_.first else "only_hop2"
