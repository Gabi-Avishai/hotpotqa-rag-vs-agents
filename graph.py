"""Title-mention graph and the expansion operator G (PREREGISTRATION.md §2.3).

Built from corpus titles and sentence text only. A -> T when a sentence of paragraph A
contains a permitted alias of title T (T != A's title); an edge to a title reaches every
text variant of it. The expansion operator never receives answers, supporting facts or
gold documents: only a base ranking, the graph and a lexical score lookup.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import data
from normalize import tok_v1

GRAPH_VERSION = "graph_v1+A2"
MIN_ALIAS_CHARS = 4
MAX_SINGLE_TOKEN_DF = 100        # preregistered; amendment A2 applies it to aliases of every length
STOPWORDS = frozenset(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into',
                       'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then',
                       'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with'])   # bm25s English list
_DISAMBIGUATOR = re.compile(r"\s\([^()]*\)$")

Alias = Tuple[str, ...]


def title_aliases(title: str) -> List[Alias]:
    """Full alias, and the short alias without a trailing parenthetical disambiguator."""
    out = [tuple(tok_v1(title))]
    m = _DISAMBIGUATOR.search(title)
    if m:
        out.append(tuple(tok_v1(title[:m.start()])))
    return [a for i, a in enumerate(out) if a and a not in out[:i]]


def alias_rejection(alias: Alias, df: Mapping[str, int], max_df: int = MAX_SINGLE_TOKEN_DF) -> Optional[str]:
    """Why an alias token tuple is dropped, or None if it is permitted."""
    if len(" ".join(alias)) < MIN_ALIAS_CHARS:
        return "short"
    if all(t in STOPWORDS for t in alias):
        return "stopwords_only"
    if all(t.isdigit() for t in alias):
        return "digits"
    if len(alias) == 1 and df.get(alias[0], 0) > max_df:
        return "frequent_single_token"
    return None


@dataclass
class TitleGraph:
    alias_title: Dict[Alias, str]                         # permitted, unambiguous aliases
    variants: Dict[str, List[str]]                        # title -> doc_ids (sorted)
    title_of: Dict[str, str]                              # doc_id -> title
    out: Dict[str, List[Tuple[str, int, int]]]            # doc_id -> [(title, sentence, offset)], by position
    incoming: Dict[str, List[str]]                        # title -> source doc_ids (sorted)
    stats: Dict[str, object] = field(default_factory=dict)

    def in_degree(self, doc_id: str) -> int:
        return len(self.incoming.get(self.title_of[doc_id], []))

    def targets(self, doc_id: str) -> List[Tuple[int, int, str]]:
        """Outgoing target documents as (sentence, offset, doc_id), in mention order then
        doc_id; every text variant of a linked title is a separate target."""
        return sorted((s, o, d) for t, s, o in self.out.get(doc_id, []) for d in self.variants[t])

    def sources(self, doc_id: str) -> List[str]:
        """Documents with an edge to ``doc_id``'s title."""
        return self.incoming.get(self.title_of[doc_id], [])

    def fingerprint(self) -> str:
        h = hashlib.sha256(GRAPH_VERSION.encode())
        for a in sorted(self.alias_title):
            h.update(("A\t" + " ".join(a) + "\t" + self.alias_title[a] + "\n").encode())
        for d in sorted(self.out):
            for t, s, o in self.out[d]:
                h.update(f"E\t{d}\t{t}\t{s}\t{o}\n".encode())
        return h.hexdigest()


def alias_paragraph_counts(sent_tokens: Mapping[str, Sequence[Sequence[str]]],
                           aliases: Iterable[Alias]) -> Dict[Alias, int]:
    """Amendment A2: for each alias, the number of distinct paragraphs containing its whole
    token sequence within one sentence. Every alias is counted independently (no
    longest-match suppression); repeated mentions in a paragraph count once."""
    aliases = set(aliases)
    lengths = defaultdict(set)
    for a in aliases:
        lengths[a[0]].add(len(a))
    found: Dict[Alias, set] = defaultdict(set)
    for doc_id, sentences in sent_tokens.items():
        for toks in sentences:
            for i, t in enumerate(toks):
                for n in lengths.get(t, ()):
                    cand = tuple(toks[i:i + n])
                    if len(cand) == n and cand in aliases:
                        found[cand].add(doc_id)
    return {a: len(found.get(a, ())) for a in aliases}


def match_sentence(tokens: Sequence[str], alias_title: Mapping[Alias, str],
                   lengths_by_first: Mapping[str, Sequence[int]]) -> List[Tuple[int, str]]:
    """Greedy left-to-right longest (by token count) non-overlapping matches, as
    (offset, title)."""
    out, i = [], 0
    while i < len(tokens):
        hit = None
        for n in lengths_by_first.get(tokens[i], ()):          # descending lengths
            cand = tuple(tokens[i:i + n])
            if len(cand) == n and cand in alias_title:
                hit = (n, alias_title[cand])
                break
        if hit:
            out.append((i, hit[1]))
            i += hit[0]
        else:
            i += 1
    return out


def build(docs: Optional[Iterable[data.Doc]] = None, max_df: int = MAX_SINGLE_TOKEN_DF) -> TitleGraph:
    t0 = time.perf_counter()
    docs = list(docs) if docs is not None else list(data.corpus().values())
    sent_tokens = {d.doc_id: [tok_v1(s) for s in d.sentences] for d in docs}
    df = Counter()
    for toks in sent_tokens.values():
        df.update({t for s in toks for t in s})
    variants: Dict[str, List[str]] = defaultdict(list)
    for d in docs:
        variants[d.title].append(d.doc_id)
    variants = {t: sorted(v) for t, v in variants.items()}

    owners: Dict[Alias, set] = defaultdict(set)
    rejected = Counter()
    for title in variants:
        for a in title_aliases(title):
            why = alias_rejection(a, df, max_df)
            if why:
                rejected[why] += 1
            else:
                owners[a].add(title)
    alias_title = {a: next(iter(ts)) for a, ts in owners.items() if len(ts) == 1}
    rejected["ambiguous"] = sum(1 for ts in owners.values() if len(ts) > 1)
    # Amendment A2: the frequency filter for aliases of every length, counted independently.
    counts = alias_paragraph_counts(sent_tokens, alias_title)
    frequent = {a for a, c in counts.items() if c > max_df}
    rejected["frequent_alias_A2"] = len(frequent)
    rejected["frequent_alias_A2_multi_token"] = sum(1 for a in frequent if len(a) > 1)
    alias_title = {a: t for a, t in alias_title.items() if a not in frequent}

    lengths = defaultdict(set)
    for a in alias_title:
        lengths[a[0]].add(len(a))
    lengths_by_first = {t: sorted(ns, reverse=True) for t, ns in lengths.items()}

    title_of = {d.doc_id: d.title for d in docs}
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    incoming: Dict[str, set] = defaultdict(set)
    for d in docs:
        first: Dict[str, Tuple[int, int]] = {}
        for si, toks in enumerate(sent_tokens[d.doc_id]):
            for off, t in match_sentence(toks, alias_title, lengths_by_first):
                if t != d.title and t not in first:
                    first[t] = (si, off)
        edges = sorted(((t, s, o) for t, (s, o) in first.items()), key=lambda e: (e[1], e[2], e[0]))
        if edges:
            out[d.doc_id] = edges
            for t, _, _ in edges:
                incoming[t].add(d.doc_id)
    g = TitleGraph(alias_title, variants, title_of, out, {t: sorted(s) for t, s in incoming.items()})
    indeg = [len(v) for v in g.incoming.values()]
    outdeg = [len(v) for v in out.values()]
    g.stats = {"titles": len(variants), "aliases_permitted": len(alias_title), "aliases_rejected": dict(rejected),
               "edges": sum(outdeg), "docs_with_out_edges": len(out), "titles_with_in_edges": len(g.incoming),
               "max_in_degree": max(indeg, default=0), "max_out_degree": max(outdeg, default=0),
               "build_s": time.perf_counter() - t0}
    return g


def load_or_build(cache_dir: str = os.path.join(data.HERE, "cache", "graph")) -> TitleGraph:
    """Build once per corpus fingerprint and graph version; cached locally."""
    key = hashlib.sha256((GRAPH_VERSION + data.verify_corpus()
                          + open(__file__).read() + open(os.path.join(data.HERE, "normalize.py")).read()).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{GRAPH_VERSION}_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    g = build()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(g, f)
    os.replace(path + ".tmp", path)
    return g


# ---------------------------------------------------------------- expansion operator

@dataclass(frozen=True)
class GraphConfig:
    backward: bool
    seed_depth: int
    retained: int = 5
    reserved: int = 5
    forward_cap: int = 3
    backward_cap: int = 2

    @property
    def name(self) -> str:
        return f"graph_{'bi' if self.backward else 'fwd'}_s{self.seed_depth}"


C4_5, C4_10 = GraphConfig(False, 5), GraphConfig(False, 10)
C5_5, C5_10 = GraphConfig(True, 5), GraphConfig(True, 10)


def expand(base: Sequence[str], g: TitleGraph, cfg: GraphConfig, lexical_score: Callable[[str], float],
           depth: int = 100) -> Tuple[List[str], List[Dict]]:
    """G_{dir,s}(X): retain X's top 5, fill up to 5 slots with graph candidates reached from
    X's top s seeds, then fill from X. ``lexical_score`` is the base lexical (B) first-round
    score, used only to rank and admit backward candidates. Returns the depth-``depth``
    ranking and the provenance of every added document."""
    retained = list(base[:cfg.retained])
    blocked = set(retained)
    cands: List[Dict] = []
    for rank, seed in enumerate(base[:cfg.seed_depth], 1):
        fwd = []
        for _, _, d in g.targets(seed):
            if d not in blocked and d != seed:
                fwd.append(d)
                blocked.add(d)
                if len(fwd) == cfg.forward_cap:
                    break
        cands += [{"doc": d, "seed_rank": rank, "direction": "forward", "seed_in_degree": g.in_degree(seed)} for d in fwd]
        if cfg.backward:
            back = sorted(((-lexical_score(d), d) for d in g.sources(seed)
                           if d not in blocked and d != seed and lexical_score(d) > 0))[:cfg.backward_cap]
            for _, d in back:
                blocked.add(d)
            cands += [{"doc": d, "seed_rank": rank, "direction": "backward", "seed_in_degree": g.in_degree(seed)}
                      for _, d in back]
    added = cands[:cfg.reserved]
    ranking = retained + [c["doc"] for c in added]
    placed = set(ranking)
    for d in base:
        if len(ranking) >= depth:
            break
        if d not in placed:
            ranking.append(d)
            placed.add(d)
    # the top 10 is retained + added + fill; positions beyond come from X in order
    return ranking[:depth], added
