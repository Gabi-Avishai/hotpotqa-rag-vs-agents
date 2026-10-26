"""Cached replay of the five case studies (reports/case_studies.md). No model calls.

    python demo.py --list
    python demo.py --case graph_success
    python demo.py --all [--verbose]

Reads the stored result and ranking files under ``results/`` and the locally cached
dataset, only. It never creates an LLM client, loads an embedding model, rebuilds a
retrieval index, or uses the network. The cases, systems and source files are fixed in
``demo_cases.toml``, and every pinned value there is checked against the stored records
before a case is shown.

**Exact replay depends on the original local artifacts.** ``results/`` is not tracked; on
a machine without those files the demo stops with a list of what is missing.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the dataset comes from the local cache
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")

import argparse
import hashlib
import json
import sys
import textwrap
import tomllib
from typing import Any, Dict, List, Mapping, Optional

import data

try:                                                    # keep cache notices off the demo screen
    import datasets
    datasets.logging.set_verbosity_error()
except Exception:                                       # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "demo_cases.toml")
RESULTS = os.path.join(HERE, "results")
BANNER = "CACHED REPLAY — NO MODEL CALLS"
POST_HOC = "Post hoc qualitative example, selected after the results were inspected (not statistically representative)."
WIDTH = 92


# ---------------------------------------------------------------- loading

def load_cases(path: str = CASES_PATH) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        cases = tomllib.load(f)["case"]
    return sorted(cases, key=lambda c: c["number"])


def artifact_paths(case: Mapping[str, Any], results: str = RESULTS) -> Dict[str, str]:
    return {name: os.path.join(results, rel) for name, rel in case["sources"].items()}


def check_artifacts(case: Mapping[str, Any], results: str = RESULTS) -> Dict[str, str]:
    """The case's stored artifacts, or a clear stop when any is absent."""
    paths = artifact_paths(case, results)
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"demo case {case['id']!r} cannot be replayed: missing stored artifact(s):\n  "
            + "\n  ".join(missing)
            + "\nExact replay needs the original local results/ directory (it is not tracked in git). "
              "A fresh run would generate new outputs, which are not guaranteed to match.")
    return paths


def record(path: str, qid: str) -> Dict[str, Any]:
    with open(path) as f:
        for line in f:
            if qid in line:
                r = json.loads(line)
                if r["qid"] == qid:
                    return r
    raise SystemExit(f"{path}: no record for {qid}")


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


class Context:
    """The question, its gold evidence and corpus titles, from the local dataset cache."""

    def __init__(self, split: str, qid: str):
        qs = data.load_split(split, unblind=split.startswith("test_"))
        self.q = next(q for q in qs if q.qid == qid)
        self.corpus = data.corpus()
        ids = data.doc_id_of()
        self.gold_ids = [ids[g] for g in self.q.gold]
        self.gold_titles = [self.corpus[d].title for d in self.gold_ids]

    def title(self, d: str) -> str:
        return self.corpus[d].title if d in self.corpus else d

    def gold_sentence(self, title: str) -> str:
        d = self.gold_ids[self.gold_titles.index(title)]
        sents = [self.corpus[d].sentences[s].strip() for t, s in self.q.supporting_facts
                 if t == title and s < len(self.corpus[d].sentences)]
        return " ".join(sents)

    def rank(self, ranking, title: str) -> Optional[int]:
        for i, d in enumerate(ranking, 1):
            if self.title(d) == title:
                return i
        return None


def gather(case: Mapping[str, Any], results: str = RESULTS) -> Dict[str, Dict[str, Any]]:
    paths = check_artifacts(case, results)
    return {name: record(p, case["qid"]) for name, p in paths.items()}


# ---------------------------------------------------------------- pins

def _ranking(case, recs, system):
    if case["kind"] == "rerank":
        return recs["X1"]["ranking"] if system == "X1" else recs["D3"]["rankings"][system]
    return recs[system]["ranked"]


def pin_problems(case: Mapping[str, Any], recs, ctx: Context) -> List[str]:
    """Every pinned value that the stored records do not reproduce."""
    p, out = case["pins"], []
    for s, ans in p.get("answers", {}).items():
        if recs[s]["output"]["answer"] != ans:
            out.append(f"answer of {s}")
    for s, ranks in p.get("gold_ranks", {}).items():
        for title, r in ranks.items():
            if ctx.rank(_ranking(case, recs, s), title) != r:
                out.append(f"rank of {title!r} in {s}")
    for s, titles in p.get("not_retrieved", {}).items():
        for title in titles:
            if ctx.rank(_ranking(case, recs, s), title) is not None:
                out.append(f"{title!r} unexpectedly retrieved by {s}")
    if "graph_added" in p:
        added = {ctx.title(a["doc"]) for a in recs["D3"]["added"]["D3"]}
        out += [f"{t!r} not graph-added" for t in p["graph_added"] if t not in added]
    agent = next((s for s in case["systems"] if s in ("agent_single", "multi_agent")), None)
    if "follow_up_queries" in p and recs[agent]["queries"][1:] != p["follow_up_queries"]:
        out.append("follow-up queries")
    if "stop_reason" in p and recs[agent].get("stop_reason") != p["stop_reason"]:
        out.append("stop reason")
    if "requirements" in p and recs[agent]["requirements"] != p["requirements"]:
        out.append("requirements")
    if "accepted" in p and [ctx.title(d) for d in recs[agent]["accepted"]] != p["accepted"]:
        out.append("accepted paragraphs")
    if "corrective_query" in p:
        v = next(e for e in recs[agent]["events"] if e["event"] == "verify")
        if not v.get("corrective") or v["corrective"][1] != p["corrective_query"]:
            out.append("corrective query")
    if "shown_not_accepted" in p:
        t = p["shown_not_accepted"]["title"]
        calls = [c for c in recs[agent]["calls"] if c["role"] == "researcher" and t in [ctx.title(d) for d in c["candidates"]]]
        if len(calls) != p["shown_not_accepted"]["researcher_calls"] or t in [ctx.title(d) for d in recs[agent]["accepted"]]:
            out.append(f"{t!r} shown-but-not-accepted count")
    return out


# ---------------------------------------------------------------- rendering

def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


def _list(ctx: Context, ids, limit: int = 10, marks: Optional[Mapping[str, str]] = None) -> List[str]:
    out = []
    for i, d in enumerate(ids[:limit], 1):
        t = ctx.title(d)
        tag = " ★ gold" if t in ctx.gold_titles else ""
        extra = (marks or {}).get(t, "")
        out.append(f"    {i:2d}. {t}{tag}{extra}")
    return out


def _answer(rec, verbose: bool) -> List[str]:
    out = [f"    answer: {rec['output']['answer']!r}"]
    if verbose:
        out.append(_wrap("rationale: " + rec["output"]["rationale"]))
    return out


def _gold_status(ctx: Context, ranking, k: Optional[int] = None) -> str:
    parts = []
    for t in ctx.gold_titles:
        r = ctx.rank(ranking, t)
        parts.append(f"{t}: {'not retrieved' if r is None else f'rank {r}'}"
                     + ("" if r is None or k is None or r <= k else f" (outside top {k})"))
    return "    gold: " + "; ".join(parts)


def render(case: Mapping[str, Any], results: str = RESULTS, verbose: bool = False) -> str:
    recs = gather(case, results)
    ctx = Context(case["split"], case["qid"])
    problems = pin_problems(case, recs, ctx)
    if problems:
        raise SystemExit(f"case {case['id']!r}: stored records do not reproduce the pinned values: {problems}")
    L = case["labels"]
    out = ["=" * WIDTH, f"  {BANNER}", "=" * WIDTH,
           f"Case {case['number']}. {case['title']}", "",
           _wrap(f"Q: {ctx.q.question}", ""), f"Gold answer: {ctx.q.answer}", "Gold evidence:"]
    out += [_wrap(f"[{t}] {ctx.gold_sentence(t)}") for t in ctx.gold_titles]
    out.append("")
    kind = case["kind"]
    if kind == "rag_pair":
        for s in case["systems"]:
            r = recs[s]
            out += [f"{L[s]}:", _gold_status(ctx, r["ranked"], r["k"]), "    top 10 retrieved:"]
            out += _list(ctx, r["ranked"], r["k"])
            out += _answer(r, verbose) + [""]
    elif kind == "rerank":
        d3, x1 = recs["D3"], recs["X1"]
        added = {ctx.title(a["doc"]): a for a in d3["added"]["D3"]}
        out += [f"BM25 alone: {_gold_status(ctx, d3['rankings']['B']).strip()}",
                f"BM25 + TAS-B fused (before the graph): {_gold_status(ctx, d3['rankings']['H']).strip()}", "",
                f"{L['D3']}:", _gold_status(ctx, d3["rankings"]["D3"], 10), "    top 10:"]
        out += _list(ctx, d3["rankings"]["D3"], 10,
                     {t: f"  [graph-added: {a['direction']} link from seed rank {a['seed_rank']}]" for t, a in added.items()})
        out += ["", f"{L['X1']}:", _gold_status(ctx, x1["ranking"], 10), "    top 10:"]
        out += _list(ctx, x1["ranking"], 10) + [""]
    elif kind == "agent":
        rag, ag = recs["d3_k10"], recs["agent_single"]
        out += [f"{L['d3_k10']}:", _gold_status(ctx, rag["ranked"], 10)] + _answer(rag, verbose) + ["", f"{L['agent_single']}:"]
        for i, (q, b) in enumerate(zip(ag["queries"], ag["batches"]), 1):
            label = "the question (= R's top 10)" if i == 1 else repr(q)
            gold_new = [ctx.title(d) for d in b if ctx.title(d) in ctx.gold_titles]
            out.append(f"    search {i}: {label} -> +{len(b)} paragraphs" + (f"  ★ adds gold: {', '.join(gold_new)}" if gold_new else ""))
            if verbose or i > 1:
                out += [f"        {ctx.title(d)}" for d in b]
        out.append(f"    stopped: {ag['stop_reason']}; {len(ag['displayed'])} paragraphs given to the answer step")
        if case.get("distractor"):
            d = next(x for x in ag["displayed"] if ctx.title(x) == case["distractor"])
            hit = next((s for s in ctx.corpus[d].sentences if ag["output"]["answer"] in s), "")
            out.append(_wrap(f"distractor in context: [{case['distractor']}] {hit.strip()}"))
        out += _answer(ag, verbose) + [""]
    elif kind == "multi":
        rag, m = recs["d3_k10"], recs["multi_agent"]
        out += [f"{L['d3_k10']}:", _gold_status(ctx, rag["ranked"], 10)] + _answer(rag, verbose) + ["", f"{L['multi_agent']}:"]
        out += [f"    requirement {i}: {r}" for i, r in enumerate(m["requirements"], 1)]
        pool = [d for b in m["batches"] for d in b]
        out.append(f"    search pool: {len(pool)} paragraphs; gold in pool: "
                   + ", ".join(t for t in ctx.gold_titles if t in [ctx.title(d) for d in pool]))
        for c in m["calls"]:
            if c["role"] == "researcher":
                cands = [ctx.title(d) for d in c["candidates"]]
                ev = next(e for e in m["events"] if e["event"] in ("research", "researcher_error") and e.get("requirement") == c["requirement"]
                          and e.get("stage") == c["stage"])
                shown_gold = [t for t in ctx.gold_titles if t in cands]
                out.append(f"    researcher (requirement {c['requirement']}, {c['stage']}): {len(cands)} candidates"
                           + (f", gold shown: {', '.join(shown_gold)}" if shown_gold else "")
                           + f" -> accepted {[ctx.title(d) for d in ev.get('accepted') or []]}")
            elif c["role"] in ("verifier", "verifier_recheck"):
                ev = next(e for e in m["events"] if e["event"] == ("verify" if c["role"] == "verifier" else "recheck")) \
                    if any(e["event"] in ("verify", "recheck") for e in m["events"]) else None
                if ev and ev["event"] == "verify":
                    corr = ev.get("corrective")
                    out.append(f"    verifier: supported {ev['supported']}" + (f"; corrective query {corr[1]!r}" if corr else ""))
        out.append(f"    accepted evidence given to the synthesizer: {[ctx.title(d) for d in m['accepted']]}")
        out += _answer(m, verbose) + [""]
    if case.get("annotation"):
        out += [_wrap("Note: " + case["annotation"], ""), ""]
    out += [_wrap("Takeaway: " + case["takeaway"], ""), "",
            "-" * WIDTH, _wrap(POST_HOC, ""),
            f"Provenance: split {case['split']}, question {case['qid']}; pinned values verified against:"]
    out += [f"    results/{rel}  (sha256 {_sha(p)}…)" for (_, rel), p in zip(case["sources"].items(), artifact_paths(case, results).values())]
    if verbose:
        out += ["", "Full stored trace:"]
        for name, r in recs.items():
            out.append(f"  [{name}] " + json.dumps({k: r[k] for k in r if k in ("displayed", "queries", "events", "stop_reason", "requirements")},
                                                  ensure_ascii=False)[:4000])
    out.append(f"  {BANNER}")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=f"{BANNER}: the five case studies, replayed from stored results.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--case")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="also print rationales and the full stored trace")
    ap.add_argument("--results", default=RESULTS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cases = load_cases()
    if args.list:
        print(BANNER)
        for c in cases:
            print(f"  {c['number']}. {c['id']:20s} {c['title']}  [{c['split']}; {' vs '.join(c['labels'][s] for s in c['systems'])}]")
        return
    chosen = cases if args.all else [c for c in cases if c["id"] == args.case]
    if not chosen:
        raise SystemExit(f"unknown case {args.case!r}; try --list")
    print("\n\n".join(render(c, args.results, args.verbose) for c in chosen))


if __name__ == "__main__":
    main()
