"""Phase 4 multi-agent investigation (PREREGISTRATION.md §4).

Planner -> (search -> researcher) per requirement -> verifier -> optional corrective search
-> corrective researcher -> verifier recheck -> synthesizer (the frozen ``answer_v2``), as
a LangGraph ``StateGraph``. Roles run sequentially; every call is a separate,
context-isolated call to the same model. Searches and routing are deterministic code.

Evidence state: ``batches`` (paragraphs retrieved, grouped by search), ``accepted``
(globally unique, ordered by first acceptance) and the candidate pool (seen minus
accepted). Every researcher sees the accepted evidence read-only and the whole pool as
selectable candidates.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import data
import llm
import prompts
from agent import drop_priority
from normalize import tok_v1

MAX_SEARCHES = 3
FIRST_TAKE = 10
FOLLOW_TAKE = 5
SEARCH_DEPTH = 100
MAX_CALLS = 7


class MAState(TypedDict, total=False):
    question: str
    requirements: List[str]
    r: int                          # current requirement (1-based)
    batches: List[List[str]]        # ids added by each executed search
    accepted: List[str]
    queries: List[str]              # executed (and skipped) query texts, in order
    pending_query: Optional[str]
    searches: int
    verification: Optional[Dict[str, Any]]
    recheck: Optional[Dict[str, Any]]
    corrective: Optional[Tuple[int, str]]
    stage: str                      # "planned" or "corrective"
    events: List[Dict[str, Any]]    # searches, skips, role outputs, errors
    calls: List[Dict[str, Any]]
    answer: Dict[str, Any]


# ---------------------------------------------------------------- deterministic pieces

def seen_ids(state: Mapping[str, Any]) -> List[str]:
    return [d for b in state["batches"] for d in b]


def candidate_ids(state: Mapping[str, Any]) -> List[str]:
    acc = set(state["accepted"])
    return [d for d in seen_ids(state) if d not in acc]


def search_skip_reason(query: Optional[str], question: str, previous: Sequence[str], searches: int) -> Optional[str]:
    """Routing table for search 2 and the corrective search (search 1 is exempt)."""
    if query is None or not tok_v1(query):
        return "no_query"
    if tok_v1(query) == tok_v1(question) or any(tok_v1(query) == tok_v1(p) for p in previous):
        return "duplicate_query"
    if searches >= MAX_SEARCHES:
        return "max_searches"
    return None


def overflow_order(batches: Sequence[Sequence[str]], candidates: Sequence[str], accepted: Sequence[str]) -> List[str]:
    """Phase 4 drop order: candidates first, then accepted, each by Phase 3's
    evidence-preserving priority over the searches they came from."""
    pri = drop_priority([b for b in batches if b])
    cs, ac = set(candidates), set(accepted)
    return [d for d in pri if d in cs] + [d for d in pri if d in ac]


def check_researcher(parsed: Dict[str, Any], n_candidates: int, has_next: bool) -> Optional[str]:
    """Runtime part of the researcher contract; None if valid."""
    if any(not 1 <= a <= n_candidates for a in parsed["accepted"]):
        return "accepted number out of range"
    if has_next and not parsed["next_query"].strip():
        return "next_query required"
    if not has_next and parsed["next_query"].strip():
        return "next_query must be empty"
    return None


def check_verifier(parsed: Dict[str, Any], n_requirements: int, sentence_counts: Sequence[int]):
    """The verifier's runtime contract (amendment A4.1): (normalized, notes, failure)."""
    return prompts.verifier_contract_v2(parsed, n_requirements, sentence_counts)


# ---------------------------------------------------------------- routing

def route_after_research(state: MAState) -> str:
    if state["stage"] == "corrective":
        return "recheck"
    return "search" if state["r"] < len(state["requirements"]) else "verify"


def route_after_verify(state: MAState) -> str:
    return "csearch" if state.get("corrective") else "answer"


# ---------------------------------------------------------------- the graph

class MultiAgent:
    def __init__(self, retriever, client, docs: Optional[Mapping[str, data.Doc]] = None, verifier: str = "v2"):
        self.retriever, self.client, self.verifier = retriever, client, verifier
        self.docs = docs if docs is not None else data.corpus()
        self.graph = self._build()

    def _build(self):
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(MAState)
        for name in ("plan", "search", "research", "verify", "csearch", "recheck", "answer"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "plan")
        g.add_edge("plan", "search")
        g.add_edge("search", "research")
        g.add_conditional_edges("research", route_after_research, {"search": "search", "verify": "verify", "recheck": "recheck"})
        g.add_conditional_edges("verify", route_after_verify, {"csearch": "csearch", "answer": "answer"})
        g.add_edge("csearch", "research")
        g.add_edge("recheck", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def _paras(self, ids):
        return [(self.docs[d].title, self.docs[d].sentences) for d in ids]

    def _fit(self, make_prompt, accepted: List[str], candidates: List[str], batches) -> Tuple[str, List[str], List[str], List[str]]:
        budget = self.client.options["num_ctx"] - self.client.options["num_predict"]
        dropped: List[str] = []
        queue = overflow_order(batches, candidates, accepted)
        while True:
            a = [d for d in accepted if d not in set(dropped)]
            c = [d for d in candidates if d not in set(dropped)]
            user = make_prompt(a, c)
            if self.client.count_tokens(llm.render_chatml(llm.DEFAULT_SYSTEM, user)) <= budget:
                return user, a, c, dropped
            if not queue:
                raise llm.ContextOverflow("the prompt does not fit even without paragraphs")
            dropped.append(queue.pop(0))

    def _call(self, role: str, user: str, schema, extra=None):
        r = self.client.generate(user, schema)
        rec = {"role": role, "key": r.key, "cached": r.cached, "prompt_tokens": r.prompt_tokens,
               "output_tokens": r.output_tokens, "wall_s": r.wall_s, "done_reason": r.done_reason,
               "error": r.error, **(extra or {})}
        return r, rec

    # -- nodes (each returns a partial state update)
    def plan(self, state: MAState) -> Dict[str, Any]:
        r, rec = self._call("planner", prompts.planner_prompt(state["question"]), prompts.PlannerOutput)
        if r.parsed is None:
            reqs, ev = [state["question"]], {"event": "planner_error", "error": r.error}
        else:
            reqs, ev = [x.strip() for x in r.parsed["requirements"]], {"event": "plan", "requirements": r.parsed["requirements"]}
        return {"requirements": reqs, "r": 0, "calls": state["calls"] + [rec], "events": state["events"] + [ev]}

    def _search(self, state: MAState, query: Optional[str], first: bool) -> Dict[str, Any]:
        events = list(state["events"])
        if not first:
            skip = search_skip_reason(query, state["question"], state["queries"], state["searches"])
            if skip:
                events.append({"event": "search_skipped", "reason": skip, "query": query, "stage": state["stage"]})
                return {"events": events, "queries": state["queries"] + ([query] if query else [])}
        seen = set(seen_ids(state))
        t0 = time.perf_counter()
        hits = self.retriever.search(query, k=SEARCH_DEPTH)
        took = time.perf_counter() - t0
        new = [d for d, _ in hits if d not in seen][:FIRST_TAKE if first else FOLLOW_TAKE]
        events.append({"event": "search", "query": query, "added": new, "stage": state["stage"],
                       "retrieval_s": took, **({"reason": "no_progress"} if not new else {})})
        return {"batches": state["batches"] + [new], "searches": state["searches"] + 1,
                "queries": state["queries"] + [query], "events": events}

    def search(self, state: MAState) -> Dict[str, Any]:
        r = state["r"] + 1
        if r == 1:                                       # search 1: unconditional, verbatim question
            upd = self._search(state, state["question"], first=True)
        else:
            upd = self._search(state, state.get("pending_query"), first=False)
        return {**upd, "r": r, "pending_query": None}

    def research(self, state: MAState) -> Dict[str, Any]:
        corrective = state["stage"] == "corrective"
        req_i = state["corrective"][0] if corrective else state["r"]
        requirement = state["requirements"][req_i - 1]
        has_next = (not corrective) and req_i < len(state["requirements"])
        next_req = state["requirements"][req_i] if has_next else None
        cands = candidate_ids(state)
        if not cands:
            return {"events": state["events"] + [{"event": "researcher_skipped", "reason": "no_candidates",
                                                  "requirement": req_i, "stage": state["stage"]}]}
        user, shown_a, shown_c, dropped = self._fit(
            lambda a, c: prompts.researcher_prompt(state["question"], requirement, self._paras(a), self._paras(c), next_req),
            list(state["accepted"]), cands, state["batches"])
        r, rec = self._call("researcher", user, prompts.ResearcherOutput,
                            {"requirement": req_i, "stage": state["stage"], "candidates": shown_c,
                             "accepted_shown": shown_a, "dropped": dropped})
        problem = r.error if r.parsed is None else check_researcher(r.parsed, len(shown_c), has_next)
        if problem:
            ev = {"event": "researcher_error", "requirement": req_i, "stage": state["stage"], "error": problem}
            return {"calls": state["calls"] + [rec], "events": state["events"] + [ev]}
        newly = [shown_c[a - 1] for a in r.parsed["accepted"]]
        accepted = state["accepted"] + [d for d in newly if d not in set(state["accepted"])]
        ev = {"event": "research", "requirement": req_i, "stage": state["stage"], "accepted": newly,
              "finding": r.parsed["finding"], "next_query": r.parsed["next_query"] or None}
        nq = (r.parsed["next_query"].strip() or None) if has_next else None
        return {"accepted": accepted, "pending_query": nq,
                "calls": state["calls"] + [rec], "events": state["events"] + [ev]}

    def _verify(self, state: MAState, role: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[str]]:
        user, shown, _, dropped = self._fit(
            lambda a, c: prompts.verifier_prompt(state["question"], state["requirements"], self._paras(a), self.verifier),
            list(state["accepted"]), [], state["batches"])
        r, rec = self._call(role, user, prompts.VerifierOutput, {"displayed": shown, "dropped": dropped})
        if r.parsed is None:
            return None, rec, [r.error]
        norm, notes, problem = check_verifier(r.parsed, len(state["requirements"]),
                                              [len(self.docs[d].sentences) for d in shown])
        rec["normalizations"] = notes
        out = None if problem else {**norm, "displayed": shown, "normalizations": notes}
        return out, rec, ([problem] if problem else [])

    def verify(self, state: MAState) -> Dict[str, Any]:
        out, rec, problem = self._verify(state, "verifier")
        if problem:
            ev = {"event": "verifier_error", "error": problem[0]}
            return {"verification": None, "corrective": None, "calls": state["calls"] + [rec],
                    "events": state["events"] + [ev]}
        corr = None if out["sufficient"] else (out["corrective_requirement"], out["corrective_query"])
        ev = {"event": "verify", "sufficient": out["sufficient"],
              "supported": [c["supported"] for c in out["requirements"]], "corrective": corr}
        return {"verification": out, "corrective": corr, "calls": state["calls"] + [rec],
                "events": state["events"] + [ev]}

    def csearch(self, state: MAState) -> Dict[str, Any]:
        st = {**state, "stage": "corrective"}
        upd = self._search(st, state["corrective"][1], first=False)
        return {**upd, "stage": "corrective"}

    def recheck(self, state: MAState) -> Dict[str, Any]:
        out, rec, problem = self._verify(state, "verifier_recheck")
        ev = ({"event": "recheck_error", "error": problem[0]} if problem else
              {"event": "recheck", "sufficient": out["sufficient"], "supported": [c["supported"] for c in out["requirements"]]})
        return {"recheck": out, "calls": state["calls"] + [rec], "events": state["events"] + [ev]}

    def answer(self, state: MAState) -> Dict[str, Any]:
        events = list(state["events"])
        context = list(state["accepted"])
        if not context:
            context = list(state["batches"][0]) if state["batches"] else []
            events.append({"event": "no_accepted_evidence"})
        user, shown, _, dropped = self._fit(
            lambda a, c: prompts.answer_prompt(state["question"], self._paras(a)), context, [], state["batches"])
        r, rec = self._call("synthesizer", user, prompts.AnswerOutput, {"displayed": shown, "dropped": dropped})
        out = r.parsed if r.parsed is not None else {"rationale": "", "answer": "", "citations": [], "supported": False}
        citations = []
        for c in out["citations"]:
            p, s = int(c["paragraph"]), int(c["sentence"])
            citations.append([shown[p - 1] if 1 <= p <= len(shown) else f"<invalid:P{p}>", s])
        ans = {"displayed": shown, "displayed_sentences": [len(self.docs[d].sentences) for d in shown],
               "overflow_dropped": len(dropped), "dropped_ids": dropped, "output": out, "citations": citations,
               "schema_error": r.parsed is None,
               "rationale_tokens": self.client.count_tokens(out["rationale"]) if out["rationale"] else 0}
        return {"answer": ans, "calls": state["calls"] + [rec], "events": events}

    def run(self, q: data.Question) -> Dict[str, Any]:
        init: MAState = {"question": q.question, "requirements": [], "r": 0, "batches": [], "accepted": [],
                         "queries": [], "pending_query": None, "searches": 0, "verification": None, "recheck": None,
                         "corrective": None, "stage": "planned", "events": [], "calls": []}
        f = self.graph.invoke(init, {"recursion_limit": 30})
        assert len(f["calls"]) <= MAX_CALLS and f["searches"] <= MAX_SEARCHES and len(seen_ids(f)) <= 20
        syn = f["calls"][-1]
        return {"qid": q.qid, "type": q.type, **f["answer"],
                "llm": {k: syn[k] for k in ("key", "cached", "prompt_tokens", "output_tokens", "wall_s", "done_reason", "error")},
                "requirements": f["requirements"], "batches": f["batches"], "accepted": f["accepted"],
                "queries": f["queries"], "verification": f["verification"], "recheck": f["recheck"],
                "corrective": f["corrective"], "events": f["events"], "calls": f["calls"],
                "searched": seen_ids(f), "searches": f["searches"], "llm_calls": len(f["calls"]),
                "retrieval_s": sum(e.get("retrieval_s", 0.0) for e in f["events"] if e["event"] == "search"),
                "retrieval_calls": f["searches"]}
