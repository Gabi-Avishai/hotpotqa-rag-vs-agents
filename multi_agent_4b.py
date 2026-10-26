"""Phase 4b: structured selection and per-requirement verification (PREREGISTRATION.md Phase 4b).

Planner -> (search -> researcher) per requirement -> one verifier call per requirement ->
optional corrective search -> corrective researcher -> recheck of the corrective
requirement -> synthesizer (the frozen ``answer_v2``), as a LangGraph ``StateGraph``.

What differs from Phase 4 (``multi_agent.py``, frozen and unchanged):

- **Selection** (``researcher_v3``, amendment A8; ``researcher_v2`` on C3): one required boolean per candidate paragraph, so a
  decision can never be missing or misaligned. A hard failure accepts every candidate the
  call was shown (fail-open); later context overflow may still remove accepted evidence.
- **Verification** (``requirement_verifier_v1``): one call per requirement, references
  restricted to the displayed sentence labels. Code derives everything else: support
  needs a reference, ``sufficient`` is the conjunction, the corrective target is the
  lowest-numbered unsupported requirement, and a hard failure counts as unsupported.
- **No new evidence**: a corrective search that adds nothing skips the corrective
  researcher and the recheck; a corrective researcher that accepts nothing skips the
  recheck.

Transport errors (the server unreachable) are not model failures: they abort the run,
which resumes from the cache.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import data
import llm
import prompts
from multi_agent import (FIRST_TAKE, FOLLOW_TAKE, MAX_SEARCHES, SEARCH_DEPTH, candidate_ids, overflow_order,
                         search_skip_reason, seen_ids)
from normalize import tok_v1

MAX_CALLS = 8
MAX_POOL = 20
RESEARCHER = "v3"        # amendment A8; C3 ran researcher_v2


class State(TypedDict, total=False):
    question: str
    requirements: List[str]
    r: int                          # current requirement (1-based)
    batches: List[List[str]]
    accepted: List[str]
    queries: List[str]
    pending_query: Optional[str]
    searches: int
    verdicts: List[Dict[str, Any]]           # the current per-requirement verdicts
    first_verification: List[Dict[str, Any]]
    corrective: Optional[Tuple[int, str]]
    stage: str                      # "planned" or "corrective"
    events: List[Dict[str, Any]]
    calls: List[Dict[str, Any]]
    answer: Dict[str, Any]


# ---------------------------------------------------------------- deterministic pieces

def truncate_query(text: str) -> str:
    """The longest prefix of whitespace-separated words with at most 12 ``tok_v1`` tokens."""
    words, out = text.split(), []
    for w in words:
        if len(tok_v1(" ".join(out + [w]))) > prompts.MAX_QUERY_TOKENS:
            break
        out.append(w)
    return " ".join(out)


def normalize_query(query: Optional[str], fallback: str) -> Tuple[str, List[str]]:
    """A model query truncated to 12 tokens; the fallback text (truncated) when it is empty."""
    notes = []
    q = truncate_query(query or "")
    if query and q != query.strip():
        notes.append("query truncated to 12 tok_v1 tokens")
    if not tok_v1(q):
        notes.append("empty query replaced by the requirement text")
        q = truncate_query(fallback)
    return q, notes


def hard_failure(r: llm.LLMResult) -> Optional[str]:
    if r.done_reason == "length":
        return "truncated (done_reason = length)"
    if r.parsed is None:
        return r.error or "invalid output"
    return None


def researcher_decisions(parsed: Optional[Mapping[str, Any]], shown: Sequence[str], failure: Optional[str]) -> List[str]:
    """Accepted ids from one researcher call; fail-open on a hard failure."""
    if failure:
        return list(shown)
    return [d for i, d in enumerate(shown, 1) if parsed[f"P{i}"]]


def requirement_verdict(parsed: Optional[Mapping[str, Any]], failure: Optional[str], requirement: str) -> Dict[str, Any]:
    """Deterministic per-requirement verdict (§4b.4)."""
    if failure:
        q, notes = normalize_query(None, requirement)
        return {"supported": False, "references": [], "query": q, "failure": failure,
                "notes": ["hard failure counted as unsupported"] + notes}
    notes, refs = [], []
    for ref in parsed["references"]:
        if ref in refs:
            continue
        refs.append(ref)
    if len(refs) < len(parsed["references"]):
        notes.append(f"removed {len(parsed['references']) - len(refs)} duplicate reference(s)")
    supported = bool(parsed["supported"]) and bool(refs)
    if parsed["supported"] and not refs:
        notes.append("supported without a reference counted as unsupported")
    q, qn = normalize_query(parsed["query"], requirement)
    return {"supported": supported, "references": refs, "query": q, "failure": None, "notes": notes + qn}


def no_evidence_verdict(requirement: str) -> Dict[str, Any]:
    q, notes = normalize_query(None, requirement)
    return {"supported": False, "references": [], "query": q, "failure": None,
            "notes": ["no accepted evidence; verifier not called"] + notes}


def corrective_target(verdicts: Sequence[Mapping[str, Any]]) -> Optional[Tuple[int, str]]:
    """The lowest-numbered unsupported requirement and its (normalized) query; None if sufficient."""
    for i, v in enumerate(verdicts, 1):
        if not v["supported"]:
            return i, v["query"]
    return None


def call_requirement_verifier(client, question: str, requirements: Sequence[str], i: int,
                              paragraphs: Sequence[Tuple[str, Sequence[str]]]) -> Tuple[Dict[str, Any], llm.LLMResult]:
    """One ``requirement_verifier_v1`` call over already-fitted paragraphs (used by the flow and the probes)."""
    schema = prompts.requirement_verifier_schema(prompts.reference_labels(paragraphs))
    r = client.generate(prompts.requirement_verifier_prompt(question, requirements, i, paragraphs), schema)
    return requirement_verdict(r.parsed, hard_failure(r), requirements[i - 1]), r


# ---------------------------------------------------------------- routing

def route_after_research(state: State) -> str:
    return "search" if state["r"] < len(state["requirements"]) else "verify"


def route_after_verify(state: State) -> str:
    return "csearch" if state.get("corrective") else "answer"


def route_after_csearch(state: State) -> str:
    return "answer" if state["events"][-1]["event"] == "no_new_evidence" else "cresearch"


def route_after_cresearch(state: State) -> str:
    return "answer" if state["events"][-1]["event"] == "no_new_evidence" else "recheck"


# ---------------------------------------------------------------- the graph

class MultiAgent4b:
    def __init__(self, retriever, client, docs: Optional[Mapping[str, data.Doc]] = None, researcher: str = RESEARCHER):
        self.retriever, self.client, self.researcher = retriever, client, researcher
        self.docs = docs if docs is not None else data.corpus()
        self.graph = self._build()

    def _build(self):
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(State)
        for name in ("plan", "search", "research", "verify", "csearch", "cresearch", "recheck", "answer"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "plan")
        g.add_edge("plan", "search")
        g.add_edge("search", "research")
        g.add_conditional_edges("research", route_after_research, {"search": "search", "verify": "verify"})
        g.add_conditional_edges("verify", route_after_verify, {"csearch": "csearch", "answer": "answer"})
        g.add_conditional_edges("csearch", route_after_csearch, {"cresearch": "cresearch", "answer": "answer"})
        g.add_conditional_edges("cresearch", route_after_cresearch, {"recheck": "recheck", "answer": "answer"})
        g.add_edge("recheck", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def _paras(self, ids):
        return [(self.docs[d].title, self.docs[d].sentences) for d in ids]

    def _fit(self, make_prompt, accepted: List[str], candidates: List[str], batches):
        """Phase 4 overflow rule: drop candidates first, then accepted, by evidence-preserving priority."""
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

    @staticmethod
    def _rec(role: str, r: llm.LLMResult, extra=None) -> Dict[str, Any]:
        return {"role": role, "key": r.key, "cached": r.cached, "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens, "wall_s": r.wall_s, "done_reason": r.done_reason,
                "error": hard_failure(r), **(extra or {})}

    # -- nodes
    def plan(self, state: State) -> Dict[str, Any]:
        r = self.client.generate(prompts.planner_prompt(state["question"]), prompts.PlannerOutput)
        rec = self._rec("planner", r)
        if rec["error"]:
            reqs, ev = [state["question"]], {"event": "planner_error", "error": rec["error"]}
        else:
            reqs, ev = [x.strip() for x in r.parsed["requirements"]], {"event": "plan", "requirements": r.parsed["requirements"]}
        return {"requirements": reqs, "r": 0, "calls": state["calls"] + [rec], "events": state["events"] + [ev]}

    def _search(self, state: State, query: Optional[str], first: bool) -> Dict[str, Any]:
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
        events.append({"event": "search", "query": query, "added": new, "stage": state["stage"], "retrieval_s": took})
        return {"batches": state["batches"] + [new], "searches": state["searches"] + 1,
                "queries": state["queries"] + [query], "events": events}

    def search(self, state: State) -> Dict[str, Any]:
        r = state["r"] + 1
        upd = self._search(state, state["question"] if r == 1 else state.get("pending_query"), first=r == 1)
        return {**upd, "r": r, "pending_query": None}

    def _research(self, state: State, req_i: int, has_next: bool) -> Dict[str, Any]:
        """One ``researcher_v2`` call for requirement ``req_i``; returns the state update and
        the newly accepted ids."""
        requirement = state["requirements"][req_i - 1]
        next_req = state["requirements"][req_i] if has_next else None
        def skipped(reason, **extra):
            ev = {"event": "researcher_skipped", "reason": reason, "requirement": req_i, "stage": state["stage"], **extra}
            upd = {"events": state["events"] + [ev]}
            if has_next:                       # no model query: search 2 uses the next requirement's text
                upd["pending_query"], ev["next_query_notes"] = normalize_query(None, next_req)
                ev["next_query"] = upd["pending_query"]
            return upd, []

        cands = candidate_ids(state)
        if not cands:
            return skipped("no_candidates")
        user, shown_a, shown_c, dropped = self._fit(
            lambda a, c: prompts.researcher_4b_prompt(state["question"], requirement, self._paras(a), self._paras(c), next_req,
                                                      self.researcher),
            list(state["accepted"]), cands, state["batches"])
        if not shown_c:
            return skipped("no_candidates_after_overflow", dropped=dropped)
        r = self.client.generate(user, prompts.researcher_v2_schema(len(shown_c), has_next))
        rec = self._rec("researcher", r, {"requirement": req_i, "stage": state["stage"], "candidates": shown_c,
                                          "accepted_shown": shown_a, "dropped": dropped})
        newly = researcher_decisions(r.parsed, shown_c, rec["error"])
        accepted = state["accepted"] + [d for d in newly if d not in set(state["accepted"])]
        ev = {"event": "researcher_error" if rec["error"] else "research", "requirement": req_i, "stage": state["stage"],
              "accepted": newly, **({"error": rec["error"], "fail_open": True} if rec["error"] else {})}
        upd = {"accepted": accepted, "calls": state["calls"] + [rec], "events": state["events"] + [ev]}
        if has_next:
            raw = None if rec["error"] else r.parsed["next_query"]
            q, notes = normalize_query(raw, next_req)
            ev["next_query"], ev["next_query_notes"] = q, notes
            upd["pending_query"] = q
        return upd, newly

    def research(self, state: State) -> Dict[str, Any]:
        upd, _ = self._research(state, state["r"], has_next=state["r"] < len(state["requirements"]))
        return upd

    def _verify_one(self, state: State, accepted: List[str], i: int, role: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        reqs = state["requirements"]
        if not accepted:
            return {"requirement": i, **no_evidence_verdict(reqs[i - 1]), "displayed": []}, None
        user, shown, _, dropped = self._fit(
            lambda a, c: prompts.requirement_verifier_prompt(state["question"], reqs, i, self._paras(a)),
            list(accepted), [], state["batches"])
        verdict, r = call_requirement_verifier(self.client, state["question"], reqs, i, self._paras(shown))
        rec = self._rec(role, r, {"requirement": i, "displayed": shown, "dropped": dropped, "notes": verdict["notes"]})
        return {"requirement": i, **verdict, "displayed": shown}, rec

    def verify(self, state: State) -> Dict[str, Any]:
        verdicts, calls = [], list(state["calls"])
        for i in range(1, len(state["requirements"]) + 1):
            v, rec = self._verify_one(state, state["accepted"], i, "verifier")
            verdicts.append(v)
            if rec:
                calls.append(rec)
        corr = corrective_target(verdicts)
        if corr is not None and not state["accepted"]:
            corr = (1, truncate_query(state["requirements"][0]))        # empty evidence: requirement 1 and its text
        ev = {"event": "verify", "supported": [v["supported"] for v in verdicts], "sufficient": corr is None,
              "failures": [v["requirement"] for v in verdicts if v["failure"]], "corrective": corr}
        return {"verdicts": verdicts, "first_verification": verdicts, "corrective": corr, "calls": calls,
                "events": state["events"] + [ev]}

    def csearch(self, state: State) -> Dict[str, Any]:
        st = {**state, "stage": "corrective"}
        upd = self._search(st, state["corrective"][1], first=False)
        events = upd["events"]
        last = events[-1]
        if last["event"] == "search_skipped" or not last["added"]:
            events = events + [{"event": "no_new_evidence", "after": "corrective_search"}]
        return {**upd, "events": events, "stage": "corrective"}

    def cresearch(self, state: State) -> Dict[str, Any]:
        upd, newly = self._research(state, state["corrective"][0], has_next=False)
        if not [d for d in newly if d not in set(state["accepted"])]:
            upd["events"] = upd["events"] + [{"event": "no_new_evidence", "after": "corrective_researcher"}]
        return upd

    def recheck(self, state: State) -> Dict[str, Any]:
        i = state["corrective"][0]
        v, rec = self._verify_one(state, state["accepted"], i, "verifier_recheck")
        verdicts = [v if w["requirement"] == i else w for w in state["verdicts"]]
        ev = {"event": "recheck", "requirement": i, "supported": v["supported"], "failure": v["failure"],
              "sufficient": all(w["supported"] for w in verdicts)}
        return {"verdicts": verdicts, "calls": state["calls"] + ([rec] if rec else []), "events": state["events"] + [ev]}

    def answer(self, state: State) -> Dict[str, Any]:
        events = list(state["events"])
        context = list(state["accepted"])
        if not context:
            context = list(state["batches"][0]) if state["batches"] else []
            events.append({"event": "no_accepted_evidence"})
        user, shown, _, dropped = self._fit(
            lambda a, c: prompts.answer_prompt(state["question"], self._paras(a)), context, [], state["batches"])
        r = self.client.generate(user, prompts.AnswerOutput)
        rec = self._rec("synthesizer", r, {"displayed": shown, "dropped": dropped})
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
        init: State = {"question": q.question, "requirements": [], "r": 0, "batches": [], "accepted": [],
                       "queries": [], "pending_query": None, "searches": 0, "verdicts": [], "first_verification": [],
                       "corrective": None, "stage": "planned", "events": [], "calls": []}
        f = self.graph.invoke(init, {"recursion_limit": 40})
        assert len(f["calls"]) <= MAX_CALLS and f["searches"] <= MAX_SEARCHES and len(seen_ids(f)) <= MAX_POOL
        syn = f["calls"][-1]
        return {"qid": q.qid, "type": q.type, **f["answer"],
                "llm": {k: syn[k] for k in ("key", "cached", "prompt_tokens", "output_tokens", "wall_s", "done_reason", "error")},
                "requirements": f["requirements"], "batches": f["batches"], "accepted": f["accepted"],
                "queries": f["queries"], "first_verification": f["first_verification"], "verdicts": f["verdicts"],
                "sufficient": all(v["supported"] for v in f["verdicts"]),
                "corrective": f["corrective"], "events": f["events"], "calls": f["calls"],
                "searched": seen_ids(f), "searches": f["searches"], "llm_calls": len(f["calls"]),
                "prompt_tokens": sum(c["prompt_tokens"] for c in f["calls"]),
                "overflow_drops": sum(len(c.get("dropped", [])) for c in f["calls"]),
                "retrieval_s": sum(e.get("retrieval_s", 0.0) for e in f["events"] if e["event"] == "search"),
                "retrieval_calls": f["searches"]}
