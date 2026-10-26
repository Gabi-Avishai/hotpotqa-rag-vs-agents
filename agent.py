"""Phase 3 single investigation agent (PREREGISTRATION.md §3).

A LangGraph ``StateGraph`` with three nodes. The LLM only returns JSON; searches and every
transition are deterministic code.

- ``search``: D3 on the current query; search 1 (the question verbatim) takes the top 10
  unseen documents, later searches the top 5 unseen.
- ``assess`` (``assess_v1``): runs after searches 1 and 2 only; decides whether to stop
  or which query to search next.
- ``answer`` (the frozen ``answer_v2``): answers from every paragraph seen, in display
  order.

Budget: at most 3 searches, 20 paragraphs and 3 LLM calls. Prompts that exceed the context
budget drop paragraphs in the evidence-preserving priority of §3.2.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import data
import llm
import prompts
from normalize import tok_v1

MAX_SEARCHES = 3
FIRST_TAKE = 10
FOLLOW_TAKE = 5
SEARCH_DEPTH = 100


class AgentState(TypedDict, total=False):
    question: str
    queries: List[str]
    batches: List[List[str]]        # ids added by each search, in rank order
    assessments: List[Dict[str, Any]]
    next_query: Optional[str]
    stop_reason: Optional[str]
    calls: List[Dict[str, Any]]      # every LLM call: kind, key, tokens, wall, dropped ids
    answer: Dict[str, Any]
    trace: List[Dict[str, Any]]


# ---------------------------------------------------------------- deterministic pieces

def drop_priority(batches: Sequence[Sequence[str]]) -> List[str]:
    """Order in which paragraphs are dropped on overflow: search 1 from its lowest rank
    upward, then earlier follow-up batches (lowest rank first), then the newest batch."""
    if not batches:
        return []
    order = list(reversed(batches[0]))
    if len(batches) > 1:
        for b in batches[1:-1]:
            order += list(reversed(b))
        order += list(reversed(batches[-1]))
    return order


def fit(client, question: str, batches: Sequence[Sequence[str]], make_prompt) -> Tuple[str, List[str], List[str]]:
    """(prompt, displayed ids in display order, dropped ids) under the overflow rule."""
    corpus = data.corpus()
    display = [d for b in batches for d in b]
    budget = client.options["num_ctx"] - client.options["num_predict"]
    dropped: List[str] = []
    queue = drop_priority(batches)
    while True:
        shown = [d for d in display if d not in set(dropped)]
        user = make_prompt(question, [(corpus[d].title, corpus[d].sentences) for d in shown])
        if client.count_tokens(llm.render_chatml(llm.DEFAULT_SYSTEM, user)) <= budget:
            return user, shown, dropped
        if not queue:
            raise llm.ContextOverflow("the prompt does not fit even without paragraphs")
        dropped.append(queue.pop(0))


def same_query(a: str, b: str) -> bool:
    return tok_v1(a) == tok_v1(b)


def route_after_search(state: AgentState) -> str:
    return "answer" if state.get("stop_reason") else "assess"


def decide_after_assess(state: AgentState, parsed: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """(stop_reason or None, next query or None), first matching rule of §3.2."""
    if parsed is None:
        return "assess_error", None
    if parsed["sufficient"]:
        return "sufficient", None
    q = parsed["next_query"]
    if not tok_v1(q):
        return "no_query", None
    if same_query(q, state["question"]) or any(same_query(q, p) for p in state["queries"]):
        return "duplicate_query", None
    return None, q


def route_after_assess(state: AgentState) -> str:
    return "answer" if state.get("stop_reason") else "search"


# ---------------------------------------------------------------- the graph

class Agent:
    def __init__(self, retriever, client):
        self.retriever, self.client = retriever, client
        self.graph = self._build()

    def _build(self):
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(AgentState)
        g.add_node("search", self.search)
        g.add_node("assess", self.assess)
        g.add_node("answer", self.answer)
        g.add_edge(START, "search")
        g.add_conditional_edges("search", route_after_search, {"assess": "assess", "answer": "answer"})
        g.add_conditional_edges("assess", route_after_assess, {"search": "search", "answer": "answer"})
        g.add_edge("answer", END)
        return g.compile()

    # nodes return partial state updates
    def search(self, state: AgentState) -> Dict[str, Any]:
        first = not state["batches"]
        query = state["question"] if first else state["next_query"]
        seen = {d for b in state["batches"] for d in b}
        t0 = time.perf_counter()
        hits = self.retriever.search(query, k=SEARCH_DEPTH)
        took = time.perf_counter() - t0
        new = [d for d, _ in hits if d not in seen][:FIRST_TAKE if first else FOLLOW_TAKE]
        batches = state["batches"] + [new]
        stop = None
        if not new:
            stop = "no_progress"
        elif len(batches) == MAX_SEARCHES:
            stop = "max_searches"
        trace = state["trace"] + [{"node": "search", "query": query, "added": new, "retrieval_s": took}]
        return {"queries": state["queries"] + [query], "batches": batches, "stop_reason": stop,
                "next_query": None, "trace": trace}

    def _call(self, kind: str, state: AgentState, make_prompt, schema):
        user, shown, dropped = fit(self.client, state["question"], state["batches"], make_prompt)
        r = self.client.generate(user, schema)
        rec = {"kind": kind, "key": r.key, "cached": r.cached, "prompt_tokens": r.prompt_tokens,
               "output_tokens": r.output_tokens, "wall_s": r.wall_s, "done_reason": r.done_reason,
               "error": r.error, "displayed": shown, "dropped": dropped}
        return r, rec, shown, dropped

    def assess(self, state: AgentState) -> Dict[str, Any]:
        r, rec, _, _ = self._call("assess", state, prompts.assess_prompt, prompts.AssessOutput)
        stop, nq = decide_after_assess(state, r.parsed)
        a = {"output": r.parsed, "raw": r.text if r.parsed is None else None, "error": r.error,
             "stop": stop, "next_query": nq}
        trace = state["trace"] + [{"node": "assess", "sufficient": None if r.parsed is None else r.parsed["sufficient"],
                                   "next_query": nq, "stop": stop, "dropped": rec["dropped"]}]
        return {"assessments": state["assessments"] + [a], "calls": state["calls"] + [rec],
                "stop_reason": stop, "next_query": nq, "trace": trace}

    def answer(self, state: AgentState) -> Dict[str, Any]:
        r, rec, shown, dropped = self._call("answer", state, prompts.answer_prompt, prompts.AnswerOutput)
        corpus = data.corpus()
        out = r.parsed if r.parsed is not None else {"rationale": "", "answer": "", "citations": [], "supported": False}
        citations = []
        for c in out["citations"]:
            p, s = int(c["paragraph"]), int(c["sentence"])
            citations.append([shown[p - 1] if 1 <= p <= len(shown) else f"<invalid:P{p}>", s])
        ans = {"displayed": shown, "displayed_sentences": [len(corpus[d].sentences) for d in shown],
               "overflow_dropped": len(dropped), "dropped_ids": dropped, "output": out, "citations": citations,
               "schema_error": r.parsed is None,
               "rationale_tokens": self.client.count_tokens(out["rationale"]) if out["rationale"] else 0}
        trace = state["trace"] + [{"node": "answer", "dropped": dropped}]
        return {"answer": ans, "calls": state["calls"] + [rec], "trace": trace}

    def run(self, q: data.Question) -> Dict[str, Any]:
        init: AgentState = {"question": q.question, "queries": [], "batches": [], "assessments": [],
                            "next_query": None, "stop_reason": None, "calls": [], "trace": []}
        final = self.graph.invoke(init, {"recursion_limit": 20})
        ans = final["answer"]
        answer_call = final["calls"][-1]
        return {"qid": q.qid, "type": q.type, **ans,
                "llm": {k: answer_call[k] for k in ("key", "cached", "prompt_tokens", "output_tokens", "wall_s",
                                                   "done_reason", "error")},
                "queries": final["queries"], "batches": final["batches"], "assessments": final["assessments"],
                "stop_reason": final["stop_reason"], "calls": final["calls"], "trace": final["trace"],
                "searched": [d for b in final["batches"] for d in b],
                "retrieval_s": sum(t["retrieval_s"] for t in final["trace"] if t["node"] == "search"),
                "retrieval_calls": len(final["batches"]), "llm_calls": len(final["calls"])}
