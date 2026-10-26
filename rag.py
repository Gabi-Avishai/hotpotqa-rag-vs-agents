"""Non-agent pipelines: BM25 RAG, oracle, closed-book, and own-context (format checks).

Every pipeline returns a plain per-question record; scoring happens separately in
``scoring.py``. The answer path is shared: render the displayed paragraphs with
``prompts.render_v1``, apply the context-overflow rule (drop the last displayed paragraph
until the prompt fits), generate with ``AnswerOutput``, and map ``[P#]`` citations back to
exact document ids. Token counts and wall time are those of the original generation,
also when a response is replayed from the cache.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import data
import llm
import prompts


def _fallback() -> Dict[str, Any]:
    return {"rationale": "", "answer": "", "citations": [], "supported": False}


def _llm_record(r: llm.LLMResult) -> Dict[str, Any]:
    return {"key": r.key, "cached": r.cached, "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens, "wall_s": r.wall_s, "done_reason": r.done_reason,
            "error": r.error}


def fit_context(client: llm.OllamaClient, question: str, docs: Sequence[data.Doc]):
    """Drop the last displayed paragraph until the answer prompt fits the context budget.
    Returns (user prompt, displayed docs, number dropped)."""
    budget = client.options["num_ctx"] - client.options["num_predict"]
    docs = list(docs)
    dropped = 0
    while True:
        user = prompts.answer_prompt(question, [(d.title, d.sentences) for d in docs])
        if client.count_tokens(llm.render_chatml(llm.DEFAULT_SYSTEM, user)) <= budget:
            return user, docs, dropped
        if not docs:
            raise llm.ContextOverflow("the answer prompt does not fit even without paragraphs")
        docs.pop()
        dropped += 1


def answer_from(client: llm.OllamaClient, question: str, docs: Sequence[data.Doc]) -> Dict[str, Any]:
    """Shared answer path for every system that shows paragraphs."""
    user, shown, dropped = fit_context(client, question, docs)
    r = client.generate(user, prompts.AnswerOutput)
    out = r.parsed if r.parsed is not None else _fallback()
    citations = []
    for c in out["citations"]:
        p, s = int(c["paragraph"]), int(c["sentence"])
        doc_id = shown[p - 1].doc_id if 1 <= p <= len(shown) else f"<invalid:P{p}>"
        citations.append([doc_id, s])
    return {
        "displayed": [d.doc_id for d in shown],
        "displayed_sentences": [len(d.sentences) for d in shown],
        "overflow_dropped": dropped,
        "output": out,
        "citations": citations,
        "schema_error": r.parsed is None,
        "rationale_tokens": client.count_tokens(out["rationale"]) if out["rationale"] else 0,
        "llm": _llm_record(r),
    }


def bm25_rag(client: llm.OllamaClient, retriever, q: data.Question, k: int, depth: int = 100) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits = retriever.search(q.question, k=depth)
    retrieval_s = time.perf_counter() - t0
    docs = data.corpus()
    rec = answer_from(client, q.question, [docs[d] for d, _ in hits[:k]])
    return {"qid": q.qid, "type": q.type, "k": k, "ranked": [d for d, _ in hits],
            "retrieval_s": retrieval_s, "retrieval_calls": 1, **rec}


def oracle(client: llm.OllamaClient, q: data.Question) -> Dict[str, Any]:
    """The two gold paragraphs (their own text variants), in supplied-context order."""
    docs, ids = data.corpus(), data.doc_id_of()
    gold = set(q.gold)
    shown = [docs[ids[c]] for c in q.context if c in gold]
    rec = answer_from(client, q.question, shown)
    return {"qid": q.qid, "type": q.type, "retrieval_s": 0.0, "retrieval_calls": 0, **rec}


def own_context(client: llm.OllamaClient, q: data.Question,
                paragraphs: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    """The question's own supplied paragraphs in supplied order (prompt-format checks on
    train questions, whose paragraphs are not in the corpus: doc ids are the titles)."""
    shown = [data.Doc(t, t, data.text_hash(s), tuple(s)) for t, s in paragraphs.items()]
    rec = answer_from(client, q.question, shown)
    return {"qid": q.qid, "type": q.type, "retrieval_s": 0.0, "retrieval_calls": 0, **rec}


def closed_book(client: llm.OllamaClient, q: data.Question) -> Dict[str, Any]:
    r = client.generate(prompts.closed_book_prompt(q.question), prompts.AnswerOutput)
    out = r.parsed if r.parsed is not None else _fallback()
    return {"qid": q.qid, "type": q.type, "retrieval_s": 0.0, "retrieval_calls": 0,
            "displayed": [], "displayed_sentences": [], "overflow_dropped": 0, "output": out,
            "citations": [], "schema_error": r.parsed is None,
            "rationale_tokens": client.count_tokens(out["rationale"]) if out["rationale"] else 0,
            "llm": _llm_record(r)}
