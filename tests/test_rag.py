"""Pipelines with a fake LLM: overflow rule, citation mapping, fallbacks, oracle order."""
import json

import pytest

import data
import llm
import prompts
import rag


class FakeClient:
    """Counts one token per word; answers with a fixed JSON output."""

    def __init__(self, output, num_ctx=8192, num_predict=512, valid=True):
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict}
        self.output, self.valid, self.prompts = output, valid, []

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, user, schema):
        self.prompts.append(user)
        parsed = self.output if self.valid else None
        return llm.LLMResult("k", json.dumps(self.output), parsed, None if self.valid else "schema: 1 error(s)",
                             10, 5, 2.5, "stop", False)


def doc(i, n_words=5, n_sent=2):
    return data.Doc(f"D{i}", f"T{i}", f"h{i}", tuple(" ".join(["w"] * n_words) for _ in range(n_sent)))


OUT = {"rationale": "Because.", "answer": "X", "supported": True,
       "citations": [{"paragraph": 1, "sentence": 1}, {"paragraph": 3, "sentence": 0}, {"paragraph": 9, "sentence": 0}]}


def test_citations_map_to_displayed_docs_and_invalid_numbers():
    c = FakeClient(OUT)
    rec = rag.answer_from(c, "q?", [doc(1), doc(2), doc(3)])
    assert rec["citations"] == [["D1", 1], ["D3", 0], ["<invalid:P9>", 0]]
    assert rec["displayed"] == ["D1", "D2", "D3"] and rec["displayed_sentences"] == [2, 2, 2]
    assert rec["overflow_dropped"] == 0 and not rec["schema_error"]
    assert rec["llm"]["wall_s"] == 2.5 and rec["rationale_tokens"] == 1


def test_overflow_drops_last_displayed_until_it_fits():
    docs = [doc(i, n_words=40) for i in range(1, 6)]
    base = len(llm.render_chatml(llm.DEFAULT_SYSTEM, prompts.answer_prompt("q?", [])).split())
    per_doc = len(prompts.render_v1([(docs[0].title, docs[0].sentences)]).split())
    c = FakeClient(OUT, num_ctx=base + 3 * per_doc + 1 + 100, num_predict=100)   # room for three
    rec = rag.answer_from(c, "q?", docs)
    assert rec["displayed"] == ["D1", "D2", "D3"] and rec["overflow_dropped"] == 2
    # a citation of [P3] maps to the last *displayed* paragraph
    assert rec["citations"][1] == ["D3", 0]


def test_schema_failure_falls_back_to_empty_unsupported():
    rec = rag.answer_from(FakeClient(OUT, valid=False), "q?", [doc(1)])
    assert rec["schema_error"] and rec["output"] == {"rationale": "", "answer": "", "citations": [], "supported": False}
    assert rec["citations"] == [] and rec["rationale_tokens"] == 0


def test_oracle_shows_gold_in_supplied_order():
    q = data.load_split("dev_smoke")[0]
    rec = rag.oracle(FakeClient(OUT), q)
    ids = data.doc_id_of()
    want = [ids[c] for c in q.context if c in set(q.gold)]
    assert rec["displayed"] == want and len(want) == 2


def test_closed_book_has_no_context():
    q = data.load_split("dev_smoke")[0]
    c = FakeClient({"rationale": "r", "answer": "a", "citations": [], "supported": False})
    rec = rag.closed_book(c, q)
    assert rec["displayed"] == [] and "Paragraphs:" not in c.prompts[0]
