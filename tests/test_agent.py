"""Phase 3 agent: the assess contract, transitions, budget, overflow, search rule,
replays, K_match, refusal and manifests (PREREGISTRATION.md §3.7)."""
import json
import os
import re

import pydantic
import pytest

import agent
import data
import llm
import prompts
import rag
import run_eval


def _prereg_assess_block():
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    section = text[text.index("### 3.3 The assess prompt"):]
    return re.search(r"Prompt:\s*```text\n(.*?)\n```", section, re.S).group(1)


def test_assess_text_equals_preregistration():
    assert prompts.ASSESS_V1 == _prereg_assess_block()


def _prereg_a3_rule():
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    section = text[text.index("### A3"):]
    return re.search(r"The v2 instruction:\s*```text\n(.*?)\n```", section, re.S).group(1)


def test_assess_v2_changes_only_the_query_instruction_as_in_amendment_a3():
    assert prompts.QUERY_RULE_V2 == _prereg_a3_rule()
    assert prompts.ASSESS_V2 == prompts.ASSESS_V1.replace(prompts._QUERY_RULE_V1, prompts.QUERY_RULE_V2)
    diff = [(a, b) for a, b in zip(prompts.ASSESS_V1.splitlines(), prompts.ASSESS_V2.splitlines()) if a != b]
    assert len(diff) == 2 and all("next_query" in a or "relevant entity" in a for a, _ in diff)
    assert prompts.assess_prompt("q", []).startswith(prompts.ASSESS_V2.split("{paragraphs}")[0])


def test_assess_v1_is_preserved():
    import hashlib
    assert hashlib.sha256(prompts.ASSESS_V1.encode()).hexdigest() == prompts.ASSESS_V1_HASH
    assert set(prompts.assess_hashes()) == {"assess_v2", "assess_schema_v1", "assess_contract_v1"}


def test_assess_schema_sends_no_description_and_has_generation_order():
    schema = prompts.AssessOutput.model_json_schema()
    assert list(schema["properties"]) == ["justification", "sufficient", "missing_info", "next_query"]
    assert "description" not in json.dumps(schema)


def J(sufficient, missing="", query="", just="x"):
    return json.dumps({"justification": just, "sufficient": sufficient, "missing_info": missing, "next_query": query})


@pytest.mark.parametrize("raw,ok", [
    (J(True), True),
    (J(True, missing="the date"), False),                           # sufficient with a non-empty field
    (J(True, query="David Cronenberg birth"), False),
    (J(False, missing="the date", query="David Cronenberg birth date"), True),
    (J(False, missing="", query="David Cronenberg"), False),         # insufficient with an empty field
    (J(False, missing="the date", query="   "), False),
    (J(False, missing="m", query=" ".join(["w"] * 12)), True),       # 12 tok_v1 tokens accepted
    (J(False, missing="m", query=" ".join(["w"] * 13)), False),      # 13 rejected
    (J(False, missing="m", query="Spider-Man " * 6 + "x"), False),   # length counted with tok_v1 (13 tokens)
])
def test_assess_contract(raw, ok):
    if ok:
        prompts.AssessOutput.model_validate_json(raw)
    else:
        with pytest.raises(pydantic.ValidationError):
            prompts.AssessOutput.model_validate_json(raw)


def st(question="What is Q?", queries=("What is Q?",)):
    return {"question": question, "queries": list(queries)}


def test_decide_after_assess_rules_in_order():
    ok = lambda **k: {"sufficient": False, "missing_info": "m", "next_query": "x", **k}
    assert agent.decide_after_assess(st(), None) == ("assess_error", None)
    assert agent.decide_after_assess(st(), {"sufficient": True, "missing_info": "", "next_query": ""}) == ("sufficient", None)
    assert agent.decide_after_assess(st(), ok(next_query="?!")) == ("no_query", None)              # empty after tok_v1
    assert agent.decide_after_assess(st(), ok(next_query="what is q")) == ("duplicate_query", None)  # = question
    s = st(queries=("What is Q?", "David Cronenberg birth date"))
    assert agent.decide_after_assess(s, ok(next_query="david  cronenberg, birth-date")) == ("duplicate_query", None)
    assert agent.decide_after_assess(s, ok(next_query="Naked Lunch director")) == (None, "Naked Lunch director")


def test_drop_priority():
    b1 = [f"a{i}" for i in range(1, 11)]
    assert agent.drop_priority([b1]) == list(reversed(b1))
    assert agent.drop_priority([b1, ["f1", "f2"], ["g1", "g2"]]) == list(reversed(b1)) + ["f2", "f1", "g2", "g1"]
    assert agent.drop_priority([b1, ["f1", "f2"]]) == list(reversed(b1)) + ["f2", "f1"]   # newest batch protected


# ---------------------------------------------------------------- stubs

class Words:
    options = {"num_ctx": 8192, "num_predict": 512}

    def count_tokens(self, text):
        return len(text.split())


class StubRetriever:
    """Ranks corpus documents in a fixed order per query; ``per_query`` overrides."""

    def __init__(self, per_query=None):
        self.ids = sorted(data.corpus())[:200]
        self.per_query = per_query or {}

    def search(self, q, k=100):
        order = self.per_query.get(q, self.ids)
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(order[:k])]

    def describe(self):
        return {"retriever": "stub"}


class StubLLM(Words):
    """Answers assess calls from a script and answer calls with a fixed valid answer,
    validating through the real schemas."""

    def __init__(self, assess_script, answer='{"rationale": "r", "answer": "A", "citations": [], "supported": true}'):
        self.script, self.answer_json, self.prompts = list(assess_script), answer, []

    def generate(self, user, schema):
        self.prompts.append((schema.__name__, user))
        text = self.script.pop(0) if schema is prompts.AssessOutput else self.answer_json
        try:
            parsed, err = schema.model_validate_json(text).model_dump(), None
        except pydantic.ValidationError as e:
            parsed, err = None, "schema"
        return llm.LLMResult("k" + str(len(self.prompts)), text, parsed, err, 10, 5, 1.0, "stop", False)


@pytest.fixture(scope="module")
def q():
    return data.load_split("dev_smoke")[0]


def run(script, q, retriever=None):
    client = StubLLM(script)
    rec = agent.Agent(retriever or StubRetriever(), client).run(q)
    return rec, client


def test_sufficient_at_first_assess_equals_R_context(q):
    rec, client = run([J(True)], q)
    assert rec["stop_reason"] == "sufficient" and len(rec["batches"]) == 1 and rec["llm_calls"] == 2
    r_ctx = rag.bm25_rag(StubLLM([]), StubRetriever(), q, 10)["displayed"]
    assert rec["batches"][0] == r_ctx == rec["displayed"]                     # search 1 = R's K=10 context
    assert [n for n, _ in client.prompts] == ["AssessOutput", "AnswerOutput"]
    assert client.prompts[0][1].startswith(prompts.ASSESS_V2.split("{paragraphs}")[0])
    assert client.prompts[1][1].startswith(prompts.ANSWER_V2.split("{paragraphs}")[0])


def test_three_searches_twenty_paragraphs_three_calls_no_assess_after_last(q):
    ids = sorted(data.corpus())[:200]
    per = {"first follow up": ids[50:120], "second follow up": ids[100:180]}
    rec, client = run([J(False, "m1", "first follow up"), J(False, "m2", "second follow up")], q, StubRetriever(per))
    assert rec["stop_reason"] == "max_searches" and len(rec["batches"]) == 3
    assert [len(b) for b in rec["batches"]] == [10, 5, 5] and len(rec["displayed"]) == 20 == len(set(rec["displayed"]))
    assert rec["llm_calls"] == 3 and [n for n, _ in client.prompts] == ["AssessOutput", "AssessOutput", "AnswerOutput"]
    assert rec["queries"] == [q.question, "first follow up", "second follow up"]


def test_no_progress_stops_when_search_adds_nothing(q):
    ids = sorted(data.corpus())[:200]
    rec, _ = run([J(False, "m", "same docs")], q, StubRetriever({"same docs": ids[:10]}))
    assert rec["stop_reason"] == "no_progress" and len(rec["batches"]) == 2 and rec["batches"][1] == []
    assert rec["llm_calls"] == 2


def test_duplicate_and_no_query_and_errors_never_search(q):
    import dataclasses
    q = dataclasses.replace(q, question="Who directed the film Naked Lunch?")
    for script, reason in ([J(False, "m", "who directed  the FILM naked-lunch")], "duplicate_query"),  \
                          ([J(False, "m", "?? !!")], "no_query"),                                   \
                          (['{"justification": "x", "sufficient": true'], "assess_error"),          \
                          ([J(True, missing="should be empty")], "assess_error"):                   # contract violation
        rec, client = run(script, q)
        assert rec["stop_reason"] == reason and len(rec["batches"]) == 1, reason
        assert [n for n, _ in client.prompts] == ["AssessOutput", "AnswerOutput"]


def test_trace_is_deterministic(q):
    strip = lambda rec: json.dumps({k: v for k, v in rec.items() if k not in ("trace", "retrieval_s")}, sort_keys=True)
    script = [J(False, "m", "follow"), J(True)]
    a, _ = run(list(script), q, StubRetriever({"follow": sorted(data.corpus())[40:80]}))
    b, _ = run(list(script), q, StubRetriever({"follow": sorted(data.corpus())[40:80]}))
    assert strip(a) == strip(b)
    assert [t["node"] for t in a["trace"]] == ["search", "assess", "search", "assess", "answer"]
    assert a["stop_reason"] == "sufficient" and a["llm_calls"] == 3


def test_overflow_is_evidence_preserving_and_logged(q):
    ids = sorted(data.corpus())[:200]
    corpus = data.corpus()
    batches = [ids[:10], ids[10:15], ids[15:20]]
    base = len(llm.render_chatml(llm.DEFAULT_SYSTEM, prompts.answer_prompt(q.question, [])).split())
    per = {d: len(prompts.render_v1([(corpus[d].title, corpus[d].sentences)]).split()) for d in ids[:20]}
    keep_all_but = lambda n: sum(per[d] for d in agent.drop_priority(batches)[n:])
    client = Words()
    client.options = {"num_ctx": base + keep_all_but(12) + 100, "num_predict": 100}   # room once 12 are dropped
    user, shown, dropped = agent.fit(client, q.question, batches, prompts.answer_prompt)
    assert dropped == list(reversed(ids[:10])) + [ids[14], ids[13]]     # search 1 first, then the earlier follow-up
    assert set(ids[15:20]) <= set(shown)                                  # the newest batch is protected
    assert shown == [d for d in ids[:20] if d in set(shown)]              # display order kept


def test_answer_record_uses_retained_ids_for_final_context(q, monkeypatch):
    rec, _ = run([J(True)], q)
    assert rec["displayed"] == rec["calls"][-1]["displayed"] and rec["dropped_ids"] == rec["calls"][-1]["dropped"]


def test_k_match_rounds_half_up_and_clips():
    mk = lambda *ns: [{"displayed": ["x"] * n} for n in ns]
    assert run_eval.k_match(mk(12, 13)) == 13          # 12.5 -> 13
    assert run_eval.k_match(mk(12, 12, 13)) == 12      # 12.33 -> 12
    assert run_eval.k_match(mk(9, 9)) == 10 and run_eval.k_match(mk(20, 20, 20)) == 20


def test_agent_refuses_blind_and_tune_splits():
    for split in ("test_agents", "test_rag", "tune"):
        with pytest.raises(SystemExit):
            run_eval.main(["agent", "--split", split])


def test_agent_manifest_config_carries_hashes_budget_and_retriever():
    cfg = run_eval.agent_config(StubRetriever())
    assert cfg["assess_v2_hash"] == prompts.assess_hashes()["assess_v2"]
    assert cfg["answer_prompt_hash"] == prompts.artifact_hashes()["answer_v2"]
    assert cfg["budget"] == {"max_searches": 3, "first_take": 10, "follow_take": 5, "max_llm_calls": 3,
                             "max_paragraphs": 20, "max_query_tokens": 12}
    assert len(cfg["agent_source_hash"]) == 64 and cfg["retriever"] == {"retriever": "stub"}


# ---------------------------------------------------------------- replays

PHASE1 = os.path.join(data.HERE, "results", "dev", "bm25_k10.jsonl")
needs = pytest.mark.skipif(not os.path.exists(PHASE1), reason="local dev results only")


class MissOn:
    options = {"num_ctx": 8192, "num_predict": 512}

    def __init__(self, real, fragment):
        self.real, self.fragment = real, fragment

    def count_tokens(self, t):
        return self.real.count_tokens(t)

    def generate(self, user, schema):
        if self.fragment in user:
            raise KeyError("cache miss in offline mode")
        return self.real.generate(user, schema)


@needs
def test_replay_aborts_on_one_cache_miss():
    client = run_eval.offline_client_for(PHASE1)
    q = data.load_split("dev")[55]
    with pytest.raises(KeyError):
        run_eval.replay_run("bm25_k10", client=MissOn(client, q.question))


@needs
def test_replay_aborts_on_one_ranking_mismatch():
    import retrieval
    real = retrieval.BM25Retriever(retrieval.SELECTED)
    target = data.load_split("dev")[3].question

    class Swapped:
        def search(self, text, k=100):
            hits = real.search(text, k)
            return hits[::-1] if text == target else hits

        def describe(self):
            return real.describe()
    with pytest.raises(RuntimeError, match="differs"):
        run_eval.replay_run("bm25_k10", client=run_eval.offline_client_for(PHASE1), retriever=Swapped())


@needs
def test_bm25_replay_reproduces_everything_from_cache():
    rows, _, _ = run_eval.replay_run("bm25_k10")
    assert len(rows) == 100 and all(r["llm"]["cached"] for r in rows)
