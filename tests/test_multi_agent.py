"""Phase 4 multi-agent system (PREREGISTRATION.md §4.9)."""
import json
import os
import re
import types

import pydantic
import pytest

import data
import llm
import multi_agent as ma
import prompts
import rag
import run_eval

# ---------------------------------------------------------------- prompt texts

def _section():
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    return text[text.index("## Phase 4 — multi-agent investigation"):text.index("## P. Post-hoc analyses")]


def _block_after(label):
    s = _section()
    return re.search(re.escape(label) + r".*?```text\n(.*?)\n```", s, re.S).group(1)


def test_prompt_texts_equal_preregistration():
    assert prompts.PLANNER_V1 == _block_after("`planner_v1`:")
    assert prompts.RESEARCHER_V1 == _block_after("`researcher_v1`, with")
    assert prompts.NEXT_QUERY_RULE_WITH == _block_after("When a next requirement exists:")
    assert prompts.NEXT_QUERY_RULE_NONE == _block_after("Otherwise:\n\n```text")[0:0] + _block_after("\nOtherwise:")
    assert prompts.VERIFIER_V1 == _block_after("`verifier_v1`:")


def _a4_block(label):
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    sec = text[text.index("### A4"):]
    return re.search(re.escape(label) + r"\s*```text\n(.*?)\n```", sec, re.S).group(1)


def test_verifier_v2_changes_only_the_two_instructions_of_amendment_a4():
    import hashlib
    assert prompts.VERIFIER_REQ_RULE_V2 == _a4_block("The v2 `requirements` instruction:")
    assert prompts.VERIFIER_QUERY_RULE_V2 == _a4_block("The v2 `corrective_query` instruction:")
    assert prompts.VERIFIER_V2 == prompts.VERIFIER_V1.replace(prompts._VERIFIER_REQ_RULE_V1, prompts.VERIFIER_REQ_RULE_V2) \
        .replace(prompts._VERIFIER_QUERY_RULE_V1, prompts.VERIFIER_QUERY_RULE_V2)
    assert hashlib.sha256(prompts.VERIFIER_V1.encode()).hexdigest() == prompts.VERIFIER_V1_HASH
    assert prompts.verifier_prompt("q", ["a"], []).startswith(prompts.VERIFIER_V2.split("{requirements}")[0])


def test_schemas_send_no_description():
    for m in (prompts.PlannerOutput, prompts.ResearcherOutput, prompts.VerifierOutput):
        assert "description" not in json.dumps(m.model_json_schema())


# ---------------------------------------------------------------- contracts

def P(*reqs):
    return json.dumps({"requirements": list(reqs)})


@pytest.mark.parametrize("raw,ok", [(P(), False), (P("a b"), True), (P("a b", "c d"), True), (P("a", "b", "c"), False),
                                    (P(" "), False), (P(" ".join(["w"] * 25)), True), (P(" ".join(["w"] * 26)), False)])
def test_planner_contract(raw, ok):
    if ok:
        prompts.PlannerOutput.model_validate_json(raw)
    else:
        with pytest.raises(pydantic.ValidationError):
            prompts.PlannerOutput.model_validate_json(raw)


def R(acc=(), nq="", finding="f"):
    return json.dumps({"accepted": list(acc), "finding": finding, "next_query": nq})


def test_researcher_contract_static_and_runtime():
    prompts.ResearcherOutput.model_validate_json(R([1, 3], "x y"))
    with pytest.raises(pydantic.ValidationError):
        prompts.ResearcherOutput.model_validate_json(R([1, 1]))
    with pytest.raises(pydantic.ValidationError):
        prompts.ResearcherOutput.model_validate_json(R([], " ".join(["w"] * 13)))
    ok = {"accepted": [1, 2], "finding": "f", "next_query": "q"}
    assert ma.check_researcher(ok, 2, True) is None
    assert ma.check_researcher({**ok, "accepted": [3]}, 2, True) == "accepted number out of range"
    assert ma.check_researcher({**ok, "accepted": [0]}, 2, True) == "accepted number out of range"
    assert ma.check_researcher({**ok, "next_query": " "}, 2, True) == "next_query required"
    assert ma.check_researcher(ok, 2, False) == "next_query must be empty"


def V(entries, sufficient, cr=0, cq=""):
    return json.dumps({"requirements": [{"supported": s, "references": [{"paragraph": p, "sentence": t} for p, t in refs]}
                                        for s, refs in entries], "sufficient": sufficient,
                       "corrective_requirement": cr, "corrective_query": cq})


def C(raw, n=2, counts=(3, 3)):
    parsed = prompts.VerifierOutput.model_validate_json(raw).model_dump()
    return prompts.verifier_contract_v2(parsed, n, list(counts))


def test_verifier_contract_v2_keeps_judgments_and_derives_control_state():
    norm, notes, err = C(V([(True, [(1, 0)]), (True, [(2, 1)])], True))
    assert err is None and norm["sufficient"] is True and notes == []
    # the LLM's aggregate is recomputed, not trusted
    norm, notes, err = C(V([(True, [(1, 0)]), (True, [(2, 1)])], False, 1, "q"))
    assert err is None and norm["sufficient"] is True and norm["corrective_requirement"] == 0 and norm["corrective_query"] == ""
    assert any("sufficient recomputed" in n for n in notes) and any("corrective fields cleared" in n for n in notes)
    # references on unsupported requirements are ignored and logged
    norm, notes, err = C(V([(True, [(1, 0)]), (False, [(2, 0)])], False, 2, "q"))
    assert err is None and norm["requirements"][1]["references"] == [] and any("ignored 1 reference" in n for n in notes)
    norm, notes, err = C(V([(False, [])], True, 0, ""), n=1)
    assert err is not None                                                  # computed insufficient: correction required


@pytest.mark.parametrize("raw,n,counts,reason", [
    (V([(True, [(1, 0)])], True), 2, (3, 3), "one entry per requirement required"),
    (V([(True, [])], True), 1, (3,), "supported=true requires at least one reference"),
    (V([(True, [(3, 0)])], True), 1, (3, 3), "invalid reference on a supported requirement"),     # paragraph not displayed
    (V([(True, [(1, 5)])], True), 1, (3,), "invalid reference on a supported requirement"),       # sentence missing
    (V([(True, [(1, 0)]), (False, [])], False, 1, "q"), 2, (3, 3), "corrective_requirement must be an unsupported requirement"),
    (V([(True, [(1, 0)]), (False, [])], False, 2, ""), 2, (3, 3), "corrective_query must be non-empty and at most 12 tok_v1 tokens"),
    (V([(False, [])], False, 1, " ".join(["w"] * 13)), 1, (3,), "corrective_query must be non-empty and at most 12 tok_v1 tokens"),
])
def test_verifier_contract_v2_hard_failures(raw, n, counts, reason):
    assert C(raw, n, counts)[2] == reason


def test_verifier_contract_v2_replay_on_recorded_outputs(tmp_path, monkeypatch):
    raw = {"k1": V([(True, [(1, 0)]), (True, [(2, 0)])], False, 1, "q"), "k2": V([(True, [(1, 0)])], True), "k3": "{bad"}
    monkeypatch.setattr(run_eval, "cached_response", lambda k: raw[k])
    rows = [{"qid": "q1", "condition": "a", "requirements": ["a", "b"],
             "calls": [{"role": "planner", "key": "p"}, {"role": "verifier", "key": "k1", "displayed": ["T1", "T2"]},
                       {"role": "verifier_recheck", "key": "k2", "displayed": ["T1"]},
                       {"role": "verifier", "key": "k3", "displayed": ["T1"]}]}]
    res = run_eval.replay_verifier_contract(rows, {"q1": {"T1": ["s"], "T2": ["s"]}})
    assert [x["failure"] for x in res] == [None, "one entry per requirement required", "schema"]
    assert res[0]["notes"] and res[0]["role"] == "verifier"


def test_overflow_order_candidates_then_accepted():
    batches = [["a1", "a2", "a3"], ["b1", "b2"]]
    assert ma.overflow_order(batches, candidates=["a1", "a3", "b2"], accepted=["a2", "b1"]) == ["a3", "a1", "b2", "a2", "b1"]


def test_search_skip_reasons():
    assert ma.search_skip_reason(None, "Q?", [], 1) == "no_query"
    assert ma.search_skip_reason("?!", "Q?", [], 1) == "no_query"
    assert ma.search_skip_reason("q", "Q?", [], 1) == "duplicate_query"
    assert ma.search_skip_reason("x y", "Q?", ["X, y"], 1) == "duplicate_query"
    assert ma.search_skip_reason("x y", "Q?", [], 3) == "max_searches"
    assert ma.search_skip_reason("x y", "Q?", [], 2) is None


# ---------------------------------------------------------------- the flow on stubs

IDS = sorted(data.corpus())[:300]


class Retriever:
    def __init__(self, per_query=None):
        self.per_query, self.calls = per_query or {}, []

    def search(self, query, k=100):
        self.calls.append(query)
        return [(d, 1.0) for d in self.per_query.get(query, IDS)[:k]]

    def describe(self):
        return {"retriever": "stub"}


class LLM:
    options = {"num_ctx": 8192, "num_predict": 512}
    ROLE = {"PlannerOutput": "planner", "ResearcherOutput": "researcher", "VerifierOutput": "verifier", "AnswerOutput": "answer"}

    def __init__(self, **scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.scripts.setdefault("answer", ['{"rationale": "r", "answer": "A", "citations": [], "supported": true}'] * 3)
        self.prompts = []

    def count_tokens(self, t):
        return len(t.split())

    def generate(self, user, schema):
        role = self.ROLE[schema.__name__]
        self.prompts.append((role, user))
        text = self.scripts[role].pop(0)
        try:
            parsed, err = schema.model_validate_json(text).model_dump(), None
        except pydantic.ValidationError:
            parsed, err = None, "schema"
        return llm.LLMResult(f"k{len(self.prompts)}", text, parsed, err, 10, 5, 1.0, "stop", False)


@pytest.fixture(scope="module")
def q():
    return data.load_split("dev_smoke")[0]


def run(q, retriever=None, **scripts):
    client = LLM(**scripts)
    rec = ma.MultiAgent(retriever or Retriever(), client).run(q)
    return rec, client


def events(rec, name=None):
    return [e for e in rec["events"] if name is None or e["event"] == name]


def test_search_1_always_runs_with_the_verbatim_question(q):
    r = Retriever()
    rec, _ = run(q, r, planner=[P("req one")], researcher=[R([1])], verifier=[V([(True, [(1, 0)])], True)])
    assert r.calls[0] == q.question and rec["batches"][0] == IDS[:10]
    assert rec["batches"][0] == rag.bm25_rag(LLM(), Retriever(), q, 10)["displayed"]       # = R's top 10
    assert not events(rec, "search_skipped")


def test_a_later_query_equal_to_the_question_is_skipped(q):
    rec, _ = run(q, planner=[P("a", "b")], researcher=[R([1], q.question.lower()), R([1])],
                 verifier=[V([(True, [(1, 0)]), (True, [(2, 0)])], True)])
    assert events(rec, "search_skipped")[0]["reason"] == "duplicate_query" and rec["searches"] == 1


def test_candidate_pool_keeps_unaccepted_gold_available(q):
    # search 1 returns both "gold" documents (IDS[0], IDS[1]); researcher 1 accepts only the first;
    # search 2 adds nothing new; researcher 2 still sees IDS[1] among the candidates and accepts it.
    r = Retriever({"next hop": IDS[:10]})
    rec, client = run(q, r, planner=[P("first fact", "second fact")],
                      researcher=[R([1], "next hop"), R([1])],
                      verifier=[V([(True, [(1, 0)]), (True, [(2, 0)])], True)])
    assert events(rec, "search")[1]["reason"] == "no_progress"
    second = [u for role, u in client.prompts if role == "researcher"][1]
    cands = second.split("Candidate paragraphs:")[1]
    assert f"[P1] {data.corpus()[IDS[1]].title}" in cands                  # the unaccepted gold is selectable
    assert f"[A1] {data.corpus()[IDS[0]].title}" in second                  # accepted evidence shown read-only
    assert rec["accepted"] == [IDS[0], IDS[1]]                               # ordered by first acceptance


def test_researcher_error_skips_search_2_but_researcher_2_still_runs(q):
    rec, _ = run(q, planner=[P("a", "b")], researcher=['{"accepted": [1, 1], "finding": "f", "next_query": ""}', R([2])],
                 verifier=[V([(True, [(1, 0)]), (True, [(1, 0)])], True)])
    assert [e["event"] for e in rec["events"]][:4] == ["plan", "search", "researcher_error", "search_skipped"]
    assert events(rec, "search_skipped")[0]["reason"] == "no_query" and events(rec, "research")[0]["requirement"] == 2


def test_no_candidates_skips_the_researcher(q):
    # search 1 returns one document, which researcher 1 accepts; search 2 and the corrective search
    # add nothing, so researcher 2 and the corrective researcher both have an empty pool.
    r = Retriever({q.question: IDS[:1], "new query": IDS[:1], "more": IDS[:1]})
    rec, client = run(q, r, planner=[P("a", "b")], researcher=[R([1], "new query")],
                      verifier=[V([(True, [(1, 0)]), (False, [])], False, 2, "more"),
                                V([(True, [(1, 0)]), (False, [])], False, 2, "again")])
    skipped = [(e["requirement"], e["stage"]) for e in events(rec, "researcher_skipped")]
    assert skipped == [(2, "planned"), (2, "corrective")]
    assert [role for role, _ in client.prompts] == ["planner", "researcher", "verifier", "verifier", "answer"]


def test_corrective_path_and_budget(q):
    rec, client = run(q, Retriever({"fix query": IDS[50:60]}), planner=[P("a", "b")],
                      researcher=[R([1], "hop two"), R([1]), R([2])],
                      verifier=[V([(True, [(1, 0)]), (False, [])], False, 2, "fix query"),
                                V([(True, [(1, 0)]), (True, [(2, 0)])], True)])
    roles = [role for role, _ in client.prompts]
    assert roles == ["planner", "researcher", "researcher", "verifier", "researcher", "verifier", "answer"]
    assert rec["llm_calls"] == 7 and rec["searches"] == 3 and len(rec["searched"]) == 20
    assert [len(b) for b in rec["batches"]] == [10, 5, 5]
    assert rec["corrective"] == (2, "fix query") and rec["recheck"]["sufficient"] is True


def test_corrective_duplicate_query_still_runs_corrective_researcher(q):
    rec, client = run(q, planner=[P("only")], researcher=[R([1]), R([2])],
                      verifier=[V([(False, [])], False, 1, q.question), V([(True, [(1, 0)])], True)])
    assert events(rec, "search_skipped")[0]["reason"] == "duplicate_query"
    assert [role for role, _ in client.prompts] == ["planner", "researcher", "verifier", "researcher", "verifier", "answer"]


def test_role_errors(q):
    rec, _ = run(q, planner=["{oops"], researcher=[R([1])], verifier=[V([(True, [(1, 0)])], True)])
    assert rec["requirements"] == [q.question] and events(rec, "planner_error")
    rec, client = run(q, planner=[P("a")], researcher=[R([1])], verifier=[V([(True, [(9, 0)])], True)])   # ungrounded
    assert events(rec, "verifier_error") and [r for r, _ in client.prompts][-1] == "answer" and rec["llm_calls"] == 4
    rec, _ = run(q, planner=[P("a")], researcher=[R([1]), R([])],
                 verifier=[V([(False, [])], False, 1, "x y"), '{"requirements": []}'])
    assert events(rec, "recheck_error")


def test_verifier_sees_no_researcher_text(q):
    rec, client = run(q, planner=[P("a", "b")], researcher=[R([1], "QUERYXYZ", "FINDINGXYZ"), R([1], "", "FINDINGABC")],
                      verifier=[V([(True, [(1, 0)]), (True, [(2, 0)])], True)])
    v = [u for role, u in client.prompts if role == "verifier"][0]
    assert "FINDINGXYZ" not in v and "FINDINGABC" not in v and "QUERYXYZ" not in v
    assert v.startswith(prompts.VERIFIER_V2.split("{requirements}")[0])


def test_synthesizer_uses_answer_v2_and_empty_evidence_fallback(q):
    rec, client = run(q, planner=[P("a")], researcher=[R([]), R([])],
                      verifier=[V([(False, [])], False, 1, "x y"), V([(False, [])], False, 1, "z")])
    syn = [u for role, u in client.prompts if role == "answer"][0]
    assert syn.startswith(prompts.ANSWER_V2.split("{paragraphs}")[0])
    assert events(rec, "no_accepted_evidence") and rec["displayed"] == IDS[:10]


def test_multi_agent_refuses_blind_splits_and_config_carries_hashes():
    for split in ("test_agents", "tune"):
        with pytest.raises(SystemExit):
            run_eval.main(["multi-agent", "--split", split])
    cfg = run_eval.multi_agent_config(Retriever())
    assert cfg["verifier_v2_hash"] == prompts.multi_agent_hashes()["verifier_v2"]
    assert cfg["answer_prompt_hash"] == prompts.artifact_hashes()["answer_v2"]
    assert cfg["budget"]["max_llm_calls"] == 7 and cfg["budget"]["max_paragraphs"] == 20
    assert run_eval.PHASE4_VERIFIER == "v2" and cfg["verifier"] == "v2"
    assert cfg["verifier_prompt_hash"] == VERIFIER_V2_HASH


def test_format_check_gates_on_synthetic_records():
    rec = {"requirements": ["a", "b"], "calls": [{"role": "planner"}, {"role": "researcher"}, {"role": "researcher"},
                                                  {"role": "verifier"}, {"role": "researcher"}, {"role": "verifier_recheck"},
                                                  {"role": "synthesizer"}],
           "events": [{"event": "plan"}, {"event": "research", "stage": "planned", "requirement": 1, "next_query": "q"},
                      {"event": "research", "stage": "planned", "requirement": 2, "next_query": None},
                      {"event": "verify", "sufficient": False},
                      {"event": "research", "stage": "corrective", "requirement": 2, "next_query": None},
                      {"event": "recheck_error"}],
           "verification": {"sufficient": True, "requirements": [{"references": [1]}]}}
    assert run_eval.role_failures(rec) == {"planner": [1, 0], "researcher": [3, 0], "verifier": [2, 1]}
    v = run_eval.ma_viability([rec])
    assert v["verifier_true_grounded"]                                     # from the initial verification
    only_recheck = {**rec, "verification": {"sufficient": False, "requirements": [{"references": []}]},
                    "recheck": {"sufficient": True, "requirements": [{"references": [{"paragraph": 1, "sentence": 0}]}]}}
    assert run_eval.ma_viability([only_recheck])["verifier_true_grounded"]  # the recheck counts (A4)
    assert v["two_requirement_plan"] and v["researcher_with_next"] and v["researcher_without_next"]
    assert v["two_requirement_path_to_verifier"] and v["verifier_false"] and v["corrective_path_with_recheck"]


# ---------------------------------------------------------------- the agent replay

AG = os.path.join(data.HERE, "results", "dev", "agent_single.jsonl")
needs = pytest.mark.skipif(not os.path.exists(AG), reason="local dev results only")


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
def test_agent_replay_aborts_on_a_cache_miss():
    client = run_eval.offline_client_for(AG)
    q = data.load_split("dev")[10]
    with pytest.raises(KeyError):
        run_eval.replay_agent(client=MissOn(client, q.question))


@needs
def test_agent_replay_reproduces_the_phase3_run():
    rows, _, _ = run_eval.replay_agent()
    assert len(rows) == 100 and all(c["cached"] for r in rows for c in r["calls"])


def test_calibration_subsets_fresh_and_disjoint():
    eda = data.load_split("eda")
    sub = run_eval.calibration_subsets(eda)
    fmt = set(list(q.qid for q in eda if q.type == "bridge")[:16] + list(q.qid for q in eda if q.type == "comparison")[:4])
    c1, c2 = ({q.qid for q in sub[k]} for k in ("c1", "c2"))
    assert len(c1) == len(c2) == 20 and not (c1 & c2) and not (c1 & fmt) and not (c2 & fmt)
    for k in ("c1", "c2"):
        assert [q.type for q in sub[k]] == ["bridge"] * 16 + ["comparison"] * 4


def test_planned_recall_ignores_corrective_stage():
    q = types.SimpleNamespace(gold=(("G1", "h"), ("G2", "h")))
    rec = {"events": [{"event": "search", "stage": "planned", "added": ["G1", "X"]},
                      {"event": "research", "stage": "planned", "accepted": ["G1", "X"]},
                      {"event": "search", "stage": "corrective", "added": ["G2"]},
                      {"event": "research", "stage": "corrective", "accepted": ["G2"]}]}
    assert run_eval.planned_recall(rec, q) == (1, 1, 2)


def test_probe_paragraphs_and_scoring():
    eda = data.load_split("eda")
    ctx = data.load_contexts("eda")
    for q in run_eval.calibration_subsets(eda)["c1"]:
        full = run_eval.probe_paragraphs(q, ctx[q.qid], True)
        part = run_eval.probe_paragraphs(q, ctx[q.qid], False)
        gold = {t for t, _ in q.gold}
        assert {t for t, _ in full} == gold and len(part) == len(full) - 1
        assert [t for t, _ in full] == [t for t in ctx[q.qid] if t in gold]          # context order
    paras = [("A", ["s0", "s1"]), ("B", ["s0"])]
    ok = prompts.VerifierOutput.model_validate_json(V([(True, [(1, 0)]), (True, [(2, 0)])], False, 1, "q")).model_dump()
    assert run_eval.score_probe(ok, 2, paras)[:2] == (True, None)
    bad = prompts.VerifierOutput.model_validate_json(V([(True, [(2, 1)])], True)).model_dump()
    assert run_eval.score_probe(bad, 1, paras)[:2] == (None, "invalid reference on a supported requirement")
    assert run_eval.score_probe(None, 1, paras)[:2] == (None, "schema")


def test_calibration_cli_parses(monkeypatch):
    seen = []
    monkeypatch.setattr(run_eval, "cmd_calib_flow", lambda a: seen.append(("flow", a.subset, a.tag)))
    monkeypatch.setattr(run_eval, "cmd_calib_probe", lambda a: seen.append(("probe", a.subset, a.tag)))
    monkeypatch.setattr("sys.argv", ["run_eval.py", "calib-flow", "--subset", "c1"])
    run_eval.main()
    monkeypatch.setattr("sys.argv", ["run_eval.py", "calib-probe", "--subset", "c2", "--tag", "v"])
    run_eval.main()
    assert seen == [("flow", "c1", None), ("probe", "c2", "v")]     # the tag is resolved by calib_tag


VERIFIER_V2_HASH = "dfa54a0f4e5430fe4d82319d1743e6e4fe32ba5a8ab3ff51112fa7b3b1e56c21"   # the ma_check_v2 and C1 prompt


def _a6_block(label):
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    sec = text[text.index("### A6"):]
    return re.search(re.escape(label) + r"\s*```text\n(.*?)\n```", sec, re.S).group(1)


def test_verifier_v1_v2_preserved_and_v3_equals_preregistration():
    import hashlib
    assert hashlib.sha256(prompts.VERIFIER_V1.encode()).hexdigest() == prompts.VERIFIER_V1_HASH
    assert hashlib.sha256(prompts.VERIFIER_V2.encode()).hexdigest() == VERIFIER_V2_HASH
    assert prompts.VERIFIER_V3 == _a6_block("`verifier_v3`:")
    assert prompts.VERIFIERS == {"v2": prompts.VERIFIER_V2, "v3": prompts.VERIFIER_V3}
    assert prompts.verifier_prompt("q", ["a"], []) == prompts.verifier_prompt("q", ["a"], [], "v2")
    with pytest.raises(ValueError):
        prompts.verifier_prompt("q", ["a"], [], "v4")


def test_verifier_v3_layout():
    u = prompts.verifier_prompt("QUESTIONX", ["first req", "second req", "third req"], [("T", ["s0", "s1"])], "v3")
    assert u.index("Question: QUESTIONX") < u.index("[R1] first req") < u.index("[P1] T")
    assert u.count("There are 3 requirements; return exactly 3 entries") == 1 and "return exactly 3 entries, judging" in u
    tail = u[u.index("Checklist:"):]
    assert "[R1] first req\n[R2] second req\n[R3] third req" in tail and u.rstrip().endswith("[R3] third req")
    assert "{" + "n}" not in u and "{checklist}" not in u and "{question}" not in u


def test_verifier_v3_examples_are_valid_and_invented():
    outs = re.findall(r"^Output: (\{.*\})$", prompts.VERIFIER_V3, re.M)
    assert len(outs) == 2
    for raw, counts, suff, corr in ((outs[0], [1, 2], True, 0), (outs[1], [2], False, 2)):
        norm, notes, err = prompts.verifier_contract_v2(prompts.VerifierOutput.model_validate_json(raw).model_dump(), 2, counts)
        assert err is None and notes == [] and norm["sufficient"] is suff and norm["corrective_requirement"] == corr
    ctx = data.load_contexts("eda")
    sub = run_eval.calibration_subsets(data.load_split("eda"))
    text = " ".join(q.question + " " + " ".join(t + " " + " ".join(s) for t, s in ctx[q.qid].items())
                    for k in ("c1", "c2") for q in sub[k])
    for name in ("Velmora", "Ostrel", "Kasdia", "Dunmere", "Hollis Brandt"):
        assert name not in text


def test_multi_agent_uses_selected_verifier(q):
    for version in ("v2", "v3"):
        client = LLM(planner=[P("a")], researcher=[R([1])], verifier=[V([(True, [(1, 0)])], True)])
        ma.MultiAgent(Retriever(), client, verifier=version).run(q)
        v = [u for role, u in client.prompts if role == "verifier"][0]
        head = prompts.VERIFIERS[version].split("{")[0]
        assert v.startswith(head) and (("Checklist:" in v) == (version == "v3"))


def test_calib_tag():
    ns = lambda **k: types.SimpleNamespace(**{"tag": None, "verifier": "v2", **k})
    assert run_eval.calib_tag(ns()) == "base" and run_eval.calib_tag(ns(verifier="v3")) == "v3"
    for bad in (ns(tag="base", verifier="v3"), ns(tag="v2x")):
        with pytest.raises(SystemExit):
            run_eval.calib_tag(bad)


def test_phase4_freeze_hashes_unchanged():
    import inspect
    import manifest
    sec = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    sec = sec[sec.index("### Phase 4 record — freeze"):]
    pinned = dict(re.findall(r"^\| `([a-z_0-9]+)`[^|]*\| `([0-9a-f]{64})` \|$", sec, re.M))
    h = prompts.multi_agent_hashes()
    for k, v in pinned.items():
        if k in h:
            assert h[k] == v, k
        elif k in ("route_after_research", "route_after_verify"):
            assert manifest.sha256_of(inspect.getsource(getattr(ma, k))) == v, k
    assert len(pinned) == 14 and pinned["verifier_v2"] == VERIFIER_V2_HASH
    src = re.search(r"`multi_agent.py` source[^|]*\| `([0-9a-f]{64})`", sec).group(1)
    assert manifest.sha256_of(inspect.getsource(ma)) == src


MA = os.path.join(data.HERE, "results", "dev", "multi_agent.jsonl")
needs_ma = pytest.mark.skipif(not os.path.exists(MA), reason="local dev results only")


@needs_ma
def test_multi_agent_replay_aborts_on_a_cache_miss():
    client = run_eval.offline_client_for(MA)
    q = data.load_split("dev")[3]
    with pytest.raises(KeyError):
        run_eval.replay_multi_agent(client=MissOn(client, q.question))


def test_stored_runs_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    assert run_eval._fresh("dev", "x_p4replay") == "x_p4replay"
    open(run_eval._path("dev", "x_p4replay"), "w").close()
    with pytest.raises(SystemExit):
        run_eval._fresh("dev", "x_p4replay")
    seen = []
    monkeypatch.setattr(run_eval, "cmd_replay_run", lambda a: seen.append((a.run, a.phase)))
    run_eval.main(["replay-run", "--run", "d3_k16", "--phase", "p4"])
    assert seen == [("d3_k16", "p4")]
    with pytest.raises(SystemExit):
        run_eval.main(["replay-run", "--run", "d3_k16"])          # the phase is required
