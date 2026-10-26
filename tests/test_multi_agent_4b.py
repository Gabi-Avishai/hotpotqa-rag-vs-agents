"""Phase 4b: structured selection and per-requirement verification (PREREGISTRATION.md §4b.11)."""
import json
import os
import re
import types

import pydantic
import pytest

import data
import llm
import manifest
import multi_agent_4b as m4
import prompts
import run_eval
from normalize import tok_v1


def _section():
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    return text[text.index("## Phase 4b"):text.index("## P. Post-hoc analyses")]


def _block(label):
    return re.search(re.escape(label) + r"\s*```text\n(.*?)\n```", _section(), re.S).group(1)


# ---------------------------------------------------------------- prompts and schemas

def test_prompt_texts_equal_preregistration_and_section_approved():
    assert prompts.RESEARCHER_V2 == _block("`researcher_v2`:")
    assert prompts.REQUIREMENT_VERIFIER_V1 == _block("`requirement_verifier_v1`:")
    assert _section().startswith("## Phase 4b — structured selection and verification (approved)")


def test_researcher_prompt_rendering():
    cands = [("T1", ["a"]), ("T2", ["b", "c"]), ("T3", ["d"])]
    u = prompts.researcher_4b_prompt("QX?", "req one", [("ACC", ["s"])], cands, "req two", version="v2")
    assert u.index("Question: QX?") < u.index("Requirement: req one") < u.index("[A1] ACC") < u.index("[P1] T1")
    assert "[P1] to [P3]" in u and '"P1" to "P3"' in u and "{n}" not in u
    assert prompts._fill(prompts.NEXT_QUERY_RULE_WITH, next_requirement="req two") in u
    last = prompts.researcher_4b_prompt("QX?", "req one", [], cands, version="v2")
    assert "Accepted evidence (already selected; read only):\n(none)" in last and "next_query" not in last


def test_requirement_verifier_prompt_rendering():
    u = prompts.requirement_verifier_prompt("QX?", ["first", "second"], 2, [("T1", ["a", "b"])])
    assert "All requirements (for context):\n1. first\n2. second" in u
    assert "Requirement to check now: second" in u and "[P1] T1\n  [0] a\n  [1] b" in u


def test_researcher_schema_required_booleans_and_no_extra_keys():
    s = prompts.researcher_v2_schema(3, True).model_json_schema()
    assert s["required"] == ["P1", "P2", "P3", "next_query"] and s["additionalProperties"] is False
    assert all(s["properties"][f"P{i}"]["type"] == "boolean" for i in (1, 2, 3))
    s2 = prompts.researcher_v2_schema(2, False).model_json_schema()
    assert s2["required"] == ["P1", "P2"] and "next_query" not in s2["properties"]
    M = prompts.researcher_v2_schema(2, False)
    for bad in ('{"P1": true}', '{"P1": true, "P2": false, "x": 1}'):
        with pytest.raises(pydantic.ValidationError):
            M.model_validate_json(bad)


def test_verifier_schema_enum_maxitems_uniqueitems_and_duplicates_pass_validation():
    paras = [("A", ["x", "y"]), ("B", ["z"])]
    labels = prompts.reference_labels(paras)
    assert labels == ["P1:S0", "P1:S1", "P2:S0"]
    V = prompts.requirement_verifier_schema(labels)
    refs = V.model_json_schema()["properties"]["references"]
    assert refs["items"]["enum"] == labels and refs["maxItems"] == 3 and refs["uniqueItems"] is True
    assert V.model_json_schema()["additionalProperties"] is False
    assert V.model_validate_json('{"references": ["P1:S0", "P1:S0"], "supported": true, "query": ""}').references == ["P1:S0", "P1:S0"]
    for bad in ('{"references": ["P3:S0"], "supported": true, "query": ""}',
                '{"references": ["P1:S0", "P1:S1", "P2:S0", "P1:S0"], "supported": true, "query": ""}',
                '{"references": [], "supported": true, "query": "", "x": 1}'):
        with pytest.raises(pydantic.ValidationError):
            V.model_validate_json(bad)


# ---------------------------------------------------------------- deterministic rules

def test_truncate_and_normalize_query():
    long = " ".join(f"w{i}" for i in range(20))
    assert len(tok_v1(m4.truncate_query(long))) == 12
    q, notes = m4.normalize_query(long, "fallback")
    assert len(tok_v1(q)) == 12 and "truncated" in notes[0]
    q, notes = m4.normalize_query("   ", "the requirement text")
    assert q == "the requirement text" and any("replaced" in n for n in notes)
    assert m4.normalize_query(None, "fb")[0] == "fb"


def test_requirement_verdict_rules():
    v = m4.requirement_verdict({"references": ["P1:S0", "P1:S0", "P2:S0"], "supported": True, "query": ""}, None, "r")
    assert v["supported"] and v["references"] == ["P1:S0", "P2:S0"] and any("duplicate" in n for n in v["notes"])
    v = m4.requirement_verdict({"references": [], "supported": True, "query": "q"}, None, "r")
    assert not v["supported"] and any("without a reference" in n for n in v["notes"])
    v = m4.requirement_verdict({"references": ["P1:S0"], "supported": False, "query": ""}, None, "req text")
    assert not v["supported"] and v["query"] == "req text"
    v = m4.requirement_verdict(None, "schema: 1 error(s)", "req text")
    assert not v["supported"] and v["failure"] and v["query"] == "req text"


def test_corrective_target_is_lowest_unsupported():
    mk = lambda s, q="q": {"supported": s, "query": q}
    assert m4.corrective_target([mk(True), mk(True)]) is None
    assert m4.corrective_target([mk(True), mk(False, "b"), mk(False, "c")]) == (2, "b")
    assert m4.corrective_target([mk(False, "a"), mk(False, "b")]) == (1, "a")


def test_researcher_fail_open():
    assert m4.researcher_decisions(None, ["a", "b"], "schema") == ["a", "b"]
    assert m4.researcher_decisions({"P1": False, "P2": True}, ["a", "b"], None) == ["b"]


def test_hard_failure_includes_truncation():
    mk = lambda parsed, reason: llm.LLMResult("k", "{}", parsed, None if parsed else "schema", 1, 1, 0.0, reason, False)
    assert m4.hard_failure(mk({"a": 1}, "length")).startswith("truncated")
    assert m4.hard_failure(mk(None, "stop")) == "schema"
    assert m4.hard_failure(mk({"a": 1}, "stop")) is None


# ---------------------------------------------------------------- the flow on stubs

DOCS = {f"D{i}": data.Doc(f"D{i}", f"Title {i}", data.text_hash([f"s{i}a", f"s{i}b"]), (f"s{i}a", f"s{i}b"))
        for i in range(1, 31)}


class Retriever:
    def __init__(self, per_query=None, default=None):
        self.per_query, self.calls = per_query or {}, []
        self.default = default if default is not None else [f"D{i}" for i in range(1, 31)]

    def search(self, query, k=100):
        self.calls.append(query)
        return [(d, 1.0) for d in self.per_query.get(query, self.default)[:k]]


class LLM:
    """Scripted responses per role; a script entry may be a callable of (user, schema)."""
    options = {"num_ctx": 8192, "num_predict": 512}
    ROLE = {"PlannerOutput": "planner", "ResearcherV2Output": "researcher",
            "RequirementVerifierOutput": "verifier", "AnswerOutput": "answer"}

    def __init__(self, **scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.scripts.setdefault("answer", ['{"rationale": "r", "answer": "A", "citations": [], "supported": true}'])
        self.prompts = []

    def count_tokens(self, t):
        return len(t.split())

    def generate(self, user, schema):
        role = self.ROLE[schema.__name__]
        self.prompts.append((role, user, schema))
        item = self.scripts[role].pop(0)
        text, reason = item(user, schema) if callable(item) else item, "stop"
        if isinstance(text, tuple):
            text, reason = text
        try:
            parsed, err = schema.model_validate_json(text).model_dump(), None
        except pydantic.ValidationError:
            parsed, err = None, "schema: 1 error(s)"
        return llm.LLMResult(f"k{len(self.prompts)}", text, parsed, err, 10, 5, 1.0, reason, False)


def P(*reqs):
    return json.dumps({"requirements": list(reqs)})


def R(keep=(), next_query=None):
    """Researcher output keeping the candidate numbers in ``keep``; sized from the schema."""
    def make(user, schema):
        n = sum(1 for k in schema.model_fields if k.startswith("P"))
        out = {f"P{i}": i in keep for i in range(1, n + 1)}
        if "next_query" in schema.model_fields:
            out["next_query"] = next_query if next_query is not None else "follow up query"
        return json.dumps(out)
    return make


def V(supported, refs=("P1:S0",), query="fix query"):
    return json.dumps({"references": list(refs) if supported else [], "supported": supported,
                       "query": "" if supported else query})


Q = types.SimpleNamespace(qid="q1", type="bridge", question="What is the question here?")


def run(retriever=None, **scripts):
    client = LLM(**scripts)
    rec = m4.MultiAgent4b(retriever or Retriever(), client, docs=DOCS).run(Q)
    return rec, client


def events(rec, name):
    return [e for e in rec["events"] if e["event"] == name]


def roles(client):
    return [r for r, _, _ in client.prompts]


def test_sufficient_path_two_requirements():
    rec, client = run(planner=[P("a", "b")], researcher=[R([1], "next q"), R([2])],
                      verifier=[V(True), V(True, ("P2:S1",))])
    assert roles(client) == ["planner", "researcher", "researcher", "verifier", "verifier", "answer"]
    assert rec["llm_calls"] == 6 and rec["sufficient"] and not events(rec, "no_new_evidence")
    assert rec["batches"] == [[f"D{i}" for i in range(1, 11)], [f"D{i}" for i in range(11, 16)]]
    assert rec["accepted"] == ["D1", "D3"]       # researcher 2's candidates: D2..D10, then D11..D15
    assert rec["queries"] == [Q.question, "next q"]


def test_researcher_two_runs_after_a_skipped_search_two():
    rec, client = run(planner=[P("a", "b")], researcher=[R([1], Q.question), R([2])], verifier=[V(True), V(True)])
    assert events(rec, "search_skipped")[0]["reason"] == "duplicate_query"
    assert roles(client).count("researcher") == 2 and rec["searches"] == 1


def test_corrective_path_rechecks_only_the_corrective_requirement():
    retr = Retriever(per_query={"fix query": [f"D{i}" for i in range(16, 31)]})
    rec, client = run(retr, planner=[P("a", "b")], researcher=[R([1]), R([1]), R([1])],
                      verifier=[V(True), V(False), V(True, ("P1:S0",))])
    assert roles(client) == ["planner", "researcher", "researcher", "verifier", "verifier", "researcher", "verifier", "answer"]
    assert rec["llm_calls"] == 8 == m4.MAX_CALLS and rec["searches"] == 3
    recheck = [p for p in client.prompts if p[0] == "verifier"][-1][1]
    assert "Requirement to check now: b" in recheck
    assert [v["supported"] for v in rec["first_verification"]] == [True, False]
    assert [v["supported"] for v in rec["verdicts"]] == [True, True] and rec["sufficient"]
    assert rec["corrective"] == (2, "fix query")


def test_no_new_evidence_skips_corrective_researcher_and_recheck():
    # the corrective query duplicates an earlier one: the search is skipped
    rec, client = run(planner=[P("a", "b")], researcher=[R([1], "dup q"), R([1])],
                      verifier=[V(True), V(False, query="dup q")])
    assert events(rec, "search_skipped")[-1]["stage"] == "corrective" and events(rec, "no_new_evidence")
    assert roles(client) == ["planner", "researcher", "researcher", "verifier", "verifier", "answer"]
    # the corrective search adds nothing new
    retr = Retriever(per_query={"fix query": ["D1", "D2"]})
    rec, client = run(retr, planner=[P("a", "b")], researcher=[R([1]), R([1])], verifier=[V(True), V(False)])
    assert events(rec, "no_new_evidence")[0]["after"] == "corrective_search"
    assert roles(client).count("researcher") == 2 and roles(client).count("verifier") == 2


def test_corrective_researcher_accepting_nothing_skips_recheck():
    retr = Retriever(per_query={"fix query": [f"D{i}" for i in range(16, 31)]})
    rec, client = run(retr, planner=[P("a", "b")], researcher=[R([1]), R([1]), R([])], verifier=[V(True), V(False)])
    assert events(rec, "no_new_evidence")[0]["after"] == "corrective_researcher"
    assert roles(client) == ["planner", "researcher", "researcher", "verifier", "verifier", "researcher", "answer"]
    assert run_eval.mechanics_violations(rec) == []


def test_researcher_hard_failure_accepts_every_shown_candidate():
    rec, _ = run(planner=[P("a")], researcher=['{"broken": true}'], verifier=[V(True)])
    assert events(rec, "researcher_error")[0]["fail_open"] and len(rec["accepted"]) == 10


def test_truncated_researcher_counts_as_a_hard_failure():
    rec, _ = run(planner=[P("a")], researcher=[lambda u, s: (R([1])(u, s), "length")], verifier=[V(True)])
    assert events(rec, "researcher_error") and len(rec["accepted"]) == 10


def test_verifier_hard_failure_counts_as_unsupported_and_corrects():
    retr = Retriever(per_query={"b": [f"D{i}" for i in range(16, 31)]})
    rec, client = run(retr, planner=[P("a", "b")], researcher=[R([1]), R([1]), R([1])],
                      verifier=[V(True), "{bad", V(True)])
    assert rec["first_verification"][1]["failure"] and rec["corrective"] == (2, "b")   # fallback: the requirement text
    assert rec["sufficient"]


def test_recheck_hard_failure_leaves_the_requirement_unsupported():
    retr = Retriever(per_query={"fix query": [f"D{i}" for i in range(16, 31)]})
    rec, _ = run(retr, planner=[P("a", "b")], researcher=[R([1]), R([1]), R([1])], verifier=[V(True), V(False), "{bad"])
    assert events(rec, "recheck")[0]["failure"] and not rec["sufficient"]
    assert [v["supported"] for v in rec["verdicts"]] == [True, False]


def test_empty_evidence_targets_requirement_one_with_its_text():
    retr = Retriever(per_query={"a long first requirement": [f"D{i}" for i in range(16, 31)]})
    rec, client = run(retr, planner=[P("a long first requirement", "b")], researcher=[R([]), R([]), R([])])
    assert roles(client).count("verifier") == 0 and rec["corrective"] == (1, "a long first requirement")
    assert all(not v["supported"] and "no accepted evidence" in v["notes"][0] for v in rec["first_verification"])
    assert events(rec, "no_new_evidence")[0]["after"] == "corrective_researcher"
    assert events(rec, "no_accepted_evidence")


def test_skipped_researcher_with_next_requirement_uses_its_text_as_query():
    retr = Retriever(default=[f"D{i}" for i in range(1, 11)], per_query={"b": [f"D{i}" for i in range(1, 16)]})
    client = LLM(planner=[P("a", "b")], researcher=[R(list(range(1, 11)), "q2"), R([1])], verifier=[V(True), V(True)])
    agent = m4.MultiAgent4b(retr, client, docs=DOCS)
    rec = agent.run(Q)
    assert rec["queries"][1] == "q2"            # researcher 1 ran; search 2 with its query
    st = {"question": Q.question, "requirements": ["a", "b"], "r": 1, "batches": [["D1"]], "accepted": ["D1"],
          "queries": [Q.question], "searches": 1, "stage": "planned", "events": [], "calls": []}
    upd, _ = agent._research(st, 1, has_next=True)
    assert upd["pending_query"] == "b" and upd["events"][-1]["reason"] == "no_candidates"


def test_verifier_sees_only_accepted_paragraphs_and_no_researcher_text():
    rec, client = run(planner=[P("a")], researcher=[R([2])], verifier=[V(True)])
    v = [u for role, u, _ in client.prompts if role == "verifier"][0]
    assert "[P1] Title 2" in v and "Title 1\n" not in v and "follow up" not in v
    labels = [s for role, _, s in client.prompts if role == "verifier"][0].model_json_schema()
    assert labels["properties"]["references"]["items"]["enum"] == ["P1:S0", "P1:S1"]


# ---------------------------------------------------------------- gate measures

def _flow(**kw):
    rec = {"qid": "q1", "events": [], "first_verification": [], "calls": [], "schema_error": False,
           "llm_calls": 5, "searches": 2, "searched": ["a"]}
    rec.update(kw)
    return rec


def test_selection_counts_use_display_after_overflow():
    q = types.SimpleNamespace(gold=(("G1", "h"), ("G2", "h")))
    key = lambda d: (d, "h")
    rec = _flow(events=[{"event": "search", "stage": "planned", "added": ["G1", "G2", "X"]},
                        {"event": "research", "stage": "planned", "accepted": ["G1", "G2", "X"]},
                        {"event": "search", "stage": "corrective", "added": ["G3"]}],
                first_verification=[{"displayed": ["G1", "X"]}, {"displayed": ["G1", "G2", "X"]}])
    c = run_eval.selection_counts(rec, q, key)
    assert c == {"gold_in_pool": 2, "gold_accepted": 2, "gold_displayed": 1, "accepted": 3, "displayed_first": 2}
    rec["first_verification"] = [{"displayed": []}, {"displayed": []}]          # verifier not called
    assert run_eval.selection_counts(rec, q, key)["gold_displayed"] == 0


def test_mechanics_violations():
    ok = _flow(events=[{"event": "no_new_evidence", "after": "corrective_search"}])
    assert run_eval.mechanics_violations(ok) == []
    bad = _flow(events=[{"event": "no_new_evidence", "after": "corrective_search"},
                        {"event": "research", "stage": "corrective"}, {"event": "recheck"}])
    assert len(run_eval.mechanics_violations(bad)) == 2
    assert run_eval.mechanics_violations(_flow(llm_calls=9)) == ["budget exceeded"]


def test_gate_report_and_role_failures():
    q = types.SimpleNamespace(qid="q1", gold=(("G1", "h"),))
    flows = [_flow(events=[{"event": "search", "stage": "planned", "added": ["G1"]},
                           {"event": "research", "stage": "planned", "accepted": ["G1"]}],
                   first_verification=[{"displayed": ["G1"]}],
                   calls=[{"role": "planner", "error": None}, {"role": "researcher", "error": None},
                          {"role": "verifier", "error": "schema"}])]
    probes = [{"qid": "q1", "probe": "complete", "sufficient": True, "calls": [{"failure": None, "notes": []}]},
              {"qid": "q1", "probe": "incomplete", "sufficient": False, "calls": [{"failure": None, "notes": []}]}]
    refs = {"q1": {"label": "correct"}}
    rep = run_eval.gate_report(flows, [q], lambda q: (lambda d: (d, "h")), probes, refs)
    assert rep["G2"] == 1.0 and rep["S3"] == 1.0 and rep["S4"] == 1.0 and rep["roles"]["verifier"] == [1, 1]
    assert rep["gates"] == {"G1": False, "G2": True, "G5": True, "G3": True, "G4": True}
    assert rep["G1_original_definition"] is False


def test_g1_amended_planner_is_diagnostic():
    """Amendment A8: one planner failure fails G1 only under the original definition."""
    q = types.SimpleNamespace(qid="q1", gold=(("G1", "h"),))
    flows = [_flow(events=[{"event": "search", "stage": "planned", "added": ["G1"]},
                           {"event": "research", "stage": "planned", "accepted": ["G1"]}],
                   first_verification=[{"displayed": ["G1"]}],
                   calls=[{"role": "planner", "error": "schema"}, {"role": "researcher", "error": None},
                          {"role": "verifier", "error": None}])]
    rep = run_eval.gate_report(flows, [q], lambda q: (lambda d: (d, "h")))
    assert rep["gates"]["G1"] is True and rep["G1_original_definition"] is False and rep["roles"]["planner"] == [1, 1]
    flows[0]["schema_error"] = True
    assert run_eval.gate_report(flows, [q], lambda q: (lambda d: (d, "h")))["gates"]["G1"] is False


# ---------------------------------------------------------------- subsets, runtime, commands

def test_calibration_subsets_c3_c4_pinned_and_disjoint():
    eda = data.load_split("eda")
    sub = run_eval.calibration_subsets(eda)
    fmt = {q.qid for q in [q for q in eda if q.type == "bridge"][:16] + [q for q in eda if q.type == "comparison"][:4]}
    ids = {k: {q.qid for q in v} for k, v in sub.items()}
    for k in ("c3", "c4"):
        assert len(ids[k]) == 30 and [q.type for q in sub[k]] == ["bridge"] * 24 + ["comparison"] * 6
        assert data.ids_hash(list(ids[k])) == run_eval.CALIB_HASHES[k]
        assert not ids[k] & fmt and not ids[k] & ids["c1"] and not ids[k] & ids["c2"]
    assert not ids["c3"] & ids["c4"]
    assert run_eval.CALIB_HASHES == {"c3": "f06f4aede1de969bb5df8081b053b421ffb93461b1ef3e36f5ccb85cae42a3fd",
                                     "c4": "82205976797ec1b2f0bc1682a3867e83837924e8d2704c43833bbb93001b363a"}


def _man(version, absent=False):
    m = {"comparison_invariants": {"split": "dev", "code": {"commit": "c", "dirty": False}}, "system_config": {}}
    if not absent:
        m["comparison_invariants"]["ollama_version"] = version
    return m


def test_ollama_version_is_a_comparison_invariant():
    manifest.check_comparable(_man("0.32.5"), _man("0.32.5"))
    with pytest.raises(manifest.IncompatibleRuns, match="ollama_version"):
        manifest.check_comparable(_man("0.32.5"), _man("0.33.0"), unrecorded_runtime_ok=True)
    with pytest.raises(manifest.IncompatibleRuns, match="ollama_version"):
        manifest.check_comparable(_man("0.32.5"), _man(None))
    diff = manifest.check_comparable(_man("0.32.5"), _man(None, absent=True), unrecorded_runtime_ok=True)
    assert "ollama_version (unknown: not recorded by the legacy run)" in diff
    blind = _man(None, absent=True)
    blind["comparison_invariants"]["split"] = "test_agents"
    with pytest.raises(manifest.IncompatibleRuns, match="legacy dev runs only"):
        manifest.check_comparable(_man("0.32.5"), blind, unrecorded_runtime_ok=True)


def test_offline_client_takes_the_recorded_version_never_the_installed_one(tmp_path):
    for inv, expected in (({"ollama_version": "0.32.5"}, "0.32.5"), ({}, None)):
        path = str(tmp_path / "run.jsonl")
        with open(path + ".manifest.json", "w") as f:
            json.dump({"comparison_invariants": {"base_llm": {"digest": "sha256:x", "quantization": "Q4"}, **inv}}, f)
        c = run_eval.offline_client_for(path)
        assert c.ollama_version == expected and c.describe()["ollama_version"] == expected


def test_schema_is_part_of_the_cache_key_and_stored(tmp_path):
    from tests.test_llm import FakeOllama, words
    fake = FakeOllama(response='{"P1": true}')
    client = llm.OllamaClient(cache_dir=str(tmp_path), transport=fake, count_tokens=words)
    a, b = prompts.researcher_v2_schema(1, False), prompts.researcher_v2_schema(1, True)
    k1, k2 = client.generate("same prompt", a).key, client.generate("same prompt", b).key
    assert k1 != k2
    assert json.load(open(client._path(k1)))["schema"] == a.model_json_schema()
    assert client.describe()["ollama_version"] == "0.0.test"


def test_g0_cases_use_invented_text_only():
    eda = data.load_split("eda")
    ctx = data.load_contexts("eda")
    text = " ".join(t + " " + " ".join(s) for q in eda[:300] for t, s in ctx[q.qid].items())
    for c in run_eval.g0_cases():
        for t, sents in c["paragraphs"]:
            assert t not in text
    assert {c["name"] for c in run_eval.g0_cases()} >= {"required_keys", "additional_properties", "enum_references"}


def test_commands_parse_and_refuse_blind_splits(monkeypatch):
    seen = []
    monkeypatch.setattr(run_eval, "cmd_calib4b_flow", lambda a: seen.append(("flow", a.subset, a.tag)))
    monkeypatch.setattr(run_eval, "cmd_calib4b_probe", lambda a: seen.append(("probe", a.subset, a.tag)))
    monkeypatch.setattr(run_eval, "cmd_replay_agent", lambda a: seen.append(("agent", a.phase)))
    run_eval.main(["calib4b-flow", "--subset", "c3"])
    run_eval.main(["calib4b-probe", "--subset", "c4", "--tag", "4b_rev"])
    run_eval.main(["replay-agent", "--phase", "p4b"])
    assert seen == [("flow", "c3", "4b"), ("probe", "c4", "4b_rev"), ("agent", "p4b")]
    with pytest.raises(SystemExit):
        run_eval.main(["calib4b-flow", "--subset", "c1"])
    for split in ("test_agents", "tune"):
        with pytest.raises(SystemExit):
            run_eval.main(["multi-agent-4b", "--split", split])


def test_config_carries_phase4b_hashes_and_budget():
    cfg = run_eval.multi_agent_4b_config()
    h = prompts.multi_agent_4b_hashes()
    assert cfg["researcher_v2_hash"] == h["researcher_v2"] and cfg["requirement_verifier_v1_hash"] == h["requirement_verifier_v1"]
    assert cfg["budget"]["max_llm_calls"] == 8 and cfg["budget"]["max_paragraphs"] == 20
    assert cfg["answer_prompt_hash"] == prompts.artifact_hashes()["answer_v2"]


def test_g0_duplicate_normalization_is_checked_offline():
    assert run_eval.offline_duplicate_check()


def test_g0_does_not_require_a_live_duplicate(monkeypatch):
    """A well-behaved model (no duplicates) passes G0; the live duplicate is a diagnostic."""
    class Clean:
        ollama_version = run_eval.PHASE4B_OLLAMA
        options = {"num_ctx": 8192, "num_predict": 512}

        def __init__(self):
            self.store = {}

        def describe(self):
            return {"ollama_version": self.ollama_version}

        def _path(self, key):
            return self.store[key]

        def generate(self, user, schema):
            props = schema.model_json_schema()["properties"]
            if "references" in props:
                out = {"references": props["references"]["items"]["enum"][:1], "supported": True, "query": ""}
            else:
                out = {k: (True if k.startswith("P") else "q") for k in props}
            text = json.dumps(out)
            key = f"k{len(self.store)}"
            import tempfile
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump({"schema": schema.model_json_schema()}, f)
            f.close()
            self.store[key] = f.name
            return llm.LLMResult(key, text, schema.model_validate_json(text).model_dump(), None, 1, 1, 0.0, "stop", False)

    res = run_eval.run_g0(Clean())
    assert res["passed"] and res["diagnostics"] == {"live_duplicate_observed": False}
    assert "duplicates_normalized_offline" in res["checks"] and "live_verifier_outputs_valid_and_normalized" in res["checks"]


def test_transport_failure_aborts_the_flow():
    """Amendment A7: an unreachable server is not a model failure; the run aborts."""
    import urllib.error

    class Down(LLM):
        def generate(self, user, schema):
            if schema.__name__ == "RequirementVerifierOutput":
                raise urllib.error.URLError("connection refused")
            return super().generate(user, schema)

    client = Down(planner=[P("a")], researcher=[R([1])])
    with pytest.raises(urllib.error.URLError):
        m4.MultiAgent4b(Retriever(), client, docs=DOCS).run(Q)


def _a8_block(label):
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    return re.search(re.escape(label) + r"\s*```text\n(.*?)\n```", text[text.index("### A8"):], re.S).group(1)


def test_researcher_v3_equals_a8_and_changes_only_the_decision_paragraph():
    assert prompts.RESEARCHER_V3 == _a8_block("`researcher_v3`:")
    assert prompts.RESEARCHER_V2 == _block("`researcher_v2`:")            # preserved
    v2, v3 = prompts.RESEARCHER_V2.split("\n\n"), prompts.RESEARCHER_V3.split("\n\n")
    changed = [i for i, (a, b) in enumerate(zip(v2, v3)) if a != b]
    assert len(v2) == len(v3) and len(changed) == 1 and v2[changed[0]].startswith("Decide for every candidate")
    assert "When unsure, keep it" in v3[changed[0]] and "Judge every candidate independently" in v3[changed[0]]
    h = prompts.multi_agent_4b_hashes()
    assert h["researcher_v2"] != h["researcher_v3"]


def test_flow_uses_researcher_v3_by_default_and_v2_on_request():
    assert m4.RESEARCHER == "v3"
    for version, marker in (("v3", "Judge every candidate independently"), (None, "Judge every candidate independently"),
                            ("v2", None)):
        client = LLM(planner=[P("a")], researcher=[R([1])], verifier=[V(True)])
        kw = {} if version is None else {"researcher": version}
        m4.MultiAgent4b(Retriever(), client, docs=DOCS, **kw).run(Q)
        u = [u for role, u, _ in client.prompts if role == "researcher"][0]
        assert (marker in u) if marker else ("Judge every candidate" not in u)
    cfg = run_eval.multi_agent_4b_config()
    assert cfg["researcher"] == "v3" and cfg["researcher_prompt_hash"] == prompts.multi_agent_4b_hashes()["researcher_v3"]
