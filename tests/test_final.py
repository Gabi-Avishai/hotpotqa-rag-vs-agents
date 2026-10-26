"""Phase 5: the final runner, its validation, the analysis and the audit (PREREGISTRATION.md §5.8).
No test split is loaded: a guard fails any test that tries."""
import json
import os
import types

import numpy as np
import pytest

import data
import final
import manifest
import metrics
import run_eval


@pytest.fixture(autouse=True)
def no_blind_access(monkeypatch):
    real_split, real_ctx = data.load_split, data.load_contexts

    def guard(real):
        def f(name, unblind=False):
            if name.startswith("test_"):
                raise AssertionError(f"blind split {name} loaded during an offline test")
            return real(name)
        return f
    monkeypatch.setattr(data, "load_split", guard(real_split))
    monkeypatch.setattr(data, "load_contexts", guard(real_ctx))


# ---------------------------------------------------------------- frozen identities and the declaration

needs_selection = pytest.mark.skipif(not os.path.exists(run_eval._path("tune", "phase2b_selection", "json")),
                                     reason="the Phase 2b selection record is local and not tracked")


@needs_selection
def test_frozen_identities_match_and_preregistration_lists_them():
    assert final.frozen_mismatches() == []
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    sec = text[text.index("## Phase 5"):text.index("## P. Post-hoc analyses")]
    assert sec.startswith("## Phase 5 — freeze and blind test (approved)")
    for group in ("split_hashes", "answer_interface", "assess"):
        assert all(v in sec for v in final.FROZEN[group].values())
    for v in (final.FROZEN["corpus"], final.FROZEN["embedding_fingerprint"], final.FROZEN["graph_fingerprint"],
              final.MODEL_DIGEST, *final.FROZEN["sources"].values()):
        assert v in sec
    assert final.K_MATCH == 16 and final.WEIGHTS == {"bridge": 0.799, "comparison": 0.201}


@needs_selection
def test_tampering_with_a_frozen_identity_is_detected(monkeypatch):
    import prompts
    real = prompts.artifact_hashes
    monkeypatch.setattr(prompts, "artifact_hashes", lambda: {**real(), "answer_v2": "x"})
    assert "answer_interface" in final.frozen_mismatches()
    monkeypatch.setattr(run_eval, "PHASE4_VERIFIER", "v3")
    assert "phase4_verifier" in final.frozen_mismatches()


def test_runner_holds_no_constant_describing_itself():
    src = open(final.__file__).read()
    assert final.runner_source_hash() not in src and manifest.eval_code_hash() not in src


def _prereg(tmp_path, block):
    p = tmp_path / "PREREGISTRATION.md"
    p.write_text("intro\n\n" + (f"{final.DECLARATION_HEADING} (date)\n\n```text\n{block}\n```\n" if block is not None else ""))
    return str(p)


def test_declaration_parsing(tmp_path):
    good = "\n".join(f"{k}: v{i}" for i, k in enumerate(final.DECLARED_KEYS))
    assert final.declared_freeze(_prereg(tmp_path, good)) == {k: f"v{i}" for i, k in enumerate(final.DECLARED_KEYS)}
    assert final.declared_freeze(_prereg(tmp_path, None)) is None
    with pytest.raises(SystemExit, match="exactly"):
        final.declared_freeze(_prereg(tmp_path, "eval_code_hash: x"))


def test_real_runs_require_the_declared_hashes_dry_runs_only_report():
    h = {k: "a" for k in final.DECLARED_KEYS}
    with pytest.raises(SystemExit, match="no approved freeze declaration"):
        final.check_declaration(h, False, declared=None)
    with pytest.raises(SystemExit, match="differ"):
        final.check_declaration(h, False, declared={**h, "runner_source_hash": "b"})
    assert final.check_declaration(h, False, declared=dict(h)) is None
    assert "differ" in final.check_declaration(h, True, declared={**h, "eval_code_hash": "b"})


def test_real_stage_refuses_before_reading_any_split_without_a_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    monkeypatch.setattr(manifest, "code_commit", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(final, "declared_freeze", lambda path=None: None)
    monkeypatch.setattr(final, "frozen_mismatches", lambda: [])
    stub = final._Described({"graph_fingerprint": final.FROZEN["graph_fingerprint"]})
    with pytest.raises(SystemExit, match="no approved freeze declaration"):
        final.run_stage("retrieval", dry_run=False, retrievers=(final._Described({"b": 1}), stub))


def test_runtime_check_refuses_other_versions_and_digests():
    final.check_runtime(types.SimpleNamespace(ollama_version=final.OLLAMA_VERSION, digest=final.MODEL_DIGEST))
    for bad in (types.SimpleNamespace(ollama_version="0.33.0", digest=final.MODEL_DIGEST),
                types.SimpleNamespace(ollama_version=final.OLLAMA_VERSION, digest="other")):
        with pytest.raises(SystemExit):
            final.check_runtime(bad)


def test_dry_run_never_loads_a_blind_split():
    for stage in final.ORDER:
        assert final.stage_split(stage, True) == "dev" and len(final.load_questions(stage, True)) == 100
    with pytest.raises(AssertionError, match="blind split"):
        final.load_questions("agents", False)


# ---------------------------------------------------------------- run_system and validation

SMOKE = data.load_split("dev_smoke")
COMMIT = {"commit": "abc", "dirty": False}


def _fn(log, fail_at=None):
    def fn(_c, q):
        if fail_at is not None and len(log) == fail_at:
            raise RuntimeError("interrupted")
        log.append(q.qid)
        return {"qid": q.qid, "value": len(log)}
    return fn


def _freeze():
    return {"commit": "abc", "eval_code_hash": manifest.eval_code_hash()}


def test_interrupted_run_resumes_finalizes_atomically_and_validates(tmp_path):
    path = str(tmp_path / "x.jsonl")
    log = []
    with pytest.raises(RuntimeError):
        final.run_system(path, "dev_smoke", SMOKE, _fn(log, fail_at=3), {"c": 1}, None, COMMIT, progress=lambda *_: None)
    assert not os.path.exists(path) and not os.path.exists(path + ".manifest.json")
    assert final.run_system(path, "dev_smoke", SMOKE, _fn(log), {"c": 1}, None, COMMIT, progress=lambda *_: None) == "finished"
    assert log == [q.qid for q in SMOKE]                       # no question run twice
    assert final.validation_problems(path, "dev_smoke", [q.qid for q in SMOKE], {"c": 1}, False, _freeze()) == []
    rows = open(path).read()
    assert final.run_system(path, "dev_smoke", SMOKE, _fn([]), {"c": 2}, None, COMMIT, progress=lambda *_: None) == "complete"
    assert open(path).read() == rows                           # immutable


def test_partial_run_refuses_a_different_commit_or_configuration(tmp_path):
    path = str(tmp_path / "x.jsonl")
    with pytest.raises(RuntimeError):
        final.run_system(path, "dev_smoke", SMOKE, _fn([], fail_at=2), {"c": 1}, None, COMMIT, progress=lambda *_: None)
    for cfg, commit in (({"c": 2}, COMMIT), ({"c": 1}, {"commit": "other", "dirty": False})):
        with pytest.raises(SystemExit, match="differs"):
            final.run_system(path, "dev_smoke", SMOKE, _fn([]), cfg, None, commit, progress=lambda *_: None)


@pytest.fixture
def stored(tmp_path):
    path = str(tmp_path / "x.jsonl")
    final.run_system(path, "dev_smoke", SMOKE, _fn([]), {"c": 1}, None, COMMIT, progress=lambda *_: None)
    return path


def _rewrite(path, rows):
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("tamper,problem", [
    (lambda p: _rewrite(p, final._load(p) + final._load(p)[:1]), "duplicate records"),
    (lambda p: _rewrite(p, final._load(p)[:-1]), "records differ from the ordered split membership"),
    (lambda p: _rewrite(p, list(reversed(final._load(p)))), "records differ from the ordered split membership"),
    (lambda p: os.remove(p + ".manifest.json"), "result or manifest missing"),
])
def test_validation_catches_tampered_results(stored, tamper, problem):
    tamper(stored)
    assert problem in final.validation_problems(stored, "dev_smoke", [q.qid for q in SMOKE], {"c": 1}, False, _freeze())


def test_validation_catches_configuration_commit_and_evaluation_hash(stored):
    qids = [q.qid for q in SMOKE]
    assert "system configuration" in final.validation_problems(stored, "dev_smoke", qids, {"c": 2}, False, _freeze())
    assert "manifest commit" in final.validation_problems(stored, "dev_smoke", qids, {"c": 1}, False, {**_freeze(), "commit": "z"})
    assert "manifest evaluation hash" in final.validation_problems(stored, "dev_smoke", qids, {"c": 1}, False,
                                                                   {**_freeze(), "eval_code_hash": "z"})
    assert "model identity" in final.validation_problems(stored, "dev_smoke", qids, {"c": 1}, True, _freeze())


class _Client:
    def __init__(self, version=final.OLLAMA_VERSION, digest=final.MODEL_DIGEST):
        self.v, self.d = version, digest

    def describe(self):
        import llm
        return {"model": final.MODEL, "digest": self.d, "quantization": final.QUANTIZATION,
                "options": dict(llm.DEFAULT_OPTIONS), "tokenizer": "t", "ollama_version": self.v}


def test_validation_of_llm_runtime_invariants(tmp_path):
    qids = [q.qid for q in SMOKE]
    ok, bad = str(tmp_path / "ok.jsonl"), str(tmp_path / "bad.jsonl")
    final.run_system(ok, "dev_smoke", SMOKE, _fn([]), {"c": 1}, _Client(), COMMIT, progress=lambda *_: None)
    assert final.validation_problems(ok, "dev_smoke", qids, {"c": 1}, True, _freeze()) == []
    final.run_system(bad, "dev_smoke", SMOKE, _fn([]), {"c": 1}, _Client(version="0.1", digest="d"), COMMIT, progress=lambda *_: None)
    probs = final.validation_problems(bad, "dev_smoke", qids, {"c": 1}, True, _freeze())
    assert "runtime" in probs and "model identity" in probs


def _fake_retrieval_stage(tmp_path, monkeypatch):
    """A complete dry-run retrieval stage with synthetic records, written through the real code."""
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    os.makedirs(final.out_dir(True))
    bm25_d, d3_d = {"retriever": "bm25-stub"}, {"retriever": "d3-stub", "graph_fingerprint": final.FROZEN["graph_fingerprint"]}
    fr = {"commit": "abc", "eval_code_hash": manifest.eval_code_hash(), "runner_source_hash": "r",
          "bm25_describe_hash": manifest.sha256_of(bm25_d), "d3_describe_hash": manifest.sha256_of(d3_d),
          "bm25_describe": bm25_d, "d3_describe": d3_d}
    final._atomic_json(os.path.join(final.out_dir(True), "freeze.json"), fr)
    qs = data.load_split("dev")
    cfgs = final.configs("retrieval", bm25_d, d3_d)
    marker = {}
    for name, cfg in cfgs.items():
        p = final.result_path("retrieval", name, True)
        final.run_system(p, "dev", qs, lambda c, q: {"qid": q.qid}, cfg, None, COMMIT, progress=lambda *_: None)
        marker[name] = {"result_sha256": final._sha_file(p), "manifest_sha256": final._sha_file(p + ".manifest.json"),
                        "config_hash": manifest.sha256_of(cfg)}
    final._atomic_json(final.marker_path("retrieval", True), {"stage": "retrieval", "split": "dev", "commit": "abc",
                                                              "systems": marker})
    return final.result_path("retrieval", "d3_rankings", True)


def test_stage_validation_recomputes_file_hashes(tmp_path, monkeypatch):
    path = _fake_retrieval_stage(tmp_path, monkeypatch)
    final.validate_stage("retrieval", True)
    with open(path, "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="files changed"):
        final.validate_stage("retrieval", True)


def test_dependent_stage_and_analysis_validate_earlier_stages(tmp_path, monkeypatch):
    path = _fake_retrieval_stage(tmp_path, monkeypatch)
    with open(path, "a") as f:
        f.write("\n")
    monkeypatch.setattr(final, "frozen_mismatches", lambda: [])
    monkeypatch.setattr(manifest, "code_commit", lambda: {"commit": "abc", "dirty": False})
    stub_b, stub_d = final._Described({"retriever": "bm25-stub"}), final._Described(
        {"retriever": "d3-stub", "graph_fingerprint": final.FROZEN["graph_fingerprint"]})
    monkeypatch.setattr(final, "current_hashes", lambda b, d: {"eval_code_hash": manifest.eval_code_hash(),
                                                                "runner_source_hash": "r",
                                                                "bm25_describe_hash": manifest.sha256_of(b),
                                                                "d3_describe_hash": manifest.sha256_of(d)})
    with pytest.raises(SystemExit, match="files changed"):
        final.run_stage("rag", dry_run=True, retrievers=(stub_b, stub_d), client=object())
    with pytest.raises(SystemExit, match="refusing to analyse|not complete|files changed"):
        final.analysis(dry_run=True)


def test_stage_order_and_completed_stage_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    with pytest.raises(SystemExit, match="needs the completion marker"):
        final.run_stage("rag", dry_run=True)
    os.makedirs(final.out_dir(True), exist_ok=True)
    open(final.marker_path("retrieval", True), "w").write("{}")
    with pytest.raises(SystemExit, match="immutable"):
        final.run_stage("retrieval", dry_run=True)
    with pytest.raises(SystemExit, match="unknown stage"):
        final.run_stage("everything", dry_run=True)


def test_real_stage_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    monkeypatch.setattr(manifest, "code_commit", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="clean"):
        final.run_stage("retrieval", dry_run=False)


# ---------------------------------------------------------------- analysis

TYPES = {f"b{i}": "bridge" for i in range(50)} | {f"c{i}": "comparison" for i in range(50)}


def test_weighted_quantile_and_question_weights():
    assert final.weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2
    assert final.weighted_quantile([1, 2, 3, 4], [0.1, 0.1, 0.1, 0.7], 0.5) == 4
    w = final.question_weights(TYPES)
    assert w["b0"] == pytest.approx(0.799 / 50) and sum(w.values()) == pytest.approx(1)
    q = final._quantiles({q: (10.0 if t == "bridge" else 100.0) for q, t in TYPES.items()}, TYPES, True)
    assert q["p50"] == 10.0 and q["p95"] == 100.0 and q["bridge"]["p50"] == 10.0     # 0.799 of the mass is bridge


def test_directional_conclusions():
    assert final.conclusion(0.05, 0.01) == "improvement demonstrated (practically meaningful)"
    assert final.conclusion(0.02, 0.01) == "improvement demonstrated"
    assert final.conclusion(-0.05, 0.01) == "worsening demonstrated"
    assert final.conclusion(0.10, 0.2) == "no clear difference demonstrated"


def test_confirmatory_family_uses_weighted_within_type_bootstrap_and_holm():
    rng = np.random.default_rng(1)
    base = {q: float(rng.random()) for q in TYPES}
    sc = {"bm25_k10": {q: {"f1": v} for q, v in base.items()},
          "d3_k10": {q: {"f1": min(1, v + 0.2)} for q, v in base.items()},
          "agent_single": {q: {"f1": min(1, v + 0.2)} for q, v in base.items()},
          "multi_agent": {q: {"f1": max(0, v + 0.1)} for q, v in base.items()}}
    res = final.confirmatory(sc, TYPES)
    direct = metrics.paired_bootstrap({q: s["f1"] for q, s in sc["d3_k10"].items()},
                                      {q: s["f1"] for q, s in sc["bm25_k10"].items()}, strata=TYPES, weights=final.WEIGHTS)
    assert res["F1"]["delta"] == direct["delta"] and res["F1"]["conclusion"].startswith("improvement demonstrated")
    assert res["F2"]["conclusion"] == "no clear difference demonstrated" and res["F3"]["conclusion"] == "worsening demonstrated"
    adj = metrics.holm({k: v["p_value"] for k, v in res.items()})
    assert all(res[k]["p_holm"] == adj[k] for k in res) and "not adjusted" in res["F1"]["note"]


def _score(f1, sup, ncit=2, valid=1):
    return {"em": float(f1 == 1), "f1": f1, "supported": float(sup), "ctx_joint_recall": 1.0, "ctx_recall": 1.0,
            "sp_strict_f1": 0.5, "sp_strict_prec": 0.5, "sp_strict_recall": 0.5, "sp_official_f1": 0.5,
            "joint_official_f1": 0.3, "has_citations": 1.0, "all_valid_citations": float(valid == ncit),
            "n_citations": ncit, "valid_citations": valid}


def test_answer_metrics_answered_f1_citations_and_context():
    sc = {q: _score(1.0 if t == "bridge" else 0.0, sup=(t == "bridge")) for q, t in TYPES.items()}
    recs = {q: {"displayed": ["a"] * (4 if t == "bridge" else 10)} for q, t in TYPES.items()}
    m = final.answer_metrics(recs, sc, TYPES, weighted=True)
    assert m["f1"]["overall"] == pytest.approx(0.799)
    assert m["f1_answered"]["bridge"] == {"f1_answered": 1.0, "n_answered": 50}
    assert m["f1_answered"]["comparison"]["n_answered"] == 0 and m["f1_answered"]["overall"] is None   # zero denominator
    assert m["per_citation_validity"]["bridge"] == {"k": 50, "n": 100, "rate": 0.5} and m["per_citation_validity"]["overall"] == 0.5
    assert m["final_context_size"]["overall"] == pytest.approx(0.799 * 4 + 0.201 * 10)
    cb = final.answer_metrics({}, {q: {"em": 0.0, "f1": 0.5, "supported": float(t == "bridge")} for q, t in TYPES.items()},
                              TYPES, weighted=True, closed_book=True)
    assert cb["low_confidence_rate"]["overall"] == pytest.approx(0.201) and "ctx_joint_recall" not in cb


def test_cost_metrics_count_cache_hits_and_unique_generations():
    rag_rec = lambda key, cached: {"llm": {"key": key, "cached": cached, "prompt_tokens": 100, "output_tokens": 10,
                                           "wall_s": 2.0}, "retrieval_s": 0.1}
    recs = {"b0": rag_rec("k1", False), "b1": rag_rec("k1", True), "c0": rag_rec("k2", False)}
    c = final.cost_metrics(recs, {"b0": "bridge", "b1": "bridge", "c0": "comparison"}, weighted=False)
    assert c["logical_calls"]["total"] == 3 and c["cache_hits"] == 1 and c["unique_new_generations"] == 2
    assert c["prompt_tokens"]["total"] == 300 and c["llm_seconds"]["p50"] == 2.0


def test_agent_behaviour_funnel_and_multi_agent_paths(monkeypatch):
    monkeypatch.setattr(final, "_key_fn", lambda: (lambda d: (d, "h")))
    q = types.SimpleNamespace(gold=(("G1", "h"), ("G2", "h")))
    rec = {"batches": [["G1", "X"], ["G2"]], "displayed": ["G1"], "accepted": ["G1"],
           "events": [{"event": "verify", "sufficient": False}, {"event": "search_skipped", "stage": "corrective"},
                      {"event": "verifier_error"}],
           "calls": [{"role": "planner"}, {"role": "researcher"}, {"role": "verifier"}, {"role": "verifier_recheck"}]}
    out = final.agent_behaviour("multi_agent", {"b0": rec}, {"b0": q}, {"b0": "bridge"}, weighted=False)
    assert out["funnel"]["search_1"]["overall"] == 0 and out["funnel"]["search_pool"]["overall"] == 1
    assert out["funnel"]["final_context"]["overall"] == 0
    assert {k: v["count"] for k, v in out["corrective"].items()} == {"attempted": 1, "executed": 0, "no_new_search_results": 1}
    assert out["role_failures"]["initial verifier"]["bridge"] == {"k": 1, "n": 1, "rate": 1.0}
    assert out["first_verification"]["counts"]["bridge"] == {"False": 1}
    single = final.agent_behaviour("agent_single", {"b0": {**rec, "stop_reason": "sufficient"}}, {"b0": q}, {"b0": "bridge"}, False)
    assert single["stop_reasons"]["counts"]["bridge"] == {"sufficient": 1} and single["stop_reasons"]["shares"]["sufficient"]["overall"] == 1


def test_retrieval_report_by_type():
    rows = lambda v: [{"qid": q, "latency_s": 0.01, "recall@10": v, "joint_recall@10": v, "title_recall@10": v,
                       "title_joint_recall@10": v} for q in TYPES]
    out = final.retrieval_report({"bm25_rankings": rows(0.5), "d3_rankings": rows(1.0)}, TYPES)
    assert out["d3_rankings"]["curves"]["joint_recall@10"] == {"bridge": 1.0, "comparison": 1.0, "overall": 1.0}
    assert out["jr10_d3_minus_bm25"]["delta"] == 0.5 and "title_joint_recall@10" in out["bm25_rankings"]["curves"]


# ---------------------------------------------------------------- audit

def test_audit_sample_and_order_are_deterministic_and_balanced():
    types_ = {f"b{i:03d}": "bridge" for i in range(50)} | {f"c{i:03d}": "comparison" for i in range(50)}
    s = final.audit_sample(types_)
    assert len(s) == 24 and sum(types_[q] == "bridge" for q in s) == 12 and s == final.audit_sample(dict(reversed(types_.items())))
    items = final.audit_items(s)
    assert len(items) == 96 and items == final.audit_items(list(reversed(s)))


def test_audit_files_blinded_with_definitions_and_unresolved_citations():
    docs = {"D1": data.Doc("D1", "Title", "h", ("first.", "second."))}
    rec = {"displayed": ["D1"], "citations": [["D1", 1], ["D1", 7]],
           "output": {"answer": "A", "supported": True, "rationale": "", "citations": []}}
    qs = {"q1": types.SimpleNamespace(question="Q?", type="bridge")}
    files = final.build_audit_files([("q1", "multi_agent")], {"multi_agent": {"q1": rec}}, qs, docs)
    sheet = files["sheet.md"]
    assert "UNRESOLVED citation" in sheet and "[Title, sentence 1] second." in sheet
    assert "takes\n   precedence" in sheet and "`cannot_determine`" in sheet and "multi_agent" not in sheet
    assert json.loads(files["key.jsonl"]) == {"item": 1, "qid": "q1", "type": "bridge", "system": "multi_agent", "supported": True}
    assert files["labels.csv"] == "item,support,citation_validity\n1,,\n"


def test_audit_directory_is_written_atomically(tmp_path):
    d = str(tmp_path / "audit")
    os.makedirs(d + ".tmp")
    open(os.path.join(d + ".tmp", "stale"), "w").write("partial")        # an interrupted build
    final.write_dir_atomic(d, {"a.txt": "x"})
    assert os.listdir(d) == ["a.txt"] and not os.path.exists(d + ".tmp")
    with pytest.raises(SystemExit, match="generated once"):
        final.write_dir_atomic(d, {"a.txt": "y"})


def _labels(tmp_path, rows):
    p = tmp_path / "labels.csv"
    p.write_text("item,support,citation_validity\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))
    return str(p)


def test_label_validation(tmp_path):
    ok = [(1, "fully_supported", "valid"), (2, "unsupported", "invalid")]
    assert final.read_labels(_labels(tmp_path, ok), 2)[2] == {"support": "unsupported", "citation_validity": "invalid"}
    for bad in ([(1, "fully_supported", "valid")], [(1, "fully", "valid"), (2, "unsupported", "invalid")],
                [(1, "fully_supported", ""), (2, "unsupported", "invalid")], [(1, "fully_supported", "valid")] * 2):
        with pytest.raises(SystemExit, match="do not validate"):
            final.read_labels(_labels(tmp_path, bad), 2)


def test_audit_summary_rates_intervals_and_zero_denominators():
    key, labels = [], {}
    i = 0
    for s in final.AUDIT_SYSTEMS:
        for t in ("bridge", "comparison"):
            for j in range(12):
                i += 1
                key.append({"item": i, "qid": f"{t}{j}", "type": t, "system": s,
                            "supported": s != "bm25_k10" or t == "bridge"})
                labels[i] = {"support": "unsupported" if j < 3 else "fully_supported",
                             "citation_validity": "valid" if j % 2 else "invalid"}
    out = final.audit_summary(key, labels)
    m = out["multi_agent"]
    assert m["bridge"]["unsupported_claim_rate"]["k"] == 3 and m["bridge"]["unsupported_claim_rate"]["n"] == 12
    assert m["balanced_audit"]["unsupported_claim_rate"]["n"] == 24 and "not a population" in m["balanced_audit"]["note"]
    assert out["bm25_k10"]["comparison"]["unsupported_claim_rate"]["rate"] is None            # no supported outputs
    assert sum(m["bridge"]["support_distribution"].values()) == 12 and m["bridge"]["citation_distribution"]["valid"] == 6
    lo, hi = m["bridge"]["unsupported_claim_rate"]["ci95"]
    assert (lo, hi) == pytest.approx(final.clopper_pearson(3, 12))


def test_clopper_pearson():
    assert final.clopper_pearson(0, 24) == pytest.approx((0, 0.1425), abs=1e-3)
    assert final.clopper_pearson(12, 24) == pytest.approx((0.2912, 0.7088), abs=1e-3)
    with pytest.raises(ValueError):
        final.clopper_pearson(0, 0)


def test_commands_parse(monkeypatch):
    seen = []
    monkeypatch.setattr(run_eval, "cmd_final", lambda a: seen.append((a.stage, a.dry_run)))
    monkeypatch.setattr(run_eval, "cmd_audit_analysis", lambda a: seen.append(("audit", a.dry_run)))
    run_eval.main(["final", "--stage", "rag", "--dry-run"])
    run_eval.main(["audit-analysis"])
    assert seen == [("rag", True), ("audit", False)]
    with pytest.raises(SystemExit):
        run_eval.main(["final", "--stage", "all"])


# ---------------------------------------------------------------- the consumer-side freeze check

def _consumer_setup(tmp_path, monkeypatch, **override):
    monkeypatch.setattr(run_eval, "RESULTS", str(tmp_path))
    os.makedirs(final.out_dir(True))
    b, d = {"retriever": "b"}, {"retriever": "d"}
    fr = {"commit": "abc", "eval_code_hash": manifest.eval_code_hash(), "runner_source_hash": final.runner_source_hash(),
          "bm25_describe_hash": manifest.sha256_of(b), "d3_describe_hash": manifest.sha256_of(d),
          "bm25_describe": b, "d3_describe": d}
    final._atomic_json(os.path.join(final.out_dir(True), "freeze.json"), {**fr, **override})
    monkeypatch.setattr(manifest, "code_commit", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(final, "frozen_mismatches", lambda: [])


def test_consumer_check_passes_only_for_the_frozen_code(tmp_path, monkeypatch):
    _consumer_setup(tmp_path, monkeypatch)
    assert final.check_consumer(True)["commit"] == "abc"


@pytest.mark.parametrize("change,problem", [
    (lambda mp: mp.setattr(manifest, "eval_code_hash", lambda: "changed"), "evaluation code differs"),
    (lambda mp: mp.setattr(final, "runner_source_hash", lambda: "changed"), "runner differs"),
    (lambda mp: mp.setattr(manifest, "code_commit", lambda: {"commit": "other", "dirty": False}), "freeze commit"),
    (lambda mp: mp.setattr(manifest, "code_commit", lambda: {"commit": "abc", "dirty": True}), "freeze commit"),
    (lambda mp: mp.setattr(final, "frozen_mismatches", lambda: ["answer_interface"]), "frozen identities differ"),
])
def test_consumer_check_refuses_changed_code_or_commit(tmp_path, monkeypatch, change, problem):
    _consumer_setup(tmp_path, monkeypatch)
    change(monkeypatch)
    with pytest.raises(SystemExit, match=problem):
        final.check_consumer(True)
    loaded = []
    monkeypatch.setattr(final, "load_questions", lambda *a: loaded.append(a))
    for consumer in (final.analysis, final.audit_sheet, final.audit_analysis):
        with pytest.raises(SystemExit, match="refusing to analyse"):
            consumer(dry_run=True)
    assert loaded == []                                   # nothing read before the check


# ---------------------------------------------------------------- audit files after creation

def _audit_world(tmp_path):
    qs = {f"{t[0]}{i:02d}": types.SimpleNamespace(question=f"Q{i}?", type=t) for t in ("bridge", "comparison") for i in range(20)}
    recs = {s: {q: {"displayed": [], "citations": [], "output": {"answer": "A", "supported": i % 2 == 0}}
                for i, q in enumerate(qs)} for s in final.AUDIT_SYSTEMS}
    items = final.audit_items(final.audit_sample({q: x.type for q, x in qs.items()}))
    d = final.write_dir_atomic(str(tmp_path / "audit"), final.build_audit_files(items, recs, qs, {}))
    return d, qs, recs


def test_audit_files_verify_and_only_labels_are_editable(tmp_path):
    d, qs, recs = _audit_world(tmp_path)
    key = final.verify_audit_files(d, qs, recs)
    assert len(key) == 96 and len({(k["qid"], k["system"]) for k in key}) == 96
    with open(os.path.join(d, "labels.csv"), "a") as f:
        f.write("# edited\n")
    final.verify_audit_files(d, qs, recs)                 # labels are not covered by the manifest
    man = json.load(open(os.path.join(d, "audit_manifest.json")))
    assert set(man) == {"sheet_sha256", "key_sha256", "n_items", "note"} and man["n_items"] == 96


def test_tampered_audit_sheet_or_key_is_refused(tmp_path):
    d, qs, recs = _audit_world(tmp_path)
    with open(os.path.join(d, "sheet.md"), "a") as f:
        f.write("x")
    with pytest.raises(SystemExit, match="changed after creation"):
        final.verify_audit_files(d, qs, recs)


def test_consistently_rewritten_key_with_a_wrong_mapping_is_refused(tmp_path):
    d, qs, recs = _audit_world(tmp_path)
    key = [json.loads(x) for x in open(os.path.join(d, "key.jsonl"))]
    j = next(i for i, k in enumerate(key) if k["system"] != key[0]["system"])
    key[0]["system"], key[j]["system"] = key[j]["system"], key[0]["system"]
    text = "".join(json.dumps(k) + "\n" for k in key)
    open(os.path.join(d, "key.jsonl"), "w").write(text)
    man = json.load(open(os.path.join(d, "audit_manifest.json")))
    man["key_sha256"] = final._h(text)
    open(os.path.join(d, "audit_manifest.json"), "w").write(json.dumps(man))
    with pytest.raises(SystemExit, match="does not match the frozen sample"):
        final.verify_audit_files(d, qs, recs)


# ---------------------------------------------------------------- prevalence-weighted behaviour

def test_behaviour_summaries_are_per_type_and_prevalence_weighted(monkeypatch):
    monkeypatch.setattr(final, "_key_fn", lambda: (lambda x: (x, "h")))
    gold = types.SimpleNamespace(gold=(("G1", "h"),))
    recs, qs = {}, {}
    for q, t in TYPES.items():
        corr = t == "bridge"                                  # every bridge question corrects; no comparison one does
        recs[q] = {"batches": [["G1"]] * (3 if corr else 2), "displayed": ["G1"], "accepted": ["G1"],
                   "events": ([{"event": "verify", "sufficient": False}, {"event": "search", "stage": "corrective"}] if corr
                              else [{"event": "verifier_error"}]),
                   "calls": [{"role": "verifier"}]}
        qs[q] = gold
    out = final.agent_behaviour("multi_agent", recs, qs, TYPES, weighted=True)
    assert out["corrective"]["executed"]["per_question_rate"]["overall"] == pytest.approx(0.799)
    assert out["corrective"]["executed"]["count"] == 50
    assert out["searches"]["mean"]["overall"] == pytest.approx(0.799 * 3 + 0.201 * 2)
    assert out["searches"]["shares"]["3"]["overall"] == pytest.approx(0.799)
    rf = out["role_failures"]["initial verifier"]
    assert rf["bridge"]["rate"] == 0 and rf["comparison"]["rate"] == 1 and rf["overall"] == pytest.approx(0.201)
    assert out["first_verification"]["shares"]["error"]["overall"] == pytest.approx(0.201)
    unweighted = final.agent_behaviour("multi_agent", recs, qs, TYPES, weighted=False)
    assert unweighted["role_failures"]["initial verifier"]["overall"] == 0.5


def test_ratio_handles_zero_denominators():
    r = final._ratio({"b0": (0, 0), "c0": (1, 2)}, {"b0": "bridge", "c0": "comparison"}, weighted=True)
    assert r["bridge"]["rate"] is None and r["overall"] is None and r["comparison"]["rate"] == 0.5
