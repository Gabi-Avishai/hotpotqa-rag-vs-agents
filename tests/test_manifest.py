"""Manifest compatibility rules."""
import copy
import json

import pytest

import data
import manifest

LLM = {"model": "qwen2.5:7b", "digest": "845dbda0ea48", "quantization": "Q4_K_M",
       "tokenizer": "Qwen/Qwen2.5-7B-Instruct@a09a354",
       "options": {"temperature": 0.0, "seed": 0, "num_ctx": 8192, "num_predict": 512}}
IDS = ["q1", "q2", "q3"]
SPLIT = "adhoc:unit"
CLEAN = {"commit": "abc", "dirty": False}


def run(system_config, llm=LLM, ids=IDS, commit=CLEAN, runtime=1.0):
    return manifest.build(SPLIT, ids, system_config, llm=llm, runtime_s=runtime, commit=commit)


BASELINE = {"system": "baseline", "retriever": "bm25", "k": 5, "answer_prompt_hash": "p1"}
IMPROVED = {"system": "improved", "retriever": "rrf+rerank", "k": 5, "answer_prompt_hash": "p1"}


def test_sections_and_contents():
    m = run(BASELINE)
    assert set(m) == {"comparison_invariants", "system_config", "run_metadata"}
    inv = m["comparison_invariants"]
    assert inv["dataset"].endswith(data.REVISION)
    assert inv["corpus_fingerprint"] == data.EXPECTED_CORPUS_FINGERPRINT
    assert inv["split_hash"] == data.ids_hash(IDS) and inv["num_ctx"] == 8192
    assert inv["base_llm"]["digest"] == "845dbda0ea48" and "options" not in inv["base_llm"]
    assert m["system_config"]["llm_options"]["seed"] == 0
    json.dumps(m)  # serialisable


def test_system_differences_are_allowed_and_reported():
    diff = manifest.check_comparable(run(BASELINE), run(IMPROVED, runtime=99.0), shared=["answer_prompt_hash"])
    assert diff == {"system": ("baseline", "improved"), "retriever": ("bm25", "rrf+rerank")}


def test_run_metadata_is_ignored():
    a, b = run(BASELINE), run(IMPROVED)
    b["run_metadata"] = {"timestamp": "other", "runtime_s": 5, "host": "elsewhere"}
    manifest.check_comparable(a, b)


@pytest.mark.parametrize("key,value", [
    ("corpus_fingerprint", "other"), ("split_hash", "other"), ("eval_code_hash", "other"),
    ("metrics_version", "2"), ("num_ctx", 4096), ("dataset", "other"), ("packages", {"numpy": "1.0"}),
])
def test_invariant_differences_are_rejected(key, value):
    b = run(IMPROVED)
    b["comparison_invariants"][key] = value
    with pytest.raises(manifest.IncompatibleRuns, match=key):
        manifest.check_comparable(run(BASELINE), b)


def test_different_base_llm_is_rejected():
    other = copy.deepcopy(LLM)
    other["digest"] = "different"
    with pytest.raises(manifest.IncompatibleRuns, match="base_llm"):
        manifest.check_comparable(run(BASELINE), run(IMPROVED, llm=other))


def test_different_question_sets_are_rejected():
    with pytest.raises(manifest.IncompatibleRuns, match="split_hash"):
        manifest.check_comparable(run(BASELINE), run(IMPROVED, ids=["q1", "q2", "q4"]))


def test_shared_components_must_match():
    changed = {**IMPROVED, "answer_prompt_hash": "p2"}
    with pytest.raises(manifest.IncompatibleRuns, match="answer_prompt_hash"):
        manifest.check_comparable(run(BASELINE), run(changed), shared=["answer_prompt_hash"])
    missing = {k: v for k, v in IMPROVED.items() if k != "answer_prompt_hash"}
    with pytest.raises(manifest.IncompatibleRuns):
        manifest.check_comparable(run(BASELINE), run(missing), shared=["answer_prompt_hash"])


def test_commit_matters_only_for_final_comparisons():
    a, b = run(BASELINE), run(IMPROVED, commit={"commit": "def", "dirty": False})
    manifest.check_comparable(a, b)                           # development: allowed
    with pytest.raises(manifest.IncompatibleRuns, match="code.commit"):
        manifest.check_comparable(a, b, final=True)
    dirty = run(IMPROVED, commit={"commit": "abc", "dirty": True})
    with pytest.raises(manifest.IncompatibleRuns, match="code.dirty"):
        manifest.check_comparable(a, dirty, final=True)
    manifest.check_comparable(a, run(IMPROVED), final=True)   # same clean commit


def test_retrieval_only_runs_have_no_llm_and_cannot_mix_with_llm_runs():
    r1, r2 = run({"retriever": "bm25"}, llm=None), run({"retriever": "rrf"}, llm=None)
    assert r1["comparison_invariants"]["base_llm"] is None
    manifest.check_comparable(r1, r2)
    with pytest.raises(manifest.IncompatibleRuns):
        manifest.check_comparable(r1, run(BASELINE))


def test_write_and_real_commit(tmp_path):
    m = manifest.build(SPLIT, IDS, BASELINE, llm=LLM)
    assert len(m["comparison_invariants"]["code"]["commit"]) == 40
    path = manifest.write(m, str(tmp_path / "results.jsonl"))
    assert json.load(open(path)) == json.loads(json.dumps(m, sort_keys=True))


def test_sha256_of_is_canonical():
    assert manifest.sha256_of({"a": 1, "b": 2}) == manifest.sha256_of({"b": 2, "a": 1})


def test_real_split_is_validated_and_corpus_is_refingerprinted(monkeypatch):
    dev = data.load_split_ids("dev")
    m = manifest.build("dev", dev, BASELINE, llm=LLM, commit=CLEAN)
    assert m["comparison_invariants"]["split_hash"] == data.EXPECTED_SPLIT_HASHES["dev"]
    assert m["comparison_invariants"]["n_questions"] == 100
    calls = []
    monkeypatch.setattr(data, "verify_corpus", lambda: calls.append(1) or "fp")
    assert manifest.build("dev", dev, BASELINE, commit=CLEAN)["comparison_invariants"]["corpus_fingerprint"] == "fp"
    assert calls == [1]


def test_manifest_fails_when_the_corpus_differs(monkeypatch):
    monkeypatch.setattr(data, "corpus_fingerprint", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="corpus fingerprint"):
        manifest.build(SPLIT, IDS, BASELINE, commit=CLEAN)


@pytest.mark.parametrize("ids", [["not-a-real-question"], "missing", "duplicated"])
def test_manifest_rejects_ids_that_are_not_the_named_split(ids):
    dev = data.load_split_ids("dev")
    if ids == "missing":
        ids = dev[1:]
    elif ids == "duplicated":
        ids = dev[1:] + [dev[1]]
    with pytest.raises(data.SplitMismatch):
        manifest.build("dev", ids, BASELINE, commit=CLEAN)


def test_manifest_rejects_unknown_non_adhoc_names():
    with pytest.raises(data.SplitMismatch, match="unknown split"):
        manifest.build("dev_subset", IDS, BASELINE, commit=CLEAN)
