"""The cached offline demo (demo.py): fixed cases, pinned values, no model calls."""
import ast
import copy
import os
import socket

import pytest

import demo

CASES = demo.load_cases()
HAVE_RESULTS = all(os.path.exists(p) for c in CASES for p in demo.artifact_paths(c).values())
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="the stored results are local and not tracked")


def test_manifest_is_fixed_and_hand_authored():
    assert [c["id"] for c in CASES] == ["graph_success", "rerank_failure", "agent_recovery", "unnecessary_search",
                                        "selection_loss"]
    assert [c["qid"] for c in CASES] == ["5a825bb455429940e5e1a874", "5ae7f56e5542993210983fd1", "5ab6f7415542991d322236f4",
                                         "5ab9547755429970cfb8ea6c", "5a87aa135542996e4f3088b8"]
    assert [c["systems"] for c in CASES] == [["bm25_k10", "d3_k10"], ["D3", "X1"], ["d3_k10", "agent_single"],
                                             ["d3_k10", "agent_single"], ["d3_k10", "multi_agent"]]
    for c in CASES:
        assert set(c["sources"]) == set(c["systems"]) and c["pins"] and c["takeaway"]


def test_demo_imports_nothing_that_can_generate_or_retrieve():
    tree = ast.parse(open(demo.__file__).read())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or "")])}
    assert not imported & {"llm", "rag", "agent", "multi_agent", "multi_agent_4b", "dense", "retrieval", "hybrid",
                           "graph", "rerank", "final", "run_eval", "sentence_transformers", "torch", "urllib", "requests"}


@needs_results
def test_every_case_renders_with_banner_post_hoc_and_provenance_and_no_model_or_network(monkeypatch):
    import llm

    def forbidden(*a, **k):
        raise AssertionError("the replay tried to call a model or the network")
    monkeypatch.setattr(llm.OllamaClient, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for c in CASES:
        for verbose in (False, True):
            text = demo.render(c, verbose=verbose)
            assert text.count(demo.BANNER) >= 2 and demo.POST_HOC.split(",")[0] in text
            assert f"question {c['qid']}" in text and all(rel in text for rel in c["sources"].values())
            assert "sha256" in text


@needs_results
def test_pins_match_the_stored_records():
    for c in CASES:
        recs = demo.gather(c)
        ctx = demo.Context(c["split"], c["qid"])
        assert demo.pin_problems(c, recs, ctx) == [], c["id"]


@needs_results
@pytest.mark.parametrize("case_id,path,value", [
    ("graph_success", ("answers", "d3_k10"), "Chicago"),
    ("rerank_failure", ("gold_ranks", "X1", "Ron Weasley"), 9),
    ("agent_recovery", ("follow_up_queries",), ["Tunisia World Cup"]),
    ("unnecessary_search", ("stop_reason",), "sufficient"),
    ("selection_loss", ("accepted",), ["Jolyon James", "Cat Stevens"]),
])
def test_a_wrong_pin_stops_the_replay(case_id, path, value):
    c = copy.deepcopy(next(x for x in CASES if x["id"] == case_id))
    target = c["pins"]
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    with pytest.raises(SystemExit, match="do not reproduce the pinned values"):
        demo.render(c)


def test_missing_artifacts_give_a_useful_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        demo.render(CASES[0], results=str(tmp_path))
    msg = str(e.value)
    assert "missing stored artifact" in msg and "not tracked" in msg and "test_agents__bm25_k10.jsonl" in msg


def test_cli(capsys, tmp_path):
    demo.main(["--list"])
    out = capsys.readouterr().out
    assert demo.BANNER in out and all(c["id"] in out for c in CASES)
    with pytest.raises(SystemExit, match="unknown case"):
        demo.main(["--case", "nope"])
    with pytest.raises(SystemExit, match="missing stored artifact"):
        demo.main(["--all", "--results", str(tmp_path)])
