"""One command line for every Phase 1 run. Results go to ``results/<split>/`` (local, not
tracked) with a manifest next to each file.

    python run_eval.py format-check                 # 20 eda questions, own contexts
    python run_eval.py grid --split tune            # 12 BM25 candidates, retrieval only
    python run_eval.py retrieval --split tune       # selected BM25: curves and latency
    python run_eval.py rag --split dev_smoke --k 5
    python run_eval.py rag --split dev --k 3        # also 5, 10
    python run_eval.py closed-book --split dev
    python run_eval.py oracle --split dev
    python run_eval.py phase2a --split tune         # Phase 2a: 9 classic configurations, retrieval only
    python run_eval.py replay-baseline              # cache-only replay of the Phase 1 K=10 dev run
    python run_eval.py phase2b --split tune         # Phase 2b: dense and hybrid, retrieval only
    python run_eval.py posthoc-x1 --split tune      # post hoc P1: X1 = cross-encoder reranking of D3
    python run_eval.py assess-check                 # Phase 3: assess_v1 format check (40 calls, eda)
    python run_eval.py agent --split dev_smoke      # Phase 3: the single agent (also --split dev)
    python run_eval.py replay-run --run d3_k10 --phase p4   # cache-only replay of a stored run (bm25_k10, d3_k16)
    python run_eval.py rag --split dev --k 13 --retriever d3   # matched-context control (K from the agent)
    python run_eval.py ma-check                     # Phase 4: miniature-flow format check (eda)
    python run_eval.py ma-check-replay              # A4.1: offline re-validation of ma_check_v2's verifier outputs
    python run_eval.py replay-multi-agent           # Phase 4: cache-only reproduction of the multi-agent dev run
    python run_eval.py ma4b-g0                      # Phase 4b: live structured-decoding mechanics gate (synthetic)
    python run_eval.py calib4b-flow --subset c3     # Phase 4b: the structured flow on a calibration subset
    python run_eval.py calib4b-probe --subset c3    # Phase 4b: per-requirement verifier probes and gates G1-G5
    python run_eval.py multi-agent-4b --split dev   # Phase 4b: the structured multi-agent system
    python run_eval.py final --stage retrieval      # Phase 5: one stage per invocation (retrieval, rag, agents)
    python run_eval.py final-analysis               # Phase 5: §5.5-5.6 from the stored runs
    python run_eval.py audit-sheet                  # Phase 5: the blinded audit sheet, key and label template
    python run_eval.py audit-analysis               # Phase 5: validated labels joined with the key (§5.7)
    python run_eval.py calib-flow --subset c1       # A5: the multi-agent flow on a calibration subset (eda)
    python run_eval.py calib-probe --subset c1      # A5: isolated verifier probes on the committed reference requirements
    python run_eval.py multi-agent --split dev_smoke  # Phase 4: the multi-agent system (also --split dev)
    python run_eval.py replay-agent                 # Phase 4: cache-only replay of the Phase 3 agent

LLM calls go through the replay cache, so an interrupted run resumes by re-running the
same command. Blind splits are refused: this script has no unblind option in Phase 1.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import agent as agent_mod
import multi_agent as ma_mod
import multi_agent_4b as ma4b_mod
import data
import dense
import feedback
import fusion
import graph
import llm
import manifest
import metrics
import prompts
import rag
import rerank
import retrieval
import scoring

RESULTS = os.path.join(data.HERE, "results")
GRID_KS = (3, 5, 10)
CURVE_KS = (2, 3, 5, 10, 20, 50)
PLATEAU = 0.02


def _path(split: str, name: str, ext: str = "jsonl") -> str:
    os.makedirs(os.path.join(RESULTS, split), exist_ok=True)
    return os.path.join(RESULTS, split, f"{name}.{ext}")


def _write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json(path: str, obj: Any) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _answer_config(system: str, **extra: Any) -> Dict[str, Any]:
    h = prompts.artifact_hashes()
    return {"system": system, "answer_prompt": "answer_v2", "answer_prompt_hash": h["answer_v2"],
            "schema": "schema_v2", "schema_hash": h["schema_v2"], "answer_validator_hash": h["answer_validator_v2"],
            "system_prompt_hash": h["system_prompt"], "renderer_hash": h["render_v1"],
            "overflow_rule": "drop last displayed paragraph until the prompt fits", **extra}


def _save(split: str, name: str, ids: Sequence[str], rows, config, client=None, runtime=None) -> str:
    path = _path(split, name)
    _write_jsonl(path, rows)
    m = manifest.build(split if not split.startswith("eda_") else "adhoc:" + split, ids, config,
                       llm=client.describe() if client else None, runtime_s=runtime)
    manifest.write(m, path)
    return path


def _progress(i: int, n: int, q: data.Question, rec: Mapping[str, Any]) -> None:
    ans = rec["output"]["answer"]
    tag = "cache" if rec.get("llm", {}).get("cached") else f"{rec.get('llm', {}).get('wall_s', 0):.1f}s"
    print(f"[{i + 1:3d}/{n}] {q.qid} {tag:>6}  {ans[:50]!r}", flush=True)


# ---------------------------------------------------------------- format check

def format_failures(rec: Mapping[str, Any]) -> List[str]:
    """Expanded format criteria (PREREGISTRATION.md §1.4 and amendment A1): schema failure,
    mis-labelled citation, and the answer flags empty / encoded / long."""
    fails = []
    if rec["schema_error"]:
        fails.append("schema")
    cv = metrics.citation_validity([tuple(c) for c in rec["citations"]],
                                   dict(zip(rec["displayed"], rec["displayed_sentences"])))
    if cv["valid"] < cv["n"]:
        fails.append("citation_label")
    fails += [f"answer_{f}" for f in scoring.answer_format_flags(rec["output"]["answer"])]
    return fails


def cmd_format_check(args) -> None:
    # Amendment A1: version 2 is checked exactly once on the same 20 questions. The
    # version-1 result (format_check.jsonl) is kept as the record of the original check.
    out = _path("eda_format", "format_check_v2")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: the v2 check runs exactly once (amendment A1)")
    eda = data.load_split("eda")
    qs = [q for q in eda if q.type == "bridge"][:16] + [q for q in eda if q.type == "comparison"][:4]
    ctx = data.load_contexts("eda")
    client = llm.OllamaClient()
    t0, rows = time.perf_counter(), []
    for i, q in enumerate(qs):
        rec = rag.own_context(client, q, ctx[q.qid])
        rec["format_failures"] = format_failures(rec)
        rec["em_f1"] = metrics.answer_scores(rec["output"]["answer"], q.answer)  # recorded, not a criterion
        rows.append(rec)
        _progress(i, len(qs), q, rec)
    failed = [r for r in rows if r["format_failures"]]
    path = _save("eda_format", "format_check_v2", [q.qid for q in qs], rows,
                 _answer_config("own_context_format_check"), client, time.perf_counter() - t0)
    print(f"\nformat failures (expanded criteria): {len(failed)}/20 (> 2 stops for review)")
    for r in failed:
        print(f"  {r['qid']}: {r['format_failures']}  answer={r['output']['answer']!r}")
    print("hashes:", json.dumps(prompts.artifact_hashes(), indent=1))
    print("saved", path)


# ---------------------------------------------------------------- BM25 selection on tune

def cmd_grid(args) -> None:
    if args.split != "tune":
        raise SystemExit("the BM25 grid is preregistered on tune only")
    qs = data.load_split("tune")
    ids = data.doc_id_of()
    rows, per_question = [], {}
    t0 = time.perf_counter()
    for rank, cfg in enumerate(retrieval.CANDIDATES, 1):
        r = retrieval.BM25Retriever(cfg)
        jr = {k: [] for k in GRID_KS}
        per_question[cfg.name] = {}
        for q in qs:
            ranked = [d for d, _ in r.search(q.question, k=max(GRID_KS))]
            gold = [ids[g] for g in q.gold]
            m = metrics.recall_at_k(ranked, gold, GRID_KS)
            for k in GRID_KS:
                jr[k].append(m[f"joint_recall@{k}"])
            per_question[cfg.name][q.qid] = {f"joint_recall@{k}": m[f"joint_recall@{k}"] for k in GRID_KS}
        row = {"preference": rank, "config": cfg.name, **{f"joint_recall@{k}": float(np.mean(jr[k])) for k in GRID_KS}}
        row["selection_score"] = float(np.mean([row[f"joint_recall@{k}"] for k in GRID_KS]))
        row["build_s"] = r.build_seconds
        rows.append(row)
        print(f"{rank:2d} {cfg.name:32s} JR@3 {row['joint_recall@3']:.3f} JR@5 {row['joint_recall@5']:.3f} "
              f"JR@10 {row['joint_recall@10']:.3f}  score {row['selection_score']:.4f}", flush=True)
    best = max(r["selection_score"] for r in rows)
    plateau = [r for r in rows if r["selection_score"] >= best - PLATEAU]
    selected = min(plateau, key=lambda r: r["preference"])
    result = {"rows": rows, "best": best, "plateau": [r["config"] for r in plateau],
              "selected": selected["config"], "per_question": per_question}
    path = _path("tune", "bm25_grid", "json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    manifest.write(manifest.build("tune", [q.qid for q in qs],
                                  {"system": "bm25_grid", "candidates": [c.name for c in retrieval.CANDIDATES],
                                   "selection": "mean JR@{3,5,10}, plateau 0.02, preference order"},
                                  runtime_s=time.perf_counter() - t0), path)
    print(f"\nbest {best:.4f}; plateau {result['plateau']}\nSELECTED: {selected['config']}\nsaved {path}")


def _selected() -> retrieval.BM25Config:
    if retrieval.SELECTED is None:
        raise SystemExit("retrieval.SELECTED is not pinned yet: run the tune grid first")
    return retrieval.SELECTED


def cmd_retrieval(args) -> None:
    if args.split != "tune":
        raise SystemExit("Phase 1 reports selected-BM25 retrieval on tune only")
    qs = data.load_split("tune")
    r = retrieval.BM25Retriever(_selected())
    for q in data.load_split("eda")[:10]:          # warm-up (train questions)
        r.search(q.question)
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        hits = r.search(q.question, k=100)
        dt = time.perf_counter() - t0
        rows.append({"qid": q.qid, "type": q.type, "latency_s": dt, "ranked": [d for d, _ in hits],
                     **scoring.retrieval_scores(q, [d for d, _ in hits], CURVE_KS)})
    path = _save("tune", "bm25_selected_retrieval", [q.qid for q in qs], rows,
                 {"system": "bm25_selected", **r.describe(), "vocabulary_size": r.vocabulary_size})
    lat = metrics.latency_stats([row["latency_s"] for row in rows])
    for k in CURVE_KS:
        print(f"K={k:2d}  recall {np.mean([x[f'recall@{k}'] for x in rows]):.3f}  "
              f"joint {np.mean([x[f'joint_recall@{k}'] for x in rows]):.3f}")
    print(f"latency p50 {lat['p50_s'] * 1000:.1f} ms, p95 {lat['p95_s'] * 1000:.1f} ms; build {r.build_seconds:.1f} s\nsaved {path}")


# ---------------------------------------------------------------- LLM runs on dev

def _dev_questions(split: str) -> List[data.Question]:
    if split not in ("dev", "dev_smoke"):
        raise SystemExit("Phase 1 LLM runs use dev or dev_smoke only")
    return data.load_split(split)


def _run_llm(split: str, name: str, config: Dict[str, Any], fn) -> None:
    qs = _dev_questions(split)
    client = llm.OllamaClient()
    t0, rows = time.perf_counter(), []
    for i, q in enumerate(qs):
        rec = fn(client, q)
        rows.append(rec)
        _progress(i, len(qs), q, rec)
    path = _save(split, name, [q.qid for q in qs], rows, config, client, time.perf_counter() - t0)
    evidence = name != "closed_book"
    scores = [scoring.score(q, r, evidence=evidence) for q, r in zip(qs, rows)]
    summ = scoring.summarize(scores)
    print("\n" + "  ".join(f"{k} {v['mean']:.3f}" for k, v in summ.items()))
    print(f"generations: {sum(not r['llm']['cached'] for r in rows)} new, "
          f"{sum(r['llm']['cached'] for r in rows)} replayed; saved {path}")


def d3_retriever():
    """The frozen overall retriever R = D3 (dense corpus matrix from the cache only)."""
    import hybrid
    ids, matrix, info = dense.corpus_embeddings("cuda")
    sel = json.load(open(_path("tune", "phase2b_selection", "json")))
    assert sel["R"] == "D3" and info["fingerprint"] == sel["embedding"]["fingerprint"]
    return hybrid.D3Retriever(retrieval.BM25Retriever(retrieval.SELECTED), dense.DenseRetriever(ids, matrix),
                              graph.load_or_build())


def cmd_rag(args) -> None:
    if args.retriever == "d3":
        r = d3_retriever()
        name = f"d3_k{args.k}"
    else:
        r = retrieval.BM25Retriever(_selected())
        name = f"bm25_k{args.k}"
    cfg = _answer_config(name, k=args.k, retriever=r.describe())
    _run_llm(args.split, name, cfg, lambda c, q: rag.bm25_rag(c, r, q, args.k))


def cmd_oracle(args) -> None:
    _run_llm(args.split, "oracle", _answer_config("oracle", context="two gold paragraphs, supplied order"),
             rag.oracle)


def closed_book_config() -> Dict[str, Any]:
    h = prompts.artifact_hashes()
    return {"system": "closed_book", "closed_book_prompt": "closed_book_v2", "closed_book_prompt_hash": h["closed_book_v2"],
            "schema": "schema_v2", "schema_hash": h["schema_v2"], "answer_validator_hash": h["answer_validator_v2"],
            "system_prompt_hash": h["system_prompt"]}


def cmd_closed_book(args) -> None:
    _run_llm(args.split, "closed_book", closed_book_config(), rag.closed_book)


# ---------------------------------------------------------------- Phase 2a (classic retrieval, tune only)

STEP1 = (("C0", retrieval.BM25Config()),
         ("C1", retrieval.BM25Config(normalize=True)),
         ("C2", retrieval.BM25Config(normalize=True, title_repeats=2)))
GRAPH_CONFIGS = (("C4@5", graph.C4_5), ("C4@10", graph.C4_10), ("C5@5", graph.C5_5), ("C5@10", graph.C5_10))
FEEDBACK_CONFIGS = (("C3a", feedback.C3A), ("C3b", feedback.C3B))
STEP2_ORDER = ("B", "C4@5", "C4@10", "C5@5", "C5@10", "C3a", "C3b")


def plateau_select(scores: Mapping[str, float], order: Sequence[str], tol: float = PLATEAU) -> Dict[str, Any]:
    """First member, in preference order, of the plateau within ``tol`` of the best score."""
    best = max(scores[n] for n in order)
    plateau = [n for n in order if scores[n] >= best - tol]
    return {"best": best, "plateau": plateau, "selected": plateau[0]}


def joint_recall_at_10(q: data.Question, ranking: Sequence[str]) -> float:
    ids = data.doc_id_of()
    return metrics.joint_recall(ranking[:10], [ids[g] for g in q.gold])


def phase2a_config(g: "graph.TitleGraph") -> Dict[str, Any]:
    from dataclasses import asdict
    return {"system": "phase2a_classic",
            "step1": {n: asdict(c) for n, c in STEP1},
            "graph": {"version": graph.GRAPH_VERSION, "fingerprint": g.fingerprint(),
                      "stats": {k: v for k, v in g.stats.items() if k != "build_s"},   # timing is not configuration
                      "min_alias_chars": graph.MIN_ALIAS_CHARS, "max_alias_df": graph.MAX_SINGLE_TOKEN_DF,
                      "configs": {n: asdict(c) for n, c in GRAPH_CONFIGS}},
            "feedback": {n: asdict(c) for n, c in FEEDBACK_CONFIGS},
            "normalization": "norm_v1", "mention_tokenization": "tok_v1",
            "selection": "overall exact Joint Recall@10; plateau 0.02; step 1 order C0,C1,C2; "
                         "step 2 order " + ",".join(STEP2_ORDER)}


def save_phase2a_selection(result: Mapping[str, Any], qids: Sequence[str], config: Mapping[str, Any],
                           runtime: float) -> str:
    """The selection summary, written atomically with its own manifest. Timing (graph build,
    runtime) is reported in the file and the manifest's run metadata, never in its
    configuration."""
    path = _path("tune", "phase2a_selection", "json")
    _write_json(path, result)
    manifest.write(manifest.build("tune", qids, config, runtime_s=runtime), path)
    return path


def cmd_phase2a(args) -> None:
    if args.split != "tune":
        raise SystemExit("Phase 2a selection is preregistered on tune only")
    qs = data.load_split("tune")
    warm = data.load_split("eda")[:10]
    rows = {q.qid: {"qid": q.qid, "type": q.type, "rankings": {}, "latency_s": {}, "added": {}, "feedback": {}}
            for q in qs}
    t_start = time.perf_counter()

    # step 1: lexical base
    retrievers = {}
    for name, cfg in STEP1:
        r = retrieval.BM25Retriever(cfg)
        retrievers[name] = r
        for w in warm:
            r.search(w.question)
        for q in qs:
            t0 = time.perf_counter()
            ranking = [d for d, _ in r.search(q.question, k=100)]
            rows[q.qid]["latency_s"][name] = time.perf_counter() - t0
            rows[q.qid]["rankings"][name] = ranking
    phase1 = {x["qid"]: x["ranked"] for x in read_jsonl(_path("tune", "bm25_selected_retrieval"))}
    assert all(rows[q.qid]["rankings"]["C0"] == phase1[q.qid] for q in qs), "C0 differs from Phase 1"
    s1 = {n: float(np.mean([joint_recall_at_10(q, rows[q.qid]["rankings"][n]) for q in qs])) for n, _ in STEP1}
    sel1 = plateau_select(s1, [n for n, _ in STEP1])
    B = sel1["selected"]
    rb = retrievers[B]

    # step 2: expansions on B
    g = graph.load_or_build()
    for w in warm:
        rb.scores(w.question)
    for q in qs:
        base = rows[q.qid]["rankings"][B]
        for name, cfg in GRAPH_CONFIGS:
            t0 = time.perf_counter()
            base_now = [d for d, _ in rb.search(q.question, k=100)]
            lex = rb.score_of(rb.scores(q.question)) if cfg.backward else (lambda d: 0.0)
            ranking, added = graph.expand(base_now, g, cfg, lex)
            rows[q.qid]["latency_s"][name] = time.perf_counter() - t0
            assert base_now == base
            rows[q.qid]["rankings"][name] = ranking
            rows[q.qid]["added"][name] = added
        for name, cfg in FEEDBACK_CONFIGS:
            t0 = time.perf_counter()
            ranking, info = feedback.rank(rb, q.question, cfg)
            rows[q.qid]["latency_s"][name] = time.perf_counter() - t0
            rows[q.qid]["rankings"][name] = ranking
            rows[q.qid]["feedback"][name] = info
    s2 = {"B": s1[B]}
    for name in STEP2_ORDER[1:]:
        s2[name] = float(np.mean([joint_recall_at_10(q, rows[q.qid]["rankings"][name]) for q in qs]))
    sel2 = plateau_select(s2, STEP2_ORDER)
    result = {"step1": {"scores": s1, **sel1}, "B": B, "step2": {"scores": s2, **sel2},
              "C": sel2["selected"] if sel2["selected"] != "B" else B,
              "n_questions": len(qs), "configurations_evaluated": len(STEP1) + len(STEP2_ORDER) - 1,
              "graph_fingerprint": g.fingerprint(), "graph_stats": g.stats}
    runtime = time.perf_counter() - t_start
    path = _save("tune", "phase2a_rankings", [q.qid for q in qs], list(rows.values()), phase2a_config(g),
                 runtime=runtime)
    save_phase2a_selection(result, [q.qid for q in qs], phase2a_config(g), runtime)
    print("step 1 (Joint Recall@10):", {k: round(v, 4) for k, v in s1.items()}, "-> B =", B)
    print("step 2 (Joint Recall@10):", {k: round(v, 4) for k, v in s2.items()})
    print(f"plateau {sel2['plateau']} -> classic champion C = {result['C']}\nsaved {path}")


# ---------------------------------------------------------------- Phase 2b (dense and hybrid, tune only)

PHASE2B_ORDER = ("D0", "D1", "D2", "D3", "D4")
G_STAR = graph.C5_5          # the frozen graph operator (best C4/C5 configuration in Phase 2a)


def dense_seeded_graph(b_ranking: Sequence[str], d1_ranking: Sequence[str], g, lexical_score):
    """D3 = G*(H) with H = RRF(B, D1); backward candidates keep B's lexical scores."""
    h = fusion.rrf([b_ranking, d1_ranking])
    ranking, added = graph.expand(h, g, G_STAR, lexical_score)
    return ranking, added, h


def unload_ollama() -> None:
    """Free the GPU before encoding: ask Ollama to unload every loaded model."""
    t = llm.http_transport()
    for m in t("/api/ps", None).get("models", []):
        t("/api/generate", {"model": m["name"], "keep_alive": 0})


def phase2b_config(dense_info: Mapping[str, Any], g) -> Dict[str, Any]:
    from dataclasses import asdict
    return {"system": "phase2b_dense_hybrid",
            "incumbent": {"C": "C5@5", "B": "C0", "graph_fingerprint": g.fingerprint()},
            "dense": {"model": dense.MODEL, "revision": dense.REVISION, "cache_key": dense_info["key"],
                      "embedding_fingerprint": dense_info["fingerprint"], "shape": dense_info["shape"],
                      "query_device": "cpu", "similarity": "dot product", "search": "exact numpy"},
            "rrf": {"c": fusion.C, "pool": fusion.POOL},
            "d3": {"seeds": "RRF(B, D1)", "operator": asdict(G_STAR), "backward_score": "B lexical"},
            "rerank": {"model": rerank.MODEL, "revision": rerank.REVISION, "max_length": rerank.MAX_LENGTH,
                       "batch": rerank.BATCH, "pool": rerank.POOL, "base": "D2", "device": "cuda"},
            "selection": "overall exact Joint Recall@10; plateau 0.02; order " + ",".join(PHASE2B_ORDER)}


def cmd_phase2b(args) -> None:
    if args.split != "tune":
        raise SystemExit("Phase 2b selection is preregistered on tune only")
    sel2a = json.load(open(_path("tune", "phase2a_selection", "json")))
    assert (sel2a["B"], sel2a["C"]) == ("C0", "C5@5"), "Phase 2b is preregistered on B = C0, C = C5@5"
    qs = data.load_split("tune")
    warm = data.load_split("eda")[:10]
    old = {x["qid"]: x for x in read_jsonl(_path("tune", "phase2a_rankings"))}
    rb = retrieval.BM25Retriever(retrieval.SELECTED)
    g = graph.load_or_build()
    assert g.fingerprint() == sel2a["graph_fingerprint"]
    t_start = time.perf_counter()

    unload_ollama()
    ids, matrix, info = dense.corpus_embeddings("cuda")
    dr = dense.DenseRetriever(ids, matrix)
    for w in warm:
        dr.search(w.question)
        rb.scores(w.question)
    rows = {}
    for q in qs:
        row = rows[q.qid] = {"qid": q.qid, "type": q.type, "rankings": {}, "latency_s": {}, "added": {}}
        t0 = time.perf_counter()
        b_rank = [d for d, _ in rb.search(q.question, k=100)]
        lex = rb.score_of(rb.scores(q.question))
        c_rank, _ = graph.expand(b_rank, g, graph.C5_5, lex)
        t_c = time.perf_counter() - t0
        assert b_rank == old[q.qid]["rankings"]["C0"] and c_rank == old[q.qid]["rankings"]["C5@5"]
        t0 = time.perf_counter()
        d1 = [d for d, _ in dr.search(q.question, k=100)]
        t_d1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        d2 = fusion.rrf([c_rank, d1])
        t_d2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        d3, added, h = dense_seeded_graph(b_rank, d1, g, lex)
        t_d3 = time.perf_counter() - t0
        row["rankings"].update({"D0": c_rank, "D1": d1, "D2": d2, "D3": d3, "B": b_rank, "H": h})
        row["added"]["D3"] = added
        row["latency_s"].update({"D0": t_c, "D1": t_d1, "D2": t_c + t_d1 + t_d2, "D3": t_c + t_d1 + t_d3})

    import torch
    ce = rerank.load_model("cuda")
    for w in warm:
        rerank.rerank(ce, w.question, rows[qs[0].qid]["rankings"]["D2"])
    for q in qs:
        row = rows[q.qid]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d4, _ = rerank.rerank(ce, q.question, row["rankings"]["D2"])
        torch.cuda.synchronize()
        row["rankings"]["D4"] = d4
        row["latency_s"]["D4"] = row["latency_s"]["D2"] + (time.perf_counter() - t0)
        row["latency_s"]["rerank_gpu"] = time.perf_counter() - t0
    ce_cpu = rerank.load_model("cpu")
    cpu_rerank = []
    for q in qs[:20]:
        t0 = time.perf_counter()
        rerank.rerank(ce_cpu, q.question, rows[q.qid]["rankings"]["D2"])
        cpu_rerank.append(time.perf_counter() - t0)

    scores = {n: float(np.mean([joint_recall_at_10(q, rows[q.qid]["rankings"][n]) for q in qs])) for n in PHASE2B_ORDER}
    sel = plateau_select(scores, PHASE2B_ORDER)
    runtime = time.perf_counter() - t_start
    cfg = phase2b_config(info, g)
    path = _save("tune", "phase2b_rankings", [q.qid for q in qs], list(rows.values()), cfg, runtime=runtime)
    result = {"scores": scores, **sel, "R": sel["selected"], "n_questions": len(qs),
              "embedding": {k: info[k] for k in ("fingerprint", "shape", "encode_s", "bytes", "peak_gpu_bytes")},
              "rerank_cpu_latency_s": metrics.latency_stats(cpu_rerank),
              "graph_fingerprint": g.fingerprint()}
    sel_path = _path("tune", "phase2b_selection", "json")
    _write_json(sel_path, result)
    manifest.write(manifest.build("tune", [q.qid for q in qs], cfg, runtime_s=runtime), sel_path)
    print("Joint Recall@10:", {k: round(v, 4) for k, v in scores.items()})
    print(f"plateau {sel['plateau']} -> overall retriever R = {sel['selected']}\nsaved {path}")


# ---------------------------------------------------------------- post hoc P1: X1 = rerank(D3), tune only

X1_CARRY_GAP = 0.02


def rankings_fingerprint(rankings: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 of a {qid: ranking} mapping in canonical JSON."""
    return manifest.sha256_of({q: list(v) for q, v in rankings.items()})


def x1_rerank(model, question: str, d3_ranking: Sequence[str]):
    """Rerank exactly D3's first 50 with D4's settings; reranked top 10, then D3's order."""
    return rerank.rerank(model, question, d3_ranking, pool=rerank.POOL)


def x1_config(d3_fingerprint: str) -> Dict[str, Any]:
    return {"system": "X1_rerank_D3", "posthoc": True, "protocol": "PREREGISTRATION.md P1",
            "base": "stored D3 tune ranking", "d3_rankings_fingerprint": d3_fingerprint,
            "rerank": {"model": rerank.MODEL, "revision": rerank.REVISION, "max_length": rerank.MAX_LENGTH,
                       "batch": rerank.BATCH, "pool": rerank.POOL, "input": "(question, title. text)",
                       "order": "logit, then doc_id", "device": "cuda"},
            "carry_rule": f"X1 - D3 overall Joint Recall@10 > {X1_CARRY_GAP}"}


def cmd_posthoc_x1(args) -> None:
    if args.split != "tune":
        raise SystemExit("post hoc X1 is defined on tune only")
    import torch
    qs = data.load_split("tune")
    stored = {x["qid"]: x["rankings"] for x in read_jsonl(_path("tune", "phase2b_rankings"))}
    d3 = {q.qid: stored[q.qid]["D3"] for q in qs}
    fp = rankings_fingerprint(d3)
    unload_ollama()
    ce = rerank.load_model("cuda")
    for q in qs[:10]:
        x1_rerank(ce, q.question, d3[q.qid])
    rows, t_start = [], time.perf_counter()
    for q in qs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ranking, _ = x1_rerank(ce, q.question, d3[q.qid])
        torch.cuda.synchronize()
        rows.append({"qid": q.qid, "type": q.type, "ranking": ranking, "rerank_gpu_s": time.perf_counter() - t0})
    ce_cpu = rerank.load_model("cpu")
    cpu = []
    for q in qs[:20]:                                   # the same 20 questions as D4's CPU timing
        t0 = time.perf_counter()
        x1_rerank(ce_cpu, q.question, d3[q.qid])
        cpu.append(time.perf_counter() - t0)
    x1_jr = float(np.mean([joint_recall_at_10(q, r["ranking"]) for q, r in zip(qs, rows)]))
    d3_jr = float(np.mean([joint_recall_at_10(q, d3[q.qid]) for q in qs]))
    runtime = time.perf_counter() - t_start
    cfg = x1_config(fp)
    path = _save("tune", "posthoc_x1_rankings", [q.qid for q in qs], rows, cfg, runtime=runtime)
    summary = {"posthoc": True, "X1_joint_recall_at_10": x1_jr, "D3_joint_recall_at_10": d3_jr,
               "difference": x1_jr - d3_jr, "carry": x1_jr - d3_jr > X1_CARRY_GAP,
               "rerank_gpu_latency_s": metrics.latency_stats([r["rerank_gpu_s"] for r in rows]),
               "rerank_cpu_latency_s": metrics.latency_stats(cpu), "d3_rankings_fingerprint": fp}
    sp = _path("tune", "posthoc_x1_summary", "json")
    _write_json(sp, summary)
    manifest.write(manifest.build("tune", [q.qid for q in qs], cfg, runtime_s=runtime), sp)
    print(f"post hoc X1: Joint Recall@10 {x1_jr:.4f} vs D3 {d3_jr:.4f} (difference {x1_jr - d3_jr:+.4f}); "
          f"carry (> {X1_CARRY_GAP}): {summary['carry']}\nsaved {path}")


# ---------------------------------------------------------------- cache-only replay of the Phase 1 baseline

def replay_baseline(client=None, split: str = "dev", k: int = 10) -> Tuple[List[Dict[str, Any]], Any]:
    """Replay the Phase 1 BM25 K=10 run from the LLM cache only (PREREGISTRATION.md §2.9).
    Aborts on any cache miss (the offline client raises), on any non-cached generation,
    and on any top-10 list that differs from Phase 1."""
    old_path = _path(split, f"bm25_k{k}")
    old = {r["qid"]: r for r in read_jsonl(old_path)}
    if client is None:
        with open(old_path + ".manifest.json") as f:
            base_llm = json.load(f)["comparison_invariants"]["base_llm"]
        client = llm.OllamaClient(offline=True, offline_digest=base_llm["digest"],
                                  offline_quantization=base_llm["quantization"])
    r = retrieval.BM25Retriever(retrieval.SELECTED)
    rows = []
    for q in data.load_split(split):
        rec = rag.bm25_rag(client, r, q, k)
        if not rec["llm"]["cached"]:
            raise RuntimeError(f"{q.qid}: generation was not a cache hit")
        if rec["ranked"][:k] != old[q.qid]["ranked"][:k] or rec["displayed"] != old[q.qid]["displayed"]:
            raise RuntimeError(f"{q.qid}: top-{k} list differs from Phase 1")
        if (rec["llm"]["wall_s"], rec["llm"]["prompt_tokens"]) != (old[q.qid]["llm"]["wall_s"], old[q.qid]["llm"]["prompt_tokens"]):
            raise RuntimeError(f"{q.qid}: cached generation metadata differs from Phase 1")
        rows.append(rec)
    return rows, client


def offline_client_for(path: str):
    """An offline, cache-only client with the LLM identity recorded in ``path``'s manifest."""
    with open(path + ".manifest.json") as f:
        inv = json.load(f)["comparison_invariants"]
    base_llm = inv["base_llm"]
    return llm.OllamaClient(offline=True, offline_digest=base_llm["digest"],
                            offline_quantization=base_llm["quantization"],
                            offline_ollama_version=inv.get("ollama_version"))   # recorded, never the installed one


REPLAYABLE = ("d3_k10", "bm25_k10", "d3_k16")


def replay_run(name: str, client=None, retriever=None, split: str = "dev") -> Tuple[List[Dict[str, Any]], Any, Any]:
    """Cache-only replay of a stored RAG run under the current code (PREREGISTRATION.md
    §3.4). Aborts on any cache miss, non-cached generation, ranking or context mismatch,
    or changed generation metadata."""
    if name not in REPLAYABLE:
        raise SystemExit(f"replayable runs: {REPLAYABLE}")
    old_path = _path(split, name)
    old = {r["qid"]: r for r in read_jsonl(old_path)}
    client = client if client is not None else offline_client_for(old_path)
    if retriever is None:
        retriever = d3_retriever() if name.startswith("d3") else retrieval.BM25Retriever(retrieval.SELECTED)
    k = int(name.split("_k")[1])
    rows = []
    for q in data.load_split(split):
        o = old[q.qid]
        if [d for d, _ in retriever.search(q.question, k=100)] != o["ranked"]:   # checked before any generation
            raise RuntimeError(f"{q.qid}: ranking differs from the stored run")
        rec = rag.bm25_rag(client, retriever, q, k)
        if not rec["llm"]["cached"]:
            raise RuntimeError(f"{q.qid}: generation was not a cache hit")
        if rec["ranked"] != o["ranked"] or rec["displayed"] != o["displayed"]:
            raise RuntimeError(f"{q.qid}: ranking or displayed context differs from the stored run")
        if (rec["llm"]["key"], rec["llm"]["wall_s"], rec["llm"]["prompt_tokens"]) != (o["llm"]["key"], o["llm"]["wall_s"], o["llm"]["prompt_tokens"]):
            raise RuntimeError(f"{q.qid}: cached generation differs from the stored run")
        rows.append(rec)
    return rows, client, retriever


def _fresh(split: str, name: str) -> str:
    """Refuse to overwrite a stored run (replays are named by the phase whose code ran them)."""
    if os.path.exists(_path(split, name)):
        raise SystemExit(f"{_path(split, name)} exists; stored runs are never overwritten")
    return name


def cmd_replay_run(args) -> None:
    name = _fresh("dev", f"{args.run}_{args.phase}replay")
    rows, client, r = replay_run(args.run)
    qs = data.load_split("dev")
    path = _save("dev", name, [q.qid for q in qs], rows,
                 _answer_config(args.run, k=int(args.run.split("_k")[1]), retriever=r.describe(), replay_of=args.run),
                 client)
    print(f"replayed {args.run}: {len(rows)} questions, all cache hits, rankings and contexts identical; saved {path}")


# ---------------------------------------------------------------- Phase 3: the single agent

ASSESS_CHECK_MAX_FAIL = 4


def assess_check_failures(q: data.Question, rec: Mapping[str, Any]) -> List[str]:
    """§3.3 failure criteria for one assess output (schema or contract error, or a
    next_query equal to the question)."""
    if rec["parsed"] is None:
        return ["schema_or_contract"]
    if not rec["parsed"]["sufficient"] and agent_mod.same_query(rec["parsed"]["next_query"], q.question):
        return ["query_equals_question"]
    return []


def removed_title(q: data.Question, paragraphs: Mapping[str, Sequence[str]]) -> str:
    """Condition (b): the gold paragraph removed for the format check."""
    import hops
    if q.type == "bridge":
        ch = hops.chain(q, paragraphs)
        return ch.second if ch.determined else ch.first
    return sorted(t for t, _ in q.gold)[1]


def cmd_assess_check(args) -> None:
    # Amendment A3: the v2 prompt is checked exactly once; the v1 result is kept.
    out = _path("eda_format", "assess_check_v2")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: the assess_v2 format check runs exactly once (amendment A3)")
    eda = data.load_split("eda")
    qs = [q for q in eda if q.type == "bridge"][:16] + [q for q in eda if q.type == "comparison"][:4]
    ctx = data.load_contexts("eda")
    client = llm.OllamaClient()
    rows = []
    for q in qs:
        for cond in ("a_full", "b_removed"):
            paras = dict(ctx[q.qid])
            removed = None
            if cond == "b_removed":
                removed = removed_title(q, ctx[q.qid])
                paras.pop(removed)
            r = client.generate(prompts.assess_prompt(q.question, list(paras.items())), prompts.AssessOutput)
            rec = {"qid": q.qid, "type": q.type, "condition": cond, "removed": removed, "parsed": r.parsed,
                   "error": r.error, "raw": r.text, "llm": {"key": r.key, "wall_s": r.wall_s, "cached": r.cached}}
            rec["failures"] = assess_check_failures(q, rec)
            rows.append(rec)
            print(f"{q.qid} {cond:9s} {'ERR ' + str(r.error) if r.parsed is None else ('sufficient' if r.parsed['sufficient'] else 'query: ' + r.parsed['next_query'])[:70]}", flush=True)
    failed = [x for x in rows if x["failures"]]
    valid = [x for x in rows if x["parsed"] is not None]
    viable = any(x["parsed"]["sufficient"] for x in valid) and any(not x["parsed"]["sufficient"] for x in valid)
    cfg = {"system": "assess_format_check", **prompts.assess_hashes()}
    _write_jsonl(out, rows)
    manifest.write(manifest.build("adhoc:eda_assess_check", [f"{x['qid']}:{x['condition']}" for x in rows], cfg,
                                  llm=client.describe()), out)
    by = {c: sum(1 for x in valid if x["condition"] == c and x["parsed"]["sufficient"]) for c in ("a_full", "b_removed")}
    print(f"\nassess_v2: failures {len(failed)}/40 (> {ASSESS_CHECK_MAX_FAIL} stops for review, no second revision); "
          f"viability gate: {viable}")
    print(f"sufficient=true in (a) full context: {by['a_full']}/20; in (b) one gold removed: {by['b_removed']}/20")
    for x in failed:
        print(f"  {x['qid']} {x['condition']}: {x['failures']} {x['error'] or ''}")
    print("hashes:", json.dumps(prompts.assess_hashes(), indent=1))


def agent_config(retriever) -> Dict[str, Any]:
    import inspect
    return {**_answer_config("agent_single", k="10+5+5"), **{f"{k}_hash": v for k, v in prompts.assess_hashes().items()},
            "agent_source_hash": manifest.sha256_of(inspect.getsource(agent_mod)),
            "budget": {"max_searches": agent_mod.MAX_SEARCHES, "first_take": agent_mod.FIRST_TAKE,
                       "follow_take": agent_mod.FOLLOW_TAKE, "max_llm_calls": 3, "max_paragraphs": 20,
                       "max_query_tokens": prompts.MAX_QUERY_TOKENS},
            "overflow_rule": "evidence-preserving: search-1 lowest ranks, then earlier follow-ups, newest batch last",
            "retriever": retriever.describe()}


def cmd_agent(args) -> None:
    if args.split not in ("dev", "dev_smoke"):
        raise SystemExit("Phase 3 agent runs use dev or dev_smoke only")
    qs = data.load_split(args.split)
    r = d3_retriever()
    client = llm.OllamaClient()
    ag = agent_mod.Agent(r, client)
    rows, t0 = [], time.perf_counter()
    for i, q in enumerate(qs):
        rec = ag.run(q)
        rows.append(rec)
        new = sum(not c["cached"] for c in rec["calls"])
        print(f"[{i + 1:3d}/{len(qs)}] {q.qid} {rec['stop_reason']:15s} searches {len(rec['batches'])} calls {len(rec['calls'])} "
              f"(new {new}) paragraphs {len(rec['displayed'])}  {rec['output']['answer'][:40]!r}", flush=True)
    path = _save(args.split, "agent_single", [q.qid for q in qs], rows, agent_config(r), client, time.perf_counter() - t0)
    scores = [scoring.score(q, rec) for q, rec in zip(qs, rows)]
    summ = scoring.summarize(scores)
    print("\n" + "  ".join(f"{k} {v['mean']:.3f}" for k, v in summ.items()))
    print(f"generations: {sum(not c['cached'] for x in rows for c in x['calls'])} new; stop reasons: "
          f"{dict(__import__('collections').Counter(x['stop_reason'] for x in rows))}; saved {path}")


# ---------------------------------------------------------------- Phase 4: multi-agent

MA_ROLE_MAX_FAIL = 0.10


class OwnContextSearch:
    """Format-check stand-in search: the question's own paragraphs in supplied order,
    whatever the query (PREREGISTRATION.md §4.6)."""

    def __init__(self, ids: Sequence[str]):
        self.ids = list(ids)

    def search(self, query, k=100):
        return [(d, 1.0) for d in self.ids[:k]]


def role_failures(rec: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Per role: [calls, failures] for one flow, from its calls and error events."""
    out = {"planner": [0, 0], "researcher": [0, 0], "verifier": [0, 0]}
    for c in rec["calls"]:
        role = {"planner": "planner", "researcher": "researcher", "verifier": "verifier",
                "verifier_recheck": "verifier"}.get(c["role"])
        if role:
            out[role][0] += 1
    for e in rec["events"]:
        role = {"planner_error": "planner", "researcher_error": "researcher", "verifier_error": "verifier",
                "recheck_error": "verifier"}.get(e["event"])
        if role:
            out[role][1] += 1
    return out


def ma_viability(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    plans2 = any(len(r["requirements"]) == 2 and not any(e["event"] == "planner_error" for e in r["events"]) for r in rows)
    research = [e for r in rows for e in r["events"] if e["event"] == "research" and e.get("stage") == "planned"]
    with_next = any(e["next_query"] for e in research)
    without_next = any(e for r in rows for e in r["events"] if e["event"] == "research"
                       and (e["requirement"] == len(r["requirements"]) or e.get("stage") == "corrective"))
    two_path = any(len(r["requirements"]) == 2 and any(e["event"] == "verify" for e in r["events"])
                   and sum(1 for e in r["events"] if e["event"] in ("research", "researcher_skipped", "researcher_error")
                           and e.get("stage") == "planned") == 2 for r in rows)
    verdicts = [e["sufficient"] for r in rows for e in r["events"] if e["event"] == "verify"]
    # the verifier role includes the recheck (PREREGISTRATION.md §4.6, amendment A4)
    grounded_true = any(v and v["sufficient"] and all(c["references"] for c in v["requirements"])
                        for r in rows for v in (r["verification"], r.get("recheck")))
    corrective = any(any(c["role"] == "verifier_recheck" for c in r["calls"])
                     and any(e.get("stage") == "corrective" and e["event"] in ("research", "researcher_skipped", "researcher_error")
                             for e in r["events"]) for r in rows)
    return {"two_requirement_plan": plans2, "researcher_with_next": with_next, "researcher_without_next": bool(without_next),
            "two_requirement_path_to_verifier": two_path, "verifier_true_grounded": grounded_true,
            "verifier_false": any(v is False for v in verdicts), "corrective_path_with_recheck": corrective}


def run_own_context_flows(qs, ctx, client, verifier: str = "v2", system=None) -> List[Dict[str, Any]]:
    """The miniature flow of §4.6: own paragraphs, conditions alternating by index.
    ``system(retriever, client, docs)`` builds the flow; the Phase 4 system by default."""
    rows = []
    for i, q in enumerate(qs):
        paras = dict(ctx[q.qid])
        cond = "a_full" if i % 2 == 0 else "b_removed"
        if cond == "b_removed":
            paras.pop(removed_title(q, ctx[q.qid]))
        docs = {t: data.Doc(t, t, data.text_hash(s), tuple(s)) for t, s in paras.items()}
        build = system or (lambda rt, cl, dc: ma_mod.MultiAgent(rt, cl, docs=dc, verifier=verifier))
        rec = build(OwnContextSearch(list(docs)), client, docs).run(q)
        rec["condition"] = cond
        rows.append(rec)
        print(f"{q.qid} {cond:9s} reqs {len(rec['requirements'])} calls {rec['llm_calls']} "
              f"events {[e['event'] for e in rec['events']]}", flush=True)
    return rows


def cmd_ma_check(args) -> None:
    # Amendment A4: verifier_v2, rerun exactly once; the v1 result is kept.
    out = _path("eda_format", "ma_check_v2")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: the verifier_v2 format check runs exactly once (amendment A4)")
    eda = data.load_split("eda")
    qs = [q for q in eda if q.type == "bridge"][:16] + [q for q in eda if q.type == "comparison"][:4]
    client = llm.OllamaClient()
    rows = run_own_context_flows(qs, data.load_contexts("eda"), client)
    per = {"planner": [0, 0], "researcher": [0, 0], "verifier": [0, 0]}
    for r in rows:
        for role, (n, f) in role_failures(r).items():
            per[role][0] += n
            per[role][1] += f
    gates = {role: f <= MA_ROLE_MAX_FAIL * n for role, (n, f) in per.items()}
    viability = ma_viability(rows)
    cfg = {"system": "ma_format_check", **prompts.multi_agent_hashes()}
    _write_jsonl(out, rows)
    manifest.write(manifest.build("adhoc:eda_ma_check", [f"{r['qid']}:{r['condition']}" for r in rows], cfg,
                                  llm=client.describe()), out)
    print("\nper role [calls, failures]:", per, "-> within 10%:", gates)
    print("viability:", viability, "-> all:", all(viability.values()))
    print("hashes:", json.dumps(prompts.multi_agent_hashes(), indent=1))


def cached_response(key: str) -> str:
    with open(os.path.join(llm.CACHE_DIR, key[:2], key + ".json")) as f:
        return json.load(f)["response"]


def replay_verifier_contract(rows: Sequence[Mapping[str, Any]], sentences: Mapping[str, Mapping[str, Sequence[str]]]):
    """Amendment A4.1: re-validate every recorded verifier and recheck output under the
    repaired contract, from the cache only, against what each call displayed."""
    import pydantic
    out = []
    for r in rows:
        for c in r["calls"]:
            if not c["role"].startswith("verifier"):
                continue
            raw = cached_response(c["key"])
            rec = {"qid": r["qid"], "condition": r.get("condition"), "role": c["role"], "key": c["key"]}
            try:
                parsed = prompts.VerifierOutput.model_validate_json(raw).model_dump()
            except pydantic.ValidationError:
                out.append({**rec, "failure": "schema", "notes": []})
                continue
            counts = [len(sentences[r["qid"]][d]) for d in c["displayed"]]
            _, notes, problem = prompts.verifier_contract_v2(parsed, len(r["requirements"]), counts)
            out.append({**rec, "failure": problem, "notes": notes})
    return out


def cmd_ma_check_replay(args) -> None:
    rows = read_jsonl(_path("eda_format", "ma_check_v2"))
    res = replay_verifier_contract(rows, data.load_contexts("eda"))
    fails = [x for x in res if x["failure"]]
    norms = [x for x in res if x["notes"]]
    out = _path("eda_format", "ma_check_v2_repaired")
    _write_jsonl(out, res)
    manifest.write(manifest.build("adhoc:eda_ma_check_replay", [f"{x['qid']}:{x['condition']}:{x['role']}:{x['key'][:12]}" for x in res],
                                  {"system": "ma_check_v2_verifier_replay", "contract": "verifier_contract_v2",
                                   **prompts.multi_agent_hashes()}), out)
    print(f"verifier calls re-validated (cache only): {len(res)}; hard failures {len(fails)} = {len(fails) / len(res):.1%} "
          f"(gate 10%: {'passes' if len(fails) <= MA_ROLE_MAX_FAIL * len(res) else 'fails'})")
    for x in fails:
        print(f"  hard failure {x['qid']} {x['condition']} {x['role']}: {x['failure']}")
    print(f"normalized calls: {len(norms)}")
    for x in norms:
        print(f"  {x['qid']} {x['condition']} {x['role']}: {x['notes']}")


# ---------------------------------------------------------------- A5 calibration

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
CALIB_TARGETS = {"S1": 0.80, "S2": 0.80, "S3": 0.70, "S4": 0.70}
PLAN_LABELS = ("correct", "incomplete", "wrong_target")


CALIB_HASHES = {   # Phase 4b: pinned membership (data.ids_hash), checked on every load
    "c3": "f06f4aede1de969bb5df8081b053b421ffb93461b1ef3e36f5ccb85cae42a3fd",
    "c4": "82205976797ec1b2f0bc1682a3867e83837924e8d2704c43833bbb93001b363a",
}


def calibration_subsets(eda: Sequence[data.Question]) -> Dict[str, List[data.Question]]:
    """A5: after the 16 bridge + 4 comparison format-check questions, in eda order, C1 is
    the next 16 + 4 and C2 the 16 + 4 after those. Phase 4b: C3 is the next 24 + 6 and C4
    the 24 + 6 after those."""
    b = [q for q in eda if q.type == "bridge"]
    c = [q for q in eda if q.type == "comparison"]
    out = {"c1": b[16:32] + c[4:8], "c2": b[32:48] + c[8:12], "c3": b[48:72] + c[12:18], "c4": b[72:96] + c[18:24]}
    for k, h in CALIB_HASHES.items():
        if data.ids_hash([q.qid for q in out[k]]) != h:
            raise RuntimeError(f"calibration subset {k} does not match its pinned hash")
    return out


def planned_recall(rec: Mapping[str, Any], q: data.Question) -> Tuple[int, int, int]:
    """S2 for one flow: (gold accepted, gold present in the planned-stage pool, accepted),
    over the accepted set after the planned researchers."""
    gold = {t for t, _ in q.gold}
    planned = lambda e: e.get("stage") == "planned"
    pool = {d for e in rec["events"] if e["event"] == "search" and planned(e) for d in e["added"]}
    acc = {d for e in rec["events"] if e["event"] == "research" and planned(e) for d in e["accepted"]}
    return len(acc & gold & pool), len(gold & pool), len(acc)


def label_sheet(rows, qs) -> str:
    """The hand-labelling sheet for S1 and the reference requirements (training labels)."""
    by = {q.qid: q for q in qs}
    ctx = data.load_contexts("eda")
    out = []
    for r in rows:
        q = by[r["qid"]]
        planner_ok = not any(e["event"] == "planner_error" for e in r["events"])
        out.append(f"## {q.qid} ({q.type}, {r['condition']})\n\n**Q:** {q.question}\n\n**Gold answer:** {q.answer}\n")
        for t, s in q.supporting_facts:
            sents = ctx[q.qid].get(t, [])
            out.append(f"- [{t}, {s}] {sents[s] if s < len(sents) else '(missing)'}")
        out.append("\n**Planner:** " + ("" if planner_ok else "(planner error; fallback) ")
                   + " | ".join(f"{i}. {x}" for i, x in enumerate(r["requirements"], 1)) + "\n")
    return "\n".join(out) + "\n"


def cmd_calib_flow(args) -> None:
    qs = calibration_subsets(data.load_split("eda"))[args.subset]
    args.tag = calib_tag(args)
    out = _path("eda_calib", f"{args.subset}_flow_{args.tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: each calibration flow runs once")
    client = llm.OllamaClient()
    rows = run_own_context_flows(qs, data.load_contexts("eda"), client, args.verifier)
    _write_jsonl(out, rows)
    cfg = {"system": f"ma_calibration_flow_{args.tag}", "subset": args.subset, "verifier": args.verifier,
           **prompts.multi_agent_hashes()}
    manifest.write(manifest.build(f"adhoc:eda_calib_{args.subset}", [r["qid"] for r in rows], cfg,
                                  llm=client.describe()), out)
    with open(_path("eda_calib", f"{args.subset}_label_sheet_{args.tag}", "md"), "w") as f:
        f.write(label_sheet(rows, qs))
    by = {q.qid: q for q in qs}
    s = np.array([planned_recall(r, by[r["qid"]]) for r in rows])
    print(f"\nS2 (pooled): gold recall {s[:, 0].sum()}/{s[:, 1].sum()} = {s[:, 0].sum() / max(s[:, 1].sum(), 1):.3f}; "
          f"precision {s[:, 0].sum()}/{s[:, 2].sum()} = {s[:, 0].sum() / max(s[:, 2].sum(), 1):.3f}")
    per = {"planner": [0, 0], "researcher": [0, 0], "verifier": [0, 0]}
    for r in rows:
        for role, (n, f) in role_failures(r).items():
            per[role][0] += n
            per[role][1] += f
    print("per role [calls, failures]:", per)
    print("initial verdicts:", [e["sufficient"] for r in rows for e in r["events"] if e["event"] == "verify"])


def calib_tag(args) -> str:
    """Output tag: 'base' is the unrevised system (verifier_v2), as in the C1 runs."""
    tag = args.tag or ("base" if args.verifier == "v2" else args.verifier)
    if (tag == "base") != (args.verifier == "v2"):
        raise SystemExit("the tag 'base' is reserved for verifier v2")
    return tag


def committed_unchanged(path: str) -> bool:
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=os.path.dirname(path), capture_output=True)
    return run("ls-files", "--error-unmatch", path).returncode == 0 and run("diff", "--quiet", "HEAD", "--", path).returncode == 0


def load_reference_requirements(subset: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(CALIB_DIR, f"requirements_{subset}.jsonl")
    if not committed_unchanged(path):
        raise SystemExit(f"{path} must be committed, unchanged, before any probe (amendment A5)")
    refs = {r["qid"]: r for r in read_jsonl(path)}
    for r in refs.values():
        assert r["label"] in PLAN_LABELS and r["requirements"] and all(x.strip() for x in r["requirements"]), r["qid"]
    return refs


def probe_paragraphs(q: data.Question, paragraphs: Mapping[str, Sequence[str]], complete: bool):
    """The gold paragraphs in context order; the incomplete probe drops one (§3.3 rule)."""
    gold = {t for t, _ in q.gold}
    keep = gold if complete else gold - {removed_title(q, paragraphs)}
    return [(t, paragraphs[t]) for t in paragraphs if t in keep]


def score_probe(raw_parsed, n_requirements: int, paras) -> Tuple[Optional[bool], Optional[str], List[str]]:
    """(computed sufficient or None on a hard failure, failure, normalization notes)."""
    if raw_parsed is None:
        return None, "schema", []
    norm, notes, problem = prompts.verifier_contract_v2(raw_parsed, n_requirements, [len(s) for _, s in paras])
    return (None if problem else norm["sufficient"]), problem, notes


def cmd_calib_probe(args) -> None:
    qs = calibration_subsets(data.load_split("eda"))[args.subset]
    refs = load_reference_requirements(args.subset)
    args.tag = calib_tag(args)
    if set(refs) != {q.qid for q in qs}:
        raise SystemExit("the reference requirements must cover exactly the subset")
    out = _path("eda_calib", f"{args.subset}_probe_{args.tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: each probe set runs once")
    ctx = data.load_contexts("eda")
    client = llm.OllamaClient()
    rows = []
    for q in qs:
        reqs = refs[q.qid]["requirements"]
        for complete in (True, False):
            paras = probe_paragraphs(q, ctx[q.qid], complete)
            r = client.generate(prompts.verifier_prompt(q.question, reqs, paras, args.verifier), prompts.VerifierOutput)
            suff, problem, notes = score_probe(r.parsed, len(reqs), paras)
            rows.append({"qid": q.qid, "probe": "complete" if complete else "incomplete", "displayed": [t for t, _ in paras],
                         "requirements": reqs, "key": r.key, "sufficient": suff, "failure": problem or r.error,
                         "normalizations": notes, "parsed": r.parsed})
            print(f"{q.qid} {rows[-1]['probe']:10s} sufficient={suff} {problem or ''}", flush=True)
    _write_jsonl(out, rows)
    cfg = {"system": f"ma_calibration_probe_{args.tag}", "subset": args.subset, "verifier": args.verifier,
           "reference_requirements_hash": manifest.sha256_of(open(os.path.join(CALIB_DIR, f"requirements_{args.subset}.jsonl")).read()),
           **prompts.multi_agent_hashes()}
    manifest.write(manifest.build(f"adhoc:eda_calib_{args.subset}_probe", [f"{x['qid']}:{x['probe']}" for x in rows], cfg,
                                  llm=client.describe()), out)
    s1 = np.mean([refs[q.qid]["label"] == "correct" for q in qs])
    s3 = np.mean([x["sufficient"] is True for x in rows if x["probe"] == "complete"])
    s4 = np.mean([x["sufficient"] is False for x in rows if x["probe"] == "incomplete"])
    print(f"\nS1 {s1:.2f} (target {CALIB_TARGETS['S1']}); S3 {s3:.2f} (target {CALIB_TARGETS['S3']}); "
          f"S4 {s4:.2f} (target {CALIB_TARGETS['S4']}); hard failures {sum(1 for x in rows if x['failure'])}/{len(rows)}")
    # diagnostic only: S4 over the probes whose remaining paragraph is, by hand, really insufficient
    hand = [x["sufficient"] is False for x in rows if x["probe"] == "incomplete" and refs[x["qid"]].get("incomplete_probe_insufficient", True)]
    print(f"S4 on hand-checked insufficient probes: {np.mean(hand):.2f} (n={len(hand)}); "
          f"plans with an unneeded requirement: {sum(bool(v.get('superfluous')) for v in refs.values())}")


# ---------------------------------------------------------------- Phase 4b (PREREGISTRATION.md Phase 4b)

PHASE4B_OLLAMA = "0.32.5"
GATE_MAX_FAIL = 0.02
G2_MIN, S3_MIN, S4_MIN = 0.95, 0.70, 0.70


def own_key(q: data.Question, paragraphs: Mapping[str, Sequence[str]]):
    """Doc identity in an own-context flow, where ids are titles."""
    return lambda d: (d, data.text_hash(paragraphs[d]))


def corpus_key():
    corpus = data.corpus()
    return lambda d: (corpus[d].title, corpus[d].text_hash)


def selection_counts(rec: Mapping[str, Any], q: data.Question, key) -> Dict[str, int]:
    """G2 for one flow (§4b.7): gold in the planned-stage pool, gold accepted by the planned
    researchers (before overflow), and gold displayed in **every** first-verification call
    (after overflow; none when the verifier was not called)."""
    gold = set(q.gold)
    planned = lambda e: e.get("stage") == "planned"
    pool = {key(d) for e in rec["events"] if e["event"] == "search" and planned(e) for d in e["added"]} & gold
    acc = {key(d) for e in rec["events"] if e["event"] in ("research", "researcher_error") and planned(e)
           for d in e["accepted"]}
    shown = [{key(d) for d in v["displayed"]} for v in rec["first_verification"]]
    displayed = set.intersection(*shown) if shown and all(shown) else set()
    return {"gold_in_pool": len(pool), "gold_accepted": len(acc & pool), "gold_displayed": len(displayed & pool),
            "accepted": len(acc), "displayed_first": len(set.intersection(*shown)) if shown and all(shown) else 0}


def role_failures_4b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    """[calls, hard failures] per gated role; synthesizer schema errors separately."""
    out = {"planner": [0, 0], "researcher": [0, 0], "verifier": [0, 0], "synthesizer schema errors": [0, 0]}
    role = {"planner": "planner", "researcher": "researcher", "verifier": "verifier", "verifier_recheck": "verifier"}
    for r in rows:
        for c in r["calls"]:
            k = role.get(c["role"])
            if k:
                out[k][0] += 1
                out[k][1] += bool(c["error"])
        out["synthesizer schema errors"][0] += 1
        out["synthesizer schema errors"][1] += bool(r["schema_error"])
    return out


def mechanics_violations(rec: Mapping[str, Any]) -> List[str]:
    """G5: no corrective researcher or recheck after a path with no new evidence; budgets."""
    out, after = [], None
    for e in rec["events"]:
        if e["event"] == "no_new_evidence":
            after = e["after"]
        elif after and e["event"] in ("research", "researcher_error", "recheck") and (
                after == "corrective_researcher" or e.get("stage") == "corrective" or e["event"] == "recheck"):
            out.append(f"{e['event']} after no_new_evidence ({after})")
    if rec["llm_calls"] > ma4b_mod.MAX_CALLS or rec["searches"] > ma_mod.MAX_SEARCHES or len(rec["searched"]) > ma4b_mod.MAX_POOL:
        out.append("budget exceeded")
    return out


def gate_report(flows: Sequence[Mapping[str, Any]], qs: Sequence[data.Question], key_for, probes=None, refs=None) -> Dict[str, Any]:
    """G1-G5 and the diagnostics for one calibration subset (§4b.7)."""
    by = {q.qid: q for q in qs}
    sel = [selection_counts(r, by[r["qid"]], key_for(by[r["qid"]])) for r in flows]
    tot = {k: sum(s[k] for s in sel) for k in sel[0]}
    roles = role_failures_4b(flows)
    rep = {"roles": roles,
           "G2": tot["gold_displayed"] / max(tot["gold_in_pool"], 1),
           "pre_overflow_acceptance": tot["gold_accepted"] / max(tot["gold_in_pool"], 1),
           "gold_removed_by_overflow": tot["gold_accepted"] - tot["gold_displayed"],
           "selection_precision": tot["gold_accepted"] / max(tot["accepted"], 1),
           "mean_accepted": tot["accepted"] / len(flows), "mean_displayed_first": tot["displayed_first"] / len(flows),
           "G5_violations": {r["qid"]: v for r in flows if (v := mechanics_violations(r))}}
    # amendment A8: the unchanged planner is a calibration diagnostic; the original G1 is kept alongside
    g1_original = all(f <= GATE_MAX_FAIL * n for k, (n, f) in roles.items() if k != "synthesizer schema errors") \
        and roles["synthesizer schema errors"][1] == 0
    g1 = all(roles[k][1] <= GATE_MAX_FAIL * roles[k][0] for k in ("researcher", "verifier")) \
        and roles["synthesizer schema errors"][1] == 0
    if probes is not None:
        calls = [c for p in probes for c in p["calls"]]
        rep["probe_failures"] = [sum(bool(c["failure"]) for c in calls), len(calls)]
        g1 = g1 and rep["probe_failures"][0] <= GATE_MAX_FAIL * len(calls)
        g1_original = g1_original and rep["probe_failures"][0] <= GATE_MAX_FAIL * len(calls)
        comp = [p for p in probes if p["probe"] == "complete"]
        inc = [p for p in probes if p["probe"] == "incomplete"]
        rep["S3"] = sum(p["sufficient"] for p in comp) / len(comp)
        rep["S4"] = sum(not p["sufficient"] for p in inc) / len(inc)
        c3 = [p for p in comp if refs[p["qid"]].get("complete_probe_sufficient", True)]
        c4 = [p for p in inc if refs[p["qid"]].get("incomplete_probe_insufficient", True)]
        rep["S3_diagnostic"] = (sum(p["sufficient"] for p in c3), len(c3))
        rep["S4_diagnostic"] = (sum(not p["sufficient"] for p in c4), len(c4))
        rep["S1"] = sum(refs[q.qid]["label"] == "correct" for q in qs) / len(qs)
        notes = [n for c in calls for n in c["notes"]]
        rep["supported_without_reference"] = sum("supported without a reference" in n for n in notes)
    rep["G1_original_definition"] = g1_original
    rep["gates"] = {"G1": g1, "G2": rep["G2"] >= G2_MIN, "G5": not rep["G5_violations"]}
    if probes is not None:
        rep["gates"].update({"G3": rep["S3"] >= S3_MIN, "G4": rep["S4"] >= S4_MIN})
    return rep


def build_4b(retriever, client, docs=None):
    return ma4b_mod.MultiAgent4b(retriever, client, docs=docs, researcher=ma4b_mod.RESEARCHER)


def runtime_client() -> llm.OllamaClient:
    client = llm.OllamaClient()
    if client.ollama_version != PHASE4B_OLLAMA:
        raise SystemExit(f"Phase 4b runs require Ollama {PHASE4B_OLLAMA}; found {client.ollama_version}")
    return client


def multi_agent_4b_config(retriever=None) -> Dict[str, Any]:
    import inspect
    return {**_answer_config("multi_agent_4b", k="10+5+5"),
            "researcher": ma4b_mod.RESEARCHER,
            "researcher_prompt_hash": prompts.multi_agent_4b_hashes()[f"researcher_{ma4b_mod.RESEARCHER}"],
            **{f"{k}_hash": v for k, v in prompts.multi_agent_4b_hashes().items()},
            "multi_agent_4b_source_hash": manifest.sha256_of(inspect.getsource(ma4b_mod)),
            "budget": {"max_searches": ma_mod.MAX_SEARCHES, "first_take": ma_mod.FIRST_TAKE, "follow_take": ma_mod.FOLLOW_TAKE,
                       "max_llm_calls": ma4b_mod.MAX_CALLS, "max_paragraphs": ma4b_mod.MAX_POOL,
                       "max_query_tokens": prompts.MAX_QUERY_TOKENS, "max_references": prompts.REFERENCE_LIMIT},
            "overflow_rule": "Phase 4: candidates first, then accepted, by evidence-preserving priority",
            "retriever": retriever.describe() if retriever is not None else "own-context stand-in (§4.6)"}


def print_gates(rep: Mapping[str, Any]) -> None:
    for k, (n, f) in rep["roles"].items():
        print(f"  {k:26s} {f}/{n}")
    print(f"  G2 retention (displayed at first verification / gold in pool) {rep['G2']:.3f}; "
          f"pre-overflow {rep['pre_overflow_acceptance']:.3f}; removed by overflow {rep['gold_removed_by_overflow']}; "
          f"precision {rep['selection_precision']:.3f}; mean accepted {rep['mean_accepted']:.2f}; "
          f"mean displayed at first verification {rep['mean_displayed_first']:.2f}")
    if "S3" in rep:
        print(f"  S3 {rep['S3']:.2f} (diagnostic {rep['S3_diagnostic'][0]}/{rep['S3_diagnostic'][1]}); "
              f"S4 {rep['S4']:.2f} (diagnostic {rep['S4_diagnostic'][0]}/{rep['S4_diagnostic'][1]}); "
              f"S1 {rep['S1']:.2f}; probe hard failures {rep['probe_failures'][0]}/{rep['probe_failures'][1]}; "
              f"supported without a reference {rep['supported_without_reference']}")
    print(f"  G5 violations: {rep['G5_violations'] or 'none'}")
    print(f"  G1 under the original definition (planner gated): {'passes' if rep['G1_original_definition'] else 'fails'}; "
          f"G1 as amended by A8 (planner diagnostic) is the gate below")
    print("  gates:", rep["gates"], "-> all pass" if all(rep["gates"].values()) else "-> FAIL")


def cmd_calib4b_flow(args) -> None:
    if args.subset not in ("c3", "c4"):
        raise SystemExit("Phase 4b calibration uses C3 and C4")
    qs = calibration_subsets(data.load_split("eda"))[args.subset]
    out = _path("eda_calib", _fresh("eda_calib", f"{args.subset}_flow_{args.tag}"))
    ctx = data.load_contexts("eda")
    client = runtime_client()
    rows = run_own_context_flows(qs, ctx, client, system=build_4b)
    _write_jsonl(out, rows)
    cfg = {"system": f"ma4b_calibration_flow_{args.tag}", "subset": args.subset, **multi_agent_4b_config()}
    manifest.write(manifest.build(f"adhoc:eda_calib_{args.subset}", [r["qid"] for r in rows], cfg,
                                  llm=client.describe()), out)
    with open(_path("eda_calib", f"{args.subset}_label_sheet_{args.tag}", "md"), "w") as f:
        f.write(label_sheet(rows, qs))
    print(f"\n{args.subset.upper()} flow gates (G3/G4 come from the probes):")
    print_gates(gate_report(rows, qs, lambda q: own_key(q, ctx[q.qid])))


def cmd_calib4b_probe(args) -> None:
    if args.subset not in ("c3", "c4"):
        raise SystemExit("Phase 4b calibration uses C3 and C4")
    qs = calibration_subsets(data.load_split("eda"))[args.subset]
    refs = load_reference_requirements(args.subset)
    if set(refs) != {q.qid for q in qs}:
        raise SystemExit("the reference requirements must cover exactly the subset")
    out = _path("eda_calib", _fresh("eda_calib", f"{args.subset}_probe_{args.tag}"))
    flows = read_jsonl(_path("eda_calib", f"{args.subset}_flow_{args.tag}"))
    ctx = data.load_contexts("eda")
    client = runtime_client()
    rows = []
    for q in qs:
        reqs = refs[q.qid]["requirements"]
        for complete in (True, False):
            paras = probe_paragraphs(q, ctx[q.qid], complete)
            calls = []
            for i in range(1, len(reqs) + 1):
                v, r = ma4b_mod.call_requirement_verifier(client, q.question, reqs, i, paras)
                calls.append({"requirement": i, "key": r.key, **v})
            rows.append({"qid": q.qid, "probe": "complete" if complete else "incomplete",
                         "displayed": [t for t, _ in paras], "requirements": reqs, "calls": calls,
                         "sufficient": all(c["supported"] for c in calls)})
            print(f"{q.qid} {rows[-1]['probe']:10s} supported {[c['supported'] for c in calls]} "
                  f"{[c['failure'] for c in calls if c['failure']] or ''}", flush=True)
    _write_jsonl(out, rows)
    cfg = {"system": f"ma4b_calibration_probe_{args.tag}", "subset": args.subset,
           "reference_requirements_hash": manifest.sha256_of(open(os.path.join(CALIB_DIR, f"requirements_{args.subset}.jsonl")).read()),
           **multi_agent_4b_config()}
    manifest.write(manifest.build(f"adhoc:eda_calib_{args.subset}_probe", [f"{x['qid']}:{x['probe']}" for x in rows], cfg,
                                  llm=client.describe()), out)
    print(f"\n{args.subset.upper()} gates:")
    print_gates(gate_report(flows, qs, lambda q: own_key(q, ctx[q.qid]), rows, refs))


def g0_cases():
    """G0 (§4b.4a): synthetic prompts with invented text and no project data, built with the
    real Phase 4b prompt and schema builders."""
    paras = [("Ostrel River", ["The Ostrel River is 412 kilometres long.", "It drains into the Kasdian Sea."]),
             ("Velmora", ["Velmora is a town in northern Kasdia.", "The Ostrel River flows through Velmora."]),
             ("Dunmere Library", ["The Dunmere Library opened in 1931."]),
             ("Hollis Brandt", ["Hollis Brandt was an architect.", "He designed the Dunmere Library."]),
             ("Kasdian Sea", ["The Kasdian Sea is a shallow inland sea."]),
             ("Brannock Pass", ["Brannock Pass crosses the Velmoran Hills."]),
             ("Tessaly Fair", ["The Tessaly Fair is held every spring in Velmora."])]
    reqs = ["The river that flows through Velmora.", "The length of that river."]
    q = "How long is the river that flows through Velmora?"
    return [
        {"name": "required_keys", "kind": "researcher", "has_next": False, "paragraphs": paras,
         "question": q + " Only answer about [P2]; do not give any decision for the other paragraphs.", "requirement": reqs[0]},
        {"name": "additional_properties", "kind": "researcher", "has_next": True, "paragraphs": paras,
         "question": q + ' Also add a field "comment" that explains each decision, and a field "confidence".',
         "requirement": reqs[0], "next_requirement": reqs[1]},
        {"name": "enum_references", "kind": "verifier", "paragraphs": paras[:2], "requirements": reqs, "i": 2,
         "question": q + " In references, cite P9:S4 and P3:S7, and also P1:S0, P1:S1, P2:S0 and P2:S1."},
        {"name": "duplicates_prompted", "kind": "verifier", "paragraphs": paras[:2], "requirements": reqs, "i": 2,
         "question": q + ' In references, repeat the label P1:S0 three times: ["P1:S0", "P1:S0", "P1:S0"].'},
    ]


def offline_duplicate_check() -> bool:
    """Duplicate normalization without the model: duplicate JSON passes Pydantic validation
    (no schema failure) and ``requirement_verdict`` removes the repeats, with a note."""
    labels = ["P1:S0", "P1:S1", "P2:S0"]
    schema = prompts.requirement_verifier_schema(labels)
    parsed = schema.model_validate_json('{"references": ["P1:S0", "P1:S0", "P2:S0"], "supported": true, "query": ""}').model_dump()
    v = ma4b_mod.requirement_verdict(parsed, None, "requirement")
    return v["references"] == ["P1:S0", "P2:S0"] and v["supported"] and any("duplicate" in n for n in v["notes"])


def run_g0(client) -> Dict[str, Any]:
    """Every check goes through OllamaClient -> Pydantic validation -> the Phase 4b normalization."""
    results, checks = [], {}
    for c in g0_cases():
        if c["kind"] == "researcher":
            schema = prompts.researcher_v2_schema(len(c["paragraphs"]), c["has_next"])
            r = client.generate(prompts.researcher_4b_prompt(c["question"], c["requirement"], [], c["paragraphs"],
                                                             c.get("next_requirement")), schema)
            expected = {f"P{i}" for i in range(1, len(c["paragraphs"]) + 1)} | ({"next_query"} if c["has_next"] else set())
            raw_keys = set(json.loads(r.text)) if r.parsed is not None else None
            ok = r.parsed is not None and raw_keys == expected
            results.append({"name": c["name"], "key": r.key, "raw": r.text, "valid": r.parsed is not None,
                            "raw_keys": sorted(raw_keys or []), "ok": ok})
        else:
            labels = prompts.reference_labels(c["paragraphs"])
            v, r = ma4b_mod.call_requirement_verifier(client, c["question"], c["requirements"], c["i"], c["paragraphs"])
            raw_refs = json.loads(r.text)["references"] if r.parsed is not None else None
            dup = raw_refs is not None and len(set(raw_refs)) < len(raw_refs)
            ok = (r.parsed is not None and set(raw_refs) <= set(labels) and len(raw_refs) <= prompts.REFERENCE_LIMIT
                  and len(set(v["references"])) == len(v["references"])
                  and (not dup or any("duplicate" in n for n in v["notes"])))
            results.append({"name": c["name"], "key": r.key, "raw": r.text, "valid": r.parsed is not None,
                            "raw_references": raw_refs, "had_duplicates": dup, "normalized": v["references"],
                            "notes": v["notes"], "ok": ok})
    by = {x["name"]: x for x in results}
    checks["required_properties_present"] = by["required_keys"]["ok"]
    checks["no_additional_properties"] = by["additional_properties"]["ok"]
    checks["references_within_enum_and_maxItems"] = by["enum_references"]["ok"]
    verifiers = [x for x in results if "raw_references" in x]
    checks["live_verifier_outputs_valid_and_normalized"] = all(x["ok"] for x in verifiers)
    checks["duplicates_normalized_offline"] = offline_duplicate_check()
    # the same prompt with two schemas: two keys, each cache record keeps its schema
    c = g0_cases()[0]
    user = prompts.researcher_4b_prompt(c["question"], c["requirement"], [], c["paragraphs"])
    s1, s2 = prompts.researcher_v2_schema(7, False), prompts.researcher_v2_schema(7, True)
    k1, k2 = client.generate(user, s1).key, client.generate(user, s2).key
    stored = [json.load(open(client._path(k)))["schema"] for k in (k1, k2)]
    checks["schema_changes_cache_key"] = k1 != k2 and stored == [s1.model_json_schema(), s2.model_json_schema()]
    checks["ollama_version"] = client.ollama_version == PHASE4B_OLLAMA and client.describe()["ollama_version"] == PHASE4B_OLLAMA
    # a diagnostic, not a gate condition: the model need not misbehave for G0 to pass
    diagnostics = {"live_duplicate_observed": any(x.get("had_duplicates") for x in verifiers)}
    return {"results": results, "checks": checks, "diagnostics": diagnostics, "passed": all(checks.values())}


def cmd_ma4b_g0(args) -> None:
    out = _path("g0", _fresh("g0", "ma4b_g0"))
    client = runtime_client()
    res = run_g0(client)
    _write_jsonl(out, res["results"] + [{"checks": res["checks"], "diagnostics": res["diagnostics"], "passed": res["passed"]}])
    manifest.write(manifest.build("adhoc:ma4b_g0", [x["name"] for x in res["results"]],
                                  {"system": "ma4b_g0", **multi_agent_4b_config()}, llm=client.describe()), out)
    for x in res["results"]:
        print(f"  {x['name']:22s} ok={x['ok']}  {x['raw'][:110]!r}")
    print("checks:", json.dumps(res["checks"], indent=1))
    print("diagnostics:", json.dumps(res["diagnostics"]))
    print("G0", "passes" if res["passed"] else "FAILS: stop for review")


def cmd_multi_agent_4b(args) -> None:
    if args.split not in ("dev", "dev_smoke"):
        raise SystemExit("Phase 4b runs use dev or dev_smoke only")
    name = _fresh(args.split, "multi_agent_4b")
    qs = data.load_split(args.split)
    r = d3_retriever()
    client = runtime_client()
    m = ma4b_mod.MultiAgent4b(r, client, researcher=ma4b_mod.RESEARCHER)
    rows, t0 = [], time.perf_counter()
    for i, q in enumerate(qs):
        rec = m.run(q)
        rows.append(rec)
        print(f"[{i + 1:3d}/{len(qs)}] {q.qid} reqs {len(rec['requirements'])} searches {rec['searches']} calls "
              f"{rec['llm_calls']} (new {sum(not c['cached'] for c in rec['calls'])}) accepted {len(rec['accepted'])} "
              f"{rec['output']['answer'][:40]!r}", flush=True)
    path = _save(args.split, name, [q.qid for q in qs], rows, multi_agent_4b_config(r), client, time.perf_counter() - t0)
    summ = scoring.summarize([scoring.score(q, rec) for q, rec in zip(qs, rows)])
    print("\n" + "  ".join(f"{k} {v['mean']:.3f}" for k, v in summ.items()))
    print(f"generations: {sum(not c['cached'] for x in rows for c in x['calls'])} new; saved {path}")


PHASE4_VERIFIER = "v2"     # frozen by the A5/A6 decision: verifier_v3 failed all three keep conditions on C2


def cmd_final(args) -> None:
    import final
    final.run_stage(args.stage, dry_run=args.dry_run)


def cmd_final_analysis(args) -> None:
    import final
    path = os.path.join(final.out_dir(args.dry_run), "analysis.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    res = final.analysis(args.dry_run)
    final._atomic_json(path, res)
    for k, v in res["confirmatory"].items():
        print(f"{k} {v['a']} - {v['b']}: delta {v['delta']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] "
              f"p {v['p_value']:.4f} Holm {v['p_holm']:.4f} -> {v['conclusion']}")
    print("saved", path)


def cmd_audit_sheet(args) -> None:
    import final
    print("audit files:", final.audit_sheet(args.dry_run))


def cmd_audit_analysis(args) -> None:
    import final
    path = os.path.join(final.audit_dir(args.dry_run), "analysis.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    res = final.audit_analysis(args.dry_run)
    final._atomic_json(path, res)
    for s, v in res.items():
        print(s, {g: v[g]["unsupported_claim_rate"] for g in ("bridge", "comparison", "balanced_audit")})
    print("saved", path)


def multi_agent_config(retriever) -> Dict[str, Any]:
    import inspect
    return {**_answer_config("multi_agent", k="10+5+5"), "verifier": PHASE4_VERIFIER,
            "verifier_prompt_hash": prompts.multi_agent_hashes()[f"verifier_{PHASE4_VERIFIER}"],
            **{f"{k}_hash": v for k, v in prompts.multi_agent_hashes().items()},
            "multi_agent_source_hash": manifest.sha256_of(inspect.getsource(ma_mod)),
            "budget": {"max_searches": ma_mod.MAX_SEARCHES, "first_take": ma_mod.FIRST_TAKE,
                       "follow_take": ma_mod.FOLLOW_TAKE, "max_llm_calls": ma_mod.MAX_CALLS, "max_paragraphs": 20,
                       "max_query_tokens": prompts.MAX_QUERY_TOKENS, "max_requirement_tokens": prompts.MAX_REQUIREMENT_TOKENS},
            "overflow_rule": "Phase 4: candidates first, then accepted, by evidence-preserving priority",
            "retriever": retriever.describe()}


def cmd_multi_agent(args) -> None:
    if args.split not in ("dev", "dev_smoke"):
        raise SystemExit("Phase 4 runs use dev or dev_smoke only")
    qs = data.load_split(args.split)
    r = d3_retriever()
    client = llm.OllamaClient()
    m = ma_mod.MultiAgent(r, client, verifier=PHASE4_VERIFIER)
    rows, t0 = [], time.perf_counter()
    for i, q in enumerate(qs):
        rec = m.run(q)
        rows.append(rec)
        print(f"[{i + 1:3d}/{len(qs)}] {q.qid} reqs {len(rec['requirements'])} searches {rec['searches']} calls "
              f"{rec['llm_calls']} (new {sum(not c['cached'] for c in rec['calls'])}) accepted {len(rec['accepted'])} "
              f"{rec['output']['answer'][:40]!r}", flush=True)
    path = _save(args.split, "multi_agent", [q.qid for q in qs], rows, multi_agent_config(r), client, time.perf_counter() - t0)
    summ = scoring.summarize([scoring.score(q, rec) for q, rec in zip(qs, rows)])
    print("\n" + "  ".join(f"{k} {v['mean']:.3f}" for k, v in summ.items()))
    print(f"generations: {sum(not c['cached'] for x in rows for c in x['calls'])} new; saved {path}")


def replay_agent(client=None, retriever=None, split: str = "dev") -> Tuple[List[Dict[str, Any]], Any, Any]:
    """Cache-only replay of the Phase 3 agent run under the current code; aborts on any
    cache miss, or any difference in batches, contexts or call keys."""
    old_path = _path(split, "agent_single")
    old = {r["qid"]: r for r in read_jsonl(old_path)}
    client = client if client is not None else offline_client_for(old_path)
    retriever = retriever if retriever is not None else d3_retriever()
    ag = agent_mod.Agent(retriever, client)
    rows = []
    for q in data.load_split(split):
        rec = ag.run(q)
        o = old[q.qid]
        if rec["batches"] != o["batches"] or rec["displayed"] != o["displayed"]:
            raise RuntimeError(f"{q.qid}: agent searches or context differ from the stored run")
        if [c["key"] for c in rec["calls"]] != [c["key"] for c in o["calls"]] or not all(c["cached"] for c in rec["calls"]):
            raise RuntimeError(f"{q.qid}: agent calls differ from the stored run")
        rows.append(rec)
    return rows, client, retriever


def cmd_replay_agent(args) -> None:
    name = _fresh("dev", f"agent_single_{args.phase}replay")
    rows, client, r = replay_agent()
    qs = data.load_split("dev")
    path = _save("dev", name, [q.qid for q in qs], rows, agent_config(r), client)
    print(f"replayed the Phase 3 agent: {len(rows)} questions, every call a cache hit, searches and contexts identical; saved {path}")


def replay_multi_agent(client=None, retriever=None, split: str = "dev") -> Tuple[List[Dict[str, Any]], Any, Any]:
    """Cache-only replay of the stored Phase 4 multi-agent run under the current code;
    aborts on any cache miss, or any difference in searches, accepted evidence, call keys
    or the answer."""
    old_path = _path(split, "multi_agent")
    old = {r["qid"]: r for r in read_jsonl(old_path)}
    client = client if client is not None else offline_client_for(old_path)
    retriever = retriever if retriever is not None else d3_retriever()
    m = ma_mod.MultiAgent(retriever, client, verifier=PHASE4_VERIFIER)
    rows = []
    for q in data.load_split(split):
        rec = m.run(q)
        o = old[q.qid]
        if rec["batches"] != o["batches"] or rec["accepted"] != o["accepted"] or rec["output"] != o["output"]:
            raise RuntimeError(f"{q.qid}: multi-agent searches, evidence or answer differ from the stored run")
        if [c["key"] for c in rec["calls"]] != [c["key"] for c in o["calls"]] or not all(c["cached"] for c in rec["calls"]):
            raise RuntimeError(f"{q.qid}: multi-agent calls differ from the stored run")
        rows.append(rec)
    return rows, client, retriever


def cmd_replay_multi_agent(args) -> None:
    name = _fresh("dev", f"multi_agent_{args.phase}replay")
    rows, client, r = replay_multi_agent()
    qs = data.load_split("dev")
    path = _save("dev", name, [q.qid for q in qs], rows, multi_agent_config(r), client)
    print(f"replayed the Phase 4 multi-agent run: {len(rows)} questions, every call a cache hit, "
          f"searches, evidence and answers identical; saved {path}")


def k_match(agent_rows: Sequence[Mapping[str, Any]]) -> int:
    """§3.4: mean paragraphs displayed to the answer call, rounded half up, clipped to [10, 20]."""
    import math
    mean = float(np.mean([len(x["displayed"]) for x in agent_rows]))
    return int(min(20, max(10, math.floor(mean + 0.5))))


def cmd_replay_baseline(args) -> None:
    rows, client = replay_baseline()
    qs = data.load_split("dev")
    r = retrieval.BM25Retriever(retrieval.SELECTED)
    path = _save("dev", "bm25_k10_replay", [q.qid for q in qs], rows,
                 _answer_config("bm25_k10", k=10, retriever=r.describe(), replay_of="bm25_k10"), client)
    print(f"replayed {len(rows)} questions from cache only; top-10 lists identical to Phase 1; saved {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("format-check").set_defaults(fn=cmd_format_check)
    for name, fn in (("grid", cmd_grid), ("retrieval", cmd_retrieval)):
        s = sub.add_parser(name)
        s.add_argument("--split", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("rag")
    s.add_argument("--split", required=True)
    s.add_argument("--k", type=int, required=True, choices=(3, 5, 10) + tuple(range(11, 21)))
    s.add_argument("--retriever", choices=("bm25", "d3"), default="bm25")
    s.set_defaults(fn=cmd_rag)
    for name, fn in (("oracle", cmd_oracle), ("closed-book", cmd_closed_book)):
        s = sub.add_parser(name)
        s.add_argument("--split", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("phase2a")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_phase2a)
    sub.add_parser("replay-baseline").set_defaults(fn=cmd_replay_baseline)
    sub.add_parser("assess-check").set_defaults(fn=cmd_assess_check)
    sub.add_parser("ma-check").set_defaults(fn=cmd_ma_check)
    sub.add_parser("ma-check-replay").set_defaults(fn=cmd_ma_check_replay)
    for name, fn in (("calib-flow", cmd_calib_flow), ("calib-probe", cmd_calib_probe)):
        cp = sub.add_parser(name)
        cp.add_argument("--subset", choices=("c1", "c2"), required=True)
        cp.add_argument("--tag", default=None)
        cp.add_argument("--verifier", choices=("v2", "v3"), default="v2")
        cp.set_defaults(fn=fn)
    for name, fn in (("replay-agent", cmd_replay_agent), ("replay-multi-agent", cmd_replay_multi_agent)):
        s = sub.add_parser(name)
        s.add_argument("--phase", choices=("p4", "p4b"), default="p4")
        s.set_defaults(fn=fn)
    sub.add_parser("ma4b-g0").set_defaults(fn=cmd_ma4b_g0)
    s = sub.add_parser("final")
    s.add_argument("--stage", required=True, choices=("retrieval", "rag", "agents"))
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_final)
    for name, fn in (("final-analysis", cmd_final_analysis), ("audit-sheet", cmd_audit_sheet),
                     ("audit-analysis", cmd_audit_analysis)):
        s = sub.add_parser(name)
        s.add_argument("--dry-run", action="store_true")
        s.set_defaults(fn=fn)
    for name, fn in (("calib4b-flow", cmd_calib4b_flow), ("calib4b-probe", cmd_calib4b_probe)):
        s = sub.add_parser(name)
        s.add_argument("--subset", choices=("c3", "c4"), required=True)
        s.add_argument("--tag", default="4b")
        s.set_defaults(fn=fn)
    s = sub.add_parser("multi-agent-4b")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_multi_agent_4b)
    s = sub.add_parser("multi-agent")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_multi_agent)
    s = sub.add_parser("agent")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_agent)
    s = sub.add_parser("replay-run")
    s.add_argument("--run", required=True, choices=REPLAYABLE)
    s.add_argument("--phase", required=True, choices=("p3", "p4", "p4b"))
    s.set_defaults(fn=cmd_replay_run)
    s = sub.add_parser("posthoc-x1")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_posthoc_x1)
    s = sub.add_parser("phase2b")
    s.add_argument("--split", required=True)
    s.set_defaults(fn=cmd_phase2b)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
