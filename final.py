"""Phase 5: the staged, immutable final runner, its analysis and the blinded audit
(PREREGISTRATION.md §5).

    python run_eval.py final --stage retrieval      # test_retrieval: BM25 C0 and D3 rankings (no LLM)
    python run_eval.py final --stage rag            # test_rag: baseline, R, closed-book, oracle
    python run_eval.py final --stage agents         # test_agents: all seven systems
    python run_eval.py final-analysis               # §5.5-5.6, from validated stored runs only
    python run_eval.py audit-sheet                  # §5.7: the blinded sheet, key and label template
    python run_eval.py audit-analysis               # §5.7: validated labels joined with the key

**The freeze declaration.** The approved evaluation-code, runner and retriever hashes live
in a tracked declaration in PREREGISTRATION.md (§A, "Phase 5 record — freeze
declaration"). Before any blind split is read, the current hashes must equal it; this
module holds no constant describing itself.

**Validation.** A stored result counts only after ``validate_system``: result and
manifest exist; exact, ordered split membership; no missing or duplicate records; the
freeze commit and evaluation hash; the expected system configuration; and the model and
runtime invariants for LLM runs. Stages are validated, file hashes included, before a
dependent stage starts and before analysis or audit generation.

**Immutability.** A system is complete when its manifest exists (renamed into place
last) and validates; it is never rerun or overwritten. A partial run resumes only when
its commit and configuration match exactly. The stage marker is written last. Each stage
is a separate invocation. ``--dry-run`` runs the same code on ``dev`` with an offline,
cache-only client; no blind split is loaded and nothing is generated.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import inspect
import json
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

import agent as agent_mod
import data
import llm
import manifest
import metrics
import multi_agent as ma_mod
import prompts
import rag
import retrieval
import scoring

# ---------------------------------------------------------------- the frozen systems (§5.1-5.3)

OLLAMA_VERSION = "0.32.5"
MODEL = "qwen2.5:7b"
MODEL_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
QUANTIZATION = "Q4_K_M"
K_MATCH = 16
WEIGHTS = {"bridge": 0.799, "comparison": 0.201}
AUDIT_SEED, AUDIT_ORDER_SEED, AUDIT_PER_TYPE = "audit-20260927:", "audit-order-20260927:", 12
AUDIT_SYSTEMS = ("bm25_k10", "d3_k10", "agent_single", "multi_agent")
SUPPORT_LABELS = ("cannot_determine", "contradicted", "fully_supported", "partially_supported", "unsupported")
CITATION_LABELS = ("valid", "invalid")
FAMILY = (("F1", "d3_k10", "bm25_k10"), ("F2", "agent_single", "d3_k10"), ("F3", "multi_agent", "agent_single"))
SECONDARY = (("matched_minus_single", "d3_k16", "agent_single"), ("matched_minus_multi", "d3_k16", "multi_agent"),
             ("multi_minus_R", "multi_agent", "d3_k10"))
MPI = 0.03

FROZEN = {
    "split_hashes": {"test_retrieval": "8654f35c1493a822245eeb07208c95999f900e59dc6adff01a5e43167a0a28ef",
                     "test_rag": "57201688a97654e73d633128498c00ff92c1f0e6984aa296b5ad506e6ab83d8d",
                     "test_agents": "6b72130cb1f09e7a21129f81d50876525c1306f4df426777f3c11e1ce470f447"},
    "corpus": "67db29c2a555974d0d16e8a3bb9c9865ee8c542d6d268ed88e0232e340fee3aa",
    "answer_interface": {"answer_v2": "0a2aa59c597d9730ed73ea358b22cd76787cde9910e12888397076b7bb003f72",
                         "schema_v2": "c29af85a08782e161cae0ab7e4eb847e19866b4bc7910d84640c30d12ab67847",
                         "answer_validator_v2": "252c82c0f31ab1013af85beeaa777b8eab0a5ffa2ba4dadec1e9961eb2d4910c",
                         "system_prompt": "16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172",
                         "render_v1": "935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6",
                         "closed_book_v2": "a89deeb1d5aafd0fccafbcb44a10e47067f610bef75dcf8bcbfa878e9cdbcb79"},
    "assess": {"assess_v2": "2221feb7f911d1cdc90150f930fab21abf0eddd63cf5fdf05d5ebc2048a17724",
               "assess_schema_v1": "c84e600907056092f6a7bc0ed85c0ff78790e5b76c72f6d8676456f7cc849915",
               "assess_contract_v1": "db8c967d1390891200e1e2afe2dd8da4536993b2f185bb7d59803f5d459a4a95"},
    "multi_agent": {"planner_v1": "e6e453e16f09adb4a0d2923c55f21f0f819798b683da7e9dbce9262a86f0b787",
                    "researcher_v1": "d9c25b49275969703649f8069a06b67eab3d4565edfce58f81c948dfe64af182",
                    "next_query_rule_with": "0ba8f669319d4af0ee2aabec223ee97c2af9367c8d25bc81a2728b98cb5e2124",
                    "next_query_rule_none": "27f512c88209cdcc733f137d55b43f13fef8153f3998b117d04ccfbb0c57804a",
                    "verifier_v2": "dfa54a0f4e5430fe4d82319d1743e6e4fe32ba5a8ab3ff51112fa7b3b1e56c21",
                    "planner_schema_v1": "53e7097fbb9ee9716b7c0963f6d0908ee68056ff6b90c99113e5f77440c2efc5",
                    "researcher_schema_v1": "eb7b7514e5d053a184f5bbd4be90733cf59f8e889d710a37b391fce6ee8e3352",
                    "verifier_schema_v1": "4a6e760e8b45672e84a86fd71c213e424545fa9d99b0657128bf605ba4af0e4a",
                    "planner_contract_v1": "36b3f5f59ae939be2b306beec7d33a1d52c96c772066a8c9c743e0a26d236760",
                    "researcher_contract_v1": "cab5565d9755ab25f2f6e73f1982e691958c9c46bc66350bef9473e88396bae2",
                    "verifier_contract_v2": "7a95d66fb32f648d1203d3b70ea9795e6666fec1a092b34cabc0f481241ec097",
                    "render_labeled": "8f6d7af4475a2278571897d3711e20800fea6b2dea84a6af317ca80354d9c937"},
    "sources": {"agent.py": "5e7b844c2416450f5903b31805b655c5b4e98fdf8dfd82c1e5619a8f562a7cdd",
                "multi_agent.py": "1c4b90a8bfa459357de435454c5b0a9d1cc1880c00ac726a38869f013c184d89"},
    "embedding_fingerprint": "5411b3a2ec46d23446ecf0d44f27de6665282f88e3d040eb88e82ea9d18a7d4c",
    "graph_fingerprint": "c0a4b543a343a2bcd9fc2d73a9b345fbb356aeed3a5cf9ac284c4ad9d0a691f3",
    "bm25": {"stem": True, "remove_stopwords": True, "k1": 1.5, "b": 0.75, "normalize": False, "title_repeats": 1},
    "phase4_verifier": "v2",
}


def frozen_mismatches() -> List[str]:
    """Every frozen system identity of §5.2-5.3 that differs now (empty: all match)."""
    import run_eval
    out = []

    def cmp(name, now, then):
        if now != then:
            out.append(name)
    cmp("split_hashes", {k: data.EXPECTED_SPLIT_HASHES[k] for k in FROZEN["split_hashes"]}, FROZEN["split_hashes"])
    cmp("corpus", data.verify_corpus(), FROZEN["corpus"])
    cmp("answer_interface", {k: prompts.artifact_hashes()[k] for k in FROZEN["answer_interface"]}, FROZEN["answer_interface"])
    cmp("assess", prompts.assess_hashes(), FROZEN["assess"])
    cmp("multi_agent", {k: prompts.multi_agent_hashes()[k] for k in FROZEN["multi_agent"]}, FROZEN["multi_agent"])
    cmp("agent.py", manifest.sha256_of(inspect.getsource(agent_mod)), FROZEN["sources"]["agent.py"])
    cmp("multi_agent.py", manifest.sha256_of(inspect.getsource(ma_mod)), FROZEN["sources"]["multi_agent.py"])
    cmp("bm25", {k: getattr(retrieval.SELECTED, k) for k in FROZEN["bm25"]}, FROZEN["bm25"])
    with open(run_eval._path("tune", "phase2b_selection", "json")) as f:
        cmp("embedding_fingerprint", json.load(f)["embedding"]["fingerprint"], FROZEN["embedding_fingerprint"])
    cmp("phase4_verifier", run_eval.PHASE4_VERIFIER, FROZEN["phase4_verifier"])
    return out


def check_runtime(client) -> None:
    """The frozen structured-decoding runtime and model (§5.1)."""
    if client.ollama_version != OLLAMA_VERSION or client.digest != MODEL_DIGEST:
        raise SystemExit(f"Phase 5 requires Ollama {OLLAMA_VERSION} with digest {MODEL_DIGEST}; "
                         f"found {client.ollama_version} / {client.digest}")


# ---------------------------------------------------------------- the freeze declaration (§5.8)

DECLARATION_HEADING = "### Phase 5 record — freeze declaration"
DECLARED_KEYS = ("eval_code_hash", "runner_source_hash", "bm25_describe_hash", "d3_describe_hash")


def declared_freeze(path: Optional[str] = None) -> Optional[Dict[str, str]]:
    """The approved hashes from the tracked declaration in PREREGISTRATION.md, or None."""
    path = path or os.path.join(data.HERE, "PREREGISTRATION.md")
    with open(path) as f:
        text = f.read()
    if DECLARATION_HEADING not in text:
        return None
    block = re.search(r"```text\n(.*?)\n```", text[text.index(DECLARATION_HEADING):], re.S)
    if block is None:
        raise SystemExit("the freeze declaration has no text block")
    out = dict(line.split(": ", 1) for line in block.group(1).splitlines() if line.strip())
    if set(out) != set(DECLARED_KEYS):
        raise SystemExit(f"the freeze declaration must list exactly {DECLARED_KEYS}")
    return out


def runner_source_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def current_hashes(bm25_describe: Mapping[str, Any], d3_describe: Mapping[str, Any]) -> Dict[str, str]:
    return {"eval_code_hash": manifest.eval_code_hash(), "runner_source_hash": runner_source_hash(),
            "bm25_describe_hash": manifest.sha256_of(dict(bm25_describe)),
            "d3_describe_hash": manifest.sha256_of(dict(d3_describe))}


_FROM_FILE = object()


def check_declaration(hashes: Mapping[str, str], dry_run: bool, declared: Any = _FROM_FILE) -> Optional[str]:
    """Real stages: the current hashes must equal the approved declaration (read from
    PREREGISTRATION.md unless given; None means there is none). A dry run only reports."""
    declared = declared_freeze() if declared is _FROM_FILE else declared
    if declared == dict(hashes):
        return None
    msg = ("no approved freeze declaration in PREREGISTRATION.md" if declared is None
           else f"current hashes differ from the declaration: "
                f"{[k for k in DECLARED_KEYS if declared.get(k) != hashes.get(k)]}")
    if not dry_run:
        raise SystemExit(msg)
    return msg


# ---------------------------------------------------------------- stages, systems, configurations (§5.4)

STAGES = {"retrieval": "test_retrieval", "rag": "test_rag", "agents": "test_agents"}
ORDER = ("retrieval", "rag", "agents")
STAGE_SYSTEMS = {"retrieval": ("bm25_rankings", "d3_rankings"),
                 "rag": ("bm25_k10", "d3_k10", "closed_book", "oracle"),
                 "agents": ("bm25_k10", "d3_k10", "d3_k16", "closed_book", "oracle", "agent_single", "multi_agent")}
NO_LLM = ("bm25_rankings", "d3_rankings")


def out_dir(dry_run: bool) -> str:
    import run_eval
    return os.path.join(run_eval.RESULTS, "final_dryrun" if dry_run else "final")


def stage_split(stage: str, dry_run: bool) -> str:
    return "dev" if dry_run else STAGES[stage]


def load_questions(stage: str, dry_run: bool) -> List[data.Question]:
    """The only place the blind splits are loaded; a dry run never loads them."""
    if dry_run:
        return data.load_split("dev")
    return data.load_split(STAGES[stage], unblind=True)


def marker_path(stage: str, dry_run: bool) -> str:
    return os.path.join(out_dir(dry_run), f"{stage}.done.json")


def result_path(stage: str, name: str, dry_run: bool) -> str:
    return os.path.join(out_dir(dry_run), f"{stage_split(stage, dry_run)}__{name}.jsonl")


class _Described:
    """A stand-in exposing a recorded ``describe()`` (for recomputing expected configurations)."""

    def __init__(self, desc):
        self._d = dict(desc)

    def describe(self):
        return dict(self._d)


def configs(stage: str, bm25_describe: Mapping[str, Any], d3_describe: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The expected system configuration of every system in a stage."""
    import run_eval
    bm25, d3 = _Described(bm25_describe), _Described(d3_describe)
    rag_cfg = lambda r, k, name: run_eval._answer_config(name, k=k, retriever=r.describe())
    table = {
        "bm25_rankings": lambda: {"system": "bm25_rankings", **bm25.describe(), "depth": 100},
        "d3_rankings": lambda: {"system": "d3_rankings", **d3.describe(), "depth": 100},
        "bm25_k10": lambda: rag_cfg(bm25, 10, "bm25_k10"),
        "d3_k10": lambda: rag_cfg(d3, 10, "d3_k10"),
        "d3_k16": lambda: rag_cfg(d3, K_MATCH, "d3_k16"),
        "closed_book": run_eval.closed_book_config,
        "oracle": lambda: run_eval._answer_config("oracle", context="two gold paragraphs, supplied order"),
        "agent_single": lambda: run_eval.agent_config(d3),
        "multi_agent": lambda: run_eval.multi_agent_config(d3),
    }
    return {name: table[name]() for name in STAGE_SYSTEMS[stage]}


def functions(stage: str, d3, bm25) -> Dict[str, Callable]:
    """name -> function(client, q) for one stage (§5.3)."""
    import run_eval

    def rankings(r):
        def fn(_c, q):
            t0 = time.perf_counter()
            hits = r.search(q.question, k=100)
            return {"qid": q.qid, "type": q.type, "latency_s": time.perf_counter() - t0, "ranked": [d for d, _ in hits],
                    **scoring.retrieval_scores(q, [d for d, _ in hits], run_eval.CURVE_KS)}
        return fn
    table = {
        "bm25_rankings": rankings(bm25), "d3_rankings": rankings(d3),
        "bm25_k10": lambda c, q: rag.bm25_rag(c, bm25, q, 10),
        "d3_k10": lambda c, q: rag.bm25_rag(c, d3, q, 10),
        "d3_k16": lambda c, q: rag.bm25_rag(c, d3, q, K_MATCH),
        "closed_book": rag.closed_book, "oracle": rag.oracle,
        "agent_single": lambda c, q: agent_mod.Agent(d3, c).run(q),
        "multi_agent": lambda c, q: ma_mod.MultiAgent(d3, c, verifier=FROZEN["phase4_verifier"]).run(q),
    }
    return {name: table[name] for name in STAGE_SYSTEMS[stage]}


# ---------------------------------------------------------------- atomic files and validation

def _atomic_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _canon(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True))


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validation_problems(path: str, split: str, qids: Sequence[str], config: Mapping[str, Any], llm_run: bool,
                        freeze: Mapping[str, Any]) -> List[str]:
    """Everything that makes a stored result unacceptable (empty: it validates)."""
    man_path = path + ".manifest.json"
    if not os.path.exists(path) or not os.path.exists(man_path):
        return ["result or manifest missing"]
    out = []
    rows = _load(path)
    got = [r.get("qid") for r in rows]
    if len(set(got)) != len(got):
        out.append("duplicate records")
    if got != list(qids):
        out.append("records differ from the ordered split membership")
    with open(man_path) as f:
        m = json.load(f)
    inv, sc = m["comparison_invariants"], m["system_config"]
    if inv["split"] != split or inv["split_hash"] != data.ids_hash(list(qids)) or inv["n_questions"] != len(qids):
        out.append("manifest split")
    if inv["code"] != {"commit": freeze["commit"], "dirty": False}:
        out.append("manifest commit")
    if inv["eval_code_hash"] != freeze["eval_code_hash"]:
        out.append("manifest evaluation hash")
    if {k: v for k, v in sc.items() if k != "llm_options"} != _canon(config):
        out.append("system configuration")
    if llm_run:
        if inv["base_llm"] is None or (inv["base_llm"]["model"], inv["base_llm"]["digest"], inv["base_llm"]["quantization"]) \
                != (MODEL, MODEL_DIGEST, QUANTIZATION):
            out.append("model identity")
        if inv.get("ollama_version") != OLLAMA_VERSION or inv["num_ctx"] != llm.DEFAULT_OPTIONS["num_ctx"]:
            out.append("runtime")
        if sc.get("llm_options") != _canon(llm.DEFAULT_OPTIONS):
            out.append("LLM options")
    elif inv["base_llm"] is not None:
        out.append("a retrieval-only run records an LLM")
    return out


def validate_system(*args, **kw) -> None:
    problems = validation_problems(*args, **kw)
    if problems:
        raise SystemExit(f"{args[0]}: stored result does not validate: {problems}")


def read_freeze(dry_run: bool) -> Dict[str, Any]:
    path = os.path.join(out_dir(dry_run), "freeze.json")
    if not os.path.exists(path):
        raise SystemExit("no freeze record: the first stage writes it")
    with open(path) as f:
        fr = json.load(f)
    if (manifest.sha256_of(fr["bm25_describe"]) != fr["bm25_describe_hash"]
            or manifest.sha256_of(fr["d3_describe"]) != fr["d3_describe_hash"]):
        raise SystemExit("the freeze record's describe dictionaries do not match their hashes")
    if not dry_run:
        check_declaration({k: fr[k] for k in DECLARED_KEYS}, dry_run=False)
    return fr


def check_consumer(dry_run: bool) -> Dict[str, Any]:
    """Before analysis or the audit reads anything: the code now executing must be the
    frozen code. The current evaluation-code and runner hashes equal the freeze record, the
    tree is clean at the freeze commit, and every frozen identity still matches."""
    fr = read_freeze(dry_run)
    now = manifest.code_commit()
    problems = []
    if manifest.eval_code_hash() != fr["eval_code_hash"]:
        problems.append("evaluation code differs from the freeze")
    if runner_source_hash() != fr["runner_source_hash"]:
        problems.append("runner differs from the freeze")
    if now != {"commit": fr["commit"], "dirty": False}:
        problems.append("not a clean tree at the freeze commit")
    bad = frozen_mismatches()
    if bad:
        problems.append(f"frozen identities differ: {bad}")
    if problems:
        raise SystemExit(f"refusing to analyse: {problems}")
    return fr


def validate_stage(stage: str, dry_run: bool, qids: Optional[Sequence[str]] = None) -> None:
    """A completed stage: marker, recomputed file hashes, and every system validated."""
    mp = marker_path(stage, dry_run)
    if not os.path.exists(mp):
        raise SystemExit(f"stage {stage!r} is not complete")
    fr = read_freeze(dry_run)
    with open(mp) as f:
        mk = json.load(f)
    if mk["commit"] != fr["commit"] or mk["stage"] != stage or mk["split"] != stage_split(stage, dry_run):
        raise SystemExit(f"stage {stage!r}: the marker does not match the freeze")
    qids = list(qids) if qids is not None else [q.qid for q in load_questions(stage, dry_run)]
    expected = configs(stage, fr["bm25_describe"], fr["d3_describe"])
    if set(mk["systems"]) != set(expected):
        raise SystemExit(f"stage {stage!r}: the marker lists other systems")
    for name, cfg in expected.items():
        path = result_path(stage, name, dry_run)
        rec = mk["systems"][name]
        if not os.path.exists(path) or _sha_file(path) != rec["result_sha256"] \
                or _sha_file(path + ".manifest.json") != rec["manifest_sha256"] or rec["config_hash"] != manifest.sha256_of(cfg):
            raise SystemExit(f"stage {stage!r}/{name}: files changed since the stage completed")
        validate_system(path, stage_split(stage, dry_run), qids, cfg, name not in NO_LLM, fr)


# ---------------------------------------------------------------- running

def run_system(path: str, split: str, qs: Sequence[data.Question], fn: Callable, config: Mapping[str, Any],
               client=None, commit: Optional[Mapping[str, Any]] = None, progress=print) -> str:
    """Run one system with append-only resumption and atomic finalization. Returns
    "complete" when a manifest already exists (the caller validates it) or "finished"."""
    if os.path.exists(path + ".manifest.json"):
        return "complete"
    commit = dict(commit) if commit is not None else manifest.code_commit()
    partial, meta_path = path + ".partial", path + ".partial.meta.json"
    meta = {"commit": commit["commit"], "dirty": commit["dirty"], "config_hash": manifest.sha256_of(dict(config)),
            "split": split, "split_hash": data.ids_hash([q.qid for q in qs])}
    if os.path.exists(partial):
        with open(meta_path) as f:
            if json.load(f) != meta:
                raise SystemExit(f"{partial}: the partial run's commit or configuration differs; refusing to resume")
    else:
        _atomic_json(meta_path, meta)
        open(partial, "w").close()
    done = _load(partial)
    if [r["qid"] for r in done] != [q.qid for q in qs[:len(done)]]:
        raise SystemExit(f"{partial}: stored records are not a prefix of the split order")
    t0 = time.perf_counter()
    with open(partial, "a") as f:
        for i, q in enumerate(qs[len(done):], len(done)):
            rec = fn(client, q)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            progress(f"[{i + 1:4d}/{len(qs)}] {os.path.basename(path)} {q.qid}")
    rows = _load(partial)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    m = manifest.build(split, [q.qid for q in qs], dict(config), llm=client.describe() if client is not None else None,
                       runtime_s=time.perf_counter() - t0, commit=commit)
    _atomic_json(path + ".manifest.json", m)                 # completion point
    os.remove(partial)
    os.remove(meta_path)
    return "finished"


def make_client(dry_run: bool):
    if dry_run:     # cache only: any prompt that differs from the dev runs aborts
        return llm.OllamaClient(offline=True, offline_digest=MODEL_DIGEST, offline_quantization=QUANTIZATION,
                                offline_ollama_version=OLLAMA_VERSION)
    client = llm.OllamaClient()
    check_runtime(client)
    return client


def run_stage(stage: str, dry_run: bool = False, retrievers=None, client=None) -> None:
    """One stage of §5.4 (one operator-started session)."""
    if stage not in ORDER:
        raise SystemExit(f"unknown stage {stage!r}")
    d = out_dir(dry_run)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(marker_path(stage, dry_run)):
        raise SystemExit(f"stage {stage!r} is complete; completed outputs are immutable")
    i = ORDER.index(stage)
    if i > 0 and not os.path.exists(marker_path(ORDER[i - 1], dry_run)):
        raise SystemExit(f"stage {stage!r} needs the completion marker of {ORDER[i - 1]!r} first")
    commit = manifest.code_commit()
    if not dry_run and commit["dirty"]:
        raise SystemExit("the working tree must be clean at the freeze commit")
    bad = frozen_mismatches()
    if bad:
        raise SystemExit(f"frozen identities differ: {bad}")
    if retrievers is None:
        import run_eval
        bm25, d3 = retrieval.BM25Retriever(retrieval.SELECTED), run_eval.d3_retriever()
    else:
        bm25, d3 = retrievers
    if d3.describe().get("graph_fingerprint") != FROZEN["graph_fingerprint"]:
        raise SystemExit("the D3 graph fingerprint differs from the freeze")
    hashes = current_hashes(bm25.describe(), d3.describe())
    note = check_declaration(hashes, dry_run)                # real stages: raises unless declared
    if note:
        print(f"dry run: {note}")
    record = {"commit": commit["commit"], **hashes, "bm25_describe": _canon(bm25.describe()),
              "d3_describe": _canon(d3.describe())}
    freeze = os.path.join(d, "freeze.json")
    if os.path.exists(freeze):
        with open(freeze) as f:
            if json.load(f) != record:
                raise SystemExit("the commit or hashes differ from the recorded freeze")
    elif i == 0:
        _atomic_json(freeze, record)
    else:
        raise SystemExit("no freeze record: the first stage records it")
    if i > 0:
        validate_stage(ORDER[i - 1], dry_run)                # loads the previous split's ids
    fr = read_freeze(dry_run)
    client = client if client is not None else (make_client(dry_run) if stage != "retrieval" else None)
    qs = load_questions(stage, dry_run)
    qids = [q.qid for q in qs]
    split = stage_split(stage, dry_run)
    cfgs, fns = configs(stage, fr["bm25_describe"], fr["d3_describe"]), functions(stage, d3, bm25)
    marker = {}
    for name, cfg in cfgs.items():
        path = result_path(stage, name, dry_run)
        status = run_system(path, split, qs, fns[name], cfg, None if name in NO_LLM else client, commit)
        validate_system(path, split, qids, cfg, name not in NO_LLM, fr)
        print(f"{stage}/{name}: {status}, validated", flush=True)
        marker[name] = {"result_sha256": _sha_file(path), "manifest_sha256": _sha_file(path + ".manifest.json"),
                        "config_hash": manifest.sha256_of(cfg)}
    _atomic_json(marker_path(stage, dry_run), {"stage": stage, "split": split, "commit": commit["commit"],
                                               "systems": marker, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"stage {stage!r} complete")


# ---------------------------------------------------------------- analysis (§5.5-5.6)

def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Weighted empirical quantile (inverse of the weighted CDF): the smallest value whose
    cumulative normalized weight reaches ``q``."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    return float(v[order][np.searchsorted(cum, q - 1e-12)])


def question_weights(types: Mapping[str, str]) -> Dict[str, float]:
    """Each question's weight: its type's prevalence divided by its type's sample size."""
    n = {t: sum(1 for x in types.values() if x == t) for t in WEIGHTS}
    return {q: WEIGHTS[t] / n[t] for q, t in types.items()}


def conclusion(delta: float, p_adj: float) -> str:
    """§5.5: directional conclusions from the Holm-adjusted p-value."""
    if p_adj < 0.05 and delta > 0:
        return "improvement demonstrated" + (" (practically meaningful)" if delta >= MPI else "")
    if p_adj < 0.05 and delta < 0:
        return "worsening demonstrated"
    return "no clear difference demonstrated"


def _paired(sa, sb, metric, types, weighted):
    a = {q: s[metric] for q, s in sa.items()}
    b = {q: s[metric] for q, s in sb.items()}
    out = metrics.paired_bootstrap(a, b, strata=types if weighted else None, weights=WEIGHTS if weighted else None)
    out["per_type"] = {t: metrics.paired_bootstrap({q: v for q, v in a.items() if types[q] == t},
                                                   {q: v for q, v in b.items() if types[q] == t}) for t in WEIGHTS}
    return out


def confirmatory(scores: Mapping[str, Mapping[str, Mapping[str, Any]]], types: Mapping[str, str]) -> Dict[str, Any]:
    """The three adjacent comparisons on F1, prevalence-weighted, bootstrapped within type, Holm."""
    res = {name: {"a": a, "b": b, **_paired(scores[a], scores[b], "f1", types, True)} for name, a, b in FAMILY}
    adj = metrics.holm({k: v["p_value"] for k, v in res.items()})
    for k in res:
        res[k]["p_holm"] = adj[k]
        res[k]["conclusion"] = conclusion(res[k]["delta"], adj[k])
        res[k]["note"] = "95% interval not adjusted for multiplicity"
    return res


def _agg(values: Mapping[str, float], types: Mapping[str, str], weighted: bool) -> Dict[str, Any]:
    """Overall (prevalence-weighted or plain) and per-type means."""
    out = {t: float(np.mean([v for q, v in values.items() if types[q] == t])) for t in WEIGHTS
           if any(types[q] == t for q in values)}
    out["overall"] = metrics.weighted_mean(values, types, WEIGHTS) if weighted else float(np.mean(list(values.values())))
    return out


def _quantiles(values: Mapping[str, float], types: Mapping[str, str], weighted: bool) -> Dict[str, Any]:
    qw = question_weights(types) if weighted else {q: 1.0 for q in values}
    out = {"p50": weighted_quantile(list(values.values()), [qw[q] for q in values], 0.5),
           "p95": weighted_quantile(list(values.values()), [qw[q] for q in values], 0.95),
           "quantiles": "weighted empirical" if weighted else "empirical"}
    for t in WEIGHTS:
        v = [x for q, x in values.items() if types[q] == t]
        if v:
            out[t] = {"p50": weighted_quantile(v, [1] * len(v), 0.5), "p95": weighted_quantile(v, [1] * len(v), 0.95)}
    return out


def answer_metrics(recs, sc, types, weighted: bool, closed_book: bool = False) -> Dict[str, Any]:
    """§5.6 answer, grounding, context and evidence measures for one system."""
    out = {}
    names = ["em", "f1", "supported"] + ([] if closed_book else
                                          ["ctx_joint_recall", "ctx_recall", "sp_strict_f1", "sp_strict_prec",
                                           "sp_strict_recall", "sp_official_f1", "joint_official_f1",
                                           "has_citations", "all_valid_citations"])
    for m in names:
        out[m] = _agg({q: float(s[m]) for q, s in sc.items()}, types, weighted)
    ans = {}
    for t in WEIGHTS:
        qs = [q for q in sc if types[q] == t]
        if qs:
            ans[t] = metrics.answered_f1({q: sc[q]["f1"] for q in qs}, {q: bool(sc[q]["supported"]) for q in qs})
    if not weighted:
        overall = metrics.answered_f1({q: s["f1"] for q, s in sc.items()}, {q: bool(s["supported"]) for q, s in sc.items()})["f1_answered"]
    elif set(ans) == set(WEIGHTS) and all(a["f1_answered"] is not None for a in ans.values()):
        overall = float(sum(WEIGHTS[t] * ans[t]["f1_answered"] for t in WEIGHTS))     # weighted per-type answered F1
    else:
        overall = None                                                                 # a type with no answered output
    out["f1_answered"] = {**ans, "overall": overall, "n_answered": sum(a["n_answered"] for a in ans.values())}
    if closed_book:
        out["low_confidence_rate"] = _agg({q: 1.0 - float(s["supported"]) for q, s in sc.items()}, types, weighted)
        out["f1_when_confident"] = out["f1_answered"]
    else:
        out["per_citation_validity"] = _ratio({q: (sc[q]["valid_citations"], sc[q]["n_citations"]) for q in sc}, types, weighted)
        out["final_context_size"] = _agg({q: float(len(r["displayed"])) for q, r in recs.items()}, types, weighted)
    return out


def _ratio(parts: Mapping[str, tuple], types: Mapping[str, str], weighted: bool) -> Dict[str, Any]:
    """A pooled ratio (numerator, denominator per question) per type, and overall: the
    prevalence-weighted mean of the per-type ratios when weighted, else the pooled ratio.
    A zero denominator gives None."""
    out = {}
    for t in WEIGHTS:
        k = sum(a for q, (a, _) in parts.items() if types[q] == t)
        n = sum(b for q, (_, b) in parts.items() if types[q] == t)
        if any(types[q] == t for q in parts):
            out[t] = {"k": k, "n": n, "rate": k / n if n else None}
    if weighted:
        rates = [out.get(t, {}).get("rate") for t in WEIGHTS]
        out["overall"] = float(sum(w * r for w, r in zip(WEIGHTS.values(), rates))) if all(r is not None for r in rates) else None
    else:
        k, n = sum(a for a, _ in parts.values()), sum(b for _, b in parts.values())
        out["overall"] = k / n if n else None
    return out


def _shares(values: Mapping[str, Any], types: Mapping[str, str], weighted: bool) -> Dict[str, Any]:
    """For a categorical value per question: per-type counts, and per-type and overall
    (prevalence-weighted when weighted) shares of each category."""
    cats = sorted({str(v) for v in values.values()})
    return {"counts": {t: dict(collections.Counter(str(v) for q, v in values.items() if types[q] == t)) for t in WEIGHTS},
            "shares": {c: _agg({q: float(str(v) == c) for q, v in values.items()}, types, weighted) for c in cats}}


def _calls(rec) -> List[Mapping[str, Any]]:
    return rec["calls"] if "calls" in rec else [rec["llm"]]


def cost_metrics(recs, types, weighted: bool) -> Dict[str, Any]:
    """Logical calls, cache hits, unique new generations, tokens, LLM and retrieval time."""
    calls = {q: _calls(r) for q, r in recs.items()}
    new_keys = {c["key"] for cs in calls.values() for c in cs if not c["cached"]}
    return {
        "logical_calls": {"total": sum(len(c) for c in calls.values()),
                          "per_question": _agg({q: float(len(c)) for q, c in calls.items()}, types, weighted)},
        "cache_hits": sum(c["cached"] for cs in calls.values() for c in cs),
        "unique_new_generations": len(new_keys),
        "prompt_tokens": {"total": sum(c["prompt_tokens"] for cs in calls.values() for c in cs),
                          "per_question": _agg({q: float(sum(c["prompt_tokens"] for c in cs)) for q, cs in calls.items()}, types, weighted)},
        "output_tokens": {"total": sum(c["output_tokens"] for cs in calls.values() for c in cs),
                          "per_question": _agg({q: float(sum(c["output_tokens"] for c in cs)) for q, cs in calls.items()}, types, weighted)},
        "llm_seconds": _quantiles({q: float(sum(c["wall_s"] for c in cs)) for q, cs in calls.items()}, types, weighted),
        "retrieval_seconds": _quantiles({q: float(r.get("retrieval_s", 0.0)) for q, r in recs.items()}, types, weighted),
    }


def _key_fn():
    corpus = data.corpus()
    return lambda d: (corpus[d].title, corpus[d].text_hash)


def agent_behaviour(name: str, recs, qs: Mapping[str, data.Question], types, weighted: bool) -> Dict[str, Any]:
    """Evidence funnel, searches, stops or corrective paths, and role failures (§5.6)."""
    key = _key_fn()
    both = lambda ids, q: float(set(qs[q].gold) <= {key(d) for d in ids})
    funnel = {"search_1": _agg({q: both(r["batches"][0] if r["batches"] else [], q) for q, r in recs.items()}, types, weighted),
              "search_pool": _agg({q: both([d for b in r["batches"] for d in b], q) for q, r in recs.items()}, types, weighted),
              "final_context": _agg({q: both(r["displayed"], q) for q, r in recs.items()}, types, weighted)}
    out = {"funnel": funnel,
           "searches": {"mean": _agg({q: float(len(r["batches"])) for q, r in recs.items()}, types, weighted),
                        **_shares({q: len(r["batches"]) for q, r in recs.items()}, types, weighted)}}
    if name == "agent_single":
        out["stop_reasons"] = _shares({q: r["stop_reason"] for q, r in recs.items()}, types, weighted)
    else:
        ev = lambda r, cond: float(any(cond(e) for e in r["events"]))
        paths = {"attempted": lambda e: e.get("stage") == "corrective" and e["event"] in ("search", "search_skipped"),
                 "executed": lambda e: e.get("stage") == "corrective" and e["event"] == "search",
                 "no_new_search_results": lambda e: e.get("stage") == "corrective" and e["event"] == "search_skipped"}
        out["corrective"] = {k: {"count": int(sum(ev(r, c) for r in recs.values())),
                                 "per_question_rate": _agg({q: ev(r, c) for q, r in recs.items()}, types, weighted)}
                             for k, c in paths.items()}
        role = {"planner": "planner", "researcher": "researchers", "verifier": "initial verifier", "verifier_recheck": "recheck"}
        err = {"planner_error": "planner", "researcher_error": "researchers", "verifier_error": "initial verifier",
               "recheck_error": "recheck"}
        out["role_failures"] = {
            k: _ratio({q: (sum(1 for e in r["events"] if err.get(e["event"]) == k),
                           sum(1 for c in r["calls"] if role.get(c["role"]) == k)) for q, r in recs.items()}, types, weighted)
            for k in ("planner", "researchers", "initial verifier", "recheck")}
        first = {q: next((str(e["sufficient"]) if e["event"] == "verify" else "error") for e in r["events"]
                         if e["event"] in ("verify", "verifier_error")) for q, r in recs.items()
                 if any(e["event"] in ("verify", "verifier_error") for e in r["events"])}
        out["first_verification"] = _shares(first, types, weighted)
        out["mean_accepted"] = _agg({q: float(len(r["accepted"])) for q, r in recs.items()}, types, weighted)
    return out


def retrieval_report(rows_by_system: Mapping[str, Sequence[Mapping[str, Any]]], types: Mapping[str, str]) -> Dict[str, Any]:
    """Recall@K and Joint Recall@K curves, exact and title-only, overall and by type (§5.6)."""
    out = {}
    for n, rows in rows_by_system.items():
        keys = [k for k in rows[0] if k.startswith(("recall@", "joint_recall@", "title_recall@", "title_joint_recall@"))]
        by = {k: {r["qid"]: float(r[k]) for r in rows} for k in keys}
        out[n] = {"curves": {k: _agg(v, types, weighted=False) for k, v in by.items()},
                  "latency_seconds": _quantiles({r["qid"]: float(r["latency_s"]) for r in rows}, types, weighted=False)}
    a, b = ({r["qid"]: float(r["joint_recall@10"]) for r in rows_by_system[n]} for n in ("d3_rankings", "bm25_rankings"))
    out["jr10_d3_minus_bm25"] = metrics.paired_bootstrap(a, b)
    return out


def analysis(dry_run: bool = False) -> Dict[str, Any]:
    """§5.5-5.6, from validated stored runs only, by the frozen code only."""
    check_consumer(dry_run)
    qsets = {st: load_questions(st, dry_run) for st in ORDER}
    for st in ORDER:
        validate_stage(st, dry_run, [q.qid for q in qsets[st]])
    out = {}
    # test_agents (prevalence-weighted)
    qs = {q.qid: q for q in qsets["agents"]}
    types = {q: x.type for q, x in qs.items()}
    recs = {n: {r["qid"]: r for r in _load(result_path("agents", n, dry_run))} for n in STAGE_SYSTEMS["agents"]}
    mans = {n: json.load(open(result_path("agents", n, dry_run) + ".manifest.json")) for n in recs}
    shared = ["answer_prompt_hash", "schema_hash", "answer_validator_hash", "system_prompt_hash", "renderer_hash"]
    for n in recs:
        if n != "multi_agent":
            manifest.check_comparable(mans["multi_agent"], mans[n], shared=[] if n == "closed_book" else shared, final=True)
    sc = {n: {q: scoring.score(qs[q], r, evidence=n != "closed_book") for q, r in rr.items()} for n, rr in recs.items()}
    out["confirmatory"] = confirmatory(sc, types)
    evid = [n for n in recs if n != "closed_book"]
    out["test_agents"] = {
        "systems": {n: {**answer_metrics(recs[n], sc[n], types, True, closed_book=n == "closed_book"),
                        "cost": cost_metrics(recs[n], types, True)} for n in recs},
        "secondary_comparisons": {name: {m: _paired(sc[a], sc[b], m, types, True) for m in ("f1", "ctx_joint_recall")}
                                  for name, a, b in SECONDARY},
        "joint_recall_family_pairs": {name: _paired(sc[a], sc[b], "ctx_joint_recall", types, True) for name, a, b in FAMILY},
        "agents": {n: agent_behaviour(n, recs[n], qs, types, True) for n in ("agent_single", "multi_agent")},
        "systems_with_evidence": evid,
    }
    # test_rag (proportional, unweighted)
    rq = {q.qid: q for q in qsets["rag"]}
    rtypes = {q: x.type for q, x in rq.items()}
    rrecs = {n: {r["qid"]: r for r in _load(result_path("rag", n, dry_run))} for n in STAGE_SYSTEMS["rag"]}
    rsc = {n: {q: scoring.score(rq[q], r, evidence=n != "closed_book") for q, r in rr.items()} for n, rr in rrecs.items()}
    out["test_rag"] = {"systems": {n: {**answer_metrics(rrecs[n], rsc[n], rtypes, False, closed_book=n == "closed_book"),
                                       "cost": cost_metrics(rrecs[n], rtypes, False)} for n in rrecs},
                       "R_minus_baseline": {m: _paired(rsc["d3_k10"], rsc["bm25_k10"], m, rtypes, False) for m in ("f1", "em")}}
    # test_retrieval (proportional, unweighted)
    tq = {q.qid: q.type for q in qsets["retrieval"]}
    out["test_retrieval"] = retrieval_report({n: _load(result_path("retrieval", n, dry_run)) for n in STAGE_SYSTEMS["retrieval"]}, tq)
    # cache reuse across splits and systems (§5.4)
    every = [c for st in ("rag", "agents") for n in STAGE_SYSTEMS[st]
             for r in _load(result_path(st, n, dry_run)) for c in _calls(r)]
    out["cache"] = {"logical_calls": len(every), "cache_hits": sum(c["cached"] for c in every),
                    "unique_new_generations": len({c["key"] for c in every if not c["cached"]})}
    return out


# ---------------------------------------------------------------- the blinded audit (§5.7)

AUDIT_INSTRUCTIONS = """Label every item in `labels.csv` (columns `support`, `citation_validity`).

**Evidence support**, applied in this order of precedence. Claims are the answer's
material factual claims; "stated" means stated by the paragraphs shown for that item.

1. `cannot_determine`: ambiguity in the answer or the context prevents applying the other labels.
2. `contradicted`: at least one material answer claim is explicitly contradicted (takes
   precedence over the labels below).
3. `fully_supported`: every material factual claim is stated, and none is contradicted.
4. `partially_supported`: at least one claim is stated, at least one is not, and none is contradicted.
5. `unsupported`: no material claim is stated or contradicted.

**Citation validity:**

- `valid`: at least one citation exists, every citation resolves to a shown sentence, and
  every cited sentence is relevant to at least one factual answer claim, whether it
  supports or contradicts that claim;
- `invalid`: otherwise, including no citations and any `UNRESOLVED` citation.

Whether every claim is covered is judged by the support label, not here.
"""


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_sample(types: Mapping[str, str]) -> List[str]:
    """12 bridge + 12 comparison ids, by the frozen hash ordering within type."""
    out = []
    for t in ("bridge", "comparison"):
        ids = sorted((q for q, x in types.items() if x == t), key=lambda q: _h(AUDIT_SEED + q))
        if len(ids) < AUDIT_PER_TYPE:
            raise ValueError(f"fewer than {AUDIT_PER_TYPE} {t} questions")
        out += ids[:AUDIT_PER_TYPE]
    return out


def audit_items(sample: Sequence[str]) -> List[tuple]:
    """(qid, system) pairs in the frozen presentation order."""
    return sorted(((q, s) for q in sample for s in AUDIT_SYSTEMS), key=lambda p: _h(f"{AUDIT_ORDER_SEED}{p[0]}:{p[1]}"))


def render_citations(rec: Mapping[str, Any], docs: Mapping[str, data.Doc]) -> List[str]:
    """Cited sentences as text; an unresolved citation is shown, never dropped."""
    shown = list(rec["displayed"])
    out = []
    for d, s in rec["citations"]:
        if d in shown and d in docs and 0 <= int(s) < len(docs[d].sentences):
            out.append(f"[{docs[d].title}, sentence {s}] {docs[d].sentences[int(s)].strip()}")
        else:
            out.append(f"[UNRESOLVED citation: {d}, sentence {s}]")
    return out


def audit_dir(dry_run: bool) -> str:
    return os.path.join(out_dir(dry_run), "audit")


def build_audit_files(items, recs, qs, docs) -> Dict[str, str]:
    """The blinded sheet, the key and the label template, as file contents."""
    sheet, key = ["# Blinded audit sheet (PREREGISTRATION.md §5.7)\n", AUDIT_INSTRUCTIONS], []
    for i, (q, s) in enumerate(items, 1):
        r = recs[s][q]
        sheet.append(f"## Item {i}\n\n**Question:** {qs[q].question}\n\n**Answer:** {r['output']['answer']}\n")
        cites = render_citations(r, docs)
        sheet.append("**Cited sentences:**\n" + ("\n".join(f"- {c}" for c in cites) if cites else "- (no citations)") + "\n")
        sheet.append("**Paragraphs shown to the answer step:**\n")
        for p, did in enumerate(r["displayed"], 1):
            sheet.append(f"[P{p}] {docs[did].title}\n" + "\n".join(f"  [{k}] {x.strip()}" for k, x in enumerate(docs[did].sentences)) + "\n")
        key.append({"item": i, "qid": q, "type": qs[q].type, "system": s, "supported": bool(r["output"]["supported"])})
    files = {"sheet.md": "\n".join(sheet), "key.jsonl": "".join(json.dumps(k) + "\n" for k in key),
             "labels.csv": "item,support,citation_validity\n" + "".join(f"{k['item']},,\n" for k in key)}
    files["audit_manifest.json"] = json.dumps(
        {"sheet_sha256": _h(files["sheet.md"]), "key_sha256": _h(files["key.jsonl"]), "n_items": len(key),
         "note": "sheet.md and key.jsonl are immutable; only labels.csv is edited"}, indent=1, sort_keys=True)
    return files


def audit_sheet(dry_run: bool = False) -> str:
    """Creates the audit directory atomically (built aside, renamed into place once)."""
    check_consumer(dry_run)
    validate_stage("agents", dry_run)
    final_dir = audit_dir(dry_run)
    if os.path.exists(final_dir):
        raise SystemExit(f"{final_dir} exists; the audit files are generated once")
    qs = {q.qid: q for q in load_questions("agents", dry_run)}
    recs = {n: {r["qid"]: r for r in _load(result_path("agents", n, dry_run))} for n in AUDIT_SYSTEMS}
    files = build_audit_files(audit_items(audit_sample({q: x.type for q, x in qs.items()})), recs, qs, data.corpus())
    return write_dir_atomic(final_dir, files)


def write_dir_atomic(final_dir: str, files: Mapping[str, str]) -> str:
    """All files built in a side directory, then renamed into place in one step; a leftover
    side directory from an interruption is never complete and is rebuilt."""
    if os.path.exists(final_dir):
        raise SystemExit(f"{final_dir} exists; it is generated once")
    tmp = final_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for name, text in files.items():
        with open(os.path.join(tmp, name), "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, final_dir)
    return final_dir


def read_labels(path: str, n_items: int) -> Dict[int, Dict[str, str]]:
    """Completed labels, validated: every item exactly once, every field a defined label."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    problems = []
    items = [r.get("item", "").strip() for r in rows]
    if sorted(items, key=lambda x: int(x) if x.isdigit() else -1) != [str(i) for i in range(1, n_items + 1)]:
        problems.append("items must be exactly 1..N, once each")
    for r in rows:
        if r.get("support", "").strip() not in SUPPORT_LABELS:
            problems.append(f"item {r.get('item')}: support {r.get('support')!r}")
        if r.get("citation_validity", "").strip() not in CITATION_LABELS:
            problems.append(f"item {r.get('item')}: citation_validity {r.get('citation_validity')!r}")
    if problems:
        raise SystemExit(f"labels do not validate: {problems[:10]}{' ...' if len(problems) > 10 else ''}")
    return {int(r["item"]): {"support": r["support"].strip(), "citation_validity": r["citation_validity"].strip()} for r in rows}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    from scipy.stats import beta
    if n == 0:
        raise ValueError("no denominator")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _rate(k: int, n: int) -> Dict[str, Any]:
    if n == 0:
        return {"k": k, "n": 0, "rate": None, "ci95": None, "note": "undefined (zero denominator)"}
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def audit_summary(key: Sequence[Mapping[str, Any]], labels: Mapping[int, Mapping[str, str]]) -> Dict[str, Any]:
    """§5.7: per system and type, and the balanced-audit rate (not a population estimate)."""
    out = {}
    for s in AUDIT_SYSTEMS:
        rows = [(k, labels[k["item"]]) for k in key if k["system"] == s]
        res = {}
        for group in ("bridge", "comparison", "balanced_audit"):
            g = [(k, l) for k, l in rows if group == "balanced_audit" or k["type"] == group]
            claimed = [(k, l) for k, l in g if k["supported"]]
            bad = sum(l["support"] in ("unsupported", "contradicted") for _, l in claimed)
            res[group] = {"unsupported_claim_rate": _rate(bad, len(claimed)), "n_outputs": len(g),
                          "support_distribution": {lab: sum(l["support"] == lab for _, l in g) for lab in SUPPORT_LABELS},
                          "citation_distribution": {lab: sum(l["citation_validity"] == lab for _, l in g) for lab in CITATION_LABELS}}
        res["balanced_audit"]["note"] = "balanced 12+12 descriptive sample; not a population-prevalence estimate"
        out[s] = res
    return out


def verify_audit_files(d: str, qs: Mapping[str, data.Question], recs) -> List[Dict[str, Any]]:
    """The immutable sheet and key: their recorded hashes, and the expected 96 item
    mappings (systems, types, supported flags) in the frozen order."""
    with open(os.path.join(d, "audit_manifest.json")) as f:
        man = json.load(f)
    with open(os.path.join(d, "sheet.md")) as f:
        sheet = f.read()
    with open(os.path.join(d, "key.jsonl")) as f:
        key_text = f.read()
    if _h(sheet) != man["sheet_sha256"] or _h(key_text) != man["key_sha256"]:
        raise SystemExit("the audit sheet or key changed after creation")
    key = [json.loads(line) for line in key_text.splitlines() if line.strip()]
    expected = audit_items(audit_sample({q: x.type for q, x in qs.items()}))
    want = [{"item": i, "qid": q, "type": qs[q].type, "system": s, "supported": bool(recs[s][q]["output"]["supported"])}
            for i, (q, s) in enumerate(expected, 1)]
    if key != want or man["n_items"] != len(want) or len(want) != 2 * AUDIT_PER_TYPE * len(AUDIT_SYSTEMS):
        raise SystemExit("the audit key does not match the frozen sample, order and records")
    return key


def audit_analysis(dry_run: bool = False) -> Dict[str, Any]:
    """§5.7, after the frozen-code check, the agents stage and the audit files are validated."""
    check_consumer(dry_run)
    validate_stage("agents", dry_run)
    qs = {q.qid: q for q in load_questions("agents", dry_run)}
    recs = {n: {r["qid"]: r for r in _load(result_path("agents", n, dry_run))} for n in AUDIT_SYSTEMS}
    d = audit_dir(dry_run)
    key = verify_audit_files(d, qs, recs)
    labels = read_labels(os.path.join(d, "labels.csv"), len(key))
    return audit_summary(key, labels)
