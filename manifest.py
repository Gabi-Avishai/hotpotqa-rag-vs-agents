"""Run manifests and the rule for which runs may be compared.

Every results file gets a manifest with three sections:

- ``comparison_invariants``: must be identical for two runs to be compared (data, split,
  evaluation code and metric definitions, base LLM and context size, packages);
- ``system_config``: what defines the system under test; differences are allowed and are
  printed next to the comparison;
- ``run_metadata``: timestamp, runtime and machine; ignored by comparisons.

The code commit is recorded in every manifest. It is an invariant only for final
comparisons, where all systems must run from one clean commit; during development, runs
from different commits are comparable when the evaluation code hash and every other
invariant agree.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import data
import metrics

PACKAGES = ("numpy", "pandas", "datasets", "bm25s", "sentence-transformers", "torch", "faiss-cpu",
            "langgraph", "langchain-core", "pydantic", "tokenizers", "scipy")
EVAL_CODE = ("data.py", "metrics.py", "scoring.py", "hops.py", "run_eval.py", "final.py")


class IncompatibleRuns(ValueError):
    """Two runs differ in something that must be identical for a valid comparison."""


def sha256_of(obj: Any) -> str:
    """Hash of a string, or of a JSON-serialisable object in canonical form."""
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def eval_code_hash() -> str:
    h = hashlib.sha256()
    for name in EVAL_CODE:
        with open(os.path.join(data.HERE, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


def code_commit() -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=data.HERE, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def package_versions(names: Iterable[str] = PACKAGES) -> Dict[str, Optional[str]]:
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def build(split: str, split_ids: Sequence[str], system_config: Mapping[str, Any],
          llm: Optional[Mapping[str, Any]] = None, runtime_s: Optional[float] = None,
          commit: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Manifest for one run. ``llm`` is ``OllamaClient.describe()`` or None for runs without
    an LLM (retrieval-only).

    The corpus is re-fingerprinted and must match the pinned value, and ``split_ids`` must
    be exactly the named split (``data.validate_split_ids``): a manifest cannot claim the
    expected corpus or split while the run used something else."""
    data.validate_split_ids(split, split_ids)
    invariants = {
        "dataset": f"{data.DATASET}/{data.CONFIG}@{data.REVISION}",
        "corpus_fingerprint": data.verify_corpus(),
        "split": split,
        "split_hash": data.ids_hash(split_ids),
        "n_questions": len(split_ids),
        "eval_code_hash": eval_code_hash(),
        "metrics_version": metrics.METRICS_VERSION,
        "base_llm": None if llm is None else {k: llm[k] for k in ("model", "digest", "quantization", "tokenizer")},
        "num_ctx": None if llm is None else llm["options"]["num_ctx"],
        # the structured-decoding runtime (Phase 4b); None when the source run never recorded it
        "ollama_version": None if llm is None else llm.get("ollama_version"),
        "packages": package_versions(),
        "code": dict(commit) if commit is not None else code_commit(),
    }
    config = dict(system_config)
    if llm is not None:
        config["llm_options"] = dict(llm["options"])
    return {
        "comparison_invariants": invariants,
        "system_config": config,
        "run_metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_s": runtime_s,
                         "host": socket.gethostname(), "platform": platform.platform(),
                         "python": platform.python_version()},
    }


def write(manifest: Mapping[str, Any], results_path: str) -> str:
    path = results_path + ".manifest.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def check_comparable(a: Mapping[str, Any], b: Mapping[str, Any], shared: Sequence[str] = (),
                     final: bool = False, unrecorded_runtime_ok: bool = False) -> Dict[str, Any]:
    """Raise ``IncompatibleRuns`` unless ``a`` and ``b`` may be compared.

    ``shared`` lists ``system_config`` keys the experiment claims are the same in both
    systems (e.g. the answer prompt hash); they must be equal. ``final`` additionally
    requires both runs to come from the same clean commit. Two recorded Ollama versions
    must always agree. ``unrecorded_runtime_ok`` (amendment A7) admits only a legacy `dev`
    run whose version was never recorded, and reports the version as unknown; it is
    refused for the blind splits, where every run records its runtime. Returns the
    allowed differences as ``{key: (a_value, b_value)}``."""
    ia, ib = dict(a["comparison_invariants"]), dict(b["comparison_invariants"])
    if unrecorded_runtime_ok and any(str(i.get("split", "")).startswith("test_") for i in (ia, ib)):
        raise IncompatibleRuns("the unrecorded-runtime exception is for legacy dev runs only, never a blind split")
    ca, cb = ia.pop("code"), ib.pop("code")
    va, vb = ia.pop("ollama_version", None), ib.pop("ollama_version", None)
    problems = [k for k in sorted(set(ia) | set(ib)) if ia.get(k) != ib.get(k)]
    runtime_note = {}
    if va != vb:
        if va is not None and vb is not None or not unrecorded_runtime_ok:
            problems.append("ollama_version")
        else:
            runtime_note = {"ollama_version (unknown: not recorded by the legacy run)": (va, vb)}
    if final:
        if ca["commit"] != cb["commit"]:
            problems.append("code.commit")
        if ca["dirty"] or cb["dirty"]:
            problems.append("code.dirty")
    sa, sb = a["system_config"], b["system_config"]
    for k in shared:
        if k not in sa or k not in sb or sa[k] != sb[k]:
            problems.append(f"shared system_config.{k}")
    if problems:
        raise IncompatibleRuns("runs are not comparable; differing: " + ", ".join(problems))
    return {**runtime_note, **{k: (sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)}}
