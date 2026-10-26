"""HotpotQA data layer: pinned loading, the shared closed corpus, splits and the blind guard.

The corpus is every context paragraph of the distractor validation split, deduplicated on
exact identity ``(title, text_hash)``. Splits are drawn deterministically from validation
(and train, for EDA); their manifest is a local file regenerated on demand and checked
against the hashes stored below, so a clean clone either reproduces them exactly or fails.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SEED = 20260927

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(HERE, "splits", "splits_v1.json")

TYPES = ("bridge", "comparison")
BLIND_SPLITS = frozenset({"test_retrieval", "test_rag", "test_agents"})

# top-level validation splits, drawn in this order (hence disjoint), validation-proportional
SPLIT_SPEC = {
    "tune": 500,
    "dev": 100,
    "test_retrieval": 1000,
}
NESTED_SPEC = {  # name: (parent, counts per type)
    "dev_smoke": ("dev", {"bridge": 16, "comparison": 4}),
    "test_rag": ("test_retrieval", {"bridge": 160, "comparison": 40}),
    "test_agents": ("test_retrieval", {"bridge": 50, "comparison": 50}),
}
EDA_SIZE = 1500
EXPECTED_SPLIT_SIZES: Dict[str, int] = {**SPLIT_SPEC, "eda": EDA_SIZE,
                                        **{n: sum(c.values()) for n, (_, c) in NESTED_SPEC.items()}}
ADHOC_PREFIX = "adhoc:"   # explicitly named ad-hoc question sets (e.g. a debugging subset)

# Recorded at the first generation (2026-09-27); any change to data, code or seed fails these.
EXPECTED_CORPUS_FINGERPRINT = "67db29c2a555974d0d16e8a3bb9c9865ee8c542d6d268ed88e0232e340fee3aa"
EXPECTED_SPLIT_HASHES: Dict[str, str] = {
    "dev": "90a6ad35c98b323a1994457b2b3119c575a99356006b9243c4490442a5835b8a",
    "dev_smoke": "ec1c0dd9cf116f784d2a6692f931fd9ec7e8fcc989076714c02d6c4a49e3e04e",
    "eda": "9e7e631a98f1c524689a2f7e327c3eaf31e1f512a6be471e7fd0661d05ec523a",
    "test_agents": "6b72130cb1f09e7a21129f81d50876525c1306f4df426777f3c11e1ce470f447",
    "test_rag": "57201688a97654e73d633128498c00ff92c1f0e6984aa296b5ad506e6ab83d8d",
    "test_retrieval": "8654f35c1493a822245eeb07208c95999f900e59dc6adff01a5e43167a0a28ef",
    "tune": "7d81b361d43676e66dd8c51129c741e344dcbdad255f84618141a1c53c0dcc40",
}


class BlindSplitError(RuntimeError):
    """Raised when a blind test split is loaded without ``unblind=True``."""


class SplitMismatch(ValueError):
    """A set of question ids does not match the split it claims to be."""


def validate_split_ids(name: str, ids: Sequence[str]) -> None:
    """Check that ``ids`` is exactly the named split: no duplicates, the expected count and
    the pinned hash. Names starting with ``adhoc:`` accept any duplicate-free id set, so a
    subset can never pass as a recognised split. Reads ids only."""
    if len(set(ids)) != len(ids):
        raise SplitMismatch(f"{name!r}: {len(ids) - len(set(ids))} duplicate id(s)")
    if name.startswith(ADHOC_PREFIX):
        if not ids:
            raise SplitMismatch(f"{name!r}: no ids")
        return
    if name not in EXPECTED_SPLIT_HASHES:
        raise SplitMismatch(f"unknown split {name!r}; name ad-hoc sets {ADHOC_PREFIX}<name>")
    if len(ids) != EXPECTED_SPLIT_SIZES[name]:
        raise SplitMismatch(f"{name!r}: {len(ids)} ids, expected {EXPECTED_SPLIT_SIZES[name]}")
    if ids_hash(ids) != EXPECTED_SPLIT_HASHES[name]:
        raise SplitMismatch(f"{name!r}: id hash differs from the pinned hash")


def text_hash(sentences: Sequence[str]) -> str:
    """Identity of a paragraph's text. The sentence segmentation is part of it, because
    supporting-fact sentence ids refer to that segmentation."""
    return hashlib.sha256("\x1f".join(sentences).encode("utf-8")).hexdigest()


def ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text_hash: str
    sentences: Tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()

    @property
    def index_text(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answer: str
    type: str
    level: str
    gold: Tuple[Tuple[str, str], ...]              # exact identities (title, text_hash)
    supporting_facts: Tuple[Tuple[str, int], ...]  # official (title, sentence_id)
    context: Tuple[Tuple[str, str], ...]           # the question's own (title, text_hash) list


# ---------------------------------------------------------------- loading

@lru_cache(maxsize=2)
def _raw(split: str):
    from datasets import load_dataset
    return load_dataset(DATASET, CONFIG, split=split, revision=REVISION)


def _to_question(ex) -> Question:
    ctx = tuple((t, text_hash(s)) for t, s in zip(ex["context"]["title"], ex["context"]["sentences"]))
    sf = tuple(zip(ex["supporting_facts"]["title"], (int(i) for i in ex["supporting_facts"]["sent_id"])))
    gold_titles = sorted(set(t for t, _ in sf))
    own = {}
    for t, h in ctx:
        own.setdefault(t, h)
    gold = tuple((t, own[t]) for t in gold_titles)
    return Question(ex["id"], ex["question"], ex["answer"], ex["type"], ex["level"], gold, sf, ctx)


# ---------------------------------------------------------------- corpus

@lru_cache(maxsize=1)
def corpus() -> Dict[str, Doc]:
    """Deduplicated corpus keyed by doc_id. A title with one text keeps the title as its id;
    a title with several texts gets ``"<title> #<hash8>"`` for every variant, so ids never
    depend on encounter order."""
    variants: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for ex in _raw("validation"):
        for t, s in zip(ex["context"]["title"], ex["context"]["sentences"]):
            variants.setdefault(t, {}).setdefault(text_hash(s), tuple(s))
    docs = {}
    for t in sorted(variants):
        for h in sorted(variants[t]):
            doc_id = t if len(variants[t]) == 1 else f"{t} #{h[:8]}"
            docs[doc_id] = Doc(doc_id, t, h, variants[t][h])
    return docs


@lru_cache(maxsize=1)
def doc_id_of() -> Dict[Tuple[str, str], str]:
    """(title, text_hash) -> doc_id."""
    return {(d.title, d.text_hash): d.doc_id for d in corpus().values()}


def corpus_fingerprint() -> str:
    lines = sorted(f"{d.doc_id}\t{d.text_hash}" for d in corpus().values())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- splits

def _counts(n: int, prevalence: Mapping[str, float]) -> Dict[str, int]:
    b = int(round(n * prevalence["bridge"]))
    return {"bridge": b, "comparison": n - b}


def make_splits() -> Dict[str, List[str]]:
    """Deterministic split ids. Reads only question ids and types (no questions, answers
    or evidence), so generating blind splits does not open them."""
    val = _raw("validation").select_columns(["id", "type"])
    pools = {ty: sorted(i for i, t in zip(val["id"], val["type"]) if t == ty) for ty in TYPES}
    n_val = sum(len(p) for p in pools.values())
    prevalence = {ty: len(pools[ty]) / n_val for ty in TYPES}

    ss = np.random.SeedSequence(SEED)
    main_rng, *nested_rngs = [np.random.default_rng(s) for s in ss.spawn(1 + len(NESTED_SPEC) + 1)]
    eda_rng = nested_rngs.pop()
    order = {ty: list(main_rng.permutation(pools[ty])) for ty in TYPES}

    splits: Dict[str, List[str]] = {}
    for name, n in SPLIT_SPEC.items():  # consumed in order, hence disjoint
        take = _counts(n, prevalence)
        splits[name] = [q for ty in TYPES for q in order[ty][:take[ty]]]
        for ty in TYPES:
            order[ty] = order[ty][take[ty]:]

    types = dict(zip(val["id"], val["type"]))
    for (name, (parent, counts)), rng in zip(NESTED_SPEC.items(), nested_rngs):
        for ty in TYPES:
            members = sorted(q for q in splits[parent] if types[q] == ty)
            splits.setdefault(name, []).extend(rng.permutation(members)[:counts[ty]].tolist())

    tr = _raw("train").select_columns(["id", "type"])
    tr_pools = {ty: sorted(i for i, t in zip(tr["id"], tr["type"]) if t == ty) for ty in TYPES}
    n_tr = sum(len(p) for p in tr_pools.values())
    eda_counts = _counts(EDA_SIZE, {ty: len(tr_pools[ty]) / n_tr for ty in TYPES})
    splits["eda"] = [q for ty in TYPES for q in eda_rng.permutation(tr_pools[ty])[:eda_counts[ty]].tolist()]
    return {k: [str(q) for q in v] for k, v in splits.items()}


def split_hashes(splits: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    return {k: ids_hash(v) for k, v in sorted(splits.items())}


def load_split_ids(name: str) -> List[str]:
    if not os.path.exists(SPLITS_PATH):
        splits = make_splits()
        os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)
        tmp = SPLITS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"revision": REVISION, "seed": SEED, "splits": splits}, f, indent=0)
        os.replace(tmp, SPLITS_PATH)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)["splits"]
    if name not in splits:
        raise SplitMismatch(f"unknown split {name!r}")
    validate_split_ids(name, splits[name])   # duplicates, count and pinned hash
    return splits[name]


def load_split(name: str, unblind: bool = False) -> List[Question]:
    """Questions of a split, in manifest order. Blind splits need ``unblind=True``."""
    if name in BLIND_SPLITS and not unblind:
        raise BlindSplitError(f"{name!r} is blind; only the final evaluation may pass unblind=True")
    ids = load_split_ids(name)
    raw = _raw("train" if name == "eda" else "validation")
    pos = {q: i for i, q in enumerate(raw["id"])}
    return [_to_question(raw[pos[q]]) for q in ids]


def load_contexts(name: str, unblind: bool = False) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Each question's own supplied paragraphs, ``{qid: {title: sentences}}``. For EDA and
    oracle-style smoke tests; systems under evaluation search the shared corpus instead."""
    if name in BLIND_SPLITS and not unblind:
        raise BlindSplitError(f"{name!r} is blind; only the final evaluation may pass unblind=True")
    raw = _raw("train" if name == "eda" else "validation")
    pos = {q: i for i, q in enumerate(raw["id"])}
    out = {}
    for q in load_split_ids(name):
        ctx = raw[pos[q]]["context"]
        out[q] = {t: tuple(s) for t, s in zip(ctx["title"], ctx["sentences"])}
    return out


def verify_corpus() -> str:
    """Fail loudly if the rebuilt corpus differs from the one the project was built on."""
    got = corpus_fingerprint()
    if got != EXPECTED_CORPUS_FINGERPRINT:
        raise RuntimeError(f"corpus fingerprint {got[:12]} != expected {EXPECTED_CORPUS_FINGERPRINT[:12]}")
    return got
