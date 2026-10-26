"""TAS-B dense retrieval, candidate D1 (PREREGISTRATION.md §2.7).

``sentence-transformers/msmarco-distilbert-base-tas-b`` at a pinned revision: CLS
pooling, 768-dimensional float32 vectors, **no normalization**, dot-product similarity,
maximum length 512. Paragraphs are encoded as ``title. text`` (original text), queries
as the original question. The corpus is encoded once (GPU, Ollama unloaded, batches of
64) and cached under a key covering every setting; queries are encoded on the CPU.
Search is exact: dot products with every paragraph, then ``deterministic_top_k`` with the
``doc_id`` tie-break.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import data
from retrieval import deterministic_top_k, id_tiebreak_rank

MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"
REVISION = "b12d9352e776979147078a8975a4885042984fd1"
MAX_LENGTH = 512
DIM = 768
BATCH = 64
TEXT_TEMPLATE = "title. text"
CACHE_DIR = os.path.join(data.HERE, "cache", "dense")


def paragraph_text(doc: data.Doc) -> str:
    return f"{doc.title}. {doc.text}"


def load_model(device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, revision=REVISION, device=device)
    m.max_seq_length = MAX_LENGTH
    assert [type(mod).__name__ for mod in m] == ["Transformer", "Pooling"], "no normalization module"
    assert m[1].get_config_dict()["pooling_mode"] == "cls", "TAS-B uses CLS pooling"
    assert m[1].get_config_dict()["embedding_dimension"] == DIM
    return m


def encode(model, texts: Sequence[str], batch_size: int = BATCH) -> np.ndarray:
    v = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=False, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def matrix_fingerprint(m: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(m, dtype=np.float32).tobytes()).hexdigest()


def cache_key(device: str) -> Dict[str, object]:
    return {"model": MODEL, "revision": REVISION, "corpus_fingerprint": data.verify_corpus(),
            "text_template": TEXT_TEMPLATE, "max_length": MAX_LENGTH, "dtype": "float32",
            "device": device, "batch_size": BATCH, "normalize": False, "pooling": "cls"}


def corpus_embeddings(device: str = "cuda") -> Tuple[List[str], np.ndarray, Dict[str, object]]:
    """(doc_ids, matrix, info). Encoded once and cached; info records the cache key, the
    matrix fingerprint, encoding time and peak GPU memory."""
    key = cache_key(device)
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"tasb_{name}")
    docs = list(data.corpus().values())
    ids = [d.doc_id for d in docs]
    if os.path.exists(path + ".npy"):
        m = np.load(path + ".npy")
        with open(path + ".json") as f:
            info = json.load(f)
        assert info["doc_ids_hash"] == data.ids_hash(ids) and info["fingerprint"] == matrix_fingerprint(m)
        return ids, m, info
    import torch
    model = load_model(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m = encode(model, [paragraph_text(d) for d in docs])
    elapsed = time.perf_counter() - t0
    info = {"key": key, "shape": list(m.shape), "fingerprint": matrix_fingerprint(m),
            "doc_ids_hash": data.ids_hash(ids), "encode_s": elapsed, "bytes": int(m.nbytes),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path + ".tmp.npy", m)
    os.replace(path + ".tmp.npy", path + ".npy")
    with open(path + ".json", "w") as f:
        json.dump(info, f, indent=1)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ids, m, info


class DenseRetriever:
    """Exact dot-product search over the cached corpus matrix; CPU query encoder."""

    def __init__(self, ids: Sequence[str], matrix: np.ndarray, query_model=None):
        self.doc_ids = np.asarray(ids)
        self.matrix = matrix
        self._tiebreak = id_tiebreak_rank(self.doc_ids)
        self.model = query_model if query_model is not None else load_model("cpu")

    def query_vector(self, question: str) -> np.ndarray:
        return encode(self.model, [question], batch_size=1)[0]

    def search(self, question: str, k: int = 100) -> List[Tuple[str, float]]:
        scores = self.matrix @ self.query_vector(question)
        top = deterministic_top_k(scores, k, self._tiebreak)
        return [(str(self.doc_ids[i]), float(scores[i])) for i in top]
