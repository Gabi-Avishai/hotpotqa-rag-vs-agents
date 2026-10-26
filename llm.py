"""Local LLM client: Ollama in raw mode, schema-constrained JSON, a replay cache and accounting.

- **Raw ChatML.** The prompt is rendered here (the same ChatML Ollama's template produces
  for qwen2.5) and sent with ``raw: true``, so the exact text the model sees is ours and is
  hashed into the cache key; no hidden default system prompt is added.
- **Pre-generation context check.** Ollama silently truncates prompts longer than
  ``num_ctx``. Before every uncached call the prompt is counted with the model's own
  tokenizer (pinned Hugging Face revision); a call whose prompt plus ``num_predict`` does
  not fit raises ``ContextOverflow``. After the call, Ollama's ``prompt_eval_count`` must
  equal the pre-count, which verifies both the tokenizer and that nothing was cut.
- **Cache.** Keyed by model digest, rendered prompt, options and output schema. Temperature
  0 and a seed do not guarantee identical first generations, so reported numbers always
  come from cached responses: reruns replay rather than regenerate.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

import data

DEFAULT_MODEL = "qwen2.5:7b"
TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 8192, "num_predict": 512}
DEFAULT_SYSTEM = "You are a careful research assistant. Answer only in the requested JSON format."
CACHE_DIR = os.path.join(data.HERE, "cache", "llm")

Transport = Callable[[str, Optional[Dict[str, Any]]], Dict[str, Any]]  # body None -> GET


class ContextOverflow(RuntimeError):
    """The prompt plus the output budget does not fit in ``num_ctx``."""


class TokenCountMismatch(RuntimeError):
    """Ollama evaluated a different number of prompt tokens than were counted."""


def render_chatml(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def http_transport(base_url: str = "http://localhost:11434", timeout: float = 900) -> Transport:
    def call(path: str, body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if body is None:
            req = urllib.request.Request(base_url + path)
        else:
            req = urllib.request.Request(base_url + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    return call


def qwen_token_counter() -> Callable[[str], int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION)
    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != TOKENIZER_SHA256:
            raise RuntimeError("tokenizer.json does not match the pinned hash")
    tok = Tokenizer.from_file(path)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def cache_key(model_digest: str, prompt: str, options: Dict[str, Any], schema: Dict[str, Any]) -> str:
    payload = {"model_digest": model_digest, "prompt": prompt, "options": options, "schema": schema}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class LLMResult:
    key: str
    text: str
    parsed: Optional[Dict[str, Any]]   # None when the output failed schema validation
    error: Optional[str]
    prompt_tokens: int
    output_tokens: int
    wall_s: float                      # of the original generation, also when replayed
    done_reason: str
    cached: bool


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.cached_calls += r.cached
        self.prompt_tokens += r.prompt_tokens
        self.output_tokens += r.output_tokens
        self.wall_s += r.wall_s


@dataclass
class OllamaClient:
    model: str = DEFAULT_MODEL
    options: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OPTIONS))
    cache_dir: str = CACHE_DIR
    transport: Transport = field(default_factory=http_transport)
    count_tokens: Callable[[str], int] = field(default_factory=qwen_token_counter)
    offline: bool = False              # replay only: a cache miss raises instead of generating
    offline_digest: Optional[str] = None      # required offline: the model digest from a run manifest
    offline_quantization: str = ""
    offline_ollama_version: Optional[str] = None   # offline: the version recorded by the source run; None = unrecorded
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        tags = self.transport("/api/tags", None) if not self.offline else None
        if tags is not None:
            match = [m for m in tags["models"] if m["name"] == self.model]
            if not match:
                raise RuntimeError(f"model {self.model!r} is not installed in Ollama")
            self.digest = match[0]["digest"]
            self.quantization = match[0]["details"].get("quantization_level", "")
            self.ollama_version = self.transport("/api/version", None)["version"]
        else:
            if not self.offline_digest:
                raise ValueError("an offline client needs the model digest recorded in a run manifest")
            self.digest, self.quantization = self.offline_digest, self.offline_quantization
            # never the installed version: a replay reports what its source run recorded
            self.ollama_version = self.offline_ollama_version

    def describe(self) -> Dict[str, Any]:
        """Model identity for run manifests."""
        return {"model": self.model, "digest": self.digest, "quantization": self.quantization,
                "options": dict(self.options), "tokenizer": f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}",
                "ollama_version": self.ollama_version}

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def generate(self, user: str, schema: Type[BaseModel], system: str = DEFAULT_SYSTEM) -> LLMResult:
        prompt = render_chatml(system, user)
        json_schema = schema.model_json_schema()
        key = cache_key(self.digest, prompt, self.options, json_schema)
        path = self._path(key)
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            result = self._result(key, rec, schema, cached=True)
            self.usage.add(result)
            return result
        if self.offline:
            raise KeyError(f"cache miss in offline mode: {key}")

        n_prompt = self.count_tokens(prompt)
        budget = self.options["num_ctx"] - self.options["num_predict"]
        if n_prompt > budget:
            raise ContextOverflow(f"prompt has {n_prompt} tokens; budget is {budget} "
                                  f"(num_ctx {self.options['num_ctx']} - num_predict {self.options['num_predict']})")
        body = {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                "format": json_schema, "options": self.options, "keep_alive": "30m"}
        t0 = time.perf_counter()
        resp = self.transport("/api/generate", body)
        wall = time.perf_counter() - t0
        if resp["prompt_eval_count"] != n_prompt:
            raise TokenCountMismatch(f"counted {n_prompt} prompt tokens, Ollama evaluated {resp['prompt_eval_count']}")
        rec = {"key": key, "model": self.model, "digest": self.digest, "options": self.options,
               "schema": json_schema, "prompt": prompt, "response": resp["response"],
               "prompt_tokens": resp["prompt_eval_count"], "output_tokens": resp["eval_count"],
               "wall_s": wall, "done_reason": resp.get("done_reason", ""),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, path)
        result = self._result(key, rec, schema, cached=False)
        self.usage.add(result)
        return result

    @staticmethod
    def _result(key: str, rec: Dict[str, Any], schema: Type[BaseModel], cached: bool) -> LLMResult:
        try:
            parsed, error = schema.model_validate_json(rec["response"]).model_dump(), None
        except ValidationError as e:
            parsed, error = None, f"schema: {e.error_count()} error(s)"
        return LLMResult(key, rec["response"], parsed, error, rec["prompt_tokens"],
                         rec["output_tokens"], rec["wall_s"], rec["done_reason"], cached)
