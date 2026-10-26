"""LLM-client tests: mocked transport for every behaviour, plus live Ollama smoke tests."""
import json
import urllib.request

import pytest
from pydantic import BaseModel

import llm


class Answer(BaseModel):
    answer: str


class AnswerWithFlag(BaseModel):
    answer: str
    supported: bool


class FakeOllama:
    """Records calls; answers /api/generate with a fixed response and a configurable count."""

    def __init__(self, response='{"answer": "Paris"}', count_offset=0, digest="sha256:aaa"):
        self.calls, self.response, self.offset, self.digest = [], response, count_offset, digest

    def __call__(self, path, body):
        self.calls.append((path, body))
        if path == "/api/tags":
            return {"models": [{"name": "qwen2.5:7b", "digest": self.digest,
                                "details": {"quantization_level": "Q4_K_M"}}]}
        if path == "/api/version":
            return {"version": "0.0.test"}
        return {"response": self.response, "prompt_eval_count": len(body["prompt"].split()) + self.offset,
                "eval_count": 5, "done_reason": "stop"}


def words(text):  # stand-in tokenizer, consistent with FakeOllama
    return len(text.split())


@pytest.fixture
def make(tmp_path):
    def _make(fake=None, **kw):
        fake = fake or FakeOllama()
        return llm.OllamaClient(cache_dir=str(tmp_path), transport=fake, count_tokens=words, **kw), fake
    return _make


def generations(fake):
    return [c for c in fake.calls if c[0] == "/api/generate"]


def test_render_chatml_matches_ollama_template():
    assert llm.render_chatml("S", "U") == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")


def test_generate_parses_counts_and_sends_raw_schema(make):
    client, fake = make()
    r = client.generate("capital of France?", Answer)
    assert r.parsed == {"answer": "Paris"} and r.error is None and not r.cached
    body = generations(fake)[0][1]
    assert body["raw"] is True and body["format"] == Answer.model_json_schema()
    assert body["options"] == llm.DEFAULT_OPTIONS
    assert client.usage.calls == 1 and client.usage.output_tokens == 5


def test_cache_replays_without_calling_ollama(make):
    client, fake = make()
    first = client.generate("q", Answer)
    second = client.generate("q", Answer)
    assert len(generations(fake)) == 1
    assert second.cached and second.text == first.text and second.key == first.key
    assert second.wall_s == first.wall_s          # original generation time is kept
    assert client.usage.calls == 2 and client.usage.cached_calls == 1


def test_cache_key_changes_with_prompt_schema_options_and_digest():
    base = dict(model_digest="d1", prompt="p", options={"temperature": 0.0, "num_ctx": 8192},
                schema=Answer.model_json_schema())
    k = llm.cache_key(**base)
    assert llm.cache_key(**base) == k
    for change in ({"prompt": "p2"}, {"schema": AnswerWithFlag.model_json_schema()},
                   {"options": {"temperature": 0.0, "num_ctx": 4096}}, {"model_digest": "d2"}):
        assert llm.cache_key(**{**base, **change}) != k, change


def test_client_misses_cache_when_system_options_or_digest_change(make, tmp_path):
    client, fake = make()
    client.generate("q", Answer)
    client.generate("q", Answer, system="another system prompt")
    client.generate("q", AnswerWithFlag)
    other, fake2 = make(FakeOllama(digest="sha256:bbb"))
    other.generate("q", Answer)
    changed, fake3 = make(options={**llm.DEFAULT_OPTIONS, "seed": 1})
    changed.generate("q", Answer)
    assert len(generations(fake)) == 3 and len(generations(fake2)) == 1 and len(generations(fake3)) == 1


def test_context_overflow_raises_before_generation(make):
    client, fake = make(options={**llm.DEFAULT_OPTIONS, "num_ctx": 40, "num_predict": 10})
    client.generate("short", Answer)                  # fits
    with pytest.raises(llm.ContextOverflow):
        client.generate("word " * 40, Answer)        # 40 + template words > 30
    assert len(generations(fake)) == 1                # the oversized prompt was never sent


def test_token_count_mismatch_raises_and_is_not_cached(make, tmp_path):
    client, fake = make(FakeOllama(count_offset=-3))
    with pytest.raises(llm.TokenCountMismatch):
        client.generate("q", Answer)
    assert not list(tmp_path.rglob("*.json"))


def test_schema_violation_is_recorded_not_retried(make):
    client, fake = make(FakeOllama(response='{"answer": 3'))
    r = client.generate("q", Answer)
    assert r.parsed is None and r.error.startswith("schema")
    assert len(generations(fake)) == 1


def test_offline_mode_replays_or_raises(make, tmp_path):
    client, _ = make()
    client.generate("q", Answer)
    with pytest.raises(ValueError, match="digest"):
        llm.OllamaClient(cache_dir=str(tmp_path), transport=None, count_tokens=words, offline=True)
    replay = llm.OllamaClient(cache_dir=str(tmp_path), transport=None, count_tokens=words, offline=True,
                              offline_digest=client.digest)
    assert replay.generate("q", Answer).cached
    with pytest.raises(KeyError):
        replay.generate("unseen", Answer)


def test_missing_model_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not installed"):
        llm.OllamaClient(model="nope:1b", cache_dir=str(tmp_path), transport=FakeOllama(), count_tokens=words)


# ---------------------------------------------------------------- live

def _ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


live = pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running")


@pytest.mark.live
@live
def test_live_smoke(tmp_path):
    client = llm.OllamaClient(cache_dir=str(tmp_path))
    assert client.digest.startswith("845dbda0ea48") and client.quantization == "Q4_K_M"
    r = client.generate("What is the capital of France? Reply with the city only.", Answer)
    assert r.error is None and "paris" in r.parsed["answer"].lower()
    assert r.done_reason == "stop" and not r.cached
    assert client.generate("What is the capital of France? Reply with the city only.", Answer).cached


@pytest.mark.live
@live
def test_live_ollama_truncates_silently_and_the_check_prevents_it(tmp_path):
    """Why the pre-generation check exists: sent directly with a small num_ctx, Ollama
    evaluates fewer tokens than the prompt has and raises no error."""
    count = llm.qwen_token_counter()
    prompt = llm.render_chatml(llm.DEFAULT_SYSTEM, "List these numbers back: " + " ".join(map(str, range(300))))
    n = count(prompt)
    resp = llm.http_transport()("/api/generate", {"model": "qwen2.5:7b", "prompt": prompt, "raw": True,
                                                  "stream": False, "options": {"num_ctx": 256, "num_predict": 8,
                                                                               "temperature": 0.0}})
    assert n > 256 and resp["prompt_eval_count"] < n   # silently cut
    client = llm.OllamaClient(cache_dir=str(tmp_path), options={**llm.DEFAULT_OPTIONS, "num_ctx": 256, "num_predict": 8})
    with pytest.raises(llm.ContextOverflow):
        client.generate("List these numbers back: " + " ".join(map(str, range(300))), Answer)
