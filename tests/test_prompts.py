"""Frozen prompts: text equals the preregistration; rendering and filling are safe."""
import json
import os
import re

import pytest

import prompts
import data


def _prereg_block(label):
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    m = re.search(re.escape(label) + r".*?```text\n(.*?)\n```", text, re.S)
    return m.group(1)


def test_templates_equal_preregistered_text():
    assert prompts.ANSWER_V1 == _prereg_block("**Answer prompt** (`answer_v1`)")
    assert prompts.CLOSED_BOOK_V1 == _prereg_block("**Closed-book prompt** (`closed_book_v1`)")


def test_render_numbers_paragraphs_and_keeps_empty_sentences():
    out = prompts.render_v1([("A", [" First. ", ""]), ("B", ["Second."])])
    assert out == "[P1] A\n  [0] First.\n  [1] \n[P2] B\n  [0] Second."


def test_fill_does_not_resubstitute_inside_values():
    p = prompts.answer_prompt("Q {paragraphs}?", [("T", ["text with {question} inside"])])
    assert "text with {question} inside" in p and p.endswith("Question: Q {paragraphs}?")
    assert '{"paragraph": 2, "sentence": 1}' in p       # JSON example braces are untouched


def test_schema_field_order_is_generation_order():
    assert list(prompts.AnswerOutput.model_json_schema()["properties"]) == ["rationale", "answer", "citations", "supported"]


def test_hashes_cover_all_frozen_artifacts():
    h = prompts.artifact_hashes()
    assert set(h) == {"answer_v2", "closed_book_v2", "schema_v2", "system_prompt", "render_v1", "answer_validator_v2"}
    assert all(len(v) == 64 for v in h.values()) and prompts.artifact_hashes() == h


def test_v1_is_preserved_unchanged():
    assert prompts.v1_hashes_now() == prompts.V1_HASHES


def test_v2_answer_rule_equals_amendment_and_only_that_rule_changed():
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    block = re.search(r"### A1.*?```text\n(.*?)\n  ```", text, re.S).group(1)
    rule = "\n".join(line[2:] for line in block.splitlines())
    assert rule == prompts.ANSWER_RULE_V2
    assert prompts.ANSWER_V2 == prompts.ANSWER_V1.replace(prompts._ANSWER_RULE_V1, rule)
    assert prompts.CLOSED_BOOK_V2 == prompts.CLOSED_BOOK_V1.replace(prompts._ANSWER_RULE_V1, rule)
    assert prompts.answer_prompt("q", []).startswith(prompts.ANSWER_V2.split("{paragraphs}")[0])


def test_schema_v2_rejects_blank_answers():
    import pydantic
    ok = '{"rationale": "", "answer": "x", "citations": [], "supported": true}'
    assert prompts.AnswerOutput.model_validate_json(ok).answer == "x"
    for blank in ('""', '" "', '"\\n\\t"'):
        with pytest.raises(pydantic.ValidationError):
            prompts.AnswerOutput.model_validate_json(ok.replace('"x"', blank))
    assert prompts.AnswerOutput.model_json_schema()["properties"]["answer"]["minLength"] == 1
    assert "pattern" not in json.dumps(prompts.AnswerOutput.model_json_schema())


def test_no_description_text_is_sent_in_schemas():
    for model in (prompts.AnswerOutput, prompts.AnswerOutputV1, prompts.CitationRef):
        assert "description" not in json.dumps(model.model_json_schema())


V2_FROZEN = {
    "answer_v2": "0a2aa59c597d9730ed73ea358b22cd76787cde9910e12888397076b7bb003f72",
    "closed_book_v2": "a89deeb1d5aafd0fccafbcb44a10e47067f610bef75dcf8bcbfa878e9cdbcb79",
    "schema_v2": "c29af85a08782e161cae0ab7e4eb847e19866b4bc7910d84640c30d12ab67847",
    "answer_validator_v2": "252c82c0f31ab1013af85beeaa777b8eab0a5ffa2ba4dadec1e9961eb2d4910c",
    "system_prompt": "16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172",
    "render_v1": "935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6",
}


def test_v2_matches_frozen_hashes_recorded_in_preregistration():
    assert prompts.artifact_hashes() == V2_FROZEN
    text = open(os.path.join(data.HERE, "PREREGISTRATION.md")).read()
    assert all(h in text for h in V2_FROZEN.values())
