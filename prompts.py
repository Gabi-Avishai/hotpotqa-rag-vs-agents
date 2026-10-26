"""Frozen prompt templates, output schema and paragraph renderer (PREREGISTRATION.md §1.4).

The template text must equal the preregistered text; ``tests/test_prompts.py`` checks it
against ``PREREGISTRATION.md``. ``artifact_hashes()`` gives the SHA-256 of every active
frozen artifact; the values recorded in the preregistration's amendment log must match.

Version 2 (amendment A1) is active for every system. Version 1 is kept unchanged as the
record of the original prompt-format check, with its hashes in ``V1_HASHES``.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Dict, List, Literal, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

import llm

ANSWER_V1 = """Answer the question using only the numbered paragraphs below.

Fill the JSON fields in this order:
- "rationale": at most two short sentences on how the evidence leads to the answer.
- "answer": the shortest answer that fully answers the question: a name, date, number,
  short phrase, or "yes"/"no". Not a full sentence.
- "citations": every sentence you relied on, as {"paragraph": P, "sentence": S} using the
  labels shown, e.g. [P2] sentence [1] is {"paragraph": 2, "sentence": 1}.
- "supported": true if the cited sentences are enough to answer the question; false if the
  paragraphs do not contain the needed information. When false, still give your best
  guess in "answer".

Paragraphs:
{paragraphs}

Question: {question}"""

CLOSED_BOOK_V1 = """Answer the question from your own knowledge. No documents are provided.

Fill the JSON fields in this order:
- "rationale": at most two short sentences.
- "answer": the shortest answer that fully answers the question: a name, date, number,
  short phrase, or "yes"/"no". Not a full sentence.
- "citations": always an empty list.
- "supported": true if you are confident the answer is correct, otherwise false. When
  false, still give your best guess in "answer".

Question: {question}"""


ANSWER_RULE_V2 = """- "answer": the shortest answer that fully answers the question: a name, date, number,
  short phrase, or "yes"/"no". Not a full sentence. Write it as ordinary plain text: never
  leave it empty, and never encode it as a function call, a type conversion or a labelled
  expression. If you are unsure, still give your best guess."""

_ANSWER_RULE_V1 = """- "answer": the shortest answer that fully answers the question: a name, date, number,
  short phrase, or "yes"/"no". Not a full sentence."""

ANSWER_V2 = ANSWER_V1.replace(_ANSWER_RULE_V1, ANSWER_RULE_V2)
CLOSED_BOOK_V2 = CLOSED_BOOK_V1.replace(_ANSWER_RULE_V1, ANSWER_RULE_V2)
assert ANSWER_V2 != ANSWER_V1 and CLOSED_BOOK_V2 != CLOSED_BOOK_V1


ASSESS_V1 = """Decide whether the numbered paragraphs below contain every fact needed to answer the
question. Do not answer the question.

Fill the JSON fields in this order:
- "justification": at most two short sentences on what the paragraphs establish and what,
  if anything, is still missing.
- "sufficient": true if the paragraphs contain every fact needed to answer the question;
  otherwise false.
- "missing_info": if not sufficient, the specific missing fact, naming the entity it is
  about (for example "the birth date of David Cronenberg"); otherwise "".
- "next_query": if not sufficient, a short search query for the missing fact that names
  the relevant entity found in the paragraphs; otherwise "".

Paragraphs:
{paragraphs}

Question: {question}"""

_QUERY_RULE_V1 = """- "next_query": if not sufficient, a short search query for the missing fact that names
  the relevant entity found in the paragraphs; otherwise ""."""
QUERY_RULE_V2 = """- "next_query": if not sufficient, a search query of at most 12 words that names the
  relevant entity found in the paragraphs; otherwise ""."""
ASSESS_V2 = ASSESS_V1.replace(_QUERY_RULE_V1, QUERY_RULE_V2)      # amendment A3: only this instruction
assert ASSESS_V2 != ASSESS_V1
ASSESS_V1_HASH = "4715e4ef90146e82715069d5d5de32352e0c9586e92867f2ee882a9d25dd209b"   # the v1 format check

MAX_QUERY_TOKENS = 12


# Phase 3 assess step. Ollama's schema can express only the field types, so the runtime
# contract (PREREGISTRATION.md §3.2) is enforced by the validator; a violation is an
# assess_error. No docstring: pydantic would copy it into the schema sent to the model.
class AssessOutput(BaseModel):
    justification: str
    sufficient: bool
    missing_info: str
    next_query: str

    @model_validator(mode="after")
    def _contract(self):
        from normalize import tok_v1
        mi, nq = self.missing_info.strip(), self.next_query.strip()
        if self.sufficient and (mi or nq):
            raise ValueError("sufficient=true requires empty missing_info and next_query")
        if not self.sufficient and not (mi and nq):
            raise ValueError("sufficient=false requires non-empty missing_info and next_query")
        if len(tok_v1(self.next_query)) > MAX_QUERY_TOKENS:
            raise ValueError(f"next_query has more than {MAX_QUERY_TOKENS} tok_v1 tokens")
        return self


# ---------------------------------------------------------------- Phase 4 (multi-agent) interfaces

PLANNER_V1 = """Break the question into the evidence that must be found to answer it. Do not answer the
question.

Fill the JSON field:
- "requirements": one or two short statements of the facts that must be found, in the
  order they must be found. Use two only when the second fact depends on an entity the
  first fact identifies, or when the question compares two named entities.

Question: {question}"""

RESEARCHER_V1 = """You are collecting evidence for one requirement of a question. Do not answer the
question.

Requirement: {requirement}

Accepted evidence (already collected; read only):
{accepted}

Candidate paragraphs:
{candidates}

Fill the JSON fields in this order:
- "accepted": the numbers of the candidate paragraphs that contain evidence for this
  requirement, for example [1, 4]; [] if none.
- "finding": at most two short sentences stating what the evidence establishes for this
  requirement.
{next_query_rule}

Question: {question}"""

NEXT_QUERY_RULE_WITH = """- "next_query": a search query of at most 12 words for the next requirement ("{next_requirement}"), naming any entity the evidence identifies."""
NEXT_QUERY_RULE_NONE = """- "next_query": "" (there is no further requirement)."""

VERIFIER_V1 = """Check whether the paragraphs below support every requirement of the question. Judge only
from the paragraphs. Do not answer the question.

Requirements:
{requirements}

Fill the JSON fields in this order:
- "requirements": one entry per requirement, in order, each with "supported" (true or
  false) and "references": the [paragraph, sentence] evidence for a supported requirement
  as {"paragraph": P, "sentence": S}, or [] if it is not supported.
- "sufficient": true only if every requirement is supported.
- "corrective_requirement": if not sufficient, the number of one unsupported requirement;
  otherwise 0.
- "corrective_query": if not sufficient, a search query of at most 12 words for that
  requirement, naming any entity found in the paragraphs; otherwise "".

Paragraphs:
{paragraphs}

Question: {question}"""

_VERIFIER_REQ_RULE_V1 = """- "requirements": one entry per requirement, in order, each with "supported" (true or
  false) and "references": the [paragraph, sentence] evidence for a supported requirement
  as {"paragraph": P, "sentence": S}, or [] if it is not supported."""
VERIFIER_REQ_RULE_V2 = """- "requirements": exactly one entry per requirement, in order, each with "supported"
  (true or false) and "references": the [paragraph, sentence] evidence for a supported
  requirement as {"paragraph": P, "sentence": S}, using the labels shown (paragraph [P1]
  is 1; sentences are numbered as shown), or [] if it is not supported (an unsupported
  requirement has no references)."""
_VERIFIER_QUERY_RULE_V1 = """- "corrective_query": if not sufficient, a search query of at most 12 words for that
  requirement, naming any entity found in the paragraphs; otherwise ""."""
VERIFIER_QUERY_RULE_V2 = """- "corrective_query": if not sufficient, a search query of at most 12 words for that
  requirement, naming any entity found in the paragraphs (never empty when not
  sufficient); otherwise ""."""
VERIFIER_V2 = (VERIFIER_V1.replace(_VERIFIER_REQ_RULE_V1, VERIFIER_REQ_RULE_V2)
               .replace(_VERIFIER_QUERY_RULE_V1, VERIFIER_QUERY_RULE_V2))      # amendment A4: format only
assert VERIFIER_V2.count("exactly one entry") == 1 and "never empty" in VERIFIER_V2
VERIFIER_TEMPLATE = VERIFIER_V2
VERIFIER_V1_HASH = "d1166226548c13d26ce9ad52af4294e23b9a8848f352efd06eb9c57cbefef8eb"   # the ma_check_v1 prompt

VERIFIER_V3 = """Check whether the paragraphs support each requirement of the question. Judge only from
the paragraphs and use no outside knowledge: a fact that no displayed sentence states is
unsupported, even if you know it to be true. Do not answer the question.

Question: {question}

There are {n} requirements; return exactly {n} entries in "requirements", one per
requirement, in the order R1, R2, ...:
{requirements}

Requirement numbers and paragraph numbers are unrelated: any paragraph may support any
requirement, and one paragraph may support several requirements.

Work in this order:
1. Judge each requirement on its own, scanning every paragraph for sentences that state
   it.
2. If such sentences exist, the requirement is supported: "supported": true, and
   "references" lists them as {"paragraph": P, "sentence": S}. Paragraph numbers start at
   1 ([P1] is 1); sentence numbers start at 0, as shown in brackets. Otherwise
   "supported": false and "references": [].
3. Only then fill the control fields:
   - "sufficient": true only if every requirement is supported;
   - "corrective_requirement": if not sufficient, the lowest-numbered unsupported
     requirement (never a supported one); otherwise 0;
   - "corrective_query": if not sufficient, a search query of at most 12 words for that
     requirement, naming any entity found in the paragraphs (never empty when not
     sufficient); otherwise "".

Example 1 (invented).
Requirements: [R1] The river that flows through Velmora. [R2] The length of that river.
Paragraphs:
[P1] Ostrel River
  [0] The Ostrel River is 412 kilometres long.
[P2] Velmora
  [0] Velmora is a town in northern Kasdia.
  [1] The Ostrel River flows through Velmora.
Output: {"requirements": [{"supported": true, "references": [{"paragraph": 2, "sentence": 1}]}, {"supported": true, "references": [{"paragraph": 1, "sentence": 0}]}], "sufficient": true, "corrective_requirement": 0, "corrective_query": ""}

Example 2 (invented).
Requirements: [R1] The architect of the Dunmere Library. [R2] The birth year of that architect.
Paragraphs:
[P1] Dunmere Library
  [0] The Dunmere Library opened in 1931.
  [1] It was designed by the architect Hollis Brandt.
Output: {"requirements": [{"supported": true, "references": [{"paragraph": 1, "sentence": 1}]}, {"supported": false, "references": []}], "sufficient": false, "corrective_requirement": 2, "corrective_query": "Hollis Brandt architect birth year"}

Paragraphs:
{paragraphs}

Checklist: return exactly {n} entries, judging each requirement against every paragraph:
{checklist}"""
VERIFIERS = {"v2": VERIFIER_V2, "v3": VERIFIER_V3}     # A6: v3 is the one verifier revision

MAX_REQUIREMENT_TOKENS = 25


def _ntok(text: str) -> int:
    from normalize import tok_v1
    return len(tok_v1(text))


# Static contracts as validators; contracts that depend on what was displayed (paragraph
# and sentence ranges, the number of requirements, whether a next requirement exists) are
# checked by the flow. No docstrings: pydantic would copy them into the schema.
class PlannerOutput(BaseModel):
    requirements: List[str]

    @model_validator(mode="after")
    def _contract(self):
        if not 1 <= len(self.requirements) <= 2:
            raise ValueError("one or two requirements")
        for r in self.requirements:
            if not r.strip() or _ntok(r) > MAX_REQUIREMENT_TOKENS:
                raise ValueError("each requirement is non-empty and at most 25 tok_v1 tokens")
        return self


class ResearcherOutput(BaseModel):
    accepted: List[int]
    finding: str
    next_query: str

    @model_validator(mode="after")
    def _contract(self):
        if len(set(self.accepted)) != len(self.accepted):
            raise ValueError("accepted numbers repeat")
        if _ntok(self.next_query) > MAX_QUERY_TOKENS:
            raise ValueError("next_query has more than 12 tok_v1 tokens")
        return self


class RequirementCheck(BaseModel):
    supported: bool
    references: List["CitationRef"]


class VerifierOutput(BaseModel):
    requirements: List[RequirementCheck]
    sufficient: bool
    corrective_requirement: int
    corrective_query: str


def verifier_contract_v2(parsed: Dict, n_requirements: int, sentence_counts: Sequence[int]):
    """Amendment A4.1. Keeps the LLM's judgments (per-requirement ``supported`` and the
    requested correction), derives control state in code, and returns
    (normalized output or None, normalization notes, hard-failure reason or None)."""
    notes: List[str] = []
    reqs = parsed["requirements"]
    if len(reqs) != n_requirements:
        return None, notes, "one entry per requirement required"
    norm = []
    for i, c in enumerate(reqs, 1):
        refs = c["references"]
        if not c["supported"]:
            if refs:
                notes.append(f"ignored {len(refs)} reference(s) on unsupported requirement {i}")
            norm.append({"supported": False, "references": []})
            continue
        if not refs:
            return None, notes, "supported=true requires at least one reference"
        for ref in refs:
            p, t = ref["paragraph"], ref["sentence"]
            if not 1 <= p <= len(sentence_counts) or not 0 <= t < sentence_counts[p - 1]:
                return None, notes, "invalid reference on a supported requirement"
        norm.append({"supported": True, "references": refs})
    sufficient = all(c["supported"] for c in norm)
    if parsed["sufficient"] != sufficient:
        notes.append(f"sufficient recomputed: {parsed['sufficient']} -> {sufficient}")
    if sufficient:
        if parsed["corrective_requirement"] != 0 or parsed["corrective_query"].strip():
            notes.append("corrective fields cleared (computed sufficient)")
        cr, cq = 0, ""
    else:
        unsupported = [i for i, c in enumerate(norm, 1) if not c["supported"]]
        cr, cq = parsed["corrective_requirement"], parsed["corrective_query"]
        if cr not in unsupported:
            return None, notes, "corrective_requirement must be an unsupported requirement"
        if not cq.strip() or _ntok(cq) > MAX_QUERY_TOKENS:
            return None, notes, "corrective_query must be non-empty and at most 12 tok_v1 tokens"
    return {"requirements": norm, "sufficient": sufficient, "corrective_requirement": cr,
            "corrective_query": cq}, notes, None


class CitationRef(BaseModel):
    paragraph: int
    sentence: int


class AnswerOutputV1(BaseModel):
    model_config = ConfigDict(title="AnswerOutput")   # the v1 schema as originally sent
    rationale: str
    answer: str
    citations: List[CitationRef]
    supported: bool


# Schema v2: the answer must contain a non-whitespace character. ``minLength`` is enforced
# by Ollama's constrained decoding; the validator rejects a whitespace-only answer, which
# then counts as a schema error (Ollama rejects regex ``pattern``). No docstring: pydantic
# would copy it into the JSON schema that is sent to the model.
class AnswerOutput(BaseModel):
    rationale: str
    answer: str = Field(min_length=1)
    citations: List[CitationRef]
    supported: bool

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer is blank")
        return v


def render_v1(paragraphs: Sequence[Tuple[str, Sequence[str]]]) -> str:
    """``[(title, sentences)]`` in display order -> numbered paragraph block."""
    blocks = []
    for p, (title, sentences) in enumerate(paragraphs, 1):
        lines = [f"[P{p}] {title}"] + [f"  [{i}] {s.strip()}" for i, s in enumerate(sentences)]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _fill(template: str, **values: str) -> str:
    """Substitute ``{name}`` slots once each, left to right, so text inside a value (e.g.
    a paragraph containing "{question}") is never substituted again. The templates' other
    braces (the JSON examples) are left alone."""
    out, rest = [], template
    for name, value in values.items():
        head, rest = rest.split("{" + name + "}", 1)
        out += [head, value]
    return "".join(out) + rest


ANSWER_TEMPLATE, CLOSED_BOOK_TEMPLATE = ANSWER_V2, CLOSED_BOOK_V2   # active versions


def answer_prompt(question: str, paragraphs: Sequence[Tuple[str, Sequence[str]]],
                  template: str = ANSWER_TEMPLATE) -> str:
    return _fill(template, paragraphs=render_v1(paragraphs), question=question)


def closed_book_prompt(question: str, template: str = CLOSED_BOOK_TEMPLATE) -> str:
    return _fill(template, question=question)


ASSESS_TEMPLATE = ASSESS_V2      # active version (amendment A3)


def assess_prompt(question: str, paragraphs: Sequence[Tuple[str, Sequence[str]]],
                  template: str = ASSESS_TEMPLATE) -> str:
    return _fill(template, paragraphs=render_v1(paragraphs), question=question)


def assess_hashes() -> Dict[str, str]:
    """Hashes of the active Phase 3 assess interface (frozen before the first dev call).
    The schema and contract are unchanged from v1."""
    return {"assess_v2": _sha(ASSESS_V2), "assess_schema_v1": _schema_hash(AssessOutput),
            "assess_contract_v1": _sha(inspect.getsource(AssessOutput._contract))}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_hash(model) -> str:
    return _sha(json.dumps(model.model_json_schema(), sort_keys=True))


def artifact_hashes() -> Dict[str, str]:
    """Hashes of the active (v2) frozen artifacts."""
    return {
        "answer_v2": _sha(ANSWER_V2),
        "closed_book_v2": _sha(CLOSED_BOOK_V2),
        "schema_v2": _schema_hash(AnswerOutput),
        "system_prompt": _sha(llm.DEFAULT_SYSTEM),
        "render_v1": _sha(inspect.getsource(render_v1)),
        "answer_validator_v2": _sha(inspect.getsource(AnswerOutput._not_blank)),
    }


# Hashes of version 1 as used in the original prompt-format check (2026-09-29).
V1_HASHES = {
    "answer_v1": "3f490daca95bb51c060f46ab8e8744a3f1e91913253ea371b6732195f647a665",
    "closed_book_v1": "90e09ebc07c35ec55d10f599e41ae7a9808b7dce6cc76507a4c408d57d43cb9a",
    "schema_v1": "b289ec00a9f3b73e012daa2e35c3c385faf24ea313c80abe2686665d8a6974fb",
    "system_prompt": "16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172",
    "render_v1": "935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6",
}


def v1_hashes_now() -> Dict[str, str]:
    """Recomputed v1 hashes, to check that version 1 is preserved unchanged."""
    return {"answer_v1": _sha(ANSWER_V1), "closed_book_v1": _sha(CLOSED_BOOK_V1),
            "schema_v1": _schema_hash(AnswerOutputV1), "system_prompt": _sha(llm.DEFAULT_SYSTEM),
            "render_v1": _sha(inspect.getsource(render_v1))}


RequirementCheck.model_rebuild()
VerifierOutput.model_rebuild()


def render_labeled(paragraphs: Sequence[Tuple[str, Sequence[str]]], label: str) -> str:
    """``render_v1`` with ``[A#]`` (or another label) in place of ``[P#]``."""
    out = render_v1(paragraphs)
    import re
    return re.sub(r"^\[P(\d+)\] ", lambda m: f"[{label}{m.group(1)}] ", out, flags=re.M)


def planner_prompt(question: str) -> str:
    return _fill(PLANNER_V1, question=question)


def researcher_prompt(question: str, requirement: str, accepted, candidates, next_requirement=None) -> str:
    rule = (_fill(NEXT_QUERY_RULE_WITH, next_requirement=next_requirement) if next_requirement is not None
            else NEXT_QUERY_RULE_NONE)
    acc = render_labeled(accepted, "A") if accepted else "(none)"
    return _fill(RESEARCHER_V1, requirement=requirement, accepted=acc, candidates=render_v1(candidates),
                 next_query_rule=rule, question=question)


def verifier_prompt(question: str, requirements: Sequence[str], paragraphs, version: str = "v2") -> str:
    if version == "v2":
        reqs = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
        return _fill(VERIFIER_TEMPLATE, requirements=reqs, paragraphs=render_v1(paragraphs), question=question)
    if version != "v3":
        raise ValueError(f"unknown verifier version {version!r}")
    reqs = "\n".join(f"[R{i}] {r}" for i, r in enumerate(requirements, 1))
    template = VERIFIER_V3.replace("{n}", str(len(requirements)))     # a count, never model or corpus text
    return _fill(template, question=question, requirements=reqs, paragraphs=render_v1(paragraphs), checklist=reqs)


def multi_agent_hashes() -> Dict[str, str]:
    """Hashes of the Phase 4 interfaces (frozen before the first dev call)."""
    src = lambda f: _sha(inspect.getsource(f))
    return {"planner_v1": _sha(PLANNER_V1), "researcher_v1": _sha(RESEARCHER_V1),
            "next_query_rule_with": _sha(NEXT_QUERY_RULE_WITH), "next_query_rule_none": _sha(NEXT_QUERY_RULE_NONE),
            "verifier_v2": _sha(VERIFIER_V2), "verifier_v3": _sha(VERIFIER_V3),
            "planner_schema_v1": _schema_hash(PlannerOutput), "researcher_schema_v1": _schema_hash(ResearcherOutput),
            "verifier_schema_v1": _schema_hash(VerifierOutput),
            "planner_contract_v1": src(PlannerOutput._contract), "researcher_contract_v1": src(ResearcherOutput._contract),
            "verifier_contract_v2": src(verifier_contract_v2), "render_labeled": src(render_labeled)}


# ---------------------------------------------------------------- Phase 4b (PREREGISTRATION.md Phase 4b)

RESEARCHER_V2 = """You are selecting evidence for one requirement of a question. Do not answer the question.

Question: {question}

Requirement: {requirement}

Accepted evidence (already selected; read only):
{accepted}

Candidate paragraphs:
{candidates}

Decide for every candidate paragraph, [P1] to [P{n}], whether to keep it for this
requirement. Keep a paragraph if it states the requirement's fact, or names or describes
an entity the requirement refers to. When unsure, keep it: a paragraph kept by mistake
costs little, but a paragraph dropped by mistake can make the question unanswerable.

Fill the JSON fields in this order:
- "P1" to "P{n}": true to keep the paragraph, false to drop it.
{next_query_rule}"""

RESEARCHER_V3 = """You are selecting evidence for one requirement of a question. Do not answer the question.

Question: {question}

Requirement: {requirement}

Accepted evidence (already selected; read only):
{accepted}

Candidate paragraphs:
{candidates}

Decide for every candidate paragraph, [P1] to [P{n}], whether to keep it for this
requirement. Judge every candidate independently: do not drop a candidate merely because
the accepted evidence or another candidate already appears sufficient. Keep a paragraph if
it states the requirement's fact, or names or describes an entity the requirement refers
to. Keep every paragraph that could identify, verify, or provide an entity or fact this
requirement needs, including a paragraph titled with such an entity and both sides of a
comparison or a multi-hop chain. When unsure, keep it: a paragraph kept by mistake costs
little, but a paragraph dropped by mistake can make the question unanswerable.

Fill the JSON fields in this order:
- "P1" to "P{n}": true to keep the paragraph, false to drop it.
{next_query_rule}"""      # amendment A8: the one permitted researcher revision
RESEARCHERS_4B = {"v2": RESEARCHER_V2, "v3": RESEARCHER_V3}

REQUIREMENT_VERIFIER_V1 = """Check whether the paragraphs below state one requirement of a question. Judge only from
the displayed sentences and use no outside knowledge: a fact that no displayed sentence
states is not supported, even if you know it to be true. Do not answer the question.

Question: {question}

All requirements (for context):
{requirements}

Requirement to check now: {requirement}

Paragraphs:
{paragraphs}

Fill the JSON fields in this order:
- "references": the labels of up to 3 displayed sentences that state the requirement to
  check, written as "P2:S1" for paragraph [P2], sentence [1]; [] if no displayed sentence
  states it.
- "supported": true only if the references state the requirement.
- "query": if it is not supported, a search query of at most 12 words that could find the
  missing fact, naming any entity found in the paragraphs; otherwise ""."""

REFERENCE_LIMIT = 3


def researcher_4b_prompt(question: str, requirement: str, accepted, candidates, next_requirement=None,
                         version: str = "v3") -> str:
    """``researcher_v2`` (C3) or ``researcher_v3`` (amendment A8; C4 onwards)."""
    rule = _fill(NEXT_QUERY_RULE_WITH, next_requirement=next_requirement) if next_requirement is not None else ""
    acc = render_labeled(accepted, "A") if accepted else "(none)"
    template = RESEARCHERS_4B[version].replace("{n}", str(len(candidates)))     # a count, never model or corpus text
    return _fill(template, question=question, requirement=requirement, accepted=acc,
                 candidates=render_v1(candidates), next_query_rule=rule)


def requirement_verifier_prompt(question: str, requirements: Sequence[str], i: int, paragraphs) -> str:
    """The check of requirement ``i`` (1-based) against the displayed paragraphs."""
    reqs = "\n".join(f"{k}. {r}" for k, r in enumerate(requirements, 1))
    return _fill(REQUIREMENT_VERIFIER_V1, question=question, requirements=reqs, requirement=requirements[i - 1],
                 paragraphs=render_v1(paragraphs))


def researcher_v2_schema(n_candidates: int, has_next: bool):
    """One required boolean per candidate ("P1".."Pn"), plus "next_query" only when a next
    requirement exists; no other keys (additionalProperties: false)."""
    fields = {f"P{i}": (bool, ...) for i in range(1, n_candidates + 1)}
    if has_next:
        fields["next_query"] = (str, ...)
    return create_model("ResearcherV2Output", __config__=ConfigDict(extra="forbid"), **fields)


def reference_labels(paragraphs) -> List[str]:
    """The valid ``P<p>:S<s>`` labels of the displayed sentences, in display order."""
    return [f"P{p}:S{s}" for p, (_, sents) in enumerate(paragraphs, 1) for s in range(len(sents))]


def requirement_verifier_schema(labels: Sequence[str]):
    """References restricted to the displayed labels (enum, at most 3, uniqueItems stated);
    Ollama does not enforce uniqueItems, and validation here does not either, so duplicates
    reach the deterministic normalization rather than becoming schema failures."""
    label_type = Literal[tuple(labels)]
    return create_model(
        "RequirementVerifierOutput", __config__=ConfigDict(extra="forbid"),
        references=(List[label_type], Field(max_length=REFERENCE_LIMIT, json_schema_extra={"uniqueItems": True})),
        supported=(bool, ...), query=(str, ...))


def multi_agent_4b_hashes() -> Dict[str, str]:
    """Hashes of the Phase 4b interfaces (frozen before the first dev call)."""
    src = lambda f: _sha(inspect.getsource(f))
    return {"researcher_v2": _sha(RESEARCHER_V2), "researcher_v3": _sha(RESEARCHER_V3),
            "requirement_verifier_v1": _sha(REQUIREMENT_VERIFIER_V1),
            "next_query_rule_with": _sha(NEXT_QUERY_RULE_WITH), "planner_v1": _sha(PLANNER_V1),
            "planner_schema_v1": _schema_hash(PlannerOutput), "planner_contract_v1": src(PlannerOutput._contract),
            "researcher_4b_prompt": src(researcher_4b_prompt), "requirement_verifier_prompt": src(requirement_verifier_prompt),
            "researcher_v2_schema": src(researcher_v2_schema), "requirement_verifier_schema": src(requirement_verifier_schema),
            "reference_labels": src(reference_labels), "render_labeled": src(render_labeled)}
