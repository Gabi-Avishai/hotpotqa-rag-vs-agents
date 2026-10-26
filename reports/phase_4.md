# Phase 4 report — multi-agent investigation

**Phase:** 4, multi-agent investigation.

**Objective:** test whether splitting the investigation into specialised roles adds
answer quality beyond the frozen retriever R = D3 and the Phase 3 single agent, and at
what cost. The roles are planner, researchers, verifier and synthesizer.
(`PREREGISTRATION.md` Phase 4, amendments A4 to A6, and the Phase 4 freeze.)

**Result:** **the multi-agent pipeline compressed the final answer context from 10–16
paragraphs to 3.69, but the compression was unsafe: Joint Recall fell to 0.67.** No
answer-quality improvement over D3 RAG or the single agent was established, while median
LLM time increased to 111 seconds. On `dev` (n = 100):

- **Answer F1:** 0.638, against R's 0.629: +0.009 [−0.072, +0.091]. The observed F1 was
  similar, and no clear difference was detected.
- **Final-context Joint Recall:** 0.67, against R's 0.84 (−0.17 [−0.27, −0.07]).
- **Evidence funnel:** 0.84 after search 1 → 0.96 over the whole search pool → 0.67 after
  researcher selection.

## Hypotheses (preregistered, exploratory, `dev`, n = 100)

| Id | Prediction | Estimate [95% CI where applicable] | Result |
|---|---|---|---|
| H4-1 | Final context < 15.95 paragraphs **and** final-context Joint Recall ≥ 0.81 | 3.69 paragraphs; JR 0.67; vs R −0.170 [−0.270, −0.070] | Fails (the context is smaller, but JR is below 0.81) |
| H4-2 | Multi − single F1 > 0 | +0.029 [−0.047, +0.106] | Holds on its point-estimate criterion; the difference is not statistically established |
| H4-3 | First verification lacking a gold paragraph judged not sufficient, at least 0.60 (proxy) | 30/33 = 0.91; 12 more ended in verifier errors | Holds, mostly through blanket caution (see below) |
| H4-4 | Corrective search when the first verification already had both gold paragraphs, at most 0.40 | 30/55 = 0.55, where 19 verifier errors could not trigger one; among valid verdicts 30/36 = 0.83 | Fails |
| H4-5 | Self-reported `supported = true` when the final context lacks a gold paragraph (proxy): lower than the single agent | Multi-agent 22/33 = 0.67; single agent 4/7 = 0.57 | Fails (not lower; the single agent's n is 7) |
| H4-6 | Median LLM time per question, multi / single ≥ 1.0 | 110.8 s / 73.4 s = 1.51 | Holds |

**H4-3 and H4-5 use missing gold paragraphs as a proxy for missing support, not proof of
it.** Alternative evidence, incomplete annotations, model knowledge or guessing can
explain some answers.

## What was built

- **`multi_agent.py`:** a LangGraph `StateGraph` with the nodes `plan`, `search`,
  `research`, `verify`, `csearch`, `recheck` and `answer`.
  - **Search:** search 1 is the question verbatim and returns D3's top 10, exactly R's
    context. Later searches add 5 unseen paragraphs each.
  - **Researchers** see the accumulated candidate pool. Accepted ids are globally unique,
    in order of first acceptance.
  - **Verifier:** judges against the accepted paragraphs only, with grounded
    `[paragraph, sentence]` references. If evidence is missing, it triggers one corrective
    search, a corrective researcher and a recheck.
  - **Budget:** at most 3 searches, 20 paragraphs and 7 LLM calls.
  - **Overflow rule:** evidence-preserving (candidates first, then accepted).
- **`prompts.py`:**
  - `planner_v1`, `researcher_v1` (with two `next_query` rule forms), and
    `verifier_v1`, `verifier_v2` and `verifier_v3`, all preserved and pinned by tests;
  - the A4.1 deterministic verifier contract (`verifier_contract_v2`): code computes
    `sufficient` itself, ignores references on unsupported requirements, and clears the
    corrective fields when the evidence is computed sufficient.
- **`run_eval.py`:**
  - `ma-check` and `ma-check-replay` (A4, A4.1);
  - `calib-flow` and `calib-probe` (A5 and A6);
  - `multi-agent`, with verifier v2 pinned in `PHASE4_VERIFIER` and recorded in every
    manifest;
  - `replay-agent` and `replay-multi-agent`;
  - `replay-run --phase`, whose outputs are named by phase and never overwritten.
- **`calibration/requirements_c1.jsonl`, `requirements_c2.jsonl`:** hand labels and
  frozen reference requirements, committed before any verifier probe on each subset.
- **Tests:** `tests/test_multi_agent.py`, including tests for:
  - prompt texts equal to the preregistration;
  - every routing row;
  - the contracts;
  - verifier independence;
  - the freeze hashes;
  - the replays aborting on a cache miss.

  **268 tests pass** in total.
- **Notebook:** `04_multi_agent.ipynb`, a read-only analysis. It reran from scratch with
  the project runner: 21 cells, no errors.

## Run history

| Commit | What it contains |
|---|---|
| `e2e095b` | The Phase 4 preregistration |
| `b847ac4` | The Phase 4 code and tests |
| `07bad11`, `d182c8c` | Amendment A4 (`verifier_v2`) and its implementation |
| `e9b4530`, `82f2885`, `f954952` | Amendment A4.1 (deterministic verifier contract), its implementation and the replayed gate |
| `bd921b7`, `f15b6c5`, `3ea06b9` | Amendment A5 (calibration protocol) and its commands |
| `8aa9707` | C1 labels and reference requirements |
| `a0110d5` | Amendment A6 (`verifier_v3`) |
| `fd0abb1` | C2 labels and reference requirements |
| `2715623` | The C2 decision and the Phase 4 freeze (verifier v2) |
| `2982b53` | Cache-only multi-agent replay; phase-named replays |

**Format check** (20 `eda` questions):

- `ma_check_v1`: the verifier failed its 10% gate (8/34). Amendment A4, a format-only
  revision, followed.
- `ma_check_v2`: 6/37 verifier failures, still over the gate.
- **A4.1:** moved the derived fields into code. An offline replay of the 37 recorded
  outputs gave 3/37 = 8.1%, and the gate passed. Both original results are preserved.

**Calibration** (A5, two fresh `eda` subsets of 20 questions, C1 and C2). Probes use
hand-corrected requirements:

| | S1 planner | S2 researchers | S3 complete → sufficient | S4 incomplete → not sufficient | Probe hard failures |
|---|---|---|---|---|---|
| C1, v2 | 0.75 | 0.90 | 0.30 | 0.70 | 13/40 |
| C2, v2 | 0.90 | 0.77 | 0.40 | 0.60 | 16/40 |
| C2, v3 | 0.90 | 0.77 | 0.15 | 0.30 | 29/40 |

- **C1 selected the verifier** (the largest shortfall, S3).
- **The one revision, `verifier_v3`, failed all three frozen keep conditions on C2:**
  - S3 dropped by 0.25;
  - S4 dropped by 0.30;
  - the pooled hard-failure rate rose from 0.37 to 0.65.

  It mostly returned a single entry for two requirements. `verifier_v2` was frozen.
- **Two probe annotations were recorded as diagnostics.** The primary S3 and S4 stay over
  all 20 questions.
  - **A gold label inconsistent with its question** (`5a847cc4`): the complete probe is
    really insufficient.
  - **An incomplete probe that is really sufficient** (`5adfc574`, and `5adf907f` in C1).

**`dev`.** All runs came from clean commits:

- **`dev_smoke`** (mechanics only), at `2715623`:
  - budgets held, no schema errors;
  - verifier failures 10/32;
  - its F1 was not used for any decision.
- **The full `dev` run,** at `2715623`:
  - 479 new generations, 2 h 21 min;
  - the 20 smoke questions were served from the cache, with identical call keys and
    answers.
- **Cache-only reproductions** of all five systems at `2982b53`:
  - multi-agent, single agent, R, BM25 K = 10, and D3 K = 16;
  - each aborts on any cache miss, or any change in searches, contexts, call keys or
    answers;
  - all reproduced exactly.

  They pass the final-comparison rule: one clean commit, evaluation-code hash
  `cd8b7302…`, split hash `90a6ad35…`, and the answer prompt, schema, validator,
  renderer and system prompt asserted equal.

## Metrics (`dev`, n = 100)

| | BM25 K = 10 | R = D3 K = 10 | D3 K = 16 (matched) | Single agent | Multi-agent |
|---|---|---|---|---|---|
| Answer F1 | 0.525 | 0.629 | 0.628 | 0.609 | 0.638 |
| EM | 0.410 | 0.470 | 0.510 | 0.470 | 0.490 |
| Final-context Joint Recall | 0.62 | 0.84 | 0.91 | **0.93** | 0.67 |
| Evidence F1 (strict) | 0.327 | 0.453 | 0.433 | 0.432 | 0.454 |
| Joint F1 (official) | 0.245 | 0.335 | 0.305 | 0.314 | 0.326 |
| Self-reported supported | 0.75 | 0.87 | 0.85 | 0.87 | 0.84 |
| All citations valid | 0.82 | 0.92 | 0.91 | 0.88 | 0.83 |
| Mean paragraphs in final answer context | 10 | 10 | 16 | 15.95 | **3.69** |
| LLM calls per question (logical) | 1 | 1 | 1 | 2.75 | 6.19 |
| Median LLM time per question | 20.6 s | 23.3 s | 34.2 s | 73.4 s | 110.8 s |

"Mean paragraphs in final answer context" counts what the synthesizer receives. It is not
the amount of text the system reads: the researchers and the verifier see many more
paragraph presentations across 619 role calls.

**Paired, multi-agent minus each other system:**

| Against | Answer F1 | EM | Final-context Joint Recall |
|---|---|---|---|
| R | +0.009 [−0.072, +0.091] | +0.020 [−0.050, +0.090] | −0.170 [−0.270, −0.070] |
| Matched control | +0.010 [−0.067, +0.088] | −0.020 [−0.100, +0.060] | −0.240 [−0.330, −0.150] |
| Single agent | +0.029 [−0.047, +0.106] | +0.020 [−0.050, +0.090] | −0.260 [−0.360, −0.170] |
| BM25 | +0.113 [+0.015, +0.211] | +0.080 [−0.010, +0.170] | +0.050 [−0.080, +0.180] |

**By type (answer F1, multi-agent − R):**

- **Bridge (n = 80):** 0.626 against 0.622, +0.004 [−0.087, +0.094].
- **Comparison (n = 20):** 0.687 against 0.659, +0.028 [−0.145, +0.201].

**Role failures** (contract or schema; a failed verification skips correction):

| Role | Failures |
|---|---|
| Planner | 1/100 |
| Researchers | 18/259 |
| Initial verifier | 31/100 |
| Recheck | 12/60 |
| Verifier, combined | 43/160 |

**Cost and behaviour:**

- **Calls:** 619 logical calls (60 questions used 7, 39 used 5, 1 used 4). 479 were new
  generations; the other 140 were cache hits, from the smoke questions and calls repeated
  within a question.
- **Searches per question:** 1 for 2 questions, 2 for 44, 3 for 54.
- **Valid initial verdicts:** 69, of which 60 were "not sufficient".
- **Corrective searches:** 60 attempted, 57 executed. Only 4 added a gold paragraph
  missing at the first verification.
- **Rechecks:** 60 in total, 48 valid and 12 failed.
  - **The 57 executed corrective-search paths:** 42 rechecks remained insufficient, 4
    became sufficient, and 11 failed.
  - **The 3 skipped corrective searches:** two repeated insufficient verdicts and one
    recheck error.

## Error analysis (exploratory, post hoc)

These breakdowns were not preregistered and do not establish causes.

| Breakdown | Finding |
|---|---|
| F1 by gold completeness of the final context | Complete (n = 67): 0.709, against R's 0.634 on the same questions. Incomplete (n = 33): 0.494, against R's 0.618 |
| Selection loss: both gold paragraphs retrieved, not both retained (n = 29) | Multi-agent 0.555 against R's 0.696, −0.142 [−0.272, −0.030]; 5 worse, 1 better, 23 equal |
| Gold paragraphs accepted, given retrieval | 165/196 = 0.84. First hop 0.86, second hop 0.84, comparison 0.80. In search 1, 0.86; found by a later search, 0.64 (9/14) |
| Verifier discrimination (valid first verdicts) | Judged sufficient: 6/36 with both gold paragraphs displayed, 3/33 with one missing |
| Valid first verdict against verifier error | F1 0.655 (n = 69) against 0.600 (n = 31). Errors: 13 invalid references, 10 corrective targets on supported requirements, 7 wrong entry counts, 1 support without a reference |
| Corrective searches that added a missing gold paragraph | 4 of 57 (3 then accepted) |
| Corrective search skipped as a duplicate query | 3 questions. In 2, the corrective researcher and the recheck repeated earlier calls exactly (no-op). In 1, the researcher re-read the 11 still-unaccepted pool paragraphs, accepted two non-gold paragraphs, and the recheck failed |
| "yes"/"no" answers to open questions | Multi-agent 4 (3 of them with a gold paragraph missing); other systems 1–2 |

**Higher F1 was associated with retaining both gold paragraphs; lower F1 was associated
with selection loss.** These post hoc groups do not establish causation. The main
evidence loss is in selection, not retrieval.

## Representative cases

- **Success, compression removing a distractor:** "'Coming Over' is the second Japanese
  single album by a boy group that debuted in what year?" The researchers kept exactly
  *Coming Over (Exo song)* and *Exo (band)*, and the answer was 2012. R, with 10
  paragraphs, answered 2016.
- **Success:** "The person who established the Human Biology journal had released how many
  publications at his death?" The accepted context was *Human Biology (journal)* and
  *Raymond Pearl*, and the answer was 841. R answered "six hundred eighty-four".
- **Selection loss:** "What year was Playford B Power Station permanently closed in by its
  Australian operator?"
  - The researchers accepted *Dunston Power Station*, *Alinta Energy* and *Blyth Power
    Station*, but not *Playford B Power Station*.
  - The answer was "no"; R answered 2015.
- **Selection loss:** the Ambrose Mendy question.
  - Only *Ambrose Mendy* was accepted; *Chris Eubank Jr.* was dropped.
  - The answer was "no"; R answered "The Ring".
- **Over-cautious verification:** with both gold paragraphs displayed and a valid
  verdict, the verifier still asked for a corrective search in 30 of 36 cases.

## Known limitations

- **One model, prompt set and budget:** a 7B quantized model, these prompts, a 7-call
  budget.
- **The verifier is known to be weak.** It reached S3 0.40 and S4 0.60 on fresh
  calibration data, and 27% of its `dev` calls failed the contract. The verifier-behaviour
  hypotheses measure this frozen verifier, not verification in general.
- **Small samples:** n = 100, with 20 comparison questions. Every F1 interval against R,
  the matched control and the single agent spans zero. That establishes neither
  equivalence nor non-inferiority.
- **Proxies:** H4-3 and H4-5 use missing gold paragraphs as a proxy for missing support.
- **No single mechanism is isolated.** Planning, querying, selection and verification
  change together. The post hoc breakdowns describe, they don't attribute causes.
- **Confirmation comes later.** Everything here is exploratory. The confirmatory
  comparison is the Holm family on `test_agents` in Phase 5, which has not started.

## Departures from the approved plan

1. **Amendments:**
   - A4: a format-only verifier revision;
   - A4.1: a deterministic interface repair, with its gate re-evaluated by an offline
     replay;
   - A5: a behavioural calibration on fresh training data;
   - A6: one verifier revision, rejected on C2.

   Each was committed before the data it governs.
2. **A diagnostic probe field:** `complete_probe_sufficient`, added for C2 after a gold
   label was found inconsistent with its question. It is reported as a diagnostic only.
3. **Replay naming:** `replay-run` wrote a fixed `_p3replay` name, which Phase 4 replays
   would have overwritten. It now takes `--phase` and refuses to overwrite a stored run. A
   cache-only `replay-multi-agent` was added. Both changes touch evaluation code, so all
   five systems were reproduced from the cache at the new clean commit. The original
   `dev` run is kept unchanged.
4. **Schedule:** the interview date was postponed. Phase 5 is deferred until a Phase 4b
   iteration has been decided (`PLAN.md`).

## Suggested improvements (Phase 4b)

Phase 4b is a separate, preregistered iteration. It runs on fresh calibration questions
(C3, C4), and the completed Phase 4 system stays untouched. Its targets:

- **Selection that cannot silently discard evidence:** per-requirement or per-paragraph
  acceptance decisions, so one malformed or over-selective list cannot drop a retrieved
  gold paragraph. This addresses the 29 selection-loss questions.
- **Verification that cannot fail on format:** schema-constrained, per-requirement
  verification calls, with deterministic aggregation of `sufficient` and of the
  corrective target.
- **No no-op corrective path:** skip the corrective researcher and the recheck when the
  corrective search adds no new results.

The improved design is frozen before it is compared on `dev`, and later on
`test_agents`.

## Phase 4b: structural repair attempt

**Motivation.** Two failure modes were measured on `dev`:

- **Selection loss:** 29 questions had both gold paragraphs retrieved but not both
  retained.
- **Verifier contract failures:** 43 of 160 verifier calls failed the output contract, and
  a failed verification skipped correction.

Phase 4b was a separately preregistered attempt to repair both structurally. It was
validated on fresh training questions, and the frozen Phase 4 system was left untouched.

**The structural changes** (`multi_agent_4b.py`; the planner, retriever, synthesizer and
budgets as in Phase 4):

- **Per-candidate researcher decisions.**
  - One required true/false decision per candidate paragraph, enforced while generating
    by Ollama's JSON-schema constraints.
  - Fail-open on a hard failure: every candidate shown is accepted.
- **Per-requirement verification.** One call per requirement, with references limited to
  the displayed sentence labels.
- **Deterministic aggregation in code:**
  - a requirement needs a reference to count as supported;
  - `sufficient` is the conjunction of the requirements;
  - the corrective target is the lowest-numbered unsupported requirement;
  - a verifier hard failure counts as unsupported, so errors no longer skip correction.
- **No-new-evidence routing.** The corrective researcher and the recheck are skipped when
  a corrective search adds nothing.

**Process:**

1. **G0, the live mechanics gate,** passed: schema constraints were enforced, and the
   runtime version was recorded.
2. **C3** (30 questions, development) failed G1 under its original definition, through one
   failure of the unchanged planner, and failed G2.
3. **Amendment A8:**
   - the planner became a calibration diagnostic (it stays in the `dev` reliability
     condition);
   - the one permitted prompt-only researcher revision followed (`researcher_v3`);
   - the verifier, which passed, was not revised.
4. **C4** (30 fresh questions, validation) ran with `researcher_v3`.

| Gate (raw counts) | C3 (`researcher_v2`) | C4 (`researcher_v3`) | Threshold |
|---|---|---|---|
| G1 researchers, hard failures | 0/59 | 0/59 | ≤ 0.02 |
| G1 verifier, flow + probes | 0/169 | 0/171 | ≤ 0.02 |
| G1 synthesizer schema errors | 0/30 | 0/30 | 0 |
| G1 planner (gated originally; diagnostic after A8) | 1/30 | 1/30 | ≤ 0.02 originally |
| **G2** gold displayed at first verification ÷ gold in the pool | 42/45 = 0.933 | **38/45 = 0.844** | ≥ 0.95 |
| G3 complete probes judged sufficient | 21/30 = 0.70 | 21/30 = 0.70 | ≥ 0.70 |
| G4 incomplete probes judged not sufficient | 23/30 = 0.77 | 24/30 = 0.80 | ≥ 0.70 |
| G5 mechanics violations | none | none | none |
| Paragraphs accepted per question | 2.67 | 2.10 | (diagnostic) |

The C4 probes were run after the G2 failure, as a diagnostic only.

**The decisive failure.** On C4, G2 = 38/45 = 0.844, against ≥ 0.95. `researcher_v3`
failed to validate the intended retention improvement: retention on fresh C4 was 0.844,
lower than the 0.933 observed for `researcher_v2` on C3. The two subsets contain different
questions, so this is not a paired comparison.

**What succeeded:**

- **Contract reliability.** The changed roles had 0 hard failures in 458 calls across C3
  and C4. The Phase 4 verifier had 43/160 failures on `dev`. That comparison is
  descriptive only, because the systems, prompts and question sets differ.
- **Verifier discrimination.** Requirements were judged supported 91/113 times with
  complete gold evidence against 52/113 with one gold paragraph removed. G3 and G4 held
  on both subsets.

**What failed: semantic evidence selection.**

- Despite explicit instructions to judge each candidate independently and keep when
  unsure, the researchers kept about 2–3 of 10 candidates.
- They lost retrieved gold on both subsets. On C4, two questions ended with no accepted
  evidence.
- They often kept a related paragraph instead of the gold one. Examples: another magazine
  instead of the paragraph naming Juxtapoz, and another graduate of the same school
  instead of the paragraph naming the city.
- Structured decoding guarantees that a decision exists for every candidate; it cannot
  make the decision right.

**Decision.** Phase 4b was rejected before `dev` (`PREREGISTRATION.md`, Phase 4b record):

- It was never run on `dev`, so **no answer-quality claim is possible** for it.
- The frozen Phase 4 system remains the multi-agent system in the Phase 5 confirmatory
  family.
- **The architectural conclusion:** structured outputs repaired contract reliability, but
  prompt-based evidence selection by the 7B model remained unreliable.

## Git diff / commit reference

- The Phase 4 `dev` run used `2715623`; the compared reproductions used `2982b53`.
- Phase 4b: implementation `b84fe81`, A7 `2a34bf7`, A8 `3e3061f`, C3 labels `020781e`, C4 labels
  `2fd4fd8`, rejection record `cc64186`.
- The analysis notebook and this report are in the commit that adds this report
  (`git log -1 -- projects/hotpot_qa/reports/phase_4.md`).
