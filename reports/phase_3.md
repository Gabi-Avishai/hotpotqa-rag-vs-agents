# Phase 3 report — a single investigation agent

**Phase:** 3, single investigation agent.

**Objective:** test whether evidence-dependent iterative search adds answer quality beyond
the frozen Phase 2 retriever R = D3, and at what cost (`PREREGISTRATION.md` Phase 3 and
amendment A3).

**Result in one sentence:** the agent finds more evidence, but doesn't answer better,
and costs three times as much. On `dev` (n = 100):

- context Joint Recall: 0.84 → 0.93;
- answer F1: 0.629 → 0.609, a difference of −0.020 [−0.080, +0.040];
- LLM time: 3.15× R's.

## Hypotheses (preregistered, exploratory, `dev`, n = 100)

| Id | Prediction | Estimate [95% CI where applicable] | Result |
|---|---|---|---|
| H3-1 | Agent − R, context Joint Recall, at least +0.03 | +0.090 [+0.040, +0.150] | Holds |
| H3-2 | (Δbridge − Δcomparison) > 0 | −0.075 [−0.250, +0.075] (bridge +0.075, n = 80; comparison +0.150, n = 20) | Not shown (point estimate runs the other way) |
| H3-3 | Agent F1 − R F1 > 0 | −0.020 [−0.080, +0.040] | Fails |
| H3-4 | Agent F1 − matched-context F1 > 0 | −0.019 [−0.081, +0.042] | Fails |
| H3-5 | `sufficient` at the first `assess` for at least 50% | 0.21 | Fails |
| H3-6 | Searching again when search 1 already had both gold, at most 40% | 0.74 (62/84) | Fails |
| H3-7 | Stopped as `sufficient` but missing a gold paragraph, at most 25% | 0.11 (4/36) | Holds |
| H3-8 | Median summed agent LLM time at least 1.8× R's | 73.4 s against 23.3 s = 3.15× | Holds |

## What was built

- **`agent.py`:** a LangGraph `StateGraph` with the nodes `search` (deterministic, over
  D3), `assess` (`assess_v2`) and `answer` (the frozen `answer_v2`).
  - Search 1 is the question itself and returns D3's top 10, exactly R's context.
  - Follow-up searches add the 5 highest-ranked unseen paragraphs each.
  - Budget: at most 3 searches, 20 paragraphs and 3 LLM calls.
  - Transitions are small pure functions, tested case by case.
  - The evidence-preserving overflow rule drops search 1's lowest ranks first and
    protects the newest batch; dropped ids are logged per call.
- **`prompts.py`:**
  - `assess_v1` is preserved.
  - `assess_v2` (A3) differs from v1 only in its `next_query` instruction.
  - `AssessOutput` enforces the runtime contract through its validator: the relation
    between `sufficient` and the other fields, the 12-token limit, and a violation
    ending as `assess_error`.
- **`run_eval.py`:**
  - `assess-check`, run once per version;
  - `agent`;
  - `replay-run`: cache-only, checking rankings before any generation, and aborting on a
    cache miss, a ranking or context mismatch, or changed metadata;
  - `k_match`;
  - `rag` accepting K from 10 to 20.
- **Tests:** `tests/test_agent.py`; **220 tests pass** in total.
- **Notebook:** `03_agent.ipynb`, read-only analysis. It reran from scratch: 6 code cells,
  no errors.

## Run history

| Commit | What it contains |
|---|---|
| `ea39204` | The Phase 3 preregistration |
| `e3ea8f3` | The Phase 3 code |
| `c3a5714` | Amendment A3, recorded before any v2 code |
| `7a870a3` | `assess_v2` |
| `37a0466` | The v2 result and the frozen hashes |

- **The `assess_v1` format check:** 7/40 failures, all `next_query` over the frozen
  12-token limit. The one permitted format-only revision followed.
- **The `assess_v2` check:** run once from the clean commit `7a870a3`: 2/40 failures,
  viability gate passed. `sufficient = true` in 7/20 full-context cases and 0/20
  one-gold-removed cases, kept as a finding.
- **All `dev` runs came from `37a0466`, with a clean tree:**
  - the `dev_smoke` smoke run (mechanics only: no crashes, no overflow drops, 2.75 calls
    per question);
  - the full `dev` agent run;
  - cache-only replays of R (D3 K = 10) and BM25 K = 10, each reproducing all 100 rankings
    and answers;
  - **K_match = 16** (mean displayed 15.95), recorded in the preregistration;
  - the matched-context control, D3 RAG at K = 16.
- **Interruptions, with no effect on results:** a laptop shutdown, the harness's
  background-task time limit, and the laptop sleeping for 10 minutes (lid closed).
  Generations are cached as they complete, so every restart replayed finished work and
  continued. The final chain ran as a detached process.
- **One manifest re-produced.** The matched control's manifest was first recorded as
  `dirty`: the analysis notebook, created while the run was going, was an untracked file.
  It was re-produced from the cache on a clean tree, with identical records and no new
  generations.
- **Manifests:** all four compared runs pass the final-comparison rule: the same clean
  commit, one evaluation-code hash (`1737931a…`), and the answer prompt, schema,
  validator, renderer and system prompt asserted equal.

## Metrics (`dev`, n = 100)

| | BM25 K = 10 | R = D3 K = 10 | D3 K = 16 (matched) | Agent |
|---|---|---|---|---|
| Answer F1 | 0.525 | **0.629** | 0.628 | 0.609 |
| EM | 0.410 | 0.470 | 0.510 | 0.470 |
| Context Joint Recall | 0.620 | 0.840 | 0.910 | **0.930** |
| Evidence F1 (strict) | 0.327 | 0.453 | 0.433 | 0.432 |
| Joint F1 (official) | 0.245 | 0.335 | 0.305 | 0.314 |
| Supported rate | 0.750 | 0.870 | 0.850 | 0.870 |
| All citations valid | 0.820 | 0.920 | 0.910 | 0.880 |
| Paragraphs shown | 10 | 10 | 16 | 15.95 |
| LLM calls per question | 1 | 1 | 1 | 2.75 |

**Paired, agent minus each other system:**

| Against | Answer F1 | Context Joint Recall | Evidence F1 (strict) |
|---|---|---|---|
| R | −0.020 [−0.080, +0.040] | +0.090 [+0.040, +0.150] | −0.022 [−0.079, +0.037] |
| Matched control | −0.019 [−0.081, +0.042] | +0.020 [−0.030, +0.070] | −0.001 [−0.063, +0.061] |
| BM25 | +0.084 [−0.003, +0.175] | +0.310 [+0.220, +0.400] | +0.105 [+0.026, +0.184] |

**By type (answer F1):**

- **Bridge (n = 80):** R 0.622, matched 0.615, agent 0.581. Agent − R = −0.040
  [−0.107, +0.025].
- **Comparison (n = 20):** R 0.659, matched 0.680, agent 0.720. Agent − R = +0.061
  [−0.067, +0.190].

**Behaviour:**

- **Stop reasons:** 44 `max_searches`, 36 `sufficient`, 14 `duplicate_query`,
  6 `assess_error`.
- **Searches per question:** 1 for 25 questions, 2 for 31, 3 for 44.
- **Cost:** 2.75 LLM calls per question; about 6,236 prompt and 254 output tokens per
  question. Summed LLM time per question: p50 73.4 s, p95 100.6 s. No overflow drops.
- **Follow-up queries:** 119 in total, of which 103 (87%) are evidence-dependent: they
  contain a content term from the retrieved paragraphs that the question lacks.
- **Recovery:** 16 questions lacked at least one gold paragraph after search 1. Follow-ups added a missing paragraph in 10 of them, and 9 finished with both; the tenth had lacked both and recovered only one.
- **`assess` calibration** (decision, and whether both gold paragraphs were present):

  | | Both gold present | A gold paragraph missing |
  |---|---|---|
  | Judged insufficient | 119 | 14 |
  | Judged sufficient | 32 | 4 |

## Why more evidence did not mean better answers (exploratory)

| Group (agent context vs R's) | n | Agent F1 | R F1 | Difference | Better / worse / same |
|---|---|---|---|---|---|
| Recovered (R lacked gold; the agent has both) | 9 | 0.370 | 0.259 | +0.111 | 2 / 1 / 6 |
| Both had both | 84 | 0.674 | 0.707 | −0.033 | 4 / 8 / 72 |
| Neither | 7 | 0.143 | 0.171 | −0.029 | 0 / 1 / 6 |

Where the agent recovered missing evidence, F1 was higher. Where R already had both gold
paragraphs, **F1 was 0.033 lower after redundant paragraphs were added, consistent with
reader distraction.** Examples: "Bert Jansch" instead of "Eric Clapton", and "Coleman
Hawkins" (the subject) instead of "Shep Meyers". This subgroup analysis is exploratory
and does not establish causation. R had both gold paragraphs for 84% of questions, so the
net effect is slightly negative.

## Representative cases

- **Success, an evidence-dependent query:** "'Coming Over' is the second Japanese single
  album by a boy group that debuted in what year?" The follow-up query "EXO debut year"
  names the bridge entity, found only in a retrieved paragraph, and adds the gold
  *Exo (band)* paragraph.
- **Unnecessary search that hurts:** "Besides the founder of Led Zeppelin, and Bert
  Jansch, what other British musician…?" R's 10 paragraphs already contain both gold paragraphs, and
  R answers "Eric Clapton" correctly. After searching again, the agent answers "Bert
  Jansch", a name taken from the question.

## Known limitations

- **One model, prompt and budget.** The result applies to a 7B quantized model, this
  prompt and this budget. A better-calibrated sufficiency judgement could change it.
- **Small samples:** n = 100, with only 20 comparison questions. Every interval on answer
  F1 spans zero.
- **The matched control is approximate:** a global context-size control, not a
  per-question match.
- **A crude query measure:** the evidence-dependence sign can count a query that names
  the wrong entity (the Henry Cadogan example still retrieved the gold paragraph).
- **Confirmation comes later:** everything here is exploratory. The confirmatory
  comparison is the Holm family on `test_agents` in Phase 5.

## Departures from the approved plan

1. **Amendment A3:** the one permitted format-only revision of the `assess` prompt.
2. **Re-produced manifest:** the matched control's manifest was re-produced from the
   cache, after an untracked notebook marked the first one dirty.
3. **Detached final chain:** it ran as a detached process, because the harness's
   30-minute limit on background tasks stopped the earlier attempt partway.

## Suggested direction for Phase 4

- **The weak component in Phase 3 is the stopping decision.** Frequent searching even
  when search 1 already contained both gold paragraphs.
- **What the planned verifier tests, and what it doesn't.** It does **not** directly stop
  the researchers' planned searches: it sees the accepted evidence after they have run,
  and can request one corrective search. As designed, Phase 4 therefore tests whether
  independent verification:
  - improves evidence selection;
  - detects missing support;
  - avoids unnecessary *corrective* searching.

  It does not test whether a verifier fixes Phase 3's early over-searching in general.
- **Keep the design small, given the deadline.** Keep the preregistered design and the
  Phase 3 retrieval budget, and state the purpose accurately rather than adding a routing
  stage.
- **Routing as a deployment recommendation.** "Retrieve with D3, search again only when
  evidence is missing" would need a more reliable sufficiency signal. It belongs in the
  deployment discussion.

## Git diff / commit reference

The Phase 3 runs used `37a0466`. The analysis notebook, this report, the slides and the
K_match record are in the commit that adds this report
(`git log -1 -- projects/hotpot_qa/reports/phase_3.md`).
