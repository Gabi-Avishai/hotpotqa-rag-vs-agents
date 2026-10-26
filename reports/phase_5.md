# Phase 5 report — frozen blind test

**Phase:** 5, freeze and blind test.

**Objective:** evaluate the frozen systems once on the untouched test splits, under the
approved freeze (`PREREGISTRATION.md` Phase 5 and the §A freeze declaration). The
analysis and the audit were defined before any test split was opened.

**Results:**

- **Retrieval:** the improved retriever R = D3 substantially improves retrieval on
  `test_retrieval`. On the larger `test_rag` split it improves answer F1, a secondary
  result.
- **Confirmatory:** none of the three confirmatory answer-F1 comparisons on `test_agents`
  was demonstrated. That does not establish equivalence.
- **Agents:** neither agent architecture improved answer F1 clearly over R.
- **Multi-agent:** its evidence-selection loss replicated. Gold-paragraph coverage in the
  search pool was 0.860; in the final answer context it was 0.596.

## The confirmatory family (`test_agents`, n = 100: 50 bridge + 50 comparison)

**Method:**

- answer F1;
- the prevalence-weighted overall difference (bridge 0.799, comparison 0.201);
- paired bootstrap resampled within type, 10,000 resamples, seed 0;
- Holm over three comparisons;
- the 95% intervals are not adjusted for multiplicity.

| | Delta [95% CI] | p | Holm p | Conclusion |
|---|---|---|---|---|
| F1: R − baseline (BM25 K = 10) | +0.070 [−0.041, +0.181] | 0.221 | 0.663 | No clear difference demonstrated |
| F2: single agent − R | +0.006 [−0.075, +0.090] | 0.883 | 1.000 | No clear difference demonstrated |
| F3: multi-agent − single agent | −0.007 [−0.113, +0.096] | 0.910 | 1.000 | No clear difference demonstrated |

**By type** (reported, not tested):

| | Bridge | Comparison |
|---|---|---|
| F1 | +0.093 [−0.045, +0.232] | −0.024 [−0.068, +0.000] |
| F2 | +0.017 [−0.083, +0.121] | −0.036 [−0.101, +0.014] |
| F3 | −0.000 [−0.133, +0.125] | −0.032 [−0.116, +0.044] |

None of the intervals excludes zero, so no clear improvement or worsening is
demonstrated, and none is shown to be absent either.

## Secondary results (no claims from p-values)

**`test_retrieval`** (n = 1,000, proportional, unweighted), Joint Recall@K:

| K | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| BM25 | 0.215 | 0.326 | 0.436 | 0.592 | 0.693 | 0.778 |
| D3 | 0.260 | 0.379 | 0.495 | 0.829 | 0.906 | 0.937 |

- **D3 − BM25 at K = 10:** +0.237 [+0.207, +0.268].
- **By type at K = 10:**

  | | Bridge | Comparison |
  |---|---|---|
  | BM25 | 0.526 | 0.856 |
  | D3 | 0.806 | 0.920 |

- **Title-only Joint Recall** is within 0.001 of exact-variant Joint Recall.
- **Retrieval latency, p50:** BM25 0.6 ms, D3 32.7 ms.

**`test_rag`** (n = 200: 160 bridge, 40 comparison; unweighted). This comparison is
**secondary**: it is not part of the Holm-controlled confirmatory family.

| | BM25 K = 10 | R | Closed-book | Oracle |
|---|---|---|---|---|
| Answer F1 | 0.537 | 0.615 | 0.246 | 0.803 |
| EM | 0.435 | 0.495 | 0.195 | 0.665 |
| Final-context Joint Recall | 0.580 | 0.855 | — | 1.000 |

R − BM25:

- **F1:** +0.077 [+0.008, +0.144]. By type: bridge +0.113, comparison −0.066.
- **EM:** +0.060 [−0.005, +0.125].
- **Final-context Joint Recall:** +0.275 [+0.205, +0.345].

## `test_agents`: all systems (prevalence-weighted overall; bridge / comparison)

| | BM25 | R | Matched control (D3 K = 16) | Single agent | Multi-agent | Oracle | Closed-book |
|---|---|---|---|---|---|---|---|
| Answer F1 | 0.544 (0.488/0.766) | 0.614 (0.581/0.742) | 0.605 (0.577/0.716) | 0.620 (0.598/0.706) | 0.613 (0.598/0.674) | 0.773 | 0.225 |
| EM | 0.492 | 0.552 | 0.548 | 0.564 | 0.556 | 0.712 | 0.168 |
| Final-context Joint Recall | 0.572 | 0.808 | 0.836 | 0.932 | 0.596 | 1.000 | — |
| Strict supporting-fact F1 | 0.388 | 0.441 | 0.447 | 0.463 | 0.449 | 0.569 | — |
| Joint F1 (official) | 0.256 | 0.294 | 0.316 | 0.322 | 0.308 | 0.444 | — |
| Share self-marked `supported = true` | 0.744 | 0.840 | 0.804 | 0.868 | 0.784 | 0.960 | 0.692 |
| F1 among outputs self-marked `supported = true` (n) | 0.584 (78) | 0.626 (87) | 0.699 (84) | 0.619 (88) | 0.670 (79) | 0.776 (90) | 0.238 (77) |
| Citations present / all citations valid | 0.95 / 0.89 | 0.96 / 0.86 | 0.97 / 0.89 | 1.00 / 0.89 | 0.94 / 0.86 | 0.99 / 0.84 | — |
| Mean paragraphs in final answer context | 10 | 10 | 16 | 16.4 | 3.7 | 2 | — |

- **Supporting-fact F1:** the official-compatible variant equals the strict one on this
  split.
- **Closed-book:** low-confidence rate 0.308.

**Secondary paired comparisons** (weighted):

| | Answer F1 | Final-context Joint Recall |
|---|---|---|
| Multi-agent − R | −0.001 [−0.092, +0.092] | −0.212 [−0.328, −0.100] |
| Matched control − single agent | −0.015 [−0.095, +0.060] | −0.096 [−0.176, −0.032] |
| Matched control − multi-agent | −0.008 [−0.107, +0.089] | +0.240 [+0.136, +0.352] |

**Agent behaviour** (weighted):

- **Single agent:**
  - Joint Recall funnel: search 1 0.808 → whole search pool 0.932 → final context 0.932;
  - 2.28 searches per question;
  - stop reasons: 0.53 reached the search limit, 0.34 judged sufficient, 0.11 duplicate
    query, 0.02 `assess` error.
- **Multi-agent:**
  - funnel: 0.808 → 0.860 → 0.596;
  - 2.47 searches per question, and 3.46 paragraphs accepted per question;
  - corrective search attempted in 63 questions (weighted 0.564) and executed in 56; 7 had
    no new search results;
  - first verification: 0.564 not sufficient, 0.096 sufficient, 0.34 hard failure.
  - Failure rates by role:

    | Role | Bridge | Comparison | Weighted |
    |---|---|---|---|
    | Initial verifier | 20/50 | 5/50 | 0.34 |
    | Recheck | 3/26 | 2/37 | 0.10 |
    | Researchers | 4/126 | 20/137 | 0.055 |
    | Planner | 0/50 | 0/50 | 0 |

**What the funnel shows.** For the multi-agent pipeline, prevalence-weighted Joint Recall
fell from 0.860 over the whole search pool to 0.596 in the final answer context. In the
balanced sample, both gold paragraphs were in the pool for 89/100 questions and in the
final context for 65/100. Selection loses retrieved evidence. Whether this lowered answer F1 is not established. The answer-F1
differences are small and imprecise, and differences in answer quality cannot be
attributed to selection from these data.

**Cost** (weighted per question; totals over 100 questions):

| | Logical calls | New generations | Cache hits | Prompt tokens per question | LLM seconds p50 / p95 |
|---|---|---|---|---|---|
| BM25 | 1.00 | 84 | 16 | 1,698 | 22.9 / 28.9 |
| R | 1.00 | 84 | 16 | 1,712 | 21.9 / 29.2 |
| Matched control | 1.00 | 100 | 0 | 2,675 | 32.8 / 40.8 |
| Single agent | 2.75 | 246 | 28 | 6,215 | 76.6 / 97.9 |
| Multi-agent | 6.13 | 605 | 21 | 7,823 | 108.0 / 148.2 |

- **Relative to R's median LLM time:** about 3.5× for the single agent and about 4.9× for
  the multi-agent system.
- **Retrieval latency, p50:** R 37 ms, single agent 100 ms, multi-agent 83 ms.
- **Cache reuse** (`test_rag` and `test_agents`): 2,200 logical calls, 113 cache hits,
  2,087 unique new generations.
- **The 16 cache hits in four systems** come from 16 questions shared by `test_rag` and
  `test_agents`, whose prompts were byte-identical (§5.4).

## The blinded audit (descriptive)

**Design** (§5.7):

- 12 bridge and 12 comparison `test_agents` questions, chosen by the frozen hash ordering;
- 4 systems: 96 outputs, labelled by hand with system names, scores, the `supported` flag
  and gold answers hidden.

The labels file has sha256 `542e6395…`, and the audit result has sha256 `0780bf5b…`.

**Unsupported-claim rate.** Among outputs self-marked `supported = true`, it is the share
labelled `unsupported` or `contradicted`. Intervals are exact (Clopper–Pearson) 95%.

| | Bridge | Comparison | Balanced audit (12 + 12) |
|---|---|---|---|
| BM25 | 3/6 = 0.50 [0.12, 0.88] | 1/7 = 0.14 [0.00, 0.58] | 4/13 = 0.31 [0.09, 0.61] |
| R | 4/8 = 0.50 [0.16, 0.84] | 2/11 = 0.18 [0.02, 0.52] | 6/19 = 0.32 [0.13, 0.57] |
| Single agent | 3/9 = 0.33 [0.07, 0.70] | 1/9 = 0.11 [0.00, 0.48] | 4/18 = 0.22 [0.06, 0.48] |
| Multi-agent | 4/10 = 0.40 [0.12, 0.74] | 1/9 = 0.11 [0.00, 0.48] | 5/19 = 0.26 [0.09, 0.51] |

The **balanced-audit rate** describes a deliberately balanced 12 + 12 sample. It is not
prevalence-weighted and not a population estimate.

**Evidence support, over 24 outputs each:**

- **Fully supported:** the single agent had the most (17/24), followed by R (15),
  multi-agent (14) and BM25 (11). In a sample this small this is not a reliable ordering.
- **Unsupported:** BM25 9, R 5, single agent 4, multi-agent 5.
- **Contradicted:** BM25 1, R 4, single agent 3, multi-agent 3.

**Citation validity** (valid of 24): BM25 17, R 21, single agent 22, multi-agent 23.
Citation validity and evidence support are separate outcomes: a valid citation can
accompany an unsupported answer, and the reverse.

**What the audit shows.** Among self-reported "supported" answers, the unsupported or
contradicted share had higher point estimates on bridge questions (between a third and a
half) than on comparison questions (about one in ten to one in five). The intervals were
wide, and no difference by type or by system was established.

## Freeze and run history

| Commit | Content |
|---|---|
| `5e3f865` | The approved Phase 5 preregistration |
| `7a9e8d4`, `3a5a03b`, `17e4530`, `fae9908` | The runner: staged and immutable, with validation, the full analysis, the audit, and consumer checks |
| `cff0d20` | The §A freeze declaration; every Phase 5 stage ran from this commit with a clean tree |

**Before unblinding:**

- 351 tests passed.
- A seven-system dry run on `dev` (cache-only, no blind access) reproduced every stored
  `dev` output exactly.

**Stages:**

| Stage | Finished | Outcome |
|---|---|---|
| `test_retrieval` | 2026-10-07 15:06 | 2 × 1,000 questions; no LLM; 36 s in total |
| `test_rag` | 2026-10-07 22:01 | 4 × 200 questions; 198 min of system runtime (BM25 73.9, R 75.4, closed-book 16.4, oracle 32.7 min) |
| `test_agents` | 2026-10-08 14:11 | 7 × 100 questions; completed after two infrastructure interruptions (below) |

**Infrastructure interruptions** (amendment A7). Both were in the `test_agents` stage,
during the single agent.

1. **The first stall.**
   - At 00:32 on 2026-10-08, Ollama's inference process stopped partway through
     processing the prompt of the single agent's 13th question. Its log stopped after 66%
     of the prompt.
   - At 00:47 the client's 900-second timeout aborted the stage (Ollama returned HTTP 500
     after 15 minutes).
2. **The second stall.**
   - A resume at 00:48 re-validated the five completed systems and accepted the single
     agent's 12 stored records.
   - Its first request queued behind the still-wedged inference process. Ollama logged no
     processing for it, and it timed out at 01:03.
3. **The resumption.**
   - After a machine restart, the stage resumed at about 09:40 from the same commit.
   - The five completed systems were re-validated without rerunning.
   - The partial single-agent run resumed only after its recorded commit, configuration and
     question order matched. The 13th question then completed normally on the fresh server.

**What the interruptions did not change:**

- No completed output was rerun or changed.
- The final stage validation confirms ordered, complete and duplicate-free records, the
  freeze commit, the configuration, and the runtime for every system.

**What survives as evidence:**

- The detailed logs of the two interrupted attempts were in a temporary directory that the
  restart cleared.
- Ollama's request-level log for that boot was not retained.
- The partial single-agent file was inspected at the time. It no longer exists: it was
  replaced when the system was finalized.
- The interruption record above therefore rests on contemporaneous observations (Ollama's
  log, the process states, and inspection of the then-present partial records) and on the
  surviving finalized artifacts and their validation, not on the attempt logs.

**Single-agent runtime:**

| Measure | Value |
|---|---|
| Resumed-session runtime (manifest), questions 13–100 | 93.6 min |
| Logical LLM wall-time attributed from call records, all 274 calls | 116.4 min |
| of which, questions 1–12 (generated before the stall) | 14.2 min |
| of which, questions 13–100 | 102.2 min |

- **Attributed wall-time** sums each call record's stored latency. A cache hit is charged
  its original generation latency, so this is neither physical generation time nor
  elapsed runtime.
- **The 93.6 min** is the resumed session's actual runtime, from its manifest.
- **Elapsed wall-clock time** for the stage was much longer, because of the stall and the
  downtime. It is not a cost measure.
- **The multi-agent system's** logical LLM wall-time attributed from call records was
  176.5 min.

## Qualitative examples (`test_agents`; post hoc)

These are post hoc qualitative examples, selected after the results were inspected. They
are neither preregistered nor representative in a statistical sense.

- **The single agent recovering missing evidence:**
  - "Sadok Sassi played for a national team that made its first World Cup in what year?"
    R's context lacked a gold paragraph and answered 1970. The agent searched "Tunisia
    first World Cup appearance year", then answered 1978, which is correct.
  - "The magazine that nominated George Rainsford for their Best Actor award in 2017 comes
    out every week on what day of the week?" R answered Monday. The agent's follow-up
    queries ("TV Choice Awards weekly magazine publication day", "TV Choice publication
    day") led it to Tuesday, which is correct.
- **Multi-agent selection loss**, from the 5 questions where both gold paragraphs were
  retrieved but not kept, and R answered correctly:
  - "The Government Digital Service is overseen by the Chief Secretary to the Treasury,
    who is from which party?"
    - The researchers kept "Shadow Chief Secretary to the Treasury", "Secretary to the
      Treasury" and "Exchequer Secretary to the Treasury".
    - They dropped both gold paragraphs (David Gauke, Government Digital Service).
    - The answer was "no"; R answered Conservative.
  - "Jolyon James performed the role of Moonshadow in a musical by a composer of what
    nationality?" The Cat Stevens paragraph was dropped in favour of another James Dooley
    paragraph. The answer was Australian; R answered British.

## Known limitations

- **Sample size.** `test_agents` is n = 100, and the overall estimates weight a 50/50
  sample to 80/20. Intervals on F1 differences are about ±0.08 to ±0.11. Effects of the
  size seen on `test_rag` (+0.08) cannot be resolved here.
- **No equivalence.** "No clear difference demonstrated" does not mean the systems are
  equivalent.
- **One model and setting.** A 7B quantized model, these prompts and these budgets.
- **The audit** is small, descriptive, labelled by one person, and balanced by type.
- **Proxies.** Self-marked `supported = true` is the model's own report. The gold-paragraph
  measures are proxies for support. The audit is the direct, small-sample check.
- **Benchmark status.** A closed corpus built from HotpotQA distractor contexts. Scores are
  not comparable with the official leaderboard.

## Departures from the approved plan

1. **The agents stage started without its own review stop.** It began automatically when
   the RAG stage completed, through a chained launcher, at the operator's request. The
   runner's checks (marker, validation, freeze) still applied. The RAG results were
   reviewed while the agents stage ran.
2. **Two infrastructure interruptions,** handled under A7, as above.
3. **The interrupted attempts' logs were lost** to a machine restart. The record rests on
   observations made at the time and on the final validation.

## Suggested improvements

- **A larger confirmatory split** to resolve differences of the size seen on `test_rag`.
- **Evidence selection** that cannot discard retrieved evidence. A Phase 4b structural
  attempt fixed reliability but not selection, so this means a different selector or
  model.
- **Routing:** search again only when the first context is insufficient, which needs a
  reliable sufficiency signal.
- **Runtime robustness:** a watchdog that restarts a stalled inference server, and attempt
  logs kept outside temporary storage.

## Git and artifact reference

- **Freeze commit:** `cff0d20`.
- **Stored results:**
  - stage markers and results under `results/final/`;
  - the analysis at `results/final/analysis.json`, sha256 `99f81c00df814b08…`;
  - the audit at `results/final/audit/analysis.json`, sha256 `0780bf5be6246b4b…`.
- **This report** is in the commit that adds it
  (`git log -1 -- projects/hotpot_qa/reports/phase_5.md`).

## Appendix A. `test_agents`: every system, weighted overall (bridge / comparison)

Overall values are prevalence-weighted (bridge 0.799, comparison 0.201); per-type values are plain means over 50 questions each. Transcribed from `results/final/analysis.json`.

| Metric | BM25 | R | Matched (K=16) | Single agent | Multi-agent | Oracle | Closed-book |
|---|---|---|---|---|---|---|---|
| Answer F1 | 0.544 (0.488 / 0.766) | 0.614 (0.581 / 0.742) | 0.605 (0.577 / 0.716) | 0.620 (0.598 / 0.706) | 0.613 (0.598 / 0.674) | 0.773 (0.779 / 0.748) | 0.225 (0.141 / 0.563) |
| EM | 0.492 (0.440 / 0.700) | 0.552 (0.520 / 0.680) | 0.548 (0.520 / 0.660) | 0.564 (0.540 / 0.660) | 0.556 (0.540 / 0.620) | 0.712 (0.720 / 0.680) | 0.168 (0.080 / 0.520) |
| Final-context Joint Recall | 0.572 (0.500 / 0.860) | 0.808 (0.780 / 0.920) | 0.836 (0.800 / 0.980) | 0.932 (0.920 / 0.980) | 0.596 (0.560 / 0.740) | 1.000 (1.000 / 1.000) | — |
| Final-context Recall | 0.786 (0.750 / 0.930) | 0.896 (0.880 / 0.960) | 0.910 (0.890 / 0.990) | 0.958 (0.950 / 0.990) | 0.750 (0.720 / 0.870) | 1.000 (1.000 / 1.000) | — |
| Strict supporting-fact F1 | 0.388 (0.345 / 0.558) | 0.441 (0.403 / 0.596) | 0.447 (0.420 / 0.552) | 0.463 (0.431 / 0.588) | 0.449 (0.426 / 0.539) | 0.569 (0.559 / 0.607) | — |
| Strict supporting-fact precision | 0.486 (0.440 / 0.667) | 0.550 (0.510 / 0.707) | 0.565 (0.547 / 0.640) | 0.546 (0.510 / 0.690) | 0.567 (0.540 / 0.673) | 0.734 (0.730 / 0.750) | — |
| Strict supporting-fact recall | 0.343 (0.298 / 0.518) | 0.398 (0.358 / 0.555) | 0.397 (0.367 / 0.517) | 0.426 (0.395 / 0.550) | 0.393 (0.370 / 0.487) | 0.494 (0.480 / 0.550) | — |
| Official-compatible supporting-fact F1 | 0.388 (0.345 / 0.558) | 0.441 (0.403 / 0.596) | 0.447 (0.420 / 0.552) | 0.463 (0.431 / 0.588) | 0.449 (0.426 / 0.539) | 0.569 (0.559 / 0.607) | — |
| Joint F1 (official) | 0.256 (0.217 / 0.412) | 0.294 (0.253 / 0.457) | 0.316 (0.288 / 0.427) | 0.322 (0.298 / 0.418) | 0.308 (0.289 / 0.384) | 0.444 (0.434 / 0.486) | — |
| Self-marked `supported = true` | 0.744 (0.720 / 0.840) | 0.840 (0.820 / 0.920) | 0.804 (0.780 / 0.900) | 0.868 (0.860 / 0.900) | 0.784 (0.780 / 0.800) | 0.960 (1.000 / 0.800) | 0.692 (0.640 / 0.900) |
| Has citations | 0.948 (0.940 / 0.980) | 0.960 (0.960 / 0.960) | 0.968 (0.960 / 1.000) | 1.000 (1.000 / 1.000) | 0.944 (0.940 / 0.960) | 0.992 (1.000 / 0.960) | — |
| All citations valid | 0.892 (0.880 / 0.940) | 0.864 (0.860 / 0.880) | 0.888 (0.880 / 0.920) | 0.892 (0.880 / 0.940) | 0.864 (0.860 / 0.880) | 0.836 (0.820 / 0.900) | — |
| Paragraphs in final answer context | 10.000 (10.000 / 10.000) | 10.000 (10.000 / 10.000) | 16.000 (16.000 / 16.000) | 16.420 (16.500 / 16.100) | 3.700 (3.640 / 3.940) | 2.000 (2.000 / 2.000) | — |
| F1 among outputs self-marked `supported = true` | 0.584 (0.543, n 36 / 0.746, n 42) | 0.626 (0.597, n 41 / 0.742, n 46) | 0.699 (0.697, n 39 / 0.707, n 45) | 0.619 (0.606, n 43 / 0.674, n 45) | 0.670 (0.664, n 39 / 0.693, n 40) | 0.776 (0.779, n 50 / 0.763, n 40) | 0.238 (0.155, n 32 / 0.570, n 45) |
| Per-citation validity (valid/cited) | 0.963 (70/73 / 86/88) | 0.940 (74/79 / 83/87) | 0.949 (71/75 / 89/93) | 0.938 (81/87 / 87/90) | 0.946 (68/72 / 78/82) | 0.899 (68/77 / 76/79) | — |

Closed-book low-confidence rate: 0.308 (0.360 / 0.100).


## Appendix B. `test_agents`: cost and latency

Per-question means are prevalence-weighted. Overall p50/p95 are weighted empirical quantiles; per-type quantiles are ordinary. LLM seconds are attributed from call records (a cache hit carries its original latency).

| System | Calls (total; per q) | Cache hits | Unique new generations | Prompt tokens (total; per q) | Output tokens (total; per q) | LLM s p50 / p95 overall (bridge; comparison) | Retrieval ms p50 / p95 overall (bridge; comparison) |
|---|---|---|---|---|---|---|---|
| BM25 | 100; 1.00 | 16 | 84 | 163,755; 1698 | 9,341; 95 | 22.9 / 28.9 (23.2/29.1; 20.6/27.1) | 1 / 2 (1/2; 1/2) |
| R | 100; 1.00 | 16 | 84 | 166,294; 1712 | 9,178; 95 | 21.9 / 29.2 (22.3/30.1; 20.2/24.3) | 37 / 45 (38/45; 36/42) |
| Matched (K=16) | 100; 1.00 | 0 | 100 | 259,259; 2675 | 9,180; 91 | 32.8 / 40.8 (32.9/40.8; 30.5/38.1) | 39 / 45 (39/45; 37/41) |
| Single agent | 274; 2.75 | 28 | 246 | 598,622; 6215 | 24,051; 245 | 76.6 / 97.9 (77.6/98.7; 73.1/91.2) | 100 / 119 (100/119; 83/118) |
| Multi-agent | 626; 6.13 | 21 | 605 | 782,237; 7823 | 37,545; 370 | 108.0 / 148.2 (104.1/148.2; 108.0/139.3) | 83 / 124 (80/124; 104/117) |
| Oracle | 100; 1.00 | 16 | 84 | 50,730; 524 | 8,434; 88 | 9.7 / 12.9 (9.9/13.4; 9.0/10.9) | 0 / 0 (0/0; 0/0) |
| Closed-book | 100; 1.00 | 16 | 84 | 20,272; 205 | 5,261; 54 | 4.7 / 6.8 (4.7/6.8; 4.6/5.8) | 0 / 0 (0/0; 0/0) |

## Appendix C. `test_rag` (secondary): every system, overall (bridge / comparison)

Unweighted: `test_rag` is a proportional sample (160 bridge, 40 comparison).

| Metric | BM25 | R | Oracle | Closed-book |
|---|---|---|---|---|
| Answer F1 | 0.537 (0.483 / 0.753) | 0.615 (0.597 / 0.687) | 0.803 (0.796 / 0.832) | 0.246 (0.143 / 0.660) |
| EM | 0.435 (0.369 / 0.700) | 0.495 (0.456 / 0.650) | 0.665 (0.637 / 0.775) | 0.195 (0.081 / 0.650) |
| Final-context Joint Recall | 0.580 (0.500 / 0.900) | 0.855 (0.844 / 0.900) | 1.000 (1.000 / 1.000) | — |
| Final-context Recall | 0.787 (0.747 / 0.950) | 0.917 (0.909 / 0.950) | 1.000 (1.000 / 1.000) | — |
| Strict supporting-fact F1 | 0.446 (0.401 / 0.624) | 0.466 (0.429 / 0.613) | 0.585 (0.576 / 0.621) | — |
| Strict supporting-fact precision | 0.571 (0.543 / 0.684) | 0.592 (0.569 / 0.683) | 0.772 (0.777 / 0.750) | — |
| Strict supporting-fact recall | 0.397 (0.346 / 0.602) | 0.412 (0.370 / 0.583) | 0.504 (0.489 / 0.565) | — |
| Official-compatible supporting-fact F1 | 0.446 (0.401 / 0.624) | 0.466 (0.429 / 0.613) | 0.585 (0.576 / 0.621) | — |
| Joint F1 (official) | 0.297 (0.251 / 0.478) | 0.328 (0.299 / 0.445) | 0.488 (0.476 / 0.532) | — |
| Self-marked `supported = true` | 0.690 (0.656 / 0.825) | 0.800 (0.787 / 0.850) | 0.925 (0.956 / 0.800) | 0.640 (0.581 / 0.875) |
| Has citations | 0.915 (0.900 / 0.975) | 0.960 (0.963 / 0.950) | 0.985 (0.994 / 0.950) | — |
| All citations valid | 0.865 (0.856 / 0.900) | 0.880 (0.881 / 0.875) | 0.870 (0.863 / 0.900) | — |
| Paragraphs in final answer context | 10.000 (10.000 / 10.000) | 10.000 (10.000 / 10.000) | 2.000 (2.000 / 2.000) | — |
| F1 among outputs self-marked `supported = true` (n) | 0.658 (138) | 0.687 (160) | 0.815 (185) | 0.325 (128) |
| Per-citation validity | 0.966 | 0.948 | 0.917 | — |

| System | Calls | Cache hits | Unique new generations | Prompt tokens (total; per q) | Output tokens | LLM s p50 / p95 | Retrieval ms p50 / p95 |
|---|---|---|---|---|---|---|---|
| BM25 | 200 | 0 | 200 | 329,758; 1649 | 18,479 | 21.8 / 28.7 | 1 / 2 |
| R | 200 | 0 | 200 | 336,572; 1683 | 18,673 | 22.1 / 28.7 | 38 / 45 |
| Oracle | 200 | 0 | 200 | 102,547; 513 | 17,004 | 9.8 / 12.5 | 0 / 0 |
| Closed-book | 200 | 0 | 200 | 40,694; 203 | 10,558 | 4.8 / 6.2 | 0 / 0 |

## Appendix D. `test_retrieval`: curves by type at every frozen K

n = 1,000 (799 bridge, 201 comparison); unweighted. Each cell: overall (bridge / comparison).

**BM25**

| Measure | K = 2 | K = 3 | K = 5 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|---|---|
| Joint Recall (exact) | 0.215 (0.191 / 0.308) | 0.326 (0.288 / 0.478) | 0.436 (0.380 / 0.657) | 0.592 (0.526 / 0.856) | 0.693 (0.631 / 0.940) | 0.778 (0.730 / 0.970) |
| Joint Recall (title-only) | 0.215 (0.191 / 0.308) | 0.327 (0.289 / 0.478) | 0.436 (0.380 / 0.657) | 0.592 (0.526 / 0.856) | 0.693 (0.631 / 0.940) | 0.778 (0.730 / 0.970) |
| Recall (exact) | 0.551 (0.543 / 0.582) | 0.623 (0.603 / 0.699) | 0.696 (0.668 / 0.808) | 0.789 (0.755 / 0.928) | 0.843 (0.810 / 0.970) | 0.887 (0.863 / 0.985) |
| Recall (title-only) | 0.551 (0.543 / 0.582) | 0.623 (0.604 / 0.699) | 0.696 (0.668 / 0.808) | 0.789 (0.755 / 0.928) | 0.843 (0.810 / 0.970) | 0.887 (0.863 / 0.985) |

Latency per query: p50 0.6 ms, p95 0.8 ms (bridge 0.6/0.8; comparison 0.5/0.7 ms).

**D3**

| Measure | K = 2 | K = 3 | K = 5 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|---|---|
| Joint Recall (exact) | 0.260 (0.194 / 0.522) | 0.379 (0.292 / 0.726) | 0.495 (0.403 / 0.861) | 0.829 (0.806 / 0.920) | 0.906 (0.885 / 0.990) | 0.937 (0.924 / 0.990) |
| Joint Recall (title-only) | 0.261 (0.195 / 0.522) | 0.379 (0.292 / 0.726) | 0.495 (0.403 / 0.861) | 0.829 (0.806 / 0.920) | 0.906 (0.885 / 0.990) | 0.937 (0.924 / 0.990) |
| Recall (exact) | 0.589 (0.549 / 0.746) | 0.661 (0.611 / 0.863) | 0.730 (0.680 / 0.930) | 0.901 (0.886 / 0.960) | 0.949 (0.937 / 0.995) | 0.968 (0.961 / 0.995) |
| Recall (title-only) | 0.589 (0.549 / 0.746) | 0.661 (0.611 / 0.863) | 0.730 (0.680 / 0.930) | 0.901 (0.886 / 0.960) | 0.949 (0.937 / 0.995) | 0.968 (0.961 / 0.995) |

Latency per query: p50 32.7 ms, p95 38.0 ms (bridge 33.0/38.4; comparison 31.4/36.2 ms).
