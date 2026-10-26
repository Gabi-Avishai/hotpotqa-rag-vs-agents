# HotpotQA: from RAG to multi-agent investigation — implementation plan

Status: **revision 3, Phase 0 approved pending these corrections**. Nothing below is implemented yet.
Every phase starts only after the previous phase has a written review.

## 1. Research question and deliverable

> When does iterative or multi-agent investigation recover enough additional evidence to
> justify its extra latency, compute and complexity over conventional RAG?

We build, evaluate and compare four systems on identical questions: a BM25 RAG baseline, an
improved non-agent RAG, a single investigation agent and a multi-agent investigation. The
interview deliverable is:

- the comparison table the assignment requires;
- a failure analysis and the five required case studies;
- a 15–20 minute deck;
- a demo that replays cached results, so it runs offline and repeats exactly;
- a deployment recommendation.

The deployment recommendation will be determined from the frozen quality, grounding,
latency, call-count and complexity results. No system is expected to win.

**Benchmark status.** This is a closed-corpus retrieval benchmark derived from the HotpotQA
distractor validation contexts. The official answer and supporting-fact metrics are reused,
but scores are **not comparable with the official distractor leaderboard**. There, each
question comes with its own ten candidate paragraphs; here every system searches a shared
66k-paragraph corpus. The README and the deck repeat this statement.

## 2. Measured constraints

All figures were measured on this machine on 2026-09-27.

| Resource | Value | Consequence |
|---|---|---|
| LLM | Local only, no paid APIs | Ollama; every LLM call costs wall-clock time, not money |
| GPU | GTX 1650 Ti, 4 GB VRAM | No 7B+ model fits entirely in VRAM, so the LLM is partly offloaded to the CPU. The GPU is shared between the LLM and the encoders |
| CPU / RAM | Ryzen 7 4800HS (16 threads), 30 GB | Online retrieval (BM25, query encoding, reranking) can run on the CPU |
| Disk | **4.1 GB free** (98% full, after clearing the pip download cache) | No new model downloads beyond a few hundred MB; see §10 |
| Ollama | 0.32.5; models installed: `qwen2.5:7b`, `llama3.1:8b`, `phi3`, `all-minilm` | Pick among installed models |

Throughput on realistic RAG prompts (numbered-sentence paragraphs, JSON output,
temperature 0). Benchmark questions came from the **train** split only, three per row.
The first three rows used `num_ctx` 4096; the last row repeats `qwen2.5:7b` at the final
`num_ctx` 8192:

| Model | `num_ctx` | In VRAM | Prompt tok/s (5 / 10 paragraphs) | Output tok/s | Wall time per call (5 / 10 par.) | Correct (of 6, informal) |
|---|---|---|---|---|---|---|
| `phi3` 3.8B | 4096 | 2.7 of 3.9 GB | ~185 / ~350 | ~24 | 6.5–7.7 s / 6.3–8.9 s | 2 |
| `qwen2.5:7b` | 4096 | 2.7 of 5.1 GB | ~110 / ~210 | ~14 | 9.2–10.8 s / 8.7–11.8 s | 4 |
| `llama3.1:8b` | 4096 | 2.7 of 5.6 GB | ~95 / ~180 | ~12 | 11.0–13.1 s / 12.4–12.8 s | 3 |
| **`qwen2.5:7b`** | **8192** | 2.7 of 5.4 GB | ~110 / ~210 | ~13 | 9.5–10.7 s / 8.8–11.8 s | 4 |

**Model: `qwen2.5:7b`**, pinned by its Ollama digest `845dbda0ea48` (Q4_K_M quantization,
to be confirmed and recorded in Phase 0).

- It was the most accurate on the samples and produced compact JSON that respects the
  schema.
- The larger context costs no measurable speed; the extra KV cache sits in CPU memory.
- Most of the time goes to prompt processing, so the number of paragraphs in context drives
  latency.
- Six questions are a smoke test, not an evaluation. The model choice is fixed now and will
  not be revisited after results are seen.
- `num_ctx` changed one output's length (29 → 15 tokens) at temperature 0. It is fixed at
  8192 for every system and recorded in every run manifest.

## 3. Working agreement

- Phase-gated: Claude proposes → Sol reviews → Claude implements only the approved phase →
  Sol inspects code, tests, outputs and failures → written review → next phase.
- Each phase ends with a completion report saved as `reports/phase_N.md`. It covers:
  Phase, Objective, Hypothesis, Files changed, Commands run, Tests and results, Experiment
  configuration, Metrics with sample sizes, Representative successes, Representative
  failures, Known limitations, Departures from the approved plan, Suggested improvements,
  and a Git diff/commit reference.
- Hypotheses, candidates and decision rules are appended to `PREREGISTRATION.md` **before**
  the results they govern are opened. Later changes are dated amendments that state their
  effect.
- Every reported metric states its split and sample size.
- Prompts, models, subsets and metrics are not changed after results are seen, except by a
  logged amendment.
- Core behaviour lives in importable, tested modules; notebooks only load results and plot
  or show them. Long LLM runs are resumable scripts, not notebook cells.
- New project directory `projects/hotpot_qa/`; `projects/fiqa_search` is not modified.
  FiQA utilities are read, adapted and re-tested, not copied blindly (§8).

## 4. Data design

### 4.1 Source

`hotpotqa/hotpot_qa`, `distractor` configuration, pinned to a Hugging Face commit hash
recorded in `data.py`. Facts checked on 2026-09-27 (and independently by review):

- Validation: 7,405 questions, **all labelled `hard`**; 5,918 bridge (79.9%), 1,487
  comparison (20.1%). Stratifying by difficulty is therefore impossible on validation;
  stratification is by question type only.
- Every validation question has exactly two gold titles, both present in its supplied
  context.
- 7,345 questions have ten context paragraphs and 60 have between two and nine, giving
  73,700 paragraph slots.
- The official test split has no labels, so every evaluated question comes from validation.

### 4.2 Shared closed corpus

- **Corpus = the context paragraphs of all 7,405 validation questions**: 73,700 paragraph
  slots, **66,581 unique titles**. The retriever searches this shared corpus. No system ever
  sees a question's own candidate set.
- Why all of validation, not only the evaluated questions: a corpus built from ~2k
  questions is small enough that gold paragraphs rarely have to compete, which would
  overstate every retriever.
- **Document identity is `(title, text_hash)`**, where `text_hash` is the SHA-256 of the
  paragraph's joined sentences.
  - 54 titles occur with two different texts; each variant is its own document.
  - A question's gold documents are the two `(title, text_hash)` variants in **its own**
    supplied context.
  - Primary retrieval metrics count a hit only for the exact gold variant.
  - Title-only recall is reported as a secondary compatibility metric.
  - Citations `(doc_id, sentence_id)` are validated against the exact variant that was in
    the context.
  - Phase 0 reports how many questions in each split are affected.
- Indexed text = `title + ". " + paragraph text`, with sentences kept separately so the LLM
  can cite them.
- Disclosure: HotpotQA picked each question's distractors by bigram TF-IDF similarity to
  *that* question. The corpus therefore contains near-miss paragraphs for exactly the
  evaluated questions. This affects every system and every split equally, and makes
  lexical retrieval harder than a random Wikipedia sample would.

### 4.3 Splits

Seeded (`seed = 20260927`) and stratified by type. `data.make_splits()` draws the question
ids deterministically.

| Split | Questions | Source | Use | Exposure |
|---|---|---|---|---|
| `eda` | 1,500 | train | Cycle 0 EDA; prompt-format smoke tests on each question's **own supplied context** (oracle-style), never on the shared corpus | open |
| `tune` | 500, proportional | validation | retrieval-only selections: BM25 settings, fusion, reranker, pool depth. **No LLM** | open, aggregates |
| `dev` | 100, proportional; a fixed 20-question prefix for smoke runs | validation | all LLM work: choosing K, prompt debugging, failure taxonomy, agent development, case-study candidates | open |
| `test_retrieval` | 1,000, proportional | validation | final retrieval-only run (no LLM): Recall and Joint Recall curves for baseline and frozen improved retrieval | **blind** |
| `test_rag` | 200, proportional (≈160 bridge / 40 comparison), drawn from `test_retrieval` | validation | baseline and improved RAG plus reference rows: the more precise estimate for non-agent systems | **blind** |
| `test_agents` | 100, **50 bridge + 50 comparison**, drawn from `test_retrieval` independently of `test_rag` | validation | the mandatory four-system table; every system runs on these same 100 | **blind** |

- `tune`, `dev` and `test_retrieval` are disjoint. `test_rag` and `test_agents` are nested
  in `test_retrieval` and may overlap each other; each is used only for its own
  comparisons.
- The remaining ~5,800 validation questions contribute only their paragraphs to the corpus.
- Train questions cannot be run end to end on the shared corpus: their gold paragraphs are
  not in it.
- `test_agents` results are reported per type, plus an overall figure weighted by
  validation prevalence (0.799 bridge, 0.201 comparison), bootstrapped within type.
- `data.load_split("test_*")` raises `BlindSplitError` unless `unblind=True`. Only the final
  evaluation script passes that flag, after the freeze in Phase 5.
- **Sample sizes are fixed now** and do not depend on how long later phases take.

## 5. Metrics

| Category | Metric | Notes |
|---|---|---|
| Answer (primary) | HotpotQA normalised token **F1**; EM | Port of the official `hotpot_evaluate_v1` normalisation, including yes/no handling; tested against hand-made cases |
| Retrieval (primary) | **Final-context Joint Recall**: share of questions whose two gold variants both reach the answer-generating prompt | Comparable across RAG and agents |
| Retrieval (diagnostic) | **Cumulative search Joint Recall**: both gold variants were retrieved at some step | Agents only; the gap to final-context recall measures evidence found and then dropped |
| Retrieval | Recall@K and Joint Recall@K, K ∈ {2, 3, 5, 10, 20, 50}; title-only variants as secondary | Retrieval-only curves on `tune` / `test_retrieval` |
| Sentence evidence (primary) | **Strict variant-aware** supporting-fact precision / recall / F1 over `(title, text_hash, sentence_id)` | Custom to this corpus; from the LLM's citations |
| Sentence evidence (secondary) | **Official-compatible** supporting-fact P/R/F1 over `(title, sentence_id)`, and the official joint answer/evidence F1 | Same definitions as `hotpot_evaluate_v1`; only these are called "official" |
| Grounding | Citation validity (the cited sentence exists in the exact variant that was in context); **incomplete-gold-evidence proxy** (citations miss a gold document); **blinded manual audit** (§6, Phase 6) | The proxy is not proof of an unsupported answer. The manual audit gives the primary grounding measure, the unsupported-claim rate |
| Behaviour | Abstention (`supported = false`) rate; **F1 on answered questions** = mean F1 over outputs with `supported = true`, reported with its count, and reported as "undefined (0 answered)" when no output is marked supported; stop-reason distribution; searches per question | Every system always outputs a best-guess answer **and** a `supported` flag; the primary F1 is computed on all questions |
| Cost | LLM calls; prompt and output tokens (Ollama counts); retrieval calls; **LLM wall-seconds** per question | Wall-clock, not GPU-seconds, because the model is partly on the CPU |
| Latency | End-to-end p50 / p95 per question, with the LLM and retrieval shares shown separately | Warm model, one question at a time |
| Engineering | Qualitative complexity/debuggability: nodes, LLM decision points, lines of orchestration code, trace length | Stated as judgement, not measurement |

**Statistics.**

- Paired bootstrap (10,000 resamples, add-one p-values) on identical question sets. The
  function refuses mismatched id sets. Tests are **two-sided**, since a more complex system
  can also be worse. Confidence intervals are two-sided 95% percentile intervals.
- **Confirmatory family**, on `test_agents` (n = 100), Holm-corrected: baseline → improved,
  improved → single agent, single agent → multi-agent.
- **Secondary**: baseline → improved on `test_rag` (n = 200), a more precise estimate of the
  non-agent gain.
- **Minimum practically meaningful difference: 3 points on the 0–1 scale** (+0.03 answer F1,
  +0.03 final-context Joint Recall). It is necessary, not sufficient: a deployment
  recommendation also weighs cost, latency and complexity.
- Expected precision: the 95% interval of a paired F1 difference is roughly ±0.07 at
  n = 100 and ±0.05 at n = 200. Differences inside those bounds are reported as "not
  demonstrated", with the interval. Confidence intervals are primary; p-values are
  secondary.

## 6. Phases

Every phase lists its hypothesis, the decisions it makes and on which split, and its gate.
Dates are indicative, and the user manages the actual schedule. The original interview
date (2026-10-04) has been postponed to a date not yet known. The calendar dates in the
headings below are the original plan. They will be replaced by milestone-based
checkpoints once the new date is known.

**Schedule update (2026-10-04).**

- Phase 4 completes with the frozen `verifier_v2` system on `dev`, followed by an
  analysis of why it succeeds or fails.
- Phase 5 (blind test) does not start until Phase 4b below has been decided. (Decided 2026-10-06: Phase 4b
  rejected; Phase 5 uses the frozen Phase 4 system.)
- Nothing runs overnight or is rushed for a deadline.

### Phase 0: data and evaluation contract (Mon 09-28)

- `data.py`: pinned load, corpus build with `(title, text_hash)` identity (§4.2), splits and
  blind guard (§4.3), corpus fingerprint.
- `metrics.py`: answer EM/F1; supporting-fact P/R/F1 in both strict variant-aware and
  official-compatible forms; official joint F1; Recall@K and Joint Recall@K (exact-variant
  and title-only); paired bootstrap; prevalence-weighted aggregation; latency stats.
- `llm.py`: an Ollama HTTP client with:
  - JSON-schema-constrained outputs (`format` = the Pydantic schema);
  - a disk cache keyed by model digest + prompt + options + schema;
  - token and wall-time accounting;
  - a hard check that the prompt fits `num_ctx` (8192), because Ollama truncates silently.
- `manifest.py`: writes a run manifest next to every results file (§8).
- Tests for all of the above (`tests/`).
- `00_eda.ipynb` on `eda` (train) plus aggregates on `tune`:
  - 3–5 worked examples with the evidence chain drawn for bridge and comparison questions;
  - paragraph and sentence lengths;
  - how often the answer string appears in a gold paragraph;
  - lexical overlap between the question and each gold paragraph, first vs second hop.
- Gate: tests pass; split sizes, disjointness, nesting and manifest hashes verified; corpus
  statistics reported.

### Phase 1: baseline RAG (Tue 09-29)

- BM25 (`bm25s`) over the corpus. A small grid on `tune`:
  - stemming on/off × stop words on/off;
  - `k1`/`b` at the defaults plus 2 alternatives.
  Selection is by the **mean of Joint Recall@{3, 5, 10}**, a pre-declared aggregate over
  the K values Phase 1 may choose. The conventional setting comes first in the preference
  order.
- One answer prompt with a fixed schema: `answer`, `citations`, `supported`, short `rationale`.
- **K is chosen on `dev`** among {3, 5, 10} by answer F1. K = 5 is the preferred default
  inside the plateau (≤ 0.02 F1 from the best). Cost: 300 calls, about 1 h.
- Reference rows on `dev`, and later on both test sets:
  - **closed-book** (no retrieval), which answers "why not a direct LLM?";
  - **oracle** (the two gold paragraphs only), the ceiling that separates retrieval failure
    from generation failure.
- Failure taxonomy on `dev`:
  - automatic buckets: neither gold variant retrieved / only one / both retrieved but the
    answer is wrong / correct answer with invalid or incomplete citations;
  - plus a hand classification of ~25 failures.
- Deliverable: baseline numbers on `dev` (n = 100) and on `tune` for retrieval (n = 500),
  the taxonomy, and one baseline failure reserved for the deck.

### Phase 2: improved non-agent RAG (Wed 09-30)

- Hypothesis, to be finalised from the Phase 1 taxonomy before any Phase 2 result is
  opened:
  - dense retrieval recovers gold paragraphs that share few words with the question;
  - fusion raises Joint Recall at the LLM's K;
  - reranking raises precision at small K, **but** may lower Joint Recall on bridge
    questions. A cross-encoder scores each paragraph against the question alone, and the
    second-hop paragraph is about the bridge entity.
- Pre-registered candidates, selected on `tune` by Joint Recall@K (K from Phase 1),
  retrieval only:
  - BM25 (the incumbent, first in preference order);
  - TAS-B;
  - RRF(BM25, TAS-B) at pool depths {50, 100};
  - each RRF pool reranked by `cross-encoder/ms-marco-MiniLM-L6-v2` or `-L12-v2`.
- Model exposure to HotpotQA, graded from model cards; the candidates are only models with
  no known direct exposure:

  | Model (revision) | Evidence | Grade |
  |---|---|---|
  | `sentence-transformers/msmarco-distilbert-base-tas-b` (`b12d935`) | Card: trained on MS MARCO | no known direct exposure — **candidate** |
  | `cross-encoder/ms-marco-MiniLM-L6-v2` (`233902d`), `-L12-v2` (`7b02352`) | Card: trained on MS MARCO | no known direct exposure — **candidate** |
  | `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5` | Card lists HotpotQA only as an MTEB evaluation set; fine-tuning mixture not itemised in the card | undocumented / possible — not used |
  | `intfloat/e5-base-v2` | Same as above | undocumented / possible — not used |
  | `intfloat/multilingual-e5-base` | Card: "KILT HotpotQA", 70k pairs in supervised fine-tuning | known direct exposure — not used |
  | `BAAI/bge-m3` | `sentence-transformers/hotpotqa` card: "used to train the BGE-M3 model" | known direct exposure — not used |

- Corpus encoding runs on the GPU with Ollama unloaded. Online retrieval runs on the CPU, so
  the LLM keeps the GPU.
- The winner runs once end to end on `dev` against the baseline.
- Deliverable: retrieval curves on `tune`, per-type breakdown, the dev comparison with a
  confidence interval, cost per query. The improved retriever is frozen; it becomes the
  agents' search tool.

### Phase 3: single investigation agent (Thu 10-01)

- LangGraph `StateGraph`. The LLM decides; the tools are deterministic.
  - State: `question, queries, seen_docs, context, missing_info, step, stop_reason, answer,
    citations, supported, trace`.
  - Nodes:
    - `search` is deterministic: the frozen Phase 2 retriever ranks the corpus for the
      query and returns **exactly the 5 highest-ranked paragraphs not already seen**.
      The **first search uses the original question** verbatim; later queries come from
      `assess`.
    - `assess` is an LLM call returning JSON `{sufficient, missing_info, next_query,
      justification}`.
    - `answer` uses the same answer prompt and schema as RAG.
  - A conditional edge on `sufficient`, the step limit and the no-progress checks.
- Hard budget:
  - at most **3 searches**, so at most **15 paragraphs**;
  - at most **4 LLM calls** (3 `assess` + 1 `answer`);
  - no pruning: the answer node sees every retrieved paragraph, so final-context and
    cumulative Joint Recall coincide for this system by construction;
  - a duplicate query (after normalisation) is rejected and ends the loop with stop reason
    `duplicate_query`;
  - a search that returns no unseen paragraph ends the loop with `no_progress`.
- Stop reasons: `sufficient`, `max_searches`, `duplicate_query`, `no_progress`.
- The trace logs queries, retrieved ids, decisions, a one-line justification and the stop
  reason. No hidden chain of thought is stored.
- What makes it agentic: query *n+1* is written from the evidence returned by query *n*
  (typically the bridge entity).
- **Matched-context control**, fixed here on `dev`: improved RAG with K equal to the agent's
  mean paragraphs in context on `dev`, rounded. This separates "iteration helped" from
  "more context helped".
- Hypothesis, to be finalised before opening results: iteration raises final-context Joint
  Recall and F1 on bridge questions, and adds little on comparison questions, which name
  both entities.
- Diagnostics on `dev`: over-searching, early stopping, loops, and the stop-reason mix.

### Phase 4: multi-agent investigation (Fri 10-02)

Same model, same search tool (5 unseen paragraphs per search), same answer schema as
Phase 3. Roles run **sequentially**, because a second hop depends on the first and there is
one GPU.

| Role | Input | Output | LLM calls |
|---|---|---|---|
| **Planner** | question | 1–2 evidence requirements, and the query for the first | 1 |
| **Researcher** (one call per requirement, in order) | requirement, evidence accepted so far, the new search results | accepted paragraphs with sentence ids, a short finding, and the query for the next requirement (written from this finding) | 1 per requirement (≤ 2) |
| **Verifier** | question, requirements, the text of the accepted paragraphs. **Not** the researcher's findings, conclusions or trace | per requirement: supported or not, with cited sentences; overall `sufficient`; if not sufficient, one corrective query for one requirement | ≤ 2 (initial, and after the one correction) |
| **Synthesizer** | question and the accepted paragraphs only | the standard answer schema, produced by the **same frozen answer prompt** as every other system; any difference must be justified and preregistered | 1 |

- Hard budget:
  - at most **2 requirements**, because HotpotQA questions are two-hop or two-entity;
  - at most **1 search per requirement** plus **1 verifier-requested corrective search**,
    so at most **3 searches**, the same as the single agent;
  - at most **7 LLM calls** (1 + 2 + 1 + 1 corrective researcher + 1 verifier recheck + 1);
    typically 5;
  - final context = the accepted paragraphs, at most **15**;
  - the same duplicate-query and no-progress rules as Phase 3.
- After the corrective round, the synthesizer runs regardless. The verifier's final verdict
  is logged beside the synthesizer's `supported` flag.
- Final-context and cumulative Joint Recall can differ here, because only accepted
  paragraphs reach the synthesizer.
- Hypothesis: independent verification reduces premature stopping and unsupported answers,
  and adds little when both gold paragraphs are already in context.
- Deliverable: dev results, one case where it helped, one where it added cost for nothing.

### Phase 4b: verifier engineering iteration (after the Phase 4 analysis)

- **Why:** the A5 calibration showed that the verifier is the multi-agent system's main
  bottleneck. `verifier_v2` reached S3 0.40 and S4 0.60 on C2, with a 40% hard-failure rate
  on the probes; the prompt-only revision (`verifier_v3`) was rejected.
- **Approach:** a separate iteration that addresses the verifier structurally. Candidates
  include:
  - constrained JSON generation whose shape follows the requirement count;
  - simpler per-requirement verifier calls;
  - deterministic aggregation.
- **Data:** fresh training questions (calibration subsets disjoint from the format check,
  C1 and C2).
- **Process:** preregistered before any result. The improved design is frozen before it is
  compared on `dev`, and later on `test_agents`.
- **Scope:** it does not change the frozen Phase 4 system or its `dev` results.

**Outcome (2026-10-06): rejected at C4, before `dev`.**

- **Why:** selection retention failed its gate on the validation set (G2 38/45 = 0.844,
  against ≥ 0.95).
- **What did work:** structured outputs removed contract failures (0 in 458 calls), and
  the per-requirement verifier met its targets on both subsets.
- **What did not:** prompt-based evidence selection by the 7B model stayed unreliable.
- **Consequence:** the frozen Phase 4 system remains the multi-agent system for Phase 5
  (`PREREGISTRATION.md`, Phase 4b record).

### Phase 5: freeze and test (Fri 10-02, overnight)

- `PREREGISTRATION.md` is completed before unblinding: every system configuration, prompt
  and schema hashes, the planned comparisons and the metrics.
- The code is committed before the run, so each run manifest names a commit.
- One command runs everything, resumable through the LLM cache:
  `python run_eval.py --split test_retrieval,test_rag,test_agents --systems all --unblind`.
  - `test_retrieval` (retrieval only, **no LLM calls**): baseline BM25 and the frozen
    improved retriever, giving Recall@K and Joint Recall@K curves on all 1,000 questions.
  - `test_rag`: baseline, improved, closed-book, oracle.
  - `test_agents`: baseline, improved, matched-context, closed-book, oracle, single agent,
    multi-agent.

### Phase 6: analysis and packaging (Sat 10-03)

- Comparison table and per-type breakdown with confidence intervals.
- Plots: quality vs cost, and Joint Recall vs answer F1.
- **Blinded audit.**
  - Sample: 24 `test_agents` questions (seeded, 12 bridge + 12 comparison) × the 4 systems =
    **96 outputs**, shuffled, with system names hidden.
  - Labelled by the user, not by an LLM, after the freeze.
  - Two independent fields per output, defined in `PREREGISTRATION.md` before auditing:
    - evidence support: `fully_supported` / `partially_supported` / `unsupported` /
      `contradicted` / `cannot_determine`;
    - citation validity: `valid` / `invalid`.
  - **Primary manual measure: the unsupported-claim rate.** Among outputs a system marked
    `supported = true`, the share whose support label is `unsupported` or `contradicted`.
  - Reported separately: partial support, invalid citations, abstentions, and the support
    distribution over all outputs.
  - All reported as counts per system with exact binomial intervals.
  - Expected labelling time: about 1.5–2 h.
- Case studies: the five required examples, taken from `dev`, or from `test_*` only after
  the results are frozen.
- README: the research progression, the benchmark-status statement (§1), the deployment
  recommendation drawn from the frozen results, and limitations.

### Phase 7: deck and demo (Sat 10-03)

- 12-slide deck following the assignment's order. The skeleton is started after Phase 1:
  each phase report from Phase 1 on adds its slide draft (architecture, numbers on `dev`, one
  example), so Saturday only replaces `dev` numbers with frozen test numbers and adds the
  comparison and recommendation slides.
- The cached demo, replaying traces from the LLM cache.
- A clean run from the README, meaning a clean execution in the documented environment
  (`pytorch_env` plus the listed packages). It is not a rebuild of the environment from
  scratch, since no lockfile is committed.
- Rehearse the nine defence questions.

## 7. Compute budget (`qwen2.5:7b` at `num_ctx` 8192, projected from the benchmark)

| Run | Calls per question | Time per question | `dev` (n = 100) | `test_rag` (n = 200) | `test_agents` (n = 100) |
|---|---|---|---|---|---|
| Closed-book | 1 (short prompt) | ~3 s | 5 min | 10 min | 5 min |
| Oracle | 1 | ~8 s | 15 min | 30 min | 15 min |
| Baseline / improved RAG (K = 5) | 1 | ~11 s | 20 min each | 40 min each | 20 min each |
| Choosing K (3 values) | 1 | ~8–15 s | ~1 h | — | — |
| Matched-context RAG (~12 paragraphs) | 1 | ~15 s | 25 min | — | 25 min |
| Single agent | ≤ 4 | ~50 s | ~1.4 h per iteration | — | ~1.4 h |
| Multi-agent | ≤ 7, typically 5 | ~65 s | ~1.8 h per iteration | — | ~1.8 h |

- Final test run: a few minutes for `test_retrieval` (no LLM), about 2 h for `test_rag`
  plus about 4.5 h for `test_agents`, fewer where
  the two sets overlap (the cache replays shared calls). That fits in one night.
- `dev` work: about 8–12 h in total, including agent iterations. Development uses the
  20-question `dev` prefix for smoke runs; the full 100 run only at declared decision points.

## 8. Code layout, reuse and provenance

```text
projects/hotpot_qa/
  PLAN.md  PREREGISTRATION.md  README.md  reports/
  data.py        corpus, splits, blind guard, expected hashes
  metrics.py     answer / supporting-fact / retrieval metrics, bootstrap, latency
  manifest.py    run manifests
  retrieval.py   BM25, deterministic top-k
  dense.py       TAS-B encoding with a cache keyed on model revision + corpus fingerprint + settings
  fusion.py      RRF
  rerank.py      cross-encoder reranking
  llm.py         Ollama client, cache, accounting
  prompts.py     versioned prompt templates and output schemas (hashed into the cache key)
  rag.py         baseline / improved / matched / closed-book / oracle pipelines (plain Python)
  agent.py       single-agent LangGraph graph
  multi_agent.py multi-agent LangGraph graph
  run_eval.py    one command: systems × split → per-question records and the summary table
  tests/
  00_eda.ipynb  01_baseline.ipynb  02_improved.ipynb  03_agent.ipynb
  04_multi_agent.ipynb  05_comparison.ipynb
  cache/  results/  splits/   (local only)
```

- Retrieval, generation, evaluation and orchestration are separate modules. LangGraph
  appears only in `agent.py` and `multi_agent.py`.
- Adapted from `semantic-search-fiqa`, each with its own tests: `fusion.rrf`,
  `retrieval.deterministic_top_k` (id tie-break), `metrics.paired_bootstrap` and
  `metrics.latency_stats`.
- The LLM is called through Ollama's HTTP API directly, not `langchain-ollama`, so caching
  and token accounting stay under our control.

**What is and is not committed.** The repository's root `.gitignore` excludes generated
files by type (`*.json`, `*.csv`, `*.txt`, `*.png`, `*.parquet`, …). The repository owner
does not want generated deliverables uploaded: no split manifests, `requirements.txt` or
result files. So the project neither adds ignore exceptions nor switches to other file
formats to get them tracked. Reproducibility comes from code instead:

- **Splits**: `data.make_splits()` regenerates the manifests deterministically. `data.py`
  stores the expected SHA-256 of each split's sorted id list and of the corpus fingerprint.
  A clean clone either reproduces identical splits and corpus or fails loudly.
- **Run manifests**: a manifest is written locally next to every results file, in three
  sections:
  - `comparison_invariants`: dataset revision, corpus fingerprint, split-manifest hash,
    evaluation-code version, base LLM digest and quantization, `num_ctx`, metric
    definitions version, code commit, package versions;
  - `system_config`: retriever and reranker (with model revisions), K, system prompts and
    schemas (with hashes), agent graph and limits, seed;
  - `run_metadata`: timestamp, runtime, machine information.

  The evaluation-code hash covers `data.py` and `metrics.py`, and from Phase 1 also
  `run_eval.py` and every module that turns system outputs into per-question scores; a
  test checks that list against the modules that call the scoring functions.

  A comparison requires identical `comparison_invariants`; `system_config` differences are
  allowed and printed with the result, and `run_metadata` is ignored. Where the experiment
  claims a component is shared, the comparison also asserts it is equal. In particular, the
  answer-generation prompt and schema hashes must match across baseline, improved,
  matched-context and the single agent's `answer` node.
- **Committed**: code, tests, the executed notebooks (with their figures inline), the
  Markdown docs (README, PLAN, PREREGISTRATION, phase reports), and the summary tables in
  README, as in the FiQA project.
- **Environment**: package versions are recorded in every run manifest and listed in the
  README; no lockfile is tracked.

## 9. Environment

- `pytorch_env` already has `bm25s` 0.3.11, `sentence-transformers` 6.1.0, `faiss` 1.15.1,
  `datasets` 5.0.0, `torch` 2.12.1 and `pandas` 3.0.3.
- To add: `langgraph` (1.2.12). A dry run shows it installs only new packages (including
  `langchain-core` and `pydantic`) and changes nothing that is already installed. About 30 MB.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Disk: 4.1 GB free | Budget: TAS-B embeddings of 66.6k paragraphs ≈ 100 MB (fp16); BM25 index and caches small; total well under 1 GB. If more space is needed, the user can choose to remove Ollama models this plan does not use (`llama3.1:8b` 4.9 GB, `phi3` 2.2 GB) |
| A 7B model controls the agent poorly | Few, simple, schema-constrained decisions per node; hard budgets. A weak agent is reported as a finding, not tuned indefinitely: time-boxed to its phase |
| GPU contention | Offline encoding with Ollama unloaded; online retrieval on the CPU |
| Silent context truncation | Assert prompt tokens < `num_ctx` on every call |
| Non-determinism | Temperature 0 with a fixed seed does not guarantee bitwise-identical first generations, since the partly-offloaded computation may not be deterministic. The cache guarantees **replayability**: every reported number comes from cached responses, and reruns replay rather than regenerate |
| Schedule slip | Sample sizes are fixed (§4.3). Cut order: fewer `dev` iterations → fewer hand-classified failures (not below 20) → simpler plots. Never cut: the comparison, the failure analysis, the cost accounting, the audit |
| LLM memorisation of Wikipedia / HotpotQA (qwen2.5 training data undocumented) | The closed-book row measures it; the citation and support metrics show whether answers are grounded |

## 11. Deliberately not used

- **Paid APIs / larger models**: excluded by budget and by the 4 GB GPU.
- **Full-Wikipedia index**: the infrastructure work would crowd out the research question.
- **Fine-tuning** (retrievers or LLM): rejected for time and experimental clarity, because it
  would confound the architecture comparison. The assignment separately forbids training or
  fine-tuning a large LLM.
- **LLM query rewriting in Cycle 2**: it adds LLM calls and blurs the line with the agent.
  Cycle 2 stays LLM-free on the retrieval side.
- **Vector database**: `faiss`/NumPy in memory is enough for 66.6k paragraphs.
- **GraphRAG / knowledge-graph construction**: disproportionate for two-hop questions.
- **CrewAI / AutoGen open-ended conversations**: harder to bound, reproduce and attribute
  than an explicit state graph.
- **Google ADK implementation**: named in the job ad, but building the system twice would
  add no evidence. The deck maps the graph onto ADK's workflow agents instead.
- **LLM-as-judge as ground truth**: at most a secondary signal; unsupported answers are
  labelled by hand.
- **Web UI**: lower interview value than traces and failure analysis.

## 12. Open questions for review

None blocking Phase 0.
