# Phase 1 completion report — BM25 baseline RAG

**Phase:** 1, BM25 baseline RAG.

**Objective:**

- Select the BM25 configuration on `tune`.
- Choose K on `dev`.
- Measure the baseline against a closed-book LLM and a gold-context reader reference.
- Build the failure taxonomy.

All of it follows `PREREGISTRATION.md` (Phase 1 and amendment A1).

**Hypotheses (preregistered):**

| Id | Prediction | Result |
|---|---|---|
| H1-a | For bridge questions, BM25 Recall@5 is lower for hop 2 than for hop 1, by at least 0.10 | **Holds**: +0.316 [+0.253, +0.380], `tune`, eligible n = 392 of 400 (8 undetermined) |
| H1-b | Joint Recall@5 is higher for comparison than for bridge questions | **Holds**: +0.238 [+0.130, +0.340], `tune`, 100 vs 400 |
| H1-c | More than half of EM = 0 baseline answers are retrieval-associated | **Not shown**: 32/59 = 0.54 [0.41, 0.66], `dev`. The point estimate is above one half, but the interval includes it |
| H1-d | F1 ordered closed-book < baseline < oracle, oracle below 1 | **Holds**: 0.224 < 0.525 < 0.744, `dev`, n = 100. Paired differences +0.301 [+0.209, +0.395] and +0.219 [+0.133, +0.307]. Oracle interval upper bound 0.818 |

## Files changed

On top of `1998de2`; uncommitted at the time of this report. All paths are under
`projects/hotpot_qa/`.

| File | Status | What it does |
|---|---|---|
| `PREREGISTRATION.md` | new | Common protocol, Phase 1 preregistration, amendment A1 |
| `retrieval.py` | new | `bm25s` BM25 over `Doc.index_text`, deterministic top-k (id tie-break, adapted from FiQA), the 12 preregistered candidates in preference order, `SELECTED` pinned from the grid |
| `prompts.py` | new | Frozen `answer_v1` / `closed_book_v1` and schema v1 (preserved, with recorded hashes); active `answer_v2` / `closed_book_v2` / schema v2; `render_v1`; single-pass slot filling; `artifact_hashes()` |
| `rag.py` | new | Baseline, oracle, closed-book and own-context pipelines; shared answer path with the overflow rule (drop the last displayed paragraph) and `[P#]` → doc-id citation mapping |
| `scoring.py` | new | Per-question scores from records (answer, strict/official evidence, joint, citation validity, context recall, incomplete-gold-evidence proxy), the malformed-answer diagnostic (labels only), summaries with bootstrap intervals, retrieval scores |
| `hops.py` | new | The Phase 0 hop-order rule as a module, plus the taxonomy evidence buckets (including `only_one_hop_undetermined`) |
| `run_eval.py` | new | One CLI: `format-check`, `grid`, `retrieval`, `rag`, `closed-book`, `oracle`. Writes records and manifests; refuses blind splits; allows the v2 format check to run only once |
| `metrics.py` | edited | Added `mean_ci` (single-system bootstrap interval) |
| `manifest.py` | edited | `EVAL_CODE` now covers `data.py`, `metrics.py`, `scoring.py`, `hops.py` and `run_eval.py` |
| `01_baseline.ipynb` | new | Reads results only: selection, curves, H1-a–H1-d, K choice, baseline profile, reference rows, taxonomy, manual labels, summary |
| `tests/test_retrieval.py`, `test_prompts.py`, `test_rag.py`, `test_scoring.py` | new | See below |
| `tests/test_metrics.py` | edited | `mean_ci` test |
| `reports/slides_draft.md` | new | Slides 1–6 drafted |

## Commands run

```bash
python -m pytest -q                              # 132 passed
python run_eval.py format-check                  # v1: 1/20 failures (preregistered criteria)
# amendment A1 written; v2 implemented and tested
python run_eval.py format-check                  # v2, run once: 2/20 failures (expanded criteria)
python run_eval.py grid --split tune             # 12 candidates, retrieval only
python run_eval.py retrieval --split tune        # selected BM25: curves, latency
python run_eval.py rag --split dev_smoke --k 5   # 20 generations
python run_eval.py rag --split dev --k 5         # 80 new + 20 replayed from dev_smoke
python run_eval.py rag --split dev --k 3
python run_eval.py rag --split dev --k 10
python run_eval.py closed-book --split dev
python run_eval.py oracle --split dev
python run_notebook.py 01_baseline.ipynb         # 10 code cells, no errors or stderr
```

**LLM compute:** 540 unique generations taking 1.90 h in total, plus 20 replayed:

| Run | New generations |
|---|---|
| format check v1 | 20 |
| format check v2 | 20 |
| `dev_smoke` | 20 |
| `dev` K = 5 / 3 / 10 | 80 / 100 / 100 |
| closed-book | 100 |
| oracle | 100 |

All cost and latency figures come from original-generation metadata in the cache.

## Tests and results

**132 passed, 0 failed**, including the 2 live Ollama tests. New in this phase:

- `test_retrieval.py` (5): id tie-break at cutoffs and within candidate subsets; the 12
  candidates in preference order; BM25 returns matched documents only, deterministically;
  stemming and stop-word handling of query terms; `describe()`.
- `test_prompts.py` (10):
  - `answer_v1` and `closed_book_v1` equal the preregistered text character for
    character; the v2 answer rule equals amendment A1, and only that rule changed;
  - v1 is preserved (recomputed hashes equal the recorded ones); the v2 hashes equal the
    frozen values recorded in the preregistration;
  - rendering keeps empty sentences numbered; slot filling never re-substitutes inside
    paragraph text; the schema field order is the generation order;
  - schema v2 rejects blank answers and contains no `pattern`; no `description` text is
    sent in any schema.
- `test_rag.py` (5): citations map to displayed documents, and out-of-range paragraph
  numbers become invalid; the overflow rule drops the last displayed paragraphs and
  citation mapping follows the displayed order; a schema failure falls back to an empty,
  unsupported output; the oracle shows the gold paragraphs in supplied order; closed-book
  shows no paragraphs.
- `test_scoring.py` (24):
  - a perfect output scores 1 on every metric; a missing gold paragraph and no citations
    are scored correctly; closed-book has no evidence metrics; outcome buckets;
    summaries skip non-applicable metrics;
  - **the hop rule reproduces the Phase 0 counts on `tune` exactly** (305 / 32 / 29 / 26
    / 8); all six evidence buckets;
  - the malformed-answer diagnostic on 15 cases, including look-alikes that must not be
    flagged ("Eric (musician)", "Star Wars: A New Hope", "1:30"); flags are labels only;
  - **every module that calls a scoring function is in the evaluation-code hash** (an AST
    scan).

## Experiment configuration

| Item | Value |
|---|---|
| BM25 | `bm25s` 0.3.11, method `lucene`, `PyStemmer` 3.1.0 English, `bm25s` English stop list (33 words), k1 = 1.5, b = 0.75, index text `title. text`, ties broken by doc id |
| K | 10 (selected on `dev`) |
| LLM | `qwen2.5:7b` (`845dbda0ea48…`, Q4_K_M), temperature 0, seed 0, `num_ctx` 8192, `num_predict` 512 |
| Answer interface | `answer_v2` `0a2aa59c…`, `closed_book_v2` `a89deeb1…`, schema v2 `c29af85a…`, validator `252c82c0…`, system prompt `16f6b152…`, `render_v1` `935e0b46…` |
| Manifests | All five `dev` runs pass `manifest.check_comparable`, with a shared answer prompt, schema, system prompt and renderer where shared, and one evaluation-code hash (`623d7ee5…`). They ran from uncommitted code on `1998de2` (`dirty = true`), which development comparisons allow |

## Metrics with sample sizes

**BM25 selection** (`tune`, n = 500):

- Selection score (mean Joint Recall@3/5/10): best 0.481 (k1 = 1.2).
- Plateau of 4 candidates; the selected configuration scored 0.478.
- Every stemmed candidate beats its unstemmed counterpart.

Joint Recall of the selected configuration by K (`tune`):

| K | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| All (500) | 0.260 | 0.356 | 0.470 | 0.608 | 0.716 | 0.768 |
| Bridge (400) | 0.230 | 0.325 | 0.422 | 0.550 | 0.660 | 0.720 |
| Comparison (100) | 0.38 | 0.48 | 0.66 | 0.84 | 0.94 | 0.96 |

- Title-only and exact matching differ by at most 0.010.
- Query latency: 0.5 ms p50, 0.7 ms p95 over all 500 `tune` queries after 10 warm-ups;
  index build 5.2 s.

**K selection** (`dev`, n = 100, answer F1):

| K | F1 | 95% interval | Mean prompt tokens | Median LLM time |
|---|---|---|---|---|
| 3 | 0.480 | [0.389, 0.571] | 667 | 7.7 s |
| 5 | 0.454 | [0.363, 0.546] | 946 | 14.6 s |
| 10 | 0.525 | [0.436, 0.612] | 1,716 | 20.6 s |

- Plateau = {10}, so **K = 10**.
- Paired differences: K10 − K5 = +0.071 [+0.007, +0.138]; K10 − K3 = +0.045
  [−0.028, +0.120].

**Baseline at K = 10** (`dev`; all 100 / bridge 80 / comparison 20):

| Metric | All | Bridge | Comparison |
|---|---|---|---|
| Answer F1 | 0.525 [0.44, 0.61] | 0.486 | 0.680 |
| EM | 0.410 | 0.388 | 0.500 |
| Context Joint Recall | 0.620 [0.52, 0.71] | 0.537 | 0.950 |
| Evidence F1, strict (= official-compatible on `dev`) | 0.327 [0.26, 0.39] | 0.270 | 0.553 |
| Evidence precision / recall | 0.415 / 0.297 | | |
| Joint F1 (official) | 0.245 | | |
| Supported rate | 0.75 | | |
| F1 on answered | 0.673 (n = 75) | | |
| Citation coverage | 0.91 | | |
| Per-citation validity | 0.944 (160 citations) | | |
| All citations valid | 0.82 | | |
| Incomplete-gold-evidence proxy | 0.71 | | |

Schema errors 0; malformed-answer flags 0; overflow drops 0.

- **Cost:** 1 retrieval call and 1 LLM call per question. 1,716 prompt tokens and 96
  output tokens on average, of which the rationale is 46 (about 48%). LLM wall time
  20.6 s p50, 26.5 s p95; retrieval 0.7 ms p50.
- **Rationale cost:** only its token count is reported. Its share of latency cannot be
  observed from a single non-streamed generation.

**Reference rows** (`dev`, n = 100):

| Row | F1 | EM | Other | Prompt tokens | Median LLM time |
|---|---|---|---|---|---|
| Closed-book | 0.224 [0.15, 0.30] | 0.150 | Low-confidence rate 0.42; F1 when confident 0.355 (n = 58) | 206 | 5.0 s |
| Oracle | 0.744 [0.67, 0.82] | 0.610 | Evidence F1 0.549; supported 0.95 | 522 | 10.3 s |

**Automatic taxonomy** (`dev`, K = 10), rows = evidence bucket, columns = answer outcome:

| Bucket | Correct | Partial | Wrong | Total |
|---|---|---|---|---|
| Both | 35 | 13 | 14 | 62 |
| Only hop 1 | 5 | 2 | 21 | 28 |
| Only hop 2 | 1 | 0 | 4 | 5 |
| Only one hop, undetermined | 0 | 0 | 1 | 1 |
| Only one entity | 0 | 0 | 1 | 1 |
| Neither | 0 | 1 | 2 | 3 |

Correct answers citing both gold paragraphs: 22 of 41.

## Manual labels (all 25 cases)

- Population: the 59 EM = 0 questions, sorted by id.
- `numpy.random.default_rng(20260929).choice(59, 25, replace=False)`; unstratified.
- Labels assigned in order of precedence. Exploratory, by one annotator who knew the
  outputs; neither blinded nor independent.

| # | Question id | Type | Bucket | F1 | Label | Evidence |
|---|---|---|---|---|---|---|
| 1 | 5a7109e95542994082a3e4fa | bridge | both | 0.50 | granularity | "Tallaght" vs "Tallaght, South Dublin" |
| 2 | 5a722c2255429971e9dc932c | bridge | both | 0.67 | granularity | "SM Lifestyle City" vs "…Cities" |
| 3 | 5a7698c2554299373536010d | bridge | only hop 1 | 0.00 | label_error | Asks the magazine's circulation (1.23 million); gold is the magazine's name |
| 4 | 5a7b1c6b55429931da12c9ca | comparison | both | 0.67 | granularity | "Johnny Edwards" vs his full name |
| 5 | 5a7e782655429934daa2fc26 | bridge | both | 0.80 | granularity | "half a million" vs "half a million acres" |
| 6 | 5a7fb6625542994857a767cc | bridge | only hop 1 | 0.00 | retrieval_hop2 | Helen Mirren paragraph absent |
| 7 | 5a827e8455429940e5e1a8dd | bridge | only hop 1 | 0.00 | retrieval_hop2 | Dame Edna Everage paragraph absent; model answered the author |
| 8 | 5a849c1b5542997175ce1ef9 | bridge | only hop 1 | 0.00 | retrieval_hop2 | Vox Day paragraph absent |
| 9 | 5a85cd295542997175ce202b | bridge | only hop 2 | 0.00 | retrieval_hop1 | Athenion paragraph absent |
| 10 | 5a874f20554299211dda2bd6 | comparison | both | 0.00 | reasoning | Rationale: Schlegel flew two missions; answer: Frimout |
| 11 | 5a898872554299669944a540 | bridge | both | 0.50 | granularity | "Anandji" vs "Anandji Virji Shah" |
| 12 | 5a8a521d5542992d82986f15 | bridge | both | 0.00 | reasoning | National-team paragraph shown; model answered the club |
| 13 | 5ab2a01d5542992953946778 | comparison | both | 0.67 | granularity | "American novelist" vs "American" |
| 14 | 5ab5627b5542992aa134a2ff | bridge | only hop 1 | 0.00 | retrieval_hop2 | United Mine Workers paragraph absent |
| 15 | 5ab5df4c5542997d4ad1f218 | bridge | both | 0.80 | granularity | "25 April 2017" vs "25 April" |
| 16 | 5ab71a6a5542991d3222377c | bridge | only hop 1 | 0.00 | other | Hop-1 sentence and rationale say 1990; the answer field says "goes to number one" |
| 17 | 5ab959905542996be2020497 | bridge | both | 0.00 | granularity | "1964-1974" vs "1964 to 1974"; normalisation joins the hyphenated years, so F1 = 0 |
| 18 | 5aba7cb455429901930fa7eb | bridge | neither | 0.00 | retrieval_both | Neither gold paragraph retrieved |
| 19 | 5abd516a5542992ac4f3825c | bridge | only hop 1 | 0.00 | label_error | Asks a religion; gold "Hindi" is a language |
| 20 | 5ac056385542997d64295a08 | bridge | both | 0.67 | granularity | "The Ring" vs "\"The Ring\" magazine" |
| 21 | 5ac2acff55429921a00ab02b | bridge | only hop 1 | 0.00 | retrieval_hop2 | Brian Doyle-Murray paragraph absent |
| 22 | 5ae1d5a3554299234fd04319 | bridge | neither | 0.00 | label_error | Ambiguous target: the corpus's Corey Dillon paragraph also satisfies the question |
| 23 | 5ae362105542991a06ce99ae | bridge | both | 0.00 | granularity | "8.7–9.2" is the paragraph's range; gold keeps its lower end; F1 = 0 from tokenisation |
| 24 | 5ae4903855429913cc2044a0 | comparison | only one entity | 0.00 | retrieval_entity | *American Dream (film)* absent |
| 25 | 5ae490a85542995ad6573d9d | bridge | only hop 1 | 0.00 | retrieval_hop2 | George Adamski paragraph absent; hop 1 names him, not the attribute |

**Counts:** granularity 10; retrieval 9 (hop 2: 6; hop 1, both and entity: 1 each);
label_error 3; reasoning 2; other 1.

## Representative successes

- **The format check worked as a gate.** It caught empty and code-like answers from
  constrained decoding on train data, before any `dev` call. With v2 there were zero
  empty answers, flagged answers and schema errors across all 500 `dev` outputs and the 20
  `dev_smoke` outputs.
- **Retrieval is worth +0.30 F1 over the bare model** (closed-book 0.22 → 0.53), with an
  interval clear of zero.
- **H1-a pinpoints the mechanism:** second-hop recall 0.52 against first-hop 0.84 on 392
  bridge questions.

## Representative failures

- **Hidden second hop** (#7, *Les Patterson Saves the World*): the film paragraph names
  Barry Humphries and Dame Edna, but the *Dame Edna Everage* paragraph holding "My
  Gorgeous Life" is not in the top 10. The model answers the author instead.
- **Reasoning with full evidence** (#10): both astronauts' paragraphs are present and the
  rationale states Schlegel's two missions, but the answer is Frimout.
- **Answer contradicting its own rationale** (#16): the rationale says 1990; the answer
  field says "goes to number one".

## Known limitations

- **Coarse K choice.** At n = 100, the K choice is coarse (K10 − K3 includes zero), and
  it will only be checked on the blind test sets. The dip at K = 5 below K = 3 is within
  noise.
- **EM undercounts correct answers.** 10 of 25 sampled EM = 0 cases are right in
  substance, and 2 of those score F1 = 0 only through official tokenisation (hyphenated
  and en-dash ranges). The official metric is kept unchanged and the effect is reported,
  not corrected.
- **Label noise is material:** 3 of 25 sampled failures have doubtful gold targets.
- **Citations are sparse:** evidence recall is 0.30, with 1.6 citations per answer; 19 of
  41 correct answers (46%) do not cite both gold paragraphs. The strict evidence metrics therefore
  measure citation completeness as much as grounding.
- **The malformed-answer diagnostic is a lower bound:** `,__1919__` escaped it in the v2
  check.
- **Forced non-empty answers produce arbitrary guesses** when the model gives up (for
  example "Olympic", "unknown", "No data"). `supported = false` marks them (25% at K = 10).
- **The manual labels are one annotator's, unblinded,** with 25 cases.
- **Development runs used uncommitted code** (allowed for development; the final runs
  require a clean commit).

## Departures from the approved plan

1. **Amendment A1** (approved): v2 prompts and schema, the malformed-answer diagnostic,
   one rerun on the same 20 questions, 2/20 failures, frozen. This adds 20 generations
   (540 unique in total, as projected).
2. **K = 10, not the default 5,** selected by the preregistered rule. This raises the
   baseline's per-question cost (1,716 prompt tokens, 20.6 s median).
3. **The hop rule moved from the Phase 0 notebook into `hops.py`,** with a test that it
   reproduces the Phase 0 counts exactly.
4. **`run_eval.py` is in the evaluation-code hash.** Editing it makes older manifests
   incomparable, which is handled by re-running older systems from cache (seconds, no new
   generations).
5. **H1-b's unpaired bootstrap is computed in the notebook,** following the preregistered
   definition (resampling within each type), not by a function in `metrics.py`.

## Suggested improvements (for review; nothing applied)

- **Phase 2 hypothesis direction.** The measured weakness is the second hop, which the
  question does not name. The planned Phase 2 techniques (dense retrieval, fusion,
  reranking) all still query with the question alone. The preregistration should
  therefore:
  - predict their gains mainly for comparison and first-hop recall;
  - track **second-hop Recall@10 on bridge questions** as a named diagnostic;
  - keep the preregistered concern that reranking can demote the second hop.
- **Answer granularity.** The frozen prompt's "shortest answer" instruction plausibly
  contributes to granularity misses. It stays frozen for comparability; the effect is
  visible through EM against F1.
- **Case studies reserved for the deck:** #7 (hidden second hop, a "baseline failure"
  candidate for a later fix) and #10 (reasoning failure with full evidence, a "remaining
  failure" candidate).

## Git diff / commit reference

Committed on branch `hotpot_qa`, on top of `1998de2`, in the commit that adds this report
(`git log -1 -- projects/hotpot_qa/reports/phase_1.md`). The five `dev` runs were produced
before that commit from the same code, and their manifests record `1998de2` with
`dirty = true`, which development comparisons allow. Generated files stay local and
untracked (`results/`, `cache/`, `splits/`).
