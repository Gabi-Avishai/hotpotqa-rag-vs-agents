# Preregistration

Decisions, hypotheses and selection rules, recorded **before** the results they govern
are opened. Later changes are dated amendments that state their reason and effect (§A).
`PLAN.md` describes the project; where the two differ, this file governs the analysis.

## 0. Protocol common to all phases

- **Data.** `hotpotqa/hotpot_qa`, `distractor`, revision `1908d6af…`. Shared corpus of
  66,635 paragraphs, fingerprint `67db29c2…`. Split ids are fixed by
  `data.EXPECTED_SPLIT_HASHES` and checked on every load and every manifest.
- **Split roles.**
  - `tune` (n = 500): retrieval-only selection; no LLM.
  - `dev` (n = 100): all LLM work, exploratory.
  - `dev_smoke` (n = 20, inside `dev`): pipeline smoke runs only; no decision rests on its
    accuracy.
  - `eda` (train, n = 1,500): prompt-format checks on each question's own supplied context.
  - `test_retrieval`, `test_rag`, `test_agents`: blind until Phase 5.
- **Benchmark status.** A closed-corpus benchmark derived from HotpotQA distractor
  validation contexts; official answer and evidence metrics are reused, but scores are not
  comparable with the official leaderboard.
- **Primary metrics.**
  - Answer: token F1, with EM alongside.
  - Retrieval: final-context Joint Recall on exact paragraph variants.
  - Evidence: strict variant-aware supporting-fact F1. Official-compatible supporting-fact
    F1 and joint F1 are secondary.
- **Minimum practically meaningful difference (MPI):** 3 points on a 0–1 scale. Necessary,
  not sufficient, for a deployment preference.
- **Statistics.**
  - Paired bootstrap, 10,000 resamples, seed 0.
  - Two-sided tests; two-sided 95% percentile intervals; add-one p-values.
  - The confirmatory family is fixed in `PLAN.md` §5 (three adjacent comparisons on
    `test_agents`, Holm-corrected). Everything on `tune` and `dev` is exploratory: point
    estimates with intervals, and no p-values used for claims.
- **Every system output** carries a best-guess `answer`, `citations` and a `supported`
  flag. The primary F1 is over all questions. "F1 on answered" is the mean F1 over outputs
  with `supported = true`, reported with its count, or as undefined when there are none.
- **LLM.** `qwen2.5:7b`, digest `845dbda0ea48…` (Q4_K_M), raw ChatML with the system prompt
  in `llm.DEFAULT_SYSTEM`. Temperature 0, seed 0, `num_ctx` 8192, `num_predict` 512.
  Responses are cached. Reported numbers come from cached responses.

## Phase 1 — BM25 baseline RAG

Recorded 2026-09-29, before any Phase 1 code exists and before any retrieval or LLM
result on `tune` or `dev` has been produced.

### 1.1 Questions

1. Which BM25 configuration to use on this corpus (`tune`, retrieval only)?
2. How many retrieved paragraphs K the baseline gives the LLM (`dev`)?
3. How the baseline performs, and how its failures divide across the failure taxonomy
   (`dev`, descriptive)?
4. Where the baseline sits between a closed-book LLM and an oracle given the gold
   paragraphs (`dev`, descriptive)?

### 1.2 Hypotheses (exploratory; they motivate Phase 2, they are not confirmatory)

The predictions come from the Phase 0 EDA. Each is stated with the estimate that will be
reported, so a failed prediction is visible.

| Id | Prediction | Split, estimate |
|---|---|---|
| H1-a | For bridge questions, BM25 retrieves the second-hop paragraph less often than the first: Recall@5 for hop 2 is lower than for hop 1 by at least 0.10 | `tune` bridge. Nominal sample 400; **eligible paired sample 392**; **8 excluded** because the Phase 0 rule leaves their hop order undetermined. All three counts are reported. Paired difference hop 1 − hop 2 in per-question hit@5 over the 392, with a bootstrap interval |
| H1-b | Joint Recall@5 is higher for comparison questions than for bridge questions | `tune` (400 bridge, 100 comparison). Difference with an unpaired bootstrap interval (resampling within type) |
| H1-c | Among baseline answers with EM = 0 at the selected K, more than half are **retrieval-associated failures**: at least one gold paragraph is missing from the context. This is an association: it does not show that retrieval rather than generation caused the error, since both can fail on the same question | `dev`. Share with a bootstrap interval; the count is also reported |
| H1-d | Answer F1 is ordered closed-book < baseline < oracle, and the oracle stays below 1 | `dev` (n = 100). Three point estimates with intervals; paired differences with intervals. The oracle is a **gold-context reader reference** (an approximate reader ceiling), not a guaranteed ceiling. Its shortfall from 1 can come from reader errors, annotation noise, answer granularity, schema errors or metric mismatch, and is not attributed to generation alone |

If H1-a and H1-c hold, the Phase 2 hypothesis will target second-hop retrieval. If they
fail, Phase 2 is re-planned from the observed taxonomy before any Phase 2 result is
opened.

### 1.3 BM25 configuration (selected on `tune`, retrieval only)

- **Implementation:** `bm25s` 0.3.11, `method="lucene"`. `PyStemmer` 3.1.0 Snowball English
  stemmer. Token pattern `\b\w\w+\b`, lowercased; one-character tokens are dropped. This is
  a known limitation: "U.S." yields no token.
- **Indexed text:** `title + ". " + paragraph text` (`Doc.index_text`). **Query:** the raw
  question, tokenized identically.
- **Ranking:** descending score, ties broken by `doc_id` in lexicographic order. Depth 100.
- **Candidates**, 12 in total, in this **preference order**:

  | Rank | Stemming | Stop words (`bm25s` English list, 33 words) | (k1, b) |
  |---|---|---|---|
  | 1 | Snowball | removed | (1.5, 0.75), the `bm25s` default |
  | 2 | Snowball | removed | (1.2, 0.75) |
  | 3 | Snowball | removed | (0.9, 0.4), the Anserini default |
  | 4–6 | Snowball | kept | same three, same order |
  | 7–9 | none | removed | same three, same order |
  | 10–12 | none | kept | same three, same order |

- **Selection score:** the mean of Joint Recall@3, @5 and @10 over the 500 `tune`
  questions, on exact variants. Averaging over the three K values the baseline may use
  keeps the choice consistent with how the retriever is used.
- **Rule:** the plateau is every candidate within **0.02** of the best selection score.
  Because the score averages three K values, 0.02 corresponds to 10 question-equivalents
  of 500, not 10 changed binary outcomes. It is a deterministic engineering tolerance,
  separate from statistical significance and from the 0.03 MPI. The selected
  configuration is the **first plateau member in the preference order**. No `dev` gate:
  BM25 is the starting point, not a challenger to an incumbent.
- **Also reported (exploratory):** for every candidate, the selection score. For the
  selected configuration, Recall@K and Joint Recall@K at K ∈ {2, 3, 5, 10, 20, 50} by type,
  exact variant and title-only. Index build time, index size, and single-query latency: CPU, 10 warm-up queries (the
  first 10 `eda` questions), then **all 500 `tune` questions** timed one query at a time;
  p50 and p95.

### 1.4 Answer generation (frozen here)

**Output schema** (`AnswerOutput`). The field order is the generation order:

```text
rationale:  string                                  # at most two short sentences
answer:     string                                  # shortest complete answer
citations:  list of {paragraph: int, sentence: int} # paragraph = the [P#] number shown
supported:  bool
```

- Citations are mapped back to `(doc_id, sentence_id)` through the rendered order. A
  paragraph number outside 1..N, where N is the number of paragraphs actually displayed
  (after any overflow dropping, and fewer than ten for some questions), maps to an
  invalid citation.

**Paragraph rendering.** Paragraphs appear in retrieval rank order. Sentences are stripped
of surrounding whitespace, and empty sentences keep their number so ids stay aligned:

```text
[P1] <title>
  [0] <sentence 0>
  [1] <sentence 1>
[P2] <title>
  ...
```

**Answer prompt** (`answer_v1`). Shared unchanged by the baseline, the improved RAG, the
matched-context control, the oracle, the single agent's answer node and the multi-agent
synthesizer:

```text
Answer the question using only the numbered paragraphs below.

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

Question: {question}
```

**Closed-book prompt** (`closed_book_v1`). Same schema; a reference row only:

```text
Answer the question from your own knowledge. No documents are provided.

Fill the JSON fields in this order:
- "rationale": at most two short sentences.
- "answer": the shortest answer that fully answers the question: a name, date, number,
  short phrase, or "yes"/"no". Not a full sentence.
- "citations": always an empty list.
- "supported": true if you are confident the answer is correct, otherwise false. When
  false, still give your best guess in "answer".

Question: {question}
```

- In closed-book, `supported` means self-reported confidence. It is not comparable with the
  evidence-based flag of the other systems and is reported separately.

**Oracle.** The answer prompt with exactly the question's two gold paragraphs (their own
text variants), in the order they appear in the question's supplied context.

**Context-overflow rule** (Phase 1: baseline and oracle contexts). If the rendered prompt
exceeds `num_ctx − num_predict` = 7,680 tokens:

- the **last displayed** paragraphs are dropped, one at a time, until it fits (for the
  baseline, the lowest-ranked; for the oracle, which has no retrieval rank, the last in
  display order);
- paragraphs are never cut mid-text;
- the number dropped is logged per question (`overflow_dropped`);
- final-context metrics use the context after dropping.

Later phases preregister their own overflow behaviour. For the agents that happens in
Phases 3–4, where paragraph priority is defined; `PLAN.md` currently specifies that the
single agent never prunes.

**Invalid outputs.** An output that fails schema validation is scored as answer `""`, no
citations and `supported = false`, and counted as a `schema_error`. There are no retries.

**Prompt-format check before any `dev` call.** 20 `eda` (train) questions — the first 16
bridge and first 4 comparison in the split's order — are run with their own supplied ten
paragraphs. The check verifies only:

- schema validity;
- that citations use the paragraph and sentence labels;
- that answers are short.

An output **fails the format check** if any of these holds:

- it fails schema validation;
- it has a **mis-labelled citation**: a paragraph number outside 1..N (N = the paragraphs
  actually displayed), or a sentence
  number that does not exist in that paragraph (an invalid citation under
  `metrics.citation_validity`);
- its answer is **not short**: more than 10 whitespace-separated words. Gold answers have
  a median of 2 words.

If more than 2 of the 20 outputs fail, the prompt wording may be revised **for format
only**, and the revision is logged as an amendment with its diff. Answer accuracy on these
questions is recorded but is not a revision criterion.

After this check, and before the first `dev` call, the SHA-256 hashes of the following are
recorded in §A, after which they are frozen:

- the answer prompt template (`answer_v1`);
- the closed-book prompt template (`closed_book_v1`);
- the output JSON schema;
- the system prompt (`llm.DEFAULT_SYSTEM`);
- the paragraph renderer (`render_v1`: the hash of its source).

**`dev_smoke`** runs the end-to-end pipeline once, for crashes, timing and logging only.
It uses the selected BM25 configuration, **K = 5**, `answer_v1` and the fixed 20
`dev_smoke` questions. No prompt or configuration change may be justified by its
accuracy. Because the inputs are identical, its 20 cached responses are reused by the
later K = 5 `dev` evaluation.

### 1.5 Choosing K (on `dev`, with the LLM)

- **Candidates:** K ∈ {3, 5, 10}, with the BM25 configuration selected in §1.3.
- **Selection metric:** answer F1 over all 100 `dev` questions.
- **Rule:** the plateau is every K within **0.02** F1 of the best. Preference order
  **5, 3, 10** (the planned default, then the cheaper option). The selected K is the first
  plateau member in that order.
- **Limitation:** at n = 100, the interval on a paired F1 difference is roughly ±0.07, so
  this is a coarse, cost-aware choice. The three F1 values are reported with intervals and
  with their prompt-token cost.

### 1.6 What is reported for the baseline (`dev`, n = 100; by type and overall)

- **Answer:** EM, F1, F1 on answered, abstention rate, schema errors.
- **Evidence:** strict and official-compatible supporting-fact P/R/F1, joint F1.
- **Grounding:** citation coverage, per-citation validity, all-valid rate, and the
  incomplete-gold-evidence proxy.
- **Retrieval:** final-context Recall and Joint Recall (exact variant; title-only
  secondary).
- **Cost:** LLM calls, prompt and output tokens, LLM wall-seconds; retrieval and LLM
  latency p50/p95 kept separate; `overflow_dropped` counts.
- **Reference rows:** the same metrics for the oracle. For closed-book: EM and F1, plus the
  **low-confidence rate** (share with `supported = false`) and **F1 when confident**. These
  replace the abstention-derived results and are excluded from every evidence-support
  comparison; closed-book has no evidence metrics.

Question-level quality metrics are reported with 95% bootstrap intervals. Raw counts,
sizes and latency percentiles are reported as they are. On `tune` (n = 500), the selected
BM25's retrieval metrics. For the `rationale` field, its **token count** is measured exactly
(by tokenizing the rationale text) and reported next to the **total response latency**. Its
separate latency cannot be observed from one non-streamed generation, so it is not
reported. The rationale is a concise evidence explanation, not a faithful account of the
model's reasoning.

### 1.7 Failure taxonomy (`dev`)

**Automatic, all 100 questions.** The evidence bucket at the selected K is crossed with the
answer outcome.

- Evidence buckets:
  - both gold paragraphs in context;
  - only hop 1 (bridge);
  - only hop 2 (bridge);
  - `only_one_hop_undetermined` (bridge, one gold paragraph present, hop order
    undetermined by the Phase 0 rule; it is never assigned a hop from paragraph order);
  - only one entity (comparison);
  - neither.
  Hop order uses the Phase 0 rule.
- Answer outcomes: correct (EM = 1); partial (EM = 0, F1 ≥ 0.5); wrong (F1 < 0.5).
- For correct answers, whether the citations cover both gold paragraphs.

**Manual sample, reproducible:**

- Population: the `dev` questions with EM = 0 for the baseline at the selected K.
- Their question ids are sorted lexicographically.
- `numpy.random.default_rng(20260929).choice(len(ids), size=min(25, len(ids)),
  replace=False)` picks positions in that sorted list (NumPy 2.5.0). The sample is
  **unstratified**: its type mix follows the failures.
- Cases are labelled in sorted-id order.

Each case gets one primary label, chosen before viewing any other system's output:

| Label | Meaning |
|---|---|
| `retrieval_hop1` / `retrieval_hop2` / `retrieval_both` / `retrieval_entity` / `retrieval_undetermined` | A needed gold paragraph is absent from the context. `retrieval_undetermined`: one bridge gold paragraph is missing and the hop order is undetermined |
| `reasoning` | Both gold paragraphs are present; the answer is wrong or incomplete |
| `granularity` | The answer is right in substance but differs in form or specificity (e.g. "Walter Coy" vs "Walter Darwin Coy") |
| `label_error` | The gold answer or gold evidence is wrong or doubtful (as in the Phase 0 *Promos* example) |
| `other` | None of the above; described |

**Label precedence.** The first rule that applies decides the label:

1. `label_error`: the benchmark target itself (gold answer or gold evidence) is doubtful.
2. `granularity`: the prediction is substantively correct.
3. A retrieval label: required gold evidence is absent from the context.
4. `reasoning`: both gold paragraphs are present.
5. `other`.

- Claude assigns the labels, each justified by a quoted sentence or by the missing
  paragraph. The full labelled table goes into the Phase 1 report, and every case is
  reviewed.
- These labels are exploratory. They are neither blinded nor independent annotation, and
  they are not the Phase 6 audit.

### 1.8 Not in Phase 1

- Dense retrieval, fusion, reranking, query rewriting.
- Any prompt change based on `dev` accuracy.
- Any use of blind splits.

### 1.9 Implementation for this phase

- `retrieval.py`: BM25 and deterministic top-k, adapted from FiQA with tests.
- `prompts.py`: the two templates and the schema, with version names and hashes.
- `rag.py`: baseline, closed-book and oracle pipelines, with the overflow rule.
- `run_eval.py`: `dev`, `dev_smoke` and `eda` runs, writing per-question records and a
  manifest.
- `01_baseline.ipynb`: reads the results, plots, taxonomy.
- `manifest.EVAL_CODE` is extended to `run_eval.py` and every module that turns outputs
  into scores, with a test that fails when a scoring module is missing from that list.
- `reports/phase_1.md`: the completion report in the agreed template, with the full
  manual-label table.
- `reports/slides_draft.md`: the slide skeleton, started here, with draft slides for the
  problem, dataset and evidence chains, evaluation protocol, and baseline with its failure
  taxonomy.
- Expected compute: 12 BM25 indexes (seconds each).
  - 540 pipeline evaluations: 20 `eda` format checks, 20 `dev_smoke`, 300 for choosing K,
    100 closed-book, 100 oracle.
  - About **520 unique LLM generations**, because the 20 `dev_smoke` responses are reused
    at K = 5. About 1.5–2 h.
  - Cost and latency always use the original-generation metadata stored in the cache
    (tokens, wall-seconds), never cache-read time.

## Phase 2 — the strongest non-agent retriever

Recorded 2026-09-29, before any Phase 2 code exists and before any Phase 2 retrieval or
LLM result has been produced. Two stages:

- **Phase 2a:** classic deterministic retrieval.
- **Phase 2b:** dense and classic–dense hybrid retrieval.

**Common rules.**

- All selection uses `tune`, retrieval only. No LLM call happens until every candidate
  has been evaluated and the overall retriever is frozen.
- The frozen winner then runs once end to end on `dev`, with K = 10 and the unchanged
  answer interface (`answer_v2`, schema v2, `render_v1`, the system prompt).
- **Candidates see only the question, corpus titles and corpus text.** They never see
  answers, supporting facts, gold documents or a question's own supplied context.
- **Gold labels are used only for scoring and diagnostics.** This includes the hop order,
  and whether the first hop is named in the question.
- **Exploratory status.** The `dev` analysis of Phase 1 BM25 rankings that motivated these
  candidates is exploratory, and so is everything on `tune` and `dev` in this phase. The
  blind test sets stay untouched.
- **Fixed conditions.**
  - The final context is exactly K = 10 unique paragraphs, exact document identity is
    unchanged, and ties are broken by `doc_id` in lexicographic order everywhere.
  - Every candidate returns a **full ranking of depth 100**. Where a method produces only
    its top 10, positions 11–100 are the base ranking's documents in order, skipping any
    already placed.
  - Recall@K for K ∈ {2, 3, 5, 10, 20, 50} is always read from that ranking. For graph
    expansion and reranking, only the top 10 positions are modified by the method; their
    Recall@20 and @50 come from the base tail, so they are not the method's own deep
    rankings and are reported with that note.

### 2.1 Motivation (Phase 1, `dev`, exploratory)

- **Bridge recall:** hop 1 reaches 0.84 at K = 5 and 0.90 at K = 10; hop 2 reaches 0.48
  and 0.61.
- **Named vs described first hops:** a first hop whose title the question names reaches
  the top 10 in 98% of cases (n = 51); one it only describes, in 75% (n = 28).
- **Missing first hops:** BM25 sometimes retrieves the second gold paragraph but not the
  one that names it. Graph traversal is therefore tested in both directions.

### 2.2 Shared text normalization `norm_v1`

Used by C1 and C2 (on both corpus text and queries) and by every alias and mention match
in the graph. It is applied before the `bm25s` tokenizer, and so before lowercasing,
stop-word removal and stemming. Displayed text and citations always use the original
text. Steps, in order:

1. Unicode NFKD, then drop every combining mark (category `Mn`).
2. Map characters that do not decompose: `ø→o`, `Ø→O`, `ß→ss`, `æ→ae`, `Æ→AE`, `œ→oe`,
   `Œ→OE`, `đ→d`, `Đ→D`, `ł→l`, `Ł→L`, `ı→i`, `þ→th`, `Þ→Th`, `ð→d`, `Ð→D`.
3. Apostrophes: map `’ ‘ ʼ ´` and backtick to `'`. Then delete a possessive `'s` at a word
   end, and delete every remaining `'` (so "O'Brien" becomes "OBrien").
4. Dashes: map `– — ‐ ‑ ‒ −` to `-`. The tokenizer then splits on `-`, as it already does
   for an ASCII hyphen.
5. Dotted acronyms: a run of two or more "letter + dot" units becomes the letters
   joined, so "U.S." → "US" and "U.S.A." → "USA". Regex:
   `\b(?:[A-Za-z]\.){2,}` → the letters only.
6. Collapse whitespace.

### 2.3 Phase 2a candidates (classic)

Selection runs in two pre-declared steps: first the lexical base, then the expansions
built on it.

**Step 1: the lexical base.** Three configurations, in preference order:

| Rank | Id | Definition |
|---|---|---|
| 1 | **C0** | The frozen Phase 1 BM25 (stemming, stop words removed, k1 = 1.5, b = 0.75, index text `title. text`) |
| 2 | **C1** | C0 with `norm_v1` applied to the corpus text and the query |
| 3 | **C2** | C1 with **title repetition**: the index text is `title. title. text`, one extra copy of the title. This is repetition, not BM25F |

The selected base is called **B**.

**Step 2: expansions on B.** Six new configurations. B stays in the comparison as the
incumbent without being re-run.

**C3 — BM25-weighted RM3-style feedback on B.** It follows RM3's structure, but
normalized BM25 scores replace RM3's query-likelihood document weights. Two
configurations: C3a (n = 3, m = 10) and C3b (n = 5, m = 20), both with λ = 0.5.

- **Feedback set.** F = B's top n documents for the question, with first-round scores
  s(d).
- **Document weights.** π(d) = s(d) / Σ_{d′∈F} s(d′).
- **Candidate terms.** Every token of B's tokenizer (same normalization, stop words and
  stemming) occurring in a document of F. No further exclusions.
- **Relevance model.** P_R(t) = Σ_{d∈F} π(d) · tf(t, d) / |d|. Both tf(t, d) and |d| are
  taken from d's **actual indexed representation in B**: its index text, after B's
  normalization, with the repeated title when B = C2. The m terms with the largest P_R
  are kept (ties by term string), and P_R is renormalized over them.
- **Query model.** P_Q(t) = c(t, Q) / |Q| over the question's tokens under B's tokenizer
  that are **in B's vocabulary**; |Q| counts only those tokens.
- **Fallbacks.** If the question has no in-vocabulary term, or F is empty, or
  Σ_{d∈F} s(d) = 0, C3 returns B's ranking unchanged, and the case is counted.
- **Final weights.** w(t) = λ · P_Q(t) + (1 − λ) · P_R(t), over the union of question
  terms and the m expansion terms.
- **Second round.** score₂(d) = Σ_t w(t) · BM25_B(t, d), where BM25_B(t, d) is t's
  contribution in B's index. The second-round ranking **replaces** the first; there is no
  merge. Each document appears once, and ties are broken by `doc_id`.
- **Drift diagnostics.** Gold paragraphs in B's top 10 but not in C3's top 10 (lost), and
  the reverse (gained), with counts and question ids.

**The title-mention graph** is used by C4 and C5, built once from corpus text only. Nodes
are the 66,635 documents, identified by `(title, text_hash)`.

- **Mention tokenization `tok_v1`.** For any string x: case-fold `norm_v1(x)`, then take
  its `\w+` tokens. Alias strings and paragraph sentences both go through this, so
  "Spider-Man" becomes the token tuple (spider, man) on both sides.
- **Aliases.** Each title T has a full alias, `tok_v1(T)`. If T ends with a parenthetical
  disambiguator, `" (…)"`, it also has a short alias: `tok_v1` of T without it. **An
  alias is a token tuple.**
- **Alias filters** are applied to the token tuple:
  - the tuple joined with single spaces must be at least 4 characters (measured **after**
    tokenization);
  - at least one token must not be in the 33-word stop list;
  - a tuple made only of digit tokens is dropped;
  - a **single-token** alias is dropped if that token occurs in more than 100 corpus
    paragraphs (document frequency of the `tok_v1` token over paragraph sentences). This
    removes titles that are ordinary words, such as "Home" or "Up".
- **Ambiguity.** An alias tuple that maps to more than one distinct title is dropped, and
  the number dropped is reported. After this step every remaining tuple maps to exactly
  one title. A title with two text variants (54 cases) is one title: an edge to it points
  to **both** variants, which are ordered by `doc_id`; no gold variant is chosen.
- **Matching.** Each sentence of a paragraph (not its title) is tokenized with `tok_v1`,
  and matches never cross sentence boundaries. The scan runs left to right: at token i,
  the **longest** alias (the greatest token count) matching tokens i, i+1, … is taken, and
  the scan continues after it. With no match, the scan moves to i + 1. Matches never
  overlap. Two different aliases cannot match the same start position with the same
  length, because each surviving tuple is unique. The mention position is (sentence
  index, token offset).
- **Edges.**
  - A → B if A's text matches an alias of B's title and B's title differs from A's title.
    This also removes edges between two variants of the same title.
  - Each edge records its first mention position (sentence index, token offset).
  - Outgoing edges of A are ordered by mention position, then by `doc_id`.
  - Incoming edges of B: every A with A → B.
- **Reported:** alias counts, and aliases dropped by length, digit, frequency and
  ambiguity; edge count; in- and out-degree distributions; build time and memory.

**Graph expansion `G_{dir,s}(X)`** is an operator on a base ranking X. In Phase 2a,
X = B. It has a direction (forward, or forward and backward), a seed depth s ∈ {5, 10},
and **5 reserved slots**:

- **Seeds:** X's top s documents. **Retained:** X's top 5.
- **The candidate list** is built by walking the seeds in rank order. For each seed:
  - **Forward** (C4 and C5): the first **3** outgoing targets, by mention position then
    `doc_id`, that are neither retained documents nor already candidates. Each text
    variant counts as one target.
  - **Backward** (C5 only): documents with an edge to the seed that are neither retained
    nor already candidates, and have a first-round **B** score above 0. The top **2** are
    taken by B score (descending), then `doc_id`. This caps popular (high in-degree)
    titles: a hub adds at most 2 documents, and only ones that also match the question
    lexically.
- **Seeds ranked 6–10** are not excluded. They can take an expansion slot, but only when
  **another seed reaches them through an edge**. Being a seed alone earns no boost.
- **The final top 10** = X's top 5, then the first candidates from the list, up to 5, then
  X's next documents in order, skipping any already placed, until 10 unique documents
  are reached. Positions 11–100 come from X, skipping placed documents.
- **The four graph configurations:** C4@5, C4@10, C5@5, C5@10, all with X = B.

**Configuration count.** Phase 2a has six candidate families (C0–C5) and **9 evaluated
configurations**: C0, C1, C2, C3a, C3b, C4@5, C4@10, C5@5 and C5@10.

### 2.4 Phase 2a selection

- **Primary metric:** overall exact-variant Joint Recall@10 on `tune` (n = 500).
- **Step 1:** the plateau is every candidate within 0.02 of the best; B is the first
  plateau member in the order C0, C1, C2.
- **Step 2:** among {B, C4@5, C4@10, C5@5, C5@10, C3a, C3b}, in that preference order (the
  graph uses no weights; RM3 adds three parameters), the classic champion **C** is the
  first member of the 0.02 plateau.
- **The 0.03 MPI** says whether C's gain over C0 is practically meaningful. It does not
  change the selection rule.

### 2.5 Phase 2a hypotheses (exploratory; `tune`; bridge hop order by the Phase 0 rule, undetermined excluded and counted)

| Id | Prediction | Estimate and threshold |
|---|---|---|
| H2a-1 | Forward expansion raises bridge hop-2 Recall@10 over B | Best C4 minus B, paired, at least **+0.05** |
| H2a-2 | Backward expansion recovers first hops when B retrieved the second gold paragraph but not the first | Among bridge questions with hop 2 but not hop 1 in B's top 10, the share with hop 1 in the top 10 of C5 (same seed depth as the best C4); predicted at least **0.30**; the count is reported |
| H2a-3 | Bidirectional beats forward-only on bridge Joint Recall@10 | C5@s minus C4@s for each s, paired; predicted **> 0** for both |
| H2a-4 | Backward expansion is noisier for high-degree seed titles | **Micro-averaged** over backward-added paragraphs in the final top 10: the share that are gold, split by the in-degree of the seed that contributed them (≥ 100 vs < 100), with both counts; predicted **lower** for ≥ 100 |
| H2a-5 | Some classic configuration improves overall Joint Recall@10 over C0 by the MPI | The best configuration minus C0, paired, at least **+0.03** |
| H2a-6 | Described first hops stay weaker than named ones under C | First-hop Recall@10 named minus unnamed, for C; predicted at least **+0.10** |
| H2a-7 | Normalization alone changes little | C1 minus C0, overall Joint Recall@10, absolute value **< 0.01** |

### 2.6 Phase 2a diagnostics (overall and by type; exact and title-only)

- **Recall:** Recall@K and Joint Recall@K; bridge hop-1 and hop-2 Recall@10; first-hop
  Recall@10, named vs unnamed; comparison Joint Recall@10.
- **Graph coverage:** the share of bridge questions (determined hop order) where the
  **forward** edge hop 1 → hop 2 exists, and where the **backward** edge hop 2 → hop 1
  exists.
- **Expansion effects:** gold paragraphs added by expansion; gold paragraphs displaced
  from B's top 10; expansion candidates per query; skipped ambiguous and high-frequency
  aliases.
- **RM3 drift:** gold paragraphs lost and gained against B, with question ids.
- **Cost:** build time, index and graph size, and p50/p95 query latency (CPU, 10 warm-up
  `eda` queries, then all 500 `tune` questions one at a time).

C is frozen before Phase 2b begins.

### 2.7 Phase 2b candidates (dense and hybrid)

**Models.** Exactly two; results describe TAS-B, not dense retrieval in general. Another
dense model may be added only as a non-selective, retrieval-only sensitivity analysis
after all required work is done.

- **Bi-encoder:** `sentence-transformers/msmarco-distilbert-base-tas-b`, revision
  `b12d9352e776979147078a8975a4885042984fd1`.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

**Dense protocol:**

- **Inputs.** Paragraph text is `title. text` (the original text, without `norm_v1`);
  query text is the original question.
- **Encoding.**
  - CLS pooling, **no normalization** of the vectors.
  - Similarity is the **dot product**, as the model card specifies.
  - Maximum length 512 tokens, 768-dimensional vectors.
  - The corpus is encoded **once**, float32, on the GPU with Ollama unloaded, in batches of
    64 via `sentence-transformers` `encode`.
  - Queries are encoded in float32 on the CPU.
- **Search.** Exact search: dot products with all 66,635 vectors, then
  `deterministic_top_k` with the `doc_id` tie-break. There is no FAISS index; the corpus
  is small enough for exact NumPy search, and approximate search would add
  nondeterminism.
- **Fingerprints.** The embeddings are cached under a key that covers the model revision,
  corpus fingerprint, text template, maximum length, dtype and device. The SHA-256 of the
  float32 matrix is recorded in every manifest.
- **Reported:** encoding time, matrix size, peak GPU memory, and query latency p50/p95.

**Candidates, in preference order:**

| Rank | Id | Definition |
|---|---|---|
| 1 | **D0** | C, the classic champion (incumbent) |
| 2 | **D1** | TAS-B alone, depth 100 |
| 3 | **D2** | RRF(C, D1): score(d) = Σ over the two rankings of 1 / (60 + rank(d)), over each ranking's top 100; top 10 by fused score. Scores are never added directly |
| 4 | **D3** | **G\*(H)**, where H = RRF(B, D1) (same formula) and **G\*** is the frozen graph operator: the best-scoring C4/C5 configuration from Phase 2a (preference order on ties), with its direction, caps and seed depth unchanged, even if it is not the Phase 2a champion. H replaces B everywhere in the operator: seeds are H's top s, retained documents are H's top 5, and the fill and the depth-100 tail come from H. The one exception is the backward-candidate score, which stays **B's lexical first-round score** |
| 5 | **D4** | Cross-encoder reranking of D2's **top 50** |

For D4:

- **Input.** `(question, title + ". " + text)`, truncated to 512 tokens, batch 32.
- **Order.** Documents are ranked by logit, with ties broken by `doc_id`. Positions 11–100
  follow D2's order.
- **Comparison.** D4 is compared with D2 because a question-only cross-encoder may
  demote hidden second hops.

**Configuration count:** 4 new configurations (D1–D4).

### 2.8 Phase 2b hypotheses (exploratory, `tune`)

| Id | Prediction | Estimate and threshold |
|---|---|---|
| H2b-1 | TAS-B helps described first hops | D1 minus C, first-hop Recall@10 on unnamed-first-hop bridge questions, at least **+0.05** |
| H2b-2 | Classic and dense make different errors | D2 overall Joint Recall@10 above both C and D1 |
| H2b-3 | Dense seeds help graph expansion | G\*(H) minus G\*(B), bridge Joint Recall@10, **> 0** |
| H2b-4 | Reranking helps direct relevance but can demote the second hop | D4 minus D2: overall Recall@3 **> 0**, while bridge hop-2 Recall@10 **< 0** |
| H2b-5 | Some dense or hybrid candidate reaches the MPI over C | The best of D1–D4 minus C, overall Joint Recall@10, at least **+0.03** |

**Selection.** Same metric and rule: the 0.02 plateau with preference order D0, D1, D2, D3,
D4. A more complex hybrid has to earn its place outside the plateau. The result is the
**overall retriever R**.

**Overlap reported** (gold paragraphs in the top 10):

- found only by C, only by D1, by both, or by neither;
- seeds contributed by dense retrieval (in H's top s but not B's);
- second hops then recovered by graph expansion (D3).

### 2.9 End-to-end evaluation (after 2a and 2b)

1. R and every relevant hash are frozen: code, graph, embeddings and configuration.
2. R runs **once** on `dev` with K = 10 and the unchanged answer interface: about 100 new
   generations, about 35 minutes.
3. The Phase 1 BM25 K = 10 run is replayed under the Phase 2 code, so that both
   manifests share the evaluation-code hash. The replay:
   - runs in **offline, cache-only mode** (`llm.OllamaClient(offline=True)`), which aborts
     on any cache miss, and so on any prompt key that differs. The model digest in the
     cache key is taken from the Phase 1 manifest (`base_llm.digest`); an offline client
     cannot query Ollama for it;
   - asserts that all 100 generations are cache hits;
   - asserts that C0 reproduces all 100 Phase 1 `dev` top-10 document lists exactly;
   - keeps the original-generation token counts and wall times.
4. R is compared with it, paired over the same 100 questions:
   - answer F1 and EM;
   - final-context Joint Recall;
   - strict and official-compatible evidence;
   - citations and `supported` behaviour;
   - calls, tokens, retrieval latency and LLM wall time;
   - bootstrap intervals throughout (exploratory).
5. The prompt does not change because of Phase 2 results.
6. If nothing reaches a practically meaningful improvement, that is reported as the
   result. If R = C0, the retriever stays BM25 and no new generation is needed. The
   cache-only replay still runs, to produce a manifest with the current evaluation-code
   hash.

### 2.10 Tests required before any experiment

- **Normalization:** symmetry between corpus and query.
- **Title repetition** in the index text.
- **BM25-weighted RM3-style feedback:** the formula on a hand-computed example;
  deterministic term selection; the fallbacks.
- **Graph aliases:** exact and short aliases; rejection of short, digit-only,
  high-frequency and ambiguous aliases; duplicate-title variants.
- **Graph edges:** forward, backward and self-edges; backward caps on a hub; candidate
  ordering; exactly 10 unique final documents; no access to gold labels (candidate
  functions receive only the question, corpus and indexes).
- **Graph operator regressions:** a seed ranked 6–10 stays eligible when another seed
  links to it; an unlinked seed ranked 6–10 gets no graph boost; D3 retains and fills
  from H, not B; D3 still ranks backward candidates by B's scores.
- **Baseline replay:** aborts on a single missing cache entry; reproduces all 100 Phase 1
  `dev` top-10 lists.
- **RRF:** deterministic, on a hand-computed example.
- **Dense:** vector dimension (768); dot product without normalization; tie-breaking.
- **Cross-encoder:** ordering.
- **Blind splits** are refused.
- **Manifests** carry every classic, graph and model setting and the graph, embedding and
  index fingerprints.

### 2.11 Deliverables

- **Code and tests:** classic, graph, dense, fusion and reranking code, with tests.
- **Analysis:** a read-only notebook `02_retrieval.ipynb`.
- **Results:** retrieval-only `tune` results; one `dev` run of R, plus the Phase 1
  baseline replayed.
- **Reports:** `reports/phase_2.md`, and updated slides.
- **Cases:** representative successes and failures.
- **Attribution:** a statement of which mechanism helped (normalization, title weighting,
  feedback, forward links, backward links, semantic retrieval, fusion, reranking) and
  which did not.

## Phase 3 — single investigation agent

Recorded 2026-09-30, before any Phase 3 code exists and before any agent output has been
produced. The research question for this phase:

> Does evidence-dependent iterative search add answer quality beyond the frozen Phase 2
> retriever R = D3, and at what cost?

### 3.1 Fixed conditions

- **Search backend.** The frozen R = D3 (`hybrid.D3Retriever`, commit `0af5786`) is the
  only search backend.
  - X1 is **not** used by any agent variant, single or multi-agent, because it failed its
    carry gate.
  - No retriever setting is changed in Phase 3.
- **LLM and answer interface.** The same `qwen2.5:7b`, options, system prompt and
  `render_v1`. The **answer** step uses the frozen `answer_v2` and schema v2 unchanged.
- **Splits.**
  - Development uses `dev_smoke` (smoke runs) and `dev` (one full run).
  - The format check uses `eda` (train) own contexts.
  - Blind splits stay closed until Phase 5.
- **Framework.** LangGraph `StateGraph` (`langgraph` 1.2.12). The LLM only returns JSON;
  every tool and every transition is deterministic code. There is no free-form tool
  calling.

### 3.2 The agent

**State:** `question`, `queries`, `seen` (ordered doc ids), `per_search` (ids added by each
search), `assessments`, `step`, `stop_reason`, `llm` (every call record), `answer`.

**Nodes and edges:**

1. **`search`** (deterministic) runs D3 on the current query and takes the
   highest-ranked documents **not already seen**:
   - search 1 uses the **original question** verbatim and takes **10**;
   - every later search takes **5**.

   Search 1 therefore gives exactly R's K = 10 RAG context. The agent starts from R's
   evidence, and anything it adds comes from evidence-dependent follow-up.
2. **`assess`** (LLM, `assess_v1`, §3.3) runs after searches 1 and 2 only. It sees the
   question and every paragraph seen so far, and returns `sufficient`, `missing_info`,
   `next_query` and a short `justification`.
3. **`answer`** (LLM, `answer_v2`) sees every paragraph seen so far, in display order:
   search 1's ten in D3 order, then search 2's five, then search 3's five.

**Transitions:**

- After `search`:
  - if the search added no unseen document → `answer` (stop `no_progress`);
  - if it was search 3 → `answer` (stop `max_searches`);
  - otherwise → `assess`.
- After `assess`, the first matching rule applies:
  - the output fails the schema or the runtime contract below → `answer` (stop
    `assess_error`). An invalid output **never** reaches `search`;
  - `sufficient = true` → `answer` (stop `sufficient`);
  - `tok_v1(next_query)` is empty → `answer` (stop `no_query`);
  - `tok_v1(next_query)` equals `tok_v1` of the question or of any earlier query →
    `answer` (stop `duplicate_query`);
  - otherwise → `search` with `next_query`.

**Runtime contract of `AssessOutput`,** enforced by validation after decoding. Ollama's
JSON schema can express only the field types.

- If `sufficient = true`, then `missing_info` and `next_query` must both be empty after
  stripping whitespace.
- If `sufficient = false`, then `missing_info` and `next_query` must both be non-empty
  after stripping.
- `next_query` has at most **12** `tok_v1` tokens.

A violation is a validation error, and so `assess_error`. **All** query comparisons and
lengths (duplicates, the 12-token limit, emptiness) use `tok_v1`, the same normalization
as the graph.

**Hard budget:**

- at most **3 searches** and **20 paragraphs** (10 + 5 + 5);
- at most **3 LLM calls** (2 `assess` + 1 `answer`). No `assess` runs after the third
  search, since no further search is allowed.

**Overflow rule (Phase 3, evidence-preserving).** Every `assess` and `answer` prompt must fit
`num_ctx − num_predict` = 7,680 tokens. If a prompt does not fit, paragraphs are dropped
one at a time, in this deterministic priority, until it fits:

1. the lowest-ranked remaining paragraph from **search 1** (its ten, from rank 10 upward);
2. then the lowest-ranked remaining paragraph from **earlier follow-up batches** (search 2
   when search 3 exists);
3. only then paragraphs from the **newest** follow-up batch, which is protected until
   everything else is gone.

- **Order and logging.** Retained paragraphs keep their original display order. Every
  `assess` and `answer` call logs the number of dropped paragraphs and their exact ids.
- **Recall.** Final-context Joint Recall uses the ids the `answer` node actually
  displayed. Cumulative Joint Recall (everything searched) is reported beside it; the two
  differ only when this rule fires.
- Paragraphs average about 150 tokens (p95 288), so 20 paragraphs fit in almost every
  case.

**Trace.** Queries, the ids added by each search, each `assess` output, transitions, the
stop reason and every LLM call record (key, tokens, wall time). No hidden chain of thought
is stored; the justification is a two-sentence field of the output schema.

**What makes it agentic:** `next_query` is written from the evidence returned so far,
typically naming the bridge entity found in a retrieved paragraph.

### 3.3 The assess prompt `assess_v1` (frozen here)

Output schema `AssessOutput`, with the field order as the generation order:

```text
justification: string   # at most two short sentences
sufficient:    bool
missing_info:  string
next_query:    string
```

Prompt:

```text
Decide whether the numbered paragraphs below contain every fact needed to answer the
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

Question: {question}
```

**Format check before any `dev` call.**

- **Inputs:** the same 20 `eda` questions as the answer-prompt check, each in two
  conditions, 40 calls in all:
  - (a) its own ten supplied paragraphs;
  - (b) the same paragraphs with one gold paragraph removed: the Phase 0 hop 2 for bridge
    questions (hop 1 when undetermined), and the lexicographically second gold title for
    comparison questions.
- **Failure criteria:** an output fails if it fails the schema or the runtime contract,
  or if `next_query` equals the question after `tok_v1`.
- **Revision rule:** if more than 4 of 40 fail, one format-only revision is allowed, logged
  as an amendment with its diff and rerun once. If more than 4 still fail, work stops for
  review.
- **Viability gate:** at least one valid `sufficient = true` and at least one valid
  `sufficient = false` output among the 40. Otherwise work stops for review. This checks
  that both branches are reachable; it does not tune accuracy.
- **Reported separately, not a criterion:** the `sufficient` decisions in condition (a)
  and in condition (b).
- **Freeze:** after the check, the hashes of `assess_v1`, `AssessOutput`, the agent graph
  source and the transition rules are recorded in §A, before the first `dev` call.

### 3.4 Runs and controls

1. **Format check** (40 calls).
2. **`dev_smoke` smoke run** of the agent (20 questions): crashes, timing and trace
   logging only. Nothing may be changed because of its accuracy.
3. **One full `dev` run** of the agent (100 questions). The 20 smoke questions are reused
   from the cache.
4. **Matched-context control.**
   - **Definition.** K_match = the mean, over the 100 `dev` questions, of the number of
     paragraphs **actually displayed to the agent's `answer` call** (after overflow). It
     is rounded half up (floor(x + 0.5)) and clipped to [10, 20].
   - **Freezing.** K_match is derived **once** from `dev`, recorded in §A, and reused
     unchanged on `test_agents`.
   - **The run.** **D3 RAG at K = K_match** runs once on `dev` with the unchanged answer
     interface and its own (Phase 1) overflow rule, and its displayed-context count is
     reported.
   - **Above K = 10,** D3's positions 11–20 come from its frozen base tail (H in order,
     skipping placed documents): graph expansion only builds D3's top ten.
   - **What it controls.** This is an approximate, **global** control of context size, not
     a per-question match. It helps distinguish iteration from additional context; it
     does not perfectly separate them.
5. **Cache-only replays of the controls under the Phase 3 evaluation code.** Adding the
   agent to `run_eval.py` changes the evaluation-code hash, so the existing D3 K = 10 and
   BM25 K = 10 runs are replayed from the cache with that same code:
   - **D3 K = 10:** reproduces all 100 rankings and cached answers exactly;
   - **BM25 K = 10:** reproduces its rankings and cached answers exactly.

   Both run in offline, cache-only mode, and abort on any cache miss or ranking mismatch.
   They generate nothing new.
6. **Comparisons, all paired over the 100 `dev` questions:** the agent against R (the D3
   replay), against the matched-context control, and against the BM25 replay.
   - Every compared manifest has the same Phase 3 evaluation-code hash.
   - Each comparison asserts equal answer prompt, schema, validator, renderer, LLM
     configuration, corpus and split invariants.
   - All comparisons are exploratory. The confirmatory test is the preregistered Holm
     family on `test_agents` in Phase 5.

**Expected LLM cost:** 40 (format check) + about 250 (agent: up to 3 calls per question,
about 2.5 on average) + 100 (matched control), roughly 390 generations and about 2.5–3
hours.

### 3.5 Phase 3 hypotheses (exploratory, `dev`, n = 100; paired bootstrap)

| Id | Prediction | Estimate and threshold |
|---|---|---|
| H3-1 | Iteration adds evidence: agent final-context Joint Recall − R's, overall | at least **+0.03** |
| H3-2 | The added evidence is on bridge questions | (Δbridge − Δcomparison) **> 0**, where Δ is the H3-1 difference (agent − R, final-context Joint Recall) within each type; unpaired bootstrap within types |
| H3-3 | Iteration improves answers over R | agent F1 − R F1 **> 0** (point estimate; the 0.03 MPI is not expected to be demonstrable at n = 100) |
| H3-4 | Iteration adds value beyond additional context (approximately; see 3.4) | agent F1 − matched-context F1 **> 0** |
| H3-5 | The agent usually stops after one search | `sufficient` at the first `assess` for at least **50%** of questions |
| H3-6 | Over-searching is limited | among questions whose search-1 context already has both gold paragraphs, the share that search again is at most **40%** |
| H3-7 | Early stopping happens | among questions stopped as `sufficient`, the share missing a gold paragraph is reported; predicted at most **25%** |
| H3-8 | Iteration costs time | median over questions of the agent's **summed** LLM wall time (every call) at least **1.8×** the median of R's stored answer-call wall time |

### 3.6 Diagnostics

- **Stop reasons:** the distribution, with counts.
- **Effort:** searches per question; paragraphs in context; LLM calls, prompt and output
  tokens, and wall time per question.
- **Recall:** final-context and cumulative Joint Recall; hop-1 and hop-2 Recall, bridge
  only.
- **Iteration outcomes:**
  - follow-ups that add a gold paragraph;
  - duplicate queries and `no_progress` loops;
  - `assess` calibration: `sufficient` against whether both gold paragraphs are present.
- **Queries:** representative traces. Queries are classified by whether they name an
  entity that appears in the retrieved paragraphs but not in the question: the operational
  sign of an evidence-dependent query.
- **By type:** everything split by bridge and comparison.

### 3.7 Tests required before any agent run

- **Search rule:** search 1 returns D3's top 10 unseen (and equals R's context); later
  searches return 5 unseen; seen documents are never repeated.
- **Budget:** each transition and stop reason on a stub LLM; at most 3 searches, 3 LLM
  calls and 20 paragraphs; no `assess` after search 3.
- **Duplicate queries:** detected after `tok_v1`, against the question and every earlier
  query.
- **The `AssessOutput` contract:**
  - `sufficient = true` with a non-empty field is an error;
  - `sufficient = false` with an empty field is an error;
  - a `next_query` of 13 tokens is an error, and one of 12 is accepted;
  - a punctuation-only `next_query` gives `no_query`;
  - a contract violation ends as `assess_error` and never reaches `search`;
  - duplicate detection and length use `tok_v1`.
- **Overflow:** the evidence-preserving priority (search 1 lowest ranks first, then earlier
  follow-ups, the newest batch protected), display order kept, dropped ids logged for
  every call, and final-context recall computed from the answer node's retained ids.
- **Replays:** the D3 and BM25 cache-only replays abort on one missing cache entry and on
  one ranking mismatch.
- **Interfaces:** the answer step uses the frozen `answer_v2` and schema v2 hashes; the
  `assess_v1` text equals this preregistration.
- **Trace:** complete, and deterministic on a stub LLM.
- **Splits:** blind splits are refused.
- **Manifests:** they carry the agent graph source hash, the prompt and schema hashes, the
  budget and the D3 fingerprints.

### 3.8 Departures from `PLAN.md` (recorded here, before implementation)

1. **Search 1 takes 10 documents instead of 5**, and the budget is 20 paragraphs instead of
   15.
   - The plan predates D3. D3 places its graph-added paragraphs at ranks 6–10, so a
     5-document first search would discard exactly the evidence that made D3 the winner.
   - Starting from R's full context makes "does iteration help beyond R?" a clean
     comparison.
   - The larger context needs the overflow rule above, instead of the plan's "no pruning".
2. **At most 3 LLM calls instead of 4.** An `assess` after the last permitted search could
   not change anything, so it is not run.

## Phase 4 — multi-agent investigation

Recorded 2026-10-01 (revision 2 of the draft), before any Phase 4 code exists and before
any multi-agent output has been produced.

**The question, stated accurately.** The verifier sees the accepted evidence *after* the
researchers have searched, and may request one corrective search. Phase 4 therefore
tests whether independent verification:

- improves **evidence selection** (a smaller, more relevant answer context);
- **detects missing support**;
- **avoids unnecessary corrective searching**.

It does **not** test whether a verifier fixes Phase 3's early over-searching in general.
No routing stage is added.

### 4.1 Fixed conditions

- **Search backend.** The frozen D3 is the only search backend. X1 is not used.
- **Retrieval budget,** the Phase 4 counterpart of Phase 3's 10 + 5 + 5 departure
  (§3.8):
  - search 1 is the question verbatim and takes D3's top 10 unseen (exactly R's
    context);
  - each later search takes the top 5 unseen;
  - at most 3 searches and **20** unique paragraphs.
- **LLM budget:** at most **7** calls.
- **Answer interface.** The synthesizer is the frozen `answer_v2` with schema v2,
  `render_v1` and the system prompt, unchanged.
- **Unchanged from Phase 3:** the model and options, and `tok_v1` for every query
  comparison and length.
- **Splits.** `dev_smoke` and `dev` for runs; `eda` own contexts for the format check.
  Blind splits stay closed.
- **Framework.** A LangGraph `StateGraph`. Roles run **sequentially**. The LLM only
  returns JSON; searches and transitions are deterministic code.

### 4.2 Evidence state

- **`seen`:** every paragraph retrieved so far, in retrieval order, grouped by the search
  that returned it.
- **`accepted`:** the paragraphs a researcher has accepted. Ids are globally unique and
  ordered by first acceptance.
- **The candidate pool:** `seen` minus `accepted`, in retrieval order.

Every researcher sees two groups:

1. **Accepted evidence,** read-only: the paragraphs accepted so far, labelled `[A1]`,
   `[A2]`, …;
2. **Candidates,** selectable: **every** searched-but-unaccepted paragraph so far,
   labelled `[P1]`, `[P2]`, …

So a paragraph retrieved early but not accepted for requirement 1 stays available to
later researchers. Researcher 2 can also resolve phrases such as "that singer" from the
evidence accepted for requirement 1.

### 4.3 Roles, flow and routing

**Planner** (`planner_v1`, 1 call) → 1–2 requirements, in the order they must be found.
Then, for each requirement r = 1, …, R in order:

1. **Planned search r.**
   - For r = 1 the query is the question verbatim (top 10 unseen).
   - For r = 2 the query is researcher 1's `next_query` (top 5 unseen).
2. **Researcher r** (`researcher_v1`, 1 call) sees the accepted evidence and the whole
   candidate pool. It returns the candidate numbers it accepts and a finding. If a next
   requirement exists, it also returns `next_query`.

Then the **verifier** (`verifier_v1`, 1 call) sees the question, the requirements and the
accepted evidence, and nothing from the researchers' findings, queries or traces.
"Independent" means a separate, context-isolated LLM call with the **same model**, not an
independent model and not a statistically independent judge. It
returns, per requirement, `supported` with grounded references, plus `sufficient` and, if
not sufficient, one corrective requirement and query.

If the verifier requests a correction:

1. a **corrective search** with its query (top 5 unseen);
2. a **corrective researcher** for that requirement, with the same inputs as any
   researcher;
3. a **verifier recheck**, whose corrective fields are logged and ignored.

Finally the **synthesizer** (`answer_v2`, 1 call) answers from the accepted evidence.

**Search 1 is unconditional.** It always runs with the question verbatim, and is exempt
from `no_query` and `duplicate_query`; those checks apply only to search 2 and the
corrective search. (`no_progress` cannot occur for search 1 on the corpus, but would be
logged.)

**Routing table** for search 2 and the corrective search. The first rule that applies
decides; the reason is always logged.

| Situation | Search | Next node |
|---|---|---|
| The query is empty after `tok_v1`, or absent (`no_query`) | skipped | the researcher for that step still runs on the accumulated candidate pool |
| The query equals the question or an earlier query after `tok_v1` (`duplicate_query`) | skipped | the same researcher, on the candidate pool |
| The search budget (3) is used up (`max_searches`) | skipped | the same researcher, on the candidate pool |
| The search returns no unseen paragraph (`no_progress`) | ran, added nothing | the same researcher, on the candidate pool |
| The candidate pool is empty when a researcher would run (`no_candidates`) | — | that researcher is skipped; next: the following requirement, else the verifier (or, after a correction, the recheck) |
| Otherwise | ran | the same researcher, on the enlarged pool |

**Invalid outputs** are logged, never retried, and handled deterministically:

| Role | On failure |
|---|---|
| Planner | one requirement, the question itself (`planner_error`) |
| Researcher | accepts nothing and gives no `next_query` (`researcher_error`); the flow continues by the table, so with two requirements search 2 is skipped as `no_query` and researcher 2 still runs |
| Verifier | no corrective search; the synthesizer follows (`verifier_error`) |
| Verifier recheck | logged only (`recheck_error`) |

- **Empty evidence.** If nothing is accepted, the synthesizer receives search 1's top 10,
  R's context (`no_accepted_evidence`).
- **Budget:** at most 1 planner + 2 researchers + 1 verifier + 1 corrective researcher + 1
  recheck + 1 synthesizer = **7** calls; at most 3 searches and 20 paragraphs.

**Overflow rule (Phase 4),** applied to every researcher, verifier and synthesizer prompt.
When a prompt exceeds 7,680 tokens, paragraphs are dropped one at a time in this order:

1. **candidates** (researcher prompts only), by Phase 3's evidence-preserving priority:
   search 1's lowest rank first, then earlier follow-up batches, with the newest batch
   protected;
2. then **accepted** paragraphs, by the same priority applied to the search each came
   from.

Display order is kept; dropped ids are logged per call. Final-context Joint Recall uses
the synthesizer's displayed ids; cumulative Joint Recall uses every paragraph searched.

**Trace:** every role output, search, skip and stop reason, and every call record
(tokens, wall time, dropped ids).

### 4.4 Schemas and runtime contracts

- **`PlannerOutput`** `{requirements: list[str]}`.
  - 1–2 items, each non-empty after stripping and at most 25 `tok_v1` tokens.
- **`ResearcherOutput`** `{accepted: list[int], finding: str, next_query: str}`.
  - Checked at runtime with N = the number of candidates shown: each accepted number lies
    in 1..N, with no repeats.
  - If a next requirement exists, `next_query` is non-empty after stripping and at most 12
    `tok_v1` tokens; otherwise `next_query` is "".
- **`VerifierOutput`** `{requirements: list[{supported: bool, references: list[{paragraph:
  int, sentence: int}]}], sufficient: bool, corrective_requirement: int, corrective_query:
  str}`.
  - Exactly one entry per requirement.
  - **Grounding:** each reference names a displayed paragraph and a sentence that exists
    in it. `supported = true` requires at least one reference; `supported = false`
    requires none.
  - `sufficient` equals "every entry is supported".
  - If sufficient: `corrective_requirement = 0` and `corrective_query = ""`. Otherwise
    `corrective_requirement` is the number of an unsupported requirement, and
    `corrective_query` is non-empty and at most 12 `tok_v1` tokens.
- **Rendering.** Requirements appear as `1. …` and `2. …`. Paragraphs use `render_v1`,
  numbered within each prompt: accepted evidence with `[A#]` in place of `[P#]`, and
  candidates and verifier paragraphs with `[P#]`.

### 4.5 Prompt texts (frozen here)

`planner_v1`:

```text
Break the question into the evidence that must be found to answer it. Do not answer the
question.

Fill the JSON field:
- "requirements": one or two short statements of the facts that must be found, in the
  order they must be found. Use two only when the second fact depends on an entity the
  first fact identifies, or when the question compares two named entities.

Question: {question}
```

`researcher_v1`, with `{next_query_rule}` taking one of the two frozen forms below:

```text
You are collecting evidence for one requirement of a question. Do not answer the
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

Question: {question}
```

When a next requirement exists:

```text
- "next_query": a search query of at most 12 words for the next requirement ("{next_requirement}"), naming any entity the evidence identifies.
```

Otherwise:

```text
- "next_query": "" (there is no further requirement).
```

When nothing has been accepted yet, `{accepted}` is the line `(none)`.

`verifier_v1`:

```text
Check whether the paragraphs below support every requirement of the question. Judge only
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

Question: {question}
```

### 4.6 Format check: a miniature flow, before any `dev` call

- **Inputs:** the same 20 `eda` questions, alternating by index between condition (a),
  their own ten paragraphs, and condition (b), the same with one gold paragraph removed
  (as in §3.3). That gives 10 flows of each.
- **The flow:** the complete multi-agent flow runs on each. The search tool is a
  deterministic stand-in: the question's own paragraphs in supplied order, giving the
  top 10 unseen for search 1 and the top 5 unseen afterwards. Later searches therefore
  usually report `no_progress`, and the following researcher works on the accumulated
  pool. This exercises:
  - both researcher prompt variants;
  - the complete two-requirement path up to verification;
  - verifier `true` and `false`;
  - the corrective researcher and recheck;
  - grounded-reference validation.

  It costs about 20 × 5–7 calls.
- **Per-role gate:** for each role (planner, researchers counted together, verifier
  counted with the recheck), failures of the schema or runtime contract must be at most
  **10% of that role's calls** (2 of 20 for the planner).
- **Revision rule:** one format-only revision per failing role is allowed, logged as an
  amendment with its diff, and the whole check is rerun once. There is no second
  revision.
- **Viability gate:** at least one valid two-requirement plan; at least one valid output
  of each researcher variant; at least one complete two-requirement path reaching the
  verifier; at least one valid verifier `sufficient = true` (with grounded references)
  and one `false`; and at least one flow that reaches the **corrective researcher and the
  verifier recheck**. Otherwise work stops for review.
- **Freeze:** the hashes of the three prompts and the two `next_query_rule` forms, the
  three schemas and contracts, the multi-agent graph source and its routing rules are
  recorded in §A before any `dev` call.

### 4.7 Runs and comparisons

1. The format check (about 110–140 calls).
2. The `dev_smoke` smoke run: mechanics only.
3. One full `dev` run: about 100 × 5–6 calls, roughly 3–4 hours.
4. **Cache-only replays under the Phase 4 evaluation code,** each aborting on any cache
   miss, ranking or context mismatch:
   - the Phase 3 agent (a new replay command, reproducing its batches, contexts and
     every call);
   - R (D3 K = 10);
   - BM25 K = 10;
   - the matched control (D3 K = 16).
5. **Comparisons,** paired over the 100 `dev` questions, all exploratory: the multi-agent
   system against the single agent, R and the matched control. Every manifest has the same
   Phase 4 evaluation-code hash, with the shared answer interface asserted equal. The
   confirmatory comparison (single → multi) is the preregistered Holm family on
   `test_agents` in Phase 5.

**Budget checks** (by construction, verified rather than hypothesized): every question
uses at most 7 calls, 3 searches and 20 paragraphs. The distribution of calls is
reported.

### 4.8 Phase 4 hypotheses (exploratory, `dev`, n = 100)

| Id | Prediction | Estimate and threshold |
|---|---|---|
| H4-1 | Better evidence selection: a smaller answer context than the single agent without losing gold completeness | mean synthesizer context **< 15.95** paragraphs, **and** final-context Joint Recall at least R's 0.84 − 0.03 (paired interval against R reported) |
| H4-2 | The multi-agent system has higher answer F1 than the single agent (planning, querying, selection and verification change together, so no single mechanism is isolated) | multi − single F1 **> 0**: paired bootstrap, point estimate |
| H4-3 | The verifier detects missing support, under the **incomplete-gold-evidence proxy** | among initial verifications whose accepted evidence lacks a gold paragraph (a proxy for missing support, not proof of it), the share judged not sufficient is at least **0.60** (raw share and count) |
| H4-4 | Few unnecessary corrective searches | among questions whose accepted evidence already has both gold paragraphs at the first verification, the share with a corrective search is at most **0.40** (raw). This is a different decision from Phase 3's 0.74 re-search rate, which is shown for context only |
| H4-5 | Fewer "supported" answers under the **incomplete-gold-evidence proxy** | among questions whose final context lacks a gold paragraph, the share of answers marked `supported = true` is lower than the single agent's (raw shares and counts). The proxy is not proof that an answer is unsupported |
| H4-6 | Cost: the multi-agent system takes at least as much LLM time as the single agent | median summed LLM time per question, multi / single **≥ 1.0** (raw ratio) |

Question-level quality differences get 95% bootstrap intervals. Behavioural shares and
latency figures are reported raw.

### 4.9 Tests required before any multi-agent run

- **Search rule:** search 1 is the question verbatim and equals R's top 10. Regression test:
  search 1 **always executes** with the verbatim question, never skipped as
  `duplicate_query` or `no_query`, while a later query equal to the question is skipped.
- **The pool:** later researchers see every accepted and unaccepted paragraph so far.
  This includes the regression case: search 1 retrieves both gold paragraphs, researcher
  1 accepts one, and researcher 2 can still accept the other.
- **Acceptance:** accepted ids are globally unique and ordered by first acceptance.
- **Routing:** every row of the routing table, for planned and corrective searches, on a
  stub LLM. This includes researcher 2 still running when search 2 is skipped, and
  `no_candidates`.
- **Errors and budget:** each role's error handling; at most 7 calls, 3 searches and 20
  paragraphs.
- **Contracts:** violating and boundary cases for every schema. This includes verifier
  references to a non-displayed paragraph or a missing sentence, `supported = true`
  without references, and `supported = false` with references.
- **Verifier independence:** the verifier prompt contains no researcher finding, query or
  trace text.
- **Synthesizer:** it uses the frozen `answer_v2` hashes; the empty-evidence fallback.
- **Overflow:** the Phase 4 rule (candidates first, then accepted), with dropped ids
  logged.
- **Prompt texts:** equal to this preregistration.
- **Replays:** the agent replay reproduces the Phase 3 records and aborts on a cache miss
  or context mismatch.
- **Splits and manifests:** blind splits refused; manifests carry every hash, the budget
  and the D3 fingerprints.

### 4.10 Departures from `PLAN.md` (recorded before implementation)

1. **Retrieval budget.** 10 + 5 + 5 and up to 20 paragraphs, instead of 5 per search and
   15: the counterpart of Phase 3's departure (§3.8).
2. **The planner does not write the first query.** Search 1 is the question verbatim, so
   that the first evidence equals R's context, as in Phase 3.
3. **Researchers return paragraph numbers rather than sentence ids,** and select from the
   whole accumulated candidate pool rather than only the newest search. The synthesizer
   produces its own sentence citations. The verifier keeps grounded sentence references.

## Phase 4b — structured selection and verification (approved)

### 4b.1 Purpose and evidential status

**Motivation.** On `dev` (n = 100), the frozen Phase 4 system lost evidence after
retrieval:

- **Evidence funnel:** Joint Recall was 0.84 after search 1, 0.96 over the whole search
  pool, and 0.67 in the synthesis context.
- **Selection loss:** 29 questions had both gold paragraphs retrieved but not both
  retained.
- **Verifier failures:** 43 of 160 verifier calls failed their contract, and the valid
  verdicts barely distinguished complete from incomplete evidence.

Phase 4b changes the **interfaces** of selection and verification so that these failure
modes are structurally impossible or deterministic. It does not tune the existing prompts.

**Evidential status.**

- Phase 4b is motivated by errors observed on `dev`. Its `dev` comparison is
  **adaptive development evidence**, even though everything below is fixed before
  another `dev` run.
- Only the untouched `test_agents` comparison in Phase 5 can confirm a result.
- The frozen Phase 4 system (`multi_agent.py`, `verifier_v2`) and its `dev` results are
  not changed. Phase 4b is a new module, `multi_agent_4b.py`.

### 4b.2 What stays fixed

- The D3 retriever.
- `planner_v1`, its schema and contract.
- The synthesizer: the frozen `answer_v2` interface over the accepted paragraphs. If no
  paragraph is accepted, it falls back to search 1's batch, as in Phase 4.
- The model and its options: `qwen2.5:7b`, temperature 0, seed 0, `num_ctx` 8192,
  `num_predict` 512.
- Search 1 is the question verbatim (R's top 10). Later searches add 5 unseen paragraphs
  each.
- The Phase 4 rules for skipping a search (`duplicate_query`, `max_searches`).
- The Phase 4 overflow rule: candidates are dropped first, then accepted paragraphs, by
  evidence-preserving priority.

### 4b.3 Structured selection: `researcher_v2`

**One call per requirement.**

- The call shows read-only context: the paragraphs already accepted, labelled `[A#]`.
- It also shows every pool paragraph not yet accepted, as `[P1]`…`[Pn]` in pool order.
- The model must return **one boolean decision per candidate paragraph**.

**The constrained output.** The JSON schema sent through Ollama's `format`:

- has a required boolean property for each candidate, `"P1"` to `"Pn"`;
- adds a required string `"next_query"` only when a next requirement exists;
- sets `additionalProperties: false`.

Ollama 0.32.5 enforces required properties while decoding. A throwaway-prompt check
before this draft confirmed it: every key was present, even when the prompt asked for
one. A decision can therefore never be missing or misaligned.

**The deterministic rules:**

- **Keeping paragraphs:** a paragraph is accepted when its decision is `true`. Accepted
  ids are globally unique, in order of first acceptance; within a call, they follow
  candidate order.
- **Fail-open selection:** if the call fails hard (transport error, invalid JSON, or
  truncation, `done_reason = length`), **every candidate shown in that call is accepted**.
  The failure is logged. A researcher failure never rejects a candidate it was shown;
  later context overflow may still remove accepted evidence.
- **`next_query`:** truncated to its first 12 `tok_v1` tokens. If it is empty, the next
  requirement's text is used instead, truncated the same way. Both cases are logged.
- **No candidates:** if no unaccepted candidates remain, the researcher is skipped
  (`no_candidates`), as in Phase 4.

`researcher_v2`:

```text
You are selecting evidence for one requirement of a question. Do not answer the question.

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
{next_query_rule}
```

- **`{n}`** is the number of candidates.
- **`{next_query_rule}`** is the frozen `NEXT_QUERY_RULE_WITH` text from Phase 4 when a
  next requirement exists, and empty otherwise.
- **`{accepted}`** is `(none)` when nothing has been accepted.

### 4b.4 Per-requirement verification: `requirement_verifier_v1`

**One call per requirement.** Each call sees:

- the question and all the requirements, for context;
- the one requirement to check;
- the accepted paragraphs only, as in Phase 4, with no researcher text.

**The constrained output** (`additionalProperties: false`):

- `"references"`: an array of at most 3 strings, with `uniqueItems: true`.
  - Each item's `enum` lists the valid `"P<p>:S<s>"` labels of the displayed sentences.
  - The enum and `maxItems` were checked on Ollama 0.32.5 the same way: labels outside the
    enum could not be produced.
  - `uniqueItems` is **not** enforced by Ollama 0.32.5 (checked: a repeated label was
    produced). Code therefore removes duplicate references, keeping first occurrences,
    and logs it.
  - References are thus **structurally valid**: every one names a displayed sentence.
    They are not necessarily evidentially correct: the model can still cite an existing
    sentence that does not state the requirement.
- `"supported"`: a boolean.
- `"query"`: a string.

**Deterministic aggregation**, all in code and logged:

- **A requirement is supported** only if `supported = true` **and** at least one
  reference is given. `supported = true` with no reference counts as unsupported.
- **`sufficient`** means every requirement is supported.
- **The corrective requirement** is the **lowest-numbered unsupported** requirement. Its
  corrective query is that call's `query`, truncated to 12 `tok_v1` tokens. If the query
  is empty, the requirement's text is used instead, truncated the same way.
- **A hard failure** (transport error, invalid JSON, or truncation) counts the
  requirement as **unsupported**. It never skips the corrective path, so errors cannot
  hide a correction.
- **No accepted paragraphs:** the verifier is not called, and every requirement is
  unsupported. The corrective requirement is **requirement 1**. Its corrective query is
  requirement 1's text, truncated to 12 `tok_v1` tokens.
- **Combining verdicts after a recheck:** each requirement keeps its stored per-requirement
  verdict, except the corrective requirement, whose verdict the recheck replaces. A
  recheck with a hard failure counts that requirement as unsupported. The final
  `sufficient` is the conjunction of the combined verdicts. A skipped recheck
  (`no_new_evidence`) leaves every stored verdict unchanged.

`requirement_verifier_v1`:

```text
Check whether the paragraphs below state one requirement of a question. Judge only from
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
  missing fact, naming any entity found in the paragraphs; otherwise "".
```

`{requirements}` lists all requirements as `1. …`, `2. …`.

### 4b.4a The structured-decoding runtime

- **Version:** the Ollama server version (`0.32.5`) is recorded in every Phase 4b manifest
  and is a comparison invariant. A run under any other version is not comparable.
- **Cache:** every cache record keeps the dynamic schema sent with the call, as the
  current client already does. The schema is part of the cache key.
- **G0, the live mechanics gate,** runs before C3. It uses synthetic prompts with
  invented text and no project data, and passes only if all of these hold:
  - required properties are always present;
  - `additionalProperties: false` admits no extra key;
  - enum-constrained references never fall outside the enum;
  - `maxItems` is respected;
  - duplicate references are removed by code;
  - changing only the schema changes the cache key.

  A failure stops work for review.

### 4b.5 Routing, the no-new-evidence rule, and budgets

1. `plan`.
2. Search 1, then the researcher for requirement 1.
3. If there are two requirements: search 2 with requirement 1's `next_query`, then the
   researcher for requirement 2.
   - Researcher 2 runs even when search 2 is skipped or adds nothing new, because it
     judges a different requirement over the unaccepted pool.
   - It is skipped only when no candidates remain.
4. Per-requirement verification. If the evidence is sufficient, go to `answer`.
5. Otherwise, one corrective search with the corrective query.
   - **If the corrective search is skipped or adds no new paragraph,** the corrective
     researcher and the recheck are **skipped** (`no_new_evidence`). The verdict stands.
   - **Otherwise,** the corrective researcher runs for the corrective requirement.
   - **If the corrective researcher accepts no new paragraph,** the recheck is skipped
     (`no_new_evidence`).
   - **Otherwise,** the recheck verifies **only the corrective requirement** against the
     updated accepted set. The other requirements keep their verdicts.
6. `answer`.

| Budget | Value |
|---|---|
| Searches | at most 3 (10 + 5 + 5 paragraphs; pool at most 20) |
| LLM calls | at most 8 for a two-requirement plan: planner 1, researchers 2, verifier 2, corrective researcher 1, recheck 1, synthesizer 1. At most 6 for a one-requirement plan |
| Output tokens per call | `num_predict` 512 |
| Latency (retention criterion, §4b.9) | median summed LLM time per question on `dev` at most 150 s |

### 4b.6 Calibration subsets C3 and C4

Two fresh `eda` subsets, disjoint from the format check, C1, C2 and each other, taken in
`eda` split order:

- **C3** (development): bridge questions 49–72 and comparison questions 13–18 (24 + 6),
  membership hash `f06f4aede1de969bb5df8081b053b421ffb93461b1ef3e36f5ccb85cae42a3fd`;
- **C4** (validation): bridge questions 73–96 and comparison questions 19–24 (24 + 6),
  membership hash `82205976797ec1b2f0bc1682a3867e83837924e8d2704c43833bbb93001b363a`.

The hashes use `data.ids_hash` (sorted ids). Tests pin both hashes and sizes, and check
that the subsets are disjoint from each other, from the format-check questions, and from
C1 and C2.

**Procedure, the same as A5 for each subset:**

- **The flow.** It uses the §4.6 own-context search stand-in. Conditions alternate by
  index: (a) all ten own paragraphs, (b) one gold paragraph removed by the §3.3 rule.
- **Hand labels and reference requirements.** They are made from the planner's outputs
  with the training labels, under the C1/C2 conventions:
  - `label`;
  - `superfluous`;
  - corrected `requirements`;
  - `incomplete_probe_insufficient`;
  - `complete_probe_sufficient`.

  They are committed as `calibration/requirements_c3.jsonl` and `requirements_c4.jsonl`
  before any probe on that subset.
- **Probes.** For every question there are two probes, one with both gold paragraphs and
  one with a gold paragraph removed. Each probe makes one `requirement_verifier_v1` call
  per reference requirement and aggregates them deterministically.

### 4b.7 Reliability gates (each must hold on the subset being evaluated)

| Gate | Measure | Threshold |
|---|---|---|
| G1 Contract reliability | Hard failures per role (planner, researchers, verifier including recheck), over the flow calls; and verifier hard failures over the probe calls. Synthesizer schema errors are reported separately | ≤ 0.02 each; synthesizer schema errors 0 |
| G2 Selection retention | Gold paragraphs **displayed in every first-verification call** (after planned selection and overflow pruning) ÷ gold paragraphs in the pool after the planned researchers, pooled over questions. A question whose verifier is not called (nothing accepted) contributes 0 | ≥ 0.95 |
| G3 Complete evidence (S3) | Probes with both gold paragraphs aggregated as sufficient, over all questions | ≥ 0.70 |
| G4 Incomplete evidence (S4) | Probes with one gold paragraph removed aggregated as not sufficient, over all questions | ≥ 0.70 |
| G5 Mechanics | No corrective researcher or recheck after a corrective path with no new evidence; no budget exceeded | none observed |

**Diagnostics** (reported, not gated):

- **Selection:**
  - pre-overflow acceptance (gold accepted ÷ gold in the pool);
  - gold paragraphs removed by overflow;
  - selection precision;
  - mean accepted and mean displayed paragraphs.
- S3 without questions flagged `complete_probe_sufficient = false`, and S4 without
  questions flagged `incomplete_probe_insufficient = false`;
- the planner's S1;
- the share of `supported = true` outputs normalized to unsupported because they gave no
  reference.

### 4b.8 Order and revision rule

1. Implement and test (no LLM calls).
2. **G0**, the live mechanics gate (§4b.4a).
3. **C3:** the flow, then labels and reference requirements (committed), then the probes.
   Gates G1–G5 are computed.
4. **If a gate fails on C3:**
   - **G2 or G3/G4:** at most **one prompt-only revision** of each failing component
     (`researcher_v2` for G2, `requirement_verifier_v1` for G3/G4). It is written as an
     amendment with its diff and rationale before C4 is opened. Schemas, aggregation and
     routing do not change.
   - **G1 or G5:** a failure indicates a defect. Work stops for review.
5. **C4:** the flow, labels and probes with the (possibly revised) components. **All gates
   must hold on C4.** There are no further revisions.
6. **If any gate fails on C4,** Phase 4b is **rejected before `dev`**. The frozen Phase 4
   system remains the multi-agent system for Phase 5, and the result is reported as a
   negative development result.
7. **If C4 passes,** every Phase 4b hash is frozen in §A before any `dev` call:
   `researcher_v2`, `requirement_verifier_v1`, schema builders, aggregation and routing
   source.

### 4b.9 `dev` runs, comparisons and the retention rule

- **Runs:** `dev_smoke` (mechanics only), then one full `dev` run. Then cache-only
  reproductions of the frozen Phase 4 system, the single agent, R, the matched control
  and BM25, all under the Phase 4b evaluation code at one clean commit.
- **Comparisons:** Phase 4b against frozen v2, R, the single agent and the matched
  control.
  - Paired over the 100 `dev` questions, with 95% bootstrap intervals.
  - Reported measures:
    - F1, EM, final-context Joint Recall;
    - the evidence funnel, with acceptance before overflow and gold removed by overflow;
    - mean paragraphs in the final answer context;
    - cost: logical calls, new generations, LLM time, prompt-token totals and overflow
      counts;
    - role failures.
  - All are labelled **adaptive development evidence**.

**Retention rule.** Phase 4b replaces the frozen Phase 4 system in the Phase 5
confirmatory family ("single agent → multi-agent" on `test_agents`) only if **all** of
these hold on `dev`:

| Condition | Threshold |
|---|---|
| R1 Safe selection | Final-context Joint Recall ≥ 0.81 (R's 0.84 − 0.03, the H4-1 threshold) |
| R2 No material loss | Phase 4b F1 − frozen v2 F1 ≥ −0.02 (point estimate) |
| R3 Latency | Median summed LLM time per question ≤ 150 s |
| R4 Reliability | Hard-failure rate ≤ 0.02 for each of planner, researchers and verifier (including the recheck); synthesizer schema errors 0 |
| R5 Selection, not keep-everything | Mean paragraphs in the final answer context **< 16**, the matched-context control's size. The comparison with R's 10 paragraphs is also reported |

**Otherwise** the frozen Phase 4 system stays in the family. Exactly one multi-agent
system is run on `test_agents`; the other is not.

### 4b.10 Phase 4b hypotheses (adaptive development evidence, `dev`, n = 100)

| Id | Prediction | Threshold |
|---|---|---|
| H4b-1 | Selection keeps the evidence that was found | Final-context JR, 4b − v2, > 0 with the whole 95% interval above 0 |
| H4b-2 | Retention given retrieval | Gold paragraphs displayed at the first verification ÷ gold in the pool (the G2 measure) ≥ 0.95 |
| H4b-3 | The verifier discriminates | Among valid first verifications, the "sufficient" rate with both gold paragraphs displayed minus the rate with one missing ≥ 0.20. A first verification is **valid** only if **every** per-requirement verifier call for that question succeeded |
| H4b-4 | Few unnecessary corrective searches | Among questions with both gold paragraphs at the first verification, a corrective search for at most 0.40 |
| H4b-5 | Answer quality | F1, 4b − v2, > 0 (point estimate; the interval is reported) |
| H4b-6 | Cost | Median LLM time per question, 4b / v2, ≤ 1.35 |

### 4b.11 Tests required before any Phase 4b LLM call

- **Prompt texts:** equal to this section. The frozen Phase 4 hashes are unchanged.
- **Schema builders:**
  - one required boolean per candidate, and `next_query` only when a next requirement
    exists;
  - reference enums equal to the displayed `P<p>:S<s>` labels, with `maxItems` 3.
- **Aggregation:** every rule in §4b.4, including `supported = true` with no references,
  the lowest-numbered unsupported requirement, query truncation and fallback, and hard
  failures counted as unsupported.
- **Fail-open selection:** a researcher hard failure accepts every candidate shown.
- **Routing:**
  - the no-new-evidence rule for a skipped corrective search, a corrective search with no
    new paragraph, and a corrective researcher that accepts nothing;
  - researcher 2 running after a skipped search 2;
  - the recheck covering only the corrective requirement.
- **Budgets:** at most 8 calls, 3 searches and 20 paragraphs.
- **Calibration subsets:** C3 and C4 membership hashes and sizes pinned. Disjoint from
  each other, from the format-check questions, and from C1 and C2.
- **Runtime:**
  - the Ollama version is recorded in manifests and checked as a comparison invariant;
  - the schema is part of the cache key;
  - duplicate references are removed.
- **Blind splits:** refused. Manifests carry every Phase 4b hash.

### 4b.12 Expected cost

- **G0:** a handful of synthetic calls, seconds.
- **C3 and C4:** about 200 flow calls and about 120 probe calls each, roughly 1.2 hours
  per subset.
- **`dev_smoke` and `dev`:** about 160 + 700 calls, roughly 3.5 hours.
- Every run needs the operator's go-ahead before it starts.

## Phase 5 — freeze and blind test (approved)

### 5.1 Status and the one-way rule

- **Blind until now.** No `test_*` question, context or result has been loaded. This
  section was written from frozen configurations and split hashes only.
- **One-way unblinding.** The final runner loads the blind splits (`unblind=True`) only
  after four steps:
  1. this section is approved;
  2. the runner is implemented and its offline tests pass;
  3. a freeze commit records the remaining hashes (§5.8);
  4. the operator gives the go-ahead.

  After that, no system, prompt, threshold, metric or analysis in this section changes.
- **Failures after unblinding:**
  - **Infrastructure failures** (A7) resume from the cache at the same commit (§5.8).
  - **Defects:** all original artifacts remain immutable. If a defect affects outputs or
    scores, the affected result is marked invalid and cannot support a primary claim. A
    corrected rerun is reported as a post-unblinding protocol deviation, alongside the
    invalid original. Unaffected results remain primary.
- **Runtime.** Every Phase 5 LLM run comes from the one freeze commit, with a clean tree.
  - Every manifest records Ollama `0.32.5` and the model digest
    `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` (`qwen2.5:7b`,
    Q4_K_M).
  - Comparisons use the final rule: the same clean commit and the same evaluation code.
  - The legacy-runtime exception (A7) does not apply to any blind split.

### 5.2 Splits (fixed in Phase 0)

| Split | n | Composition | Membership hash |
|---|---|---|---|
| `test_retrieval` | 1,000 | proportional | `8654f35c1493a822245eeb07208c95999f900e59dc6adff01a5e43167a0a28ef` |
| `test_rag` | 200 | proportional (≈ 160 bridge / 40 comparison), inside `test_retrieval` | `57201688a97654e73d633128498c00ff92c1f0e6984aa296b5ad506e6ab83d8d` |
| `test_agents` | 100 | 50 bridge + 50 comparison, inside `test_retrieval`, drawn independently of `test_rag` | `6b72130cb1f09e7a21129f81d50876525c1306f4df426777f3c11e1ce470f447` |

Corpus fingerprint `67db29c2a555974d0d16e8a3bb9c9865ee8c542d6d268ed88e0232e340fee3aa`.

### 5.3 The frozen systems

**Shared answer interface** (every system except closed-book):

| Artifact | SHA-256 |
|---|---|
| `answer_v2` | `0a2aa59c597d9730ed73ea358b22cd76787cde9910e12888397076b7bb003f72` |
| schema v2 | `c29af85a08782e161cae0ab7e4eb847e19866b4bc7910d84640c30d12ab67847` |
| answer validator v2 | `252c82c0f31ab1013af85beeaa777b8eab0a5ffa2ba4dadec1e9961eb2d4910c` |
| system prompt | `16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172` |
| `render_v1` | `935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6` |
| `closed_book_v2` (closed-book only) | `a89deeb1d5aafd0fccafbcb44a10e47067f610bef75dcf8bcbfa878e9cdbcb79` |

**The seven systems:**

| System | Definition | Frozen in |
|---|---|---|
| **Baseline RAG** | BM25 C0 (`bm25s` 0.3.11 lucene, Snowball, stop words removed, k1 1.5, b 0.75), K = 10 | §1.3–1.5, A1 |
| **Improved RAG (R)** | D3 = G*(RRF(BM25 C0, TAS-B)), K = 10 | §2, Phase 2 selection |
| Matched-context control | D3 RAG at K = K_match = 16 | Phase 3 record |
| Closed-book | `closed_book_v2`, no context | A1 |
| Oracle | the gold paragraphs only | §1 |
| **Single agent** | Phase 3 LangGraph agent with `assess_v2` | §3, A3 |
| **Multi-agent** | the frozen Phase 4 system with `verifier_v2` | §4, A4–A6, Phase 4 freeze |

**R's components:**

- TAS-B `msmarco-distilbert-base-tas-b` at revision `b12d9352e776979147078a8975a4885042984fd1`, with corpus embedding
  fingerprint `5411b3a2ec46d23446ecf0d44f27de6665282f88e3d040eb88e82ea9d18a7d4c` (`results/tune/phase2b_selection.json`);
- RRF with c = 60 and a pool of 100;
- the expansion operator: seed depth 5, 5 retained, 5 reserved, forward cap 3, backward
  cap 2;
- graph fingerprint `c0a4b543a343a2bcd9fc2d73a9b345fbb356aeed3a5cf9ac284c4ad9d0a691f3`.

**Single agent:**

- prompt and contracts:
  - `assess_v2`: `2221feb7f911d1cdc90150f930fab21abf0eddd63cf5fdf05d5ebc2048a17724`;
  - assess schema v1: `c84e600907056092f6a7bc0ed85c0ff78790e5b76c72f6d8676456f7cc849915`;
  - assess contract v1: `db8c967d1390891200e1e2afe2dd8da4536993b2f185bb7d59803f5d459a4a95`;
- `agent.py` source: `5e7b844c2416450f5903b31805b655c5b4e98fdf8dfd82c1e5619a8f562a7cdd`;
- budget: 3 searches, 20 paragraphs, 3 LLM calls.

**Multi-agent:**

- every hash in the Phase 4 freeze record;
- `multi_agent.py` source: `1c4b90a8bfa459357de435454c5b0a9d1cc1880c00ac726a38869f013c184d89`;
- `run_eval.PHASE4_VERIFIER = "v2"`;
- budget: 3 searches, 20 paragraphs, 7 LLM calls.

**Excluded, with the reason recorded:**

- Phase 4b: rejected at C4.
- `verifier_v3`: rejected on C2.
- The X1 cross-encoder: failed its post hoc check.
- PRF candidates: not selected in Phase 2.

### 5.4 Runs

| Split | Systems | New generations (approx.) |
|---|---|---|
| `test_retrieval` | BM25 C0 and D3 rankings to depth 100 | none (no LLM) |
| `test_rag` | baseline, R, closed-book, oracle | about 800 |
| `test_agents` | all seven systems | about 1,400 (single agent about 2.75 and multi-agent about 6.2 calls per question) |

- **Order:** `test_retrieval`, then `test_rag`, then `test_agents`.
- **Time:** about 10 hours of generation, in resumable sessions. Each session starts only
  with the operator's go-ahead.
- **Cache reuse:** a cache hit occurs only when the complete frozen cache key matches:
  model digest, rendered prompt (including the system prompt), options and schema. Reuse
  may occur across splits or systems, and is reported as logical calls, cache hits and
  unique new generations.
- **Sessions:** each split is a separate stage of the `final` command (§5.8). A stage runs
  only when invoked explicitly, so the operator starts each session.

### 5.5 The confirmatory family (`test_agents`, n = 100)

- **Metric:** answer token F1 over all 100 questions.
- **Estimate:** the prevalence-weighted overall difference. It weights the mean paired
  difference within each type by validation prevalence: 0.799 bridge, 0.201 comparison.
- **Inference:**
  - paired bootstrap with resampling **within type** (`metrics.paired_bootstrap` with
    strata and weights), 10,000 resamples, seed 0;
  - two-sided add-one p-values;
  - 95% percentile intervals.
- **The three adjacent comparisons:**
  - **F1:** improved RAG (R) − baseline RAG;
  - **F2:** single agent − R;
  - **F3:** multi-agent − single agent.
- **Multiplicity:** Holm–Bonferroni over the three p-values, family-wise α = 0.05.
- **Conclusions, by direction:**

  | Condition | Conclusion |
  |---|---|
  | delta > 0 and Holm-adjusted p < 0.05 | Improvement demonstrated |
  | delta < 0 and Holm-adjusted p < 0.05 | Worsening demonstrated |
  | Otherwise | No clear difference demonstrated |

  - A **practically meaningful improvement** additionally requires delta ≥ 0.03 (the MPI).
  - An interval containing zero is never read as equivalence.
- **Reporting:** the ordinary 95% bootstrap interval is reported alongside the Holm-adjusted
  p-value. The interval itself is not adjusted for multiplicity.
- **One confirmatory endpoint.** Answer F1 is the sole confirmatory endpoint.
  Final-context Joint Recall remains the primary mechanistic retrieval measure, but it is
  not a second confirmatory hypothesis family.
- **Per-type differences** (n = 50 each) are reported with intervals but not tested.

### 5.6 Secondary and exploratory analyses (no claims from p-values)

- **`test_rag` (n = 200, proportional, unweighted):** R − baseline F1 and EM, with paired
  intervals. This is the more precise non-agent estimate (secondary).
- **`test_agents`, all systems, per type and prevalence-weighted overall:**
  - **Answer and grounding:** EM, F1 on answered (with its count), self-reported supported
    rate, citation coverage and validity.
  - **Context and evidence:** final-context Joint Recall; strict supporting-fact F1;
    official-compatible supporting-fact F1 and joint F1.
  - **Paired comparisons:** matched control against the single agent and the multi-agent;
    multi-agent − R.
  - **Reference rows:**
    - oracle;
    - closed-book: EM, F1, the low-confidence rate and F1 when confident. No evidence
      metrics.
- **Agents:**
  - the evidence funnel (search 1 → pool → final context);
  - stop reasons and searches;
  - corrective searches and no-op paths;
  - role failures;
  - mean paragraphs in the final answer context.
- **Cost:** logical LLM calls, cache hits, unique new generations, prompt and output
  tokens, LLM wall-seconds, and retrieval latency.
  - **Percentiles on `test_agents`:** p50 and p95 per question use **weighted empirical
    quantiles**, each question weighted by its type's prevalence divided by its type's
    sample size (0.799/50 for bridge, 0.201/50 for comparison). Per-type quantiles are
    reported as well.
  - **On the proportional splits:** ordinary quantiles.
- **`test_retrieval` (n = 1,000, proportional, unweighted):**
  - Recall@K and Joint Recall@K for K ∈ {2, 3, 5, 10, 20, 50}, exact and title-only, by
    type and overall;
  - a paired interval for D3 − BM25 Joint Recall@10.
- **Weighting rule:** overall `test_agents` figures are always prevalence-weighted, with
  the bootstrap within type. `test_rag` and `test_retrieval` are proportional samples, so
  they are unweighted. The audit (§5.7) is exempt: it is a deliberately balanced
  descriptive sample.

### 5.7 The blinded audit (run in Phase 6; defined now)

- **Sample:** 12 bridge and 12 comparison questions from `test_agents`. Within each type,
  question ids are sorted by `sha256("audit-20260927:" + qid)` and the first 12 are taken.
  The sample is deterministic and independent of every result.
- **Systems:** baseline RAG, R, single agent, multi-agent: **96 outputs**.
- **Presentation:**
  - outputs are ordered by `sha256("audit-order-20260927:" + qid + ":" + system)`;
  - system names, scores, the `supported` flag, the gold answer and the gold supporting
    facts are hidden;
  - the labeller sees the question, the answer, the cited sentences, and the paragraphs
    displayed to the answer call;
  - a citation that does not resolve to a displayed sentence is shown visibly as
    unresolved, never omitted.
- **Labeller:** the user, not an LLM.
- **Labels, two independent fields per output.**
  - **Evidence support,** applied in this order of precedence (claims are the answer's
    material factual claims; "stated" means stated by the displayed context):

    | Label | Rule |
    |---|---|
    | `cannot_determine` | Ambiguity in the answer or the context prevents applying the other labels |
    | `contradicted` | At least one material answer claim is explicitly contradicted. Takes precedence over the labels below |
    | `fully_supported` | Every material factual claim is stated, and none is contradicted |
    | `partially_supported` | At least one claim is stated, at least one is not, and none is contradicted |
    | `unsupported` | No material claim is stated or contradicted |

  - **Citation validity:**
    - `valid`: at least one citation exists, every citation resolves to a displayed
      sentence, and every cited sentence is relevant to at least one factual answer claim,
      whether it supports or contradicts that claim;
    - `invalid`: otherwise, including no citations and any unresolved citation.
    - Whether every claim is covered is judged separately, by the support label.
- **The primary manual measure: the unsupported-claim rate.** Among the outputs a system
  marked `supported = true`, it is the share labelled `unsupported` or `contradicted`.
- **Reported per system:**
  - bridge and comparison counts separately, with exact (Clopper–Pearson) 95% intervals;
  - the support and citation distributions over all outputs.
- **Pooled figures:** if the pooled 24-output rate and its Clopper–Pearson interval are
  shown, they are called the **balanced-audit rate**. They are not a population-prevalence
  estimate, and the audit is exempt from the general weighting rule.
- **Status:** descriptive. The audit is not part of the confirmatory family.

### 5.8 The final runner (implemented only after this section is approved)

- **The command:** `run_eval.py final --stage retrieval|rag|agents`.
  - **Order:** each invocation runs one stage, and the stages must run in §5.4 order. A
    stage refuses to start before the previous stage's completion marker exists. There is
    no "run everything" mode, so the operator starts each session.
  - **It refuses to start unless all of these hold:**
    - the tree is clean at the freeze commit;
    - every hash in §5.2–5.3 matches;
    - Ollama is `0.32.5` with the full frozen digest.
- **Immutability and resumption:**
  - A completed test output, meaning a results file plus its manifest plus the stage's
    completion marker, is immutable and is never overwritten.
  - A partial run may resume only if its recorded commit and configuration exactly match
    the freeze commit and the frozen configuration. Otherwise the stage refuses.
- **Atomic finalization:**
  - Records are appended to a partial file during a run.
  - Each system's results file and manifest are written to temporary paths and renamed into
    place only when the system is complete.
  - The stage marker is written last, by atomic rename.
  - An interruption therefore cannot make a partial result appear complete.
- **The analysis command:** reads only stored runs and computes §5.5–5.6 exactly as
  specified.
- **The audit sheet generator:** builds the §5.7 sheet.
- **Offline tests before unblinding** (no `test_*` access):
  - hash and refusal checks;
  - the weighted within-type bootstrap and Holm on synthetic data;
  - the audit sampler and order on synthetic ids;
  - a dry run of the full runner on `dev` (`--dry-run`), which reuses the cache and loads
    no blind split.
- **The freeze commit records in §A,** before unblinding: the evaluation-code hash, the
  runner source hash, and the D3 and BM25 `describe()` hashes.

### 5.9 Stop points

1. This draft, for review.
2. After the runner, the analysis code and their offline tests (no `test_*` access).
3. Before the one-way unblinding: the operator's go-ahead.
4. Between run sessions: the operator's go-ahead for each long run.

### 5.10 Departures from `PLAN.md`

1. **The command:** one `final` command replaces the planned multi-flag command line.
2. **Excluded systems:** Phase 4b and the other rejected variants are excluded (§5.3).
3. **Audit labels:** `cannot_determine` and citation validity are defined here in full,
   before any audit.

## P. Post-hoc analyses

Analyses defined **after** inspecting results. They never replace a preregistered
selection or outcome, and every result and manifest they produce is marked
`posthoc: true`.

### P1 — X1: cross-encoder reranking of D3 (recorded 2026-09-30, before X1 is run)

**Motivation.** D4 (cross-encoder reranking of D2) did unexpectedly well on `tune`. It
raised Recall@3, and it raised second-hop Recall@10 over D2, contrary to H2b-4. That
revealed an untested composition: reranking the selected retriever D3. X1 is defined
only after the Phase 2b `tune` results were seen.

**Definition.**

- X1 reranks **exactly the first 50** documents of the stored D3 `tune` ranking. That
  ranking is identified by the SHA-256 of its canonical JSON, recorded in X1's manifest.
- It returns the reranked top 10, followed by D3's remaining documents in D3's order,
  with no duplicates, to depth 100.
- It uses D4's exact settings: `cross-encoder/ms-marco-MiniLM-L6-v2` at revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`; input `(question, title. text)`; maximum
  length 512; batch 32; ranking by logit with ties broken by `doc_id`.
- Nothing is tuned: not the pool size, the model, any weight or any other setting.

**Carry rule.** The primary comparison is overall exact-variant Joint Recall@10 on `tune`
(n = 500).

- **X1 − D3 > 0.02:** D3 falls outside the existing plateau, and X1 is carried forward as
  a post hoc challenger.
- **A gain of at least +0.03** also clears the project's practical-importance threshold.
- **A gain between 0 and 0.02** is reported, but does not justify carrying the extra
  cross-encoder.

**Reported.**

- Overall, bridge and comparison Joint Recall@10; Recall@3; bridge hop-1 and hop-2
  Recall@10.
- Paired X1 − D3 intervals, and X1 compared with D4.
- Graph-added gold paragraphs that are retained, promoted or removed.
- Reranking latency: on the GPU for all 500 questions, and on the CPU for the same 20
  questions used for D4.

**Status.**

- **D3 remains the frozen preregistered winner R.** X1 is a post hoc challenger. The
  Phase 2b hypotheses and selection table are not rewritten.
- **If X1 clears the carry gate:**
  - it gets one parallel `dev` run with the identical K = 10 answer interface, and the
    completed D3 evaluation is kept;
  - D3 and X1 both go into later retrieval evaluations;
  - every X1-backed result is labelled post hoc and exploratory, and X1 stays outside the
    confirmatory Holm family;
  - if X1 reaches the blind test, its configuration is frozen before any blind split is
    opened, and its history stays "candidate added after inspecting Phase 2b `tune`
    results".

## A. Amendment log

| Id | Date | Phase | Change | Reason | Effect |
|---|---|---|---|---|---|
| A1 | 2026-09-29 | 1 | Answer and closed-book prompts, and the output schema, move to version 2; the prompt-format criteria are expanded by a malformed-answer diagnostic; version 2 is checked once on the same 20 `eda` questions | Violations of the output interface found diagnostically in the version-1 check; the preregistered trigger did not fire (details below) | Every system sharing the answer prompt uses v2; about 20 more generations (Phase 1: ~560 pipeline evaluations, ~540 unique generations) |
| A2 | 2026-09-29 | 2a | Alias-frequency filter extended from single-token aliases to aliases of every length (> 100 paragraphs, counted independently per alias) | Corpus-only structural audit of the graph, before any Phase 2 `tune` result: multi-token generic aliases dominated the graph | 83 aliases (29% of links) removed before the first `tune` run; tokenization, directions, budgets and selection unchanged |
| A3 | 2026-09-30 | 3 | `assess_v1` → `assess_v2`: the `next_query` instruction states the 12-word limit | The `assess_v1` format check failed 7/40 (limit 4); all 7 were queries over the frozen 12-token limit | One format-only revision, rerun once on the same 40 cases; no second revision |
| A4 | 2026-10-01 | 4 | `verifier_v1` → `verifier_v2`: format instructions only; the format-check viability gate counts the verifier recheck, as §4.6 specifies | The Phase 4 format check failed the verifier's per-role gate (8/34; limit 10%), all format or grounding errors; the implemented viability check wrongly ignored the recheck | One format-only revision of the verifier; the whole check rerun once (`ma_check_v2`); no second revision |
| A4.1 | 2026-10-01 | 4 | Verifier interface repair: aggregate `sufficient` computed in code; references on unsupported requirements ignored and logged; corrective fields cleared when computed sufficient | `ma_check_v2` still failed the verifier gate (6/37), mostly through redundant or derived fields; verifier errors skip corrective search, which would bias H4-4 | Gate re-evaluated by an offline replay of the 37 recorded verifier outputs; both original results preserved |
| A5 | 2026-10-01 | 4 | Behavioural calibration on two fresh `eda` subsets (C1, C2): stage diagnostics S1–S4 with hand-corrected reference requirements; at most one revision (A6), kept only if it validates on C2 | The verifier judged all 17 valid initial verdicts in `ma_check_v2` not sufficient; the cause (planning, selection or verifier) is undiagnosed | About 300–450 `eda` calls; `dev` and the blind splits untouched; the chosen version is frozen before `dev` |
| A6 | 2026-10-01 | 4 | `verifier_v2` → `verifier_v3`: question first, labelled requirements with an explicit count, a fixed order of work, a final checklist, no outside knowledge, lowest-numbered unsupported correction, two invented examples | C1 diagnostics: S3 0.30 (target 0.70) is the largest shortfall; 13/40 probe hard failures | The verifier prompt only; validated on C2 against v2 with the A5 keep rule plus a no-higher hard-failure-rate condition |
| A7 | 2026-10-04 | 4b | Infrastructure failures abort the run instead of counting as model hard failures; a legacy-runtime exception for comparisons with pre-4b `dev` runs; the G0 duplicate check made deterministic | Counting an unreachable server as model failures would silently fail open or unsupported; legacy runs never recorded the Ollama version; G0 must not require model misbehaviour | Recorded before G0 and any Phase 4b LLM call |
| A8 | 2026-10-04 | 4b | Calibration G1 gates the changed roles only (researchers, verifier, synthesizer schema); the unchanged planner becomes a calibration diagnostic and stays in R4 on `dev`. `researcher_v2` → `researcher_v3`, the one permitted prompt-only researcher revision | C3: G1 failed through one failure of the unchanged planner (a 2% gate over 30 calls allows none); G2 failed (42/45) through selection, with researchers keeping about 2.7 of 10 candidates | C3 results kept as recorded; C4 validates `researcher_v3`; the verifier is not revised |

### A1 — version 2 of the answer interface (2026-09-29, before any `dev` call)

**What the original check found.** The preregistered prompt-format check (§1.4) ran on the
20 `eda` questions with version 1 and produced **1/20 failures** (one mis-labelled
citation), below the revision threshold. The preregistered trigger therefore **did not
fire**. Inspecting the outputs diagnostically found four violations of the output
interface that the version-1 criteria did not cover:

- **two empty answers**, both with `supported = false`, despite the instruction to give a
  best guess;
- **two serialization-like answers**, `math:2001-2016` and `strconv(1887)`, next to
  sensible rationales.

These are defects of the interface, independent of whether an answer is correct, which is
why the change is made. The version-1 result stays on record
(`results/eda_format/format_check.jsonl` with its manifest).

**Version-1 hashes** (preserved; recomputed by `prompts.v1_hashes_now()`):

| Artifact | SHA-256 |
|---|---|
| `answer_v1` | `3f490daca95bb51c060f46ab8e8744a3f1e91913253ea371b6732195f647a665` |
| `closed_book_v1` | `90e09ebc07c35ec55d10f599e41ae7a9808b7dce6cc76507a4c408d57d43cb9a` |
| schema v1 | `b289ec00a9f3b73e012daa2e35c3c385faf24ea313c80abe2686665d8a6974fb` |
| system prompt | `16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172` |
| `render_v1` | `935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6` |

**Version-2 changes.**

- In both `answer_v2` and `closed_book_v2`, the `"answer"` rule becomes:

  ```text
  - "answer": the shortest answer that fully answers the question: a name, date, number,
    short phrase, or "yes"/"no". Not a full sentence. Write it as ordinary plain text: never
    leave it empty, and never encode it as a function call, a type conversion or a labelled
    expression. If you are unsure, still give your best guess.
  ```

  Every other line of both prompts is unchanged from version 1.
- **Schema v2**: `answer` has `minLength: 1`, which Ollama enforces while decoding, and a
  validator rejects an answer without a non-whitespace character. A rejected output is a
  schema error, scored as in §1.4. A regex `pattern` cannot be used, because Ollama
  rejects schemas containing one (checked: HTTP 400). Checking on a throwaway prompt also
  showed that `minLength` alone lets the model return a single space.
- The system prompt and the paragraph renderer are unchanged.

**Malformed-answer diagnostic** (`scoring.answer_format_flags`; deterministic; it only
labels answers — it never rewrites an answer or changes EM or F1):

- `empty`: the answer has no non-whitespace character;
- `encoded`: the whole stripped answer matches `[A-Za-z_][A-Za-z0-9_.]*\(.*\)` (a function
  call with no space before the parenthesis), or the answer starts with
  `[a-z_][a-z0-9_]*:\S` (a lowercase identifier joined to a colon with no space after);
- `long`: more than 10 whitespace-separated words.

The flags are recorded for every output of every system from now on and reported as
their own counts.

**Expanded format criteria.** An output fails if it fails the schema, has a mis-labelled
citation (§1.4), or has any of the three flags.

**Procedure.**

- Version 2 is run **exactly once** on the same 20 `eda` questions. The code refuses a
  second run.
- If no more than 2 of the 20 fail the expanded criteria, the version-2 hashes are
  recorded below and frozen.
- If more than 2 fail, work stops for review. There is no further tuning on these
  questions.

**Result of the version-2 check** (run once, 2026-09-29): **2/20 failures** under the
expanded criteria, so version 2 is frozen. The failures:

- `math.min(2001, 2016)`: flagged `encoded`;
- the same mis-labelled citation as in version 1 (a sentence number that does not exist
  in *Innocence (Björk song)*).

There were no empty answers and no schema errors. Two observations are recorded without
changing anything, as the procedure requires:

- `,__1919__` is malformed but matches neither `encoded` pattern. The diagnostic
  undercounts serialization-like answers, and its counts are a lower bound. Scoring is
  unaffected; the official normalization strips punctuation, so this answer scores as
  correct.
- With empty answers ruled out, low-confidence outputs contain arbitrary guesses (for
  example "Yes" for a "what year" question). They score 0 as before. `supported` and the
  rationale remain the signal of abstention.

**Version-2 hashes (frozen):**

| Artifact | SHA-256 |
|---|---|
| `answer_v2` | `0a2aa59c597d9730ed73ea358b22cd76787cde9910e12888397076b7bb003f72` |
| `closed_book_v2` | `a89deeb1d5aafd0fccafbcb44a10e47067f610bef75dcf8bcbfa878e9cdbcb79` |
| schema v2 | `c29af85a08782e161cae0ab7e4eb847e19866b4bc7910d84640c30d12ab67847` |
| answer validator v2 | `252c82c0f31ab1013af85beeaa777b8eab0a5ffa2ba4dadec1e9961eb2d4910c` |
| system prompt | `16f6b152986937eb921b888a7ece5481fc3bbab534b0e7ac8cf883ad63bf9172` |
| `render_v1` | `935e0b46eb31926412702ca36a731746a819dfb1c51682b68176ac6688686fc6` |

### A2 — alias-frequency filter for aliases of every length (2026-09-29, before any Phase 2 `tune` result)

**How the defect was found.** After implementing the graph exactly as preregistered, a
structural audit of the built graph, using corpus text only and no labels or results,
showed that the frequency filter of §2.3 applied only to single-token aliases.

- **The failure mode.** Removing a disambiguator often turns the title of a work into an
  everyday phrase, and case-folding lets that phrase match ordinary text:

  | Alias | Title | Paragraphs matched |
  |---|---|---|
  | "the first" | *The First (album)* | 7,470 |
  | "new york" | *New York (Glee)* | 2,242 |
  | "the world" | *The World (film)* | 2,155 |
  | "the american" | *The American (2010 film)* | 2,067 |
  | "high school" | *High School (1968 film)* | 1,054 |
  | "to play" | *To (play)* (a full title, whose parentheses vanish under tokenization) | 436 |

- **Pre-amendment statistics (preserved).** 73,388 permitted aliases; 138,113 edges.
  **83 aliases** matched more than 100 paragraphs (50 of them shortened titles), and they
  accounted for **29% of all alias matches**.
- **Why it matters.** Forward candidates are taken in mention order, so these false links
  displace real ones.

No Phase 2 `tune` result had been produced or inspected when the defect was found.

**Correction.** The frequency filter now applies to aliases of **every** length, full and
shortened alike:

1. After the preregistered filters (length, stop words, digits, single-token frequency)
   and the ambiguity rule, count for every surviving alias the **distinct corpus
   paragraphs** containing its complete token sequence.
   - Sentences are tokenized with `tok_v1`, and a match must lie within one sentence, as
     in graph matching.
   - A paragraph counts once, however often it contains the alias.
   - Each alias is counted **independently**, with no longest-match suppression, so a
     common phrase cannot hide inside a longer alias and reappear after filtering.
   - All corpus paragraphs are counted, including the title's own.
2. Drop every alias found in **more than 100** paragraphs. This is the preregistered
   threshold, not chosen by experiment. For single-token aliases it coincides with the
   existing rule.
3. Build the graph once from the remaining aliases, with the frozen greedy, longest-first
   matching.

**Unchanged:** tokenization, graph directions, expansion budgets and caps, candidate set
and selection rules.

**Known cost.** Some legitimate but generic hubs are removed as well: for example "United
Kingdom", "Hong Kong", "World War I", "Billboard 200" and "Hip hop". Graph coverage of
the gold hops stays a reported diagnostic. It is not a tuning criterion for the alias
rules.

### A3 — `assess_v2`, the one permitted format-only revision (2026-09-30, before any `dev` call)

**The `assess_v1` check** (§3.3; commit `e3ea8f3`; results kept as
`results/eda_format/assess_check_v1.jsonl`) produced **7 failures in 40**, above the limit
of 4.

- **Cause:** all 7 are validation errors from the frozen runtime contract: a `next_query`
  longer than 12 `tok_v1` tokens. The prompt asked for "a short search query" without
  stating the limit, and the model wrote question-length queries.
- **Viability gate:** passed; both branches occurred.
- **Reported, not a criterion:** `sufficient = true` in 6 of 20 full-context cases and in
  1 of 20 one-gold-removed cases.

**The revision.** Only the `next_query` instruction changes:

```diff
-- "next_query": if not sufficient, a short search query for the missing fact that names
-  the relevant entity found in the paragraphs; otherwise "".
+- "next_query": if not sufficient, a search query of at most 12 words that names the
+  relevant entity found in the paragraphs; otherwise "".
```

The v2 instruction:

```text
- "next_query": if not sufficient, a search query of at most 12 words that names the
  relevant entity found in the paragraphs; otherwise "".
```

**Unchanged:** the schema (`AssessOutput`), the runtime contract and its 12-token limit,
the search policy, and every other instruction. `assess_v1` and its results are
preserved.

**Procedure.**

- `assess_v2` is run **once**, from a clean commit, on the same 40 cases (the same 20
  `eda` questions and two conditions). Results go to `assess_check_v2`.
- If more than 4 fail, or the viability gate fails, work stops for review. **There is no
  second revision.**
- Otherwise the `assess_v2` hash is frozen here, and Phase 3 continues with `dev_smoke`.
- The cautious stopping behaviour is **not** adjusted. It stays a preregistered finding
  (H3-5, H3-6).

**Result of the `assess_v2` check** (run once, 2026-09-30, clean commit `7a870a3`): **2
failures in 40**, both on the same question (`5abdcce855429965af743e57`, both conditions),
each a `next_query` over 12 tokens.

- **Viability gate:** passed.
- **Reported, not a criterion:** `sufficient = true` in 7 of 20 full-context cases and in
  0 of 20 one-gold-removed cases. The cautious stopping behaviour is kept as a finding.

`assess_v2` is frozen.

**Frozen Phase 3 hashes (SHA-256):**

| Artifact | Hash |
|---|---|
| `assess_v2` | `2221feb7f911d1cdc90150f930fab21abf0eddd63cf5fdf05d5ebc2048a17724` |
| `AssessOutput` schema (unchanged from v1) | `c84e600907056092f6a7bc0ed85c0ff78790e5b76c72f6d8676456f7cc849915` |
| Runtime contract (unchanged from v1) | `db8c967d1390891200e1e2afe2dd8da4536993b2f185bb7d59803f5d459a4a95` |
| Agent graph source (`agent.py`) | `5e7b844c2416450f5903b31805b655c5b4e98fdf8dfd82c1e5619a8f562a7cdd` |
| Transition rules (`route_after_search`, `decide_after_assess`, `route_after_assess`) | `f497262e825cded804fe192b101afee6ad077a1fe96adf14cb1655a67064a9df` |
| `answer_v2`, schema v2, validator, system prompt, `render_v1` | unchanged; see A1 |

### Phase 3 record — K_match (2026-09-30)

**K_match = 16.** It is derived once from the agent's `dev` run (commit `37a0466`), as the
mean number of paragraphs displayed to its `answer` call, 15.95, rounded half up. It is
reused unchanged for the matched-context control on `test_agents`.

### A4 — `verifier_v2`, a format-only repair (2026-10-01, before any `dev` call)

**The `ma_check_v1` result** (clean commit `b847ac4`, 20 `eda` flows, saved locally):

| Role | Calls | Failures | Within 10%? |
|---|---|---|---|
| Planner | 20 | 1 | Yes |
| Researchers | 53 | 4 | Yes |
| Verifier (with the recheck) | 34 | 8 | **No** |

The verifier's failures:

| Failure | Count |
|---|---|
| A reference to paragraph 0 (paragraphs are labelled from `[P1]`) | 4 |
| The wrong number of requirement entries | 2 |
| References on an unsupported requirement | 1 |
| An empty corrective query | 1 |

**The viability gate,** as implemented, required a valid grounded `sufficient = true` from the *initial* verification only. §4.6 counts the verifier together with the recheck, and the recheck produced one such output. The implementation is corrected to match §4.6. All other viability conditions passed.

**Observations, not a reason for this revision:**

- All 14 valid initial verdicts were "not sufficient".
- Only 2 of those 14 verifications were shown both gold paragraphs. In the full-context
  flows, the researchers accepted both gold paragraphs in only 3 of 10.

Verifier caution is therefore not established as the main problem. Planning or evidence
selection may matter more. This is diagnosed separately (see A5) and not changed here.

**The revision.** Only two instructions of `verifier_v1` change:

```diff
-- "requirements": one entry per requirement, in order, each with "supported" (true or
-  false) and "references": the [paragraph, sentence] evidence for a supported requirement
-  as {"paragraph": P, "sentence": S}, or [] if it is not supported.
+- "requirements": exactly one entry per requirement, in order, each with "supported"
+  (true or false) and "references": the [paragraph, sentence] evidence for a supported
+  requirement as {"paragraph": P, "sentence": S}, using the labels shown (paragraph [P1]
+  is 1; sentences are numbered as shown), or [] if it is not supported (an unsupported
+  requirement has no references).
-- "corrective_query": if not sufficient, a search query of at most 12 words for that
-  requirement, naming any entity found in the paragraphs; otherwise "".
+- "corrective_query": if not sufficient, a search query of at most 12 words for that
+  requirement, naming any entity found in the paragraphs (never empty when not
+  sufficient); otherwise "".
```

The v2 `requirements` instruction:

```text
- "requirements": exactly one entry per requirement, in order, each with "supported"
  (true or false) and "references": the [paragraph, sentence] evidence for a supported
  requirement as {"paragraph": P, "sentence": S}, using the labels shown (paragraph [P1]
  is 1; sentences are numbered as shown), or [] if it is not supported (an unsupported
  requirement has no references).
```

The v2 `corrective_query` instruction:

```text
- "corrective_query": if not sufficient, a search query of at most 12 words for that
  requirement, naming any entity found in the paragraphs (never empty when not
  sufficient); otherwise "".
```

**Unchanged:** the schemas, contracts, routing, planner, researchers and every other
instruction. The criteria for judging sufficiency are untouched.

**Procedure.** The whole format check is rerun **once** on the same 20 flows
(`ma_check_v2`), from a clean commit. Planner and researcher calls with unchanged prompts
replay from the cache. If any role exceeds its 10% gate, or the corrected viability gate
fails, work stops for review; there is no second format revision.

### A4.1 — deterministic verifier interface repair (2026-10-01, before any `dev` call)

**The `ma_check_v2` result** (clean commit `d182c8c`; saved as
`results/eda_format/ma_check_v2.jsonl`):

| Role | Calls | Failures | Gate (10%) |
|---|---|---|---|
| Planner | 20 | 1 | Passes |
| Researchers | 56 | 5 | Passes |
| Verifier, with the recheck | 37 | 6 | **Fails (16%)** |

All viability conditions passed. Recovered from the cache, the six failed verifier calls
contain:

| Failure | Count |
|---|---|
| The wrong number of requirement entries | 2 |
| References attached to `supported = false` (one overlaps a wrong-count failure) | 3 |
| Every requirement supported, but `sufficient = false` | 1 |
| Unsupported requirements without a valid corrective target and query | 1 |

**Why change the interface rather than accept the errors.** A verifier error skips the
corrective search (§4.3). Leaving these failures in place would artificially lower the
corrective-search rate that H4-4 measures. Aggregate sufficiency and references on
unsupported requirements are derived or irrelevant fields. That control state belongs in
code, not in redundant LLM output.

**The repaired contract** (`verifier contract v2`; the prompt `verifier_v2` and the JSON
schema are unchanged):

1. **LLM judgments are kept.** Each per-requirement `supported` value, and the requested
   correction (`corrective_requirement`, `corrective_query`), are the LLM's judgments.
2. **`sufficient = all(supported)` is computed in code.** The LLM's own `sufficient`
   field is logged and ignored.
3. **References on `supported = false` are ignored and logged.** References are
   validated only for supported requirements.
4. **When the computed `sufficient` is true,** the corrective fields are cleared
   deterministically (and logged).
5. **Hard failures remain:**
   - the wrong number of requirement entries;
   - `supported = true` without at least one reference;
   - an invalid reference on a supported requirement (a paragraph not displayed, or a
     sentence that does not exist);
   - when the computed `sufficient` is false, a corrective requirement that is not an
     unsupported requirement, or a corrective query that is empty or over 12 `tok_v1`
     tokens.

**The gate, re-evaluated offline.**

- **Method:** the 37 recorded verifier outputs of `ma_check_v2` are re-validated under
  the repaired contract, from the cache only, against the paragraphs and requirements
  each call actually displayed. No LLM call is made, and every normalization is logged.
- **Expected:** 3 hard failures in 37 (8.1%), within the 10% gate.
- **Rule:** the gate passes only if the replay confirms at most 10%. Otherwise work stops
  for review.
- **Records:** both original results (`ma_check_v1`, `ma_check_v2`) are preserved; the
  replay is saved separately as `ma_check_v2_repaired`.

**Unchanged:** the prompts, schemas sent to the model, routing, planner and researchers.
The A5 calibration then runs with the repaired verifier contract.


**Replay result** (`python run_eval.py ma-check-replay`, clean commit `82f2885`; saved as
`results/eda_format/ma_check_v2_repaired.jsonl` with its manifest): **3 hard failures in
37 (8.1%). The gate passes.**

| Hard failure | Count |
|---|---|
| The wrong number of requirement entries | 2 |
| A corrective requirement that is not an unsupported requirement | 1 |

Three calls were normalized: two recheck outputs had one reference each ignored on an
unsupported requirement, and one verifier output had `sufficient` recomputed from false to
true, with its corrective fields cleared.

### A5 — behavioural calibration on fresh training data (2026-10-01, before any `dev` call)

**Principle.** A failed preliminary check on training data is grounds for diagnosis and
transparent iteration, provided that:

- the iteration is documented;
- any behavioural revision is validated on *fresh* data;
- `dev` and the blind splits play no part.

The cautious behaviour is **not** adjusted merely to make H4-4 pass. It may be improved
through this calibration cycle. The cycle runs after the deterministic interface repair
(A4.1) passes its replayed gate (it did: 3/37), and entirely before `dev`.

**Data.** Two fresh subsets of `eda` (train), disjoint from the 20 format-check questions
and from each other, taken in `eda` split order:

- **C1** (calibration): the next 16 bridge and 4 comparison questions;
- **C2** (validation): the 16 bridge and 4 comparison questions after those.

Within each subset, conditions alternate by index: (a) own ten paragraphs, (b) one gold
paragraph removed (§3.3 rule). The search stand-in is the same as in §4.6.

**Reference requirements.** For every C1 and C2 question, the planner's requirements
are checked by hand against the training labels (gold answer and supporting facts).
Incomplete or wrong requirements are corrected by hand, each correction with a one-line
justification. The corrected requirements are committed
(`calibration/requirements_c1.jsonl`, `calibration/requirements_c2.jsonl`) before any
verifier probe on that subset is run. The C2 file is written only after the A6 decision,
starting from the unrevised planner's C2 outputs. It is one label set, used for both
systems, even if A6 revises the planner.

**Stage diagnostics,** each computed on C1 with train gold labels, which are allowed on
`eda`:

| Stage | Measure | Target |
|---|---|---|
| S1 Planner | Share of the planner's own plans whose requirements, if satisfied, yield the gold answer. Labelled by hand per question as correct / incomplete / wrong target (the same labelling that produces the reference requirements) | ≥ 0.80 correct |
| S2 Researchers | Gold recall of the accepted set after the planned researchers, over the gold paragraphs present in the planned-stage pool, pooled over questions; precision reported alongside | ≥ 0.80 |
| S3 Verifier, complete evidence | An isolated verifier probe on every C1 question: its frozen reference requirements plus exactly the two gold paragraphs. The share judged sufficient | ≥ 0.70 |
| S4 Verifier, incomplete evidence | The same probe with one gold paragraph (the §3.3 removal rule). The share judged not sufficient | ≥ 0.70 |

The probes are separate verifier calls outside the flow (2 × 20), under the A4.1
contract; a probe with a hard failure counts against its target. Because they use the
reference requirements, they isolate the verifier from planning and selection errors.
S1 and S2 measure the system as it runs, with the planner's own requirements.

**Decision rule.**

- At most **one** behavioural revision, of the single component with the largest
  shortfall (target − observed) among S1–S4.
- If no stage falls short, nothing is revised.
- The revision is a prompt change to that component only. It is written, with its diff
  and rationale, as amendment A6 before C2 is run.

**Validation on C2.** The revised component is run through the same diagnostics on C2.

- **Kept if:** its targeted measure improves by at least **0.15** (absolute) over the
  unrevised component on C2, and no other stage measure drops by more than 0.15.
- **Otherwise:** the unrevised component is kept.

To make that comparison, the unrevised system is also run on C2, with cache replays where
prompts are unchanged. Either way, the chosen version is frozen, with its hashes recorded,
before any `dev` call.

**Cost:** about 120 flow calls + 40 probe calls on C1, the same on C2, and twice for the
unrevised and revised systems where they differ. Roughly 300–450 calls, 2–3 hours.

**Order.** C1 flow → C1 labelling and reference requirements (committed) → C1 probes →
diagnosis and, if any shortfall, A6 (committed) → C2 flow(s) → C2 labelling and reference
requirements (committed) → C2 probes → keep/revert decision → freeze. Every run needs the
operator's go-ahead before it starts.

### A6 — verifier_v3, the one behavioural revision (2026-10-01, before C2 and any `dev` call)

**The C1 diagnostics** (A5; flow `results/eda_calib/c1_flow_base.jsonl`, reference
requirements `calibration/requirements_c1.jsonl` at `8aa9707`, probes
`results/eda_calib/c1_probe_base.jsonl`; all from clean commits):

| Stage | Observed | Target | Shortfall |
|---|---|---|---|
| S1 Planner (hand labels) | 15/20 = 0.75 | ≥ 0.80 | 0.05 |
| S2 Researchers (pooled gold recall) | 27/30 = 0.90 (precision 0.55) | ≥ 0.80 | 0 |
| S3 Verifier, complete evidence | 6/20 = 0.30 | ≥ 0.70 | **0.40** |
| S4 Verifier, incomplete evidence | 14/20 = 0.70 | ≥ 0.70 | 0 |

S4 on the 19 probes whose remaining paragraph was checked by hand to be insufficient
(diagnostic only): 14/19 = 0.74.

**Selected component.** The largest shortfall is S3, so under the A5 decision rule the
verifier is revised. Nothing else is revised.

**Verifier hard failures under the A4.1 contract.**

- **C1 probes: 13/40** (8 complete-evidence, 5 incomplete-evidence probes):

  | Failure | Count |
  |---|---|
  | Corrective requirement that the output itself marks supported | 6 |
  | The wrong number of requirement entries | 3 |
  | Invalid reference on a supported requirement | 3 |
  | `supported = true` without a reference | 1 |

- **C1 flow:** 13 of 29 verifier and recheck calls (7 wrong entry count, 4 corrective
  requirement marked supported, 2 invalid references).

**Diagnosis.** S3 does not fail through caution alone. Three connected problems:

1. **Only the first requirement is judged properly.** 19 of the 40 parsed probe outputs
   are exactly "R1 supported, R2 unsupported", including 12 of the 14 complete-evidence
   probes that were not judged sufficient.
2. **Requirement order is confused with the paragraph-to-evidence mapping.** Requirement 1
   sometimes cites the sentence that supports requirement 2.
3. **The control fields contradict the output's own support decisions.**

Telling the verifier simply to be less cautious would be inappropriate, because S4 is
exactly at its target.

**The revision** (`verifier_v3`; the only change):

1. The question comes first.
2. Requirements are labelled `[R1]`, `[R2]`, ...
3. The exact count is stated: "There are N requirements; return exactly N entries."
4. Requirement and paragraph numbers are stated to be unrelated.
5. A fixed order of work: judge each requirement on its own against every paragraph,
   assign references, and only then fill the control fields.
6. The requirements are repeated after the paragraphs as a final checklist.
7. Outside knowledge is forbidden: a fact that no displayed sentence states is
   unsupported.
8. The corrective requirement is the lowest-numbered unsupported requirement, never a
   supported one.
9. Two short invented examples: both requirements supported by different paragraphs
   (requirement 1 by paragraph 2), and requirement 2 unsupported with correction target
   2. They show that paragraphs are numbered from 1 and sentences from 0. The invented
   names appear in no C1 or C2 question or paragraph (tested).

`verifier_v3`:

```text
Check whether the paragraphs support each requirement of the question. Judge only from
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
{checklist}
```

`{n}` is the number of requirements. `{requirements}` and `{checklist}` are the same
`[Ri]` lines, one per requirement.

**The diff from `verifier_v2`:**

```diff
--- verifier_v2
+++ verifier_v3
@@ -1,19 +1,46 @@
-Check whether the paragraphs below support every requirement of the question. Judge only
-from the paragraphs. Do not answer the question.
+Check whether the paragraphs support each requirement of the question. Judge only from
+the paragraphs and use no outside knowledge: a fact that no displayed sentence states is
+unsupported, even if you know it to be true. Do not answer the question.

-Requirements:
+Question: {question}
+
+There are {n} requirements; return exactly {n} entries in "requirements", one per
+requirement, in the order R1, R2, ...:
 {requirements}

-Fill the JSON fields in this order:
-- "requirements": exactly one entry per requirement, in order, each with "supported"
-  (true or false) and "references": the [paragraph, sentence] evidence for a supported
-  requirement as {"paragraph": P, "sentence": S}, using the labels shown (paragraph [P1]
-  is 1; sentences are numbered as shown), or [] if it is not supported (an unsupported
-  requirement has no references).
-- "sufficient": true only if every requirement is supported.
-- "corrective_requirement": if not sufficient, the number of one unsupported requirement;
-  otherwise 0.
-- "corrective_query": if not sufficient, a search query of at most 12 words for that
-  requirement, naming any entity found in the paragraphs (never empty when not
-  sufficient); otherwise "".
+Requirement numbers and paragraph numbers are unrelated: any paragraph may support any
+requirement, and one paragraph may support several requirements.
+
+Work in this order:
+1. Judge each requirement on its own, scanning every paragraph for sentences that state
+   it.
+2. If such sentences exist, the requirement is supported: "supported": true, and
+   "references" lists them as {"paragraph": P, "sentence": S}. Paragraph numbers start at
+   1 ([P1] is 1); sentence numbers start at 0, as shown in brackets. Otherwise
+   "supported": false and "references": [].
+3. Only then fill the control fields:
+   - "sufficient": true only if every requirement is supported;
+   - "corrective_requirement": if not sufficient, the lowest-numbered unsupported
+     requirement (never a supported one); otherwise 0;
+   - "corrective_query": if not sufficient, a search query of at most 12 words for that
+     requirement, naming any entity found in the paragraphs (never empty when not
+     sufficient); otherwise "".
+
+Example 1 (invented).
+Requirements: [R1] The river that flows through Velmora. [R2] The length of that river.
+Paragraphs:
+[P1] Ostrel River
+  [0] The Ostrel River is 412 kilometres long.
+[P2] Velmora
+  [0] Velmora is a town in northern Kasdia.
+  [1] The Ostrel River flows through Velmora.
+Output: {"requirements": [{"supported": true, "references": [{"paragraph": 2, "sentence": 1}]}, {"supported": true, "references": [{"paragraph": 1, "sentence": 0}]}], "sufficient": true, "corrective_requirement": 0, "corrective_query": ""}
+
+Example 2 (invented).
+Requirements: [R1] The architect of the Dunmere Library. [R2] The birth year of that architect.
+Paragraphs:
+[P1] Dunmere Library
+  [0] The Dunmere Library opened in 1931.
+  [1] It was designed by the architect Hollis Brandt.
+Output: {"requirements": [{"supported": true, "references": [{"paragraph": 1, "sentence": 1}]}, {"supported": false, "references": []}], "sufficient": false, "corrective_requirement": 2, "corrective_query": "Hollis Brandt architect birth year"}

@@ -22,2 +49,3 @@

-Question: {question}+Checklist: return exactly {n} entries, judging each requirement against every paragraph:
+{checklist}
```

**Unchanged:**

- the JSON schema sent to the model (`verifier_schema_v1`);
- the A4.1 contract (`verifier_contract_v2`). It still accepts any unsupported corrective
  requirement; v3 only asks for the lowest-numbered one;
- the model and its options;
- the planner, researchers and synthesizer;
- routing, budgets and the paragraph renderer.

`verifier_v1` and `verifier_v2` are preserved and pinned by tests. v3 adds about 400
prompt tokens to each verifier call; the existing overflow rule applies.

| Artifact | SHA-256 |
|---|---|
| `verifier_v1` | `d1166226548c13d26ce9ad52af4294e23b9a8848f352efd06eb9c57cbefef8eb` |
| `verifier_v2` | `dfa54a0f4e5430fe4d82319d1743e6e4fe32ba5a8ab3ff51112fa7b3b1e56c21` |
| `verifier_v3` | `c744d3282f06c30c7938025b18b0bc0c71a593640ac3471b46fa787723692eae` |
| `verifier_schema_v1` | `4a6e760e8b45672e84a86fd71c213e424545fa9d99b0657128bf605ba4af0e4a` |
| `verifier_contract_v2` | `7a95d66fb32f648d1203d3b70ea9795e6666fec1a092b34cabc0f481241ec097` |

**C2, as A5 specifies:**

1. Run the unrevised system (`verifier_v2`, tag `base`) and the revised system
   (`verifier_v3`) through the C2 flow. The v3 flow replays the cached planner and
   planned-researcher calls, which are unchanged, so S1 and S2 are identical by
   construction. Only the verifier, and what follows it, can differ.
2. Label the unrevised planner's C2 plans by hand and commit
   `calibration/requirements_c2.jsonl`.
3. Run the 40 C2 probes for each system.

**The keep rule** (A5, plus one condition fixed here, before any C2 result):

- v3 is kept only if:
  - S3 improves by at least 0.15 (absolute) over v2 on C2;
  - no other stage measure drops by more than 0.15;
  - **v3's verifier hard-failure rate on C2 is not higher than v2's**, pooled over the
    flow's verifier and recheck calls and the 40 probes.
- Otherwise v2 is kept.
- Each system's hard-failure rate is reported as a safety diagnostic, separately for the
  flow and the probes.

Either way, the chosen verifier is frozen, with its hashes, before any `dev` call.

### A6 result — the C2 decision (2026-10-04)

**Runs.** All from clean commits, run as locked pairs with v2 first:

- the C2 flows at `a0110d5`;
- the C2 reference requirements, committed at `fd0abb1`
  (sha256 `446fb18fb36e4e23d828ba9aef98bc6327fe8292e86e91b46b98740a7c8b1290`);
- the C2 probes at `fd0abb1`.

| C2 | `verifier_v2` | `verifier_v3` |
|---|---|---|
| S1 (hand labels; identical by construction) | 0.90 | 0.90 |
| S2 (pooled gold recall; identical by construction) | 23/30 = 0.77 | 23/30 = 0.77 |
| **S3, primary (all 20)** | 8/20 = **0.40** | 3/20 = **0.15** |
| **S4, primary (all 20)** | 12/20 = **0.60** | 6/20 = **0.30** |
| S3 diagnostic, without `5a847cc4` (gold answer inconsistent with the question) | 8/19 | 3/19 |
| S4 diagnostic, without `5adfc574` (remaining paragraph sufficient) | 12/19 | 6/19 |
| Probe hard failures | 16/40 | 29/40 |
| Flow hard failures (verifier and recheck calls) | 9/27 | 13/25 |
| **Pooled hard-failure rate** | 25/67 = **0.373** | 42/65 = **0.646** |

**The keep rule** (A5 and A6):

| Condition | Result |
|---|---|
| S3 improves by at least 0.15 | −0.25: **fails** |
| No other stage measure drops by more than 0.15 | S4 −0.30: **fails** |
| v3's pooled hard-failure rate is not higher than v2's | 0.646 against 0.373: **fails** |

**Decision: `verifier_v2` is kept.** v3 is retained in the code, with its hash, as the
record of the rejected revision.

**Why v3 failed.** 25 of its 29 probe failures return one entry for a two-requirement
plan, despite the stated count, the checklist and the examples. It did remove corrective
targets on supported requirements (v2: 4 on the probes; v3: 0).

**v2 also falls short on fresh data** (S3 0.40, S4 0.60, hard failures 40% on the probes).
The Phase 4 `dev` run measures the frozen v2 system as it is, and its analysis reports
this limitation alongside H4-4.

### Phase 4 record — freeze (2026-10-04, before any `dev` call)

The Phase 4 system runs with `verifier_v2` (`run_eval.PHASE4_VERIFIER`, recorded in every
Phase 4 manifest) and the A4.1 contract.

| Artifact | SHA-256 |
|---|---|
| `planner_v1` | `e6e453e16f09adb4a0d2923c55f21f0f819798b683da7e9dbce9262a86f0b787` |
| `researcher_v1` | `d9c25b49275969703649f8069a06b67eab3d4565edfce58f81c948dfe64af182` |
| `next_query_rule_with` | `0ba8f669319d4af0ee2aabec223ee97c2af9367c8d25bc81a2728b98cb5e2124` |
| `next_query_rule_none` | `27f512c88209cdcc733f137d55b43f13fef8153f3998b117d04ccfbb0c57804a` |
| `verifier_v2` | `dfa54a0f4e5430fe4d82319d1743e6e4fe32ba5a8ab3ff51112fa7b3b1e56c21` |
| `planner_schema_v1` | `53e7097fbb9ee9716b7c0963f6d0908ee68056ff6b90c99113e5f77440c2efc5` |
| `researcher_schema_v1` | `eb7b7514e5d053a184f5bbd4be90733cf59f8e889d710a37b391fce6ee8e3352` |
| `verifier_schema_v1` | `4a6e760e8b45672e84a86fd71c213e424545fa9d99b0657128bf605ba4af0e4a` |
| `planner_contract_v1` | `36b3f5f59ae939be2b306beec7d33a1d52c96c772066a8c9c743e0a26d236760` |
| `researcher_contract_v1` | `cab5565d9755ab25f2f6e73f1982e691958c9c46bc66350bef9473e88396bae2` |
| `verifier_contract_v2` (A4.1) | `7a95d66fb32f648d1203d3b70ea9795e6666fec1a092b34cabc0f481241ec097` |
| `render_labeled` | `8f6d7af4475a2278571897d3711e20800fea6b2dea84a6af317ca80354d9c937` |
| `multi_agent.py` source (graph and nodes) | `1c4b90a8bfa459357de435454c5b0a9d1cc1880c00ac726a38869f013c184d89` |
| `route_after_research` | `152dd1df259855f9f2ee8a1a3b887b37c1e94915eedf435334568088082e0ca3` |
| `route_after_verify` | `daef44a653a1187391e4affa5b64a7e1c6126014affa3aacb981f1a07656cf77` |

**Budgets:** at most 3 searches (10 + 5 + 5 paragraphs, search depth 100) and at most 7
LLM calls per question.

**Next** (§4.7): `dev_smoke`, one full `dev` run, the four cache-only replays, then the
exploratory comparisons.

**Later work.** Any later engineering iteration on the verifier is a separate,
separately preregistered phase on fresh training questions. It does not alter this frozen
system or its `dev` results.

### A7 — Phase 4b: infrastructure failures, the legacy runtime, and G0 (2026-10-04, before G0)

**1. Infrastructure failures.**

- **The rule:** infrastructure failures, including an unreachable Ollama server, abort
  the run. They are not counted as model contract failures.
- **Restarting:** completed calls stay cached, so a restart resumes without regenerating
  them.
- **Still hard failures:** invalid model output (invalid JSON or schema failure) and
  truncation (`done_reason = length`).
- **What this replaces:** the "transport error" item in the hard-failure definitions of
  §4b.3 and §4b.4.
- **Why:** counting an unreachable server as model failures would make every remaining
  call fail open (selection) or unsupported (verification). The run would then measure
  the infrastructure, not the system.
- **Test:** a transport exception raised during the flow propagates and aborts the run.

**2. The legacy-runtime exception.**

- **When it applies:** runs from before Phase 4b never recorded the Ollama version.
  Comparisons between a Phase 4b run and those legacy `dev` runs (or their cache-only
  replays) may use `unrecorded_runtime_ok=True`.
- **What gets reported:** the legacy runs' version is **unknown**. It is not inferred,
  from the installation date of the Ollama binary or otherwise.
- **Still refused:**
  - two recorded, differing versions;
  - the exception on a blind split (`test_*`). Every Phase 5 run records its runtime.

**3. The G0 duplicate check** (§4b.4a).

- **Gate conditions:**
  - duplicate normalization is checked deterministically: duplicate JSON passes Pydantic
    validation and `requirement_verdict` removes the repeats;
  - every live verifier output must be structurally valid and normalized.
- **Not a gate condition:** whether a live duplicate occurs. It is reported as the
  diagnostic `live_duplicate_observed`, so G0 never requires the model to misbehave.

### A8 — Phase 4b after C3: the G1 scope and `researcher_v3` (2026-10-04, before C4)

**C3 results, kept as recorded** (flow at `2a34bf7`; reference requirements at `020781e`,
sha256 `7c8f9fc9…`; probes at `020781e`; Ollama 0.32.5; n = 30):

| Gate | Raw count | Rate | Threshold | Result |
|---|---|---|---|---|
| G1 planner | 1/30 | 0.033 | ≤ 0.02 | Fails |
| G1 researchers | 0/59 | 0 | ≤ 0.02 | Passes |
| G1 verifier (flow, including recheck) | 0/59 | 0 | ≤ 0.02 | Passes |
| G1 verifier (probes) | 0/110 | 0 | ≤ 0.02 | Passes |
| G1 synthesizer schema errors | 0/30 | | 0 | Passes |
| G2 gold displayed at first verification ÷ gold in the pool | 42/45 | 0.933 | ≥ 0.95 | Fails |
| G3 complete probes judged sufficient | 21/30 | 0.70 | ≥ 0.70 | Passes |
| G4 incomplete probes judged not sufficient | 23/30 | 0.77 | ≥ 0.70 | Passes |
| G5 mechanics violations | 0 of 30 flows | | none | Passes |

**Under the original definition, G1 failed on C3.** That result stands in every report.

**1. The G1 scope on C3 and C4.**

- **Gated:** researchers and verifier at ≤ 0.02 each; synthesizer schema errors at 0.
- **Diagnostic:** planner hard failures are fully reported, but do not gate calibration.
- **Unchanged on `dev`:** the planner stays inside R4, the 100-question retention
  condition.
- **Why:**
  - **Scope:** Phase 4b did not modify `planner_v1`, and its fallback (the question as one
    requirement) worked.
  - **Discreteness:** a 2% threshold over 30 calls permits no failure at all.

**2. The one permitted researcher revision: `researcher_v2` → `researcher_v3`.**

- **The C3 losses:** three gold paragraphs were lost, all in selection and none to
  overflow. The researchers kept about 2.7 of 10 candidates.
- **The shared pattern:** the researcher stops retaining candidates once another
  paragraph appears sufficient.
- **The change:** one paragraph of the prompt. Schema, routing, aggregation and budgets
  are unchanged.

`researcher_v3`:

```text
You are selecting evidence for one requirement of a question. Do not answer the question.

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
{next_query_rule}
```

**The diff from `researcher_v2`:**

```diff
--- researcher_v2
+++ researcher_v3
@@ -13,5 +13,9 @@
 Decide for every candidate paragraph, [P1] to [P{n}], whether to keep it for this
-requirement. Keep a paragraph if it states the requirement's fact, or names or describes
-an entity the requirement refers to. When unsure, keep it: a paragraph kept by mistake
-costs little, but a paragraph dropped by mistake can make the question unanswerable.
+requirement. Judge every candidate independently: do not drop a candidate merely because
+the accepted evidence or another candidate already appears sufficient. Keep a paragraph if
+it states the requirement's fact, or names or describes an entity the requirement refers
+to. Keep every paragraph that could identify, verify, or provide an entity or fact this
+requirement needs, including a paragraph titled with such an entity and both sides of a
+comparison or a multi-hop chain. When unsure, keep it: a paragraph kept by mistake costs
+little, but a paragraph dropped by mistake can make the question unanswerable.

```

**3. The verifier is not revised.**

- G3 and G4 passed, and there were 0 contract failures in 169 verifier calls.
- Requirements were judged supported 45/55 times on complete evidence against 26/55 on
  incomplete evidence.
- A revision of a passing component is outside the §4b.8 rule.

**Order:** C3 is not rerun. C4, the fresh validation set, runs with `researcher_v3` and
`requirement_verifier_v1`, and all gates (G1 as amended here) must hold there.

### Phase 4b record — rejected at C4 (2026-10-06, before any Phase 4b `dev` call)

**C4, the validation set:**

- **Commits and runtime:**
  - the flow ran at `3e3061f` with `researcher_v3` and `requirement_verifier_v1`;
  - the reference requirements were committed at `2fd4fd8` (sha256 `7cb75abd…`);
  - the probes ran at `2fd4fd8`;
  - Ollama 0.32.5; n = 30.
- **The probes are diagnostic only.** They were run after the G2 failure and could not
  change the outcome.

| Gate | Raw count | Rate | Threshold | Result |
|---|---|---|---|---|
| G1 researchers | 0/59 | 0 | ≤ 0.02 | Passes |
| G1 verifier (flow) | 0/55 | 0 | ≤ 0.02 | Passes |
| G1 verifier (probes) | 0/116 | 0 | ≤ 0.02 | Passes |
| G1 synthesizer schema errors | 0/30 | | 0 | Passes |
| G1 planner (diagnostic under A8) | 1/30 | 0.033 | | G1 fails under the original definition |
| **G2** gold displayed at first verification ÷ gold in the pool | **38/45** | **0.844** | ≥ 0.95 | **Fails** |
| G3 complete probes judged sufficient (diagnostic) | 21/30 | 0.70 | ≥ 0.70 | Holds |
| G4 incomplete probes judged not sufficient (diagnostic) | 24/30 | 0.80 | ≥ 0.70 | Holds |
| G5 mechanics violations | 0 of 30 flows | | none | Passes |

**Diagnostics:**

- **Selection:**
  - pre-overflow acceptance 38/45, with 0 gold paragraphs removed by overflow;
  - precision 38/63;
  - 2.10 paragraphs accepted per question;
  - 2 questions with no accepted evidence.
- **Verifier:**
  - S4 without the flagged probe: 24/29;
  - requirements judged supported 46/58 times on complete evidence against 26/58 on
    incomplete evidence;
  - S1 = 24/30.

**Decision.** G2 failed on the validation set, so under §4b.8 **Phase 4b is rejected
before `dev`**.

- Phase 4b is not run on `dev`, and no further revision is made.
- The frozen Phase 4 system (`multi_agent.py`, `verifier_v2`) stays the multi-agent system
  in the Phase 5 confirmatory family.

**`researcher_v3` did not validate the intended retention improvement.** Retention on
fresh C4 was 0.844, lower than the 0.933 observed for `researcher_v2` on C3. C3 and C4
contain different questions, so this is not a paired comparison.

**What Phase 4b established:**

- **Structured outputs repaired contract reliability.** Across C3 and C4 the changed
  roles (researchers and verifier) had 0 hard failures in 458 calls, against 43/160
  verifier failures for Phase 4 on `dev`.
- **The per-requirement verifier met both behavioural targets on two fresh subsets.** It
  distinguished complete from incomplete evidence: requirements were judged supported
  91/113 times on complete evidence against 52/113 on incomplete evidence.
- **Prompt-based evidence selection by the 7B model remained unreliable.** The researchers
  kept about 2–3 of 10 candidates despite explicit keep-when-unsure instructions, and lost
  retrieved gold on both subsets.

### Phase 5 record — freeze declaration (2026-10-06, before any `test_*` access)

The approved hashes of the Phase 5 evaluation code, runner and retrievers. They were
reproduced independently at clean commit `fae9908` (351 tests passing; a seven-system,
cache-only dry run on `dev` reproducing every stored output). The runner refuses any real
stage, and every consumer refuses to analyse, unless the current hashes equal these.

```text
eval_code_hash: 2885d1e1381280c1222dc41d28843371c9394a915497ab990ad4aa926322eb7d
runner_source_hash: 535f05ad0f1fb7c3665e93d2a5fe99753ad7e318d0090a9b3b3fd36d3e60770e
bm25_describe_hash: 6c8b86493e57bcef48a4300505d605c4ad47fca07ba20aa9abc69da39f395606
d3_describe_hash: ba30b1d41ef49a8fec61a412d98eaf1010e05c79cd7cfe3d1c37a928134ced43
```

The commit that adds this declaration is the freeze commit. All Phase 5 stages run from it.
