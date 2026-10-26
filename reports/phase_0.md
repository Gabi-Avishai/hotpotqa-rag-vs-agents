# Phase 0 completion report — data and evaluation contract

**Phase:** 0, data and evaluation contract.

**Objective:** Pin the data. Build the shared corpus and deterministic splits with a blind
guard. Implement and test the metrics (cross-checked against the official scorer), the
local LLM client and run manifests. Understand the data in an EDA notebook with worked
evidence chains.

**Hypothesis:** None tested. Phase 0 builds infrastructure and describes the data; no
system is evaluated.

## Files changed

Base commit: `7e4e031` (plan). All paths are under `projects/hotpot_qa/`.

| File | Status | What it does |
|---|---|---|
| `data.py` | new | Loads the dataset at a pinned revision. Builds the corpus with `(title, text_hash)` identity. Draws deterministic splits and stores their expected hashes. Blind guard (`BlindSplitError`). `load_contexts` returns each question's own paragraphs, for EDA and oracle-style use only |
| `metrics.py` | new | Answer EM/F1 (official). Supporting-fact scoring in strict and official-compatible forms, official joint metric, citation validity. Recall@K and Joint Recall@K (exact variant and title-only). F1 on answered questions. Prevalence-weighted mean. Paired bootstrap (plain or stratified-weighted). Holm correction. Latency statistics |
| `llm.py` | new | Ollama client: raw ChatML prompts, JSON-schema outputs, token counting before generation, a disk cache for replay, usage accounting, and an offline replay mode |
| `manifest.py` | new | Run manifests with three sections, and the comparability check |
| `run_notebook.py` | adapted | FiQA's cell-by-cell notebook runner; only the usage example changed |
| `00_eda.ipynb` | new | EDA: question mix, 4 worked evidence chains, hop statistics, word-overlap analysis with one figure, corpus length and context budget |
| `tests/test_data.py`, `test_metrics.py`, `test_llm.py`, `test_manifest.py`, `conftest.py` | new | 87 tests |
| `pytest.ini` | new | Test paths; marker for the live Ollama tests |
| `.gitignore` | new | Ignores `splits/`, `cache/` and `results/` (local generated files; adds ignores only) |
| `PLAN.md` | edited | The two non-blocking review items: the definition of "F1 on answered questions", and the synthesizer reusing the frozen answer prompt |

## Commands run

```bash
# environment (pytorch_env)
pip install langgraph==1.2.12      # only new packages; nothing already installed changed
pip install pytest                 # was missing

# corpus, splits, pinned hashes (reads only ids and types for blind splits)
python -c "import data; data.make_splits(); data.verify_corpus()"

# official scorer, pinned by commit and SHA-256 (kept in the ignored cache/, not copied into the project)
curl -o cache/official/hotpot_evaluate_v1.py \
  https://raw.githubusercontent.com/hotpotqa/hotpot/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py

# tokenizer check against Ollama (4 prompts, raw mode + JSON schema)
# notebook authored cell by cell in a persistent kernel, then re-executed from scratch:
python run_notebook.py 00_eda.ipynb            # 7 code cells, 28 s, no stderr output
python -m pytest -q                            # 87 passed in 25 s (70 before the corrections below)
```

## Tests and results

**87 passed, 0 failed** (`python -m pytest -q`, including the 2 live Ollama tests).

| File | Tests | What they establish |
|---|---|---|
| `test_data.py` | 24 | Revision pinned. Corpus counts: 66,581 titles, 54 of them duplicated, 66,635 documents; fingerprint matches. Duplicate-title ids don't depend on reading order. The text hash includes the sentence segmentation. Splits are deterministic and match the pinned hashes. Sizes and type counts are exact. `tune`, `dev` and `test_retrieval` are disjoint; `dev_smoke` ⊂ `dev`, `test_rag` ⊂ `test_retrieval`, `test_agents` ⊂ `test_retrieval`. All three blind splits refuse `load_split` and `load_contexts`. On open splits, gold identities are exact, lie in the question's own context, and are in the corpus. Contexts agree with question identities. **Split-id validation**: every split's exact ids pass, in any order (ids only). Altered, missing, duplicated, appended-duplicate and extra ids are rejected under `dev`. Another split's ids are rejected under `test_agents`. Unknown names are rejected. `adhoc:` sets must be non-empty and duplicate-free |
| `test_metrics.py` | 28 | **Cross-check against the official scorer**: its `eval()` run on files built from 600 open-split questions (`tune` + `dev`) with synthetic predictions, compared on all 12 official means (tolerance 1e-12). Per-item agreement with the official `f1_score` / `exact_match_score` on 2,000 random answer pairs, including yes/no/noanswer. Strict vs official-compatible scoring on a duplicated title (wrong variant: official F1 1.0, strict 0.5). Citation validity, and **an output without citations is not valid** (`all_valid` 0, `valid_fraction` None). Coverage and per-citation validity are aggregated separately. **Exact-variant vs title-only retrieval** (wrong variant ranked first: exact Joint Recall@3 = 0, title-only = 1). Final-context vs cumulative Joint Recall. F1 on answered questions, including the "no answered outputs" case, and rejection of mismatched id sets. Hand-computed weighted means. Clear errors for empty input, a question without a stratum, and a stratum without a weight (previously dropped silently). **Bootstrap**: mismatched ids rejected, empty input rejected, a constant difference gives a zero-width interval, p = 2/2001 (never 0), antisymmetry, seed reproducibility, stratified-weighted resampling (Δ = 0.8 × 0.2 = 0.16). Holm. Latency, including an error for empty input |
| `test_llm.py` | 12 | Mocked transport: raw ChatML identical to Ollama's qwen2.5 template; schema sent as `format`; the **cache key changes with prompt, schema, options and model digest** (unit level), and the client misses the cache when the system prompt, schema, digest or seed change. Replay doesn't call Ollama and keeps the original wall time. **The context check raises before the request is sent.** A token-count mismatch raises and nothing is cached. A schema violation is recorded, not retried. Offline replay; missing model. **Live**: smoke test on `qwen2.5:7b` (digest `845dbda0ea48…`, Q4_K_M) plus replay; **silent-truncation demonstration**: sent directly with `num_ctx` 256, Ollama evaluated fewer tokens than the prompt had and raised no error, and the client's check raised `ContextOverflow` for the same prompt |
| `test_manifest.py` | 23 | Three sections present. The real `dev` split is validated, and the corpus is re-fingerprinted on every build. A different corpus fails the build. Wrong, missing and duplicated ids and unknown split names are rejected. `system_config` differences are allowed and reported; `run_metadata` is ignored. Each invariant change is rejected: corpus, split hash, evaluation-code hash, metric version, `num_ctx`, dataset, packages, base LLM. Different question sets are rejected. **Shared components** (e.g. the answer-prompt hash) must match and must be present. The commit counts only for final comparisons, which also reject dirty trees. Retrieval-only runs can't be mixed with LLM runs. Writing works; the real commit is read |

## Experiment configuration (what was pinned)

| Item | Value |
|---|---|
| Dataset | `hotpotqa/hotpot_qa`, `distractor`, revision `1908d6afbbead072334abe2965f91bd2709910ab` |
| Corpus fingerprint | `67db29c2a555974d0d16e8a3bb9c9865ee8c542d6d268ed88e0232e340fee3aa` (66,635 documents) |
| Split seed | 20260927 |
| Official scorer | `hotpot_evaluate_v1.py` @ `3635853403a8735609ee997664e1528f4480762a`, SHA-256 `d35fc91a…fecc02c0` |
| LLM | `qwen2.5:7b`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, Q4_K_M, 7.6B |
| Tokenizer | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`, `tokenizer.json` SHA-256 `c0382117…ce87539` |
| LLM options | temperature 0, seed 0, `num_ctx` 8192, `num_predict` 512; raw ChatML with an explicit system prompt |
| Metrics version | `METRICS_VERSION = "1"`; MPI 0.03 |
| Packages added | `langgraph` 1.2.12 (with `langchain-core`, `pydantic`, …), `pytest` |

**Split hashes** (SHA-256 of the sorted id list):

| Split | n | Bridge / comparison | Hash |
|---|---|---|---|
| `eda` (train) | 1,500 | — | `9e7e631a98f1c524…` |
| `tune` | 500 | 400 / 100 | `7d81b361d43676e6…` |
| `dev` | 100 | 80 / 20 | `90a6ad35c98b323a…` |
| `dev_smoke` ⊂ `dev` | 20 | 16 / 4 | `ec1c0dd9cf116f78…` |
| `test_retrieval` | 1,000 | 799 / 201 | `8654f35c1493a822…` |
| `test_rag` ⊂ `test_retrieval` | 200 | 160 / 40 | `57201688a97654e7…` |
| `test_agents` ⊂ `test_retrieval` | 100 | 50 / 50 | `6b72130cb1f09e7a…` |

`test_rag` and `test_agents` share 16 questions. Full hashes are in `data.EXPECTED_SPLIT_HASHES`.

**The tokenizer count equals Ollama's** `prompt_eval_count` on every call checked: 30 = 30
and 264 = 264 tokens, each twice, plus the live smoke test. The client raises on any
mismatch, so a later divergence would stop a run rather than pass silently.

### Blind splits were not opened

- `make_splits()` reads only the `id` and `type` columns. `load_split` and `load_contexts`
  raise for all three blind splits (tested).
- No code in the project passes `unblind=True`. Tests load only `tune`, `dev` and
  `dev_smoke`; the notebook loads only `eda` and `tune`.
- Exposure before the splits existed (2026-09-27): aggregate counts over all 7,405
  validation questions (type and level counts, context slots, unique and duplicated
  titles), plus the reviewer's aggregate check that each question's two gold titles occur
  in its own context. No individual validation example was printed, and no system was
  scored on validation. The model benchmark used train questions only.
- Corpus-level statistics in the notebook (paragraph lengths, template debris) use
  paragraph text only, which includes paragraphs belonging to blind questions, but no
  labels.

## Metrics with sample sizes (EDA, from `00_eda.ipynb`)

| Measurement | `eda` (train, n = 1,500) | `tune` (validation, n = 500) |
|---|---|---|
| Bridge share | 80.7% | 80.0% |
| Yes/no answers among comparison questions | 32.2% (of 289) | 30.0% (of 100) |
| Levels | easy 285, medium 960, hard 255 | hard 500 |
| Bridge: hop-1 title named in question | 67.3% (of 1,211) | 65.0% (of 400) |
| Bridge: **hop-2 title named in question** | **14.0%** | **8.0%** |
| Comparison: both titles named | 100% | 100% |
| Bridge: answer string in hop-2 paragraph | 82.9% | 82.0% |
| Bridge hop order decided by supporting sentences / full paragraph / question / answer / undetermined | 76.5 / 8.8 / 7.9 / 5.2 / 1.7% | 76.2 / 8.0 / 7.2 / 6.5 / 2.0% |
| Question-word overlap (stemmed, stop words removed): hop 1 / hop 2 / comparison entity / own distractors | 0.530 / 0.444 / 0.407 / ~0.38 | 0.507 / 0.438 / 0.418 / ~0.38 |
| Gold paragraph outranked by ≥ 3 of its own distractors: hop 1 / hop 2 / comparison entity | 18.6 / 33.3 / 28.7% | 20.5 / **35.5** / 25.0% |

Corpus (66,635 paragraphs, text only):

- Rendered paragraph length: median 133, mean 147, p95 288, maximum 2,321 Qwen tokens.
- Sentences per paragraph: median 4, p95 8, maximum 85.
- Template debris (`{'1': …}`): 215 paragraphs (0.3%). Empty paragraphs: 0.

## Representative successes

- **The official-scorer cross-check** agrees on all 12 official means over 600 open-split
  questions, across exact, partial, wrong, yes/no, punctuation and empty answers, and
  evidence sets with missing, extra and wrong facts.
- **Ollama's silent truncation is real, and the check stops it**: with a 256-token
  context, Ollama evaluated only part of a 300-number prompt and returned normally; the
  client raised before sending the same prompt.
- **A clean two-hop chain, *Naked Lunch* → *David Cronenberg*** (notebook §2): the name
  needed to retrieve the second hop appears only in the first hop's text.

## Representative failures (found during the phase, and fixed)

- **The live test caught a bug**: `/api/tags` needs GET, but the client always sent POST
  (HTTP 405). The mocked tests could not see it. Fixed: a request with no body is now
  sent as GET.
- **The first hop-ordering rule mis-ordered example 1** (*Promos* → *Chris
  Diamantopoulos*): neither paragraph's supporting sentences name the other. Replaced by
  a four-rule cascade; the rule that decided each case is recorded and reported.
- **The first answer-location check matched yes/no answers inside words** ("no" in
  "north"). Replaced by whole-token matching; yes/no answers are reported separately.
- **A count double-counted comparison questions**, and an empty-slice warning appeared for
  questions with no distractors. Both fixed before the final run.

## Known limitations

- Title-mention detection is a lower bound: paraphrases ("the 1991 film") are not
  detected. The bridge hop order is heuristic and undetermined for about 2% of questions.
- Word overlap is a proxy for BM25 (no term weighting, no length normalisation).
  Distractor-based outranking uses each question's own 8 distractors, not the 66k-paragraph
  corpus.
- **Label noise**: in example 1 the linking sentence is not marked as supporting, and the
  gold answer does not fit the question's "sports star" (Ryan Howard). Error analysis
  needs a label-error category.
- Cached replays report the original generation's wall time. Usage totals therefore
  describe generation cost, not replay time.
- The worst-case context is not yet handled by a rule: one paragraph has 2,321 tokens, so
  a large-K context can occasionally exceed the budget and raise `ContextOverflow`.
- Train questions (`eda`) mix difficulty levels; validation is all `hard`. Chain
  statistics differ only slightly between them.

## Departures from the approved plan

1. **The code commit is an invariant only for final comparisons.** Taken literally, a
   commit invariant would make development runs from different commits incomparable
   (for example a Phase 1 baseline against a Phase 3 agent). Development comparisons
   require an identical **evaluation-code hash** (`data.py` + `metrics.py`) and every
   other invariant. Final comparisons (`final=True`) also require the same clean commit.
2. **`dev_smoke` is a separately drawn stratified subset** of `dev` (16 bridge, 4
   comparison), not a literal "prefix", because `dev`'s stored order is grouped by type.
3. **Raw ChatML with an explicit system prompt** instead of Ollama's template with its
   default system prompt ("You are Qwen…"), so the exact prompt is ours, hashed and
   countable. The throughput benchmark used the template path; the change affects only a
   few prompt tokens.
4. **Added dependency: the Qwen2.5 tokenizer** (7 MB, pinned revision and hash), needed
   because Ollama has no token-counting endpoint.
5. **The official scorer is not copied into the project**: its repository has no licence
   file. The tests download it into the ignored `cache/` at a pinned commit and verify its
   SHA-256.
6. **Added `data.load_contexts`** (each question's own paragraphs) for EDA and future
   oracle-style checks; it is blind-guarded like `load_split`.
7. **`pytest` installed** into `pytorch_env`, where it was missing.
8. **`holm` implemented now** rather than at the comparison stage.

## Suggested improvements

- A **context-overflow rule for Phase 1**, preregistered before the first LLM run.
  Proposal: drop the lowest-ranked paragraphs until the prompt fits, and log each
  occurrence as a stop reason.
- A **label-error category** in the failure taxonomy.
- **Keep the corpus text unchanged** despite the template debris (0.3%). Cleaning it
  would change the benchmark for a negligible gain.
- **`PREREGISTRATION.md`** to be created with the Phase 1 proposal: the BM25 grid, the
  choice of K, the answer prompt and schema, the closed-book and oracle rows, and the
  overflow rule.

## Corrections (2026-09-28)

After the first review, these were changed. Tests went from 70 to 87; the notebook was
re-executed with no errors and unchanged results.

1. **Manifests verify their inputs instead of restating expectations.**
   - `manifest.build()` takes the corpus fingerprint from `data.verify_corpus()`, so it
     re-fingerprints the corpus and fails on a mismatch.
   - It checks the supplied ids with the new `data.validate_split_ids()`: no duplicates,
     the expected count, and the pinned hash for every recognised split.
   - Arbitrary id sets are accepted only under an explicit `adhoc:<name>`. A manifest for
     `dev` built from `["not-a-real-question"]` now raises `SplitMismatch`.
   - `data.load_split_ids()` uses the same check. It reads ids only, so no blind question
     is opened.
2. **Citation validity no longer rewards citing nothing.** `citation_validity()` now
   returns `has_citations`; `all_valid` is 0 when there are no citations, and
   `valid_fraction` is None. The new `aggregate_citations()` reports coverage (the share
   of outputs with citations) separately from pooled per-citation validity and from the
   all-valid rate over all outputs.
3. **Aggregation validates ids.** `answered_f1()` uses the same identical-id check as the
   paired bootstrap. `weighted_mean()` and `latency_stats()` raise on empty input.
   `weighted_mean()`, and with it the weighted bootstrap, raises when a question has no
   stratum or a stratum has no weight. Before, such questions were dropped silently.
   This last defect was found while making these changes.
4. **The evaluation-code hash** will be extended in Phase 1 to `run_eval.py` and every
   module that turns outputs into scores. Recorded in `PLAN.md` §8.

## Git diff / commit reference

**Commited at eb43562** The working tree contains the files listed above, on top of
`7e4e031` on branch `hotpot_qa`. Generated local files stay untracked and ignored:
`splits/splits_v1.json`, `cache/official/`, and the Hugging Face and Ollama caches.
