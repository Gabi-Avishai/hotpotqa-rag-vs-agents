# Phase 2 report — the strongest non-agent retriever

**Status:** Phase 2 is complete.

- **Phase 2a** (classic): reviewed and approved. Champion C = C5@5.
- **Phase 2b** (dense and hybrid): the overall retriever is **R = D3**, frozen and run once
  end to end on `dev`.
- **Post hoc interaction check X1** (reranking D3): failed its carry gate.

Waiting for review before Phase 3.

## Phase 2a — classic retrieval

**Objective:** select the strongest classic, deterministic retriever on `tune`, retrieval
only, under `PREREGISTRATION.md` §2 and amendment A2.

**Hypotheses (preregistered, exploratory, `tune`):**

| Id | Prediction | Estimate [95% CI] | Result |
|---|---|---|---|
| H2a-1 | Best forward configuration (C4@5) − B: bridge hop-2 Recall@10 at least +0.05 | +0.237 [+0.191, +0.286] | Holds |
| H2a-2 | C5@5 recovers hop 1 when B's top 10 has only hop 2: at least 0.30 | 0.63 (17 of 27) | Holds |
| H2a-3 | Bidirectional beats forward-only on bridge Joint Recall@10 | s = 5: +0.040 [+0.020, +0.062]; s = 10: +0.045 [+0.022, +0.070] | Holds |
| H2a-4 | Backward additions are noisier for seeds with in-degree ≥ 100 | No seed has in-degree ≥ 100 after A2 (maximum 98) | Not testable; threshold not redefined |
| H2a-5 | Some classic configuration beats C0 by at least +0.03 overall | C5@5: +0.188 [+0.144, +0.234] | Holds |
| H2a-6 | Under C, named − unnamed first-hop Recall@10 at least +0.10 | +0.088 [+0.023, +0.155] (n = 258 / 134, unpaired) | Not shown: the gap is above zero but below +0.10 |
| H2a-7 | \|C1 − C0\| overall Joint Recall@10 < 0.01 | +0.000 | Holds |

### Files changed

All paths are under `projects/hotpot_qa/`.

- **Committed first:**
  - `7874013`: the preregistration for Phases 2a and 2b.
  - `4598f10`: amendment A2.
- **Committed in `ca6a587`:**

| File | Status | What it does |
|---|---|---|
| `normalize.py` | new | `norm_v1` (the six preregistered steps) and `tok_v1` |
| `retrieval.py` | edited | `BM25Config.normalize` and `title_repeats`, with defaults that keep C0 and its name unchanged. Helpers: full score vectors, per-term scores, indexed document tokens, ranking from a score vector |
| `feedback.py` | new | BM25-weighted RM3-style feedback (C3), with fallbacks and diagnostics |
| `graph.py` | new | Aliases, filters, the A2 frequency filter (counted independently), greedy longest-match edges, cached build with a fingerprint, and the expansion operator `G_{dir,s}(X)` |
| `llm.py` | edited | An offline client must be given the model digest from a manifest |
| `run_eval.py` | edited | `phase2a` (9 configurations, the two-step plateau selection, an assertion that C0 matches Phase 1) and `replay-baseline` (cache-only replay with abort conditions) |
| `tests/test_phase2a.py` | new | 46 tests (see below) |
| `tests/test_llm.py` | edited | The offline-digest requirement |
| `02_retrieval.ipynb` | new | Read-only Phase 2a analysis |

### Commands run

```bash
python -m pytest -q                        # 176 passed, before any Phase 2 tune result was opened
python run_eval.py phase2a --split tune    # 43 s; writes results/tune/phase2a_rankings.jsonl and phase2a_selection.json
python run_notebook.py 02_retrieval.ipynb  # 8 code cells, no errors or stderr

# bookkeeping corrections after review (no change to any result):
python -m pytest -q                        # 178 passed
python run_eval.py phase2a --split tune    # rerun: rankings, additions, feedback, scores, plateaus, selection
                                           # and graph fingerprint identical to the first run
python run_notebook.py 02_retrieval.ipynb  # rerun from scratch, no errors or stderr
```

### Tests and results

**178 passed, 0 failed.** The 46 new Phase 2a tests cover:

- `norm_v1`, step by step; `tok_v1`; normalization applied symmetrically to corpus and
  query ("Bjork" finds "Björk" under C1 and not under C0);
- title repetition, with C0 unchanged;
- feedback: a hand-computed example (weights and term tie-break), determinism, and both
  fallbacks;
- aliases: full and shortened; the length, stop-word, digit and single-token-frequency
  rejections;
- A2:
  - an alias found in 101 paragraphs is removed; one found in exactly 100 is kept;
  - repeated mentions in a paragraph count once;
  - the filter applies to full titles as well as shortened ones;
  - the independent count is 101 for "new york" even though every "new york city"
    mention hides it under greedy matching;
  - a rare useful link survives;
  - rebuilding is deterministic, independent of input order;
- matching: longest match first, no overlaps, no matches across sentences; self-edges
  and same-title variants; both variants reached; ambiguous shortened aliases dropped;
- the operator:
  - the forward cap and order; exactly 10 unique documents;
  - **a seed ranked 6–10 stays eligible when another seed links to it**, and **an
    unlinked seed ranked 6–10 gets no boost**;
  - retained documents are never candidates;
  - the backward cap, score order and zero-score exclusion on a 500-source hub;
  - **D3 retains and fills from H, not B**, and **D3 ranks backward candidates by B's
    scores**;
  - candidate functions take no gold information;
- the plateau rule; blind and `dev` splits refused for `phase2a`; the manifest carries
  every classic and graph setting and the graph fingerprint;
- the Phase 2a manifest configuration is deterministic and carries no timing; the selection
  file is written atomically with its own manifest;
- **the replay aborts on a single missing cache entry**, and **reproduces all 100 Phase 1
  `dev` top-10 lists** from the cache alone.

### Experiment configuration

| Item | Value |
|---|---|
| Graph | `graph_v1+A2`, fingerprint `c0a4b543a343a2bc…` |
| Aliases | 73,295 permitted. Rejected: 625 frequent single-token, 265 short, 22 digit-only, 6 stop-word-only, 1,255 ambiguous, 93 by A2 (all multi-token) |
| Edges | 98,027. 48,778 paragraphs have outgoing edges; 18,565 titles are linked to. Maximum in-degree 98, median 2 |
| Graph build | 21 s; about 760 MB peak |
| C3 | a: n = 3, m = 10; b: n = 5, m = 20; λ = 0.5 |
| C4/C5 | Retain 5, reserve 5; forward cap 3, backward cap 2 |
| Selection | Step 1 B = **C0** (C2 at +0.018 is inside the plateau). Step 2 plateau = {C5@5, C5@10} → **C = C5@5** |
| Count | 9 configurations in 6 families |

C0 reproduces all 500 Phase 1 `tune` rankings (depth 100) exactly; this is asserted in
the run.

### Metrics (`tune`, n = 500; bridge 400, comparison 100)

| Configuration | Joint Recall@10 | Bridge | Comparison | vs C0 [95% CI] | Hop-1 R@10 | Hop-2 R@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| C0 (B) | 0.608 | 0.550 | 0.84 | — | 0.903 | 0.620 | 0.5 ms |
| C1 | 0.608 | 0.550 | 0.84 | +0.000 | 0.906 | 0.620 | 0.6 ms |
| C2 | 0.626 | 0.562 | 0.88 | +0.018 [+0.002, +0.034] | 0.926 | 0.620 | 0.5 ms |
| C3a | 0.530 | 0.515 | 0.59 | −0.078 [−0.114, −0.042] | 0.852 | 0.602 | 2.3 ms |
| C3b | 0.566 | 0.545 | 0.65 | −0.042 [−0.074, −0.010] | 0.865 | 0.625 | 3.2 ms |
| C4@5 | 0.774 | 0.772 | 0.78 | +0.166 [+0.124, +0.210] | 0.880 | 0.857 | 0.7 ms |
| C4@10 | 0.762 | 0.765 | 0.75 | +0.154 [+0.110, +0.200] | 0.867 | 0.862 | 0.5 ms |
| **C5@5** | **0.796** | **0.812** | 0.73 | **+0.188 [+0.144, +0.234]** | 0.916 | 0.855 | 0.7 ms |
| C5@10 | 0.792 | 0.810 | 0.72 | +0.184 [+0.138, +0.230] | 0.913 | 0.855 | 0.8 ms |

- Hop recall is over the 392 bridge questions with a determined hop order; 8 are
  undetermined.
- Title-only Joint Recall equals exact-variant Joint Recall for every configuration to
  three decimals.

**Graph diagnostics:**

- **Coverage** (392 bridge questions): the forward edge hop 1 → hop 2 exists for **0.816**;
  the backward edge hop 2 → hop 1 for 0.046. Backward expansion mostly follows the hop 1 →
  hop 2 edge in reverse: from a retrieved hop 2, it finds hop 1 because hop 1 mentions it.
- **C5@5 additions:** 4.47 documents per query; 166 gold added and 35 gold displaced
  from C0's top 10, so +131 net. Precision of additions: forward 143/1,638 = 0.087;
  backward 23/598 = 0.038.
- **First-hop Recall@10, named vs unnamed:** C0 0.942 / 0.828; C5@5 0.946 / 0.858.
- **C1 is a genuine null.** Normalization changed 41,785 of 66,635 index texts and 107 of
  500 questions, and every depth-100 ranking. The top-10 set changed for 34 questions,
  but Joint Recall@10 for none.
- **Feedback drift:**
  - C3a lost 82 gold paragraphs and gained 29 (64 questions lost, 25 gained);
  - C3b lost 60 and gained 28;
  - no fallbacks.

### Representative success

`5a7119f75542994082a3e58e`: "When was the composer of *Persian Surgery Dervishes* born?"
*Terry Riley* is **not in BM25's top 100**. C5@5 adds it at rank 6 by a forward link from
the rank-1 gold paragraph. No single query with the question could have reached it.

### Representative failure

`5a745a6055429974ef308be4` (comparison): "Which … was completed first, *Something Wicked
This Way Comes* or *The Wind in the Willows*?"

- BM25 has both gold paragraphs, at ranks 6 and 10.
- C5@5's reserved slots are filled by forward links from generic single-word aliases:
  *Wicked (Future song)*, *Ugly (…)*, *Intro (…)*. They sit under the 100-paragraph
  threshold, so they survive the filters.
- Both gold paragraphs are pushed down, and one leaves the top 10.

This is the mechanism behind the comparison loss (0.84 → 0.73; 13 comparison questions
lose Joint Recall@10).

### Known limitations

- **The graph helps bridge questions and costs comparison questions.** The selection
  metric is overall, so C5@5 wins with that trade-off built in. Type-aware packing would
  be a new, post hoc candidate and is not proposed here.
- **Noise survives below the A2 threshold:** for example *The First (album)* via "the
  first album" (97 paragraphs), and single-word aliases such as "wicked" or "café". The
  alias rules are not tuned further.
- **H2a-4 is untestable.** After A2, the observed maximum in-degree was 98, below the
  preregistered 100.
- **Recall@20 and @50 of graph configurations** come from the base tail (the preregistered
  convention) and are not the graph's own deep ranking.
- **All results are on `tune`** (exploratory). *(Phase 2a checkpoint, before Phase 2b:
  the end-to-end effect on answers had not yet been measured. It is reported in the
  end-to-end section below.)*

### Departures from the approved plan

1. **Amendment A2** (approved before any result): the frequency filter applies to aliases
   of every length, counted independently. It removed 93 aliases (the pre-amendment count
   of 83 used greedy matching).
2. **H2a-4 is not testable** (above).
3. **The C4/C5 latency** includes the base BM25 search and, for C5, the score vector used
   for backward candidates.

### Suggested next step (Phase 2a checkpoint, before Phase 2b)

At the Phase 2a checkpoint, the recommendation was to proceed to Phase 2b as
preregistered:

- C = C5@5 is the incumbent.
- G\* = C5@5 (the best graph configuration), used for D3 = G\*(H).
- Phase 2b's chance to help lies in the remaining unnamed-first-hop gap (0.858 against
  0.946) and in comparison questions, which the graph weakened.

### Bookkeeping corrections (after review, no change to results)

- **The selection summary** (`phase2a_selection.json`) is written atomically and has its
  own manifest.
- **Graph build time** is no longer part of any manifest's `system_config`. It is kept in
  the selection file for reporting only. The same fix applies to the Phase 1 `retrieval`
  command's configuration.
- **Two regression tests** were added.
- **The rerun reproduces everything:** Phase 2a rankings, additions, feedback diagnostics,
  scores, plateaus, the selection (C = C5@5) and the graph fingerprint (`c0a4b543…`) are
  identical.

### Git diff / commit reference

Phase 2a was committed in `ca6a587` (report cleanup `fa8e294`).

## Phase 2b — dense and hybrid retrieval

**Objective:** test whether TAS-B dense retrieval, fusion, dense-seeded graph expansion or
cross-encoder reranking improves on C = C5@5 (`PREREGISTRATION.md` §2.7–2.8). Selection
is retrieval-only on `tune`, with the preference order D0 → D1 → D2 → D3 → D4.

### Selection (`tune`, n = 500) — preregistered, unchanged

| Candidate | Joint Recall@10 | Bridge | Comparison | vs D0 [95% CI] | Hop-1 R@10 | Hop-2 R@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| D0 = C5@5 | 0.796 | 0.812 | 0.73 | — | 0.916 | 0.855 | 1.2 ms |
| D1 = TAS-B | 0.568 | 0.478 | 0.93 | −0.228 [−0.280, −0.176] | 0.923 | 0.533 | 35 ms |
| D2 = RRF(C, D1) | 0.686 | 0.620 | 0.95 | −0.110 [−0.156, −0.064] | 0.939 | 0.663 | 36 ms |
| **D3 = G\*(RRF(B, D1))** | **0.826** | **0.818** | 0.86 | **+0.030 [+0.004, +0.056]** | 0.939 | 0.855 | 36 ms |
| D4 = rerank(D2 top 50) | 0.752 | 0.692 | 0.99 | −0.044 [−0.092, +0.002] | 0.944 | 0.742 | 269 ms (GPU) |

The plateau is {D3}, so **R = D3**. That is +0.218 over Phase 1's BM25 on `tune`.

### Hypotheses (preregistered, exploratory)

| Id | Prediction | Estimate [95% CI] | Result |
|---|---|---|---|
| H2b-1 | D1 − C, first-hop R@10 on unnamed first hops, at least +0.05 | −0.067 [−0.134, +0.000] (n = 134) | Contradicted |
| H2b-2 | D2 above both C and D1 | D2 0.686, C 0.796, D1 0.568 | Contradicted |
| H2b-3 | G\*(H) − G\*(B), bridge Joint Recall@10 > 0 | +0.005 [−0.022, +0.030] | Not shown |
| H2b-4 | D4 − D2: Recall@3 > 0 | +0.061 [+0.039, +0.083] | Holds |
| H2b-4 | D4 − D2: bridge hop-2 R@10 < 0 | +0.079 [+0.043, +0.115] | Contradicted (see X1) |
| H2b-5 | Best of D1–D4 − C, at least +0.03 | D3: +0.030 [+0.004, +0.056] | Not shown |

### Diagnostics

- **Overlap:** of 1,000 gold paragraphs in the top 10, 167 are found only by C5@5, 64
  only by TAS-B, 712 by both, and 57 by neither.
- **Dense seeds:** on average 1.9 per query are new (in H's top 5 but not BM25's top 5),
  and 77 of them are gold. D3's graph step reaches 6 bridge second hops that D0 misses,
  and loses 11 that D0 had.
- **Where D3's gain comes from:** mostly comparison questions (net +13 of 100); bridge is
  net +2 of 400. TAS-B restores comparison recall that graph expansion had cost (0.73 →
  0.86) without losing the bridge gains.
- **Cost:**
  - Encoding the corpus takes 10.7 minutes on the GPU, once. The float32 matrix is 205 MB
    (fingerprint `5411b3a2…`), with a peak of 1.50 GB of GPU memory.
  - Queries are encoded on the CPU, about 35 ms each.
  - The cross-encoder takes 230 ms per query on the GPU and 1.09 s on the CPU.

### Freeze and implementation checks

- **Freeze:** R = D3 was frozen in `0af5786` before its `dev` run.
- **Reproduction:** `hybrid.D3Retriever` reproduces all 500 stored `tune` D3 rankings
  exactly (depth 100). D0 inside Phase 2b reproduces the stored C5@5 rankings.
- **API adjustment:** `sentence-transformers` 6.1 reports pooling through its
  configuration (`pooling_mode: cls`). The CLS-pooling and no-normalization checks read
  it from there.

## End-to-end evaluation on `dev` (n = 100, K = 10)

R = D3 ran **once** with the unchanged answer interface. The Phase 1 BM25 K = 10 run was
replayed from the cache only: all 100 answers were cache hits, and the top-10 lists were
identical. Both manifests pass the final-comparison rule: same clean commit, identical
invariants, and the shared answer interface asserted equal.

| Metric | BM25 K = 10 | D3 K = 10 | D3 − BM25 [95% CI] |
|---|---|---|---|
| Answer F1 | 0.525 | **0.629** | **+0.104 [+0.020, +0.189]** |
| EM | 0.410 | 0.470 | +0.060 [−0.020, +0.140] |
| Context Joint Recall | 0.620 | 0.840 | +0.220 [+0.120, +0.320] |
| Evidence F1, strict (= official-compatible) | 0.327 | 0.453 | +0.126 [+0.042, +0.209] |
| Joint F1 (official) | 0.245 | 0.335 | +0.090 [+0.016, +0.165] |
| Supported rate | 0.750 | 0.870 | +0.120 [+0.030, +0.210] |
| All citations valid | 0.820 | 0.920 | +0.100 [+0.010, +0.190] |
| Incomplete-gold-evidence proxy | 0.710 | 0.640 | −0.070 [−0.170, +0.040] |

- **By type:** answer F1 on bridge questions rises by +0.136 [+0.034, +0.238] (n = 80);
  comparison questions change by −0.021 [−0.064, +0.000] (n = 20).
- **Cost:** 1 retrieval call and 1 LLM call per question in both systems. Retrieval p50
  is 34 ms against 0.6 ms; prompt tokens 1,752 against 1,716. LLM p50 is 23.3 s against
  20.6 s, from original-generation metadata; this is observed, not explained.
- **Context:** the answer-F1 gain (+0.104) is about half of the Phase 1 gap to the
  gold-context reader reference (+0.219).

**Bug found and fixed during this step.** The replay command did not pass its offline
client to the manifest writer. The replay manifest therefore lacked `base_llm` and
`num_ctx`, and the comparison check rejected the pair.

- **Fix:** `fb70fc7`, with a regression assertion that the replay manifest names the
  Phase 1 LLM.
- **Re-production:** both `dev` manifests were re-produced under the fixed code. The
  replay is cache-only again. The D3 records came from the cache of the single original D3
  run, with zero new generations, and are identical apart from retrieval timing.

## Post-hoc interaction check — X1 = cross-encoder reranking of D3

**Not a Phase 2b candidate.**

- **Definition:** recorded as `PREREGISTRATION.md` P1 (`80fb16d`) after the Phase 2b
  `tune` results and before X1 ran. X1 reranks exactly D3's first 50 with D4's settings.
- **Run:** clean commit `064ff67`. The manifest is marked `posthoc: true` and pins the D3
  ranking fingerprint.
- **Carry rule:** carry X1 forward only if X1 − D3 > 0.02 in overall Joint Recall@10.
- **Status:** the selection table and hypothesis outcomes above are unchanged, and
  **D3 remains R.**

| `tune` (n = 500) | D3 (R) | X1 (post hoc) | D4 |
|---|---|---|---|
| Joint Recall@10 | **0.826** | 0.750 | 0.752 |
| Bridge / comparison | 0.818 / 0.86 | 0.692 / 0.98 | 0.692 / 0.99 |
| Recall@3 | 0.675 | 0.753 | 0.754 |
| Hop-1 / hop-2 R@10 | 0.939 / 0.855 | 0.944 / 0.742 | 0.944 / 0.742 |

- **X1 − D3 (paired):**
  - Joint Recall@10: −0.076 [−0.120, −0.034]. **Not carried; no X1 `dev` run.**
  - Bridge: −0.125 [−0.177, −0.075]; comparison: +0.120 [+0.060, +0.190].
  - Recall@3: +0.078 [+0.057, +0.099].
  - Hop-2 R@10: −0.112 [−0.161, −0.064]; hop-1 R@10: +0.005 [−0.020, +0.031].
- **Graph-added gold paragraphs in D3's top 10 (157):** 82 removed from the top 10 (52%),
  54 promoted, 4 unchanged, 17 demoted but kept.
- **X1 against D4:** −0.002 [−0.012, +0.006]. Their ordered top 10s are identical for
  440 of 500 questions, share 9 documents for 57 and 8 for 3. The cross-encoder largely
  determines the final ranking whether D2 or D3 supplies the candidates. This also
  explains H2b-4: reranking looked harmless on D2 only because fusion had already diluted
  the graph's additions.
- **Latency:** 226 ms p50 on the GPU (all 500), 1.05 s on the CPU (the same 20 questions
  as D4).

**The architectural conclusion: the order of individually reasonable components
matters.**

```text
question-only fusion  → graph expansion           D3: keeps hidden-hop evidence
graph expansion       → question-only reranking   X1: often removes hidden-hop evidence
```

In this experiment, question–paragraph reranking improved direct relevance but removed
52% of the graph-added gold evidence, undoing much of the multi-hop gain. This applies to
this pointwise MS MARCO cross-encoder and this pipeline order. A multi-hop-aware
reranker, protected graph slots, or reranking before graph expansion could behave
differently; none of those was tested.

## Phase 2 conclusions

| Mechanism | Helped? | Evidence |
|---|---|---|
| Lexical normalization | No | +0.000 Joint Recall@10 (`tune`) |
| Title weighting | Marginally | +0.018, inside the plateau |
| Pseudo-relevance feedback | No, harmful | −0.042 / −0.078, query drift |
| Forward links | **Yes, most** | Hop-2 R@10 0.62 → 0.86; +0.166 overall |
| Backward links | Yes | +0.040 on bridge; recovers 17 of 27 missing first hops |
| Semantic retrieval (TAS-B) alone | No overall; yes for comparison | 0.568 overall, 0.93 on comparison |
| Fusion of final lists | No | 0.686: dilutes graph additions |
| Dense seeds for the graph (D3) | **Yes, modestly** | +0.030; restores comparison recall |
| Cross-encoder reranking | No for multi-hop; yes for direct relevance | Recall@3 +0.06–0.08; removes 52% of graph-added gold |

End to end on `dev`, the retrieval gains carry over to answers: F1 +0.104
[+0.020, +0.189], mostly on bridge questions. The Phase 3 agents start from **D3** as
their frozen search backend; X1 is not used.

### Known limitations (Phase 2b)

- **Single models:** conclusions are about TAS-B and one MS MARCO MiniLM cross-encoder.
- **D3's gain is small:** the `tune` gain is exactly +0.030, but its interval
  [+0.004, +0.056] does not establish a gain of at least 0.03. D3 was selected by the
  frozen plateau rule. The `dev` answer improvement is encouraging but exploratory
  (n = 100; comparison n = 20).
- **Deep recall:** the graph and reranker figures at Recall@20/50 come from the base
  tail, per the preregistered convention.
- **Exploratory status:** all results are exploratory; the blind splits remain unopened.

### Departures from the approved plan

1. **The replay-manifest bug** (fixed in `fb70fc7`); both `dev` manifests were
   re-produced from the cache, with no new generations.
2. **X1 was added post hoc** under protocol P1. It failed its gate and does not affect R.
3. **The cross-encoder's latency** is reported on the GPU (with Ollama unloaded) and on
   20 questions on the CPU.

## Git references

- Preregistration `7874013`; amendment A2 `4598f10`.
- Phase 2a `ca6a587`, `fa8e294`.
- Phase 2b freeze `0af5786`; replay fix `fb70fc7`.
- Post hoc protocol P1 `80fb16d`; X1 code `064ff67`.
- This report's final version is in the commit that adds the Phase 2b sections
  (`git log -1 -- projects/hotpot_qa/reports/phase_2.md`).
