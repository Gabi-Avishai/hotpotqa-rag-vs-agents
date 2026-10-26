# HotpotQA: from RAG to multi-agent investigation

> **When does iterative or multi-agent investigation recover enough extra evidence to
> justify its latency, compute and complexity over conventional RAG?**

Four systems answer the same multi-hop questions, all with one local model (`qwen2.5:7b`,
4-bit, through Ollama; no paid APIs):

- a BM25 RAG baseline;
- an improved non-agent RAG;
- a single investigation agent;
- a multi-agent investigation (planner, researchers, verifier, synthesizer).

System-selection rules, confirmatory hypotheses and final analyses were preregistered before their results were
opened. Amendments and post hoc analyses are explicitly labelled.
The final comparison ran once, on blind test splits, from a frozen commit.

> **Benchmark status.** This is a **custom closed-corpus benchmark** derived from the
> HotpotQA distractor validation contexts. Every system searches one shared corpus of
> 66,635 paragraphs, instead of each question's own ten candidate paragraphs. The
> official answer and supporting-fact metrics are reused, but **the scores are not
> comparable with the official HotpotQA leaderboard.**

## Results (frozen blind test)

**`test_agents`:** 100 questions (50 bridge + 50 comparison), weighted to validation
prevalence (0.799 / 0.201).

| | BM25 RAG | D3 RAG (R) | Single agent | Multi-agent |
|---|---|---|---|---|
| Answer F1 | 0.544 | 0.614 | 0.620 | 0.613 |
| Both gold paragraphs in the answer context (Joint Recall) | 0.572 | 0.808 | 0.932 | 0.596 |
| LLM calls per question | 1.0 | 1.0 | 2.75 | 6.13 |
| Median LLM time per question | 23 s | 22 s | 77 s | 108 s |

**Confirmatory family:** answer F1, with a paired bootstrap within type and Holm
correction over three adjacent comparisons.

| Comparison | Delta | 95% CI | Holm-adjusted p | Conclusion |
|---|---|---|---|---|
| R − BM25 | +0.070 | [−0.041, +0.181] | 0.66 | No clear difference demonstrated |
| Single agent − R | +0.006 | [−0.075, +0.090] | 1.00 | No clear difference demonstrated |
| Multi-agent − single agent | −0.007 | [−0.113, +0.096] | 1.00 | No clear difference demonstrated |

At n = 100 the intervals are wide. "No clear difference" is not evidence of equivalence.

**Secondary results** (outside the confirmatory family):

- **Retrieval** (`test_retrieval`, n = 1,000): D3 raises Joint Recall@10 from 0.592 to
  0.829, +0.237 [+0.207, +0.268]. Most of the gain is on bridge questions.
- **Answers** (`test_rag`, n = 200): R − BM25 answer F1 is +0.077 [+0.008, +0.144].
- **Multi-agent evidence loss:** weighted Joint Recall was 0.860 over the search pool and
  0.596 in the final answer context. Researcher selection discards retrieved evidence.
  This replicates the development finding.
- **Blinded manual audit** (96 outputs, descriptive):
  - among answers each system self-marked `supported = true`, 22–32% were labelled
    unsupported or contradicted;
  - point estimates were higher on bridge questions;
  - no difference by system or type was established.

## Recommendation

**Use D3 RAG.**

- It substantially improves the chance that both evidence paragraphs reach the model.
- It improved answer F1 on the larger secondary split.
- Its retrieval costs about 37 ms per question, against about 22 s for the LLM call.

**Neither agent demonstrated an answer-F1 improvement,** and they cost about **3.5×**
(single agent) and **4.9×** (multi-agent) R's median LLM time. Agents would need two
things this 7B model did not provide: a reliable sufficiency signal, to search again only
when evidence is missing, and evidence selection that cannot discard retrieved
paragraphs. These results apply to this model, these prompts, these budgets and this
benchmark.

## Research progression

Each phase ends with a completion report in `reports/`. Development work used `tune` (500
questions, retrieval only) and `dev` (100 questions). The test splits stayed blind until
the freeze.

| Phase | Question | Finding (development, unless marked test) | Report |
|---|---|---|---|
| 0 | Data and evaluation contract | Closed corpus, exact paragraph identities, seeded splits, blind guard; the official scorer reproduced exactly | `phase_0.md` |
| 1 | BM25 RAG baseline | BM25 top 10: F1 0.525 on `dev`; failures are mostly missing second-hop evidence | `phase_1.md` |
| 2 | The strongest non-agent retriever | A title-mention graph expands fused BM25 + TAS-B seeds (D3): Joint Recall@10 0.84 and F1 0.63 on `dev`. A question-only cross-encoder at the end removed graph-added evidence | `phase_2.md` |
| 3 | Single investigation agent | More evidence (Joint Recall 0.93), not better answers; the stopping decision is the weak component | `phase_3.md` |
| 4 | Multi-agent investigation | The answer context shrinks to 3.7 paragraphs, but selection loses gold evidence (Joint Recall 0.67); the verifier is unreliable | `phase_4.md` |
| 4b | Structural repair (rejected) | Schema-constrained, per-requirement outputs removed contract failures in the changed researcher and verifier roles (0/458; the unchanged planner is reported separately), but selection retention failed validation (38/45) | `phase_4.md` |
| 5 | Frozen blind test | The tables above | `phase_5.md` |

`PLAN.md` is the project plan. `PREREGISTRATION.md` governs every analysis decision,
amendment and freeze record (§A). Both, and the phase reports, are kept as the original
research records. Presentation-preparation materials were intentionally omitted from this
repository, so references to `reports/slides_draft.md` in those records point to a file
that is not included.

## Repository layout

| Path | Contents |
|---|---|
| `data.py`, `metrics.py`, `scoring.py`, `hops.py` | Data loading, corpus, splits and the blind guard; metrics and the official-compatible scorer |
| `retrieval.py`, `normalize.py`, `graph.py`, `dense.py`, `fusion.py`, `hybrid.py`, `rerank.py`, `feedback.py` | BM25, the title-mention graph, TAS-B, RRF fusion, the D3 retriever, the cross-encoder, pseudo-relevance feedback |
| `llm.py`, `prompts.py`, `rag.py` | The Ollama client (schema-constrained JSON, disk cache, context guard), frozen prompts, the RAG pipeline |
| `agent.py`, `multi_agent.py`, `multi_agent_4b.py` | The single agent, the multi-agent system and the rejected Phase 4b variant (LangGraph) |
| `manifest.py`, `run_eval.py`, `final.py` | Run manifests and comparability rules; every experiment command; the frozen, staged Phase 5 runner, analysis and audit |
| `00_eda.ipynb` … `04_multi_agent.ipynb` | Analysis notebooks, one per phase |
| `05_final_analysis.ipynb` | Figures over the frozen Phase 5 results |
| `calibration/` | Hand-labelled reference requirements for the agent calibration subsets |
| `tests/` | The test suite (no model calls) |

`results/`, `cache/` and `splits/` are generated locally and not tracked.

## Reproducing

**Environment:**

- Python 3.12 with the pinned top-level dependencies in `requirements.txt`
  (`pip install -r requirements.txt`). It is not a complete lockfile, so transitive
  versions may differ from the environment that produced the results;
- Ollama **0.32.5** with `qwen2.5:7b` (Q4_K_M, digest
  `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`);
- a 4 GB GPU is enough.

**Tests.** Tests make no model calls. On a clean machine, the first run downloads the
pinned HotpotQA dataset; subsequent runs can use the local dataset cache.

```bash
python -m pytest -q
```

**Development runs.** See the docstring at the top of `run_eval.py` for every command.
For example:

```bash
python run_eval.py rag --split dev --retriever d3 --k 10
python run_eval.py agent --split dev
python run_eval.py multi-agent --split dev
```

**The frozen test.** The reported run used freeze commit `cff0d20` of the original
development repository (this repository starts from a single snapshot). It runs one stage
per invocation.

```bash
python run_eval.py final --stage retrieval     # test_retrieval, no LLM
python run_eval.py final --stage rag           # test_rag
python run_eval.py final --stage agents        # test_agents
python run_eval.py final-analysis              # the preregistered analysis, from validated results only
python run_eval.py audit-sheet                 # the blinded audit sheet
python run_eval.py audit-analysis              # joins the hand labels with the hidden key
```

The runner refuses to start unless the evaluation code, the runner and the retrievers
match the declared freeze hashes, and the working tree is clean. Completed outputs are
immutable, and an interrupted stage resumes from its stored records.

- **`--dry-run`** runs the same code on `dev` from the cache, without opening any test
  split.
- **Reported numbers come from cached generations.** Temperature 0 and a fixed seed do not
  make fresh generations bit-identical, so a rerun replays the cache.

**What a clean clone can and cannot reproduce.** `results/` and `cache/` are not tracked.

- **Exact reproduction** of the reported run needs the original `cff0d20` artifacts and
  LLM cache, which are not part of this repository.
- **A clean public clone can:**
  - rebuild the corpus and splits from the pinned dataset (their hashes are checked);
  - regenerate the prerequisites, including re-selecting the retrievers on `tune`;
  - conduct a fresh frozen run from its own clean commit, which the runner records in a
    new freeze record, and then run the analysis and audit commands on it.
- **A fresh run may differ.** Fresh generations, even at temperature 0, are not guaranteed
  to match the recorded ones, so the new run's outputs and scores may differ from those
  reported above.

## Offline demo (cached replay)

```bash
python demo.py --list               # the five case studies
python demo.py --case graph_success # one case
python demo.py --all --verbose      # every case, with rationales and the full stored trace
```

- **What it shows:** the five case studies in `reports/case_studies.md`, replayed from
  stored results. Each case compares only the systems relevant to it.
- **No live calls:** it makes no model calls and needs no network. It loads no embedding
  model and rebuilds no index.
- **Fixed cases:** the cases, systems and source files are fixed in `demo_cases.toml`.
  Every pinned value is checked against the stored records before a case is shown.
- **Exact replay** needs the original local `results/` directory. Without it, the demo
  stops and lists the missing files.

## Limitations

- **Sample size:** the confirmatory split has n = 100. Differences of about 0.08 F1, the
  size seen on `test_rag`, cannot be resolved there.
- **One setting:** one local 7B model, one prompt set and fixed budgets. Stronger models
  or other prompts could change the agent results.
- **Proxies:** self-marked `supported = true` is the model's own report. Gold-paragraph
  coverage is a proxy for support. The audit is a small, single-labeller, descriptive
  check.
- **Interruptions:** two Ollama server stalls interrupted the final agents stage. It
  resumed safely from stored records (`reports/phase_5.md`).
- **Benchmark status:** a closed-corpus benchmark, not leaderboard-comparable (see above).
