# Five case studies

**What these are.** Post hoc qualitative examples, selected after the results were
inspected. They illustrate mechanisms that the preregistered measurements quantify. They
are neither preregistered nor representative in a statistical sense.

**Sources.**

- Cases 1 and 3–5 come from the frozen blind test (`test_agents`, commit `cff0d20`).
- Case 2 comes from the cross-encoder experiment on `tune`, a retrieval-only split.
- Quoted sentences are the gold supporting facts.

| # | Mechanism | Split | Outcome |
|---|---|---|---|
| 1 | Graph expansion recovers the hidden hop | `test_agents` | D3 right, BM25 wrong |
| 2 | Question-only reranking removes it again | `tune` | Graph-added gold pushed out of the top 10 |
| 3 | An evidence-dependent agent search | `test_agents` | Single agent right, R wrong |
| 4 | Unnecessary search distracts | `test_agents` | R right, single agent wrong |
| 5 | Multi-agent selection loss | `test_agents` | R right, multi-agent wrong |

## 1. Graph success: following the link the first paragraph names

*"Folie à Deux is a studio album by the American rock band formed in which city?"*
The gold answer is **Wilmette**.

**Evidence:**

- *Folie à Deux (album):* "…the fourth studio album by American rock band **Fall Out
  Boy**."
- *Fall Out Boy:* "Fall Out Boy is an American rock band formed in **Wilmette,
  Illinois**, a suburb of Chicago, in 2001."

**What happened:**

- **BM25** retrieved only the album paragraph. The question never names Fall Out Boy, so
  BM25 could not find the second-hop paragraph. The model answered "Chicago", citing a
  distractor that places the band in Chicago.
- **D3** ranks the album paragraph first. Its graph expansion follows the forward title
  mention ("Fall Out Boy") from that top seed and places *Fall Out Boy* at rank 6. With
  both paragraphs in context, the model answered "Wilmette".

**Mechanism:** a bridge question hides its second hop. Its title appears only inside the
first paragraph. Following that link is cheap, about 37 ms per query, and deterministic.

**At scale:** D3 raised Joint Recall@10 from 0.592 to 0.829 on `test_retrieval`, with most
of the gain on bridge questions (0.526 → 0.806).

## 2. Reranking failure: a question-only reranker undoes the graph

*"Who is the author of the series in which Rupert Grint plays one of the three main
characters?"* The gold answer is **J. K. Rowling**. This is a `tune` question, from the
post hoc X1 cross-encoder experiment.

**Evidence:**

- *Rupert Grint:* "He rose to prominence playing **Ron Weasley**, one of the three main
  characters in the 'Harry Potter' film series."
- *Ron Weasley:* "Ronald Bilius 'Ron' Weasley is a fictional character in **J. K.
  Rowling's** 'Harry Potter' series."

**What happened:**

- **BM25 and the fused BM25 + TAS-B list** rank *Rupert Grint* first but never retrieve
  *Ron Weasley*.
- **D3's graph** follows "Ron Weasley" from the top seed and places it at rank 6, so both
  gold paragraphs are in the top 10.
- **The MS MARCO cross-encoder** then scores each paragraph against the question alone.
  *Ron Weasley* shares little vocabulary with the question, and falls to rank 12, outside
  the context. Generic cast lists and other franchises take its place.

**Mechanism:** a reranker that judges relevance to the question cannot see that a
paragraph is relevant through the *first* paragraph. It removes exactly the evidence the
graph exists to add.

**At scale:** on `tune`, reranking D3 removed 52% (82/157) of the graph-added gold
paragraphs from the top 10. This specific loss (both gold paragraphs in D3's top 10, and a
graph-added gold paragraph pushed out by the reranker) occurred in 81 bridge questions. Component order matters: fuse, then expand. In this
pipeline, the tested question-only reranker should not follow graph expansion.

## 3. Useful agent recovery: an evidence-dependent query

*"Sadok Sassi played for a national team that made its first World Cup in what year?"*
The gold answer is **1978**.

**Evidence:**

- *Sadok Sassi:* "He was a goalkeeper and played for Club Africain and the **Tunisian
  national team**."
- *Tunisia national football team:* "They have qualified for four FIFA World Cups, **the
  first one in 1978**…"

**What happened:**

- **R's ten paragraphs** held only the Sadok Sassi paragraph. The model inferred an answer
  from a qualification-round sentence: "1970", wrong.
- **The single agent** began from the same ten paragraphs. Its `assess` step judged them
  insufficient and wrote the query **"Tunisia first World Cup appearance year"**. "Tunisia"
  comes from the retrieved paragraph, not the question. That search added the *Tunisia
  national football team* paragraph, and the agent answered "1978".

**Mechanism:** iteration helps exactly when the first context lacks the second hop and the
retrieved text names it.

**At scale:** the single agent raised final-context Joint Recall to 0.932, against R's
0.808, on `test_agents`. Its answer F1 was not demonstrably better: +0.006 [−0.075,
+0.090].

## 4. Unnecessary search: more context, a worse answer

*"St Dionis Backchurch was rebuilt after a major conflagration that consumed how many
parish churches?"* The gold answer is **87**.

**Evidence:**

- *St Dionis Backchurch:* "…it was rebuilt after the **Great Fire of London**…"
- *Great Fire of London:* "It consumed 13,200 houses, **87 parish churches**, St Paul's
  Cathedral…"

**What happened:**

- **R** had both gold paragraphs at ranks 1 and 4, and answered "87".
- **The single agent** started from the same context, but `assess` judged it insufficient.
  Two more searches ("churches destroyed in great fire of london", "number of parish
  churches destroyed…") added ten more paragraphs about London churches.
- One of them states, "The office of Christopher Wren rebuilt **51** parish churches." The
  agent answered "51", citing it.

**Mechanism:** the stopping decision is the agent's weak component. Extra paragraphs that
are plausible but off-target can displace a correct answer. This is consistent with
reader distraction, which needs no retrieval failure at all.

**At scale:** the single agent reached its search limit for about half of `test_agents`
(weighted 0.53) and stopped as "sufficient" for only 0.34. On `dev`, where R already held
both gold paragraphs, the agent's F1 was 0.033 lower.

## 5. Multi-agent selection loss: retrieved, then discarded

*"Jolyon James performed the role of Moonshadow in a musical by a composer of what
nationality?"* The gold answer is **British**.

**Evidence:**

- *Jolyon James:* "He performed the role of the Moonshadow in **Cat Stevens's** musical…"
- *Cat Stevens:* "…commonly known by his former stage name Cat Stevens, is a **British**
  singer-songwriter…"

**What happened:**

- **The planner** produced two sound requirements:
  1. "The musical in which Jolyon James played Moonshadow."
  2. "The nationality of the composer of that musical."
- **Search 1** (R's own top 10) already contained both gold paragraphs.
- **The researcher for requirement 1** accepted *Jolyon James*.
- **The researcher for requirement 2** accepted *James Dooley (composer)*, an American film
  composer whose title matches "composer". It did not accept *Cat Stevens*, though it was
  among its candidates.
- **The verifier** judged requirement 2 unsupported, and its corrective query named the
  right entity: "nationality of the composer of Cat Stevens's musical".
- **The corrective researcher** saw *Cat Stevens* again among its candidates, and accepted
  *Jackie James* instead.
- **The synthesizer** saw only those three paragraphs and answered "Australian". R, with
  all ten paragraphs, answered "British".

**Mechanism:** a researcher choosing evidence for "the composer" matched the word
"composer" instead of following the entity the first paragraph named.

- Unaccepted paragraphs stay available to later researchers.
- They reach the verifier and the synthesizer only if a researcher accepts them.
- Here, three researcher calls saw *Cat Stevens*, and none accepted it.

**At scale:** on `test_agents`, the multi-agent system's weighted Joint Recall fell from
0.860 in the search pool to 0.596 in the answer context. In the balanced sample, both gold
paragraphs were in the pool for 89/100 questions and in the context for 65/100. This
replicates the `dev` finding. Whether the loss lowered answer F1 is not established.

## What the five cases show together

- **Getting the second hop into context is the core problem** (cases 1, 3).
- **Ordering and selection can silently remove it again** (cases 2, 5).
- **More evidence is not automatically better** (case 4).

Hence the recommendation: deterministic link-following retrieval (D3) with a single model
call. An agent pays off only with a reliable sufficiency signal and a selector that cannot
discard retrieved evidence.
