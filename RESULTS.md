# Query results — hand-read validation

Captured from the live `workspace.cpt.articles_index` (GTE embeddings, Hybrid, 22 articles).
Each row is `[id, title, category, url, score]`; higher score = closer in meaning.

## 1. Semantic "meaning test" — no filter

**Query:** `how do I measure if people actually trust the AI?`

```
a12  How to Measure AI Adoption Without Vanity Metrics          positioning     0.663
a10  Are AI Agents Rewriting the Consumer Adoption Curve?       positioning     0.646
a13  Measuring AI That Acts: Adoption Is a Trust Ladder...      positioning     0.623
a04  Measuring Agent Autonomy                                   business_acumen 0.598
a11  Agentic AI Adoption: Closing the Daily-Use Gap            positioning     0.591
```

The query shares almost no words with these titles, yet it surfaced the measuring-adoption / trust cluster — including the "Trust Ladder" article at #3. Pure semantic retrieval.

## 2. Metadata filter — `category = peripheral`

**Query:** `chunking and embeddings for retrieval`

```
a18  The Ultimate Guide to Chunking Strategies for RAG...       peripheral  0.741
a19  Chunking Strategies for RAG                                peripheral  0.625
a20  Mosaic AI Vector Search (documentation)                    peripheral  0.552
a21  Announcing Hybrid Search GA in Mosaic AI Vector Search     peripheral  0.539
a17  Building Effective Agents                                  peripheral  0.495
```

All `peripheral`. The two chunking articles top it; "Building Effective Agents" (peripheral but not about chunking) ranks last — correct relevance ordering.

## 3. Metadata filter — `category = business_acumen`

**Query:** `how do companies get real ROI from AI`

```
a01  The State of AI in 2026: On the Road to ROI               business_acumen 0.627
a06  You're Not Ready for AI Agents Unless You Can Answer...   business_acumen 0.540
a08  Six Stages of Deploying Enterprise AI...                  business_acumen 0.532
a07  Three Roles That Shape Discovery and Engagement...        business_acumen 0.531
a04  Measuring Agent Autonomy                                  business_acumen 0.513
```

A ROI query tops with the McKinsey "On the Road to ROI" report — a bullseye. All `business_acumen`.

## 4. Same query as (2), filter REMOVED — proves the filter works

**Query:** `chunking and embeddings for retrieval` (no filter)

```
a18  The Ultimate Guide to Chunking Strategies for RAG...       peripheral  0.741
a19  Chunking Strategies for RAG                                peripheral  0.625
a20  Mosaic AI Vector Search (documentation)                    peripheral  0.552
a16  Agent Memory: Start Designing What Deserves to Be...      positioning 0.548   <-- cross-category
a21  Announcing Hybrid Search GA in Mosaic AI Vector Search     peripheral  0.539
```

Without the filter, `a16 Agent Memory` (category `positioning`) ranks in at #4 — pushing a peripheral article out of the top 5. Hard evidence the filter was genuinely excluding other categories. (Sensible near-match: "what an agent should *remember*" is semantically close to "embeddings / retrieval.")

## Takeaway

Semantic ranking, metadata filtering, and the measurable difference between them all work. Meaning finds the candidates; the category filter enforces the scope.
