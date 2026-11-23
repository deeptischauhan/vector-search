# Keel — Vector Search Reading Recommender (RAG, deployed on Databricks)

Part of the **Keel** capstone (Career Transition Planner): a **RAG reading-recommender** built, served, and *honestly evaluated* end-to-end on Databricks.

- **Layer 1 — retrieval.** Given a learner's context, return the most relevant articles by **meaning**, scoped by **category**, on **Databricks AI Search** (formerly Mosaic AI Vector Search).
- **Layer 2 — deploy.** Turn the retriever into a **governed, served AI feature**: MLflow model → Unity Catalog (`@champion` alias) → MLflow-tracked eval → a **Model Serving** endpoint called over REST.
- **Layer 3 — generate + evaluate honestly.** Add grounded, cited answers with a **Foundation Model**, then measure answer quality (**faithfulness + sufficiency**) with an **LLM-as-judge** — and hand-review the result instead of trusting the aggregate.

Built on **Databricks Free Edition**. Doubles as hands-on prep for the Databricks Generative AI Engineer Associate exam (Vector Search, MLflow, Unity Catalog model registry, Model Serving, Foundation Model APIs, LLM-as-judge eval).

## What it does

```
curated articles (Delta table)
      → embed `content` with GTE → AI Search Delta Sync index (Hybrid)
      → retrieve by meaning + filter by category
      → wrap as MLflow model → register in Unity Catalog (@champion) → serve over REST
      → generate a grounded, cited answer with a Foundation Model (Llama 3.3 70B)
      → evaluate answers for faithfulness + sufficiency (LLM-as-judge)
```

## Files

| File | What it is |
|---|---|
| `setup.sql` | Prepare the source Delta table: Change Data Feed, primary key, deleted-file retention. |
| `query_vector_search.py` | Connect to the index and run semantic + filtered queries. |
| `chunking_lab.py` | Pure-Python chunking demo — chunk size/overlap on a long doc. |
| `retrieval_eval.py` | Retrieval-eval harness: Recall@k + MRR against a labeled gold set. |
| `retriever_model.py` | **Layer 2** — the retriever as an MLflow "model from code". |
| `deploy_retriever.py` | **Layer 2** — deploy lifecycle: log → register → alias → evaluate → serve → query. |
| `rag_answer.py` | **Layer 3** — grounded, cited generation + faithfulness/sufficiency LLM-judge eval. |
| `data/articles_corpus.csv` | The article corpus (id, title, url, source, category, skill, content). |
| `RESULTS.md` | Captured retrieval query outputs with commentary. |

## Build steps (short)

1. Load `data/articles_corpus.csv` → `workspace.cpt.articles`; run `setup.sql`.
2. Create the AI Search endpoint `keel-vector-search` and a Delta Sync index `articles_index` (embed `content` with `databricks-gte-large-en`, Hybrid, include `category`).
3. Query + validate retrieval (`query_vector_search.py`, `retrieval_eval.py`).
4. **Layer 2:** log/register/alias/evaluate/serve the retriever (`retriever_model.py`, `deploy_retriever.py`).
5. **Layer 3:** generate grounded answers + run the answer eval (`rag_answer.py`).

## Evaluation & limitations (the honest version)

Retrieval and generation both score well on clean cases — **Recall@5 = 100% / MRR 0.931** on a 17-question gold set, and **faithfulness 1.00** with correct refusals on out-of-corpus questions. But those numbers are *saturated*, and the interesting part is what hand-review found underneath them:

- **The gold set is small and easy**, so 100% recall is a ceiling effect, not evidence of robustness. A bigger, graded, adversarial set is the real test.
- **The system rejects blatant false premises** ("will agents *guarantee* value?" → "no, ~6% see significant profit impact [a01]") but **subtly accepts leading ones**: asked how the (enterprise) *trust ladder* applies to *consumer* products, it bridged sources the corpus doesn't connect. Nameable failure mode: **premise-acceptance / conflation.**
- **Answers are capped by corpus detail.** The corpus `content` is one-paragraph *summaries* (kept that way for licensing safety), so grounding is over summaries, not full text — e.g. "which embedding model does Vector Search require?" couldn't be answered because the summary lacks that specific.
- **The LLM judge is lenient and variable.** Same model family judging its own output: it scored the conflation case faithful, missed an "ensures success" overclaim, and scored identical refusals 0 then 1 across runs. **An independent, stricter judge is needed to trust the number.**

So the honest headline is not "100% faithful" — it's *"no faithfulness failures on clear cases by a lenient same-model judge; hand-review surfaced a conflation mode and a corpus-detail limit."*

## Free Edition note

Custom models serve on **CPU only**, with capped active endpoints, **no SLA**, and **scale-to-zero** (cold-start on first call). Perfect for a portfolio demo; a production deployment would move to a paid workspace for uptime — the architecture and skills are identical.
