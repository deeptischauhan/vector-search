# Keel — Vector Search Reading Recommender (RAG, deployed on Databricks)

Part of the **Keel** capstone (Career Transition Planner): a **RAG retrieval layer** that recommends what a learner should *read*, built and **served end-to-end on Databricks**.

- **Layer 1 — retrieval.** Given a learner's context, return the most relevant articles by **meaning**, scoped by **category** (business acumen / peripheral skills / positioning), on **Databricks AI Search** (formerly Mosaic AI Vector Search).
- **Layer 2 — deploy.** Turn that retriever into a **governed, evaluated, served AI feature**: MLflow model → Unity Catalog (with a `@champion` alias) → MLflow-tracked evaluation → a **Model Serving** endpoint you call over REST.

Built on **Databricks Free Edition**. Doubles as hands-on prep for the Databricks Generative AI Engineer Associate exam (Vector Search, MLflow, Unity Catalog model registry, Model Serving).

## What it does

```
curated articles (Delta table)
      → embed `content` with GTE (databricks-gte-large-en)
      → Databricks AI Search Delta Sync index (Hybrid: vector + keyword)
      → query by meaning + filter by category
      → wrap as an MLflow model → register in Unity Catalog (@champion)
      → serve behind a Model Serving endpoint
      → POST a question → ranked reads back as JSON
```

- **Semantic search** — matches on meaning, not keywords.
- **Metadata filtering** — scope results to a `category`.
- **Hybrid index** — vector + keyword retrieval together.
- **Served feature** — callable over REST, no notebook required.

## Files

| File | What it is |
|---|---|
| `setup.sql` | Prepare the source Delta table: Change Data Feed, primary key, deleted-file retention. |
| `query_vector_search.py` | Notebook code: connect to the index and run semantic + filtered queries. |
| `chunking_lab.py` | Pure-Python chunking demo — see how chunk size/overlap split a long doc. |
| `retrieval_eval.py` | Retrieval-eval harness over the index: Recall@k + MRR against a labeled gold set. |
| `retriever_model.py` | **Layer 2** — the retriever packaged as an MLflow "model from code". |
| `deploy_retriever.py` | **Layer 2** — the full deploy lifecycle: log → register → alias → evaluate → serve → query. |
| `data/articles_corpus.csv` | The article corpus (id, title, url, source, category, skill, content). |
| `RESULTS.md` | Captured query outputs with commentary (the hand-read validation). |

## Prerequisites

- A Databricks workspace with **Unity Catalog** (Free Edition works).
- An **embedding model** available (Free Edition offers `databricks-gte-large-en`, `bge-large-en`, `qwen3-embedding` via Unity Gateway). This project uses **GTE**.
- One **AI Search endpoint** (Free Edition allows exactly one; Direct Vector Access is **not** supported, so a **Delta Sync index** is required).

## Build steps

### Layer 1 — retrieval

1. **Load the corpus** `data/articles_corpus.csv` into a Unity Catalog Delta table `workspace.cpt.articles` (Catalog → Create or modify table from files). `id` is the primary key.
2. **Prepare the table** — run `setup.sql`: enable Change Data Feed, set `id` NOT NULL + PRIMARY KEY, extend deleted-file retention to 30 days.
3. **Create the AI Search endpoint** (`keel-vector-search`).
4. **Create a Delta Sync index** `workspace.cpt.articles_index` on the endpoint:
   - Source table `workspace.cpt.articles`; primary key `id`.
   - Embeddings: **Compute embeddings** · source column `content` · model `databricks-gte-large-en`.
   - Index type **Hybrid** · Update mode **Triggered**.
   - Columns to index: include `category` (for filtering) + `title`, `url`, `source`, `skill`.
5. **Query it** — run `query_vector_search.py` (see `RESULTS.md`). Validate with `retrieval_eval.py`.

### Layer 2 — deploy (`retriever_model.py` + `deploy_retriever.py`)

6. **Log** the retriever as an MLflow model-from-code, **with the index declared as a resource** (so the served model can authenticate to it).
7. **Register** it in Unity Catalog as `workspace.cpt.keel_article_retriever` and set a movable **`@champion`** alias.
8. **Evaluate** the champion model against the 17-question gold set; log Recall@k + MRR to MLflow so the scores attach to the version.
9. **Serve** it: Serving → Create serving endpoint → `keel-retriever`, the champion version, CPU 4 GB, Scale to zero → Create.
10. **Call** `https://<workspace-host>/serving-endpoints/keel-retriever/invocations` (auth token required).

## Concepts (why each piece matters)

- **Embedding model** — turns text into a vector capturing meaning; similar meaning → nearby vectors. The index embeds both the stored articles and the query, then returns the closest.
- **Chunking** — splitting long documents into passages before embedding (knobs: chunk size + overlap). See `chunking_lab.py`.
- **Change Data Feed (CDF)** — records row-level changes so the Delta Sync index re-embeds only changed rows.
- **Deleted-file retention** — extended to 30 days so a Triggered sync never fails looking for vacuumed files.
- **Metadata filtering** — `category` rides along (not embedded) so queries can be scoped.
- **Hybrid search** — combines vector (semantic) and keyword retrieval.
- **MLflow model-from-code** — package the model as a file, not a pickle; the modern, reproducible approach.
- **Unity Catalog model registry + alias** — a versioned, governed model; apps call `@champion`, so versions swap underneath without downstream changes.
- **Resource-based serving auth** — declaring the index as a `resource` at log time lets the serving container authenticate to it (no notebook token on the server).

## Result

- **Retrieval, hand-read** across all three categories (full outputs in `RESULTS.md`): a ROI query tops with the McKinsey "On the Road to ROI" report; a no-shared-keywords "how do I measure if people actually trust the AI?" query surfaces the trust-ladder / measuring-adoption cluster; category filters cleanly isolate results.
- **Retrieval, measured:** **Recall@5 = 100% (17/17), MRR = 0.931** over the labeled gold set — recorded in MLflow against the model version.
- **Deployed:** the retriever answers a live REST call to the `keel-retriever` serving endpoint.

## Free Edition note

Custom models serve on **CPU only**, with capped active endpoints, **no SLA**, and **scale-to-zero** (cold-start on the first call after idle). Perfect for a portfolio demo; for a production deployment you'd move to a paid workspace for real uptime — the architecture and skills are identical.
