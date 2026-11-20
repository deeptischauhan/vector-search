# Keel Vector Search — query examples
# Run in a Databricks notebook (Python, serverless compute).
# Inside a Databricks notebook, authentication to the index is automatic — no token needed.

# --- Cell 1: install the SDK, then restart the Python kernel ---
# %pip install databricks-vectorsearch
# dbutils.library.restartPython()

# --- Cell 2: connect to the index ---
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="keel-vector-search",
    index_name="workspace.cpt.articles_index",
)


def show(results):
    """Print each result row: [id, title, category, url, score]."""
    for row in results["result"]["data_array"]:
        print(row)


# --- Cell 3: plain semantic search (no filter) — the "meaning test" ---
# Query shares almost no keywords with the target articles, yet finds them.
show(index.similarity_search(
    query_text="how do I measure if people actually trust the AI?",
    columns=["id", "title", "category", "url"],
    num_results=5,
))

# --- Cell 4: metadata filtering — scope to one category ---
show(index.similarity_search(
    query_text="chunking and embeddings for retrieval",
    columns=["id", "title", "category", "url"],
    filters={"category": "peripheral"},
    num_results=5,
))

# --- Cell 5: a different category ---
show(index.similarity_search(
    query_text="how do companies get real ROI from AI",
    columns=["id", "title", "category", "url"],
    filters={"category": "business_acumen"},
    num_results=5,
))

# --- Cell 6: remove the filter to see its effect ---
# Rerun Cell 4's query WITHOUT filters and compare — a cross-category article
# (e.g. 'Agent Memory', category=positioning) ranks in, proving the filter
# was genuinely excluding other categories.
show(index.similarity_search(
    query_text="chunking and embeddings for retrieval",
    columns=["id", "title", "category", "url"],
    num_results=5,
))
