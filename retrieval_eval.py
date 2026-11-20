# Keel Vector Search - Retrieval Eval Harness
# Run in a Databricks notebook (Python). Measures how good retrieval is over
# workspace.cpt.articles_index, using a small labeled gold set.
#
# Two stages (the RAG eval pattern):
#   Stage 1 - RETRIEVAL: did the correct chunk come back in top-k?  (Recall@k, MRR)
#   Stage 2 - ANSWER (optional): is the generated answer good?       (LLM-as-judge)
# Stage 1 is the only thing new vs the planner/tutor evals; it isolates
# retrieval failures from generation failures.

# --- Cell 1: install + connect ---
# %pip install databricks-vectorsearch
# dbutils.library.restartPython()

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name="keel-vector-search",
    index_name="workspace.cpt.articles_index",
)

# --- The gold set: a realistic question -> the article id that SHOULD answer it ---
# Questions are phrased differently from the titles on purpose, so we test
# semantic matching, not keyword overlap. (A set of acceptable ids also works;
# single id kept here for simplicity.)
GOLD = [
    ("how do I measure whether people actually trust a deployed AI product?", "a13"),
    ("what are the stages of rolling out enterprise AI and where do teams fail?", "a08"),
    ("how should I split long documents into pieces for a RAG system?", "a18"),
    ("what vector search does Databricks provide and how do I query it?", "a20"),
    ("how do enterprises actually get financial return from AI?", "a01"),
    ("what can Mistral's agent API do - connectors, tools, memory?", "a02"),
    ("what are good patterns for building effective AI agents?", "a17"),
    ("what should I check before deploying AI agents?", "a06"),
    ("how I built an evaluation framework for an AI pipeline", "a15"),
    ("how do I design memory for an AI agent?", "a16"),
    ("stop using vanity metrics to measure AI adoption", "a12"),
    ("stakeholder analysis for AI governance", "a14"),
    ("what is hybrid search in a vector index?", "a21"),
    ("power/interest stakeholder engagement for change", "a05"),
    ("Mistral's enterprise chat product and its model", "a03"),
    ("measuring how autonomous an AI agent really is", "a04"),
    ("closing the gap between AI adoption and daily use", "a11"),
]

K = 5


def evaluate(k=K):
    hits = 0
    rr_sum = 0.0
    print(f"{'expected':<9}{'rank':<6}{'hit':<5}query")
    print("-" * 70)
    for question, expected in GOLD:
        res = index.similarity_search(
            query_text=question,
            columns=["id", "title", "category"],
            num_results=k,
        )
        ids = [row[0] for row in res["result"]["data_array"]]
        rank = ids.index(expected) + 1 if expected in ids else 0   # 1-based; 0 = miss
        hit = rank > 0
        hits += hit
        rr_sum += (1.0 / rank) if rank else 0.0
        print(f"{expected:<9}{(rank or '-'):<6}{('Y' if hit else 'N'):<5}{question[:52]}")
    n = len(GOLD)
    print("-" * 70)
    print(f"Recall@{k}:  {hits}/{n} = {hits / n:.0%}   (how often the right article is in the top {k})")
    print(f"MRR:       {rr_sum / n:.3f}            (1/rank of the right article, averaged; 1.0 = always #1)")
    print("\nRead the MISSES (hit=N) and low ranks: that's where chunking/retrieval is weak.")


evaluate()


# ===========================================================================
# Stage 2 (OPTIONAL) - answer quality via LLM-as-judge.
# Same idea as your planner/tutor judge: retrieve -> answer -> judge.
# Needs an LLM call (Anthropic key, or a Databricks foundation model). Skeleton:
#
# def answer_and_judge(question, k=K):
#     res = index.similarity_search(query_text=question,
#                                   columns=["id", "title", "content"], num_results=k)
#     context = "\n\n".join(row[2] for row in res["result"]["data_array"])  # content col
#     # 1) answer: ask an LLM to answer ONLY from `context`
#     # 2) judge: ask a second LLM call to score faithfulness (0/1) + sufficiency (0-2)
#     #    - faithfulness: is every claim supported by the context?
#     #    - sufficiency:  did the context contain enough to answer?
#     # Return the scores; average across GOLD like Stage 1.
#     ...
#
# Diagnosis rule: low Recall@k -> retrieval/chunking problem;
#                 high Recall@k but low answer score -> generation/prompt problem.
# ===========================================================================


# ===========================================================================
# Comparing CHUNKING strategies:
# To compare (e.g. whole-article vs chunked), you'd build a SECOND index over a
# chunked table and run evaluate() against each, then compare Recall@k / MRR.
# On Free Edition (one index / one search unit) you'd delete articles_index first,
# or reuse the single slot. The method is identical - just a different index name.
# ===========================================================================
