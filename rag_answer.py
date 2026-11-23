# Keel Vector Search — Layer 3: grounded RAG answers + honest answer eval
# Run in a Databricks notebook (serverless). Builds on the Layer-1 index: retrieve
# the most relevant article summaries, generate a grounded, cited answer with a
# Foundation Model, then evaluate answer quality (faithfulness + sufficiency) with
# an LLM-as-judge.
#
# HONEST SCOPE: the corpus `content` is a one-paragraph summary per article (kept
# that way for licensing safety), so answers are grounded in curated SUMMARIES,
# not full article text. See "Findings / limitations" at the bottom.

# --- setup ---
# %pip install -U mlflow databricks-vectorsearch
# dbutils.library.restartPython()
import json, re, statistics as st
import mlflow
from databricks.vector_search.client import VectorSearchClient
from mlflow.deployments import get_deploy_client

index = VectorSearchClient().get_index(
    endpoint_name="keel-vector-search",
    index_name="workspace.cpt.articles_index")
llm = get_deploy_client("databricks")
GEN_MODEL = "databricks-meta-llama-3-3-70b-instruct"


# --- retrieve + generate a grounded, cited answer ---
def rag_answer(question, k=5):
    res = index.similarity_search(
        query_text=question,
        columns=["id", "title", "category", "url", "content"], num_results=k)
    srcs = [{"id": r[0], "title": r[1], "category": r[2], "url": r[3], "content": r[4]}
            for r in res["result"]["data_array"]]
    context = "\n\n".join(f"[{s['id']}] {s['title']}\n{s['content']}" for s in srcs)
    system = ("You are Keel's reading assistant for people moving into AI roles. "
        "Answer using ONLY the sources provided. Cite sources inline by bracket tag, e.g. [a06]. "
        "If the sources don't contain enough to answer, say so plainly and point to the closest source. "
        "Be concise: 3-5 sentences.")
    # temperature 0 for a grounded, reproducible answer
    resp = llm.predict(endpoint=GEN_MODEL, inputs={"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"}],
        "max_tokens": 400, "temperature": 0.0})
    return resp["choices"][0]["message"]["content"], srcs


# --- LLM-as-judge: faithfulness (0/1) + sufficiency (0-2) ---
# NOTE: this judges with the SAME model family it generated with -> self-eval bias.
# A strict rubric + temperature 0 mitigate it, but an independent, stronger judge
# would be more trustworthy. Hand-review found this judge misses subtle overclaims
# and source-conflation (see findings).
STRICT_RUBRIC = (
 "You are a STRICT evaluator of a RAG answer. Judge ONLY against the provided sources. Be harsh.\n"
 "faithfulness = 0 if the answer contains ANY of: a claim not directly stated in the sources; an "
 "overclaim or guarantee the sources do not make; conflating two sources into a claim neither supports; "
 "or accepting a false premise without flagging it. faithfulness = 1 ONLY if every sentence is directly "
 "supported and it makes no claim beyond the sources.\n"
 "sufficiency = 0 if it doesn't or can't answer; 1 if it only partially answers or the sources only "
 "partly cover it; 2 if it fully and directly answers from the sources.\n"
 'Return ONLY JSON: {"faithfulness":0 or 1,"sufficiency":0 or 1 or 2,"reason":"one sentence naming the weakest point"}')


def judge(question, answer, sources):
    context = "\n\n".join(f"[{s['id']}] {s['title']}: {s['content']}" for s in sources)
    resp = llm.predict(endpoint=GEN_MODEL, inputs={"messages": [
        {"role": "system", "content": STRICT_RUBRIC},
        {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer}\n\nSOURCES:\n{context}"}],
        "max_tokens": 220, "temperature": 0.0})
    txt = resp["choices"][0]["message"]["content"]
    try:
        return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception:
        return {"faithfulness": -1, "sufficiency": -1, "reason": "parse error: " + txt[:120]}


# --- eval set: easy + adversarial + out-of-corpus (refusal) ---
EVAL = [
    "what should I check before deploying AI agents?",                   # answerable
    "how do I measure whether people actually trust a deployed AI?",     # answerable
    "will deploying AI agents guarantee business value?",                # FALSE PREMISE (a01: most pilots fail)
    "how does the trust ladder apply to consumer AI products?",          # CONFLATION bait (ladder = enterprise)
    "what embedding model does Databricks Vector Search require?",       # PRECISION (summary lacks the detail)
    "how do I fine-tune a Llama model on my own GPUs?",                  # OUT OF CORPUS -> refuse
    "what is Mistral's current stock price?",                            # OUT OF CORPUS -> refuse
]


def run_eval(questions=EVAL):
    rows = []
    for q in questions:
        a, srcs = rag_answer(q)
        v = judge(q, a, srcs)
        rows.append((q, v, a))
        print(f"F={v['faithfulness']} S={v['sufficiency']} | {q[:58]}")
        print("   ", v["reason"])
    ok = [r for r in rows if r[1]["faithfulness"] >= 0]
    faith = st.mean(r[1]["faithfulness"] for r in ok)
    suff = st.mean(r[1]["sufficiency"] for r in ok)
    print(f"\nfaithfulness {faith:.2f} | sufficiency {suff:.2f}  (n={len(ok)})")
    with mlflow.start_run(run_name="rag-answer-eval"):
        mlflow.log_metrics({"faithfulness_mean": faith, "sufficiency_mean": suff, "n": len(ok)})
    return rows


if __name__ == "__main__":
    run_eval()


# ===========================================================================
# FINDINGS / LIMITATIONS (hand-reviewed, not just the aggregate score)
# ---------------------------------------------------------------------------
# - Grounding + refusal work on clear cases: answers cite faithfully, and the
#   system refuses out-of-corpus questions instead of confabulating.
# - Rejects blatant false premises (e.g. "will agents GUARANTEE value?" -> "no,
#   ~6% see significant profit impact [a01]") but SUBTLY ACCEPTS leading ones:
#   "how does the trust ladder apply to CONSUMER products?" was answered as if the
#   (enterprise) trust-ladder sources applied to consumer, bridging sources the
#   corpus doesn't connect. Nameable failure mode: premise-acceptance / conflation.
# - Answers are capped by CORPUS DETAIL: "which embedding model?" couldn't be
#   answered because the summary (a20) only says "embedding model options" -
#   a direct consequence of grounding in summaries, not full text.
# - The LLM judge is LENIENT and VARIABLE: it scored faithfulness 1.00 even on the
#   conflation case above, missed an "ensures success" overclaim in testing, and
#   scored identical refusals 0 then 1 across runs. So the honest headline is NOT
#   "100% faithful" - it is "no faithfulness failures on clear cases by a lenient
#   same-model judge; hand-review found a conflation mode and a corpus-detail limit;
#   an independent, stricter judge is needed to trust the number."
# ===========================================================================
