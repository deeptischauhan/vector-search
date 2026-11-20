# Keel Vector Search — Layer 2 deploy lifecycle
# Run in a Databricks notebook (serverless). Turns the Layer-1 retriever into a
# served, governed, evaluated AI feature:
#   log (MLflow) -> register (Unity Catalog) + @champion alias -> evaluate ->
#   serve (Model Serving endpoint) -> query.
# Pairs with retriever_model.py (the model-from-code).

# --- Cell 0: install + restart ---
# %pip install -U mlflow databricks-vectorsearch
# dbutils.library.restartPython()

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models.resources import DatabricksVectorSearchIndex

UC_MODEL = "workspace.cpt.keel_article_retriever"
INDEX = "workspace.cpt.articles_index"
ENDPOINT = "keel-retriever"
K = 5

mlflow.set_registry_uri("databricks-uc")  # register models into Unity Catalog
input_example = pd.DataFrame({"query": ["how do enterprises actually get ROI from AI?"]})


# --- Step 1+2: log the model-from-code to MLflow, WITH the index as a resource ---
# Declaring the Vector Search index as a `resource` is what lets the SERVED model
# authenticate to the index (a serving container has no notebook token). The
# platform injects a short-lived, scoped credential for exactly that resource.
with mlflow.start_run(run_name="keel-article-retriever") as run:
    info = mlflow.pyfunc.log_model(
        name="retriever",
        python_model="retriever_model.py",
        input_example=input_example,
        pip_requirements=["mlflow", "databricks-vectorsearch", "pandas"],
        resources=[DatabricksVectorSearchIndex(index_name=INDEX)],
    )
print("model_uri:", info.model_uri)


# --- Step 3: register in Unity Catalog + move the @champion alias ---
mv = mlflow.register_model(info.model_uri, UC_MODEL)
MlflowClient().set_registered_model_alias(UC_MODEL, "champion", mv.version)
print("champion ->", UC_MODEL, "v", mv.version)


# --- Step 4: evaluate the @champion model, tracked in MLflow ---
# Reuses the Layer-1 retrieval gold set; runs it THROUGH the registered model so
# the scores attach to the model version (later versions compete on the same bar).
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
champ = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}@champion")
preds = champ.predict(pd.DataFrame({"query": [q for q, _ in GOLD]}))
hits, rr = 0, 0.0
for (q, gold), hitlist in zip(GOLD, preds):
    ids = [h["id"] for h in hitlist]
    rank = ids.index(gold) + 1 if gold in ids else 0
    hits += rank > 0
    rr += (1.0 / rank) if rank else 0.0
n = len(GOLD)
recall, mrr = hits / n, rr / n
print(f"Recall@{K}: {hits}/{n} = {recall:.0%}   MRR: {mrr:.3f}")
with mlflow.start_run(run_name="retriever-eval"):
    mlflow.log_metrics({f"recall_at_{K}": recall, "mrr": mrr})
    mlflow.log_param("eval_gold_n", n)
    mlflow.log_param("model_version", mv.version)
# Our run: Recall@5 = 100% (17/17), MRR = 0.931.
# NOTE: for GENERATED answers (full RAG) use mlflow.genai.evaluate with LLM-judge
# scorers; for pure retrieval, Recall@k / MRR is the right metric.


# --- Step 5: create the serving endpoint ---
# Easiest via the UI: Serving -> Create serving endpoint -> name "keel-retriever",
# entity workspace.cpt.keel_article_retriever (the @champion version), CPU 4 GB
# (smallest; GPU is not on Free Edition), Scale to zero ON -> Create.
# First build takes ~5-15 min. Programmatic equivalent:
#
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import (
#     EndpointCoreConfigInput, ServedEntityInput)
# w = WorkspaceClient()
# w.serving_endpoints.create(
#     name=ENDPOINT,
#     config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(
#         entity_name=UC_MODEL, entity_version=str(mv.version),
#         workload_size="Small", scale_to_zero_enabled=True)]))


# --- Step 6: call the live endpoint ---
# The REST URL is https://<workspace-host>/serving-endpoints/keel-retriever/invocations
# (needs an auth token; it is NOT an anonymous public URL).
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
resp = client.predict(
    endpoint=ENDPOINT,
    inputs={"dataframe_records": [{"query": "what should I check before deploying AI agents?"}]},
)
print(resp)
