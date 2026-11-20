# Keel Vector Search — the retriever as an MLflow "model from code"
# (Layer 2). MLflow packages this whole file as the model; `set_model` at the
# bottom marks the object to serve. Loaded by deploy_retriever.py via
# python_model="retriever_model.py".

import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from databricks.vector_search.client import VectorSearchClient


class ArticleRetriever(PythonModel):
    """Query -> ranked articles from the Keel AI Search index."""

    def load_context(self, context):
        # runs once when the model loads (notebook or serving container)
        self.vsc = VectorSearchClient()
        self.index = self.vsc.get_index(
            endpoint_name="keel-vector-search",
            index_name="workspace.cpt.articles_index",
        )

    def predict(self, context, model_input, params=None):
        # model_input: a DataFrame with a 'query' column (optional 'k')
        out = []
        for _, row in model_input.iterrows():
            k = int(row["k"]) if "k" in row and not pd.isna(row.get("k")) else 5
            res = self.index.similarity_search(
                query_text=row["query"],
                columns=["id", "title", "category", "url"],
                num_results=k,
            )
            hits = res["result"]["data_array"]
            out.append([
                {"id": r[0], "title": r[1], "category": r[2], "url": r[3], "score": r[4]}
                for r in hits
            ])
        return out


# tells MLflow which object in this file is the model
mlflow.models.set_model(ArticleRetriever())
