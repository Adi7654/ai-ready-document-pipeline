# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Capability check (run once, before the embedding notebook)
# MAGIC Answers four questions so Session 5 is built on what Free Edition actually allows:
# MAGIC
# MAGIC | # | Question | Why it matters |
# MAGIC |---|---|---|
# MAGIC | A | Can we download an open multilingual model (`intfloat/multilingual-e5-small`) from Hugging Face? | The same model must run here **and** on the public website |
# MAGIC | B | Does it put the same meaning in different languages close together? | That's what makes cross-lingual search work |
# MAGIC | C | How fast does it embed on serverless CPU, on the driver and spread across workers? | Tells us how long the full 40,975-chunk run takes |
# MAGIC | D | Which Databricks model endpoints and Vector Search can we use? | Embedding endpoints, an LLM for Session 6, and the vector index |
# MAGIC
# MAGIC Every test is wrapped so a failure prints a reason instead of stopping the notebook.
# MAGIC **Send screenshots of the RESULTS cell at the end** (and of any cell that prints FAIL).

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, time, traceback, platform
from pyspark.sql import functions as F

CATALOG = "workspace"
CHUNKS = f"{CATALOG}.docs_gold.chunks"
MODEL_NAME = "intfloat/multilingual-e5-small"
VS_ENDPOINT = "docs_vs_endpoint"

results = {}

def record(key, ok, detail):
    results[key] = (ok, detail)
    print(f"{'PASS' if ok else 'FAIL'}  {key}: {detail}")

try:
    import torch
    torch_info = f"torch {torch.__version__}, threads {torch.get_num_threads()}"
except Exception as e:
    torch_info = f"torch not importable: {e}"
record("environment", True, f"Python {platform.python_version()}, CPUs on driver {os.cpu_count()}, {torch_info}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A · Download the model from Hugging Face

# COMMAND ----------

model = None
try:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    record("A_model_download", True,
           f"{MODEL_NAME} loaded in {time.time() - t0:.0f}s, {model.get_sentence_embedding_dimension()} dimensions, "
           f"max {model.max_seq_length} tokens")
except Exception as e:
    record("A_model_download", False, f"{type(e).__name__}: {str(e)[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B · Does meaning match across languages?
# MAGIC e5 models expect `query: ` before a question and `passage: ` before a document.

# COMMAND ----------

if model is not None:
    try:
        import numpy as np
        passages = {
            "de": "Ein Großbrand hat das Dach der Kathedrale Notre-Dame in Paris zerstört.",
            "es": "Un gran incendio destruyó el tejado de la catedral de Notre Dame en París.",
            "fr": "Un violent incendie a détruit la toiture de la cathédrale Notre-Dame de Paris.",
            "ru": "Сильный пожар уничтожил крышу собора Парижской Богоматери.",
            "tr": "Büyük bir yangın Paris'teki Notre Dame Katedrali'nin çatısını yok etti.",
            "de_other": "Die Bundesbank senkt ihre Wachstumsprognose für das kommende Jahr.",
            "ru_other": "Сборная России по футболу проиграла товарищеский матч.",
        }
        q = model.encode(["query: What happened to Notre-Dame cathedral?"], normalize_embeddings=True)
        p = model.encode([f"passage: {t}" for t in passages.values()], normalize_embeddings=True)
        scores = dict(zip(passages, (p @ q.T).ravel()))
        for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {k:9s} {v:.3f}")
        on_topic = min(v for k, v in scores.items() if "_other" not in k)
        off_topic = max(v for k, v in scores.items() if "_other" in k)
        record("B_cross_lingual", on_topic > off_topic,
               f"English question: lowest on-topic score {on_topic:.3f} vs highest off-topic {off_topic:.3f}")
    except Exception as e:
        record("B_cross_lingual", False, f"{type(e).__name__}: {str(e)[:300]}")
else:
    record("B_cross_lingual", False, "skipped: model not loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C1 · Speed on the driver (real chunks)

# COMMAND ----------

TOTAL_CHUNKS = spark.table(CHUNKS).count()
if model is not None:
    try:
        sample = [r["chunk_text"] for r in spark.table(CHUNKS).select("chunk_text").limit(256).collect()]
        t0 = time.time()
        model.encode([f"passage: {t}" for t in sample], batch_size=32, normalize_embeddings=True)
        rate = len(sample) / (time.time() - t0)
        record("C1_driver_speed", True,
               f"{rate:.1f} chunks/s on the driver → all {TOTAL_CHUNKS:,} chunks ≈ {TOTAL_CHUNKS / rate / 60:.0f} min")
    except Exception as e:
        record("C1_driver_speed", False, f"{type(e).__name__}: {str(e)[:300]}")
else:
    record("C1_driver_speed", False, "skipped: model not loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C2 · Speed spread across workers (pandas UDF on 2,000 chunks)
# MAGIC Workers download the model themselves, so this also tests whether they can reach Hugging Face.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import types as T

@pandas_udf(T.ArrayType(T.FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_NAME, device="cpu")
    vecs = m.encode([f"passage: {t}" for t in texts], batch_size=32, normalize_embeddings=True)
    return pd.Series([v.tolist() for v in vecs])

try:
    t0 = time.time()
    n = (spark.table(CHUNKS).select("chunk_id", "chunk_text").limit(2000).repartition(8)
         .select(embed_udf("chunk_text").alias("v")).where(F.size("v") > 0).count())
    secs = time.time() - t0
    record("C2_worker_speed", True,
           f"{n:,} chunks in {secs:.0f}s ({n / secs:.1f}/s incl. model load) → all {TOTAL_CHUNKS:,} ≈ "
           f"{TOTAL_CHUNKS / (n / secs) / 60:.0f} min")
except Exception as e:
    record("C2_worker_speed", False, f"{type(e).__name__}: {str(e)[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## D1 · Databricks model endpoints (embeddings + LLMs)

# COMMAND ----------

chat_endpoint, embed_endpoints = None, []
try:
    from mlflow.deployments import get_deploy_client
    endpoints = get_deploy_client("databricks").list_endpoints()
    by_task = {}
    for e in endpoints:
        by_task.setdefault(e.get("task") or "other", []).append(e["name"])
    for task, names in sorted(by_task.items()):
        print(f"{task}: {', '.join(sorted(names))}")
    embed_endpoints = sorted(by_task.get("llm/v1/embeddings", []))
    chats = sorted(by_task.get("llm/v1/chat", []))
    preferred = [n for n in chats if "llama" in n and "70b" in n] + [n for n in chats if "llama" in n] + chats
    chat_endpoint = preferred[0] if preferred else None
    record("D1_endpoints", True, f"{len(endpoints)} endpoints; embeddings: {embed_endpoints or 'none'}; "
                                 f"chat model picked: {chat_endpoint or 'none'}")
except Exception as e:
    record("D1_endpoints", False, f"{type(e).__name__}: {str(e)[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## D2 · Can we call an LLM? (needed in Session 6)

# COMMAND ----------

if chat_endpoint:
    try:
        t0 = time.time()
        answer = spark.sql(f"""
            SELECT ai_query('{chat_endpoint}',
              'Translate to English in one short sentence: Сильный пожар уничтожил крышу собора Парижской Богоматери.') AS a
        """).first()["a"]
        record("D2_llm_call", True, f"{chat_endpoint} answered in {time.time() - t0:.0f}s: {answer[:150]}")
    except Exception as e:
        record("D2_llm_call", False, f"{type(e).__name__}: {str(e)[:300]}")
else:
    record("D2_llm_call", False, "skipped: no chat endpoint listed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## D3 · Vector Search endpoint
# MAGIC Free Edition allows one endpoint. If none exists, this starts creating it now (it takes ~10–20 min to
# MAGIC come online, so it's ready by the time the embeddings are).

# COMMAND ----------

try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if VS_ENDPOINT not in existing and not existing:
        vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
        record("D3_vector_search", True, f"created endpoint '{VS_ENDPOINT}' (provisioning; check Compute → Vector Search)")
    else:
        record("D3_vector_search", True, f"existing endpoints: {existing}")
except Exception as e:
    record("D3_vector_search", False, f"{type(e).__name__}: {str(e)[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RESULTS (screenshot this)

# COMMAND ----------

for key, (ok, detail) in results.items():
    print(f"{'PASS' if ok else 'FAIL'}  {key:18s} {detail}")
