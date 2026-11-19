# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Capability check 2: pick the embedding path
# MAGIC | Path | Test |
# MAGIC |---|---|
# MAGIC | **A** · `multilingual-e5-small` on our workers | Save the model to a Volume once (workers can't reach Hugging Face), then embed 2,000 chunks across workers |
# MAGIC | **B** · Databricks-hosted `databricks-qwen3-embedding-0-6b` | Same cross-lingual test as before, then embed 500 real chunks with `ai_query` and time it |
# MAGIC
# MAGIC **Send a screenshot of the RESULTS cell** (and any FAIL).

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf

CATALOG = "workspace"
CHUNKS = f"{CATALOG}.docs_gold.chunks"
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DIR = f"/Volumes/{CATALOG}/docs_gold/models/multilingual-e5-small"
QWEN = "databricks-qwen3-embedding-0-6b"
QWEN_QUERY_PREFIX = "Instruct: Given a news question, retrieve news passages that answer it\nQuery: "

TOTAL = spark.table(CHUNKS).count()
results = {}

def record(key, ok, detail):
    results[key] = (ok, detail)
    print(f"{'PASS' if ok else 'FAIL'}  {key}: {detail}")

PASSAGES = {
    "de": "Ein Großbrand hat das Dach der Kathedrale Notre-Dame in Paris zerstört.",
    "es": "Un gran incendio destruyó el tejado de la catedral de Notre Dame en París.",
    "fr": "Un violent incendie a détruit la toiture de la cathédrale Notre-Dame de Paris.",
    "ru": "Сильный пожар уничтожил крышу собора Парижской Богоматери.",
    "tr": "Büyük bir yangın Paris'teki Notre Dame Katedrali'nin çatısını yok etti.",
    "de_other": "Die Bundesbank senkt ihre Wachstumsprognose für das kommende Jahr.",
    "ru_other": "Сборная России по футболу проиграла товарищеский матч.",
}
QUESTION = "What happened to Notre-Dame cathedral?"

def margin_report(key, q, p):
    q = q / np.linalg.norm(q)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    scores = dict(zip(PASSAGES, (p @ q)))
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:9s} {v:.3f}")
    on = min(v for k, v in scores.items() if "_other" not in k)
    off = max(v for k, v in scores.items() if "_other" in k)
    record(key, on > off, f"lowest on-topic {on:.3f} vs highest off-topic {off:.3f} (margin {on - off:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A1 · Save e5-small to a Volume (once)

# COMMAND ----------

try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.docs_gold.models")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(MODEL_NAME, device="cpu").save(MODEL_DIR)
    record("A1_saved_to_volume", True, MODEL_DIR)
except Exception as e:
    record("A1_saved_to_volume", False, f"{type(e).__name__}: {str(e)[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A2 · Embed 2,000 chunks across workers, loading the model from the Volume

# COMMAND ----------

@pandas_udf(T.ArrayType(T.FloatType()))
def e5_udf(texts: pd.Series) -> pd.Series:
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(max(1, torch.get_num_threads()))
    m = SentenceTransformer(MODEL_DIR, device="cpu")
    vecs = m.encode([f"passage: {t}" for t in texts], batch_size=32, normalize_embeddings=True)
    return pd.Series([v.tolist() for v in vecs])

try:
    t0 = time.time()
    n = (spark.table(CHUNKS).select("chunk_text").limit(2000).repartition(16)
         .select(e5_udf("chunk_text").alias("v")).where(F.size("v") == 384).count())
    rate = n / (time.time() - t0)
    record("A2_e5_workers", n == 2000,
           f"{n:,} chunks at {rate:.1f}/s incl. model load → all {TOTAL:,} ≈ {TOTAL / rate / 60:.0f} min")
except Exception as e:
    msg = str(e)
    record("A2_e5_workers", False, f"{type(e).__name__}: {msg[-600:]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B1 · Qwen3 endpoint: cross-lingual test

# COMMAND ----------

try:
    texts = [QWEN_QUERY_PREFIX + QUESTION] + list(PASSAGES.values())
    rows = spark.createDataFrame([(i, t) for i, t in enumerate(texts)], "i int, t string") \
        .selectExpr("i", f"ai_query('{QWEN}', t) AS v").orderBy("i").collect()
    vecs = np.array([r["v"] for r in rows], dtype=np.float32)
    print(f"dimensions: {vecs.shape[1]}")
    margin_report("B1_qwen_cross_lingual", vecs[0], vecs[1:])
except Exception as e:
    record("B1_qwen_cross_lingual", False, f"{type(e).__name__}: {str(e)[:400]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B2 · Qwen3 endpoint: speed on 500 real chunks (watch for rate-limit errors)

# COMMAND ----------

try:
    t0 = time.time()
    n = (spark.table(CHUNKS).select("chunk_text").limit(500)
         .selectExpr(f"ai_query('{QWEN}', chunk_text) AS v").where("size(v) > 0").count())
    rate = n / (time.time() - t0)
    record("B2_qwen_speed", n == 500, f"{n} chunks at {rate:.1f}/s → all {TOTAL:,} ≈ {TOTAL / rate / 60:.0f} min")
except Exception as e:
    record("B2_qwen_speed", False, f"{type(e).__name__}: {str(e)[:400]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RESULTS (screenshot this)

# COMMAND ----------

for key, (ok, detail) in results.items():
    print(f"{'PASS' if ok else 'FAIL'}  {key:22s} {detail}")
