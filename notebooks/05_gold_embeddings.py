# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Gold: embeddings + vector index
# MAGIC Turns every chunk into a **1,024-number meaning vector** with the multilingual model
# MAGIC `databricks-qwen3-embedding-0-6b`, then builds a **Vector Search index** over them.
# MAGIC
# MAGIC | Step | What happens |
# MAGIC |---|---|
# MAGIC | 1 · Find work | Only chunks that are **new or changed** (by `chunk_hash`) since the last run get embedded |
# MAGIC | 2 · Embed | `ai_query` on the Databricks-hosted model; failed rows are retried once |
# MAGIC | 3 · Check (blocking) | Every chunk has exactly one vector of 1,024 numbers; nothing stale is left |
# MAGIC | 4 · Write | MERGE into `docs_gold.chunk_vectors` (Change Data Feed on, so the index syncs only changes) |
# MAGIC | 5 · Index | Delta Sync index `docs_gold.chunk_index` on the Vector Search endpoint |
# MAGIC | 6 · Try it | Cross-language searches, with and without a language filter |
# MAGIC
# MAGIC Why this model: in the capability check it separated on-topic from off-topic sentences across five languages
# MAGIC 4× more clearly than `multilingual-e5-small` (margin +0.31 vs +0.07) and embedded 127 chunks/s,
# MAGIC against under 7/s for e5 on serverless CPU.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
from pyspark.sql import functions as F

CATALOG = "workspace"
CHUNKS = f"{CATALOG}.docs_gold.chunks"
VECTORS = f"{CATALOG}.docs_gold.chunk_vectors"
INDEX = f"{CATALOG}.docs_gold.chunk_index"
VS_ENDPOINT = "docs_vs_endpoint"

MODEL = "databricks-qwen3-embedding-0-6b"
DIMENSIONS = 1024
# Qwen3 embeddings: documents are embedded as-is; questions get a short instruction in front
QUERY_PREFIX = "Instruct: Given a news question, retrieve news passages that answer it\nQuery: "

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Find the chunks that need embedding

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {VECTORS} (
  chunk_id STRING NOT NULL, doc_id STRING, chunk_index INT, language STRING, source STRING,
  title STRING, chunk_text STRING, url STRING, topic STRING, published_date DATE,
  chunk_hash STRING, embedding ARRAY<FLOAT>, model STRING, _embedded_at TIMESTAMP
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

chunks = spark.table(CHUNKS)
existing = spark.table(VECTORS).select("chunk_id", F.col("chunk_hash").alias("old_hash"), "model")
todo = (chunks.join(existing, "chunk_id", "left")
        .where(F.col("old_hash").isNull() | (F.col("old_hash") != F.col("chunk_hash")) |
               (F.col("model") != F.lit(MODEL)))
        .drop("old_hash", "model"))
todo_count = todo.count()
print(f"Chunks in gold: {chunks.count():,} · to embed now: {todo_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Embed (title + chunk text), retrying failures once
# MAGIC `failOnError => false` returns an error message per row instead of failing the whole batch.

# COMMAND ----------

def embed(df):
    return (df.withColumn("embed_input", F.concat_ws("\n", F.col("title"), F.col("chunk_text")))
              .withColumn("r", F.expr(f"ai_query('{MODEL}', embed_input, failOnError => false)"))
              .select("*", F.col("r.result").cast("array<float>").alias("embedding"), F.col("r.errorMessage").alias("error"))
              .drop("r", "embed_input"))

embedded = None
if todo_count:
    t0 = time.time()
    embed(todo).write.mode("overwrite").saveAsTable(f"{CATALOG}.docs_gold._embed_batch")
    batch = spark.table(f"{CATALOG}.docs_gold._embed_batch")
    failed = batch.where("embedding IS NULL OR size(embedding) <> {0}".format(DIMENSIONS))
    failed_count = failed.count()
    print(f"First pass: {todo_count - failed_count:,} embedded, {failed_count:,} failed "
          f"in {time.time() - t0:.0f}s")
    if failed_count:
        display(failed.select("chunk_id", "error").limit(5))
        retry = embed(failed.drop("embedding", "error"))
        ok_first = batch.where(f"size(embedding) = {DIMENSIONS}")
        embedded = ok_first.unionByName(retry)
    else:
        embedded = batch
    embedded.createOrReplaceTempView("new_vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Quality checks (blocking, before the write)

# COMMAND ----------

if todo_count:
    s = embedded.agg(
        F.count("*").alias("rows"),
        F.countDistinct("chunk_id").alias("ids"),
        F.sum(F.when(F.col("embedding").isNull() | (F.size("embedding") != DIMENSIONS), 1).otherwise(0)).alias("bad"),
        F.sum(F.when(F.exists("embedding", lambda x: F.isnan(x)), 1).otherwise(0)).alias("nan"),
    ).first()
    checks = {
        "every new chunk embedded": s["rows"] == todo_count and s["ids"] == todo_count,
        f"every vector has {DIMENSIONS} numbers": s["bad"] == 0,
        "no NaN values": s["nan"] == 0,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise Exception("Embedding checks failed: vectors not written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · MERGE into `chunk_vectors`
# MAGIC Vectors for chunks that no longer exist are deleted, so the index never returns stale text.

# COMMAND ----------

if todo_count:
    r = spark.sql(f"""
    MERGE INTO {VECTORS} t
    USING (SELECT chunk_id, doc_id, chunk_index, language, source, title, chunk_text, url, topic,
                  published_date, chunk_hash, embedding FROM new_vectors) s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED THEN UPDATE SET
      t.doc_id = s.doc_id, t.chunk_index = s.chunk_index, t.language = s.language, t.source = s.source,
      t.title = s.title, t.chunk_text = s.chunk_text, t.url = s.url, t.topic = s.topic,
      t.published_date = s.published_date, t.chunk_hash = s.chunk_hash, t.embedding = s.embedding,
      t.model = '{MODEL}', t._embedded_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT
      (chunk_id, doc_id, chunk_index, language, source, title, chunk_text, url, topic, published_date,
       chunk_hash, embedding, model, _embedded_at)
      VALUES (s.chunk_id, s.doc_id, s.chunk_index, s.language, s.source, s.title, s.chunk_text, s.url, s.topic,
              s.published_date, s.chunk_hash, s.embedding, '{MODEL}', current_timestamp())
    """).first().asDict()
    print("MERGE:", r)

# Runs every time, even when nothing was embedded:
#  - labels (language, source, title, url, date…) follow the chunks table WITHOUT re-embedding,
#    e.g. when a language label is corrected upstream, the vector is still valid but its filter value isn't
#  - vectors whose chunk is gone are deleted
META = ["doc_id", "chunk_index", "language", "source", "title", "chunk_text", "url", "topic", "published_date"]
changed = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in META)
d = spark.sql(f"""
MERGE INTO {VECTORS} t
USING (SELECT chunk_id, chunk_hash, {", ".join(META)} FROM {CHUNKS}) s
ON t.chunk_id = s.chunk_id
WHEN MATCHED AND t.chunk_hash = s.chunk_hash AND ({changed}) THEN UPDATE SET {", ".join(f"t.{c} = s.{c}" for c in META)}
WHEN NOT MATCHED BY SOURCE THEN DELETE
""").first().asDict()
print("Labels synced / stale vectors deleted:", d)
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.docs_gold._embed_batch")

v = spark.table(VECTORS)
total, n_chunks = v.count(), chunks.count()
print(f"PASS  vectors match chunks: {total:,} = {n_chunks:,}" if total == n_chunks
      else f"FAIL  vectors {total:,} vs chunks {n_chunks:,}")
assert total == n_chunks, "vector table out of sync with chunks"

mismatched = spark.sql(f"""
SELECT count(*) AS n FROM {VECTORS} t JOIN {CHUNKS} s USING (chunk_id)
WHERE NOT (t.language <=> s.language) OR NOT (t.source <=> s.source) OR t.chunk_hash <> s.chunk_hash
""").first()["n"]
print(f"PASS  labels match chunks" if mismatched == 0 else f"FAIL  {mismatched:,} vectors with stale labels")
assert mismatched == 0, "vector labels out of sync with chunks"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Vector Search index (Delta Sync, self-managed embeddings)
# MAGIC Created once; later runs just trigger a sync, which only processes changed rows.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

def wait(check, what, timeout_min=30):
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        ok, state = check()
        if ok:
            print(f"{what}: ready ({time.time() - t0:.0f}s)")
            return
        print(f"{what}: {state} … waiting")
        time.sleep(30)
    raise TimeoutError(f"{what} not ready after {timeout_min} min")

def endpoint_state():
    st = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state", "UNKNOWN")
    return st == "ONLINE", st

if VS_ENDPOINT not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")   # first run only (~10–20 min to come online)
wait(endpoint_state, f"endpoint {VS_ENDPOINT}")

existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
if INDEX not in existing_indexes:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT, index_name=INDEX, source_table_name=VECTORS,
        pipeline_type="TRIGGERED", primary_key="chunk_id",
        embedding_dimension=DIMENSIONS, embedding_vector_column="embedding")
    print(f"Index {INDEX} created")
else:
    index = vsc.get_index(VS_ENDPOINT, INDEX)
    index.sync()
    print(f"Index {INDEX} sync triggered")

def index_state():
    st = index.describe().get("status", {})
    return bool(st.get("ready")) and st.get("indexed_row_count", 0) >= total, \
        f"{st.get('detailed_state', '?')}, {st.get('indexed_row_count', 0):,}/{total:,} rows"

wait(index_state, f"index {INDEX}", timeout_min=45)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Try it: cross-language search
# MAGIC The question is embedded with the same model (plus the query instruction), then the index returns the
# MAGIC closest chunks. A `language` filter limits results to one newspaper's language.

# COMMAND ----------

import pandas as pd

def embed_query(q):
    return spark.sql(f"SELECT ai_query('{MODEL}', :q) AS v", args={"q": QUERY_PREFIX + q}).first()["v"]

def search(question, k=10, language=None):
    res = index.similarity_search(
        query_vector=embed_query(question), num_results=k,
        columns=["chunk_id", "doc_id", "language", "source", "title", "chunk_text"],
        filters={"language": language} if language else None)
    cols = [c["name"] for c in res["manifest"]["columns"]]
    df = pd.DataFrame(res["result"].get("data_array", []), columns=cols)
    if not df.empty:
        df["snippet"] = df["chunk_text"].str.slice(0, 140)
    return df.drop(columns=["chunk_text"], errors="ignore")

QUESTIONS = [
    "Brexit negotiations between the UK and the European Union",
    "refugees crossing the Mediterranean Sea",
    "football World Cup results",
]
for q in QUESTIONS:
    hits = search(q, k=20)
    print(f"\n=== {q}\nlanguages in top 20: {hits['language'].value_counts().to_dict()}")
    display(hits.head(5)[["language", "source", "score", "title", "snippet"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Same question, Russian newspaper only

# COMMAND ----------

ru = search("refugees crossing the Mediterranean Sea", k=5, language="ru")
display(ru[["language", "source", "score", "title", "snippet"]])

# COMMAND ----------

lang_counts = v.groupBy("language").count().orderBy("language")
display(lang_counts)
print(f"Vectors:          {total:,} × {DIMENSIONS} dimensions ({MODEL})")
print(f"Embedded this run: {todo_count:,}")
print(f"Index:            {INDEX} on {VS_ENDPOINT}")
