# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest
# MAGIC Loads raw MLSUM news articles (JSON Lines, one file per language and split) into
# MAGIC `docs_bronze.articles`.
# MAGIC
# MAGIC - Picks the **latest** date folder in the landing zone automatically
# MAGIC - Reads every `*.jsonl` file; the **language** and **split** come from the file name (`de_val.jsonl` → `de`, `val`)
# MAGIC - Takes a **deterministic sample** of up to `SAMPLE_PER_LANGUAGE` articles per language: the same
# MAGIC   articles every run, so results are reproducible
# MAGIC - Keeps the source fields **as they are** (raw strings) and adds audit columns:
# MAGIC   `_ingested_at`, `_source_file`, `_load_date`
# MAGIC - Adds a stable `doc_id` (hash of the URL) used as the key in every later layer

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
LANDING = f"/Volumes/{CATALOG}/docs_bronze/landing/mlsum"
TARGET = f"{CATALOG}.docs_bronze.articles"

SAMPLE_PER_LANGUAGE = 2100   # Russian has only ~1,500 articles, so all of them are kept

# MLSUM labels Turkish as "tu"; use the ISO 639-1 code instead
LANGUAGE_CODES = {"de": "de", "es": "es", "fr": "fr", "ru": "ru", "tu": "tr"}

# COMMAND ----------

folders = sorted(f.name.rstrip("/") for f in dbutils.fs.ls(LANDING) if f.isDir())
load_date = folders[-1]
base = f"{LANDING}/{load_date}"
files = sorted(f.name for f in dbutils.fs.ls(base) if f.name.endswith(".jsonl"))
print(f"Loading from: {base}")
print(f"Files found:  {files}")

# COMMAND ----------

# MAGIC %md ## Read all files, tag language and split from the file name

# COMMAND ----------

raw = (
    spark.read
    .schema("title STRING, text STRING, summary STRING, topic STRING, url STRING, date STRING")
    .json(f"{base}/*.jsonl")
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_file_name", F.col("_metadata.file_name"))
)

lang_map = F.create_map(*[F.lit(x) for kv in LANGUAGE_CODES.items() for x in kv])
raw = (
    raw.withColumn("source_language", F.regexp_extract("_file_name", r"^([a-z]+)_", 1))
       .withColumn("language", lang_map[F.col("source_language")])
       .withColumn("split", F.regexp_extract("_file_name", r"_([a-z]+)\.jsonl$", 1))
)

print("Articles per language in the source files:")
raw.groupBy("language").count().orderBy("language").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deterministic sample per language
# MAGIC Articles are ordered by a hash of their URL, then the first N per language are kept.
# MAGIC Unlike `sample()`, this picks the same articles on every run.

# COMMAND ----------

w = Window.partitionBy("language").orderBy(F.xxhash64("url", "title"))
sampled = (
    raw.withColumn("_rank", F.row_number().over(w))
       .filter(F.col("_rank") <= SAMPLE_PER_LANGUAGE)
       .drop("_rank", "_file_name")
)

bronze = (
    sampled
    .withColumn("doc_id", F.sha2(F.coalesce(F.col("url"), F.col("title"), F.col("text")), 256))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_load_date", F.lit(load_date))
    .select("doc_id", "language", "source_language", "split", "title", "text", "summary",
            "topic", "url", "date", "_ingested_at", "_source_file", "_load_date")
)

(bronze.write.mode("overwrite").format("delta")
       .option("overwriteSchema", "true")
       .saveAsTable(TARGET))

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

result = spark.table(TARGET)
total = result.count()
summary = (result.groupBy("language")
                 .agg(F.count("*").alias("articles"),
                      F.round(F.avg(F.length("text"))).cast("int").alias("avg_text_chars"))
                 .orderBy("language"))
display(summary) if "display" in globals() else summary.show()
print(f"TOTAL: {total:,} articles in {TARGET}")
