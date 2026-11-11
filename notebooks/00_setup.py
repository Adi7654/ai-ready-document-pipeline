# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog objects for the document pipeline:
# MAGIC - Schemas: `docs_bronze`, `docs_silver`, `docs_gold` (separate from the retail project's schemas)
# MAGIC - Volume: `docs_bronze.landing`, the landing zone for raw files
# MAGIC - A date-stamped folder for today's load: `/Volumes/<catalog>/docs_bronze/landing/mlsum/<YYYY-MM-DD>`
# MAGIC
# MAGIC Safe to re-run: every statement uses `IF NOT EXISTS`.

# COMMAND ----------

from datetime import date

CATALOG = "workspace"   # default catalog in Databricks Free Edition

# COMMAND ----------

for schema in ["docs_bronze", "docs_silver", "docs_gold"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"Schema ready: {CATALOG}.{schema}")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.docs_bronze.landing")
print(f"Volume ready: {CATALOG}.docs_bronze.landing")

# COMMAND ----------

load_date = date.today().isoformat()
landing_path = f"/Volumes/{CATALOG}/docs_bronze/landing/mlsum/{load_date}"
dbutils.fs.mkdirs(landing_path)
print(f"Upload the MLSUM .jsonl files to: {landing_path}")
