# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Export data for the public website
# MAGIC The website (GitHub Pages) is static: no server, no model, no API key. Databricks prepares everything it needs:
# MAGIC
# MAGIC | File | What it holds | How it's made |
# MAGIC |---|---|---|
# MAGIC | `articles.json` | Every article: headline, **English headline**, newspaper, date, short snippet, link | Headlines translated once with the LLM (incremental, stored in `docs_gold.article_titles_en`) |
# MAGIC | (inside `articles.json`) | For each article, its **closest articles in every other language** | Article vector = mean of its chunk vectors; cosine similarity across all 9,722 articles |
# MAGIC | `answers.json` | Featured questions with full RAG answers (coverage, summaries, sources) | Latest answer per question from `docs_gold.rag_log` (ask them in `06_rag_search`) |
# MAGIC | `meta.json` | Counts, sources, generation time | |
# MAGIC
# MAGIC Only headlines, ~35-word snippets and links to the original articles are published, all after PII masking.
# MAGIC Files are written to the Volume `docs_gold.site`; download them into the repo's `docs/data/` folder.

# COMMAND ----------

import json, math, os, time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
CHUNKS = f"{CATALOG}.docs_gold.chunks"
VECTORS = f"{CATALOG}.docs_gold.chunk_vectors"
LOG = f"{CATALOG}.docs_gold.rag_log"
TITLES_EN = f"{CATALOG}.docs_gold.article_titles_en"
SITE_DIR = f"/Volumes/{CATALOG}/docs_gold/site"

LLM = "databricks-meta-llama-3-3-70b-instruct"
SNIPPET_WORDS = 35
NEIGHBOURS_PER_LANGUAGE = 3
# Tuned by reading sample pairs per score band (cell "Threshold evidence" below):
# 0.55–0.65 = unrelated or same broad topic; 0.65–0.70 = sometimes the same event; 0.83+ = the same story
SAME_STORY_SCORE = 0.75      # shown as "Same story"
NEIGHBOUR_MIN_SCORE = 0.67   # 0.67–0.75 shown as "Related coverage"; below is not published
PLACEHOLDER = r"\[(?:PERSON|EMAIL|PHONE|URL|IP|IBAN)\]"

LANGUAGES = ["de", "es", "fr", "ru", "tr", "en"]
LANGUAGE_NAMES = {"de": "German", "es": "Spanish", "fr": "French", "ru": "Russian", "tr": "Turkish", "en": "English"}
SOURCES = {"de": "Süddeutsche Zeitung", "es": "El País", "fr": "Le Monde",
           "ru": "Moskovsky Komsomolets", "tr": "Internet Haber", "en": "El País (English edition)"}

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.docs_gold.site")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Articles (one row each, from the first chunk)

# COMMAND ----------

articles = (spark.table(CHUNKS).where("chunk_index = 0")
            .select("doc_id", "language", "source", "title", "url", "topic", "published_date", "date_precision",
                    F.array_join(F.slice(F.split("chunk_text", " "), 1, SNIPPET_WORDS), " ").alias("snippet")))
n_articles = articles.count()
print(f"Articles: {n_articles:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · English headlines (incremental: only new or changed titles are translated)

# COMMAND ----------

spark.sql(f"""CREATE TABLE IF NOT EXISTS {TITLES_EN}
              (doc_id STRING, title_hash STRING, title_en STRING, translated BOOLEAN, _translated_at TIMESTAMP)""")


def placeholders(col):
    """Sorted list of masking placeholders in a text, e.g. ['[PERSON]', '[PERSON]']."""
    return F.array_sort(F.regexp_extract_all(F.coalesce(col, F.lit("")), F.lit(PLACEHOLDER), F.lit(0)))


# Translate titles that are new or changed, plus earlier translations whose placeholders don't match the original
# (the LLM sometimes invented [PERSON] for brands or ordinary words)
old = spark.table(TITLES_EN).select("doc_id", F.col("title_hash").alias("old_hash"), F.col("title_en").alias("old_en"))
todo = (articles.withColumn("title_hash", F.sha2("title", 256)).join(old, "doc_id", "left")
        .where(F.col("old_hash").isNull() | (F.col("old_hash") != F.col("title_hash")) |
               (placeholders(F.col("title")) != placeholders(F.col("old_en"))))
        .select("doc_id", "language", "title", "title_hash"))
todo_count = todo.count()
print(f"Headlines to translate: {todo_count:,}")

LANG_NAME_MAP = F.create_map(*[F.lit(x) for kv in LANGUAGE_NAMES.items() for x in kv])


def translate(df, strict=False):
    """One LLM call per headline. strict=True names the exact placeholders the translation must contain."""
    rule = (F.concat(F.lit("The headline contains exactly these placeholders: "),
                     F.array_join(placeholders(F.col("title")), ", "),
                     F.lit(". Your translation must contain exactly these placeholders and no other square brackets. "))
            if strict else
            F.lit("If the headline contains placeholders in square brackets such as [PERSON], copy them unchanged. "
                  "Never add a placeholder that is not in the headline; translate names, brands and words normally. "))
    prompt = F.concat(F.lit("Translate this "), LANG_NAME_MAP[F.col("language")], F.lit(" news headline into natural English. "),
                      rule, F.lit("Return only the translated headline, nothing else.\n\nHeadline: "), F.col("title"))
    return (df.withColumn("prompt_text", prompt)
              .withColumn("r", F.expr(f"""ai_query('{LLM}', prompt_text,
                    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 60), failOnError => false)"""))
              .withColumn("title_en", F.trim(F.regexp_replace(
                  F.split(F.col("r.result"), "\n").getItem(0),
                  r'^(?:English:|Translation:)?\s*["“”«»]*|["“”«»]*$', "")))
              .select("doc_id", "language", "title", "title_hash", "title_en"))


def is_valid(df):
    return (F.col("title_en").isNotNull() & (F.length("title_en") > 0) &
            (placeholders(F.col("title")) == placeholders(F.col("title_en"))))


if todo_count:
    t0 = time.time()
    english = todo.where("language = 'en'").withColumn("title_en", F.col("title"))
    translate(todo.where("language <> 'en'")).unionByName(english) \
        .write.mode("overwrite").saveAsTable(f"{CATALOG}.docs_gold._titles_batch")
    batch = spark.table(f"{CATALOG}.docs_gold._titles_batch")
    bad = batch.where(~is_valid(batch))
    bad_count = bad.count()
    print(f"First pass: {todo_count - bad_count:,} valid, {bad_count:,} empty or with wrong placeholders → strict retry")
    if bad_count:
        retried = translate(bad.select("doc_id", "language", "title", "title_hash"), strict=True)
        batch = batch.where(is_valid(batch)).unionByName(retried)
    final = batch.select("doc_id", "title_hash",
                         F.when(is_valid(batch), F.col("title_en")).otherwise(F.col("title")).alias("title_en"),
                         is_valid(batch).alias("translated"))
    final.write.mode("overwrite").saveAsTable(f"{CATALOG}.docs_gold._titles_final")
    spark.sql(f"""MERGE INTO {TITLES_EN} t USING {CATALOG}.docs_gold._titles_final s ON t.doc_id = s.doc_id
                  WHEN MATCHED THEN UPDATE SET t.title_hash = s.title_hash, t.title_en = s.title_en,
                                               t.translated = s.translated, t._translated_at = current_timestamp()
                  WHEN NOT MATCHED THEN INSERT (doc_id, title_hash, title_en, translated, _translated_at)
                       VALUES (s.doc_id, s.title_hash, s.title_en, s.translated, current_timestamp())""")
    for tmp in ["_titles_batch", "_titles_final"]:
        spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.docs_gold.{tmp}")
    print(f"Translated in {time.time() - t0:.0f}s")

t = spark.table(TITLES_EN).join(articles.select("doc_id", "language"), "doc_id")
stats = t.agg(F.count("*").alias("n"), F.sum(F.col("translated").cast("int")).alias("ok")).first()
print(f"English headlines: {stats['ok']:,} of {stats['n']:,} translated (the rest keep the original headline)")
check = t.join(articles.select("doc_id", "title"), "doc_id").where(
    "translated AND language <> 'en'").where(placeholders(F.col("title")) != placeholders(F.col("title_en"))).count()
print(f"{'PASS' if check == 0 else 'FAIL'}  translated headlines keep exactly the original placeholders ({check} mismatches)")
assert check == 0
display(t.join(articles.select("doc_id", "title"), "doc_id").where("language <> 'en'")
        .select("language", "title", "title_en").orderBy(F.rand(7)).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Same story in other newspapers
# MAGIC Article vector = normalised mean of its chunk vectors. For each article, the closest articles **in each other
# MAGIC language** are kept if their cosine similarity is at least `NEIGHBOUR_MIN_SCORE`.

# COMMAND ----------

# Averaging 1,024 columns in Spark is slow to plan; collect chunk vectors (40,975 × 1,024 floats ≈ 170 MB) and average in numpy
pdf = spark.table(VECTORS).select("doc_id", "language", "embedding").toPandas()
pdf["embedding"] = pdf["embedding"].map(lambda v: np.asarray(v, dtype=np.float32))
grouped = pdf.groupby(["doc_id", "language"])["embedding"].apply(lambda s: np.mean(np.stack(s.values), axis=0)).reset_index()
del pdf
doc_ids = grouped["doc_id"].tolist()
langs = np.array(grouped["language"].tolist())
M = np.stack(grouped["embedding"].values).astype(np.float32)
M /= np.linalg.norm(M, axis=1, keepdims=True)
print(f"Article vectors: {M.shape}")

neighbours = {d: {} for d in doc_ids}
best_cross = np.zeros(len(doc_ids), dtype=np.float32)
masks = {l: (langs == l) for l in LANGUAGES}
t0 = time.time()
for start in range(0, len(doc_ids), 1000):
    S = M[start:start + 1000] @ M.T
    for row in range(S.shape[0]):
        i = start + row
        own = langs[i]
        top_any = -1.0
        for l in LANGUAGES:
            if l == own or not masks[l].any():
                continue
            idx = np.where(masks[l])[0]
            scores = S[row, idx]
            k = min(NEIGHBOURS_PER_LANGUAGE, len(idx))
            top = idx[np.argpartition(-scores, k - 1)[:k]]
            top = top[np.argsort(-S[row, top])]
            top_any = max(top_any, float(S[row, top[0]]))
            keep = [(doc_ids[j], round(float(S[row, j]), 3)) for j in top if S[row, j] >= NEIGHBOUR_MIN_SCORE]
            if keep:
                neighbours[doc_ids[i]][l] = keep
        best_cross[i] = top_any
print(f"Similarity computed in {time.time() - t0:.0f}s")

q = np.percentile(best_cross, [10, 25, 50, 75, 90]).round(3)
with_match = sum(1 for d in doc_ids if neighbours[d])
print(f"Best cross-language score per article · p10 {q[0]} · p25 {q[1]} · median {q[2]} · p75 {q[3]} · p90 {q[4]}")
same = sum(1 for d in doc_ids if any(s >= SAME_STORY_SCORE for lst in neighbours[d].values() for _, s in lst))
print(f"Articles with a 'Same story' match (≥ {SAME_STORY_SCORE}): {same:,} · with any published match "
      f"(≥ {NEIGHBOUR_MIN_SCORE}): {with_match:,} of {len(doc_ids):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Threshold evidence: random best-match pairs per score band
# MAGIC Read these to check where pairs stop being the same story. Bands below `NEIGHBOUR_MIN_SCORE` aren't kept in
# MAGIC `neighbours`, so they are sampled from the similarity of each article's best match.

# COMMAND ----------

import random
titles = {r["doc_id"]: (r["language"], r["title_en"]) for r in t.select("doc_id", "language", "title_en").collect()}
random.seed(1)
pairs = [(s, a, b) for a in doc_ids for l, lst in neighbours[a].items() for b, s in lst[:1]]
for lo in [0.65, 0.70, 0.75, 0.80]:
    band = [p for p in pairs if lo <= p[0] < lo + 0.05]
    print(f"\n=== {lo:.2f}–{lo + 0.05:.2f}  ({len(band):,} pairs)")
    for s, a, b in random.sample(band, min(6, len(band))):
        print(f"{s:.3f} | {titles[a][0]}: {titles[a][1][:60]}  ⟷  {titles[b][0]}: {titles[b][1][:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity check: the 8 strongest cross-language pairs (should be the same story)

# COMMAND ----------

pairs = sorted({tuple(sorted((d, n))): s for d in doc_ids for l, lst in neighbours[d].items()
                for n, s in lst}.items(), key=lambda kv: -kv[1])[:8]
display(pd.DataFrame([{"score": s, "a_lang": titles[a][0], "a_title_en": titles[a][1],
                       "b_lang": titles[b][0], "b_title_en": titles[b][1]} for (a, b), s in pairs]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Featured RAG answers (latest per question from `rag_log`)

# COMMAND ----------

answers = []
if spark.catalog.tableExists(LOG):
    latest = (spark.table(LOG)
              .withColumn("r", F.row_number().over(Window.partitionBy(F.lower(F.trim("question"))).orderBy(F.desc("asked_at"))))
              .where("r = 1").orderBy("asked_at").collect())
    for r in latest:
        answers.append({
            "question": r["question"], "asked_at": str(r["asked_at"])[:10],
            "comparison": r["comparison"],
            "coverage": json.loads(r["coverage_json"]),
            "summaries": json.loads(r["summaries_json"]),
            "sources": json.loads(r["sources_json"]),
            "global_top20": json.loads(r["global_top20_json"]) if r["global_top20_json"] else {}})
print(f"Featured answers: {len(answers)}")
for a in answers:
    print(" -", a["question"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Write the JSON files

# COMMAND ----------

rows = (articles.join(t.select("doc_id", "title_en"), "doc_id", "left")
        .orderBy("language", F.desc("published_date")).collect())
index_of = {r["doc_id"]: i for i, r in enumerate(rows)}
out = []
for i, r in enumerate(rows):
    d = r["published_date"]
    date = "" if d is None else (d.strftime("%Y-%m") if r["date_precision"] == "month" else d.isoformat())
    out.append({
        "i": i, "l": r["language"], "t": r["title"], "e": r["title_en"] or r["title"], "u": r["url"],
        "d": date, "p": r["snippet"], "c": r["topic"] or "",
        "n": {l: [[index_of[n], s] for n, s in lst if n in index_of] for l, lst in neighbours.get(r["doc_id"], {}).items()}})

meta = {"generated_at": time.strftime("%Y-%m-%d"), "articles": len(out),
        "by_language": {l: sum(1 for a in out if a["l"] == l) for l in LANGUAGES},
        "sources": SOURCES, "language_names": LANGUAGE_NAMES, "neighbour_min_score": NEIGHBOUR_MIN_SCORE,
        "same_story_score": SAME_STORY_SCORE,
        "featured_answers": len(answers)}

os.makedirs(SITE_DIR, exist_ok=True)
for name, obj in [("articles.json", out), ("answers.json", answers), ("meta.json", meta)]:
    path = f"{SITE_DIR}/{name}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{name:14s} {os.path.getsize(path) / 1e6:6.2f} MB  → {path}")

# COMMAND ----------

import re
# the domain must end in a 2+ letter TLD (Latin or Cyrillic), so photo credits like "CC@2.0" are not flagged
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.(?:[A-Za-z]{2,}|[\u0400-\u04FF]{2,})\b")
checks = {
    "every article exported": len(out) == n_articles,
    "every article has an English headline": all(a["e"] for a in out),
    "every article has a link": all(a["u"] for a in out),
    "no unmasked e-mail in snippets": not any(EMAIL.search(a["p"] or "") for a in out),
    "articles.json under 25 MB": os.path.getsize(f"{SITE_DIR}/articles.json") < 25e6,
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
assert all(checks.values()), "export checks failed"
