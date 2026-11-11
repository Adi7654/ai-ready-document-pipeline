# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: clean, deduplicate, detect language
# MAGIC Turns raw bronze articles into clean, unique, language-verified documents in `docs_silver.articles`.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **Clean** | Strips leftover HTML tags and entities, normalizes whitespace, keeps paragraph breaks, parses dates |
# MAGIC | **Filter** | Rejects empty and very short articles |
# MAGIC | **Exact duplicates** | Same normalized text → keep one |
# MAGIC | **Near-duplicates** | **MinHash + LSH** on 5-word shingles → pairs with estimated Jaccard similarity ≥ 0.8 |
# MAGIC | **Language detection** | Script + stopword scoring, compared against the language label from the source file |
# MAGIC
# MAGIC Every removed article goes to `docs_silver.rejected` with a **reason**, so nothing disappears silently.
# MAGIC Everything is plain Spark SQL: no ML libraries or package installs needed.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
BRONZE = f"{CATALOG}.docs_bronze.articles"
SILVER = f"{CATALOG}.docs_silver.articles"
REJECTED = f"{CATALOG}.docs_silver.rejected"

MIN_TEXT_CHARS = 200        # shorter articles are too thin to be useful for retrieval
SHINGLE_WORDS = 5           # near-duplicate shingles: 5 consecutive words
MAX_WORDS = 2000            # only the first 2,000 words are fingerprinted (bounds the work per article)
NUM_HASHES = 32             # MinHash signature length
BANDS, ROWS = 8, 4          # LSH: 8 bands x 4 rows = 32 hashes
NEAR_DUP_THRESHOLD = 0.8    # estimated Jaccard similarity that counts as a near-duplicate

# COMMAND ----------

# MAGIC %md ## 1. Clean the text

# COMMAND ----------

def clean(col):
    c = F.coalesce(F.col(col), F.lit(""))
    c = F.regexp_replace(c, r"<[^>]+>", " ")                      # leftover HTML tags
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'),
                         ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">")]:
        c = F.regexp_replace(c, entity, char)
    c = F.regexp_replace(c, r"[ \t ]+", " ")                  # runs of spaces and tabs
    c = F.regexp_replace(c, r" *\n *", "\n")                       # spaces around line breaks
    c = F.regexp_replace(c, r"\n{3,}", "\n\n")                     # at most one blank line
    return F.trim(c)


bronze = spark.table(BRONZE)
bronze_count = bronze.count()

cleaned = (
    bronze
    .withColumn("title_clean", clean("title"))
    .withColumn("text_clean", clean("text"))
    .withColumn("text_chars", F.length("text_clean"))
    # German and Turkish sources only record the month ("00/02/2019"): use the 1st and flag it
    .withColumn("date_precision",
                F.when(F.col("date").startswith("00/"), "month")
                 .when(F.col("date").rlike(r"^\d{2}/\d{2}/\d{4}$"), "day"))
    .withColumn("published_date",
                F.expr("to_date(try_to_timestamp(regexp_replace(date, '^00/', '01/'), 'dd/MM/yyyy'))"))
    .withColumn("date_precision", F.when(F.col("published_date").isNotNull(), F.col("date_precision")))
)

# COMMAND ----------

# MAGIC %md ## 2. Reject empty and very short articles

# COMMAND ----------

with_reason = cleaned.withColumn(
    "reject_reason",
    F.when(F.col("text_chars") == 0, "empty_text")
     .when(F.col("text_chars") < MIN_TEXT_CHARS, "too_short"),
)
rejected_quality = with_reason.filter("reject_reason IS NOT NULL")
kept = with_reason.filter("reject_reason IS NULL").drop("reject_reason")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exact duplicates
# MAGIC Same text after lower-casing and removing all non-letters/digits → same `content_hash`. Keep the lowest `doc_id`.

# COMMAND ----------

kept = kept.withColumn(
    "content_hash",
    F.sha2(F.regexp_replace(F.lower("text_clean"), r"[^\p{L}\p{N}]+", ""), 256),
)
w = Window.partitionBy("content_hash").orderBy("doc_id")
kept = kept.withColumn("_dup_rank", F.row_number().over(w))
rejected_exact = kept.filter("_dup_rank > 1").withColumn("reject_reason", F.lit("exact_duplicate"))
kept = kept.filter("_dup_rank = 1").drop("_dup_rank")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Near-duplicates: MinHash + LSH
# MAGIC 1. Split each article into overlapping **5-word shingles**.
# MAGIC 2. **MinHash signature:** for each of 32 seeded hash functions, keep the smallest shingle hash.
# MAGIC    Two articles' signatures agree on a position with probability equal to their Jaccard similarity.
# MAGIC 3. **LSH banding:** split the signature into 8 bands of 4. Articles sharing any whole band become
# MAGIC    **candidate pairs**, so we never compare all pairs (about 49M for 10K articles).
# MAGIC 4. **Estimate similarity** for candidates only; ≥ 0.8 → near-duplicate. The article with the higher `doc_id` is removed.

# COMMAND ----------

words = F.slice(
    F.filter(F.split(F.regexp_replace(F.lower("text_clean"), r"[^\p{L}\p{N}]+", " "), " "),
             lambda x: x != ""),
    1, MAX_WORDS,
)
shingles = F.expr(
    f"transform(sequence(1, greatest(size(_w) - {SHINGLE_WORDS} + 1, 1)), "
    f"i -> concat_ws(' ', slice(_w, i, {SHINGLE_WORDS})))"
)
def seeded_hash(seed):
    # A factory, not `lambda s, seed=seed`: PySpark passes (element, index) to two-argument lambdas
    return lambda s: F.xxhash64(s, F.lit(seed))

signature = F.array(*[
    F.array_min(F.transform(F.col("_sh"), seeded_hash(seed))) for seed in range(NUM_HASHES)
])

sigs = (kept.select("doc_id", "language", "text_clean")
            .withColumn("_w", words)
            .withColumn("_sh", shingles)
            .withColumn("sig", signature)
            .select("doc_id", "language", "sig"))

bands = sigs.select(
    "doc_id", "language", "sig",
    F.posexplode(F.array(*[
        F.xxhash64(F.slice("sig", b * ROWS + 1, ROWS)) for b in range(BANDS)
    ])).alias("band", "band_key"),
)

a, b = bands.alias("a"), bands.alias("b")
candidates = (
    a.join(b, (F.col("a.language") == F.col("b.language")) &
              (F.col("a.band") == F.col("b.band")) &
              (F.col("a.band_key") == F.col("b.band_key")) &
              (F.col("a.doc_id") < F.col("b.doc_id")))
     .select(F.col("a.doc_id").alias("keep_id"), F.col("b.doc_id").alias("drop_id"),
             F.col("a.sig").alias("sig_a"), F.col("b.sig").alias("sig_b"))
     .dropDuplicates(["keep_id", "drop_id"])
)
pairs = (
    candidates
    .withColumn("similarity", F.expr(
        f"size(filter(sequence(1, {NUM_HASHES}), i -> element_at(sig_a, i) = element_at(sig_b, i))) / {NUM_HASHES}"))
    .filter(F.col("similarity") >= NEAR_DUP_THRESHOLD)
    .select("keep_id", "drop_id", "similarity")
)

near_dup_ids = pairs.groupBy("drop_id").agg(
    F.first("keep_id").alias("duplicate_of"), F.max("similarity").alias("similarity"))
rejected_near = (kept.join(near_dup_ids, kept.doc_id == near_dup_ids.drop_id)
                     .drop("drop_id", "duplicate_of", "similarity")
                     .withColumn("reject_reason", F.lit("near_duplicate")))
kept = kept.join(near_dup_ids, kept.doc_id == near_dup_ids.drop_id, "left_anti")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Language detection
# MAGIC - **Russian:** share of Cyrillic letters ≥ 30%
# MAGIC - **German, Spanish, French, Turkish:** a score per language from two signals:
# MAGIC   - **stopwords:** hits from a list of very common, language-specific words (words shared between
# MAGIC     these languages, like "de" or "la", are left out)
# MAGIC   - **distinctive letters**, each worth 2 points: Turkish ı ğ ş İ, Spanish ñ ¿ ¡, German ä ß,
# MAGIC     French è ê œ ù. Turkish attaches grammar to word endings, so it has fewer separate stopwords;
# MAGIC     the letters catch short Turkish and Spanish articles that stopwords alone miss.
# MAGIC - The highest score wins. A total score below 5 → `unknown`. `lang_confidence` = winning score ÷ total score.
# MAGIC
# MAGIC The detected language is then compared with the label from the source file: a free quality check.

# COMMAND ----------

STOPWORDS = {
    "de": ["der", "die", "und", "das", "nicht", "ist", "ein", "eine", "mit", "auch", "sich", "auf",
           "den", "dem", "wird", "wurde", "werden", "für", "von", "zu", "im", "bei", "nach", "sie"],
    "es": ["el", "los", "las", "del", "que", "y", "por", "para", "con", "una", "al", "como",
           "más", "pero", "sus", "fue", "ha", "este", "esta", "también", "entre", "sobre"],
    "fr": ["le", "les", "et", "est", "une", "du", "pour", "dans", "qui", "sur", "pas", "au",
           "avec", "ce", "sont", "aux", "été", "plus", "cette", "mais", "il", "elle"],
    "tr": ["ve", "bir", "bu", "için", "ile", "olarak", "olan", "gibi", "daha", "çok", "ancak",
           "kadar", "sonra", "değil", "her", "ise", "göre", "olduğu", "yapılan", "üzerinde"],
}

sample_words = F.split(
    F.regexp_replace(F.lower(F.substring("text_clean", 1, 3000)), r"[^\p{L}]+", " "), " ")
lang_words = kept.withColumn("_lw", sample_words)

def is_stopword(stop):
    stop_array = F.array(*[F.lit(x) for x in stop])
    return lambda w: F.array_contains(stop_array, w)

# Letters that occur in only one of these four languages (Turkish also uses ö ü ç, so those are left out)
DISTINCT_LETTERS = {"de": "äÄß", "es": "ñÑ¿¡", "fr": "èêœùÈÊŒ", "tr": "ıİğĞşŞ"}
sample_text = F.substring("text_clean", 1, 3000)

for lang, stop in STOPWORDS.items():
    letters = F.length(F.regexp_replace(sample_text, f"[^{DISTINCT_LETTERS[lang]}]", ""))
    lang_words = lang_words.withColumn(
        f"_hits_{lang}", F.size(F.filter("_lw", is_stopword(stop))) + 2 * letters)

hit_cols = [f"_hits_{l}" for l in STOPWORDS]
cyr_share = (F.length(F.regexp_replace(F.substring("text_clean", 1, 3000), r"[^\p{IsCyrillic}]", ""))
             / F.greatest(F.length(F.regexp_replace(F.substring("text_clean", 1, 3000), r"[^\p{L}]", "")), F.lit(1)))
best_hits = F.greatest(*[F.col(c) for c in hit_cols])
total_hits = sum(F.col(c) for c in hit_cols)
best_lang = F.coalesce(*[F.when(F.col(f"_hits_{l}") == best_hits, F.lit(l)) for l in STOPWORDS])

detected = (
    lang_words
    .withColumn("_cyr", cyr_share)
    .withColumn("detected_language",
                F.when(F.col("_cyr") >= 0.3, "ru")
                 .when(total_hits < 5, "unknown")
                 .otherwise(best_lang))
    .withColumn("lang_confidence",
                F.round(F.when(F.col("_cyr") >= 0.3, F.col("_cyr"))
                         .otherwise(best_hits / F.greatest(total_hits, F.lit(1))), 3))
    .withColumn("language_match", F.col("detected_language") == F.col("language"))
    .drop("_lw", "_cyr", *hit_cols)
)

# COMMAND ----------

# MAGIC %md ## 6. Write silver and rejected tables

# COMMAND ----------

silver = detected.select(
    "doc_id", "language", "detected_language", "lang_confidence", "language_match",
    F.col("title_clean").alias("title"), F.col("text_clean").alias("text"),
    "text_chars", F.size(F.split("text_clean", r"\s+")).alias("word_count"),
    "topic", "url", "published_date", "date_precision", "content_hash", "split",
    "_ingested_at", "_load_date",
).withColumn("_processed_at", F.current_timestamp())

(silver.write.mode("overwrite").format("delta")
       .option("overwriteSchema", "true").saveAsTable(SILVER))

reject_cols = ["doc_id", "language", "reject_reason", "text_chars", "url"]
rejected = (rejected_quality.select(*reject_cols)
            .unionByName(rejected_exact.select(*reject_cols))
            .unionByName(rejected_near.select(*reject_cols))
            .withColumn("_processed_at", F.current_timestamp()))
(rejected.write.mode("overwrite").format("delta")
         .option("overwriteSchema", "true").saveAsTable(REJECTED))

# COMMAND ----------

# MAGIC %md ## 7. Summary

# COMMAND ----------

s = spark.table(SILVER)
r = spark.table(REJECTED)
silver_count, rejected_count = s.count(), r.count()

per_language = (
    spark.table(BRONZE).groupBy("language").agg(F.count("*").alias("bronze"))
    .join(r.groupBy("language").pivot("reject_reason",
          ["empty_text", "too_short", "exact_duplicate", "near_duplicate"]).count(), "language", "left")
    .join(s.groupBy("language").agg(
          F.count("*").alias("silver"),
          F.count(F.when(~F.col("language_match"), True)).alias("lang_mismatch")), "language", "left")
    .fillna(0).orderBy("language")
)
display(per_language) if "display" in globals() else per_language.show()

mismatch = s.filter("NOT language_match").count()
print("Language mismatches (label → detected):")
(s.filter("NOT language_match").groupBy("language", "detected_language").count()
  .orderBy(F.desc("count")).show())
print(f"Bronze articles:        {bronze_count:,}")
print(f"Rejected:               {rejected_count:,}")
print(f"Silver articles:        {silver_count:,}")
print(f"Language label matches: {silver_count - mismatch:,} / {silver_count:,} "
      f"({100 * (silver_count - mismatch) / max(silver_count, 1):.1f}%)")
dated = s.filter("published_date IS NOT NULL").count()
day_precision = s.filter(F.col("date_precision") == "day").count()
month_precision = s.filter(F.col("date_precision") == "month").count()
print(f"Articles with a date:   {dated:,} / {silver_count:,} "
      f"(day precision: {day_precision:,}, month precision: {month_precision:,})")
