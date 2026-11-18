# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold: chunks for retrieval
# MAGIC Splits every masked article into **overlapping, sentence-aligned chunks**: the unit that gets embedded (Session 5)
# MAGIC and handed to the LLM (Session 6). Output: `docs_gold.chunks`.
# MAGIC
# MAGIC | Rule | Value | Why |
# MAGIC |---|---|---|
# MAGIC | Split on | Paragraphs, then sentences | A chunk never cuts a sentence in half |
# MAGIC | Target size | ~150 words, hard cap 220 | Fits multilingual embedding models (≈512 tokens), focused enough to match one topic |
# MAGIC | Overlap | Up to 30 words (whole sentences) | A fact that sits on a chunk boundary is still found |
# MAGIC | Short tail | Merged into the previous chunk | No tiny, meaningless chunks |
# MAGIC | Text used | `text_masked` from `docs_silver.articles_masked` | PII never reaches gold |
# MAGIC
# MAGIC The table is written with **MERGE** on `chunk_id`, and each chunk carries a `chunk_hash`.
# MAGIC Unchanged chunks are left alone, so Session 5 only embeds chunks that are new or changed.
# MAGIC
# MAGIC **Blocking checks:** every article has chunks, no empty or oversized chunks, no broken `[PLACEHOLDER]`,
# MAGIC and **no text lost** (the chunks, minus their overlap, add up to every word of the article).

# COMMAND ----------

import re
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

CATALOG = "workspace"
SOURCE = f"{CATALOG}.docs_silver.articles_masked"
TARGET = f"{CATALOG}.docs_gold.chunks"
SUMMARY = f"{CATALOG}.docs_gold.chunk_summary"

TARGET_WORDS = 150    # close a chunk once it reaches this size
MAX_WORDS = 220       # never exceed this (long sentences are split on words)
OVERLAP_WORDS = 30    # carry up to this many words (whole sentences) into the next chunk
MIN_TAIL_WORDS = 40   # a last chunk smaller than this is merged into the previous one

# One newspaper per MLSUM language
SOURCES = {"de": "Süddeutsche Zeitung", "es": "El País", "fr": "Le Monde",
           "ru": "Moskovsky Komsomolets", "tr": "Internet Haber"}

# The same whitespace definition is used to split words here and to count them in the coverage check
WHITESPACE = r"[ \t\n\x0b\f\r   -     　]+"

# COMMAND ----------

# MAGIC %md
# MAGIC ## The chunking function

# COMMAND ----------

WS_RE = re.compile(WHITESPACE)
# A sentence ends with . ! ? or … (optionally followed by a closing quote or bracket) and then whitespace
SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"»”’')\]]))\s+")


def words_of(s):
    return [w for w in WS_RE.split(s) if w]


def split_sentences(text):
    """Paragraphs, then sentences, as lists of words. Sentences longer than MAX_WORDS are cut into pieces."""
    sentences = []
    for paragraph in re.split(r"\n+", text or ""):
        for sentence in SENTENCE_END.split(paragraph.strip()):
            w = words_of(sentence)
            for i in range(0, len(w), MAX_WORDS):
                sentences.append(w[i:i + MAX_WORDS])
    return sentences


def chunk_article(text):
    """Returns [(chunk_text, word_count, overlap_words)].
    overlap_words = how many leading words repeat the end of the previous chunk."""
    sentences = split_sentences(text)
    chunks = []                      # each item: [list of sentences, overlap_words]
    current, overlap = [], 0

    def size(sents):
        return sum(len(s) for s in sents)

    for s in sentences:
        new_words = size(current) - overlap
        room = MAX_WORDS - size(current)
        if 0 < new_words < MIN_TAIL_WORDS and len(s) > room:
            # a very long sentence would leave this chunk tiny: fill it with the sentence's first words
            current, s = current + [s[:room]], s[room:]
            new_words = size(current) - overlap
        if new_words > 0 and (size(current) >= TARGET_WORDS or size(current) + len(s) > MAX_WORDS):
            chunks.append([current, overlap])
            # carry trailing whole sentences (up to OVERLAP_WORDS) into the next chunk
            carry = []
            for prev in reversed(current):
                if size(carry) + len(prev) > OVERLAP_WORDS:
                    break
                carry.insert(0, prev)
            if size(carry) + len(s) > MAX_WORDS:
                carry = []
            current, overlap = carry, size(carry)
        current = current + [s]

    if size(current) - overlap > 0:
        tail_new = size(current) - overlap
        if chunks and tail_new < MIN_TAIL_WORDS and size(chunks[-1][0]) + tail_new <= MAX_WORDS:
            # merge the short tail into the previous chunk (only its new words)
            prev_sents, prev_overlap = chunks[-1]
            chunks[-1] = [prev_sents + [[w for s in current for w in s][overlap:]], prev_overlap]
        else:
            chunks.append([current, overlap])

    out = []
    for sents, ov in chunks:
        flat = [w for s in sents for w in s]
        out.append((" ".join(flat), len(flat), ov))
    return out


CHUNK_SCHEMA = T.ArrayType(T.StructType([
    T.StructField("chunk_text", T.StringType()),
    T.StructField("word_count", T.IntegerType()),
    T.StructField("overlap_words", T.IntegerType()),
]))
chunk_udf = F.udf(chunk_article, CHUNK_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick test on a sample text

# COMMAND ----------

_sample = ("[PERSON] said the fire started in the attic. " * 40).strip()
for i, (txt, n, ov) in enumerate(chunk_article(_sample)):
    print(f"chunk {i}: {n} words, {ov} overlap · {txt[:70]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk all articles

# COMMAND ----------

articles = spark.table(SOURCE)
source_map = F.create_map(*[F.lit(x) for kv in SOURCES.items() for x in kv])

chunks = (
    articles
    .select(
        "doc_id", "url", "topic", "published_date", "date_precision",
        F.col("language").alias("source_language"),
        # search by the language the text is really in (e.g. 34 English articles from the Spanish file)
        F.when(F.col("detected_language") != "unknown", F.col("detected_language"))
         .otherwise(F.col("language")).alias("language"),
        source_map[F.col("language")].alias("source"),
        F.col("title_masked").alias("title"),
        F.posexplode(chunk_udf("text_masked")).alias("chunk_index", "chunk"))
    .select(
        F.sha2(F.concat_ws("|", "doc_id", F.col("chunk_index").cast("string")), 256).alias("chunk_id"),
        "doc_id", "chunk_index", "language", "source_language", "source", "title",
        F.col("chunk.chunk_text").alias("chunk_text"),
        F.col("chunk.word_count").alias("word_count"),
        F.col("chunk.overlap_words").alias("overlap_words"),
        "url", "topic", "published_date", "date_precision")
    .withColumn("chunks_in_doc", F.count("*").over(Window.partitionBy("doc_id")))
    .withColumn("chunk_hash", F.sha2(F.concat_ws("|", "title", "chunk_text"), 256))
)
chunks.createOrReplaceTempView("new_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality checks (blocking)
# MAGIC Run **before** writing, so a bad run never reaches gold.

# COMMAND ----------

# an opened placeholder that never closes, or a placeholder cut at the end of a chunk
PLACEHOLDER_BROKEN = r"\[(?:PERSON|EMAIL|PHONE|URL|IP|IBAN)(?!\])|\[[A-Z]{1,5}$"

article_count = articles.count()
stats = chunks.agg(
    F.countDistinct("doc_id").alias("docs"),
    F.count("*").alias("chunks"),
    F.sum(F.when(F.col("word_count") == 0, 1).otherwise(0)).alias("empty"),
    F.sum(F.when(F.col("word_count") > MAX_WORDS, 1).otherwise(0)).alias("oversized"),
    F.sum(F.when(F.col("chunk_text").rlike(PLACEHOLDER_BROKEN), 1).otherwise(0)).alias("broken_placeholders"),
).first()

# No text lost: per article, chunk words minus overlap must equal the article's word count
source_words = articles.select(
    "doc_id",
    F.size(F.filter(F.split("text_masked", WHITESPACE), lambda w: w != "")).alias("article_words"))
chunk_words = chunks.groupBy("doc_id").agg(
    F.sum(F.col("word_count") - F.col("overlap_words")).alias("chunk_words"))
lost = (source_words.join(chunk_words, "doc_id", "left")
        .where(F.coalesce("chunk_words", F.lit(0)) != F.col("article_words")))
lost_count = lost.count()

checks = {
    "every article has chunks": stats["docs"] == article_count,
    "no empty chunks": stats["empty"] == 0,
    f"no chunk over {MAX_WORDS} words": stats["oversized"] == 0,
    "no broken placeholders": stats["broken_placeholders"] == 0,
    "no text lost": lost_count == 0,
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{article_count:,} articles → {stats['chunks']:,} chunks")

if not all(checks.values()):
    if lost_count:
        display(lost.limit(10)) if "display" in globals() else lost.show(10, truncate=False)
    raise Exception("Chunk quality checks failed: gold table not written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write with MERGE
# MAGIC New chunks are inserted, changed ones updated (their `chunk_hash` differs), and chunks whose article
# MAGIC disappeared or got shorter are deleted. Unchanged rows aren't touched, which keeps Session 5 incremental.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TARGET} (
  chunk_id STRING, doc_id STRING, chunk_index INT, language STRING, source_language STRING, source STRING,
  title STRING, chunk_text STRING, word_count INT, overlap_words INT, chunks_in_doc BIGINT,
  url STRING, topic STRING, published_date DATE, date_precision STRING, chunk_hash STRING,
  _updated_at TIMESTAMP
) USING DELTA
""")

merge_result = spark.sql(f"""
MERGE INTO {TARGET} t
USING new_chunks s
ON t.chunk_id = s.chunk_id
WHEN MATCHED AND (t.chunk_hash <> s.chunk_hash OR t.chunks_in_doc <> s.chunks_in_doc OR t.language <> s.language) THEN
  UPDATE SET t.doc_id = s.doc_id, t.chunk_index = s.chunk_index, t.language = s.language,
             t.source_language = s.source_language, t.source = s.source, t.title = s.title,
             t.chunk_text = s.chunk_text, t.word_count = s.word_count, t.overlap_words = s.overlap_words,
             t.chunks_in_doc = s.chunks_in_doc, t.url = s.url, t.topic = s.topic,
             t.published_date = s.published_date, t.date_precision = s.date_precision,
             t.chunk_hash = s.chunk_hash, t._updated_at = current_timestamp()
WHEN NOT MATCHED THEN
  INSERT (chunk_id, doc_id, chunk_index, language, source_language, source, title, chunk_text, word_count,
          overlap_words, chunks_in_doc, url, topic, published_date, date_precision, chunk_hash, _updated_at)
  VALUES (s.chunk_id, s.doc_id, s.chunk_index, s.language, s.source_language, s.source, s.title, s.chunk_text,
          s.word_count, s.overlap_words, s.chunks_in_doc, s.url, s.topic, s.published_date, s.date_precision,
          s.chunk_hash, current_timestamp())
WHEN NOT MATCHED BY SOURCE THEN DELETE
""")
m = merge_result.first().asDict()
print("MERGE:", {k: m[k] for k in m})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary per language

# COMMAND ----------

gold = spark.table(TARGET)
summary = (gold.groupBy("language", "source")
           .agg(F.countDistinct("doc_id").alias("articles"),
                F.count("*").alias("chunks"),
                F.round(F.count("*") / F.countDistinct("doc_id"), 1).alias("chunks_per_article"),
                F.round(F.avg("word_count"), 0).cast("int").alias("avg_words"),
                F.min("word_count").alias("min_words"),
                F.max("word_count").alias("max_words"))
           .orderBy("language"))
summary.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(SUMMARY)
display(summary) if "display" in globals() else summary.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: one article's chunks

# COMMAND ----------

example_doc = gold.where("chunks_in_doc = 3").select("doc_id").first()["doc_id"]
example = (gold.where(F.col("doc_id") == example_doc).orderBy("chunk_index")
           .select("chunk_index", "language", "source", "word_count", "overlap_words",
                   F.substring("chunk_text", 1, 160).alias("chunk_start")))
display(example) if "display" in globals() else example.show(truncate=False)

# COMMAND ----------

total = gold.count()
print(f"Articles chunked:   {gold.select('doc_id').distinct().count():,}")
print(f"Chunks in gold:     {total:,}")
print(f"Avg words / chunk:  {gold.agg(F.avg('word_count')).first()[0]:.0f}")
print(f"Quality checks:     {len(checks)}/{len(checks)} passed")
