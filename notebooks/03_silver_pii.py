# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver: PII masking
# MAGIC Masks personal data in every article before it can reach chunks, embeddings or an LLM.
# MAGIC Output: `docs_silver.articles_masked`.
# MAGIC
# MAGIC | PII type | Method | Replaced with |
# MAGIC |---|---|---|
# MAGIC | Email addresses | Pattern | `[EMAIL]` |
# MAGIC | URLs | Pattern | `[URL]` |
# MAGIC | IP addresses | Pattern | `[IP]` |
# MAGIC | IBANs (bank accounts) | Pattern | `[IBAN]` |
# MAGIC | Phone numbers | Pattern + at least 9 digits; dates, years and plain numbers excluded; "call/tel" context decides ambiguous cases | `[PHONE]` |
# MAGIC | Person names | **spaCy multilingual NER** (`xx_ent_wiki_sm`) + Turkish title rule; spans that aren't capitalised like a name are rejected (and counted) | `[PERSON]` |
# MAGIC
# MAGIC Then it **verifies** the result:
# MAGIC 1. **Residual check:** the patterns run again on the masked text; any hit fails the job.
# MAGIC 2. **Independent audit:** Databricks `ai_mask()` (an LLM) reads a small sample of masked articles.
# MAGIC    Anything it still masks is PII the pipeline missed, so we get a measured miss rate per language.

# COMMAND ----------

# MAGIC %pip install spacy https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import re
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
SOURCE = f"{CATALOG}.docs_silver.articles"
TARGET = f"{CATALOG}.docs_silver.articles_masked"
SUMMARY = f"{CATALOG}.docs_silver.pii_summary"
AUDIT = f"{CATALOG}.docs_silver.pii_audit"

AUDIT_PER_LANGUAGE = 6      # ai_mask is rate-limited (built for < 100 rows), so the audit is a small sample
AUDIT_SNIPPET_CHARS = 1200  # the audit reads the first part of each sampled article
PARTITIONS = 16             # spread the NER work across workers

PII_TYPES = ["email", "url", "ip", "iban", "phone", "person"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## The masking function
# MAGIC Patterns run first (they're exact), then the NER model looks for names in what's left.
# MAGIC Each article's counts per PII type are kept, for the pipeline-health dashboard.

# COMMAND ----------

PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "url":   r"(?i)\b(?:https?://|www\.)[^\s<>\"')]+",
    "ip":    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "iban":  r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b",
    # not inside a longer word/number; a trailing "." or "," is fine (end of sentence), "3.5" is not
    "phone": r"(?<![\w/.,-])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}(?![\w/]|[.,]\d)",
}
DATE_LIKE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$|^\d{4}[./-]\d{1,2}[./-]\d{1,2}$")
NUMBER_RANGE = re.compile(r"^\d{4}\s*[-–]\s*\d{4}$")                    # "2018-2019"
THOUSANDS = re.compile(r"^\d{1,3}(?:[ .,\u00A0\u202F]\d{3})+$")           # "1 500 000", "1.500.000"
YEAR_LIST = re.compile(r"^(?:(?:19|20)\d{2}[\s.,-]*)+$")                  # "2017 2018 2019"
# Words that mean "phone"/"call" just before a number decide ambiguous cases like Spanish "912 345 678"
PHONE_CONTEXT = re.compile(
    r"(?i)(tel\b|tel\.|telefon|teléfono|téléphone|phone|call|llame|llamar|appel|anrufen|"
    r"numara|ara(yın|yin)?\b|тел|телефон|звоните|whatsapp|fax|móvil|mobile|handy)")

# Turkish news puts a title in front of names ("Cumhurbaşkanı Recep Tayyip Erdoğan"), and the
# model (trained on Wikipedia in other languages) then often tags the whole phrase as an organisation.
TURKISH_TITLES = ["Cumhurbaşkanı", "Başbakan", "Bakan", "Bakanı", "Başkan", "Başkanı", "Sayın",
                  "Milletvekili", "Genel Başkanı", "Valisi", "Belediye Başkanı"]
TITLE_PREFIX = re.compile(r"^(?:(?:%s)\s+)+" % "|".join(sorted(map(re.escape, TURKISH_TITLES), key=len, reverse=True)))


def _is_phone(match_text, before=""):
    t = match_text.strip()
    digits = re.sub(r"\D", "", t)
    if len(digits) < 9 or DATE_LIKE.match(t) or NUMBER_RANGE.match(t) or YEAR_LIST.match(t):
        return False
    if t.startswith("+"):
        return True                                  # international format: always a phone
    if PHONE_CONTEXT.search(before):
        return True                                  # "Llame al 912 345 678"
    return not THOUSANDS.match(t)                    # "1 500 000" with no phone context: a number


def mask_patterns(text, counts):
    """Replace pattern-based PII; add the number of replacements per type to `counts`."""
    for kind in ["email", "url", "ip", "iban"]:
        text, n = re.subn(PATTERNS[kind], f"[{kind.upper()}]", text)
        counts[kind] += n

    def phone_sub(m):
        before = m.string[max(0, m.start() - 30):m.start()]
        if _is_phone(m.group(0), before):
            counts["phone"] += 1
            return "[PHONE]"
        return m.group(0)
    text = re.sub(PATTERNS["phone"], phone_sub, text)
    return text


# Names in all five languages are capitalised. A "person" whose first or last word is lowercase
# ("yıla çıkarıldı") is a model error, mostly in Turkish, which the model was not trained on.
NAME_PARTICLES = {"de", "del", "la", "las", "los", "da", "do", "dos", "di", "du", "le", "van",
                  "von", "der", "den", "y", "e", "bin", "ibn", "al", "el", "ben", "st", "zu"}


def looks_like_name(text):
    words = [w for w in re.split(r"[\s\-]+", text.strip()) if w]
    if not words:
        return False
    def cap(w):
        w = w.lstrip("'\"«»„“(")
        return bool(w) and w[0].isupper()
    # last word capitalised; every other word capitalised or a name particle ("Charles de Gaulle", "de Gaulle")
    return cap(words[-1]) and all(cap(w) or w.lower() in NAME_PARTICLES for w in words)


def person_spans(doc, language, counts=None):
    """Character spans of person names in a spaCy doc."""
    spans = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            if looks_like_name(ent.text):
                spans.append((ent.start_char, ent.end_char))
            elif counts is not None:
                counts["person_rejected"] += 1
        elif language == "tr" and ent.label_ in ("ORG", "MISC"):
            m = TITLE_PREFIX.match(ent.text)
            rest = ent.text[m.end():] if m else ""
            # a title followed by 2-4 capitalised words is a person ("Cumhurbaşkanı Recep Tayyip Erdoğan")
            if m and 2 <= len(rest.split()) <= 4 and all(w[:1].isupper() for w in rest.split()):
                spans.append((ent.start_char + m.end(), ent.end_char))
    return spans


def mask_persons(doc, text, language, counts):
    spans = person_spans(doc, language, counts)
    counts["person"] += len(spans)
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + "[PERSON]" + text[end:]
    return text


def mask_partition(batches):
    """mapInPandas: loads the model once per worker, then masks every article in each batch."""
    import spacy
    nlp = spacy.load("xx_ent_wiki_sm")
    for pdf in batches:
        rows = []
        titles, texts, langs = [], [], []
        for t, x, lang in zip(pdf["title"].fillna(""), pdf["text"].fillna(""), pdf["nlp_language"]):
            counts = dict.fromkeys(PII_TYPES + ["person_rejected"], 0)
            t = mask_patterns(t, counts)
            x = mask_patterns(x, counts)
            rows.append(counts)
            titles.append(t); texts.append(x); langs.append(lang)
        title_docs = nlp.pipe(titles, batch_size=64)
        text_docs = nlp.pipe(texts, batch_size=16)
        out_titles, out_texts = [], []
        for counts, t, x, lang, td, xd in zip(rows, titles, texts, langs, title_docs, text_docs):
            out_titles.append(mask_persons(td, t, lang, counts))
            out_texts.append(mask_persons(xd, x, lang, counts))
        result = pdf[["doc_id"]].copy()
        result["title_masked"] = out_titles
        result["text_masked"] = out_texts
        for kind in PII_TYPES:
            result[f"pii_{kind}"] = [c[kind] for c in rows]
        result["person_spans_rejected"] = [c["person_rejected"] for c in rows]
        yield result


OUT_SCHEMA = ("doc_id STRING, title_masked STRING, text_masked STRING, "
              + ", ".join(f"pii_{k} INT" for k in PII_TYPES) + ", person_spans_rejected INT")

# COMMAND ----------

# MAGIC %md ## Mask all articles

# COMMAND ----------

articles = spark.table(SOURCE)
# Use the detected language where it is known (e.g. English articles that were labeled Spanish)
articles = articles.withColumn(
    "nlp_language",
    F.when(F.col("detected_language") != "unknown", F.col("detected_language")).otherwise(F.col("language")))

masked = (articles.select("doc_id", "title", "text", "nlp_language")
          .repartition(PARTITIONS, "doc_id")
          .mapInPandas(mask_partition, OUT_SCHEMA))

result = (
    articles.select("doc_id", "language", "detected_language", "topic", "url",
                    "published_date", "date_precision", "word_count", "_load_date")
    .join(masked, "doc_id")
    .withColumn("pii_total", sum(F.col(f"pii_{k}") for k in PII_TYPES))
    .withColumn("_processed_at", F.current_timestamp())
)
(result.write.mode("overwrite").format("delta")
       .option("overwriteSchema", "true").saveAsTable(TARGET))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 1: residual patterns (blocking)
# MAGIC The same patterns run again on the masked text. Emails, URLs, IPs and IBANs must be **zero**;
# MAGIC otherwise the job fails before anything downstream uses the data.

# COMMAND ----------

m = spark.table(TARGET)
residual = {}
for kind in ["email", "url", "ip", "iban"]:
    hits = F.size(F.regexp_extract_all(F.concat_ws(" ", "title_masked", "text_masked"),
                                       F.lit(PATTERNS[kind]), F.lit(0)))
    residual[kind] = m.select(hits.alias("n")).agg(F.sum("n")).first()[0] or 0
print("Residual pattern hits after masking:", residual)
if any(residual.values()):
    raise Exception(f"Unmasked PII remains: {residual}")

article_count = spark.table(SOURCE).count()
masked_count = m.count()
if masked_count != article_count:
    raise Exception(f"Row count mismatch: {article_count} articles in, {masked_count} masked out")
print(f"Row counts match: {masked_count:,} articles")

# COMMAND ----------

# MAGIC %md ## PII summary per language

# COMMAND ----------

summary = (m.groupBy("language")
            .agg(F.count("*").alias("articles"),
                 *[F.sum(f"pii_{k}").alias(k) for k in PII_TYPES],
                 F.sum("pii_total").alias("total"),
                 F.sum("person_spans_rejected").alias("person_spans_rejected"),
                 F.round(F.avg("pii_total"), 1).alias("per_article"))
            .orderBy("language"))
summary.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(SUMMARY)
display(summary) if "display" in globals() else summary.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 2: independent audit with `ai_mask()`
# MAGIC An LLM reads the start of a few masked articles per language and masks any person, email or phone
# MAGIC it still finds. It also re-masks our own placeholders (`[PERSON]` becomes `[MASKED]`), so a real
# MAGIC **miss** = `[MASKED]` count − placeholders it overwrote. This gives a measured miss rate instead of an
# MAGIC assumed one. Not blocking: it's a sample.

# COMMAND ----------

sample = (m.withColumn("_r", F.row_number().over(
                Window.partitionBy("language").orderBy(F.xxhash64("doc_id"))))
           .filter(F.col("_r") <= AUDIT_PER_LANGUAGE)
           .select("doc_id", "language",
                   F.substring("text_masked", 1, AUDIT_SNIPPET_CHARS).alias("snippet")))
sample.createOrReplaceTempView("audit_sample")

audit = spark.sql("""
    SELECT doc_id, language, snippet,
           ai_mask(snippet, array('person', 'email', 'phone')) AS llm_masked
    FROM audit_sample
""")
PLACEHOLDER = r"\[(?:PERSON|EMAIL|PHONE|URL|IP|IBAN)\]"
count_of = lambda col, pattern: F.size(F.regexp_extract_all(F.col(col), F.lit(pattern), F.lit(0)))
audit = (audit
    .withColumn("llm_masks", count_of("llm_masked", r"\[MASKED\]"))
    .withColumn("placeholders_overwritten",
                count_of("snippet", PLACEHOLDER) - count_of("llm_masked", PLACEHOLDER))
    .withColumn("missed", F.greatest(F.col("llm_masks") - F.col("placeholders_overwritten"), F.lit(0)))
    .withColumn("_processed_at", F.current_timestamp()))
audit.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(AUDIT)

audit_summary = (spark.table(AUDIT).groupBy("language")
                 .agg(F.count("*").alias("sampled"),
                      F.sum("missed").alias("missed_items"),
                      F.sum(F.when(F.col("missed") > 0, 1).otherwise(0)).alias("articles_with_a_miss"))
                 .orderBy("language"))
display(audit_summary) if "display" in globals() else audit_summary.show()

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

totals = m.agg(*[F.sum(f"pii_{k}").alias(k) for k in PII_TYPES]).first().asDict()
a = spark.table(AUDIT).agg(F.count("*").alias("n"), F.sum("missed").alias("missed"),
                           F.sum(F.when(F.col("missed") > 0, 1).otherwise(0)).alias("with_miss")).first()
print(f"Articles masked:        {masked_count:,}")
print(f"PII items masked:       {sum(totals.values()):,}  " +
      ", ".join(f"{k}: {v:,}" for k, v in totals.items()))
print(f"Residual pattern hits:  {sum(residual.values())} (blocking check passed)")
print(f"ai_mask audit:          {a['missed'] or 0} missed items in {a['with_miss'] or 0} of {a['n']} sampled articles")
