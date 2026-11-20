# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · RAG search: how did each newspaper cover it?
# MAGIC Ask a question in English → find coverage in **every** newspaper → check it's really on topic →
# MAGIC summarise each outlet in English → compare them, with sources.
# MAGIC
# MAGIC | Step | What happens | Fixes (from the Session 5 test searches) |
# MAGIC |---|---|---|
# MAGIC | 1 · Retrieve per language | One filtered vector search **per newspaper**, not one global top 20 | Russian and Turkish never reached a global top 20 |
# MAGIC | 2 · Group by article | Chunks of the same article collapse into one result (best score, its 2 best chunks as the passage) | One article filled 3 of the top 5 places |
# MAGIC | 3 · Relevance check | A score floor, then the LLM answers YES/NO: *is this article about the question?* | "Closest" isn't "relevant" (handball for football) |
# MAGIC | 4 · Coverage table | Relevant articles per newspaper and their share of that newspaper's articles | Compare coverage fairly across outlets of different sizes |
# MAGIC | 5 · Summaries | One English summary per newspaper, citing its excerpts; then one comparison: answer, agreement, differences | The LLM answers only from retrieved text |
# MAGIC | 6 · Evaluation | 20 auto-generated test questions with a known answer article; global vs per-language retrieval | Measures the fix instead of claiming it |
# MAGIC
# MAGIC Every question and answer is logged to `docs_gold.rag_log`.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, re, time
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
VECTORS = f"{CATALOG}.docs_gold.chunk_vectors"
INDEX = f"{CATALOG}.docs_gold.chunk_index"
VS_ENDPOINT = "docs_vs_endpoint"
LOG = f"{CATALOG}.docs_gold.rag_log"
EVAL = f"{CATALOG}.docs_gold.rag_eval"

EMBED_MODEL = "databricks-qwen3-embedding-0-6b"
QUERY_PREFIX = "Instruct: Given a news question, retrieve news passages that answer it\nQuery: "
LLM = "databricks-meta-llama-3-3-70b-instruct"

LANGUAGES = ["de", "es", "fr", "ru", "tr", "en"]
LANGUAGE_NAMES = {"de": "German", "es": "Spanish", "fr": "French", "ru": "Russian", "tr": "Turkish", "en": "English"}
SOURCES = {"de": "Süddeutsche Zeitung", "es": "El País", "fr": "Le Monde",
           "ru": "Moskovsky Komsomolets", "tr": "Internet Haber", "en": "El País (English edition)"}

CHUNKS_PER_LANGUAGE = 30   # chunks retrieved per newspaper before grouping by article
ARTICLES_TO_CHECK = 6      # best articles per newspaper sent to the relevance check
MIN_SCORE = 0.35           # chunks below this similarity are never considered
PASSAGE_CHARS = 1500       # excerpt length sent to the LLM per article

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building blocks
# MAGIC The pure-Python parts (grouping, parsing, coverage) are kept separate from the Databricks calls so they can be tested.

# COMMAND ----------

def group_by_article(chunks, max_chunks=2):
    """Collapse chunks into articles: best score per article, and its best `max_chunks` chunks (in reading order)
    joined as the passage. Returns articles sorted by score with a 1-based rank."""
    cols = ["doc_id", "language", "source", "title", "url", "published_date", "score", "passage", "rank"]
    if chunks is None or chunks.empty:
        return pd.DataFrame(columns=cols)
    chunks = chunks.sort_values("score", ascending=False)
    best = chunks.drop_duplicates("doc_id").set_index("doc_id")
    top = chunks.groupby("doc_id", sort=False).head(max_chunks).sort_values(["doc_id", "chunk_index"])
    passages = top.groupby("doc_id")["chunk_text"].agg(" … ".join).rename("passage")
    out = best.join(passages).reset_index().sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out[[c for c in cols if c in out.columns]]


def parse_yes_no(text):
    """LLM relevance answers: True for YES, False for NO or anything unreadable (fail closed)."""
    m = re.match(r"\W*(yes|no)\b", (text or "").strip().lower())
    return bool(m and m.group(1) == "yes")


def coverage_row(lang, checked, relevant, total_articles, global_top20):
    shown = f"{relevant}+" if checked and relevant == checked == ARTICLES_TO_CHECK else str(relevant)
    return {"language": lang, "source": SOURCES[lang],
            "articles_checked": checked, "relevant_articles": shown,
            "share_of_outlet_pct": round(100 * relevant / total_articles, 2) if total_articles else 0.0,
            "in_global_top_20": global_top20}

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
index = VectorSearchClient(disable_notice=True).get_index(VS_ENDPOINT, INDEX)

ARTICLES_PER_LANGUAGE = {r["language"]: r["n"] for r in
                         spark.table(VECTORS).groupBy("language").agg(F.countDistinct("doc_id").alias("n")).collect()}
print("Articles per newspaper language:", ARTICLES_PER_LANGUAGE)

SEARCH_COLUMNS = ["chunk_id", "doc_id", "chunk_index", "language", "source", "title", "url", "published_date", "chunk_text"]


def embed_query(question):
    return spark.sql(f"SELECT ai_query('{EMBED_MODEL}', :q) AS v", args={"q": QUERY_PREFIX + question}).first()["v"]


def search_chunks(query_vector, language=None, k=CHUNKS_PER_LANGUAGE):
    res = index.similarity_search(query_vector=query_vector, num_results=k, columns=SEARCH_COLUMNS,
                                  filters={"language": language} if language else None)
    cols = [c["name"] for c in res["manifest"]["columns"]]
    return pd.DataFrame(res["result"].get("data_array") or [], columns=cols)


def llm_batch(prompts, max_tokens=300):
    """Run many prompts through the LLM in one Spark job. Returns answers in order (None for a failed row)."""
    if not prompts:
        return []
    df = spark.createDataFrame(pd.DataFrame({"id": range(len(prompts)), "prompt": prompts}))
    rows = (df.selectExpr("id", f"""ai_query('{LLM}', prompt,
                modelParameters => named_struct('temperature', 0.0, 'max_tokens', {max_tokens}),
                failOnError => false) AS r""")
              .selectExpr("id", "r.result AS answer").orderBy("id").collect())
    return [r["answer"] for r in rows]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompts
# MAGIC `[PERSON]`, `[EMAIL]` … are masked personal data from Session 3: the LLM is told to keep them, never to guess.

# COMMAND ----------

def judge_prompt(question, a):
    return f"""You check search results for a news research tool.
Question: {question}
Article from {a['source']} ({LANGUAGE_NAMES.get(a['language'], a['language'])}). Title: {a['title']}
Excerpt: {str(a['passage'])[:PASSAGE_CHARS]}

Does this article report on the topic of the question, not just a loosely related subject?
Answer with one word: YES or NO."""


def summary_prompt(question, lang, arts):
    excerpts = "\n\n".join(f"[{i}] {a['title']}\n{str(a['passage'])[:PASSAGE_CHARS]}" for i, a in enumerate(arts, 1))
    return f"""You are a news analyst. Using ONLY the excerpts below from {SOURCES[lang]} ({LANGUAGE_NAMES[lang]}),
write 2-3 sentences in English on what this newspaper reported about: "{question}".
Cite the excerpts you use as [1], [2], [3]. Translate any quotation into English. Text like [PERSON] is redacted personal data: keep it, never guess a name.
Do not add anything that is not in the excerpts.

{excerpts}"""


def comparison_prompt(question, summaries, no_coverage):
    body = "\n\n".join(f"{SOURCES[l]}: {s}" for l, s in summaries.items())
    missing = ", ".join(SOURCES[l] for l in no_coverage) or "none"
    return f"""Question: {question}

Below are English summaries of how different newspapers covered this question.
Write, using ONLY these summaries:
Answer: 2-3 sentences answering the question.
Agreement: 1-2 bullet points on what the newspapers agree on.
Differences: 1-2 bullet points on what only some report, or how their framing differs.
Name the newspapers. Newspapers with no relevant coverage: {missing}.

{body}"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## `ask()`: the full RAG flow

# COMMAND ----------

def ask(question, show=True, log=True):
    t0 = time.time()
    qvec = embed_query(question)

    # Plain global search, kept only to show what it would have returned
    global_top = group_by_article(search_chunks(qvec, None, 60)).head(20)
    global_counts = global_top["language"].value_counts().to_dict()

    # 1-2 · per-newspaper retrieval, grouped by article, score floor
    candidates = {}
    for lang in LANGUAGES:
        arts = group_by_article(search_chunks(qvec, lang))
        candidates[lang] = arts[arts["score"] >= MIN_SCORE].head(ARTICLES_TO_CHECK).to_dict("records")

    # 3 · relevance check, all newspapers in one LLM batch
    flat = [(lang, a) for lang in LANGUAGES for a in candidates[lang]]
    verdicts = [parse_yes_no(v) for v in llm_batch([judge_prompt(question, a) for _, a in flat], max_tokens=3)]
    relevant = {lang: [] for lang in LANGUAGES}
    for (lang, a), ok in zip(flat, verdicts):
        if ok:
            relevant[lang].append(a)

    # 4 · coverage table
    coverage = pd.DataFrame([coverage_row(l, len(candidates[l]), len(relevant[l]),
                                          ARTICLES_PER_LANGUAGE.get(l, 0), global_counts.get(l, 0))
                             for l in LANGUAGES])

    # 5 · per-newspaper summaries, then one comparison
    covered = [l for l in LANGUAGES if relevant[l]]
    texts = llm_batch([summary_prompt(question, l, relevant[l][:3]) for l in covered], max_tokens=300)
    summaries = {l: (t or "(summary failed)").strip() for l, t in zip(covered, texts)}
    comparison = (llm_batch([comparison_prompt(question, summaries, [l for l in LANGUAGES if l not in covered])],
                            max_tokens=500)[0] or "(comparison failed)") if summaries else \
        "No newspaper in the archive has relevant coverage of this question."
    sources = [{"language": l, "n": i, "title": a["title"], "source": a["source"], "url": a["url"],
                "score": round(float(a["score"]), 3)}
               for l in covered for i, a in enumerate(relevant[l][:3], 1)]
    seconds = round(time.time() - t0, 1)

    if log:
        spark.createDataFrame([{
            "question": question, "seconds": float(seconds), "coverage_json": coverage.to_json(orient="records", force_ascii=False),
            "summaries_json": json.dumps(summaries, ensure_ascii=False), "comparison": comparison,
            "sources_json": json.dumps(sources, ensure_ascii=False, default=str),
            "global_top20_json": json.dumps(global_counts)}]) \
            .withColumn("asked_at", F.current_timestamp()) \
            .write.mode("append").option("mergeSchema", "true").saveAsTable(LOG)

    if show:
        print(f"Q: {question}   ({seconds}s)")
        print(f"Plain global top 20 would have shown: {global_counts}")
        display(coverage)
        html = [f"<h3>{question}</h3><div style='white-space:pre-wrap'>{comparison}</div><h4>By newspaper</h4>"]
        for l in LANGUAGES:
            html.append(f"<p><b>{SOURCES[l]}</b> ({LANGUAGE_NAMES[l]}): "
                        f"{summaries.get(l, '<i>No relevant coverage found.</i>')}</p>")
            for s in [s for s in sources if s["language"] == l]:
                html.append(f"<div style='margin-left:1em;font-size:90%'>[{s['n']}] "
                            f"<a href='{s['url']}' target='_blank'>{s['title']}</a> · score {s['score']}</div>")
        displayHTML("".join(html))
    return {"coverage": coverage, "summaries": summaries, "comparison": comparison, "sources": sources}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try it

# COMMAND ----------

r1 = ask("How did the Brexit negotiations between the UK and the EU develop?")

# COMMAND ----------

r2 = ask("What happened to refugees and migrants crossing the Mediterranean Sea?")

# COMMAND ----------

r3 = ask("What is the relationship between Russia and Turkey?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Evaluation: does per-language retrieval fix the bias?
# MAGIC For 4 random articles per main language, the LLM writes an **English** question the article answers
# MAGIC (paraphrased, not copied). Then we check where that article ranks:
# MAGIC - **Global:** one plain search over all languages (the Session 5 approach)
# MAGIC - **Per-language:** the search filtered to the article's newspaper (the RAG approach)
# MAGIC
# MAGIC Hit@1 / Hit@5 = the right article is first / in the top 5. **Judge recall** = how often the relevance
# MAGIC check keeps the right article when it's in the top 6 (a check that's too strict would hide real coverage).

# COMMAND ----------

EVAL_LANGUAGES = ["de", "es", "fr", "ru", "tr"]
EVAL_PER_LANGUAGE = 4

picked = (spark.table(VECTORS).where(F.col("language").isin(EVAL_LANGUAGES) & (F.col("chunk_index") == 0))
          .withColumn("h", F.xxhash64("doc_id"))
          .withColumn("r", F.row_number().over(
              Window.partitionBy("language").orderBy("h")))
          .where(F.col("r") <= EVAL_PER_LANGUAGE)
          .select("doc_id", "language", "title", "chunk_text").toPandas())

gen = llm_batch([f"""Here is the start of a {LANGUAGE_NAMES[r.language]} news article.
Title: {r.title}
Text: {r.chunk_text[:1200]}

Write ONE short English question (at most 15 words) that a reader could ask and that this article answers.
Paraphrase in your own words; do not translate the title word for word. Return only the question."""
                 for r in picked.itertuples()], max_tokens=40)
picked["question"] = [(q or "").strip().strip('"') for q in gen]


def rank_of(doc_id, articles):
    hit = articles.index[articles["doc_id"] == doc_id]
    return int(articles.loc[hit[0], "rank"]) if len(hit) else None


rows = []
for r in picked.itertuples():
    qvec = embed_query(r.question)
    glob = group_by_article(search_chunks(qvec, None, 60))
    local = group_by_article(search_chunks(qvec, r.language))
    rows.append({"doc_id": r.doc_id, "language": r.language, "question": r.question,
                 "global_rank": rank_of(r.doc_id, glob), "per_language_rank": rank_of(r.doc_id, local),
                 "passage": local.loc[local["doc_id"] == r.doc_id, "passage"].head(1).tolist(),
                 "title": r.title, "source": SOURCES[r.language]})
ev = pd.DataFrame(rows)

# Judge recall: does the relevance check keep the right article when it was retrieved in the top 6?
in_top6 = ev[ev["per_language_rank"].notna() & (ev["per_language_rank"] <= ARTICLES_TO_CHECK)]
judged = llm_batch([judge_prompt(r.question, {"source": r.source, "language": r.language, "title": r.title,
                                              "passage": r.passage[0]}) for r in in_top6.itertuples()], max_tokens=3)
ev["judge_kept"] = None
ev.loc[in_top6.index, "judge_kept"] = [parse_yes_no(j) for j in judged]

for col, name in [("global_rank", "global"), ("per_language_rank", "per_lang")]:
    ev[f"{name}_hit1"] = ev[col].fillna(999) <= 1
    ev[f"{name}_hit5"] = ev[col].fillna(999) <= 5

spark.createDataFrame(ev.drop(columns=["passage"]).astype({"global_rank": "Int64", "per_language_rank": "Int64",
                                                           "judge_kept": "boolean"}))\
    .withColumn("_evaluated_at", F.current_timestamp())\
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL)

display(ev[["language", "question", "global_rank", "per_language_rank", "judge_kept"]])

# COMMAND ----------

summary = (ev.groupby("language")
           .agg(questions=("question", "count"),
                global_hit5=("global_hit5", "mean"), per_language_hit1=("per_lang_hit1", "mean"),
                per_language_hit5=("per_lang_hit5", "mean"),
                judge_recall=("judge_kept", lambda s: s.dropna().astype(bool).mean() if s.notna().any() else None))
           .reset_index())
total = {"language": "ALL", "questions": len(ev), "global_hit5": ev["global_hit5"].mean(),
         "per_language_hit1": ev["per_lang_hit1"].mean(), "per_language_hit5": ev["per_lang_hit5"].mean(),
         "judge_recall": ev["judge_kept"].dropna().astype(bool).mean()}
summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
for c in ["global_hit5", "per_language_hit1", "per_language_hit5", "judge_recall"]:
    summary[c] = (summary[c].astype(float) * 100).round(0)
display(summary)
print(f"Retrieval hit@5: global {total['global_hit5']:.0%} → per-language {total['per_language_hit5']:.0%} · "
      f"hit@1 per-language {total['per_language_hit1']:.0%} · judge recall {total['judge_recall']:.0%}")
