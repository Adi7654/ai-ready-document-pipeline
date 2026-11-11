# AI-Ready Document Pipeline on Databricks

A data engineering pipeline that turns raw multilingual news articles into **clean, deduplicated, PII-masked, chunked and embedded** Delta tables, ready for retrieval-augmented generation (RAG). Built on **Databricks** with **PySpark**, **Delta Lake** and **Unity Catalog**.

> 🚧 Status: in progress. Sections marked _TODO_ are filled in as each stage is completed.

---

## Why this project

LLM applications are only as good as the data they retrieve. Raw documents arrive duplicated, inconsistently formatted, in several languages, and containing personal data that must not reach a model or a vector index. This pipeline applies the same data engineering discipline as an analytics lakehouse (layers, quality checks, lineage, incremental processing) to **unstructured text for AI**.

## Architecture

```
MLSUM .jsonl files ──► Unity Catalog Volume (landing zone)
                            │  01_bronze_ingest: raw articles + audit columns
                            ▼
                   docs_bronze.articles   (raw)
                            │  02_silver_clean: clean, exact + near-duplicate removal (MinHash/LSH), language detection
                            ▼
                   docs_silver.articles   (clean)  +  docs_silver.rejected (reasons)
                            │  03_silver_pii: mask emails, URLs, IPs, IBANs, phones (patterns) + names (spaCy NER),
                            │                 residual check (blocking) + LLM audit with ai_mask()
                            ▼
                   docs_silver.articles_masked  +  pii_summary  +  pii_audit
                            │  TODO: chunking
                            ▼
                   docs_gold.*            (chunks + embeddings) ──► vector search ──► RAG
```

## Dataset

[MLSUM](https://github.com/recitalAI/MLSUM): real news articles in **5 languages**: German (de), Spanish (es), French (fr), Russian (ru) and Turkish (tr). Each article has a title, full text, summary, topic, URL and date.

This project uses a **deterministic sample of ~10K articles**: up to 2,100 per language from the validation splits, plus all Russian validation and test articles (~1,500). MLSUM is licensed for **non-commercial research use only**, so the raw files are not stored in this repository. See *How to run*.

## Layers

### Bronze: raw ingestion
- Raw `.jsonl` files land in a Unity Catalog Volume: `/Volumes/workspace/docs_bronze/landing/mlsum/<load_date>/`.
- `01_bronze_ingest` reads every file, takes the **language** and **split** from the file name, and keeps a **deterministic sample per language** (ordered by a hash of the URL), so every run picks the same articles.
- Source fields are kept as they are. Each article gets a stable `doc_id` (SHA-256 of its URL) plus audit columns `_ingested_at`, `_source_file`, `_load_date`.
- **Articles ingested: 9,907**

| Language | Articles | Avg. text length (chars) |
|---|---|---|
| German (de) | 2,100 | 3,691 |
| Spanish (es) | 2,100 | 4,507 |
| French (fr) | 2,100 | 3,751 |
| Russian (ru) | 1,507 (all available) | 6,624 |
| Turkish (tr) | 2,100 | 1,620 |

Turkish articles are about 4× shorter than Russian ones, which affects how many chunks each language produces later.

![Bronze summary](docs/bronze_summary.png)

### Silver: clean, deduplicate, detect language
`02_silver_clean` turns 9,907 raw articles into **9,722 clean, unique, language-verified documents**. Every removed article goes to `docs_silver.rejected` with its reason, so nothing disappears silently.

| Step | Method |
|---|---|
| Clean | Strip HTML tags and entities, normalize whitespace, keep paragraph breaks |
| Filter | Reject empty articles and articles under 200 characters |
| Exact duplicates | SHA-256 of the normalized text; keep one per hash |
| Near-duplicates | **MinHash + LSH** on 5-word shingles (32 hashes, 8 bands × 4 rows); estimated Jaccard ≥ 0.8 |
| Language detection | Cyrillic share for Russian; for the others, stopword hits plus letters unique to one language (Turkish ı ğ ş, Spanish ñ ¿ ¡, German ä ß, French è ê œ ù); English scored too, to catch mislabeled articles |
| Dates | Parse `dd/MM/yyyy`; German and Turkish sources only record the month (`00/MM/yyyy`), so those become the 1st of the month with `date_precision = 'month'` |

**Results**

| Language | Bronze | Exact duplicates | Near-duplicates | Silver |
|---|---|---|---|---|
| German (de) | 2,100 | 0 | 2 | 2,098 |
| Spanish (es) | 2,100 | 45 | 0 | 2,055 |
| French (fr) | 2,100 | 13 | 0 | 2,087 |
| Russian (ru) | 1,507 | 0 | 0 | 1,507 |
| Turkish (tr) | 2,100 | 48 | **77** | 1,975 |
| **Total** | **9,907** | **106** | **79** | **9,722** |

- **185 duplicates removed (1.9%).** Turkish had the most (6%), mostly near-duplicates: the same wire story republished with small edits, which exact matching alone would miss.
- **Language detection matches the source label for 99.7% of articles (9,688 of 9,722).** The first version reached 98.0%. A breakdown showed 164 of the 199 misses were Turkish marked `unknown`, because Turkish attaches grammar to word endings and has fewer separate stopwords. Adding letters unique to each language fixed them.
- **The remaining mismatches are a labeling error in the source, not detector errors.** The 34 "Spanish" articles that didn't match are actually **English**: El País also publishes an English edition, and some of those articles ended up in MLSUM's Spanish file. The detector now scores English too, so they're flagged as `detected_language = 'en'`. Downstream steps use the detected language.
- **Dates: 100% parsed** (5,649 to the day, 4,073 to the month). The first version parsed only 58% because of the month-only format.

![Silver summary](docs/silver_summary.png)

### Silver: PII masking and verification
`03_silver_pii` masks personal data in all 9,722 articles before any text can reach chunks, embeddings or an LLM. Output: `docs_silver.articles_masked`, with per-article counts of what was masked.

| PII type | Method | Replaced with |
|---|---|---|
| Email, URL, IP, IBAN | Regular expressions | `[EMAIL]` `[URL]` `[IP]` `[IBAN]` |
| Phone numbers | Pattern + at least 9 digits; dates, number ranges, year lists and thousands-separated numbers are excluded; a leading `+` or a nearby "tel / call / llame / телефон" decides ambiguous cases | `[PHONE]` |
| Person names | **spaCy multilingual NER** (`xx_ent_wiki_sm`), run per partition with `mapInPandas`; a Turkish rule adds names after titles (Cumhurbaşkanı, Bakan…); spans not capitalised like a name are rejected and counted | `[PERSON]` |

**Two checks verify the result:**
1. **Residual check (blocking):** the email, URL, IP and IBAN patterns run again on the masked text. Any hit, or a row-count mismatch, fails the job.
2. **Independent audit:** Databricks `ai_mask()` (an LLM) re-reads a sample of masked articles. Anything *it* still masks is PII the pipeline missed. Since `ai_mask` is rate-limited, the audit uses 6 articles per language (first 1,200 characters each).

**Results**

| Language | Articles | Email | URL | Phone | Person | Total | Per article |
|---|---|---|---|---|---|---|---|
| German (de) | 2,098 | 30 | 65 | 14 | 25,755 | 25,864 | 12.3 |
| Spanish (es) | 2,055 | 19 | 196 | 34 | 29,705 | 29,954 | 14.6 |
| French (fr) | 2,087 | 15 | 218 | 56 | 23,728 | 24,017 | 11.5 |
| Russian (ru) | 1,507 | 4 | 11 | 19 | 28,137 | 28,171 | 18.7 |
| Turkish (tr) | 1,975 | 0 | 34 | 73 | 25,103 | 25,210 | 12.8 |
| **Total** | **9,722** | **68** | **524** | **196** | **132,428** | **133,216** | **13.7** |

No IPs or IBANs appeared in the data. **Residual check: 0 hits (passed).**

| Audit (ai_mask) | Sampled | Missed items | Articles with a miss |
|---|---|---|---|
| German | 6 | 3 | 2 |
| Spanish | 6 | 0 | 0 |
| French | 6 | 3 | 2 |
| Russian | 6 | 6 | 1 |
| Turkish | 6 | 9 | 2 |
| **Total** | **30** | **21** | **7** |

- **The first audit measured the wrong thing.** It reported 50 misses in 13 of 30 articles, but most of those were the LLM re-masking our own `[PERSON]` placeholders. The fixed audit subtracts placeholders it overwrote and counts only *new* masks: **21 real misses in 7 articles**.
- **Turkish was over-masked.** The NER model tagged lowercase Turkish words as names. A rule requiring name-like capitalisation (allowing particles like *de*, *van*, *bin*, so "Charles de Gaulle" still passes) cut person masks from 141,180 to **132,428** (8,752 fewer, −6%); every rejected span is counted in `person_spans_rejected`.
- **Phones needed context, not just a pattern.** The first pattern missed numbers followed by a full stop, and treated Spanish `915 123 456` numbers as ordinary thousands. Excluding dates, ranges and year lists, plus a "call/tel" context rule, fixed both without masking prices or statistics.
- **Remaining risk is known and measured:** most misses are names the small NER model doesn't recognise, concentrated in Russian and Turkish. A larger model (or `ai_mask` itself, once it scales) would close this gap.

![PII summary](docs/pii_summary.png)
![PII audit](docs/pii_audit.png)

### Gold: chunks and embeddings
_TODO_

## Design Decisions
- **Deterministic sampling instead of `sample()`:** reproducible results, so metrics stay comparable between runs.
- **`doc_id` from a hash of the URL:** a stable key that survives re-ingestion and links every chunk back to its source article.
- **Language from the file name, detected again later:** the source label becomes a quality check for the language detector.
- **MinHash + LSH in plain Spark SQL:** finds near-duplicates without comparing all ~49M pairs and without ML libraries that serverless compute may not support.
- **Rejected rows are kept with a reason:** every removal is auditable, not silent.
- **A lightweight, deterministic language detector:** no package installs, fast, and explainable; at larger scale it could be swapped for fastText.
- **`date_precision` instead of guessing a day:** month-only dates stay usable without pretending to be exact.
- **Patterns before NER:** exact patterns run first, so the model never sees (or splits) emails and URLs, and placeholders are never re-tagged as names.
- **Placeholders, not deletion:** `[PERSON]`, `[EMAIL]`… keep sentence structure for chunking and embeddings, and make masking countable.
- **spaCy loaded once per partition with `mapInPandas`:** runs a Python NLP model at Spark scale without loading it per row.
- **A blocking check plus a measured audit:** deterministic PII must be 0 or the job fails; fuzzy PII (names) gets an independent miss rate instead of a guess.
- **`ai_mask` as an auditor, not the masker:** it's rate-limited on this tier, so it checks a sample rather than processing all 9,722 articles.
- _TODO: add more as built_

## Repository Structure
```
├── data/raw/      # place MLSUM .jsonl files here (not committed; see How to run)
├── notebooks/     # Databricks notebooks
└── docs/          # architecture diagram, screenshots
```

## How to Run
1. Create a [Databricks Free Edition](https://www.databricks.com/learn/free-edition) workspace.
2. Download the MLSUM files `de_val`, `es_val`, `fr_val`, `tu_val`, `ru_val` and `ru_test` (`.jsonl`) from the [MLSUM repository](https://github.com/recitalAI/MLSUM).
3. Run `00_setup`, then upload the files to the landing folder it prints.
4. Run `01_bronze_ingest`, `02_silver_clean`, then `03_silver_pii` (installs spaCy and its multilingual model from PyPI/GitHub).

## Author
**Aditya Raj**: [LinkedIn](https://linkedin.com/in/aditya-raj-5a1a6a21b)
