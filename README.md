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
                            │  TODO: text cleaning, deduplication, language detection
                            ▼
                   docs_silver.*          (clean)
                            │  TODO: PII masking, quality checks
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
- **Articles ingested:** _TODO_

### Silver: clean and standardize
_TODO: text normalization, near-duplicate removal, language detection_

### PII masking and quality checks
_TODO_

### Gold: chunks and embeddings
_TODO_

## Design Decisions
- **Deterministic sampling instead of `sample()`:** reproducible results, so metrics stay comparable between runs.
- **`doc_id` from a hash of the URL:** a stable key that survives re-ingestion and links every chunk back to its source article.
- **Language from the file name, detected again later:** the source label becomes a quality check for the language detector.
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
4. Run `01_bronze_ingest`.

## Author
**Aditya Raj**: [LinkedIn](https://linkedin.com/in/aditya-raj-5a1a6a21b)
