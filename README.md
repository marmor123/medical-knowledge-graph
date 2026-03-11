# Medical Knowledge Graph (Offline-First)

Distill dense medical reference PDFs into an interactive knowledge graph using VLMs and embedded graph databases.

## 🛠️ ETL Pipeline

### 1. Ingestion
Converts PDF pages to high-resolution images and uses **Qwen2-VL** to extract structured clinical data (text, tables, shorthand).
- `src/etl/rasterizer.py`: PDF to Image conversion.
- `src/etl/vlm_parser.py`: VLM-based extraction.
- `src/etl/processor.py`: Orchestrator with incremental saving.

### 2. Entity Resolution & Schema
Decouples physical mentions from conceptual identities using a dual-layer schema.
- **Stage 1 (Optional):** Extract abbreviation dictionary from the book (pages 417+).
  ```bash
  python -m src.etl.processor --pdf "book.pdf" --start 417 --mode abbrev
  python -m src.etl.compile_abbreviations
  ```
- **Stage 2:** Resolve mentions to UMLS CUIs using **SapBERT** and the custom dictionary.
  - `src/etl/resolver.py`: Resolves mentions to UMLS CUIs.
  - `src/db/schema.cypher`: Graph schema definition.
  - `src/etl/db_loader.py`: Ingests resolved data into **Kùzu**.

### 3. Validation (LLM-as-a-Judge)
Ensures data integrity and extraction quality.
- `src/etl/validator.py`: Samples triples from Kùzu and evaluates resolution accuracy.
- **Run validation:**
  ```bash
  python -m src.etl.validator --db data/db --limit 50
  ```

## 🧪 Testing
Run the full test suite to verify extraction and database logic:
```bash
$env:PYTHONPATH="."
pytest tests/test_etl.py
pytest tests/test_er_schema.py
```

## 📊 Database Schema
- **Entity**: Canonical concept (CUI, Name).
- **Mention**: Text found in the document (Text, Role).
- **Chunk**: Source provenance (Source, Page, Text).
- **Relationships**:
  - `(Mention)-[:REFERS_TO]->(Entity)`
  - `(Mention)-[:APPEARS_IN]->(Chunk)`
