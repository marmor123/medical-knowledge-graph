# Implementation Plan: Finalize Full Ingestion of Pocket Medicine 9th Edition

## Approach
We will execute the complete ETL pipeline to extract, resolve, and load the entire medical textbook into the knowledge graph.
- **Why this solution:** The pipeline is already modularized and tested. Incremental saving in `processor.py` ensures resilience against interruptions. The dual-layer Entity-Mention schema provides the necessary semantic depth.
- **Alternatives considered:** One-shot loading without validation (too risky for medical data).

## Steps
1. **Pilot Run (Verification)** (15 min)
   - Run ingestion for the first 10 pages to verify schema adherence.
   - `python -m src.etl.processor --pdf "Pocket Medicine 9th Edition 2026.pdf" --limit 10`
   - Inspect `data/interim/raw_chunks.json`.

2. **Full scale Ingestion** (Several hours, CPU dependent)
   - Run ingestion for the entire document.
   - `python -m src.etl.processor --pdf "Pocket Medicine 9th Edition 2026.pdf"`
   - Monitor for performance or memory issues.

3. **Database Loading** (20 min)
   - Load the resolved chunks into the production Kùzu database.
   - Update `db_loader.py` to point to `data/db`.
   - `python -m src.etl.db_loader`

4. **Validation & Reporting** (15 min)
   - Run the LLM-as-a-judge validator on the final graph.
   - `python -m src.etl.validator --db data/db --limit 100`

## Timeline
| Phase | Duration |
|-------|----------|
| Pilot | 15 min |
| Ingestion | ~4-6 hours (CPU) |
| DB Load | 20 min |
| Validation | 15 min |
| **Total** | **~7 hours** |
