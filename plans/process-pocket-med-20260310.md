# Implementation Plan: Medical PDF Processing (VLM)

## Approach
We will execute the previously implemented layout-aware ETL pipeline.
- **VLM Strategy:** Use Qwen3-VL (local/remote as configured) to extract structured clinical data (text, tables, shorthand) while preserving page citations.
- **Batching:** Process in batches to manage memory and handle potential failures.

## Steps
1. **Pre-flight Check** (10 min)
   - Verify Poppler installation (`pdftocairo` or `pdftoppm`).
   - Check if CUDA is available for `transformers`.
   - Ensure `Pocket Medicine 9th Edition 2026.pdf` is accessible.

2. **Sample Extraction (Pilot)** (15 min)
   - Run `processor.py` on a limited set of pages (e.g., 1-5).
   - `python -m src.etl.processor --pdf "Pocket Medicine 9th Edition 2026.pdf" --limit 5`
   - Inspect `data/interim/raw_chunks.json` for schema adherence and extraction quality.

3. **Full Scale Processing** (Variable - ~2-4 hours)
   - Execute the pipeline for the entire document.
   - Monitor logs for VLM timeout or JSON parsing errors.

4. **Data Verification** (20 min)
   - Validate JSON integrity.
   - Ensure every chunk has a `source_file` and `page_number`.

## Timeline
| Phase | Duration |
|-------|----------|
| Check | 10 min |
| Pilot | 15 min |
| Full | 3h |
| Verify| 20 min |
| **Total** | **~4 hours** |

## Rollback Plan
- If VLM fails, revert to manual text extraction + heuristic table parsing (fallback).

## Security Checklist
- [x] Local execution (if Qwen3-VL is local).
- [x] Pydantic validation for all LLM outputs.
