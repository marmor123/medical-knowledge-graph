# Implementation Plan: Medical Knowledge Graph - Phase 1 (Ingestion)

### ## Approach
We will implement a layout-aware ingestion pipeline using Python. This phase focuses on converting dense medical PDFs into structured JSON chunks while preserving reading order and tabular data.
- **Why this solution:** Traditional OCR (Tesseract/Marker) fails on multi-column medical layouts. Vision-Language Models (VLMs) like Qwen3-VL "understand" visual semantics, ensuring perfect table reconstruction and reading order.
- **Alternatives considered:** Nougat (too slow/hallucinatory for non-academic layouts), Unstructured (struggles with complex column wraps).

### ## Steps
1. **Environment Setup** (10 min)
   ```bash
   python -m venv venv
   ./venv/Scripts/activate
   pip install pdf2image transformers qwen-vl-utils pydantic torch torchvision
   ```

2. **PDF Rasterization Service** (15 min)
   - File: `src/etl/rasterizer.py`
   - Use `pdf2image` to convert PDF pages to 300 DPI PNGs.
   ```python
   from pdf2image import convert_from_path
   def rasterize_pdf(pdf_path, output_dir):
       # Implementation using convert_from_path
       pass
   ```

3. **VLM Inference Pipeline** (30 min)
   - File: `src/etl/vlm_parser.py`
   - Implement `Qwen3-VL` local inference loop.
   - Use `Pydantic` for strict JSON output enforcement.
   ```python
   class PageChunk(BaseModel):
       text: str
       tables: List[str]
       metadata: Dict[str, Any]
   ```

4. **Chunking & Metadata Attachment** (15 min)
   - File: `src/etl/processor.py`
   - Attach source filename, page number, and section headers to each chunk.

5. **Validation & Storage** (10 min)
   - Save chunks to `data/interim/raw_chunks.json`.

### ## Timeline
| Phase | Duration |
|-------|----------|
| Dependencies | 10 min |
| Rasterizer | 15 min |
| VLM Pipeline | 30 min |
| Processing | 15 min |
| Validation | 10 min |
| **Total** | **1 hour 20 min** |

### ## Rollback Plan
1. Delete `venv` and re-install from `requirements.txt`.
2. Revert to `marker` (heuristic parser) if VLM inference latency is too high for local dev.

### ## Security Checklist
- [x] Local LLM execution (No data sent to cloud)
- [x] Input PDF path validation
- [x] Pydantic schema validation
- [ ] Error handling for corrupted PDF pages
