# Tech Stack: Medical Knowledge Graph

## Frontend (React + TypeScript)
- **Framework:** Vite + React (TypeScript)
- **Styling:** Vanilla CSS (preferred for flexibility and speed)
- **Graph Rendering:** Sigma.js (WebGL-based) for high-performance network rendering.
- **Search:** MiniSearch or FlexSearch (Client-side, typo-tolerant index).
- **Architecture:** Web Workers for graph physics offloading.

## Database (Graph)
- **Engine:** Kùzu (Embedded graph DB).
- **Client-Side:** Compiled to WebAssembly (WASM) for in-browser Cypher queries.

## ETL Pipeline (Python)
- **PDF Extraction:** `pdf2image` (rasterized 300 DPI) + `Qwen3-VL` (locally via `transformers`).
- **Data Structuring:** `Pydantic` for schema enforcement on VLM outputs.
- **Normalization:** ClinicalBERT or BlueBERT for abbreviation disambiguation.

## Testing & Quality
- **Frontend:** Vitest
- **ETL:** Pytest
- **Validation:** LLM-as-a-Judge (OpenAI o1 or Claude 3.5 Opus) for extraction fidelity.
