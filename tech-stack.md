# Technical Architecture & Stack

## Phase 1: Data Ingestion & ETL (Python)
* **PDF Parsing:** `pdf2image` to convert textbook pages to high-res images.
* **Vision-Language Model (VLM):** `Qwen3-VL` (running locally via `transformers` and `qwen-vl-utils`) to read multi-column layouts, tables, and clinical shorthand.
* **Structured Output:** `Pydantic` must be used strictly to force Qwen3-VL to output valid JSON matching our graph schema. 
* **Entity Resolution:** Python scripts to merge duplicate nodes (e.g., mapping "SOB" and "Dyspnea" to the same master node) while preserving distinct page citations on the edges.

## Phase 2: Database (Kùzu / WASM)
* **Engine:** `Kùzu` (embedded graph DB). We use this because it compiles to WebAssembly, allowing complex Cypher queries entirely in the user's browser without a backend server.
* **Schema:** Nodes (Symptom, Diagnosis, Lab, Treatment) and Edges (LEADS_TO, REQUIRES_LAB, INDICATES). 

## Phase 3: Frontend (React + TypeScript)
* **Framework:** Vite + React.
* **Search:** Client-side index (e.g., `MiniSearch` or `FlexSearch`) for instant node lookup.
* **Graph Rendering:** `Sigma.js` (WebGL based) for 60fps rendering of thousands of nodes.
* **Physics:** Graph layout algorithms (Force-directed) must be offloaded to a Web Worker so the main UI thread never freezes.