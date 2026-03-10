# Product Specification: Offline Medical Knowledge Graph (MVP)

## 1. Core Objective
Distill dense, unstructured medical reference PDFs (specifically "Pocket Medicine") into an interactive, offline-first knowledge graph. The app allows medical interns to trace symptoms to differential diagnoses (DDx), labs, and treatments via a visual, interactive decision tree.

## 2. Strict Constraints (The "Do Nots")
* **Zero Hallucination Tolerance:** The system MUST NOT generate medical advice on the fly. It acts strictly as an interactive visualizer for the pre-compiled graph database.
* **No Runtime LLM Calls:** The Knowledge Graph (KG) is compiled via an offline ETL pipeline. The live React app relies solely on querying the local/WASM database.
* **Absolute Provenance:** Every single node (e.g., "Fever") and edge (e.g., "Fever -> leads to -> Sepsis") MUST include metadata pointing to the exact source (Book Title, Chapter, Page Number).

## 3. User Experience (UX)
* **Entry:** User types a symptom or lab finding into a typo-tolerant, offline search bar.
* **Visualization:** The screen displays a node. Clicking it expands its branches (Next Steps, Physical Exam reminders, DDx).
* **Cross-referencing:** Users can add multiple nodes to the canvas to see where their branches intersect (e.g., intersecting "Fever" and "Throat Pain").