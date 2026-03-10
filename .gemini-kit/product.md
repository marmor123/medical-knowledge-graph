# Product Context: Medical Knowledge Graph

## Product Description
Distill dense, unstructured medical reference PDFs (specifically "Pocket Medicine") into an interactive, offline-first knowledge graph. The app allows medical interns to trace symptoms to differential diagnoses (DDx), labs, and treatments via a visual, interactive decision tree.

## Primary Users
Medical interns, residents, and clinicians in high-stakes clinical environments.

## Main Goals
1. **Zero Hallucination:** Strictly visualize pre-compiled graph data; no on-the-fly generation.
2. **Offline-First:** Operate entirely client-side via WASM/WebGL without runtime cloud dependencies.
3. **Absolute Provenance:** Every node and edge must include metadata pointing to the exact source (Book Title, Chapter, Page).

## Main Features
- **Symptom Entry:** Typo-tolerant offline search for symptoms or lab findings.
- **Interactive Visualization:** Node-based decision trees with expandable branches (Next Steps, DDx, etc.).
- **Cross-referencing:** Multi-node canvas to visualize intersections of clinical pathways.
- **Strict Citations:** Built-in citation rendering for every clinical recommendation.
