# **Architectural Blueprint for Offline Medical Knowledge Distillation and Clinical Decision Support**

## **Executive Overview**

The translation of dense, unstructured medical textbooks into computable, highly structured clinical decision-support systems represents a monumental challenge within the field of medical informatics. Traditional Retrieval-Augmented Generation (RAG) architectures, which rely primarily on vector similarity search and on-the-fly Large Language Model (LLM) inference, often fail in high-stakes clinical environments. These failures stem from unpredictable latency, high computational inference costs, and the persistent, unacceptable risk of model hallucinations where non-existent clinical relationships are fabricated during the generation phase.

This analysis details the architectural design, algorithmic strategies, and technical roadmap for an alternative, highly deterministic paradigm: the comprehensive distillation of complex medical PDFs into a static, offline-compiled knowledge graph. By shifting the computational burden of natural language processing and LLM reasoning entirely to the preprocessing phase, the resulting application guarantees zero-latency execution, perfect reproducibility, and absolute citation fidelity. Operating entirely client-side via WebAssembly (WASM) and WebGL, the system provides an interactive, privacy-preserving clinical decision-support tool that does not rely on external cloud infrastructure at runtime.

The ensuing report meticulously evaluates state-of-the-art layout-aware parsing mechanisms for document ingestion, ontology-driven semantic normalization pipelines for clinical shorthand, conditional graph schema designs for medical logic, lightweight embedded databases for in-browser execution, and WebGL-accelerated frontend visualizations for massive network rendering. It culminates in an agentic development roadmap and an automated validation framework to ensure rigorous factuality and extraction recall.

## **Ingestion and Layout-Aware Parsing of Dense Medical Texts**

The foundational layer of any knowledge distillation pipeline relies entirely on the precise extraction of the source material. Medical textbooks represent arguably the most hostile environment for standard optical character recognition (OCR) and text extraction due to their reliance on multi-column layouts, nested clinical tables, complex mathematical or chemical formulas, marginalia, and footnotes. The failure to preserve reading order or tabular hierarchies during this initial phase inevitably corrupts downstream entity extraction, leading to a flawed knowledge graph.

### **Evaluation of Open-Source Parsing Pipelines**

Historically, programmatic document parsing relied on heuristic-based bounding box identification and traditional OCR engines like Tesseract. The evaluation of contemporary open-source parsing pipelines reveals significant divergence in capabilities for handling the structural complexity of medical literature.

| Parsing Tool | Core Architecture | Table Preservation | Reading Order | Optimal Use Case | Performance Characteristics |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Nougat** | Transformer-based Vision Encoder-Decoder | Poor to Moderate | Moderate | Academic papers with dense mathematical formulas in LaTeX. | High accuracy for equations; suffers from slow autoregressive generation and hallucination loops on non-academic layouts.1 |
| **Marker** | Multi-stage heuristic and ML pipeline | Moderate | Good | High-speed, bulk conversion of standard digital PDFs to Markdown. | Extremely fast CPU processing; struggles to retain intricate row-span/col-span properties in nested clinical tables.4 |
| **Unstructured** | ML-driven layout analysis (YOLOX) with OCR fallback | Moderate | Good | General enterprise document chunking for standard RAG pipelines. | Highly modular; however, text flow and paragraph integrity are frequently disrupted by embedded multi-column wraps.5 |

Nougat, developed specifically for academic documents, excels at transcribing mathematical equations into LaTeX format.2 However, its autoregressive nature results in slow processing speeds, and it occasionally falls into repetition loops when encountering the highly stylized, multi-column layouts typical of modern medical textbooks.1 Marker provides a much faster alternative, effectively converting digital PDFs to Markdown, but it fundamentally struggles to retain the logical hierarchy of complex clinical tables, often flattening them into unreadable text strings.4 Unstructured offers a highly modular approach, combining computer vision models with OCR engines, but frequently fragments paragraphs when interrupted by embedded figures, rendering cross-sentence relation extraction impossible.5

### **The Vision-Language Model (VLM) Paradigm Shift**

The emergence of multimodal Vision-Language Models (VLMs) fundamentally alters the document ingestion landscape. Instead of relying on brittle, multi-stage layout-analysis pipelines, VLMs process the entire rendered PDF page as a high-resolution image, utilizing spatial reasoning to decode the semantic structure simultaneously with the text. This represents a shift from "reading" a document to "understanding" its visual semantics.7

Models such as GPT-4o and Claude 3.5 Sonnet demonstrate unprecedented, state-of-the-art performance in complex document understanding.9 Claude 3.5 Sonnet, in particular, exhibits superior capabilities in preserving tabular structures.10 When prompted with specific extraction schemas, these models can transcribe nested clinical schedules of activities directly into structured JSON or Markdown, retaining the implicit relationships between headers and nested cell values that traditional OCR destroys.10 Furthermore, VLMs natively preserve the correct reading order across complex multi-column layouts and intuitively bypass irrelevant page headers, footers, and watermarks, which otherwise pollute the semantic space.13

For the highest fidelity extraction in the proposed architecture, a hybrid rendering approach is required. Each PDF page is rasterized into high-resolution images (e.g., utilizing Poppler or PyMuPDF with a minimum of 300 DPI) and passed through Claude 3.5 Sonnet or an advanced open-weights document understanding model such as Qwen2.5-VL or MeDocVL.15 The extraction must be strictly constrained to JSON, yielding an array of text and table chunks. Crucially, each chunk must be meticulously tagged with its exact bounding box coordinates, source page number, and hierarchical section header.16 This structural preservation is the prerequisite for establishing the strict citation metadata required in the final knowledge graph.

## **Semantic Normalization and Medical Shorthand Decoding**

Extracting clean text from a PDF is merely a structural prerequisite; the true utility of the decision-support system is derived from semantic understanding. Medical texts, particularly those designed as quick-reference pocket medicine guides, are notoriously dense with clinical shorthand, acronyms, and domain-specific vernacular. A knowledge graph populated with raw, unnormalized text will suffer from severe entity fragmentation. For example, "Type 2 Diabetes," "T2DM," "NIDDM," and "Non-insulin-dependent diabetes mellitus" would exist as isolated, disconnected nodes, destroying the graph's ability to facilitate cohesive multi-hop reasoning.

### **Decoding Clinical Shorthand via NLP Pipelines**

The disambiguation of medical abbreviations is highly context-dependent and cannot be solved through simple dictionary lookups. For instance, the abbreviation "MS" can denote "Multiple Sclerosis," "Mitral Stenosis," or "Mental Status," while "pt" can refer to "patient" or "posterior tibialis," depending entirely on the surrounding narrative context.18

To decode this shorthand before graph construction, the architecture must implement a specialized natural language processing pipeline. The One-to-All (OTA) framework, backed by Bidirectional Encoder Representations from Transformers (BERT), provides a robust mechanism for this task.19 By leveraging the bidirectional context tokens surrounding an abbreviation, fine-tuned models such as ClinicalBERT or BlueBERT can achieve macro-accuracies exceeding 95% on abbreviation disambiguation tasks across standard clinical datasets.19

Alternatively, passing the extracted textual chunks through an optimized LLM prompt—augmented with few-shot examples derived from the MeDAL (Medical Dataset for Abbreviation Disambiguation for Natural Language Understanding) corpus or the Clinical Abbreviation Sense Inventory (CASI)—enables highly accurate, zero-shot expansion of clinical shorthand.21 Furthermore, privacy-preserving machine learning techniques, such as web-scale reverse substitution, have been proven to train highly effective disambiguation models without requiring access to sensitive, protected health information.24 The pipeline must enforce a strict rule: all abbreviations must be algorithmically expanded to their full lexical forms before any relation extraction occurs.

### **Integration of UMLS and SNOMED CT for Entity Resolution**

To facilitate seamless cross-chapter entity resolution, every extracted medical concept must be anchored to a standardized clinical ontology. The Unified Medical Language System (UMLS) Metathesaurus, and specifically its integration with the Systematized Nomenclature of Medicine – Clinical Terms (SNOMED CT), serves as the optimal semantic backbone for this endeavor.25 SNOMED CT provides a comprehensive, scientifically validated framework that maps over 350,000 distinct medical concepts and their inherent hierarchical relationships.27

The normalization pipeline operates sequentially. First, the LLM extracts candidate entities from the decoded text chunk. Subsequently, a specialized biomedical embedding model, such as BioBERT, generates dense vector representations of these extracted text strings.30 These embeddings are then queried against a localized vector database (e.g., ChromaDB or FAISS) that has been pre-populated with the official SNOMED CT terminology and their corresponding descriptions.30

By calculating the cosine similarity between the extracted entity's vector and the standardized ontology vectors, the system resolves disparate textual representations into a single, canonical Concept Unique Identifier (CUI).30 This rigorous normalization process ensures that when the final graph is compiled, all references to a specific pathology or pharmacological agent—regardless of their original phrasing or the textbook chapter they originated from—converge onto a singular, merged node. This topological centralization is critical; it is the mechanism that allows the graph to fuse isolated facts into a cohesive web of medical knowledge.

## **Graph Schema Design for Conditional Clinical Logic**

Standard Resource Description Framework (RDF) triples, which operate on a simplistic Subject \-\> Predicate \-\> Object paradigm, are fundamentally ill-equipped to represent the nuanced conditional logic inherent in clinical decision support. Medical guidelines rarely dictate absolute truths; rather, they formulate rules based on specific patient states, formatted as: *IF Patient has Condition A, AND Patient is taking Drug B, THEN recommend Intervention C, UNLESS Comorbidity D is present.*

Attempting to model this complexity with standard binary edges results in catastrophic information loss or mathematically invalid cyclical loops.

### **Reification and Hypergraph Methodologies**

To accurately represent n-ary relationships and multi-conditional logic, the graph schema must employ techniques known as reification or hypergraph modeling.33 In a traditional labeled property graph, reification involves creating an intermediate entity—such as an "Event," "Rule," "Context," or "Protocol" node—rather than attempting to connect entities directly with a single, overburdened edge.34

Instead of generating a simplistic and clinically dangerous (Drug)--\>(Disease) edge, the architectural schema defines a central (Clinical\_Protocol) node. This protocol node serves as a logical hub, connecting to the requisite medical entities via highly specific, directional edges:

* (Clinical\_Protocol)--\>(Disease)  
* (Clinical\_Protocol)--\>(Symptom/Lab\_Value)  
* (Clinical\_Protocol)--\>(Comorbidity)  
* (Clinical\_Protocol)--\>(Drug/Procedure)

This hypergraph-inspired structure inherently supports complex conditional branching.38 When an end-user inputs patient parameters matching the Observation and the Disease, the system traverses the graph to the central Clinical\_Protocol node, verifies the absence of the connected Comorbidity, and subsequently follows the path to the recommended Intervention. This schema accurately mirrors human clinical reasoning by encapsulating the entire context of a medical decision into a cohesive, queryable subgraph.

### **Preserving Multi-Source Citations on Merged Nodes**

Because the knowledge graph fuses information from multiple textbook chapters into singular conceptual nodes, a single clinical protocol may possess multiple document origins. To maintain absolute trust and auditability in a medical context, the schema must explicitly separate the semantic domain (the medical concepts) from the lexical domain (the source text).17

The architecture achieves this by utilizing a Map-Reduce-Align extraction pattern 39 and establishing (Chunk) nodes that represent the raw, unedited text extracted directly from the PDFs. Every semantic node (e.g., Disease, Intervention) and every reified Clinical\_Protocol node is linked back to its originating (Chunk) node via a or edge.17

The (Chunk) node acts as a rigid metadata container, holding properties such as Source\_File, Page\_Number, Section\_Header, and Bounding\_Box\_Coordinates.17 Consequently, when the frontend application displays a clinical recommendation, it executes a reverse traversal from the semantic node back to all associated chunk nodes. This allows the UI to render exact, multi-source citations, complete with page numbers and original text snippets, without relying on an LLM to "remember" or hallucinate document provenance.39

## **Database Evaluation for Offline Compilation**

A primary directive of this architectural design is that the final application must operate as a static, offline-compiled tool. This requirement eliminates dependencies on continuous server hosting, mitigates cloud latency, completely circumvents runtime LLM inference costs, and ensures patient data privacy by keeping all queries local to the user's device. Therefore, the chosen graph database must be exceptionally lightweight, embeddable, and capable of executing complex multi-hop Cypher queries strictly on the client side.

### **Comparative Analysis of Graph Database Engines**

| Database Engine | Architecture / Language | Storage Model | Offline / Client-Side Capability | Optimal Use Case |
| :---- | :---- | :---- | :---- | :---- |
| **Neo4j** | Java-based, Client-Server | Disk-based Property Graph | Low (Requires JVM / Heavy runtime) | Enterprise scale, highly mutable data, centralized servers.42 |
| **Memgraph** | C++, Client-Server | In-Memory Property Graph | Low (Requires server daemon) | High-speed real-time streaming analytics and write-heavy workloads.42 |
| **ArangoDB** | C++, Client-Server | Multi-model (Graph, Document) | Low (Complex deployment) | Polyglot persistence, enterprise search and complex clustering.44 |
| **SQLite** | C, Embedded | Relational (Graph via recursive CTEs) | High (WASM compiled) | Simple local storage; degrades heavily on deep graph traversals.46 |
| **Kùzu** | C++, Embedded | Columnar Property Graph | Extreme (Native WASM compilation) | Heavy analytical multi-hop graph queries executed entirely in-browser/edge.48 |

While Neo4j remains the undisputed industry standard for enterprise knowledge graphs due to its robust Cypher support and mature ecosystem, its reliance on the Java Virtual Machine (JVM) renders it fundamentally unsuitable for lightweight, offline web distribution.42 Memgraph offers a highly performant, C++ based in-memory alternative that routinely outperforms Neo4j in speed benchmarks, but it still operates on a client-server paradigm requiring a background daemon, making it difficult to package into a standalone static site.42

SQLite, when compiled to WebAssembly (WASM), is ubiquitous for offline web applications. However, it is a relational database. Mapping a highly connected, multi-hop knowledge graph into relational tables requires executing recursive Common Table Expressions (CTEs). As the traversal depth of a query increases, the performance of recursive CTEs degrades exponentially, making SQLite computationally prohibitive for complex clinical logic.46

### **The Kùzu-WASM Advantage**

The optimal, purpose-built engine for this specific offline architectural requirement is **Kùzu**.48 Engineered from the ground up as an embeddable, in-process columnar graph database written in C++, Kùzu is widely considered the "DuckDB of graph databases".50

Most critically, Kùzu provides native WebAssembly (WASM) bindings, allowing the entire graph database engine to execute locally within the user's web browser without any backend server.48 Kùzu utilizes a columnar sparse row-based (CSR) adjacency list and factorized query execution, which allows it to perform worst-case optimal joins (WCOJ).49 This highly optimized architecture prevents the browser's JavaScript V8 engine from exhausting memory limits when executing deep graph traversals.

During the preprocessing and development phase, Python scripts utilize the LlamaIndex or LangChain frameworks to structure the extracted data and populate a local Kùzu database file using standard Cypher syntax.56 For production deployment, this pre-compiled binary database file is simply packaged alongside the frontend HTML/JS assets and the Kùzu-WASM library. When the clinician loads the static application, the graph database is mounted directly into the browser's memory, guaranteeing instantaneous, read-only query execution with absolute privacy, as no clinical data ever traverses a network.59

## **Frontend Visualization and Cognitive Ergonomics**

Visualizing large-scale knowledge graphs containing tens of thousands of medical nodes and edges presents significant rendering bottlenecks and induces severe cognitive overload for the end user. If a clinician is presented with an unreadable "hairball" of data, the decision-support utility of the system is reduced to zero. Therefore, the frontend architecture must prioritize hardware-accelerated rendering and algorithmic curation to aggressively mitigate visual clutter.

### **WebGL Rendering and Web Worker Physics Offloading**

Traditional Document Object Model (DOM) based SVG rendering or HTML5 Canvas libraries, such as D3.js or Vis-network, suffer catastrophic frame rate degradation when node and edge counts exceed 2,000 to 5,000 elements.60 For a comprehensive medical knowledge graph extracted from a dense textbook, a WebGL-accelerated rendering engine is absolutely mandatory.

A comparative analysis of high-performance graph visualization libraries identifies **Sigma.js** as the superior open-source choice for this architecture.62 Sigma.js exclusively utilizes the GPU via WebGL to render tens of thousands of nodes and edges smoothly at 60 frames per second, vastly outperforming Canvas-based alternatives.63

However, the visual rendering pipeline is only half of the performance equation. The graph layout algorithm—typically a force-directed physics simulation like ForceAtlas2 that calculates the repulsion and attraction of nodes based on their edge weights—requires intense, continuous CPU computation. If this physics simulation is executed on the main browser thread, it will freeze the UI, rendering the application unresponsive to user input.67

The architectural solution requires isolating the graphology physics calculations within a dedicated **Web Worker**.68 The Web Worker asynchronously computes the spatial coordinates of the nodes in a background thread and streams the updated positions back to the main thread via continuous message passing. This allows Sigma.js to seamlessly animate the graph as it settles into its steady state without ever interrupting the clinician's interaction with the user interface.68

### **Algorithmic Curation for Visual Clutter Mitigation**

To ensure the application remains an effective, readable clinical tool, the visualization must employ algorithmic strategies to surface only the most relevant information.

**PageRank Node Sizing:** Not all medical concepts carry equal structural weight. A core pathology like "Hypertension" acts as a massive hub within the graph, whereas a highly specific genetic marker or secondary side effect may be a peripheral leaf node. By executing the PageRank algorithm across the knowledge graph prior to compilation, the system calculates a global relevance score for every node based on its topological centrality and the density of its incoming edges.70 The frontend utilizes these pre-calculated scores to dynamically scale the physical radius and label visibility of the nodes. Crucial, highly-connected clinical concepts dominate the visual hierarchy, while peripheral nodes remain visually unobtrusive until specifically queried.71

**Progressive Disclosure:** The user interface must employ a progressive disclosure paradigm to prevent overwhelming the user.74 Upon initial load, the application renders only the highest-level macro-nodes (e.g., primary disease categories or organ systems). When a user clicks or hovers over a specific clinical entity, the application executes a localized Cypher query against the Kùzu-WASM database to retrieve only its immediate 1-hop or 2-hop neighborhood, expanding the graph dynamically. Edges and nodes that are completely unrelated to the active clinical inquiry are aggressively faded or hidden, instantly clarifying the logical pathway from diagnosis to intervention.

## **Automated Validation Strategy: LLM-as-a-Judge**

Because the extraction process relies heavily on the generative and reasoning capabilities of VLMs and LLMs, an automated, rigorous validation pipeline is necessary to ensure the resulting offline graph contains zero hallucinations and achieves high recall of the source textbook's knowledge. Human evaluation of tens of thousands of extracted clinical relationships is prohibitively slow, unscalable, and expensive. Therefore, an "LLM-as-a-Judge" framework is deployed to programmatically audit the extraction pipeline prior to database compilation.75

The validation strategy utilizes an independent, highly capable reasoning model (e.g., OpenAI o1 or Claude 3.5 Opus) that is completely decoupled from the models used in the extraction pipeline. This judge model evaluates the extracted JSON graph structures directly against the raw text chunks generated in the initial parsing phase.

### **Core Evaluation Metrics**

| Metric | Definition | LLM-as-a-Judge Implementation Strategy |
| :---- | :---- | :---- |
| **Extraction Recall (Completeness)** | The proportion of critical facts present in the source text that were successfully captured and represented in the graph. | **Referenceless QAG (Question-Answer Generation):** The judge is provided the source text and asked to generate 5-10 critical clinical questions. It then queries the extracted JSON to verify if the graph can answer those questions, outputting a binary pass/fail to calculate a recall score.77 |
| **Factuality (Precision/Hallucination)** | The proportion of extracted graph triples that are strictly, factually supported by the source text. | **Reference-based Single Output:** The judge reviews each extracted semantic triple alongside the specific source text chunk cited in its metadata. If the triple cannot be directly inferred from the text, it is flagged as a hallucination.20 |
| **Schema Adherence** | The structural validity of the generated output. | **Deterministic Validation:** Does not require an LLM. Programmatic unit tests verify the output against the strict Pydantic/JSON schema definition to ensure valid database ingestion.81 |

The LLM-as-a-Judge methodology relies on providing the judge with a highly constrained prompt containing a rigid evaluation rubric.20 Furthermore, instead of requesting an arbitrary numerical score (which is prone to variance), the judge is instructed to generate a "Chain of Thought" rationale prior to rendering a binary decision. By utilizing the log-probabilities of the output tokens, the system can calculate a more continuous, weighted confidence score.78 Any subgraphs or relationships flagged for low factuality or high hallucination probability are automatically dropped from the pipeline or queued for manual human review prior to offline compilation, ensuring the clinical integrity of the final product.79

## **5-Phase Agentic AI Development Roadmap**

The development of the distillation pipeline requires a deterministic, heavily orchestrated workflow. The following five-phase roadmap relies on constrained, programmatic prompts chained together via frameworks like LlamaIndex or LangChain, rather than autonomous, open-ended agents, ensuring the structural rigidity required for medical data.

### **Phase 1: Layout-Aware Ingestion and Chunking**

The initial phase bypasses standard OCR in favor of multimodal processing. The PDF textbook is rasterized into high-resolution images.

* **System Prompt:** "You are an expert document parsing assistant. Analyze the provided image of a medical textbook page. Extract all text, preserving the exact reading order, including multi-column flows. If a table or schedule of activities is present, transcribe it entirely using structured Markdown format. Discard all page numbers, running headers, and footers. Output only the extracted content."  
  The resulting clean text is algorithmically chunked by section boundaries, and rigid metadata (Source File, Page Number, Section) is securely attached to each chunk object.

### **Phase 2: Entity Extraction, Shorthand Decoding, and Normalization**

This phase isolates the semantic entities from the narrative text and standardizes them.

* **System Prompt:** "You are a medical informatics expert. Review the following text chunk. Step 1: Identify and expand all clinical shorthand and acronyms to their full terminology based on standard medical vernacular (e.g., 'MS' to 'Multiple Sclerosis' based on context). Step 2: Extract all medical entities (Diseases, Drugs, Symptoms, Procedures) and the relationships between them. Output strictly in JSON format matching the provided schema."  
  Following extraction, the entities are embedded via BioBERT and mapped against the local SNOMED CT vector database. The raw text strings are replaced with their official Concept Unique Identifiers (CUIs).

### **Phase 3: Schema Reification and Logic Structuring**

Raw relational triples are converted into the hypergraph-style conditional schema required for clinical decision support.

* **System Prompt:** "Analyze the following extracted medical entities and relationships. Your task is to identify conditional clinical logic (e.g., 'Drug X is used for Disease Y only if Symptom Z is present'). Map these relationships into a reified structure using a central 'Clinical\_Protocol' node. The output must adhere to the provided graph schema, ensuring all conditional logic is represented via directional edges connected to the central protocol node."

### **Phase 4: Automated Verification (LLM-as-a-Judge)**

The independent judge model verifies the integrity of the proposed graph structures against the raw chunks generated in Phase 1 to prevent hallucinations.

* **System Prompt:** "You are an impartial medical auditor. You are provided with a raw text chunk from a medical textbook and a set of structured knowledge graph relationships extracted from it. Your task is to ensure zero hallucinations. For every relationship provided, verify that it is explicitly and factually stated in the text. Output a JSON list containing each relationship and a boolean 'is\_factual' flag. Provide a brief rationale for any relationship marked false."

### **Phase 5: Offline Compilation and UI Generation**

Once the extracted data passes validation, the nodes and edges are exported to a structured format (such as Parquet or CSV) and bulk-loaded into the local Kùzu database using its COPY command for maximum ingest efficiency.59 The Kùzu binary file is compiled. The frontend React application, bundled with Sigma.js and the Web Worker physics scripts, is finalized. The entire package is exported as a static set of web assets, ready for deployment on any standard web host or local file system without backend infrastructure.

## **Academic Fair Use and Legal Considerations**

The extraction of data from copyrighted medical textbooks for the creation of a private knowledge graph necessitates a brief review of intellectual property frameworks. In the United States, the legal doctrine of fair use heavily relies on the concept of "transformative use".84

The computational analysis of text, often referred to as Text and Data Mining (TDM), is generally viewed favorably under fair use when the purpose is non-expressive.86 The pipeline described herein does not aim to reproduce, display, or distribute the expressive, narrative prose of the original textbook authors. Instead, it utilizes machine learning to extract atomic, factual relationships and reassembles them into a fundamentally different structure—a computable, topological network designed for algorithmic conditional logic.84

This process is highly transformative; it creates an entirely new utility (a clinical decision-support tool) that does not serve as a market substitute for the original educational reading material.85 Furthermore, if the application is developed strictly for private, non-commercial, or internal academic research purposes, the first statutory factor of fair use (the purpose and character of the use) weighs heavily in favor of the developer.84 Consequently, maintaining the application offline and restricting its commercial distribution significantly mitigates the risk of copyright infringement while maximizing the utility of the mined clinical intelligence.

## **Conclusion**

The architecture outlined above represents a highly robust framework for bypassing the inherent vulnerabilities of generative AI in medical contexts. By utilizing cutting-edge Vision-Language Models for precise, layout-aware ingestion, enforcing semantic consistency via SNOMED CT vector alignments, and structuring complex logic through graph reification, the system extracts the true utility of dense medical texts.

The strategic decision to compile this data offline into a Kùzu WebAssembly database, visualized through Web Worker-accelerated WebGL via Sigma.js, ensures that the resulting clinical decision-support tool is secure, instantaneous, and completely immune to the runtime hallucinations that plague standard RAG architectures. This paradigm prioritizes rigorous pre-computation and automated LLM-as-a-judge validation, delivering a reliable, evidence-grounded asset suitable for the stringent demands of high-stakes medical informatics.

#### **Works cited**

1. Marker: Convert PDF to Markdown quickly with high accuracy | Hacker News, accessed on March 10, 2026, [https://news.ycombinator.com/item?id=38482007](https://news.ycombinator.com/item?id=38482007)  
2. Nougat: Neural Optical Understanding for Academic Documents \- arXiv, accessed on March 10, 2026, [https://arxiv.org/pdf/2308.13418](https://arxiv.org/pdf/2308.13418)  
3. INFINITY-PARSER: LAYOUT-AWARE REINFORCEMENT LEARNING FOR SCANNED DOCUMENT PARSING \- OpenReview, accessed on March 10, 2026, [https://openreview.net/pdf?id=M3GgDDGYec](https://openreview.net/pdf?id=M3GgDDGYec)  
4. Best document parser : r/Rag \- Reddit, accessed on March 10, 2026, [https://www.reddit.com/r/Rag/comments/1mhe1t4/best\_document\_parser/](https://www.reddit.com/r/Rag/comments/1mhe1t4/best_document_parser/)  
5. A Comparative Study of PDF Parsing Tools Across Diverse Document Categories \- arXiv, accessed on March 10, 2026, [https://arxiv.org/html/2410.09871v1](https://arxiv.org/html/2410.09871v1)  
6. How to Parse a PDF, Part 1 \- Unstructured, accessed on March 10, 2026, [https://unstructured.io/blog/how-to-parse-a-pdf-part-1](https://unstructured.io/blog/how-to-parse-a-pdf-part-1)  
7. Comparative Analysis of AI OCR Models for PDF to Structured Text | IntuitionLabs, accessed on March 10, 2026, [https://intuitionlabs.ai/articles/ai-ocr-models-pdf-structured-text-comparison](https://intuitionlabs.ai/articles/ai-ocr-models-pdf-structured-text-comparison)  
8. Beyond Text Extraction: The 2025 Open OCR Revolution Powered by Vision-Language Models | by TechEon, accessed on March 10, 2026, [https://atul4u.medium.com/beyond-text-extraction-the-2025-open-ocr-revolution-powered-by-vision-language-models-89ad33d36bbf](https://atul4u.medium.com/beyond-text-extraction-the-2025-open-ocr-revolution-powered-by-vision-language-models-89ad33d36bbf)  
9. Preserving Table Structure for Better Retrieval \- Unstructured, accessed on March 10, 2026, [https://unstructured.io/blog/preserving-table-structure-for-better-retrieval](https://unstructured.io/blog/preserving-table-structure-for-better-retrieval)  
10. Comparison Analysis: Claude 3.5 Sonnet vs GPT-4o \- Vellum AI, accessed on March 10, 2026, [https://www.vellum.ai/blog/claude-3-5-sonnet-vs-gpt4o](https://www.vellum.ai/blog/claude-3-5-sonnet-vs-gpt4o)  
11. Benchmarking proprietary and open-source language and vision-language models for gastroenterology clinical reasoning \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12749705/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12749705/)  
12. I Tested 12 “Best-in-Class” PDF Table Extraction Tools, and the Results Were Appalling | by Mark Kramer | Medium, accessed on March 10, 2026, [https://medium.com/@kramermark/i-tested-12-best-in-class-pdf-table-extraction-tools-and-the-results-were-appalling-f8a9991d972e](https://medium.com/@kramermark/i-tested-12-best-in-class-pdf-table-extraction-tools-and-the-results-were-appalling-f8a9991d972e)  
13. Turn Complex Documents into Usable Data with VLM, NVIDIA Nemotron Parse 1.1, accessed on March 10, 2026, [https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/](https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/)  
14. From PDFs to AI-ready structured data: a deep dive \- Explosion, accessed on March 10, 2026, [https://explosion.ai/blog/pdfs-nlp-structured-data](https://explosion.ai/blog/pdfs-nlp-structured-data)  
15. MeDocVL: A Visual Language Model for Medical Document Understanding and Parsing, accessed on March 10, 2026, [https://arxiv.org/html/2602.06402v1](https://arxiv.org/html/2602.06402v1)  
16. \\pipeline: Unlocking Trillions of Tokens in PDFs with Vision Language Models \- arXiv, accessed on March 10, 2026, [https://arxiv.org/html/2502.18443v1](https://arxiv.org/html/2502.18443v1)  
17. Knowledge Graph Generation \- Neo4j, accessed on March 10, 2026, [https://neo4j.com/blog/developer/knowledge-graph-generation/](https://neo4j.com/blog/developer/knowledge-graph-generation/)  
18. Deciphering clinical abbreviations with a privacy protecting machine learning system \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9718734/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718734/)  
19. Disambiguating Clinical Abbreviations by One-to-All Classification: Algorithm Development and Validation Study \- Semantic Scholar, accessed on March 10, 2026, [https://pdfs.semanticscholar.org/cdc0/43b8d8b3f91f27bf9beb97ea23cc4c00827e.pdf](https://pdfs.semanticscholar.org/cdc0/43b8d8b3f91f27bf9beb97ea23cc4c00827e.pdf)  
20. Disambiguating Clinical Abbreviations by One-to-All Classification: Algorithm Development and Validation Study \- JMIR Medical Informatics, accessed on March 10, 2026, [https://medinform.jmir.org/2024/1/e56955](https://medinform.jmir.org/2024/1/e56955)  
21. MeDAL Dataset \- Kaggle, accessed on March 10, 2026, [https://www.kaggle.com/datasets/xhlulu/medal-emnlp](https://www.kaggle.com/datasets/xhlulu/medal-emnlp)  
22. Medical Abbreviation Disambiguation with Large Language Models: Zero- and Few-Shot Evaluation on the MeDAL Dataset \- bioRxiv.org, accessed on March 10, 2026, [https://www.biorxiv.org/content/10.1101/2025.09.12.675926v1.full.pdf](https://www.biorxiv.org/content/10.1101/2025.09.12.675926v1.full.pdf)  
23. Disambiguation of acronyms in clinical narratives with large language models | Journal of the American Medical Informatics Association | Oxford Academic, accessed on March 10, 2026, [https://academic.oup.com/jamia/article/31/9/2040/7699035](https://academic.oup.com/jamia/article/31/9/2040/7699035)  
24. Deciphering clinical abbreviations with privacy protecting ML \- Google Research, accessed on March 10, 2026, [https://research.google/blog/deciphering-clinical-abbreviations-with-privacy-protecting-ml/](https://research.google/blog/deciphering-clinical-abbreviations-with-privacy-protecting-ml/)  
25. Leveraging Medical Knowledge Graphs Into Large Language Models for Diagnosis Prediction: Design and Application Study \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11894347/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11894347/)  
26. Use of SNOMED CT in Large Language Models: Scoping Review \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11494256/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494256/)  
27. Clinical Knowledge Graph: Structure & Applications \- Emergent Mind, accessed on March 10, 2026, [https://www.emergentmind.com/topics/clinical-knowledge-graph](https://www.emergentmind.com/topics/clinical-knowledge-graph)  
28. Systematized Nomenclature of Medicine–Clinical Terminology (SNOMED CT) Clinical Use Cases in the Context of Electronic Health Record Systems \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9941898/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9941898/)  
29. Integrating SNOMED CT into the UMLS: An Exploration of Different Views of Synonymy and Quality of Editing \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC1174894/](https://pmc.ncbi.nlm.nih.gov/articles/PMC1174894/)  
30. Large language models for intelligent RDF knowledge graph construction: results from medical ontology mapping \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12061982/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061982/)  
31. KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis \- arXiv, accessed on March 10, 2026, [https://arxiv.org/html/2412.16833v1](https://arxiv.org/html/2412.16833v1)  
32. Brittleness and Promise: Knowledge Graph–Based Reward Modeling for Diagnostic Reasoning \- arXiv, accessed on March 10, 2026, [https://arxiv.org/html/2509.18316v1](https://arxiv.org/html/2509.18316v1)  
33. How to Create a Knowledge Graph?, accessed on March 10, 2026, [https://web.stanford.edu/class/cs520/2020/notes/How\_To\_Create\_A\_Knowledge\_Graph.html](https://web.stanford.edu/class/cs520/2020/notes/How_To_Create_A_Knowledge_Graph.html)  
34. Reification in Graph Databases: Benefits and Challenges | by Volodymyr Pavlyshyn, accessed on March 10, 2026, [https://ai.plainenglish.io/reification-in-graph-databases-benefits-and-challenges-f1334f9b28b6](https://ai.plainenglish.io/reification-in-graph-databases-benefits-and-challenges-f1334f9b28b6)  
35. An empirical study on Resource Description Framework reification for trustworthiness in knowledge graphs \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8634049/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8634049/)  
36. Graph databases, complex data, and the case for a structured hypergraph \- TypeDB, accessed on March 10, 2026, [https://typedb.com/blog/the-case-for-a-structured-hypergraph](https://typedb.com/blog/the-case-for-a-structured-hypergraph)  
37. Graph Data Modeling: All About Relationships | by David Allen | Neo4j Developer Blog, accessed on March 10, 2026, [https://medium.com/neo4j/graph-data-modeling-all-about-relationships-5060e46820ce](https://medium.com/neo4j/graph-data-modeling-all-about-relationships-5060e46820ce)  
38. Dynamic Control Flow in CUDA Graphs with Conditional Nodes | NVIDIA Technical Blog, accessed on March 10, 2026, [https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)  
39. Designing Reliable LLM Pipelines Across PDFs, Use Cases, and ..., accessed on March 10, 2026, [https://community.databricks.com/t5/technical-blog/ai-newsletter-series-part-2-designing-a-useful-llm-pipeline/ba-p/145873](https://community.databricks.com/t5/technical-blog/ai-newsletter-series-part-2-designing-a-useful-llm-pipeline/ba-p/145873)  
40. Investigations on using Evidence-Based GraphRag Pipeline using LLM Tailored for Answering USMLE Medical Exam Questions \- medRxiv, accessed on March 10, 2026, [https://www.medrxiv.org/content/10.1101/2025.05.03.25325604.full.pdf](https://www.medrxiv.org/content/10.1101/2025.05.03.25325604.full.pdf)  
41. Advanced RAG Techniques for High-Performance LLM Applications \- Graph Database & Analytics \- Neo4j, accessed on March 10, 2026, [https://neo4j.com/blog/genai/advanced-rag-techniques/](https://neo4j.com/blog/genai/advanced-rag-techniques/)  
42. Memgraph vs Neo4j: Graph Database Comparison \- PuppyGraph, accessed on March 10, 2026, [https://www.puppygraph.com/blog/memgraph-vs-neo4j](https://www.puppygraph.com/blog/memgraph-vs-neo4j)  
43. Neo4j vs Memgraph — How to choose a graph database? | by Ivan Despot | Medium, accessed on March 10, 2026, [https://gdespot.medium.com/neo4j-vs-memgraph-how-to-choose-a-graph-database-babdd8d0f81d?source=author\_recirc-----297b103c37cc----3----------------------------](https://gdespot.medium.com/neo4j-vs-memgraph-how-to-choose-a-graph-database-babdd8d0f81d?source=author_recirc-----297b103c37cc----3----------------------------)  
44. The Battle Between Graph Databases: ArangoDB vs Neo4J \- Incora Software, accessed on March 10, 2026, [https://incora.software/insights/graph-databases-arango-vs-neo4j](https://incora.software/insights/graph-databases-arango-vs-neo4j)  
45. ArangoDB vs. Memgraph, accessed on March 10, 2026, [https://memgraph.com/blog/arangodb-vs-memgraph](https://memgraph.com/blog/arangodb-vs-memgraph)  
46. LocalStorage vs. IndexedDB vs. Cookies vs. OPFS vs. WASM-SQLite | RxDB \- JavaScript Database, accessed on March 10, 2026, [https://rxdb.info/articles/localstorage-indexeddb-cookies-opfs-sqlite-wasm.html](https://rxdb.info/articles/localstorage-indexeddb-cookies-opfs-sqlite-wasm.html)  
47. Benchmark of data processing libraries on the browser including Arquero, Sqlite WASM and Duckdb WASM \- GitHub, accessed on March 10, 2026, [https://github.com/timlrx/browser-data-processing-benchmarks](https://github.com/timlrx/browser-data-processing-benchmarks)  
48. Kùzu Wasm, accessed on March 10, 2026, [https://unswdb.github.io/kuzu-wasm/](https://unswdb.github.io/kuzu-wasm/)  
49. GitHub \- kuzudb/kuzu: Embedded property graph database built for speed. Vector search and full-text search built in. Implements Cypher., accessed on March 10, 2026, [https://github.com/kuzudb/kuzu](https://github.com/kuzudb/kuzu)  
50. prrao87/kuzudb-study: Benchmark study on Kuzu, an embedded graph database, on an artificial social network dataset \- GitHub, accessed on March 10, 2026, [https://github.com/prrao87/kuzudb-study](https://github.com/prrao87/kuzudb-study)  
51. What is the difference between using Neo4j for graph analytics and using python networkx for graph analytics? \- Neo4j Community, accessed on March 10, 2026, [https://community.neo4j.com/t/what-is-the-difference-between-using-neo4j-for-graph-analytics-and-using-python-networkx-for-graph-analytics/31005](https://community.neo4j.com/t/what-is-the-difference-between-using-neo4j-for-graph-analytics-and-using-python-networkx-for-graph-analytics/31005)  
52. Neo4j vs Memgraph \- How to Choose a Graph Database?, accessed on March 10, 2026, [https://memgraph.com/blog/neo4j-vs-memgraph](https://memgraph.com/blog/neo4j-vs-memgraph)  
53. Graphical Database Architecture For Clinical Trials \- NC A\&T SU Bluford Library's Aggie Digital Collections and Scholarship, accessed on March 10, 2026, [https://digital.library.ncat.edu/cgi/viewcontent.cgi?article=1321\&context=theses](https://digital.library.ncat.edu/cgi/viewcontent.cgi?article=1321&context=theses)  
54. Comparing relational to graph database \- Getting Started \- Neo4j, accessed on March 10, 2026, [https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/graphdb-vs-rdbms/](https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/graphdb-vs-rdbms/)  
55. Kùzu, an extremely fast embedded graph database \- The Data Quarry, accessed on March 10, 2026, [https://thedataquarry.com/blog/embedded-db-2/](https://thedataquarry.com/blog/embedded-db-2/)  
56. GraphRAG Implementation with LlamaIndex \- V2, accessed on March 10, 2026, [https://developers.llamaindex.ai/python/examples/cookbooks/graphrag\_v2/](https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/)  
57. Improved Knowledge Graph Creation with LangChain and LlamaIndex \- Memgraph, accessed on March 10, 2026, [https://memgraph.com/blog/improved-knowledge-graph-creation-langchain-llamaindex](https://memgraph.com/blog/improved-knowledge-graph-creation-langchain-llamaindex)  
58. From RAG to GraphRAG: Knowledge Graphs, Ontologies and Smarter AI | GoodData, accessed on March 10, 2026, [https://www.gooddata.com/blog/from-rag-to-graphrag-knowledge-graphs-ontologies-and-smarter-ai/](https://www.gooddata.com/blog/from-rag-to-graphrag-knowledge-graphs-ontologies-and-smarter-ai/)  
59. In-Browser Codebase to Knowledge Graph generator : r/LocalLLaMA \- Reddit, accessed on March 10, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1nqgio2/inbrowser\_codebase\_to\_knowledge\_graph\_generator/](https://www.reddit.com/r/LocalLLaMA/comments/1nqgio2/inbrowser_codebase_to_knowledge_graph_generator/)  
60. A Comparison of Javascript Graph / Network Visualisation Libraries \- Cylynx, accessed on March 10, 2026, [https://www.cylynx.io/blog/a-comparison-of-javascript-graph-network-visualisation-libraries/](https://www.cylynx.io/blog/a-comparison-of-javascript-graph-network-visualisation-libraries/)  
61. Graph visualization efficiency of popular web-based libraries \- PMC \- NIH, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12061801/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061801/)  
62. Top 10 JavaScript Libraries for Knowledge Graph Visualization \- Focal AI, accessed on March 10, 2026, [https://www.getfocal.co/post/top-10-javascript-libraries-for-knowledge-graph-visualization](https://www.getfocal.co/post/top-10-javascript-libraries-for-knowledge-graph-visualization)  
63. Sigma.js, accessed on March 10, 2026, [https://www.sigmajs.org/](https://www.sigmajs.org/)  
64. Top 13 JavaScript graph visualization libraries \- Linkurious, accessed on March 10, 2026, [https://linkurious.com/blog/top-javascript-graph-libraries/](https://linkurious.com/blog/top-javascript-graph-libraries/)  
65. Renderers \- Sigma.js, accessed on March 10, 2026, [https://www.sigmajs.org/docs/advanced/renderers/](https://www.sigmajs.org/docs/advanced/renderers/)  
66. React Sigma.js: The Practical Guide to Interactive Graph ... \- Menudo, accessed on March 10, 2026, [https://www.menudo.com/react-sigma-js-the-practical-guide-to-interactive-graph-visualization-in-react/](https://www.menudo.com/react-sigma-js-the-practical-guide-to-interactive-graph-visualization-in-react/)  
67. You Want a Fast, Easy-To-Use, and Popular Graph Visualization Tool? Pick Two\!, accessed on March 10, 2026, [https://memgraph.com/blog/you-want-a-fast-easy-to-use-and-popular-graph-visualization-tool](https://memgraph.com/blog/you-want-a-fast-easy-to-use-and-popular-graph-visualization-tool)  
68. Running JS physics in a webworker \- proof of concept \- DEV Community, accessed on March 10, 2026, [https://dev.to/jerzakm/running-js-physics-in-a-webworker-part-1-proof-of-concept-ibj](https://dev.to/jerzakm/running-js-physics-in-a-webworker-part-1-proof-of-concept-ibj)  
69. The Best Libraries and Methods to Render Large Force-Directed Graphs on the Web, accessed on March 10, 2026, [https://weber-stephen.medium.com/the-best-libraries-and-methods-to-render-large-network-graphs-on-the-web-d122ece2f4dc](https://weber-stephen.medium.com/the-best-libraries-and-methods-to-render-large-network-graphs-on-the-web-d122ece2f4dc)  
70. Graph Algorithms in Neo4j: PageRank, accessed on March 10, 2026, [https://neo4j.com/blog/graph-data-science/graph-algorithms-neo4j-pagerank/](https://neo4j.com/blog/graph-data-science/graph-algorithms-neo4j-pagerank/)  
71. Approaches to measure class importance in Knowledge Graphs | PLOS One, accessed on March 10, 2026, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252862](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252862)  
72. PageRank Algorithm for Graph Databases \- Memgraph, accessed on March 10, 2026, [https://memgraph.com/blog/pagerank-algorithm-for-graph-databases](https://memgraph.com/blog/pagerank-algorithm-for-graph-databases)  
73. PageRank Algorithm-Based Recommendation System for Construction Safety Guidelines, accessed on March 10, 2026, [https://www.mdpi.com/2075-5309/14/10/3041](https://www.mdpi.com/2075-5309/14/10/3041)  
74. Designing and Engineering for Progressive Disclosure \- Jim Nielsen's Blog, accessed on March 10, 2026, [https://blog.jim-nielsen.com/2019/designing-and-engineering-progressive-disclosure/](https://blog.jim-nielsen.com/2019/designing-and-engineering-progressive-disclosure/)  
75. LLM-as-a-Judge: automated evaluation of search query parsing using large language models \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12319771/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12319771/)  
76. Using LLM-as-a-judge ‍⚖️ for an automated and versatile evaluation \- Hugging Face Open-Source AI Cookbook, accessed on March 10, 2026, [https://huggingface.co/learn/cookbook/en/llm\_judge](https://huggingface.co/learn/cookbook/en/llm_judge)  
77. Introduction to LLM Metrics | DeepEval by Confident AI, accessed on March 10, 2026, [https://deepeval.com/docs/metrics-introduction](https://deepeval.com/docs/metrics-introduction)  
78. LLM-as-a-Judge Simply Explained: The Complete Guide to Run LLM Evals at Scale, accessed on March 10, 2026, [https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)  
79. Can LLMs be Good Graph Judge for Knowledge Graph Construction? \- ACL Anthology, accessed on March 10, 2026, [https://aclanthology.org/2025.emnlp-main.554.pdf](https://aclanthology.org/2025.emnlp-main.554.pdf)  
80. Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs, accessed on March 10, 2026, [https://arxiv.org/html/2511.12817v1](https://arxiv.org/html/2511.12817v1)  
81. Essential LLM evaluation metrics for AI quality control: From error analysis to binary checks, accessed on March 10, 2026, [https://langwatch.ai/blog/essential-llm-evaluation-metrics-for-ai-quality-control](https://langwatch.ai/blog/essential-llm-evaluation-metrics-for-ai-quality-control)  
82. Operationalizing Large Language Models for Clinical Research Data Extraction: Methods, Quality Control, and Governance \- PMC, accessed on March 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12932350/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12932350/)  
83. Can LLMs be Good Graph Judge for Knowledge Graph Construction? \- arXiv.org, accessed on March 10, 2026, [https://arxiv.org/html/2411.17388v3](https://arxiv.org/html/2411.17388v3)  
84. Fair Use \- Copyright Information, accessed on March 10, 2026, [https://copyright.psu.edu/copyright-basics/fair-use/](https://copyright.psu.edu/copyright-basics/fair-use/)  
85. Code of Best Practices in Fair Use for Scholarly Research in Communication \- Center for Media and Social Impact, accessed on March 10, 2026, [https://cmsimpact.org/code/code-best-practices-fair-use-scholarly-research-communication/](https://cmsimpact.org/code/code-best-practices-fair-use-scholarly-research-communication/)  
86. Text and Data Mining and Fair Use in the United States1 Background No \- ARL.org, accessed on March 10, 2026, [https://www.arl.org/wp-content/uploads/2015/06/TDM-5JUNE2015.pdf](https://www.arl.org/wp-content/uploads/2015/06/TDM-5JUNE2015.pdf)  
87. Copyright Safety for Generative AI | Published in Houston Law Review, accessed on March 10, 2026, [https://houstonlawreview.org/article/92126-copyright-safety-for-generative-ai](https://houstonlawreview.org/article/92126-copyright-safety-for-generative-ai)  
88. Fair use rights to conduct text and data mining and use artificial intelligence tools are essential for UC research and teaching \- Office of Scholarly Communication, accessed on March 10, 2026, [https://osc.universityofcalifornia.edu/2024/03/fair-use-tdm-ai-restrictive-agreements/](https://osc.universityofcalifornia.edu/2024/03/fair-use-tdm-ai-restrictive-agreements/)  
89. Copyright and the Progress of Science: Why Text and Data Mining Is Lawful, accessed on March 10, 2026, [https://lawreview.law.ucdavis.edu/sites/g/files/dgvnsk15026/files/media/documents/53-2\_Carroll.pdf](https://lawreview.law.ucdavis.edu/sites/g/files/dgvnsk15026/files/media/documents/53-2_Carroll.pdf)