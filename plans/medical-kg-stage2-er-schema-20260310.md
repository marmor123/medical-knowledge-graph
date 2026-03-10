# Implementation Plan: Medical Knowledge Graph - Stage 2 (Entity Resolution & Kùzu Schema)

### ## Approach
We will transition from raw text chunks to a dual-layer **Entity-Mention** schema. This decouples the physical occurrence of a term (Mention) from its canonical clinical concept (Entity), solving the "multiple topics per page" and "entity resolution" issues.
- **Why this solution:** Decoupling allows a single node (e.g., "Diabetes") to point to all its mentions (as a risk factor, disease, or treatment context) without duplicating the central concept. SapBERT-tiny provides high-precision medical entity linking in a small footprint.
- **Alternatives considered:** Flat triples (Subject-Predicate-Object), which lose provenance and struggle with n-ary medical logic.

### ## Steps
1. **Implement SapBERT Resolution Service** (30 min)
   - File: `src/etl/resolver.py`
   - Use `transformers` to load `cambridgeltl/SapBERT-UMLS-2020AB-all-prediction-v2-tiny`.
   - Implement `MedicalResolver` class to map extracted strings to UMLS CUIs.
   ```python
   class MedicalResolver:
       def __init__(self, model_name="cambridgeltl/SapBERT-UMLS-2020AB-all-prediction-v2-tiny"):
           self.tokenizer = AutoTokenizer.from_pretrained(model_name)
           self.model = AutoModel.from_pretrained(model_name)
       
       def resolve(self, text: str) -> str:
           # Logic to map to nearest CUI
           pass
   ```

2. **Update VLM Prompting & Models** (15 min)
   - File: `src/etl/vlm_parser.py`
   - Update `MedicalPageChunk` to include a list of `Mentions` with `role` and `context`.
   - Adjust VLM prompt to explicitly distinguish between clinical roles.

3. **Design Kùzu Schema** (20 min)
   - File: `src/db/schema.cypher`
   - Define nodes: `Entity` (CUI, Name), `Mention` (Text, Role), `Chunk` (Source, Page).
   - Define edges: `REFERS_TO`, `APPEARS_IN`, `LEADS_TO` (reified via Protocol nodes).

4. **ETL Orchestration for DB Loading** (30 min)
   - File: `src/etl/db_loader.py`
   - Load JSON chunks -> Resolve Entities -> Bulk insert into Kùzu via Python API.

5. **Validation with LLM-as-a-Judge** (20 min)
   - File: `src/etl/validator.py`
   - Verify that resolved entities correctly link back to their source mentions.

### ## Timeline
| Phase | Duration |
|-------|----------|
| SapBERT Service | 30 min |
| VLM Schema Update | 15 min |
| Kùzu Schema Design | 20 min |
| DB Loader | 30 min |
| Validation | 20 min |
| **Total** | **~2 hours** |

### ## Rollback Plan
1. Revert to `raw_chunks.json` if resolution accuracy is below 80%.
2. Clear Kùzu database using `rm -rf data/db/` and reload from backup.

### ## Security Checklist
- [x] Local SapBERT inference (Privacy-preserving).
- [x] Schema enforcement for database integrity.
- [ ] Error handling for failed entity mappings.
