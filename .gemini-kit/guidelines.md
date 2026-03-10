# Guidelines: Medical Knowledge Graph

## Code Style
- **TypeScript (Frontend):** 
  - Strict mode enabled.
  - Functional components with Hooks.
  - JSDoc for all core graph logic and WASM interactions.
- **Python (ETL):** 
  - PEP 8 compliance.
  - Type hints for all function signatures.
  - Docstrings explaining each ETL phase.
  - Pydantic models for all data parsing and validation.

## Commit Conventions
- **Conventional Commits:** (feat, fix, refactor, docs, chore, test, style).
- **Scope-based:** e.g., `feat(etl): add Qwen3-VL parser`, `fix(ui): improve node hover interaction`.

## Project Standards
- **Zero Hallucination:** Code changes must not introduce generative logic that deviates from the pre-compiled graph data.
- **Offline Integrity:** All runtime logic must be verified to work without an internet connection.
- **Provenance Integrity:** Every data extraction step must strictly preserve page/chapter metadata.

## Workflow
1. **Plan first:** All changes must be preceded by a `/plan`.
2. **Scout:** Always understand the codebase before making changes.
3. **Test:** Write and run tests for every new feature or bug fix.
4. **Review:** Peer review (via agent or human) before final commit.
