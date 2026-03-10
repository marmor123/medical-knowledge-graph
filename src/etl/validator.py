import os
import json
import kuzu
import argparse
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class ValidationResult(BaseModel):
    mention: str
    resolved_entity: str
    role: str
    sentence_context: str
    is_correct: bool
    reason: str
    suggested_correction: Optional[str] = None

class MedicalValidator:
    """
    Validates Kùzu graph data using an LLM-as-a-Judge approach.
    """
    def __init__(self, db_path: str = "data/db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def fetch_triples(self, limit: int = 20) -> List[Dict]:
        """Retrieves Mention-Entity-Chunk triples for validation."""
        query = (
            "MATCH (m:Mention)-[:REFERS_TO]->(e:Entity), "
            "(m)-[:APPEARS_IN]->(c:Chunk) "
            "RETURN m.text, m.role, e.name, c.text_content "
            f"LIMIT {limit}"
        )
        res = self.conn.execute(query)
        
        samples = []
        while res.has_next():
            row = res.get_next()
            samples.append({
                "mention": row[0],
                "role": row[1],
                "entity": row[2],
                "context": row[3]
            })
        return samples

    def evaluate_sample(self, sample: Dict, mock: bool = True) -> ValidationResult:
        """Evaluates a single sample using an LLM or Mock logic."""
        mention = sample["mention"]
        entity = sample["entity"]
        role = sample["role"]
        context = sample["context"]

        if mock:
            # Simple heuristic mock
            is_correct = True
            reason = "Automatic pass (Mock Mode)"
            # Simulate a failure if the mention is very different from entity
            if mention.lower() not in entity.lower() and len(mention) > 3:
                # Unless it's a known synonym from our dictionary
                known_synonyms = {"SOB": "Shortness of breath", "Dyspnea": "Shortness of breath"}
                if known_synonyms.get(mention) != entity:
                    is_correct = False
                    reason = f"Mention '{mention}' seems unrelated to Entity '{entity}'"
            
            return ValidationResult(
                mention=mention,
                resolved_entity=entity,
                role=role,
                sentence_context=context,
                is_correct=is_correct,
                reason=reason
            )
        else:
            # In a real scenario, this would call OpenAI/Anthropic API
            # For now, we remain offline-first as per mandates
            return self.evaluate_sample(sample, mock=True)

    def run_validation(self, output_report: str = "data/interim/validation_report.json", limit: int = 20):
        """Orchestrates the validation process and saves a report."""
        print(f"Sampling {limit} triples from Kùzu...")
        samples = self.fetch_triples(limit=limit)
        
        results = []
        correct_count = 0
        
        for sample in samples:
            eval_res = self.evaluate_sample(sample)
            results.append(eval_res.dict())
            if eval_res.is_correct:
                correct_count += 1
        
        accuracy = (correct_count / len(results)) * 100 if results else 0
        
        report = {
            "metadata": {
                "db_path": self.db_path,
                "total_samples": len(results),
                "accuracy": f"{accuracy:.2f}%"
            },
            "detailed_results": results
        }
        
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        print(f"Validation complete. Accuracy: {accuracy:.2f}%. Report saved to {output_report}")
        return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical KG Validator")
    parser.add_argument("--db", type=str, default="data/db", help="Path to Kùzu DB")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to validate")
    
    args = parser.parse_args()
    
    try:
        validator = MedicalValidator(db_path=args.db)
        validator.run_validation(limit=args.limit)
    except Exception as e:
        print(f"Validator failed: {e}")
