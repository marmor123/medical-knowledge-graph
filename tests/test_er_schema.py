import os
import json
import pytest
import shutil
import kuzu
from src.etl.resolver import MedicalResolver
from src.etl.db_loader import KuzuLoader
from src.etl.validator import MedicalValidator

def test_resolver_semantic():
    """Verify SapBERT can resolve synonyms to same CUI."""
    resolver = MedicalResolver()
    
    # "SOB" and "Shortness of breath" should ideally map to C0013404
    res1 = resolver.resolve("SOB")
    res2 = resolver.resolve("Shortness of breath")
    
    assert res1["cui"] == "C0013404"
    assert res2["cui"] == "C0013404"
    assert res1["confidence"] > 0.7

def test_kuzu_schema_load_jsonl():
    """Verify end-to-end load of Entity-Mention-Chunk schema using JSONL."""
    db_path = "tests/test_db_load"
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)
        
    loader = KuzuLoader(db_path=db_path)
    
    # Create sample data in JSONL format
    sample_data = [
        {
            "source_file": "test.pdf",
            "page_number": 45,
            "text_content": "Patient presents with SOB and history of Diabetes.",
            "mentions": [
                {"text": "SOB", "role": "Symptom"},
                {"text": "Diabetes", "role": "Diagnosis"}
            ]
        }
    ]
    
    sample_file = "tests/sample_load.jsonl"
    with open(sample_file, "w") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
        
    loader.load_chunks(sample_file)
    conn = loader.conn
    
    # Check Entity count
    res = conn.execute("MATCH (e:Entity) RETURN count(*)")
    count = res.get_next()[0]
    assert count >= 2
    
    # Check REFERS_TO relationship
    res = conn.execute("MATCH (m:Mention)-[:REFERS_TO]->(e:Entity) WHERE m.text = 'SOB' RETURN e.cui")
    cui = res.get_next()[0]
    assert cui == "C0013404"

def test_validator_logic():
    """Verify MedicalValidator can sample data and generate report."""
    # Use a different path to avoid lock issues with previous test if not fully closed
    db_path = "tests/test_db_validator"
    if os.path.exists(db_path):
        if os.path.isdir(db_path): shutil.rmtree(db_path)
        else: os.remove(db_path)

    # First populate it
    loader = KuzuLoader(db_path=db_path)
    sample_data = [{"source_file": "v.pdf", "page_number": 1, "text_content": "test", "mentions": [{"text": "SOB", "role": "Symptom"}]}]
    sample_file = "tests/validator_sample.jsonl"
    with open(sample_file, "w") as f:
        f.write(json.dumps(sample_data[0]) + "\n")
    loader.load_chunks(sample_file)

    # Now validate
    loader.close() # RELEASE LOCK
    validator = MedicalValidator(db_path=db_path)
    samples = validator.fetch_triples(limit=5)
    assert len(samples) > 0
    
    report_path = "tests/test_validation_report.json"
    report = validator.run_validation(output_report=report_path, limit=5)
    
    assert os.path.exists(report_path)
    assert "accuracy" in report["metadata"]

if __name__ == "__main__":
    pytest.main([__file__])
