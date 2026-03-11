import json
from src.etl.vlm_parser import MedicalPageChunk

def test_recovery_logic():
    print("Testing Parser Recovery Logic...")

    # Case 1: The "Unterminated String" error from Page 4
    # (Simulating a raw string that has a missing close quote or brace)
    broken_raw_4 = """
    {
      "source_file": "test",
      "page_number": 4,
      "text_content": "T-wave inversion (TWI; ≥1 mm; deep if ≥5 mm",
      "mentions": [{"text": "TWI", "role": "Symptom"
    """
    
    # Case 2: The "Input should be a valid list" error from Page 7
    # (Simulating a dict where a list was expected)
    broken_raw_7 = """
    {
      "source_file": "test",
      "page_number": 7,
      "text_content": "CCTA results...",
      "clinical_shorthand_detected": {
        "shorthand": "CCTA",
        "full_term": "Coronary CT Angiography"
      }
    }
    """

    # We can't use VLMParser.parse_page because it tries to load the model
    # So we test the repair and Pydantic validation directly
    from json_repair import repair_json
    
    print("\n--- Testing Case 1 (Truncation) ---")
    repaired_4 = repair_json(broken_raw_4)
    print(f"Repaired JSON: {repaired_4}")
    data_4 = json.loads(repaired_4)
    chunk_4 = MedicalPageChunk(**data_4)
    print(f"✅ Success! Mentions found: {len(chunk_4.mentions)}")

    print("\n--- Testing Case 2 (Dict-to-List Coercion) ---")
    data_7 = json.loads(broken_raw_7)
    chunk_7 = MedicalPageChunk(**data_7)
    print(f"✅ Success! Shorthand coerced to list: {isinstance(chunk_7.clinical_shorthand_detected, list)}")
    print(f"Shorthand count: {len(chunk_7.clinical_shorthand_detected)}")

if __name__ == "__main__":
    try:
        import json_repair
        test_recovery_logic()
    except ImportError:
        print("Please install json-repair to run this test: pip install json-repair")
