import json
import os

def compile_abbreviations(input_file="data/interim/abbreviations_raw.jsonl", output_file="data/interim/abbreviations.json"):
    """
    Takes raw VLM output from abbreviation pages and creates a clean dictionary.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    abbreviation_map = {}
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunk = json.loads(line)
            
            # 1. Extract from clinical_shorthand_detected field
            shorthand_list = chunk.get("clinical_shorthand_detected", [])
            for item in shorthand_list:
                s = item.get("shorthand")
                f_t = item.get("full_term")
                if s and f_t:
                    abbreviation_map[s.strip()] = f_t.strip()
            
            # 2. Extract from mentions if role was 'Treatment' or 'Diagnosis' but it looks like an acronym
            mentions = chunk.get("mentions", [])
            for m in mentions:
                text = m.get("text", "")
                # If it's all caps and short, it's likely an acronym
                if text.isupper() and 2 <= len(text) <= 6:
                    # We only add if it's not already there
                    if text not in abbreviation_map:
                        abbreviation_map[text] = "UNKNOWN_EXPANSION"

    # Save to a clean JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(abbreviation_map, f, indent=4)
        
    print(f"✅ Compiled {len(abbreviation_map)} abbreviations to {output_file}.")

if __name__ == "__main__":
    compile_abbreviations()
