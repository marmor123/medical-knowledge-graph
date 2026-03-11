import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import numpy as np

class MedicalResolver:
    """
    Resolves medical mentions to canonical Concept Unique Identifiers (CUIs).
    Uses a hybrid approach:
    1. Local abbreviation dictionary (extracted from the book).
    2. Vectorized SapBERT semantic search for canonical mapping.
    """
    def __init__(self, 
                 model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 abbrev_path: str = "data/interim/abbreviations.json",
                 threshold: float = 0.7):
        print(f"Initializing MedicalResolver with {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = threshold
        
        # 1. Load Abbreviation Dictionary
        self.abbreviations = {}
        if os.path.exists(abbrev_path):
            try:
                with open(abbrev_path, "r", encoding="utf-8") as f:
                    self.abbreviations = json.load(f)
                print(f"Loaded {len(self.abbreviations)} abbreviations from {abbrev_path}")
            except Exception as e:
                print(f"Warning: Failed to load abbreviations: {e}")

        # 2. Local Canonical Concept DB (The "Identity" nodes in our graph)
        self.concept_db = {
            "Diabetes Mellitus": "C0011849",
            "Type 2 Diabetes": "C0011849",
            "Shortness of breath": "C0013404",
            "Dyspnea": "C0013404",
            "Myocardial Infarction": "C0027051",
            "Heart Attack": "C0027051",
            "Smoking": "C0037369",
            "Hypertension": "C0020538",
            "High Blood Pressure": "C0020538",
            "Fever": "C0015967",
            "LDH High": "C0202054",
            "Abdominal Aortic Aneurysm": "C0000768"
        }
        
        self.concept_names = list(self.concept_db.keys())
        self.db_matrix = None
        self._build_index()

    def _build_index(self):
        """Builds a semantic index of the canonical concepts."""
        embeddings = []
        for name in self.concept_names:
            emb = self._get_embedding(name)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        self.db_matrix = torch.cat(embeddings, dim=0)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Generates a SapBERT embedding."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def resolve(self, mention_text: str) -> Dict[str, str]:
        """
        Resolves a mention. If it's a known abbreviation, resolves the expansion.
        """
        original_text = mention_text.strip()
        search_text = original_text
        
        # Check abbreviation dictionary first
        if original_text in self.abbreviations:
            search_text = self.abbreviations[original_text]
            print(f"Mapping abbreviation: {original_text} -> {search_text}")

        # Vectorized search
        mention_emb = self._get_embedding(search_text)
        mention_emb = mention_emb / mention_emb.norm(dim=-1, keepdim=True)
        
        similarities = torch.mm(mention_emb, self.db_matrix.t()).squeeze(0)
        best_match_idx = torch.argmax(similarities).item()
        highest_sim = similarities[best_match_idx].item()
        best_match_name = self.concept_names[best_match_idx]
        
        if highest_sim > self.threshold:
            return {
                "cui": self.concept_db[best_match_name],
                "canonical_name": best_match_name,
                "confidence": float(highest_sim),
                "resolved_via": "abbreviation" if search_text != original_text else "semantic"
            }
        else:
            return {
                "cui": "UNKNOWN",
                "canonical_name": original_text,
                "confidence": float(highest_sim),
                "resolved_via": "none"
            }

if __name__ == "__main__":
    # Create dummy abbrev file for testing if it doesn't exist
    os.makedirs("data/interim", exist_ok=True)
    with open("data/interim/abbreviations.json", "w") as f:
        json.dump({"AAA": "Abdominal Aortic Aneurysm", "SOB": "Shortness of breath"}, f)
        
    resolver = MedicalResolver()
    print(f"Resolving 'AAA': {resolver.resolve('AAA')}")
    print(f"Resolving 'Diabetes': {resolver.resolve('Diabetes')}")
