import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import numpy as np

class MedicalResolver:
    """
    Resolves medical mentions to canonical Concept Unique Identifiers (CUIs).
    Uses SapBERT for semantic embedding and a local concept dictionary for mapping.
    """
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        print(f"Initializing MedicalResolver with {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Local MVP dictionary (In a real scenario, this would be a large vector DB or FAISS index)
        self.concept_db = {
            "Diabetes Mellitus": "C0011849",
            "Type 2 Diabetes": "C0011849",
            "Shortness of breath": "C0013404",
            "SOB": "C0013404",
            "Dyspnea": "C0013404",
            "Myocardial Infarction": "C0027051",
            "Heart Attack": "C0027051",
            "Smoking": "C0037369",
            "Hypertension": "C0020538",
            "High Blood Pressure": "C0020538",
            "Fever": "C0015967",
            "LDH High": "C0202054"
        }
        
        # Pre-calculate embeddings for the dictionary for semantic matching
        self.db_embeddings = {}
        self._build_index()

    def _build_index(self):
        """Builds a semantic index of the concept dictionary."""
        for name in self.concept_db.keys():
            self.db_embeddings[name] = self._get_embedding(name)

    def _get_embedding(self, text: str):
        """Generates a SapBERT embedding for a given string."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=25)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use [CLS] token embedding as sentence representation
        return outputs.last_hidden_state[0, 0, :].numpy()

    def resolve(self, mention_text: str) -> Dict[str, str]:
        """
        Maps a mention string to the closest canonical entity in the dictionary.
        Returns a dict with 'cui' and 'canonical_name'.
        """
        mention_emb = self._get_embedding(mention_text)
        
        best_match = None
        highest_sim = -1.0
        
        for name, db_emb in self.db_embeddings.items():
            # Cosine similarity
            sim = np.dot(mention_emb, db_emb) / (np.linalg.norm(mention_emb) * np.linalg.norm(db_emb))
            if sim > highest_sim:
                highest_sim = sim
                best_match = name
        
        # Threshold for resolution (MVP: 0.7)
        if highest_sim > 0.7:
            return {
                "cui": self.concept_db[best_match],
                "canonical_name": best_match,
                "confidence": float(highest_sim)
            }
        else:
            return {
                "cui": "UNKNOWN",
                "canonical_name": mention_text,
                "confidence": float(highest_sim)
            }

if __name__ == "__main__":
    resolver = MedicalResolver()
    print(resolver.resolve("SOB"))
    print(resolver.resolve("Type 2 Diabetes"))
