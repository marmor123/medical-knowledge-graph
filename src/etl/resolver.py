import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import numpy as np

class MedicalResolver:
    """
    Resolves medical mentions to canonical Concept Unique Identifiers (CUIs).
    Uses SapBERT for semantic embedding and a vectorized matrix for fast mapping.
    """
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        print(f"Initializing MedicalResolver with {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to evaluation mode
        
        # Local MVP dictionary
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
        
        self.concept_names = list(self.concept_db.keys())
        self.db_matrix = None
        self._build_index()

    def _build_index(self):
        """Builds a semantic index of the concept dictionary using a vectorized matrix."""
        embeddings = []
        for name in self.concept_names:
            emb = self._get_embedding(name)
            # Normalize embedding upfront for faster cosine similarity via dot product
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        
        # Stack into (N, D) matrix
        self.db_matrix = torch.cat(embeddings, dim=0)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Generates a SapBERT embedding for a given string as a PyTorch tensor."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=25).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use [CLS] token embedding as sentence representation
        return outputs.last_hidden_state[:, 0, :]

    def resolve(self, mention_text: str) -> Dict[str, str]:
        """
        Maps a mention string to the closest canonical entity in the dictionary using vectorization.
        Returns a dict with 'cui' and 'canonical_name'.
        """
        mention_emb = self._get_embedding(mention_text)
        # Normalize
        mention_emb = mention_emb / mention_emb.norm(dim=-1, keepdim=True)
        
        # Vectorized cosine similarity: (1, D) @ (D, N) -> (1, N)
        # Since both are normalized, dot product is cosine similarity
        similarities = torch.mm(mention_emb, self.db_matrix.t()).squeeze(0)
        
        best_match_idx = torch.argmax(similarities).item()
        highest_sim = similarities[best_match_idx].item()
        best_match_name = self.concept_names[best_match_idx]
        
        # Threshold for resolution (MVP: 0.7)
        if highest_sim > 0.7:
            return {
                "cui": self.concept_db[best_match_name],
                "canonical_name": best_match_name,
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
