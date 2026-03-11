import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import numpy as np
import re

class MedicalResolver:
    """
    Resolves medical mentions to canonical Concept Unique Identifiers (CUIs).
    Enhanced to handle multi-mappings and clinical shorthand symbols.
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
        
        # Symbol normalization map
        self.symbol_map = {
            "Δ": "change in ",
            "β": "beta ",
            "α": "alpha ",
            "↑": "increased ",
            "↓": "decreased ",
            "→": "leads to ",
            "Σ": "sum of "
        }
        
        self.abbreviations = {}
        if os.path.exists(abbrev_path):
            try:
                with open(abbrev_path, "r", encoding="utf-8") as f:
                    self.abbreviations = json.load(f)
                print(f"Loaded {len(self.abbreviations)} abbreviations.")
            except: pass

        # Expanded MVP Concept DB
        self.concept_db = {
            "Diabetes Mellitus": "C0011849",
            "Diabetes": "C0011849",
            "Shortness of breath": "C0013404",
            "SOB": "C0013404",
            "Dyspnea": "C0013404",
            "Myocardial Infarction": "C0027051",
            "Heart Attack": "C0027051",
            "Smoking": "C0037369",
            "Hypertension": "C0020538",
            "High Blood Pressure": "C0020538",
            "Fever": "C0015967",
            "LDH High": "C0202054",
            "Abdominal Aortic Aneurysm": "C0000768",
            "Beta-blocker": "C0001645",
            "Mental Status Change": "C0233407",
            "Aortic Insufficiency": "C0003504",
            "Adrenal Insufficiency": "C0001623"
        }
        
        self.concept_names = list(self.concept_db.keys())
        self.db_matrix = None
        self._build_index()

    def _normalize_text(self, text: str) -> str:
        """Replaces medical symbols with their text equivalents."""
        for sym, replacement in self.symbol_map.items():
            text = text.replace(sym, replacement)
        return text.strip()

    def _build_index(self):
        """Builds semantic index."""
        embeddings = []
        for name in self.concept_names:
            normalized = self._normalize_text(name)
            emb = self._get_embedding(normalized)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        self.db_matrix = torch.cat(embeddings, dim=0)

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def resolve(self, mention_text: str, context: str = "") -> Dict[str, str]:
        """
        Resolves a mention using symbol normalization, abbreviation expansion, 
        and context-aware multi-mapping selection.
        """
        original_text = self._normalize_text(mention_text)
        candidates = [original_text]
        
        # 1. Expand abbreviation
        if mention_text in self.abbreviations:
            expansion = self.abbreviations[mention_text]
            # Handle multi-mappings
            candidates.extend(expansion.split(' ')) 

        best_overall_match = None
        highest_overall_sim = -1.0
        resolution_method = "semantic"

        for cand in candidates:
            if not cand.strip(): continue
            cand_norm = self._normalize_text(cand)
            emb = self._get_embedding(cand_norm)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            
            similarities = torch.mm(emb, self.db_matrix.t()).squeeze(0)
            idx = torch.argmax(similarities).item()
            sim = similarities[idx].item()
            
            # If we have context, boost candidates that appear in context
            if context and cand.lower() in context.lower():
                sim += 0.1 

            if sim > highest_overall_sim:
                highest_overall_sim = sim
                best_overall_match = self.concept_names[idx]
                if cand != original_text: resolution_method = "abbreviation"

        if highest_overall_sim > self.threshold:
            return {
                "cui": self.concept_db[best_overall_match],
                "canonical_name": best_overall_match,
                "confidence": float(highest_overall_sim),
                "resolved_via": resolution_method
            }
        else:
            return {
                "cui": "UNKNOWN",
                "canonical_name": mention_text,
                "confidence": float(highest_overall_sim),
                "resolved_via": "none"
            }

if __name__ == "__main__":
    resolver = MedicalResolver()
    # Test symbol normalization
    print(f"Resolving 'ΔMS': {resolver.resolve('ΔMS')}")
    # Test multi-mapping via context (Hypothetical)
    print(f"Resolving 'AI' in Cardiology context: {resolver.resolve('AI', context='heart valve aortic')}")
