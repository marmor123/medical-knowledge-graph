import os
import json
import torch
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Try to import json_repair for more robust parsing
try:
    import json_repair
except ImportError:
    json_repair = None

# Define our strict output schema
class TableStructure(BaseModel):
    title: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]

class MedicalMention(BaseModel):
    text: str
    role: str = Field(description="One of: Symptom, Diagnosis, LabValue, RiskFactor, Treatment")
    context: Optional[str] = Field(None, description="Optional surrounding context for disambiguation")

class MedicalPageChunk(BaseModel):
    source_file: str
    page_number: int
    text_content: str
    mentions: List[MedicalMention]
    tables: List[TableStructure]
    clinical_shorthand_detected: List[Dict[str, str]] = Field(description="Maps shorthand to full terms if found")

class VLMParser:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the Qwen2-VL model for document understanding.
        """
        print(f"Loading model {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Model loaded on {self.device}.")

    def parse_page(self, image_path: str, page_number: int, source_file: str) -> Optional[MedicalPageChunk]:
        """
        Processes a single page image and returns structured medical data with role-based mentions.
        """
        print(f"Parsing page {page_number} from {image_path}...")
        
        # Refined prompt for dual-layer schema
        prompt = (
            "You are a medical informatics expert. Analyze this medical textbook page image. "
            "1. Extract all text preserving reading order. "
            "2. Identify all medical concepts and categorize them as 'mentions' with a role: "
            "   - 'Symptom', 'Diagnosis', 'LabValue', 'RiskFactor', 'Treatment'. "
            "3. Identify clinical shorthand (e.g., 'SOB', 'CAD') and map them to their full expansion. "
            "4. Reconstruct all tables into structured format. "
            "Output strictly in valid JSON format matching this schema: "
            "{'source_file': 'str', 'page_number': int, 'text_content': 'str', "
            "'mentions': [{'text': 'str', 'role': 'str', 'context': 'str'}], "
            "'tables': [{'title': 'str', 'headers': [], 'rows': [[]]}], "
            "'clinical_shorthand_detected': [{'shorthand': 'str', 'full_term': 'str'}]}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{os.path.abspath(image_path)}",
                        "max_pixels": 512 * 512,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Robust JSON extraction and repair
            data = None
            if json_repair:
                # Use json_repair to handle malformed strings or truncation
                try:
                    data = json_repair.loads(output_text)
                except:
                    pass
            
            if not data:
                # Fallback to manual extraction if json_repair fails or is missing
                json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                else:
                    data = json.loads(output_text) # Last ditch effort
            
            # Ensure mandatory fields are present
            data["source_file"] = source_file
            data["page_number"] = page_number
            
            # Validate with Pydantic
            return MedicalPageChunk(**data)
        except Exception as e:
            print(f"Error parsing page {page_number}: {e}")
            print(f"Raw output snippet: {output_text[:500] if 'output_text' in locals() else 'None'}...")
            return None

if __name__ == "__main__":
    pass
