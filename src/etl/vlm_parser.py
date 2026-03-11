import os
import json
import torch
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
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
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)

class MedicalMention(BaseModel):
    text: str
    role: str = Field(description="MUST be one of: Symptom, Diagnosis, LabValue, RiskFactor, Treatment")
    context: Optional[str] = Field(None, description="Optional surrounding context")

class MedicalPageChunk(BaseModel):
    source_file: str
    page_number: int
    text_content: str
    mentions: List[MedicalMention] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)
    clinical_shorthand_detected: List[Dict[str, str]] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def coerce_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field in ['mentions', 'tables', 'clinical_shorthand_detected']:
                if field in data and isinstance(data[field], dict):
                    data[field] = [data[field]]
        return data

class VLMParser:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the Qwen2-VL model with GPU optimizations for Colab.
        """
        print(f"Initializing VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model on {self.device} with {self.torch_dtype}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=self.torch_dtype, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded.")

    def _manual_repair(self, s: str) -> str:
        """Last resort repair for common VLM truncation issues."""
        s = s.strip()
        if s.endswith(','): s = s[:-1]
        
        open_braces = s.count('{')
        close_braces = s.count('}')
        if open_braces > close_braces:
            s += '}' * (open_braces - close_braces)
            
        open_brackets = s.count('[')
        close_brackets = s.count(']')
        if open_brackets > close_brackets:
            s += ']' * (open_brackets - close_brackets)
            
        return s

    def parse_page(self, image_path: str, page_number: int, source_file: str) -> Optional[MedicalPageChunk]:
        """
        Processes a single page image and returns structured medical data.
        """
        print(f"--- Processing Page {page_number} ---")
        
        # Highly prescriptive prompt to force role adherence and avoid truncation
        prompt = (
            "Analyze this medical textbook page. Extract all clinical data into a valid JSON object.\n\n"
            "CONSTRAINTS:\n"
            "1. 'mentions': List all medical concepts (Symptoms, Diseases, etc.).\n"
            "   - EACH 'role' MUST be exactly one of: [Symptom, Diagnosis, LabValue, RiskFactor, Treatment].\n"
            "   - DO NOT use any other role names.\n"
            "2. 'clinical_shorthand_detected': List pairs of {'shorthand': '...', 'full_term': '...'}.\n"
            "3. 'tables': Reconstruct any tables found.\n"
            "4. 'text_content': Extract the full text of the page last.\n"
            "5. If a field has no data, return an empty list [].\n\n"
            "OUTPUT FORMAT:\n"
            "{\n"
            "  \"mentions\": [{\"text\": \"...\", \"role\": \"...\", \"context\": \"...\"}],\n"
            "  \"clinical_shorthand_detected\": [],\n"
            "  \"tables\": [],\n"
            "  \"text_content\": \"...\"\n"
            "}"
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

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                # We use a high max_new_tokens to ensure complex pages aren't cut off
                generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # DEBUG LOGGING
            print(f"DEBUG: RAW VLM OUTPUT (Page {page_number}):\n{output_text}\nDEBUG: END RAW OUTPUT")

            clean_text = output_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            data = None
            if json_repair:
                try:
                    data = json_repair.loads(clean_text)
                except:
                    pass
            
            if not data:
                try:
                    json_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        data = json.loads(self._manual_repair(clean_text))
                except Exception as e:
                    print(f"Standard json.loads failed: {e}")
                    raise e
            
            # Ensure mandatory fields are injected by code
            data["source_file"] = source_file
            data["page_number"] = page_number
            
            # Ensure text_content exists (fallback if model failed to return it)
            if "text_content" not in data:
                data["text_content"] = "Extraction incomplete"
            
            return MedicalPageChunk(**data)
        except Exception as e:
            print(f"CRITICAL ERROR parsing page {page_number}: {e}")
            return None

if __name__ == "__main__":
    pass
