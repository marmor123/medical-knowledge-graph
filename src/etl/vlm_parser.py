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
    mentions: List[MedicalMention] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)
    clinical_shorthand_detected: List[Dict[str, str]] = Field(default_factory=list, description="Maps shorthand to full terms if found")

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
        # If it ends with a dangling key or truncated value
        if s.endswith(','): s = s[:-1]
        
        # Count braces
        open_braces = s.count('{')
        close_braces = s.count('}')
        if open_braces > close_braces:
            s += '}' * (open_braces - close_braces)
            
        # Count brackets
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
        
        prompt = (
            "You are a medical informatics expert. Analyze this medical textbook page image. "
            "Output strictly in valid JSON format. "
            "Do not include any conversational filler or markdown code blocks like ```json. "
            "Schema: {'source_file': 'str', 'page_number': int, 'text_content': 'str', "
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

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # DEBUG LOGGING
            print(f"DEBUG: RAW VLM OUTPUT (Page {page_number}):\n{output_text}\nDEBUG: END RAW OUTPUT")

            # Clean markdown if present
            clean_text = output_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            data = None
            if json_repair:
                try:
                    data = json_repair.loads(clean_text)
                except Exception as e:
                    print(f"json_repair failed: {e}")
            
            if not data:
                try:
                    # Try extraction via regex
                    json_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        data = json.loads(self._manual_repair(clean_text))
                except Exception as e:
                    print(f"Standard json.loads failed after repair attempt: {e}")
                    raise e
            
            # Ensure mandatory fields
            data["source_file"] = source_file
            data["page_number"] = page_number
            
            return MedicalPageChunk(**data)
        except Exception as e:
            print(f"CRITICAL ERROR parsing page {page_number}: {e}")
            return None

if __name__ == "__main__":
    pass
