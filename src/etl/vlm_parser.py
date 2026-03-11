import os
import json
import torch
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from PIL import Image

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
except ImportError:
    from transformers import AutoModel as AutoModelForVision2Seq
    from transformers import AutoProcessor, BitsAndBytesConfig

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
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """
        Initializes the state-of-the-art Qwen3-VL 8B model.
        Optimized for Kaggle T4 GPUs (using float16 instead of bfloat16).
        """
        print(f"Initializing SOTA VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # T4 Optimization: Use float16 (T4 does not support bfloat16 natively)
        # bfloat16 will run on T4 but it is extremely slow due to lack of hardware support.
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading Qwen3-8B on {self.device} with {self.torch_dtype}...")
        
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if torch.cuda.is_available():
            print("🚀 Enabling NF4 4-bit quantization for Qwen3-8B...")
            try:
                import bitsandbytes
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("❌ bitsandbytes not found. Quantization disabled.")

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            print(f"Primary load failed: {e}. Falling back to basic AutoModel...")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model loaded successfully. Class: {self.model.__class__.__name__}")

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
        Processes a single page image using Qwen3-8B visual reasoning.
        """
        print(f"--- Processing Page {page_number} (Qwen3-8B) ---")
        
        prompt = (
            "You are a medical informatics expert. Analyze this medical textbook page. "
            "Extract all clinical data into a valid JSON object.\n\n"
            "CONSTRAINTS:\n"
            "1. 'mentions': List all medical concepts.\n"
            "   - EACH 'role' MUST be exactly one of: [Symptom, Diagnosis, LabValue, RiskFactor, Treatment].\n"
            "2. 'clinical_shorthand_detected': List pairs of {'shorthand': '...', 'full_term': '...'}.\n"
            "3. 'tables': Reconstruct any tables found.\n"
            "4. 'text_content': Extract the full text of the page last.\n"
            "5. If a field has no data, return an empty list [].\n\n"
            "OUTPUT FORMAT:\n"
            "{\n"
            "  \"mentions\": [],\n"
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
                        "max_pixels": 1280 * 1280, # Increased for tiny 'Pocket Medicine' text
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
            
            print(f"DEBUG: RAW VLM OUTPUT (Page {page_number}):\n{output_text[:500]}...\nDEBUG: END RAW OUTPUT")

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
            
            data["source_file"] = source_file
            data["page_number"] = page_number
            if "text_content" not in data:
                data["text_content"] = "Extraction incomplete"
            
            return MedicalPageChunk(**data)
        except Exception as e:
            print(f"CRITICAL ERROR parsing page {page_number}: {e}")
            return None

if __name__ == "__main__":
    pass
