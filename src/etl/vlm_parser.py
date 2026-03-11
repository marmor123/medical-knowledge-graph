import os
import json
import torch
import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration
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
    is_negated: bool = Field(default=False, description="True if the text indicates the absence of the concept (e.g., 'no fever')")

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
            # 1. Handle aliasing
            if 'clinical_concepts' in data and 'mentions' not in data:
                data['mentions'] = data.pop('clinical_concepts')
            if 'clinical_shorthand' in data and 'clinical_shorthand_detected' not in data:
                data['clinical_shorthand_detected'] = data.pop('clinical_shorthand')
            
            # 2. Handle single-item wrapping
            for field in ['mentions', 'tables', 'clinical_shorthand_detected']:
                if field in data and isinstance(data[field], dict):
                    data[field] = [data[field]]
            
            # 3. Handle malformed table rows
            if 'tables' in data and isinstance(data['tables'], list):
                for table in data['tables']:
                    if 'rows' in table and isinstance(table['rows'], list):
                        for i, row in enumerate(table['rows']):
                            if isinstance(row, dict):
                                table['rows'][i] = list(row.values())
        return data

class VLMParser:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", abbrev_path: str = "data/interim/abbreviations.json"):
        """
        Initializes the VLM with extreme RAM efficiency and strict device mapping for parallel jobs.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Initializing VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load Abbreviation Context
        self.abbrev_context = ""
        if os.path.exists(abbrev_path):
            try:
                with open(abbrev_path, "r", encoding="utf-8") as f:
                    abbrevs = json.load(f)
                    sample_list = [f"{k}: {v}" for k, v in list(abbrevs.items())[:150]]
                    self.abbrev_context = "\n".join(sample_list)
                print(f"Loaded {len(abbrevs)} abbreviations for context.")
            except: pass

        print(f"[{time.strftime('%H:%M:%S')}] Loading weights...")
        
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": {"": 0} if torch.cuda.is_available() else "cpu", # Force isolated device mapping
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            print("🚀 Enabling NF4 Quantization...")
            try:
                import bitsandbytes
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except: pass

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"[{time.strftime('%H:%M:%S')}] VLM Ready.")

    def parse_page(self, image_path: str, page_number: int, source_file: str, mode: str = "standard", quality: str = "high") -> Optional[MedicalPageChunk]:
        """
        Processes a single page image using Qwen3-8B visual reasoning.
        """
        print(f"--- Processing Page {page_number} (Mode: {mode}) ---")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        pixel_map = {"high": 1280, "medium": 768, "low": 512}
        px = pixel_map.get(quality, 1280)

        if mode == "abbrev":
            prompt = (
                "You are a medical lexicographer. This image is a page from a medical abbreviation dictionary.\n"
                "Extract every abbreviation and its full expansion.\n"
                "Format strictly as JSON:\n"
                "{\n"
                "  \"clinical_shorthand_detected\": [{\"shorthand\": \"AAA\", \"full_term\": \"abdominal aortic aneurysm\"}],\n"
                "  \"text_content\": \"[Full OCR of the page here]\"\n"
                "}"
            )
        else:
            context_block = f"\nCLINICAL CONTEXT (Common abbreviations in this book):\n{self.abbrev_context}\n" if self.abbrev_context else ""
            prompt = (
                "You are a medical informatics expert. Analyze this medical textbook page.\n"
                "Extract all clinical data into a valid JSON object.\n"
                f"{context_block}\n"
                "CRITICAL CONSTRAINTS:\n"
                "1. YOU MUST USE THESE EXACT KEYS: 'mentions', 'clinical_shorthand_detected', 'tables', 'text_content'.\n"
                "2. 'mentions': List all medical concepts. EACH 'role' MUST be exactly one of: [Symptom, Diagnosis, LabValue, RiskFactor, Treatment].\n"
                "   - 'is_negated': Set to true ONLY if the text explicitly excludes the finding (e.g., 'no fever', 'absence of SOB').\n"
                "3. 'text_content': YOU MUST EXTRACT THE FULL TEXT OF THE PAGE HERE.\n"
                "4. If a field has no data, return an empty list [].\n\n"
                "OUTPUT FORMAT: {\"mentions\": [{\"text\": \"...\", \"role\": \"...\", \"is_negated\": false}], \"clinical_shorthand_detected\": [], \"tables\": [], \"text_content\": \"...\"}"
            )

        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}", "max_pixels": px * px}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Use json_repair or regex fallback
        data = None
        if json_repair:
            try: data = json_repair.loads(output_text)
            except: pass
        if not data:
            json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
            if json_match: data = json.loads(json_match.group(1))
            else: data = json.loads(output_text)
        
        data["source_file"] = source_file
        data["page_number"] = page_number
        if "text_content" not in data: data["text_content"] = "Extraction incomplete"
        
        return MedicalPageChunk(**data)

if __name__ == "__main__":
    pass
