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
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", abbrev_path: str = "data/interim/abbreviations.json"):
        """
        Initializes the VLM with extreme RAM efficiency and explicit device mapping.
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
                    sample_list = [f"{k}: {v}" for k, v in list(abbrevs.items())[:100]]
                    self.abbrev_context = "\n".join(sample_list)
            except: pass

        print(f"[{time.strftime('%H:%M:%S')}] Loading weights...")
        
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": {"": 0} if torch.cuda.is_available() else "cpu", # Force local GPU
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            print("🚀 Enabling NF4 Quantization...")
            try:
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
        pixel_map = {"high": 1280, "medium": 768, "low": 512}
        px = pixel_map.get(quality, 1280)

        if mode == "abbrev":
            prompt = "Extract medical abbreviations and their full expansions from this page. Output JSON."
        else:
            context_block = f"\nABBREVIATIONS:\n{self.abbrev_context}\n" if self.abbrev_context else ""
            prompt = (
                "Extract clinical concepts (Symptom, Diagnosis, LabValue, RiskFactor, Treatment) "
                f"and tables from this page. {context_block} Output JSON."
            )

        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}", "max_pixels": px * px}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
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
        return MedicalPageChunk(**data)

if __name__ == "__main__":
    pass
