import os
import json
import torch
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from PIL import Image

# Official Transformers classes for Qwen3-VL
try:
    from transformers import (
        Qwen3VLForConditionalGeneration, 
        AutoProcessor, 
        BitsAndBytesConfig
    )
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
        from transformers import AutoProcessor, BitsAndBytesConfig
        HAS_QWEN3 = True
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration
        from transformers import AutoProcessor, BitsAndBytesConfig
        HAS_QWEN3 = False

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
        Initializes the state-of-the-art Qwen3-VL 8B model.
        Loads abbreviation context if available to improve extraction precision.
        """
        print(f"Initializing VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load Abbreviation Context for Prompt Injection
        self.abbrev_context = ""
        if os.path.exists(abbrev_path):
            try:
                with open(abbrev_path, "r", encoding="utf-8") as f:
                    abbrevs = json.load(f)
                    # Limit to top 150 relevant abbreviations to save prompt tokens
                    sample_list = [f"{k}: {v}" for k, v in list(abbrevs.items())[:150]]
                    self.abbrev_context = "\n".join(sample_list)
                print(f"Loaded {len(abbrevs)} abbreviations for prompt context.")
            except:
                pass

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
                print("❌ bitsandbytes not found.")

        # Load the model
        try:
            print(f"Attempting load with class: {Qwen3VLForConditionalGeneration.__name__}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs
            )
        except Exception as e:
            print(f"Primary load failed: {e}. Falling back to AutoModelForVision2Seq...")
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, **load_kwargs
            )

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model loaded successfully. Class: {self.model.__class__.__name__}")

    def parse_page(self, image_path: str, page_number: int, source_file: str, mode: str = "standard") -> Optional[MedicalPageChunk]:
        """
        Processes a single page image using Qwen3-8B visual reasoning.
        """
        print(f"--- Processing Page {page_number} (Mode: {mode}) ---")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
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
            # Inject abbreviation context into the prompt
            context_block = ""
            if self.abbrev_context:
                context_block = f"\nCLINICAL CONTEXT (Common abbreviations in this book):\n{self.abbrev_context}\n"

            prompt = (
                "You are a medical informatics expert. Analyze this medical textbook page.\n"
                "Extract all clinical data into a valid JSON object.\n"
                f"{context_block}\n"
                "CONSTRAINTS:\n"
                "1. 'mentions': List all medical concepts.\n"
                "   - EACH 'role' MUST be exactly one of: [Symptom, Diagnosis, LabValue, RiskFactor, Treatment].\n"
                "2. 'clinical_shorthand_detected': List pairs of {'shorthand': '...', 'full_term': '...'}.\n"
                "3. 'tables': Reconstruct any tables found.\n"
                "4. 'text_content': Extract the full text of the page last.\n"
                "OUTPUT FORMAT: {\"mentions\": [], \"clinical_shorthand_detected\": [], \"tables\": [], \"text_content\": \"...\"}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{os.path.abspath(image_path)}",
                        "max_pixels": 1280 * 1280, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Use json_repair or regex fallback
        data = None
        if json_repair:
            try: data = json_repair.loads(output_text)
            except: pass
        
        if not data:
            json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(output_text)
        
        data["source_file"] = source_file
        data["page_number"] = page_number
        if "text_content" not in data: data["text_content"] = "Extraction incomplete"
        
        return MedicalPageChunk(**data)

if __name__ == "__main__":
    pass
