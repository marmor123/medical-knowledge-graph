import os
import json
import argparse
import time
import traceback
from typing import List, Optional
from .rasterizer import rasterize_pdf
from .vlm_parser import VLMParser, MedicalPageChunk

# Singleton-like cache for the parser to prevent redundant loading in notebooks
_GLOBAL_VLM_PARSER = None

def get_vlm_parser(model_name: str = "Qwen/Qwen3-VL-8B-Instruct") -> VLMParser:
    """Helper to ensure the VLM is only loaded once per session."""
    global _GLOBAL_VLM_PARSER
    if _GLOBAL_VLM_PARSER is None:
        _GLOBAL_VLM_PARSER = VLMParser(model_name=model_name)
    return _GLOBAL_VLM_PARSER

def log_error(error_msg: str):
    """Logs error messages to a dedicated file for remote debugging."""
    with open("kaggle_errors.log", "a", encoding="utf-8") as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"--- ERROR: {timestamp} ---\n")
        f.write(error_msg + "\n")
        f.write(traceback.format_exc() + "\n\n")

def run_ingestion(pdf_path: str, output_file: str, limit: int = None, mock: bool = False, parser: Optional[VLMParser] = None):
    """
    Main entry point for the medical PDF ingestion pipeline.
    Uses JSONL for O(1) incremental saving and robust error logging.
    """
    print(f"Starting ingestion for {pdf_path}...")
    
    # Setup directories
    image_folder = "data/interim/images"
    os.makedirs(image_folder, exist_ok=True)
    
    # Phase 1: Rasterize PDF to images
    image_paths = rasterize_pdf(pdf_path, image_folder, mock=mock)
    if not image_paths:
        print("Failed to rasterize PDF. Aborting.")
        return
        
    if limit:
        image_paths = image_paths[:limit]
        print(f"Limited ingestion to first {limit} pages.")
        
    # Phase 2: VLM Parsing
    processed_pages = set()
    source_filename = os.path.basename(pdf_path)
    
    # Load existing progress from JSONL
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        processed_pages.add(chunk["page_number"])
            print(f"Detected {len(processed_pages)} already processed pages.")
        except Exception as e:
            print(f"Progress check warning: {e}")

    pages_to_process = []
    for i, img_path in enumerate(image_paths):
        page_num = i + 1
        if page_num not in processed_pages:
            pages_to_process.append((img_path, page_num))

    if not pages_to_process:
        print("All requested pages are already processed. Skipping VLM initialization.")
    elif mock:
        print("MOCK MODE: Simulating VLM parsing...")
        with open(output_file, "a", encoding="utf-8") as f:
            for img_path, page_num in pages_to_process:
                chunk = {
                    "source_file": source_filename,
                    "page_number": page_num,
                    "text_content": f"Mock text content for page {page_num}",
                    "mentions": [],
                    "tables": [],
                    "clinical_shorthand_detected": []
                }
                f.write(json.dumps(chunk) + "\n")
    else:
        # Use provided parser or get/create the global one
        if parser is None:
            parser = get_vlm_parser()
            
        with open(output_file, "a", encoding="utf-8") as f:
            for img_path, page_num in pages_to_process:
                try:
                    page_data = parser.parse_page(img_path, page_num, source_filename)
                    if page_data:
                        # O(1) Append to JSONL
                        f.write(json.dumps(page_data.dict()) + "\n")
                        f.flush() # Ensure it's written to disk immediately
                    else:
                        err_msg = f"VLM returned None for page {page_num}"
                        print(err_msg)
                        log_error(err_msg)
                except Exception as e:
                    err_msg = f"CRITICAL failure on page {page_num}: {e}"
                    print(err_msg)
                    log_error(err_msg)
                
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Knowledge Graph Ingestion Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the source medical PDF")
    parser.add_argument("--out", type=str, default="data/interim/raw_chunks.jsonl", help="Output path for JSONL chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pages to process")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no VLM)")
    
    args = parser.parse_args()
    run_ingestion(args.pdf, args.out, args.limit, args.mock)
