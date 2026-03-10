import os
import json
import argparse
from typing import List
from .rasterizer import rasterize_pdf
from .vlm_parser import VLMParser, MedicalPageChunk

def run_ingestion(pdf_path: str, output_file: str, limit: int = None, mock: bool = False):
    """
    Main entry point for the medical PDF ingestion pipeline.
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
    results = []
    source_filename = os.path.basename(pdf_path)
    
    # Load existing progress if available
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing chunks from {output_file}.")
        except:
            pass

    processed_pages = {r["page_number"] for r in results}
    pages_to_process = []
    
    for i, img_path in enumerate(image_paths):
        page_num = i + 1
        if page_num not in processed_pages:
            pages_to_process.append((img_path, page_num))

    if not pages_to_process:
        print("All requested pages are already processed. Skipping VLM initialization.")
    elif mock:
        print("MOCK MODE: Simulating VLM parsing...")
        for img_path, page_num in pages_to_process:
            results.append({
                "source_file": source_filename,
                "page_number": page_num,
                "text_content": f"Mock text content for page {page_num}",
                "mentions": [],
                "tables": [],
                "clinical_shorthand_detected": []
            })
    else:
        parser = VLMParser()
        for img_path, page_num in pages_to_process:
            page_data = parser.parse_page(img_path, page_num, source_filename)
            if page_data:
                results.append(page_data.dict())
                # Incremental save
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
            else:
                print(f"Skipping page {page_num} due to error.")
                
    # Phase 3: Final save (redundant but safe)
    print(f"Final save of {len(results)} chunks to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Knowledge Graph Ingestion Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the source medical PDF")
    parser.add_argument("--out", type=str, default="data/interim/raw_chunks.json", help="Output path for JSON chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pages to process")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no VLM)")
    
    args = parser.parse_args()
    run_ingestion(args.pdf, args.out, args.limit, args.mock)
