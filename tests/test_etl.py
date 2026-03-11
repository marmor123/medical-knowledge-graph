import os
import json
import pytest
from src.etl.rasterizer import rasterize_pdf
from src.etl.vlm_parser import MedicalPageChunk
from src.etl.processor import run_ingestion

def test_rasterizer_real_pdf():
    """Verify PyMuPDF can open and rasterize at least 1 page of the target PDF."""
    pdf_path = "Pocket Medicine 9th Edition 2026.pdf"
    output_folder = "tests/temp_images"
    
    # Clean up from previous runs
    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
        
    image_paths = rasterize_pdf(pdf_path, output_folder, dpi=72) # Low DPI for speed
    assert len(image_paths) > 0
    assert os.path.exists(image_paths[0])
    assert image_paths[0].endswith(".png")

def test_mock_ingestion_pipeline_jsonl():
    """Verify the end-to-end pipeline in mock mode using JSONL."""
    pdf_path = "Pocket Medicine 9th Edition 2026.pdf"
    output_file = "tests/test_chunks.jsonl"
    
    if os.path.exists(output_file):
        os.remove(output_file)

    # 1. First run (1 page)
    run_ingestion(pdf_path, output_file, limit=1, mock=True)
    
    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    
    data = json.loads(lines[0])
    assert data["page_number"] == 1

    # 2. Second run (Incremental - total 2 pages)
    run_ingestion(pdf_path, output_file, limit=2, mock=True)
    
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2 # O(1) append confirmed if it didn't duplicate line 1
    
    assert json.loads(lines[0])["page_number"] == 1
    assert json.loads(lines[1])["page_number"] == 2

def test_schema_validation():
    """Verify MedicalPageChunk Pydantic model with sample data."""
    sample_data = {
        "source_file": "test.pdf",
        "page_number": 1,
        "text_content": "Example medical text.",
        "mentions": [
            {"text": "Diabetes", "role": "Diagnosis"}
        ],
        "tables": [
            {
                "title": "Table 1",
                "headers": ["Col1", "Col2"],
                "rows": [["val1", "val2"]]
            }
        ]
    }
    
    chunk = MedicalPageChunk(**sample_data)
    assert chunk.page_number == 1
    assert len(chunk.mentions) == 1
    assert chunk.tables[0].headers == ["Col1", "Col2"]

if __name__ == "__main__":
    pytest.main([__file__])
