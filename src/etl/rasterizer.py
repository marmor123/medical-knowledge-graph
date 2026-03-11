import os
import fitz  # PyMuPDF
from PIL import Image
from typing import List, Optional

def rasterize_pdf(pdf_path: str, output_folder: str, dpi: int = 300, start_page: int = 1, limit: Optional[int] = None, mock: bool = False) -> List[str]:
    """
    Converts specific PDF pages into high-resolution images (PNG) using PyMuPDF.
    Optimized to only render the requested range.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if mock:
        print("MOCK RASTERIZATION: Generating dummy images...")
        image_paths = []
        for i in range(2): 
            image_path = os.path.join(output_folder, f"page_{i+1}.png")
            Image.new('RGB', (800, 1000), color = (73, 109, 137)).save(image_path)
            image_paths.append(image_path)
        return image_paths

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Calculate range
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, start_idx + limit) if limit else total_pages
        
        print(f"Rasterizing {pdf_path} (Pages {start_idx+1} to {end_idx}) at {dpi} DPI...")
        
        image_paths = []
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        for i in range(start_idx, end_idx):
            image_name = f"page_{i+1}.png"
            image_path = os.path.join(output_folder, image_name)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                continue

            # Render specific page
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            pix.save(image_path)
            image_paths.append(image_path)
            print(f"Saved {image_path}")
            
        doc.close()
        return image_paths
    except Exception as e:
        print(f"Error during rasterization with PyMuPDF: {e}")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rasterize_pdf(sys.argv[1], "data/interim/images")
