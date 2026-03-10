import os
import fitz  # PyMuPDF
from PIL import Image
from typing import List

def rasterize_pdf(pdf_path: str, output_folder: str, dpi: int = 300, mock: bool = False) -> List[str]:
    """
    Converts PDF pages into high-resolution images (PNG) using PyMuPDF.
    
    Args:
        pdf_path: Path to the source PDF file.
        output_folder: Directory to save the rasterized images.
        dpi: Dots per inch for the output images (default 300).
        mock: If True, generates dummy images to bypass PDF processing.
        
    Returns:
        A list of paths to the generated image files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if mock:
        print("MOCK RASTERIZATION: Generating dummy images...")
        image_paths = []
        for i in range(2): # Default to 2 pages for mock
            image_path = os.path.join(output_folder, f"page_{i+1}.png")
            Image.new('RGB', (800, 1000), color = (73, 109, 137)).save(image_path)
            image_paths.append(image_path)
        return image_paths

    print(f"Rasterizing {pdf_path} at {dpi} DPI using PyMuPDF...")
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Matrix for scaling (DPI / 72 since default is 72 DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        for i, page in enumerate(doc):
            image_name = f"page_{i+1}.png"
            image_path = os.path.join(output_folder, image_name)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                continue

            # Render page to a pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Save the pixmap to a PNG file
            pix.save(image_path)
            image_paths.append(image_path)
            print(f"Saved {image_path}")
            
        doc.close()
        return image_paths
    except Exception as e:
        print(f"Error during rasterization with PyMuPDF: {e}")
        return []

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        rasterize_pdf(sys.argv[1], "data/interim/images")
