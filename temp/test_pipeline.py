#!/usr/bin/env python
# Test script for the OCR pipeline

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from main import process_with_standard_ocr, get_mlx_status

def test_standard_ocr_pipeline():
    """Test the standard OCR pipeline."""
    print("Testing standard OCR pipeline...")
    
    # Check MLX status
    mlx_status = get_mlx_status()
    print(f"MLX Status: {mlx_status}")
    
    # Create test directory
    test_dir = Path('test_book')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Create test images to simulate a captured book
    for i in range(1, 4):
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f'Test page {i}', fill=(0, 0, 0))
        img_path = test_dir / f'test_book_{i:03d}.png'
        img.save(img_path)
        print(f"Created test image: {img_path}")
    
    # Process the test book
    process_with_standard_ocr(test_dir, 'test_book')
    
    # Check results
    print(f"\nFiles in {test_dir} directory:")
    for f in sorted(os.listdir(test_dir)):
        print(f"- {f}")
    
    # Check for final PDF
    final_pdf = test_dir / "test_book_ocr.pdf"
    if final_pdf.exists():
        print(f"\nSuccess! Final PDF created: {final_pdf}")
    else:
        print(f"\nWarning: Final PDF not found at {final_pdf}")
    
    print("Test completed.")

if __name__ == "__main__":
    test_standard_ocr_pipeline()