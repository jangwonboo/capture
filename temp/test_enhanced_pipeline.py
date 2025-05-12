#!/usr/bin/env python
# Test script for the enhanced OCR pipeline

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from llm_ocr import image_folder_to_enhanced_searchable_pdf, merge_enhanced_pdfs, check_mlx_status

def test_enhanced_ocr_pipeline():
    """Test the enhanced OCR pipeline with MLX acceleration."""
    print("Testing enhanced OCR pipeline with MLX...")
    
    # Check MLX status
    mlx_status = check_mlx_status()
    print(f"MLX Status: {mlx_status}")
    
    # Create test directory
    test_dir = Path('test_enhanced')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Create test images to simulate a captured book
    for i in range(1, 4):
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f'Test page {i} with OCR text that needs enhancement', fill=(0, 0, 0))
        img_path = test_dir / f'test_enhanced_{i:03d}.png'
        img.save(img_path)
        print(f"Created test image: {img_path}")
    
    # Set MLX parameters
    mlx_params = {
        "use_mlx": True,
        "model_size": "tiny",  # Use tiny model for faster testing
        "batch_size": 4        # Small batch size for testing
    }
    
    # Process the test book with enhanced OCR
    print("\nProcessing test images with enhanced OCR...")
    pdf_files = image_folder_to_enhanced_searchable_pdf(
        str(test_dir),
        prefix="test_enhanced",
        lang="eng",
        use_openai=False,
        mlx_params=mlx_params
    )
    
    print(f"\nCreated {len(pdf_files)} enhanced PDF files")
    
    # Merge enhanced PDFs
    if pdf_files:
        merged_pdf = test_dir / "test_enhanced_merged.pdf"
        merge_enhanced_pdfs(str(test_dir), "test_enhanced", str(merged_pdf))
        print(f"Merged PDFs into: {merged_pdf}")
    
    # Check results
    print(f"\nFiles in {test_dir} directory:")
    for f in sorted(os.listdir(test_dir)):
        print(f"- {f}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_enhanced_ocr_pipeline()