#!/usr/bin/env python
# Test script for PDF creation and merging

import os
import shutil
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import functions from llm_ocr
from llm_ocr import create_searchable_pdf_with_enhanced_text, merge_enhanced_pdfs

def test_pdf_creation():
    """Test PDF creation and merging with a simple example."""
    test_dir = 'test_merge'
    
    # Clean up and recreate test directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Create test images (white 300x300 images)
    img1 = Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 255)
    img2 = Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 255)
    
    # Save test images
    img1_path = f'{test_dir}/test_img1.png'
    img2_path = f'{test_dir}/test_img2.png'
    img1.save(img1_path)
    img2.save(img2_path)
    print(f'Created test images: {img1_path}, {img2_path}')
    
    # Create PDFs from images with enhanced text
    pdf1_path = f'{test_dir}/test_img1_enhanced_ocr.pdf'
    pdf2_path = f'{test_dir}/test_img2_enhanced_ocr.pdf'
    
    try:
        create_searchable_pdf_with_enhanced_text(img1_path, 'Test text for page 1', pdf1_path)
        print(f'Successfully created PDF: {pdf1_path}')
    except Exception as e:
        print(f'Error creating PDF 1: {e}')
    
    try:
        create_searchable_pdf_with_enhanced_text(img2_path, 'Test text for page 2', pdf2_path)
        print(f'Successfully created PDF: {pdf2_path}')
    except Exception as e:
        print(f'Error creating PDF 2: {e}')
    
    # Merge PDFs
    merged_pdf_path = f'{test_dir}/merged.pdf'
    try:
        merge_enhanced_pdfs(test_dir, 'test_img', merged_pdf_path)
        print(f'Successfully merged PDFs: {merged_pdf_path}')
    except Exception as e:
        print(f'Error merging PDFs: {e}')
    
    # List files in test directory
    print('\nFiles in test directory:')
    for f in os.listdir(test_dir):
        print(f'- {f}')
    
    print(f'\nTest completed. Test files are in: {os.path.abspath(test_dir)}')

if __name__ == "__main__":
    test_pdf_creation()