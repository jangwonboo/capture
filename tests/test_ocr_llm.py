#!/usr/bin/env python3
"""
Test script for OCR and LLM enhancement functionality.
Tests OCR with different parameters and LLM enhancement with different models.
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path
import time

def run_test(command, description, delay=0):
    """Run a test command and report if it was successful."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(command, shell=True, check=False)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if delay > 0:
                print(f"Waiting {delay} seconds...")
                time.sleep(delay)
            return True
        else:
            print(f"❌ FAILED: Exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def capture_test_image(output_dir, title="test_image"):
    """Capture a test image for OCR processing."""
    command = f"python run.py --title {title} --output-dir {output_dir} --pages 1"
    success = run_test(command, f"Capturing test image for OCR tests")
    
    if success:
        # Find the captured image
        image_files = glob.glob(f"{output_dir}/{title}_*.png")
        if image_files:
            return image_files[0]
    
    return None

def main():
    """Run OCR and LLM enhancement tests."""
    parser = argparse.ArgumentParser(description='Test OCR and LLM enhancement')
    parser.add_argument('--skip-capture', action='store_true',
                        help='Skip capturing new test images')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM enhancement tests')
    parser.add_argument('--input-image', type=str,
                        help='Use specified image for OCR tests instead of capturing')
    
    args = parser.parse_args()
    
    # Create test output directory
    test_output = Path("test_ocr_results")
    os.makedirs(test_output, exist_ok=True)
    print(f"Test output will be saved to: {test_output.absolute()}")
    
    # Get or capture test image
    test_image = None
    if args.input_image:
        if os.path.exists(args.input_image):
            test_image = args.input_image
            print(f"Using provided image: {test_image}")
        else:
            print(f"⚠️ Specified image not found: {args.input_image}")
    
    if test_image is None and not args.skip_capture:
        print("Capturing test image...")
        test_image = capture_test_image(test_output)
    
    if test_image is None and not args.input_image and not args.skip_capture:
        print("❌ Failed to capture or find test image. Cannot proceed with OCR tests.")
        return 1
    
    success_count = 0
    total_tests = 0
    
    # Test OCR with different language options
    if test_image:
        # Test 1: Basic OCR with default settings
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"ocr_default\" --output-dir {test_output} --ocr --pages 1",
            description="OCR with default settings"
        ):
            success_count += 1
        
        # Test 2: OCR with English language only
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"ocr_eng\" --output-dir {test_output} --ocr --language eng --pages 1",
            description="OCR with English language only"
        ):
            success_count += 1
        
        # Test 3: OCR with different PSM settings
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"ocr_psm6\" --output-dir {test_output} --ocr --psm 6 --pages 1",
            description="OCR with PSM 6 (Assume single uniform block of text)"
        ):
            success_count += 1
        
        # Test 4: OCR with higher DPI
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"ocr_dpi600\" --output-dir {test_output} --ocr --dpi 600 --pages 1",
            description="OCR with higher DPI (600)"
        ):
            success_count += 1
    
        # LLM enhancement tests
        if not args.skip_llm:
            # Test 5: LLM enhancement with tiny model
            total_tests += 1
            if run_test(
                command=f"python run.py --title \"llm_tiny\" --output-dir {test_output} --ocr --enhance --model-size tiny --pages 1",
                description="OCR with LLM enhancement (tiny model)"
            ):
                success_count += 1
            
            # Test 6: LLM enhancement with small model
            total_tests += 1
            if run_test(
                command=f"python run.py --title \"llm_small\" --output-dir {test_output} --ocr --enhance --model-size small --pages 1",
                description="OCR with LLM enhancement (small model)"
            ):
                success_count += 1
            
            # Test 7: LLM enhancement with no-MLX option
            total_tests += 1
            if run_test(
                command=f"python run.py --title \"llm_no_mlx\" --output-dir {test_output} --ocr --enhance --model-size tiny --no-mlx --pages 1",
                description="OCR with LLM enhancement (no MLX)"
            ):
                success_count += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"OCR/LLM TEST SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    if test_image is None:
        print("⚠️ Some tests were skipped because no test image was available.")
    
    return 0 if success_count == total_tests or test_image is None else 1

if __name__ == "__main__":
    sys.exit(main()) 