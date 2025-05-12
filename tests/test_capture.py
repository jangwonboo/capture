#!/usr/bin/env python3
"""
Test script for e-book capture tool functionality.
Tests key features like window resizing, monitor selection, and capture options.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_test(command, description):
    """Run a test command and report if it was successful."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(command, shell=True, check=False)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
        else:
            print(f"❌ FAILED: Exit code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run a series of tests to verify e-book capture functionality."""
    parser = argparse.ArgumentParser(description='Test e-book capture functionality')
    parser.add_argument('--target-window', type=str, default="",
                      help='Window title to use for testing (default: none)')
    parser.add_argument('--skip-resize', action='store_true', 
                      help='Skip window resize tests')
    parser.add_argument('--skip-monitor', action='store_true',
                      help='Skip monitor selection tests')
    parser.add_argument('--skip-ocr', action='store_true',
                      help='Skip OCR tests')
    parser.add_argument('--skip-capture', action='store_true',
                      help='Skip capture tests')
    
    args = parser.parse_args()
    
    # Create a test output directory
    test_output = Path("test_results")
    os.makedirs(test_output, exist_ok=True)
    print(f"Test output will be saved to: {test_output.absolute()}")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: List available monitors
    total_tests += 1
    if run_test(
        command="python run.py --list-monitors",
        description="List available monitors"
    ):
        success_count += 1
    
    # Test 2: List available book formats
    total_tests += 1
    if run_test(
        command="python run.py --list-formats",
        description="List available book formats"
    ):
        success_count += 1
    
    # Window resize tests
    if not args.skip_resize and args.target_window:
        # Test 3: Resize window to paperback format
        total_tests += 1
        if run_test(
            command=f"python run.py --window \"{args.target_window}\" --book-format paperback --scale 0.7",
            description="Resize window to paperback format"
        ):
            success_count += 1
            time.sleep(2)  # Give time to observe changes
            
        # Test 4: Resize window to small-tablet format
        total_tests += 1
        if run_test(
            command=f"python run.py --window \"{args.target_window}\" --book-format small-tablet --scale 0.6",
            description="Resize window to small-tablet format"
        ):
            success_count += 1
            time.sleep(2)  # Give time to observe changes
    elif not args.skip_resize:
        print("\n⚠️ Skipping window resize tests - no target window specified")
        print("   Use --target-window \"Window Title\" to test window resizing")
    
    # Monitor selection tests
    if not args.skip_monitor:
        # Get monitor count first
        result = subprocess.run("python run.py --list-monitors", 
                                shell=True, capture_output=True, text=True)
        
        # Try to parse monitor info from output
        monitor_count = 0
        for line in result.stdout.splitlines():
            if line.strip().startswith("0") or line.strip().startswith("1"):
                monitor_count += 1
        
        if monitor_count > 1:
            # Test 5: Capture with specific monitor
            total_tests += 1
            if run_test(
                command=f"python run.py --title \"monitor_test\" --monitor 0 --output-dir {test_output} --pages 1",
                description="Capture from primary monitor"
            ):
                success_count += 1
        else:
            print("\n⚠️ Skipping multi-monitor tests - only one monitor detected")
    
    # Capture tests
    if not args.skip_capture:
        # Test 6: Basic capture
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"basic_capture\" --output-dir {test_output} --pages 2 --delay 1",
            description="Basic screen capture (2 pages)"
        ):
            success_count += 1
        
        # Test 7: Capture with cropping
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"crop_test\" --output-dir {test_output} --pages 1 --crop 10,10,90,90",
            description="Capture with cropping (10% on each side)"
        ):
            success_count += 1
    
    # OCR tests
    if not args.skip_ocr:
        # Test 8: OCR on captured image
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"ocr_test\" --output-dir {test_output} --pages 1 --ocr",
            description="OCR on captured image"
        ):
            success_count += 1
        
        # Test 9: OCR with LLM enhancement (only if MLX is available)
        # This will be skipped if MLX isn't available
        total_tests += 1
        if run_test(
            command=f"python run.py --title \"llm_test\" --output-dir {test_output} --pages 1 --ocr --enhance --model-size tiny",
            description="OCR with LLM enhancement (tiny model)"
        ):
            success_count += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    return 0 if success_count == total_tests else 1

if __name__ == "__main__":
    sys.exit(main()) 