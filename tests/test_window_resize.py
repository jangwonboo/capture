#!/usr/bin/env python3
"""
Test script specifically for window resizing functionality.
Tests multiple book formats with different scale factors.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_resize_test(window_title, book_format, scale=0.8, padding=5, delay=3.0):
    """Run a window resize test with the given parameters."""
    command = (
        f"python run.py --window \"{window_title}\" "
        f"--book-format {book_format} --scale {scale} --padding {padding}"
    )
    
    print(f"\n{'='*80}")
    print(f"TEST: Resize window to {book_format} format (scale: {scale}, padding: {padding}%)")
    print(f"COMMAND: {command}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(command, shell=True, check=False)
        success = result.returncode == 0
        
        if success:
            print(f"✅ SUCCESS: Window resize to {book_format}")
            print(f"Waiting {delay} seconds to observe changes...")
            time.sleep(delay)
        else:
            print(f"❌ FAILED: Exit code {result.returncode}")
        
        return success
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run a series of window resize tests."""
    parser = argparse.ArgumentParser(description='Test window resizing functionality')
    parser.add_argument('--window', '-w', type=str, required=True,
                        help='Window title to resize (required)')
    parser.add_argument('--delay', '-d', type=float, default=3.0,
                        help='Delay between tests to observe changes (seconds)')
    parser.add_argument('--formats', type=str, 
                        default="a4,a5,paperback,small-tablet,kindle",
                        help='Comma-separated list of book formats to test')
    parser.add_argument('--list-formats', action='store_true',
                        help='List available formats and exit')
    
    args = parser.parse_args()
    
    # If requested, list formats and exit
    if args.list_formats:
        subprocess.run("python run.py --list-formats", shell=True)
        return 0
    
    # Get formats to test
    formats_to_test = args.formats.split(',')
    print(f"Will test {len(formats_to_test)} book formats on window: \"{args.window}\"")
    print(f"Formats: {', '.join(formats_to_test)}")
    
    success_count = 0
    total_tests = 0
    
    # Test each format with different scale factors
    for book_format in formats_to_test:
        # Test with default scale (0.8)
        total_tests += 1
        if run_resize_test(
            window_title=args.window,
            book_format=book_format,
            scale=0.8,
            delay=args.delay
        ):
            success_count += 1
        
        # Test with smaller scale (0.6)
        total_tests += 1
        if run_resize_test(
            window_title=args.window,
            book_format=book_format,
            scale=0.6,
            delay=args.delay
        ):
            success_count += 1
        
        # Test with larger scale (0.9)
        total_tests += 1
        if run_resize_test(
            window_title=args.window,
            book_format=book_format,
            scale=0.9,
            padding=0,  # No padding for this test
            delay=args.delay
        ):
            success_count += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"RESIZE TEST SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    return 0 if success_count == total_tests else 1

if __name__ == "__main__":
    sys.exit(main()) 