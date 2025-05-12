#!/usr/bin/env python3
"""
E-book Page Capture Tool - Main script

A tool to capture screenshots of e-book pages for OCR processing.
Supports:
- Multiple book formats and window sizing
- Automatic page turning
- OCR text extraction
- Multi-monitor support
- Optional LLM text enhancement
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Import core modules
from core.window import get_window_by_title, resize_window_to_book_format, get_monitor_info
from core.capture import capture_screen, consecutive_capture, crop_image
from core.ocr import perform_ocr, ocr_image_file, ocr_multiple_files, save_ocr_results
from core.llm import LLMEnhancer

# Import utility modules
from utils.config import get_settings, get_book_formats, get_output_dir, get_cli_options
from utils.logging_utils import configure_logging
from utils.platform_utils import get_system_name, is_apple_silicon

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='E-book Page Capture Tool')
    
    # Load CLI options from configuration
    cli_options = get_cli_options()
    
    # Add required arguments
    for name, option in cli_options.get('required', {}).items():
        kwargs = {
            'help': option.get('help', ''),
            'required': option.get('required', False)
        }
        
        # Add type if specified
        if 'type' in option:
            kwargs['type'] = eval(option['type'])
            
        # Add default if specified
        if 'default' in option:
            kwargs['default'] = option['default']
            
        # Add choices if specified
        if 'choices' in option:
            kwargs['choices'] = option['choices']
            
        # Add action if specified
        if 'action' in option:
            kwargs['action'] = option['action']
        
        parser.add_argument(*option['flags'], **kwargs)
    
    # Add option groups
    for group_name, group_info in cli_options.get('groups', {}).items():
        arg_group = parser.add_argument_group(group_info.get('title', group_name))
        
        # Add options within this group
        for option_name, option in group_info.get('options', {}).items():
            kwargs = {
                'help': option.get('help', '')
            }
            
            # Add type if specified
            if 'type' in option:
                kwargs['type'] = eval(option['type'])
                
            # Add default if specified
            if 'default' in option:
                kwargs['default'] = option['default']
                
            # Add choices if specified
            if 'choices' in option:
                kwargs['choices'] = option['choices']
                
            # Add action if specified
            if 'action' in option:
                kwargs['action'] = option['action']
            
            arg_group.add_argument(*option['flags'], **kwargs)
    
    return parser.parse_args()

def list_book_formats():
    """List available book formats."""
    book_formats = get_book_formats()
    
    print("Available book formats:")
    print("-" * 80)
    print(f"{'Name':<15} {'Width (mm)':<12} {'Height (mm)':<12} {'Description'}")
    print("-" * 80)
    
    for name, format_info in book_formats.items():
        # Skip device formats which are in pixels, not mm
        if format_info.get('is_device', False):
            continue
            
        width = format_info.get('width', 0)
        height = format_info.get('height', 0)
        description = format_info.get('description', '')
        
        print(f"{name:<15} {width:<12} {height:<12} {description}")
    
    print("\nDevice formats (dimensions in pixels):")
    print("-" * 80)
    print(f"{'Name':<15} {'Width (px)':<12} {'Height (px)':<12} {'Description'}")
    print("-" * 80)
    
    for name, format_info in book_formats.items():
        # Only show device formats
        if not format_info.get('is_device', False):
            continue
            
        width = format_info.get('width', 0)
        height = format_info.get('height', 0)
        description = format_info.get('description', '')
        
        print(f"{name:<15} {width:<12} {height:<12} {description}")

def list_monitors():
    """List available monitors."""
    monitor_info = get_monitor_info()
    
    print("Available monitors:")
    print("-" * 60)
    print(f"{'Index':<8} {'Position':<15} {'Size':<15} {'Primary'}")
    print("-" * 60)
    
    for i, (x, y, width, height) in enumerate(monitor_info['monitors']):
        is_primary = (i == monitor_info.get('primary_index', 0))
        position = f"({x}, {y})"
        size = f"{width}x{height}"
        primary_str = "Yes" if is_primary else "No"
        
        print(f"{i:<8} {position:<15} {size:<15} {primary_str}")
    
    print(f"\nVirtual screen size: {monitor_info['all'][0]}x{monitor_info['all'][1]}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle list options
    if args.list_formats:
        list_book_formats()
        return 0
    
    if args.list_monitors:
        list_monitors()
        return 0
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.title:
        output_dir = get_output_dir(args.title)
    else:
        output_dir = get_output_dir("capture_" + time.strftime("%Y%m%d_%H%M%S"))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        quiet=args.quiet,
        title=args.title,
        output_dir=output_dir
    )
    
    logger = logging.getLogger('main')
    logger.info(f"E-book Page Capture Tool started")
    logger.info(f"System: {get_system_name()}")
    logger.info(f"Output directory: {output_dir}")
    
    # If window and book format are specified, resize the window
    if args.window and args.book_format:
        logger.info(f"Resizing window '{args.window}' to {args.book_format} format")
        
        success = resize_window_to_book_format(
            window_title=args.window,
            book_format=args.book_format,
            scale_factor=args.scale,
            padding_percent=args.padding,
            target_monitor=args.monitor
        )
        
        if not success:
            logger.warning("Window resize failed. Continuing with capture anyway.")
    
    # Check for crop parameter
    crop_percentage = None
    if args.crop:
        try:
            logger.debug("FLOW: Entering try block in main() - parsing crop parameter")
            crop_values = [float(x) for x in args.crop.split(',')]
            if len(crop_values) == 4:
                crop_percentage = tuple(crop_values)
                logger.info(f"Using crop percentages: {crop_percentage}")
            else:
                logger.error("Crop requires 4 values: left,top,right,bottom")
        except ValueError:
            logger.error("Invalid crop values. Expected numbers separated by commas.")
    
    # Prepare base filename
    if args.title:
        base_filename = args.title.replace(' ', '_')
    else:
        base_filename = "capture"
    
    # Capture images
    logger.info(f"Capturing {args.pages} pages, starting from page {args.start}")
    
    captured_files = consecutive_capture(
        window_title=args.window,
        output_dir=output_dir,
        base_filename=base_filename,
        num_pages=args.pages,
        key=args.key,
        delay=args.delay,
        page_turn_delay=args.page_delay,
        output_format=args.format,
        start_number=args.start,
        monitor_index=args.monitor,
        crop_percentage=crop_percentage,
        quality=args.quality
    )
    
    logger.info(f"Captured {len(captured_files)} images")
    
    # Perform OCR if requested
    if args.ocr and captured_files:
        logger.info("Performing OCR on captured images")
        
        language = args.language or "eng+kor"
        
        # Initialize LLM enhancer if requested
        llm_enhancer = None
        if args.enhance:
            logger.info(f"Initializing LLM enhancer with model size: {args.model_size}")
            llm_enhancer = LLMEnhancer(
                model_size=args.model_size,
                use_mlx=not args.no_mlx
            )
        
        # Process all captured images
        ocr_results = ocr_multiple_files(
            image_files=captured_files,
            language=language,
            psm=args.psm,
            dpi=args.dpi,
            llm_enhancer=llm_enhancer
        )
        
        # Save OCR results to text files
        if ocr_results:
            logger.info("Saving OCR results")
            save_ocr_results(ocr_results, output_dir)
    
    logger.info("Processing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())