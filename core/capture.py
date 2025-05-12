"""
Screen capture functionality for the e-book page capture tool.
"""

import os
import time
import logging
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image

from platforms import get_platform_handler
from core.window import get_window_rect

logger = logging.getLogger('capture')

def capture_screen(window_title: Optional[str] = None, monitor_index: Optional[int] = None) -> Optional[Image.Image]:
    """
    Capture the screen or a specific window.
    
    Args:
        window_title: Title of the window to capture (if None, captures entire screen or monitor)
        monitor_index: Index of monitor to capture (if None and window_title is None, captures primary monitor)
        
    Returns:
        PIL Image of the captured screen or window
    """
    platform_handler = get_platform_handler()
    return platform_handler.capture_screen(window_title, monitor_index)

def capture_with_delay(
    window_title: Optional[str], 
    delay: float = 0.5, 
    monitor_index: Optional[int] = None
) -> Optional[Image.Image]:
    """
    Capture the screen or window with a delay.
    
    Args:
        window_title: Title of the window to capture
        delay: Delay in seconds before capture
        monitor_index: Index of monitor to capture
        
    Returns:
        PIL Image of the captured screen or window
    """
    logger.info(f"Waiting {delay}s before capture...")
    time.sleep(delay)
    
    return capture_screen(window_title, monitor_index)

def crop_image(
    image: Image.Image, 
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    crop_percentage: Optional[Tuple[float, float, float, float]] = None
) -> Image.Image:
    """
    Crop an image based on absolute coordinates or percentage.
    
    Args:
        image: PIL Image to crop
        crop_rect: Tuple of (left, top, right, bottom) in absolute pixels
        crop_percentage: Tuple of (left, top, right, bottom) in percentages (0-100)
        
    Returns:
        Cropped PIL Image
    
    Note:
        Either crop_rect or crop_percentage must be provided, not both.
    """
    if crop_rect is None and crop_percentage is None:
        logger.warning("No crop parameters provided, returning original image")
        return image
    
    if crop_rect is not None and crop_percentage is not None:
        logger.warning("Both crop_rect and crop_percentage provided, using crop_rect")
    
    # Get image dimensions
    width, height = image.size
    
    # Calculate crop coordinates
    if crop_rect is not None:
        # Use absolute pixel values
        left, top, right, bottom = crop_rect
    else:
        # Use percentage values
        left_pct, top_pct, right_pct, bottom_pct = crop_percentage
        left = int(width * left_pct / 100)
        top = int(height * top_pct / 100)
        right = int(width * (1 - right_pct / 100))
        bottom = int(height * (1 - bottom_pct / 100))
    
    # Ensure coordinates are within image bounds
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    
    logger.info(f"Cropping image: ({left}, {top}, {right}, {bottom})")
    return image.crop((left, top, right, bottom))

def capture_and_save(
    window_title: Optional[str], 
    output_path: str,
    output_format: str = "png", 
    delay: float = 0.5,
    monitor_index: Optional[int] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    crop_percentage: Optional[Tuple[float, float, float, float]] = None,
    quality: Optional[int] = None
) -> Optional[str]:
    """
    Capture, optionally crop, and save an image.
    
    Args:
        window_title: Title of the window to capture
        output_path: Path to save the captured image
        output_format: Image format to save as ('png', 'jpg', 'jpeg', 'tiff', etc.)
        delay: Delay in seconds before capture
        monitor_index: Index of monitor to capture
        crop_rect: Tuple of (left, top, right, bottom) in absolute pixels
        crop_percentage: Tuple of (left, top, right, bottom) in percentages (0-100)
        quality: Quality for JPEG compression (0-100)
        
    Returns:
        Path to the saved image or None if capture failed
    """
    # Capture the screen or window
    img = capture_with_delay(window_title, delay, monitor_index)
    
    if img is None:
        logger.error("Capture failed")
        return None
    
    # Crop if needed
    if crop_rect is not None or crop_percentage is not None:
        img = crop_image(img, crop_rect, crop_percentage)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the image
    try:
        logger.debug(f"FLOW: Entering try block in capture_and_save() - saving image to: {output_path}")
        # Determine quality parameters for different formats
        save_options = {}
        if output_format.lower() in ['jpg', 'jpeg'] and quality is not None:
            save_options['quality'] = quality
        elif output_format.lower() == 'png':
            # PNG compression is lossless, but setting optimize saves some space
            save_options['optimize'] = True
        elif output_format.lower() == 'webp' and quality is not None:
            save_options['quality'] = quality
        
        # Save image
        img.save(output_path, format=output_format.upper(), **save_options)
        logger.info(f"Saved image to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def consecutive_capture(
    window_title: Optional[str], 
    output_dir: str,
    base_filename: str,
    num_pages: int,
    key: str = "right",
    delay: float = 0.5,
    page_turn_delay: float = 0.3,
    output_format: str = "png",
    start_number: int = 1,
    monitor_index: Optional[int] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    crop_percentage: Optional[Tuple[float, float, float, float]] = None,
    quality: Optional[int] = None
) -> List[str]:
    """
    Capture multiple pages by automatically pressing navigation keys between captures.
    
    Args:
        window_title: Title of the window to capture
        output_dir: Directory to save the captured images
        base_filename: Base name for the saved images
        num_pages: Number of pages to capture
        key: Key to press to navigate to next page
        delay: Delay in seconds before each capture
        page_turn_delay: Delay after pressing key before next capture
        output_format: Image format to save as
        start_number: Starting page number
        monitor_index: Index of monitor to capture
        crop_rect: Tuple of (left, top, right, bottom) in absolute pixels
        crop_percentage: Tuple of (left, top, right, bottom) in percentages (0-100)
        quality: Quality for JPEG compression (0-100)
        
    Returns:
        List of paths to the saved images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get platform-specific handler
    platform_handler = get_platform_handler()
    
    # List to store output paths
    output_paths = []
    
    # Loop for each page
    for i in range(num_pages):
        page_num = start_number + i
        output_filename = f"{base_filename}_{page_num:04d}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(f"Capturing page {page_num}/{start_number + num_pages - 1}")
        
        # Capture and save the current page
        saved_path = capture_and_save(
            window_title=window_title,
            output_path=output_path,
            output_format=output_format,
            delay=delay,
            monitor_index=monitor_index,
            crop_rect=crop_rect,
            crop_percentage=crop_percentage,
            quality=quality
        )
        
        if saved_path:
            output_paths.append(saved_path)
        else:
            logger.error(f"Failed to capture page {page_num}")
        
        # If not the last page, press the key to go to the next page
        if i < num_pages - 1:
            logger.debug(f"Pressing {key} key to navigate to next page")
            if not platform_handler.send_keystroke(key):
                logger.error(f"Failed to send keystroke '{key}'")
            
            # Wait after page turn
            if page_turn_delay > 0:
                logger.debug(f"Waiting {page_turn_delay}s after page turn")
                time.sleep(page_turn_delay)
    
    logger.info(f"Captured {len(output_paths)} pages")
    return output_paths