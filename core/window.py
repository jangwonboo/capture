"""
Window management functionality for the e-book page capture tool.
"""

import logging
import platform
from typing import Optional, Dict, Any, Tuple

from utils.config import get_book_formats
from platforms import get_platform_handler

logger = logging.getLogger('window')

def get_window_by_title(window_title: str) -> Optional[Any]:
    """
    Find a window by its title and bring it to the foreground.
    
    Args:
        window_title: Title of the window to find
        
    Returns:
        Window handle/ID or boolean depending on platform
    """
    if not window_title:
        logger.debug("No window title provided, skipping window focus")
        return None
    
    logger.info(f"Attempting to focus window: '{window_title}'")
    
    # Get platform-specific handler
    platform_handler = get_platform_handler()
    return platform_handler.get_window_by_title(window_title)

def get_window_rect(window_title: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the current position and size of a window.
    
    Args:
        window_title: Title of the window
    
    Returns:
        Tuple of (x, y, width, height) or None if window not found
    """
    platform_handler = get_platform_handler()
    return platform_handler.get_window_rect(window_title)

def resize_window_to_book_format(
    window_title: str, 
    book_format: str, 
    scale_factor: float = 0.8, 
    padding_percent: int = 5, 
    target_monitor: Optional[int] = None
) -> bool:
    """
    Resize a window to match a specific book format with proper aspect ratio.
    
    Args:
        window_title: Title of the window to resize
        book_format: Format name from book_formats.yaml or dict with width and height in mm
        scale_factor: How much of the screen height to use (0.0-1.0)
        padding_percent: Extra padding around the format (percentage)
        target_monitor: Index of monitor to use (None = auto-detect monitor with window)
        
    Returns:
        Boolean indicating success or failure
    """
    if not window_title:
        logger.warning("No window title provided, skipping window resize")
        return False
    
    # Get book format definitions
    book_formats = get_book_formats()
    
    # Get book format dimensions
    if isinstance(book_format, str):
        if book_format not in book_formats:
            logger.error(f"Unknown book format: {book_format}")
            logger.info(f"Available formats: {', '.join(book_formats.keys())}")
            return False
        format_spec = book_formats[book_format]
    elif isinstance(book_format, dict) and 'width' in book_format and 'height' in book_format:
        format_spec = book_format
    else:
        logger.error("Invalid book format specification")
        return False
    
    # Get platform-specific handler
    platform_handler = get_platform_handler()
    
    # Get monitor information
    monitor_info = platform_handler.get_monitor_info()
    logger.info(f"Detected {len(monitor_info['monitors'])} monitors")
    
    # Get target monitor dimensions based on where the window is currently located
    # or use the specified target monitor
    target_monitor_index = 0  # Default to primary monitor
    
    if target_monitor is not None and 0 <= target_monitor < len(monitor_info['monitors']):
        # Use specified monitor
        target_monitor_index = target_monitor
        logger.info(f"Using specified monitor {target_monitor}")
    else:
        # Try to detect which monitor contains the window
        window_rect = platform_handler.get_window_rect(window_title)
        if window_rect:
            window_x, window_y = window_rect[0], window_rect[1]
            
            # Check which monitor contains the window center point
            window_center_x = window_x + window_rect[2] // 2
            window_center_y = window_y + window_rect[3] // 2
            
            # Convert from absolute to normalized coordinates
            if 'virtual_screen' in monitor_info:
                virt_x, virt_y = monitor_info['virtual_screen'][0], monitor_info['virtual_screen'][1]
                window_center_x -= virt_x
                window_center_y -= virt_y
            
            # Find the monitor that contains this point
            for i, (mon_x, mon_y, mon_width, mon_height) in enumerate(monitor_info['monitors']):
                if (mon_x <= window_center_x < mon_x + mon_width and 
                    mon_y <= window_center_y < mon_y + mon_height):
                    target_monitor_index = i
                    logger.info(f"Window detected on monitor {i}")
                    break
            else:
                logger.info(f"Window not found on any monitor, using primary monitor")
                target_monitor_index = monitor_info.get('primary_index', 0)
        else:
            logger.info(f"Could not detect window position, using primary monitor")
            target_monitor_index = monitor_info.get('primary_index', 0)
    
    # Get target monitor dimensions
    if target_monitor_index < len(monitor_info['monitors']):
        mon_x, mon_y, screen_width, screen_height = monitor_info['monitors'][target_monitor_index]
        logger.info(f"Target monitor dimensions: {screen_width}x{screen_height} at ({mon_x},{mon_y})")
    else:
        # Fallback to primary monitor
        screen_width, screen_height = monitor_info['primary']
        mon_x, mon_y = 0, 0
        logger.info(f"Using primary monitor: {screen_width}x{screen_height}")
    
    # Check if it's a device format (with pixel dimensions) or a book format (with mm dimensions)
    is_device_format = format_spec.get('is_device', False)
    
    if is_device_format:
        # For devices, we already have the dimensions in pixels
        device_width_px = format_spec['width']
        device_height_px = format_spec['height']
        
        # Apply scale factor directly to the device dimensions
        target_width_px = int(device_width_px * scale_factor)
        target_height_px = int(device_height_px * scale_factor)
        
        # Make sure the dimensions fit on target monitor
        if target_width_px > screen_width:
            reduction_factor = screen_width / target_width_px
            target_width_px = screen_width
            target_height_px = int(target_height_px * reduction_factor)
            
        if target_height_px > screen_height:
            reduction_factor = screen_height / target_height_px
            target_height_px = screen_height
            target_width_px = int(target_width_px * reduction_factor)
        
        logger.info(f"Resizing window to match {format_spec.get('description', book_format)} format")
        logger.info(f"Original device resolution: {device_width_px}x{device_height_px}")
        logger.info(f"Scaled window size: {target_width_px}x{target_height_px} pixels")
    else:
        # Extract book width and height in mm
        book_width_mm = format_spec['width']
        book_height_mm = format_spec['height']
        
        # Calculate target window dimensions in pixels
        # Calculate the maximum height based on scale factor
        max_height_px = int(screen_height * scale_factor)
        
        # Calculate the ratio of pixels to mm
        px_per_mm = max_height_px / book_height_mm
        
        # Calculate target width and height
        target_width_px = int(book_width_mm * px_per_mm)
        target_height_px = max_height_px
        
        # Apply padding
        padding_width = int(target_width_px * (padding_percent / 100))
        padding_height = int(target_height_px * (padding_percent / 100))
        target_width_px += padding_width
        target_height_px += padding_height
        
        # Ensure dimensions don't exceed screen
        if target_width_px > screen_width:
            reduction_factor = screen_width / target_width_px
            target_width_px = screen_width
            target_height_px = int(target_height_px * reduction_factor)
        
        logger.info(f"Resizing window to match {format_spec.get('description', book_format)} format")
        logger.info(f"Book dimensions: {book_width_mm}mm x {book_height_mm}mm")
        logger.info(f"Target window size: {target_width_px}x{target_height_px} pixels")
    
    # Calculate window position (center on target monitor)
    x = mon_x + (screen_width - target_width_px) // 2
    y = mon_y + (screen_height - target_height_px) // 2
    
    # Now convert back to absolute coordinates if needed
    if 'virtual_screen' in monitor_info:
        virt_x, virt_y = monitor_info['virtual_screen'][0], monitor_info['virtual_screen'][1]
        abs_x = virt_x + x
        abs_y = virt_y + y
    else:
        abs_x, abs_y = x, y
    
    logger.info(f"Setting window position to ({abs_x}, {abs_y})")
    
    # Use platform handler to resize the window
    return platform_handler.resize_window(window_title, target_width_px, target_height_px, abs_x, abs_y)

def send_keystroke(key_name: str) -> bool:
    """
    Send a keystroke to the active application.
    
    Args:
        key_name: Name of the key to press ('right', 'left', 'space', 'enter', etc.)
        
    Returns:
        Boolean indicating success or failure
    """
    logger.debug(f"Sending keystroke: {key_name}")
    
    # Get platform-specific handler
    platform_handler = get_platform_handler()
    return platform_handler.send_keystroke(key_name)

def get_monitor_info() -> Dict[str, Any]:
    """
    Get information about all monitors in a multi-monitor setup.
    
    Returns:
        Dictionary containing monitor information
    """
    platform_handler = get_platform_handler()
    return platform_handler.get_monitor_info()