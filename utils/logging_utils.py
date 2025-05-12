"""
Logging utilities for the e-book page capture tool.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

from .config import get_settings

def configure_logging(
    log_level: str = "INFO", 
    log_file: Optional[str] = None, 
    quiet: bool = False, 
    title: Optional[str] = None, 
    output_dir: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path
        quiet: Suppress console output if True
        title: Book title (for default log file name)
        output_dir: Output directory (for default log file location)
    """
    # Get log level
    log_level_value = getattr(logging, log_level.upper())
    
    # Get settings
    settings = get_settings()
    log_format = settings['logging']['format']
    console_format = settings['logging']['console_format']
    
    # Determine log file path
    if log_file:
        log_file_path = log_file
    elif output_dir and title:
        # Create default log file in output directory
        log_file_path = os.path.join(output_dir, f"ebook_capture_{title}.log")
    else:
        # Fallback to a default location
        home = Path.home()
        log_dir = home / "Documents" / "ebook_capture" / "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = log_dir / "ebook_capture.log"
    
    # Ensure the log file directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create handlers
    handlers = []
    
    # File handler
    try:
        logging.debug("FLOW: Entering try block in configure_logging() - creating file handler")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file at {log_file_path}: {e}")
        print("Falling back to console logging only")
    
    # Console handler (unless quiet mode)
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(console_format))
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level_value,
        handlers=handlers,
        force=True  # Override earlier configuration
    )
    
    # Create and configure module loggers
    loggers = {
        'main': logging.getLogger('main'),
        'capture': logging.getLogger('capture'),
        'ocr': logging.getLogger('ocr'),
        'llm': logging.getLogger('llm'),
        'window': logging.getLogger('window'),
        'mlx': logging.getLogger('mlx'),
        'pdf': logging.getLogger('pdf'),
        'config': logging.getLogger('config'),
        'platform': logging.getLogger('platform'),
        'test': logging.getLogger('test')
    }
    
    # Set levels for each logger
    for name, logger in loggers.items():
        logger.setLevel(log_level_value)
    
    # Log configuration
    logging.info(f"Logging configured: level={log_level}, file={log_file_path}")
    
    return loggers