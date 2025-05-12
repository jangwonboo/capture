"""
Configuration utilities for loading and managing application settings.
"""

import os
import platform
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Determine base directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / 'config'

# Setup logger
logger = logging.getLogger('config')

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing configuration values
    """
    try:
        logger.debug(f"FLOW: Entering try block in load_yaml_config() - loading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {file_path}: {e}")
        return {}

def get_settings() -> Dict[str, Any]:
    """
    Load and return the application settings.
    
    Returns:
        Dictionary containing application settings
    """
    settings_path = os.path.join(CONFIG_DIR, 'settings.yaml')
    return load_yaml_config(settings_path)

def get_book_formats() -> Dict[str, Any]:
    """
    Load and return the book format definitions.
    
    Returns:
        Dictionary containing book format definitions
    """
    formats_path = os.path.join(CONFIG_DIR, 'book_formats.yaml')
    return load_yaml_config(formats_path)

def get_cli_options() -> Dict[str, Any]:
    """
    Load and return the CLI options configuration.
    
    Returns:
        Dictionary containing CLI options configuration
    """
    cli_options_path = os.path.join(CONFIG_DIR, 'cli_options.yaml')
    return load_yaml_config(cli_options_path)

def get_output_dir(title: str) -> str:
    """
    Get output directory based on application settings and book title.
    Creates a folder structure: output_base/title/
    
    Args:
        title: Title of the book (folder name)
        
    Returns:
        Full path to the output directory
    """
    settings = get_settings()
    system = platform.system()
    
    # Get default output directory based on platform
    if system == 'Windows':
        base_dir = settings['output']['default_output_dir']['windows']
        # Replace environment variables
        base_dir = os.path.expandvars(base_dir)
    elif system == 'Darwin':  # macOS
        base_dir = settings['output']['default_output_dir']['macos']
        # Expand ~ to user's home directory
        base_dir = os.path.expanduser(base_dir)
    else:  # Linux
        base_dir = settings['output']['default_output_dir']['linux']
        base_dir = os.path.expanduser(base_dir)
    
    # Create book-specific subfolder
    output_dir = os.path.join(base_dir, title)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def get_tesseract_path() -> str:
    """
    Get the appropriate Tesseract OCR path based on the platform.
    
    Returns:
        Path to Tesseract executable
    """
    settings = get_settings()
    system = platform.system()
    
    if system == 'Windows':
        return settings['ocr']['tesseract_paths']['windows']
    elif system == 'Darwin':  # macOS
        # Try multiple possible paths for macOS
        possible_paths = settings['ocr']['tesseract_paths']['macos']
        
        # First check predefined paths
        for path in possible_paths:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                return path
        
        # If not found, try using 'which' command
        try:
            logger.debug("FLOW: Entering try block in get_tesseract_path() - using 'which' command")
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Error using 'which' command: {e}")
        
        # Default to 'tesseract' and hope it's in PATH
        return 'tesseract'
    elif system == 'Linux':
        return settings['ocr']['tesseract_paths']['linux']
    else:
        # Default command, hoping it's in PATH
        return 'tesseract'