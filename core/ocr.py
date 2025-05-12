"""
OCR functionality for the e-book page capture tool.
"""

import os
import logging
import tempfile
from typing import Optional, Dict, Any, List
from PIL import Image
import subprocess

from utils.config import get_settings, get_tesseract_path

logger = logging.getLogger('ocr')

def perform_ocr(
    image: Image.Image,
    language: Optional[str] = None,
    dpi: int = 300,
    psm: int = 3
) -> str:
    """
    Perform OCR on the given image.
    
    Args:
        image: PIL Image to perform OCR on
        language: Tesseract language code (default: from settings)
        dpi: DPI value for OCR
        psm: Tesseract Page Segmentation Mode (PSM)
        
    Returns:
        Extracted text as string
    """
    settings = get_settings()
    
    # Get tesseract path
    tesseract_path = get_tesseract_path()
    
    # Get default language if not specified
    if language is None:
        language = settings['ocr']['default_language']
    
    logger.info(f"Performing OCR with language: {language}, PSM: {psm}")
    
    # Create a temporary file for the image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
        temp_img_path = temp_img_file.name
    
    # Create a temporary file for the output text
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt_file:
        temp_txt_path = temp_txt_file.name
    
    try:
        logger.debug("FLOW: Entering try block in perform_ocr() - saving image and running tesseract")
        # Save image to temporary file
        image.save(temp_img_path, format='PNG', dpi=(dpi, dpi))
        
        # Build tesseract command
        output_base = os.path.splitext(temp_txt_path)[0]
        cmd = [
            tesseract_path,
            temp_img_path,
            output_base,
            '-l', language,
            '--psm', str(psm),
            '--dpi', str(dpi)
        ]
        
        # Run tesseract
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"OCR Error: {result.stderr}")
            return ""
        
        # Read output text file
        if os.path.exists(temp_txt_path):
            with open(temp_txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"OCR complete: {len(text)} characters extracted")
            return text
        else:
            logger.error(f"OCR output file not found: {temp_txt_path}")
            return ""
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""
    finally:
        # Clean up temporary files
        try:
            logger.debug("FLOW: Entering finally block in perform_ocr() - cleaning up temp files")
            if os.path.exists(temp_img_path):
                os.unlink(temp_img_path)
            if os.path.exists(temp_txt_path):
                os.unlink(temp_txt_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")

def ocr_image_file(
    image_path: str,
    language: Optional[str] = None,
    dpi: int = 300,
    psm: int = 3
) -> str:
    """
    Perform OCR on an image file.
    
    Args:
        image_path: Path to the image file
        language: Tesseract language code (default: from settings)
        dpi: DPI value for OCR
        psm: Tesseract Page Segmentation Mode (PSM)
        
    Returns:
        Extracted text as string
    """
    try:
        logger.debug(f"FLOW: Entering try block in ocr_image_file() - loading image: {image_path}")
        # Load the image
        image = Image.open(image_path)
        
        # Perform OCR
        return perform_ocr(image, language, dpi, psm)
    except Exception as e:
        logger.error(f"Error loading image for OCR: {e}")
        return ""

def ocr_multiple_files(
    image_files: List[str],
    language: Optional[str] = None,
    dpi: int = 300,
    psm: int = 3,
    llm_enhancer = None
) -> Dict[str, str]:
    """
    Perform OCR on multiple image files.
    
    Args:
        image_files: List of paths to image files
        language: Tesseract language code (default: from settings)
        dpi: DPI value for OCR
        psm: Tesseract Page Segmentation Mode (PSM)
        llm_enhancer: Optional LLMEnhancer instance for text enhancement
        
    Returns:
        Dictionary mapping file paths to extracted text
    """
    results = {}
    
    for image_path in image_files:
        logger.info(f"Processing OCR for: {image_path}")
        text = ocr_image_file(image_path, language, dpi, psm)
        
        # Enhance text with LLM if enhancer is provided
        if llm_enhancer is not None and text:
            try:
                logger.debug(f"FLOW: Entering try block in ocr_multiple_files() - enhancing text with LLM")
                logger.info(f"Enhancing OCR text with LLM for: {image_path}")
                enhanced_text = llm_enhancer.enhance_text(text)
                if enhanced_text:
                    text = enhanced_text
                    logger.info("Text enhancement successful")
                else:
                    logger.warning("LLM enhancement returned empty result, using original OCR text")
            except Exception as e:
                logger.error(f"Error enhancing text with LLM: {e}")
                # Fall back to original OCR text
        
        results[image_path] = text
    
    return results

def save_ocr_text(text: str, output_path: str) -> bool:
    """
    Save OCR text to a file.
    
    Args:
        text: Text to save
        output_path: Path to save the text
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        logger.debug(f"FLOW: Entering try block in save_ocr_text() - saving text to: {output_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved OCR text to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving OCR text: {e}")
        return False

def save_ocr_results(results: Dict[str, str], output_dir: str, extension: str = ".txt") -> Dict[str, str]:
    """
    Save multiple OCR results to files.
    
    Args:
        results: Dictionary mapping file paths to extracted text
        output_dir: Directory to save the text files
        extension: File extension for the output files
        
    Returns:
        Dictionary mapping input file paths to output file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    for image_path, text in results.items():
        # Derive output filename from input filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}{extension}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the text
        if save_ocr_text(text, output_path):
            output_paths[image_path] = output_path
    
    return output_paths