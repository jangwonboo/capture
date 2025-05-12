import os
import argparse
import platform
import sys
from pathlib import Path
from typing import List

from capture import (
    capture_and_save_pages,
    image_folder_to_searchable_pdf,
    merge_pdfs
)

# Import the new LLM-based OCR enhancement module
try:
    from llm_ocr import image_folder_to_enhanced_searchable_pdf, check_mlx_status, merge_enhanced_pdfs
    LLM_OCR_AVAILABLE = True
except ImportError as e:
    print(f"LLM OCR enhancement not available: {e}")
    print("Will use standard OCR processing")
    LLM_OCR_AVAILABLE = False

# Check for MLX support on Apple Silicon
MLX_AVAILABLE = False
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine().startswith('arm')
if IS_APPLE_SILICON:
    try:
        # Try to import MLX utils
        from mlx_utils import MLX_AVAILABLE, get_device_info
        if MLX_AVAILABLE:
            print("MLX support detected for Apple Silicon acceleration!")
            device_info = get_device_info()
            print(f"Device: {device_info.get('device', 'Unknown')}")
            print(f"MLX version: {device_info.get('mlx_version', 'Unknown')}")
    except ImportError:
        print("MLX utils not available. Apple Silicon acceleration won't be used.")
        print("Install MLX for optimal performance: pip install mlx")

def get_default_output_dir():
    """Get a suitable default output directory based on the platform."""
    system = platform.system()
    home = Path.home()
    
    if system == 'Windows':
        return str(home / 'Documents' / 'bookscan')
    elif system == 'Darwin':  # macOS
        return str(home / 'Documents' / 'bookscan')
    else:  # Linux or other
        return str(home / 'bookscan')

def get_tesseract_path():
    """Get default tesseract path based on platform."""
    system = platform.system()
    
    if system == 'Windows':
        return 'C:\\Program Files\\Tesseract-OCR\\tessdata'
    elif system == 'Darwin':  # macOS
        return '/usr/local/share/tessdata'
    elif system == 'Linux':
        return '/usr/share/tesseract-ocr/4.00/tessdata'
    else:
        return ''  # Hope it's in PATH

def process_book(
    win_title: str, 
    book: str, 
    pg_start: int, 
    num_pages: int, 
    output_dir: str, 
    delay: int = 5, 
    next_action: str = "right_key", 
    click_coords: tuple = (1100, 1040),
    use_llm: bool = False,
    use_openai: bool = False,
    lang: str = "eng+kor",
    use_mlx: bool = True
) -> None:
    """Process a book by capturing pages, converting to searchable PDF, and merging."""
    folder = Path(output_dir) / book
    
    # Create output directory if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)
    
    # Capture pages from screen
    output_files = capture_and_save_pages(
        win_title=win_title,
        book=book,
        pg_start=pg_start,
        no_pages=num_pages,
        out_folder=str(folder),
        delay=delay,
        next=next_action,
        click_coords=click_coords
    )
    
    # Process images to create searchable PDFs
    if use_llm and LLM_OCR_AVAILABLE:
        print(f"Using LLM-enhanced OCR processing for {book}")
        
        # Set environment variable for MLX
        if IS_APPLE_SILICON and use_mlx:
            os.environ['USE_MLX'] = '1' 
            print("MLX acceleration enabled for OCR processing")
            
            # Setup MLX parameters
            mlx_params = {
                "use_mlx": True,
                "model_size": os.environ.get('LLM_OCR_MODEL_SIZE', 'small'),
                "batch_size": 8  # Default batch size, could be optimized further
            }
            
            # Try to get optimal batch size from mlx_utils if available
            try:
                from mlx_utils import get_optimal_batch_size
                mlx_params["batch_size"] = get_optimal_batch_size()
                print(f"Using optimal MLX batch size: {mlx_params['batch_size']}")
            except ImportError:
                pass
        else:
            mlx_params = None
        
        try:
            # Process with LLM-enhanced OCR
            pdf_files = image_folder_to_enhanced_searchable_pdf(
                str(folder),
                prefix=book,
                lang=lang,
                use_openai=use_openai,
                mlx_params=mlx_params
            )
            
            # Merge the enhanced PDFs
            if pdf_files:
                final_pdf = folder / f"{book}_enhanced_ocr.pdf"
                merge_enhanced_pdfs(str(folder), book, str(final_pdf))
                print(f"Processing complete for {book}. Enhanced PDF: {final_pdf}")
            else:
                print("No enhanced PDFs were created. Falling back to standard OCR.")
                # Fall back to standard OCR if enhanced PDFs failed
                process_with_standard_ocr(folder, book)
                
        except Exception as e:
            print(f"Error in LLM-enhanced OCR processing: {e}")
            print("Falling back to standard OCR...")
            # Fall back to standard OCR if the LLM enhancement fails
            process_with_standard_ocr(folder, book)
    else:
        # Use standard OCR processing
        process_with_standard_ocr(folder, book)

def process_with_standard_ocr(folder: Path, book: str):
    """Process with standard OCR method as fallback."""
    print(f"Using standard OCR processing for {book}")
    image_folder_to_searchable_pdf(str(folder), book)
    
    # Merge individual PDFs into one final PDF
    final_pdf = folder / f"{book}_ocr.pdf"
    merge_pdfs(str(folder), book, str(final_pdf))
    
    print(f"Processing complete for {book}. Final PDF: {final_pdf}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Capture pages from e-book reader and create searchable PDF.')
    
    parser.add_argument('--books', nargs='+', default=["book1"],
                        help='List of book names to process')
    parser.add_argument('--window-title', default="Kindle",
                        help='Window title of the e-book reader application')
    parser.add_argument('--start-page', type=int, default=1,
                        help='Starting page number')
    parser.add_argument('--num-pages', type=int, default=10,
                        help='Number of pages to capture')
    parser.add_argument('--delay', type=int, default=500,
                        help='Delay between page captures in milliseconds')
    parser.add_argument('--output-dir', default=get_default_output_dir(),
                        help='Base output directory for all books')
    parser.add_argument('--next-action', choices=['right_key', 'left_btn'], default='right_key',
                        help='Action to move to the next page')
    parser.add_argument('--click-x', type=int, default=1100,
                        help='X coordinate for mouse click if using left_btn')
    parser.add_argument('--click-y', type=int, default=1040,
                        help='Y coordinate for mouse click if using left_btn')
    parser.add_argument('--tesseract-path', default=get_tesseract_path(),
                        help='Path to Tesseract tessdata directory')
    
    # Add new LLM-related arguments
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM-enhanced OCR processing')
    parser.add_argument('--use-openai', action='store_true',
                        help='Use OpenAI API for LLM enhancement (requires API key)')
    parser.add_argument('--openai-api-key', 
                        help='OpenAI API key (can also be set as OPENAI_API_KEY environment variable)')
    parser.add_argument('--lang', default='eng+kor',
                        help='OCR language(s) to use (e.g., "eng", "kor", "eng+kor")')
    
    # Add Apple Silicon MLX options
    if IS_APPLE_SILICON:
        parser.add_argument('--disable-mlx', action='store_true',
                           help='Disable MLX acceleration on Apple Silicon')
        parser.add_argument('--mlx-model-size', choices=['tiny', 'small', 'medium'], default='small',
                           help='Size of MLX model to use (tiny=fastest, medium=most accurate)')
    
    return parser.parse_args()

def get_mlx_status():
    """Check MLX status and return results as a string."""
    return check_mlx_status()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set Tesseract OCR data path if provided
    if args.tesseract_path:
        os.environ['TESSDATA_PREFIX'] = args.tesseract_path
    
    # Set OpenAI API key if provided
    if args.openai_api_key and args.use_openai:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    # Configure MLX settings if on Apple Silicon
    if IS_APPLE_SILICON:
        use_mlx = not getattr(args, 'disable_mlx', False)
        
        if hasattr(args, 'mlx_model_size'):
            os.environ['LLM_OCR_MODEL_SIZE'] = args.mlx_model_size
            if use_mlx:
                print(f"Using MLX with model size: {args.mlx_model_size}")
    else:
        use_mlx = False
    
    # Warn if LLM features are requested but not available
    if args.use_llm and not LLM_OCR_AVAILABLE:
        print("WARNING: LLM-enhanced OCR was requested but dependencies are not available.")
        print("Install with: pip install -e '.[llm]'")
        if args.use_openai:
            print("To use OpenAI, make sure you provide a valid API key with --openai-api-key")
        
        # Ask user if they want to continue with standard OCR
        if input("Continue with standard OCR? (y/n): ").lower() != 'y':
            sys.exit(1)
    
    # Process each book
    for book in args.books:
        process_book(
            win_title=args.window_title,
            book=book,
            pg_start=args.start_page,
            num_pages=args.num_pages,
            output_dir=args.output_dir,
            delay=args.delay,
            next_action=args.next_action,
            click_coords=(args.click_x, args.click_y),
            use_llm=args.use_llm,
            use_openai=args.use_openai,
            lang=args.lang,
            use_mlx=use_mlx
        )

if __name__ == '__main__':
    main()
