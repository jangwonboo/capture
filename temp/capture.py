import glob
import os
import sys
import time
import platform
from pathlib import Path
from typing import List, Tuple, Optional

import pytesseract
import pyautogui
from PIL import Image, ImageGrab
from pdfrw import PdfReader, PdfWriter

# Configure Tesseract path based on platform
system = platform.system()
if system == 'Windows':
    TESSERACT_CMD = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
elif system == 'Darwin':  # macOS
    # Check multiple possible locations for Tesseract on macOS
    possible_paths = ['/usr/local/bin/tesseract', '/opt/homebrew/bin/tesseract']
    TESSERACT_CMD = None
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            TESSERACT_CMD = path
            break
    
    # If not found, try using 'which' command
    if TESSERACT_CMD is None:
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                TESSERACT_CMD = result.stdout.strip()
            else:
                TESSERACT_CMD = 'tesseract'  # Default fallback
        except Exception:
            TESSERACT_CMD = 'tesseract'  # Default fallback
elif system == 'Linux':
    TESSERACT_CMD = '/usr/bin/tesseract'
else:
    TESSERACT_CMD = 'tesseract'  # Default command, hoping it's in PATH

# Platform-specific configurations
if system == 'Windows':
    try:
        import pywinauto as pywin
        PYWINAUTO_AVAILABLE = True
    except ImportError:
        PYWINAUTO_AVAILABLE = False
else:
    PYWINAUTO_AVAILABLE = False

def get_window_by_title(win_title: str):
    """Get a window by title, using platform-specific methods."""
    if PYWINAUTO_AVAILABLE and system == 'Windows':
        try:
            app = pywin.Application().connect(found_index=0, title_re=win_title, backend="win32")
            return app.top_window()
        except Exception as e:
            print(f"Windows: Error connecting to window: {e}")
            return None
    elif system == 'Darwin':  # macOS
        try:
            # For macOS, we use AppleScript through subprocess
            import subprocess
            script = f'''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                if name of frontApp contains "{win_title}" then
                    return true
                else
                    return false
                end if
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            return result.stdout.strip() == 'true'
        except Exception as e:
            print(f"macOS: Error checking window: {e}")
            return None
    else:  # Linux or other
        # For Linux, we'll just assume the window is active
        print("Platform-specific window finding not implemented for this OS.")
        print("Please ensure the target window is active and in focus.")
        return True

def capture_screen():
    """Capture the screen using platform-independent method."""
    return ImageGrab.grab()

def send_keystroke(key: str):
    """Send keystrokes in a platform-independent way."""
    if key == '{RIGHT}':
        pyautogui.press('right')
    elif key == '{LEFT}':
        pyautogui.press('left')
    elif key == '{SPACE}':
        pyautogui.press('space')
    else:
        pyautogui.press(key)

def capture_and_save_pages(
    win_title: str, 
    book: str, 
    pg_start: int, 
    no_pages: int, 
    out_folder: str,
    delay: int = 300, 
    next: str = "right_key", 
    click_coords: Tuple[int, int] = (1000, 1000)
) -> List[str]:
    """
    Capture screenshots of pages from a window and save them as images.
    
    Args:
        win_title: Window title to capture from
        book: Book name for file naming
        pg_start: Starting page number
        no_pages: Number of pages to capture
        out_folder: Output folder path
        delay: Delay between captures in milliseconds
        next: Action to move to the next page ("right_key" or "left_btn")
        click_coords: Coordinates for mouse click if using "left_btn"
        
    Returns:
        List of saved image file paths
    """
    output_list: List[str] = []
    out_path = Path(out_folder)
    
    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting capture of {no_pages} pages from {win_title}")
    print(f"Please make sure the window is in focus and visible")
    
    # Give the user a moment to focus the window
    print("Starting in 3 seconds...")
    time.sleep(3)

    for i in range(pg_start, pg_start + no_pages):
        try:
            # Check if window is available
            window = get_window_by_title(win_title)
            if not window and system != 'Linux':
                print(f"Window '{win_title}' not found. Please ensure it's open and try again.")
                continue
            
            # Create the image file name and path
            filename = f"{book}_{str(i).zfill(3)}.png"
            filepath = out_path / filename
            
            # For Windows, try to use pywinauto if available for more precise capture
            if PYWINAUTO_AVAILABLE and system == 'Windows' and isinstance(window, object):
                img = window.capture_as_image()
            else:
                # Otherwise use PyAutoGUI for cross-platform support
                img = capture_screen()
                
            print(f"Saving to {filepath}")
            img.save(filepath)
            output_list.append(str(filepath))

            # Calculate and print completion percentage
            pct = (i - pg_start + 1) / no_pages * 100
            print(f"{pct:.1f}% complete")

            # Move to the next page based on specified action
            if next == "right_key":
                send_keystroke('{RIGHT}')
            elif next == "left_btn":
                pyautogui.click(click_coords[0], click_coords[1])

            # Wait the specified interval before capturing the next image
            time.sleep(delay / 1000.0)

        except Exception as e:
            print(f"Error during capture: {e}")
            time.sleep(delay / 1000.0)
            continue

    return output_list


def image_folder_to_searchable_pdf(folder: str, prefix: str = None) -> None:
    """
    Convert images in a folder to searchable PDFs with OCR.
    
    Args:
        folder: Folder containing images
        prefix: Prefix for image filenames to filter by
    """
    # Configure Tesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    # Get all image files matching the pattern
    pattern = f"{prefix}*.png" if prefix else "*.png"
    image_files_pattern = os.path.join(folder, pattern)
    image_files: List[str] = glob.glob(image_files_pattern)
    
    for file in image_files:
        try:
            print(f"Converting to a searchable PDF: {file}")
            image = Image.open(file)
            pdf_data = pytesseract.image_to_pdf_or_hocr(image, extension='pdf', lang='eng+kor')
            output_filename = f"{os.path.splitext(file)[0]}_ocr.pdf"
            
            with open(output_filename, 'wb') as f:
                f.write(pdf_data)

        except Exception as e:
            print(f"Error processing file {file}: {e}")


def image_folder_to_text(folder: str, prefix: str = None) -> None:
    """
    Extract text from images in a folder and save to a text file.
    
    Args:
        folder: Folder containing images
        prefix: Prefix for image filenames to filter by
    """
    # Configure Tesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    # Get all image files matching the pattern
    pattern = f"{prefix}*.png" if prefix else "*.png"
    image_files_pattern = os.path.join(folder, pattern)
    image_files: List[str] = glob.glob(image_files_pattern)
    text_file = os.path.join(folder, f"{prefix or 'output'}.txt")

    with open(text_file, 'wb') as f:
        for file in image_files:
            try:
                print(f"Running OCR on {file}")
                image = Image.open(file)
                text_data = pytesseract.image_to_string(image, lang='eng+kor')
                # Convert text to bytes and write to file
                f.write(text_data.encode('utf-8'))
                f.write(b"\n\n--- Page Break ---\n\n")
            except Exception as e:
                print(f"Error processing file {file}: {e}")


def image_files_to_searchable_pdf(image_files: List[str]) -> None:
    """
    Convert a list of image files to individual searchable PDFs.
    
    Args:
        image_files: List of image file paths
    """
    # Configure Tesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    for file in image_files:
        try:
            print(f"Converting to a searchable PDF: {file}")
            image = Image.open(file)
            pdf_data = pytesseract.image_to_pdf_or_hocr(image, extension='pdf', lang='eng+kor')
            output_filename = f"{os.path.splitext(file)[0]}.pdf"
            
            with open(output_filename, 'wb') as f:
                f.write(pdf_data)
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")


def merge_pdfs(folder: str, prefix: str, output_filename: str) -> None:
    """
    Merge multiple PDF files in a folder into a single PDF.
    
    Args:
        folder: Folder containing PDF files
        prefix: Prefix for PDF filenames to filter by
        output_filename: Path for the merged output PDF
    """
    pdf_files = glob.glob(os.path.join(folder, f'{prefix}*_ocr.pdf'))
    pdf_files.sort()  # Sort to maintain page order
    pdf_writer = PdfWriter()

    for file in pdf_files:
        try:
            print(f"Merging PDF: {file}")
            pdf_reader = PdfReader(file)
            pdf_writer.addpages(pdf_reader.pages)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    print(f"Writing merged PDF to {output_filename}")
    with open(output_filename, 'wb') as f:
        pdf_writer.write(f)
