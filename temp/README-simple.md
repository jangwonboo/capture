# Simple E-book Capture Tool

A streamlined tool for capturing pages from e-book readers (Ridibooks or Kindle), performing LLM-aided OCR, and creating searchable PDF files.

## Features

- Captures screenshots from e-book readers
- Automatically focuses on e-book reader windows
- Applies image preprocessing for better OCR results
- Performs OCR on captured images with optional LLM enhancement
- Creates searchable PDF files
- Merges individual PDFs into a single document
- Support for multiple languages (English, Korean, etc.)
- Cross-platform support for Windows, macOS, and Linux
- Comprehensive logging system

## Requirements

- Python 3.6+
- Tesseract OCR installed on your system
- Dependencies listed in `requirements-simple.txt`

## Installation

1. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`

2. Install Python dependencies:
   ```
   pip install -r requirements-simple.txt
   ```

3. Additional platform-specific requirements:
   - Windows: No additional requirements
   - macOS: No additional requirements
   - Linux: `sudo apt-get install xdotool` (for window focusing)

## Usage

Basic usage with default settings:

```
python simple_capture.py -t "my_book" -r "Kindle"
```

This will capture 10 pages from your Kindle reader, starting from page 1, using the right arrow key to navigate between pages.

### Command-line Arguments

#### Reader Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-r` | `--reader` | E-book reader application (e.g., "Kindle", "Ridibooks") | None |

#### Book Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-t` | `--title` | Title of the book (used for file naming and output folder) | `ebook` |
| `-s` | `--start` | Starting page number | `1` |
| `-e` | `--end` | Ending page number | `start + 9` |

#### Capture Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-d` | `--delay` | Delay between page captures in seconds | `0.5` |
| `-o` | `--output` | Base output path (files stored in output/title folder) | Platform-specific user documents folder |

#### Page Turning Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-p` | `--page-turn` | Method to turn pages (`key` or `click`) | `key` |
| `-k` | `--key` | Key to press for next page - Supported keys: right, left, space, enter, pagedown, pageup, right_arrow, left_arrow, down_arrow, up_arrow | `right` |
| `-c` | `--click-coord` | Coordinates as a tuple for mouse click (e.g., "(800,500)") | "(1100,1040)" |

#### OCR Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-l` | `--lang` | OCR language(s) (e.g., eng, kor, kor+eng) | `kor+eng` |
|  | `--llm` | Use LLM to enhance OCR results | `False` |

#### Logging Settings
| Short | Long | Description | Default |
|-------|------|-------------|---------|
|  | `--log-level` | Set logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
|  | `--log-file` | Custom log file path | Output directory/ebook_capture_TITLE.log |
| `-q` | `--quiet` | Suppress console output (log to file only) | `False` |

### Examples

Capture Kindle pages with keyboard navigation:

```
python simple_capture.py -r "Kindle" -t "my_novel" -s 15 -e 35 --llm
```

Use mouse click to navigate in Ridibooks:

```
python simple_capture.py -r "Ridibooks" -t "manga" -s 1 -e 50 -p click -c "(800,500)"
```

Use page down key for navigation:

```
python simple_capture.py -r "PDF Reader" -t "textbook" -s 1 -e 20 -k pagedown
```

Specify a custom output directory and language:

```
python simple_capture.py -t "korean_book" -s 30 -e 60 -o "./my_books" -l kor --llm
```

Debug mode with detailed logging:

```
python simple_capture.py -t "debugging_book" --log-level DEBUG
```

Run in quiet mode (useful for automated scripts):

```
python simple_capture.py -t "background_job" -q --log-file "/tmp/capture.log"
```

## How It Works

1. The script locates and focuses on your e-book reader window
2. It captures a screenshot of the e-book reader
3. The image is saved and processed for better OCR results
4. Tesseract OCR extracts the text from the image
5. If enabled, the LLM enhancement improves OCR accuracy
6. A searchable PDF is created for each page
7. All PDFs are merged into a single document

## Troubleshooting

- **Poor OCR quality**: Try adjusting the language parameter to match your book's language
- **Capture issues**: Make sure the e-book reader window is fully visible and in focus
- **Navigation problems**: Adjust the delay between captures or try different page turning methods
- **Memory errors with LLM**: The transformer models can require significant memory. If you encounter memory issues, try disabling the LLM enhancement
- **Window focusing issues**: If the script can't find your e-book reader window, use `--reader` with the exact window title or a unique part of it
- **Linux window focusing**: Install xdotool with `sudo apt-get install xdotool` to enable window focusing on Linux

## Log Files

The script creates detailed log files which can be helpful for troubleshooting. By default, logs are stored in the same directory as the output files with the name `ebook_capture_TITLE.log`. You can specify a different log location with the `--log-file` parameter.

Use `--log-level DEBUG` for more detailed logging when troubleshooting issues.