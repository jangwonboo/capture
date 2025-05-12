from setuptools import setup, find_packages

setup(
    name="capture",
    version="0.2.0",
    author="John",
    description="A cross-platform tool for capturing pages from e-book readers with LLM-enhanced OCR",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pytesseract",
        "pyautogui",
        "Pillow",
        "pdfrw",
        "pathlib",
        "opencv-python",
        "pyscreeze",
        "pdf2image",
        "numpy",
    ],
    extras_require={
        "llm": [
            "tqdm",
            "transformers",
            "sentence-transformers",
            "openai",
            "langchain",
        ],
        "gpu": [
            "torch",
        ],
    },
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "capture=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
        "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture",
    ],
) 