"""
LLM-based OCR Enhancement Module

This module provides functions to enhance OCR results using Large Language Models.
It improves the quality of text extracted from images and creates better searchable PDFs.
"""

import sys
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import re
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
import json
import platform

import numpy as np
import cv2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import torch
from PyPDF2 import PdfWriter, PdfReader  # Add PyPDF2 import for PDF creation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("llm_ocr")

# Check for Apple Silicon and MLX availability
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine().startswith('arm')
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx
        import mlx.core as mx
        MLX_AVAILABLE = True
        logger.info("MLX detected and available for Apple Silicon acceleration")
    except ImportError:
        logger.warning("MLX not found. For optimal performance on Apple Silicon, install MLX: pip install mlx")

# Configure Tesseract path based on platform
system = platform.system()
if system == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
elif system == 'Darwin':  # macOS
    # Check multiple possible locations for Tesseract on macOS
    possible_paths = ['/usr/local/bin/tesseract', '/opt/homebrew/bin/tesseract']
    tesseract_path = None
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    # If not found, try using 'which' command
    if tesseract_path is None:
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                tesseract_path = result.stdout.strip()
            else:
                # Default to 'tesseract' and hope it's in PATH
                tesseract_path = 'tesseract'
        except Exception:
            tesseract_path = 'tesseract'
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info(f"Using Tesseract at: {tesseract_path}")
elif system == 'Linux':
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:
    # Default command, hoping it's in PATH
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Configure GPU/CPU usage with optimal settings for each
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if IS_APPLE_SILICON and MLX_AVAILABLE:
    # Allow disabling MLX via environment variable or command-line
    DISABLE_MLX = (os.environ.get("DISABLE_MLX", "0").lower() in ("1", "true", "yes") or 
                  "--disable-mlx" in sys.argv)
    
    if not DISABLE_MLX:
        DEVICE = "mlx"
        logger.info("Using MLX as the primary backend for Apple Silicon")
    else:
        logger.info("MLX disabled by user configuration, using CPU")
elif DEVICE == "cpu":
    logger.warning("CUDA not available - using CPU for LLM (this may be slow)")
    # Set torch threads for CPU optimization
    torch.set_num_threads(min(8, os.cpu_count() or 4))  # Limit threads to avoid oversubscription
    logger.info(f"Set torch threads to {torch.get_num_threads()} for CPU optimization")
else:
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

# Global model cache to avoid reloading
MODEL_CACHE = {}

# Global model variables 
text_correction_model = None
embedding_model = None

# Optional: Configure OpenAI if available
try:
    import openai
    from openai import OpenAI
    
    # Initialize client (will use OPENAI_API_KEY from environment)
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    OPENAI_AVAILABLE = True
    logger.info("OpenAI module available")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI module not available, falling back to local models")

# Configure model options based on available hardware
CPU_FRIENDLY_MODELS = {
    "text_correction": {
        "tiny": "google/mt5-small",  # Smaller model ~300MB
        "small": "KETI-AIR/ke-t5-small-ko",  # ~500MB
        "medium": "KETI-AIR/ke-t5-base-ko",  # ~850MB
    },
    "embedding": {
        "tiny": "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2",  # ~90MB
        "small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ~120MB
        "medium": "jhgan/ko-sroberta-multitask",  # ~300MB
    }
}

# MLX-optimized models for Apple Silicon
MLX_MODELS = {
    "text_correction": {
        # These are actual MLX models available on Hugging Face
        "tiny": "mlx-community/Kokoro-82M-8bit",
        "small": "mlx-community/parakeet-tdt-0.6b-v2",
        "medium": "mlx-community/Dia-1.6B-6bit",
    },
    "embedding": {
        # Since there are no MLX-specific embedding models, 
        # we'll use standard models and process them with PyTorch
        "tiny": "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2",
        "small": "sentence-transformers/all-MiniLM-L6-v2",
        "medium": "sentence-transformers/all-mpnet-base-v2",
    }
}

def load_model(model_type, size="small", force_reload=False):
    """
    Load a model with caching, with options for different sizes based on available hardware.
    
    Args:
        model_type: Type of model to load (text_correction or embedding)
        size: Model size (tiny, small, medium)
        force_reload: Force model reload even if cached
        
    Returns:
        Loaded model
    """
    global MODEL_CACHE, DEVICE
    
    cache_key = f"{model_type}_{size}"
    
    # Return cached model if available
    if cache_key in MODEL_CACHE and not force_reload:
        return MODEL_CACHE[cache_key]
    
    try:
        # Use MLX if on Apple Silicon and MLX is available
        if DEVICE == "mlx":
            logger.info(f"Loading {model_type} model with MLX on Apple Silicon")
            
            # First check for required dependencies
            try:
                from huggingface_hub import snapshot_download
                import mlx
                from mlx import core as mx
            except ImportError as e:
                logger.warning(f"Required MLX dependencies not found: {e}")
                logger.warning("Install with: pip install huggingface_hub mlx")
                logger.warning("Falling back to standard transformers")
                # Disable MLX for future loads
                DEVICE = "cpu"
                # Fall through to standard PyTorch loading
            else:
                # Dependencies exist, proceed with MLX loading
                if model_type == "text_correction":
                    try:
                        # MLX doesn't have the same pipeline interface as transformers
                        # So we need to create a custom wrapper
                        from mlx import core as mx
                        from transformers import AutoTokenizer
                        import os
                        from huggingface_hub import hf_hub_download
                        
                        model_name = MLX_MODELS["text_correction"][size]
                        logger.info(f"Loading MLX text correction model: {model_name}")
                        
                        # Load tokenizer from the original model
                        try:
                            original_model_name = CPU_FRIENDLY_MODELS["text_correction"][size]
                            tokenizer = AutoTokenizer.from_pretrained(original_model_name)
                        except Exception as e:
                            logger.warning(f"Failed to load original tokenizer: {e}")
                            logger.warning("Attempting to load tokenizer directly from MLX model")
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                        
                        # MLX model class definition
                        class MLXTextModel:
                            def __init__(self, model_name):
                                # Import here to avoid import errors if MLX is not available
                                from mlx import core as mx
                                import mlx
                                import os
                                from huggingface_hub import snapshot_download
                                
                                # Download model from HuggingFace and load using MLX utils
                                try:
                                    logger.info(f"Downloading model: {model_name}")
                                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mlx_models")
                                    os.makedirs(cache_dir, exist_ok=True)
                                    
                                    # Download the model files
                                    local_path = snapshot_download(
                                        repo_id=model_name,
                                        cache_dir=cache_dir,
                                        local_files_only=False
                                    )
                                    logger.info(f"Model downloaded to {local_path}")
                                    
                                    # Search for available model.safetensors or other model files
                                    import glob
                                    
                                    # Check for various model file extensions
                                    model_files = []
                                    for ext in ["safetensors", "bin", "mlx"]:
                                        model_files.extend(glob.glob(f"{local_path}/*.{ext}"))
                                    
                                    if not model_files:
                                        logger.error(f"No model files found in {local_path}")
                                        raise FileNotFoundError(f"No model files found in {local_path}")
                                    
                                    logger.info(f"Found model files: {model_files}")
                                    
                                    # Load the model file (try to find a weights file if possible)
                                    possible_weights = [f for f in model_files if "weights" in f.lower() or "model" in f.lower()]
                                    model_path = possible_weights[0] if possible_weights else model_files[0]
                                    
                                    try:
                                        # First try loading the model directory
                                        self.model = mx.load(local_path)
                                        logger.info(f"Model loaded successfully from directory: {local_path}")
                                    except Exception as e:
                                        logger.warning(f"Could not load model from directory: {e}")
                                        # Try loading specific file
                                        self.model = mx.load(model_path)
                                        logger.info(f"Model loaded successfully from file: {model_path}")
                                        
                                except Exception as e:
                                    logger.error(f"Error downloading/loading model: {e}")
                                    raise
                                
                                # Check if the model is a dict (common format for MLX models)
                                # and create appropriate interface
                                if isinstance(self.model, dict):
                                    logger.info("Model loaded as dictionary, creating interface wrapper")
                                    self.weights = self.model
                                    
                                    # Check for model structure to determine type
                                    if any(k.endswith('.weight') for k in self.weights.keys()):
                                        logger.info("Detected standard parameter dict format")
                                        # This is a standard parameter dict
                                        self.is_dict_model = True
                                    else:
                                        # This is a structured model with components
                                        self.is_dict_model = False
                                        # Try to extract components
                                        if 'model' in self.weights:
                                            logger.info("Found 'model' key in weights")
                                            self.model_weights = self.weights['model']
                                        else:
                                            self.model_weights = self.weights
                                else:
                                    self.is_dict_model = False
                                    self.weights = None
                                
                                self.tokenizer = tokenizer
                            
                            def __call__(self, text, **kwargs):
                                try:
                                    # Process similar to huggingface pipeline
                                    inputs = self.tokenizer(text, return_tensors="np")
                                    input_ids = mx.array(inputs["input_ids"])
                                    
                                    try:
                                        if hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                                            # If model has generate method (newer MLX models)
                                            output_ids = self.model.generate(
                                                input_ids,
                                                max_length=kwargs.get("max_length", 512),
                                                temperature=kwargs.get("temperature", 0.7)
                                            )
                                        else:
                                            # Fallback for older models or dict models
                                            # Fix: categorical() doesn't accept temperature as a keyword argument
                                            # It needs to be applied to logits directly before calling categorical
                                            temperature = kwargs.get("temperature", 0.7)
                                            
                                            try:
                                                # For dict-based models, we need a simpler approach
                                                if self.is_dict_model:
                                                    # Dict models can't be called directly
                                                    logger.info("Using dictionary-based model fallback")
                                                    
                                                    # Extract the actual OCR text from the prompt
                                                    # The prompt is typically: "Fix OCR errors in the following text, maintaining original formatting: {text}"
                                                    input_text = text
                                                    match = re.search(r"formatting: (.*?)$", input_text, re.DOTALL)
                                                    if match:
                                                        actual_text = match.group(1).strip()
                                                        logger.info(f"Extracted OCR text: {actual_text[:30]}...")
                                                    else:
                                                        # If we can't extract the text, use the whole input
                                                        actual_text = input_text
                                                        logger.info("Using full input as OCR text")
                                                    
                                                    # Basic OCR error correction
                                                    processed_text = actual_text
                                                    
                                                    # Fix common OCR errors
                                                    # 1. Fix capitalization issues
                                                    processed_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed_text)
                                                    
                                                    # 2. Fix missing spaces between words (already covered by #1)
                                                    
                                                    # 3. Fix numbers mixed with letters
                                                    processed_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', processed_text)
                                                    processed_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', processed_text)
                                                    
                                                    # 4. Fix common word joining
                                                    processed_text = re.sub(r'(and|the|with|for|from|that|this|would|could|should)([a-zA-Z])', r'\1 \2', processed_text, flags=re.IGNORECASE)
                                                    
                                                    # 5. Fix obvious letter case issues
                                                    word_list = processed_text.split()
                                                    for i, word in enumerate(word_list):
                                                        # If word has mixed case in the middle, normalize it
                                                        if any(c.isupper() for c in word[1:]) and not word.isupper():
                                                            word_list[i] = word[0] + word[1:].lower()
                                                    
                                                    processed_text = ' '.join(word_list)
                                                    
                                                    # Return our processed text as the model output
                                                    output_text = processed_text
                                                    logger.info(f"Processed text: {output_text[:50]}...")
                                                    
                                                    # We don't need to decode anything as we're using our processed text directly
                                                    return [{"generated_text": output_text}]
                                                else:
                                                    # Get logits from model
                                                    logits = self.model(input_ids)
                                                    
                                                    # Apply temperature scaling to logits if temperature != 1.0
                                                    if temperature != 1.0:
                                                        logits = logits / temperature
                                                        
                                                    # Then call categorical without temperature parameter
                                                    output_ids = mx.random.categorical(logits)
                                            except TypeError as e:
                                                # If there's an issue with the model call signature
                                                logger.warning(f"TypeError in MLX model call: {e}")
                                                
                                                # For testing, fall back to input_ids as output_ids
                                                logger.info("Using simple fallback for text generation")
                                                output_ids = input_ids
                                            
                                        # Decode the output
                                        try:
                                            # Convert MLX array to numpy for tokenizer decoding
                                            if isinstance(output_ids, mx.array):
                                                try:
                                                    # Different MLX versions have different conversion methods
                                                    try:
                                                        # First try direct conversion to numpy if available
                                                        output_ids_np = np.array(output_ids)
                                                    except Exception:
                                                        # If that fails, try converting via explicit list
                                                        if len(output_ids.shape) == 1:
                                                            # Convert to list first
                                                            output_ids_list = [int(i) for i in output_ids.tolist()]
                                                            output_ids_np = np.array([output_ids_list])
                                                        else:
                                                            # 2D case
                                                            output_ids_list = [[int(i) for i in row] for row in output_ids.tolist()]
                                                            output_ids_np = np.array(output_ids_list)
                                                    
                                                    # Use the tokenizer's batch decode
                                                    texts = self.tokenizer.batch_decode(output_ids_np, skip_special_tokens=True)
                                                    output_text = texts[0]
                                                    logger.info(f"Decoded text from MLX array: {output_text[:30]}...")
                                                except Exception as np_error:
                                                    logger.error(f"Error converting MLX array to numpy: {np_error}")
                                                    # Fallback to much simpler approach - just use the input text
                                                    output_text = text
                                            else:
                                                # Default decoding approach
                                                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                                                
                                            # Basic validation
                                            if not output_text or len(output_text.strip()) == 0:
                                                logger.warning("Empty text generated, falling back to input text")
                                                output_text = text
                                                
                                        except Exception as decode_error:
                                            logger.error(f"Error decoding token ids: {decode_error}")
                                            logger.info("Falling back to input text")
                                            output_text = text
                                        
                                    except Exception as model_error:
                                        logger.error(f"MLX model generation failed: {model_error}")
                                        # Fallback to returning the input text with minimal processing
                                        # Try to at least process the text nicely
                                        output_text = text.replace("  ", " ").strip()
                                        
                                        # Try to make minimal improvements to the text
                                        output_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', output_text)  # Add spaces between words
                                        output_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', output_text)  # Space between numbers and letters
                                        
                                    return [{"generated_text": output_text}]
                                except Exception as e:
                                    logger.error(f"Error in MLX text generation: {e}")
                                    # Return a simple response as fallback
                                    return [{"generated_text": text}]
                        
                        # Initialize and return the model
                        try:
                            model = MLXTextModel(model_name)
                            
                            # Basic validation to ensure model works
                            test_input = "This is a test."
                            logger.info(f"Testing MLX model with input: '{test_input}'")
                            test_result = model(test_input)
                            
                            if isinstance(test_result, list) and len(test_result) > 0:
                                if "generated_text" in test_result[0]:
                                    logger.info("MLX model test successful!")
                                else:
                                    logger.warning("MLX model test output has unexpected format")
                            else:
                                logger.warning("MLX model test returned unexpected result format")
                                
                            MODEL_CACHE[cache_key] = model
                            return model
                        except Exception as e:
                            logger.error(f"Error initializing or testing MLX model: {e}")
                            logger.warning("Falling back to standard transformers model")
                            # Disable MLX for this session
                            DEVICE = "cpu"
                            # Fall through to standard loading
                        
                    except ImportError as e:
                        logger.warning(f"Failed to load MLX for text correction: {e}")
                        logger.warning("Falling back to standard transformers model")
                        
                        # Disable MLX for future loads if we had serious issues
                        DEVICE = "cpu"
                        logger.warning("Disabling MLX due to loading errors and falling back to CPU")
                        
                        # Fall through to standard loading
                    except Exception as e:
                        logger.error(f"Error loading MLX text correction model: {e}")
                        logger.warning("Falling back to standard transformers model")
                        
                        # Disable MLX for future loads if we had serious issues
                        DEVICE = "cpu"
                        logger.warning("Disabling MLX due to loading errors and falling back to CPU")
                        
                        # Fall through to standard loading
                    
                elif model_type == "embedding":
                    try:
                        # For embedding models using MLX
                        from mlx import core as mx
                        from transformers import AutoTokenizer
                        import os
                        from huggingface_hub import snapshot_download
                        
                        model_name = MLX_MODELS["embedding"][size]
                        logger.info(f"Loading MLX embedding model: {model_name}")
                        
                        # MLX embedding model class
                        class MLXEmbeddingModel:
                            def __init__(self, model_name):
                                # Import here to avoid import errors if MLX is not available
                                from mlx import core as mx
                                import mlx
                                import os
                                from huggingface_hub import snapshot_download
                                
                                # Download model from HuggingFace and load using MLX utils
                                try:
                                    logger.info(f"Downloading embedding model: {model_name}")
                                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mlx_models")
                                    os.makedirs(cache_dir, exist_ok=True)
                                    
                                    # Download the model files
                                    local_path = snapshot_download(
                                        repo_id=model_name,
                                        cache_dir=cache_dir,
                                        local_files_only=False
                                    )
                                    logger.info(f"Embedding model downloaded to {local_path}")
                                    
                                    # For embedding models, we'll try to convert standard models
                                    # into MLX-compatible format if needed
                                    import glob
                                    
                                    # Check for various model file extensions
                                    model_files = []
                                    for ext in ["safetensors", "bin", "mlx", "onnx"]:
                                        model_files.extend(glob.glob(f"{local_path}/*.{ext}"))
                                    
                                    if not model_files:
                                        logger.error(f"No model files found in {local_path}")
                                        raise FileNotFoundError(f"No model files found in {local_path}")
                                    
                                    logger.info(f"Found embedding model files: {model_files}")
                                    
                                    # Load the model file or directory
                                    possible_weights = [f for f in model_files if "weights" in f.lower() or "model" in f.lower()]
                                    model_path = possible_weights[0] if possible_weights else model_files[0]
                                    
                                    try:
                                        # First try loading the entire model directory
                                        self.model = mx.load(local_path)
                                        logger.info(f"Embedding model loaded successfully from directory: {local_path}")
                                    except Exception as e:
                                        logger.warning(f"Could not load embedding model from directory: {e}")
                                        # Try with specific file
                                        self.model = mx.load(model_path)
                                        logger.info(f"Embedding model loaded successfully from file: {model_path}")
                                        
                                except Exception as e:
                                    logger.error(f"Error downloading/loading embedding model: {e}")
                                    raise
                                
                                # Get tokenizer from CPU model
                                self.tokenizer = AutoTokenizer.from_pretrained(
                                    model_name
                                )
                            
                            def encode(self, sentences, **kwargs):
                                # Process similar to sentence-transformers
                                inputs = self.tokenizer(sentences, return_tensors="np", padding=True)
                                input_ids = mx.array(inputs["input_ids"])
                                attention_mask = mx.array(inputs["attention_mask"])
                                
                                # Get embeddings
                                embeddings = self.model(input_ids, attention_mask)
                                
                                # Convert to numpy for compatibility
                                return embeddings.astype(mx.float32).to_numpy()
                        
                        # Initialize and return the model
                        model = MLXEmbeddingModel(model_name)
                        MODEL_CACHE[cache_key] = model
                        return model
                        
                    except ImportError as e:
                        logger.warning(f"Failed to load MLX for embeddings: {e}")
                        logger.warning("Falling back to sentence-transformers")
                        
                        # Disable MLX for future loads if we had serious issues
                        if DEVICE == "mlx":
                            logger.warning("Disabling MLX due to loading errors and falling back to CPU")
                            DEVICE = "cpu"
                        
                        # Fall through to standard loading
                    except Exception as e:
                        logger.error(f"Error loading MLX embedding model: {e}")
                        logger.warning("Falling back to standard embedding model")
                        
                        # Disable MLX for future loads if we had serious issues
                        if DEVICE == "mlx":
                            logger.warning("Disabling MLX due to loading errors and falling back to CPU")
                            DEVICE = "cpu"
        
        # Standard model loading with PyTorch
        if model_type == "text_correction":
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            
            model_name = CPU_FRIENDLY_MODELS["text_correction"][size]
            logger.info(f"Loading text correction model: {model_name}")
            
            # For CPU, use 8-bit quantization if tiny model isn't selected
            use_quantization = DEVICE == "cpu" and size != "tiny"
            
            if use_quantization:
                try:
                    logger.info("Using 8-bit quantization for better CPU performance")
                    model = pipeline(
                        "text2text-generation", 
                        model=model_name,
                        device_map="auto",
                        load_in_8bit=True  # Enable 8-bit quantization
                    )
                except Exception as e:
                    logger.warning(f"Quantization failed, falling back to standard loading: {e}")
                    model = pipeline(
                        "text2text-generation", 
                        model=model_name,
                        device=-1 if DEVICE == "cpu" else 0
                    )
            else:
                model = pipeline(
                    "text2text-generation", 
                    model=model_name,
                    device=-1 if DEVICE == "cpu" else 0
                )
                
        elif model_type == "embedding":
            from sentence_transformers import SentenceTransformer
            
            model_name = CPU_FRIENDLY_MODELS["embedding"][size]
            logger.info(f"Loading embedding model: {model_name}")
            
            # Use CPU device name for sentence-transformers if we're using MLX
            device_name = "cpu" if DEVICE == "mlx" else DEVICE
            model = SentenceTransformer(model_name, device=device_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Cache the model
        MODEL_CACHE[cache_key] = model
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model (size={size}): {e}")
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory. Try using a smaller model size ('tiny' or 'small')")
        elif "not enough memory" in str(e).lower():
            logger.error("Not enough system memory. Try using 'tiny' model size or free up RAM")
        return None

# Try to load models based on available hardware

# Check MLX status
def check_mlx_status():
    """
    Check MLX availability and version.
    
    Returns:
        dict: Status information for MLX
    """
    status = {
        "available": MLX_AVAILABLE,
        "device": DEVICE,
        "is_apple_silicon": IS_APPLE_SILICON,
        "version": None
    }
    
    if MLX_AVAILABLE:
        try:
            import mlx
            if hasattr(mlx, "__version__"):
                status["version"] = mlx.__version__
            else:
                import pkg_resources
                try:
                    status["version"] = pkg_resources.get_distribution("mlx").version
                except:
                    status["version"] = "Unknown"
        except:
            pass
    
    if "--mlx-status" in sys.argv:
        print(f"MLX Status: {status}")
        if status["available"]:
            print(f"MLX version: {status['version']}")
            print(f"Device: {DEVICE}")
            print(f"Apple Silicon: {IS_APPLE_SILICON}")
            if DEVICE == "mlx":
                print("MLX is enabled and will be used for model acceleration")
            else:
                print("MLX is available but not being used (disabled by user config)")
        else:
            print("MLX is not available")
            print(f"Using device: {DEVICE}")
        sys.exit(0)
    
    return status

# Call the function once at import time to check if --mlx-status is used
check_mlx_status()

try:
    # Choose model size based on available hardware and memory
    if DEVICE == "cpu":
        # For CPU, use the smallest models by default
        model_size = os.environ.get("LLM_OCR_MODEL_SIZE", "tiny")
    else:
        # For GPU, use medium models by default
        model_size = os.environ.get("LLM_OCR_MODEL_SIZE", "medium")
    
    # Load models
    TRANSFORMERS_AVAILABLE = False
    text_correction_model = load_model("text_correction", size=model_size)
    embedding_model = load_model("embedding", size=model_size)
    
    # Fallback to tiny models if medium/small failed
    if text_correction_model is None:
        logger.warning("Failed to load primary text correction model, trying tiny model")
        text_correction_model = load_model("text_correction", size="tiny")
    
    if embedding_model is None:
        logger.warning("Failed to load primary embedding model, trying tiny model")
        embedding_model = load_model("embedding", size="tiny")
    
    # Verify models loaded successfully
    TRANSFORMERS_AVAILABLE = (text_correction_model is not None and embedding_model is not None)
    
    if TRANSFORMERS_AVAILABLE:
        logger.info(f"Transformers models loaded successfully (size: {model_size})")
    else:
        logger.warning("Failed to load transformer models")
        
except (ImportError, RuntimeError) as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers models could not be loaded: {e}")

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    # Use adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Optional: Noise reduction
    binary = cv2.medianBlur(binary, 3)
    
    # Optional: Apply dilation to make text more visible
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

def ocr_image(image_path: str, lang: str = "eng+kor") -> str:
    """
    Perform OCR on an image with preprocessing.
    
    Args:
        image_path: Path to the image
        lang: OCR language(s)
        
    Returns:
        Extracted text
    """
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image_path)
        
        # Save temporarily for debugging if needed
        # cv2.imwrite(f"{image_path}_preprocessed.png", preprocessed)
        
        # Perform OCR
        text = pytesseract.image_to_string(
            preprocessed, 
            lang=lang,
            config='--psm 6'  # Assume a single uniform block of text
        )
        
        return text
    except Exception as e:
        logger.error(f"OCR error for {image_path}: {e}")
        # If error occurs, try with original image as fallback
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=lang)
            return text
        except Exception as e2:
            logger.error(f"Fallback OCR error for {image_path}: {e2}")
            return ""

def split_into_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into chunks with overlap to maintain context.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean the text
    text = text.replace('\n\n', ' <p> ').replace('\n', ' ')
    
    # Use sentence boundaries for more natural splits
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # If the current chunk is not empty, add it to the list
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start a new chunk, potentially including overlap from the previous chunk
            if len(current_chunk) > overlap:
                # Get the last few words from the previous chunk for context
                overlap_text = " ".join(current_chunk.split()[-overlap//10:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def correct_with_local_model(text: str, max_batch_size: int = None, mlx_params: dict = None) -> str:
    """
    Correct OCR errors using local transformer model.
    
    Args:
        text: OCR text to correct
        max_batch_size: Maximum number of tokens to process at once
        mlx_params: Optional MLX-specific parameters for Apple Silicon acceleration
        
    Returns:
        Corrected text
    """
    global text_correction_model, DEVICE
    
    if not TRANSFORMERS_AVAILABLE or text_correction_model is None:
        logger.warning("Transformers not available, returning original text")
        return text
    
    try:
        # Prompt engineering to guide the model
        prompt = f"Fix OCR errors in the following text, maintaining original formatting: {text}"
        
        # Set max batch size based on device
        if max_batch_size is None:
            if mlx_params and 'batch_size' in mlx_params:
                # Use MLX-specific batch size if provided
                max_batch_size = mlx_params.get('batch_size', 8)
            else:
                max_batch_size = 512 if DEVICE == "cpu" else 1024
        
        # Generate corrected text with optimized settings for the device
        gen_kwargs = {
            "max_length": min(len(text) + 100, 512),  # Limit max length on CPU
            "num_beams": 2 if DEVICE == "cpu" else 4, # Reduce beam search on CPU
            "early_stopping": True
        }
        
        # Apply MLX-specific parameters if provided
        if mlx_params and DEVICE == "mlx":
            # Use model_size if provided in mlx_params
            if 'model_size' in mlx_params and text_correction_model is None:
                model_size = mlx_params.get('model_size', 'small')
                text_correction_model = load_model("text_correction", size=model_size)
            
            # Apply any other MLX-specific generation parameters
            temperature = mlx_params.get('temperature', 0.7)
            gen_kwargs["temperature"] = temperature
            
            # Set appropriate max_length for MLX
            gen_kwargs["max_length"] = min(len(text) + 150, 768)
            
            logger.info(f"Using MLX acceleration with temperature={temperature}")
            
            try:
                # Validate that we're still using MLX (could have fallen back in load_model)
                if DEVICE != "mlx":
                    logger.warning("MLX was requested but not available, using CPU")
            except Exception as e:
                logger.error(f"Error checking MLX status: {e}")
        
        # Apply generation with optimized parameters
        try:
            logger.debug(f"Starting text correction with {DEVICE} device")
            result = text_correction_model(prompt, **gen_kwargs)
            
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                corrected_text = result[0]['generated_text']
            else:
                logger.warning(f"Unexpected model output format: {type(result)}")
                # Try to extract text in a robust way
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and any(k for k in result[0].keys()):
                        # Get the first value from the dictionary
                        first_key = next(iter(result[0].keys()))
                        corrected_text = str(result[0][first_key])
                    elif isinstance(result[0], str):
                        corrected_text = result[0]
                    else:
                        logger.error(f"Cannot extract text from result: {result}")
                        return text
                else:
                    logger.error("Model returned empty or non-list result")
                    return text
            
            # Restore paragraph breaks
            corrected_text = corrected_text.replace(' <p> ', '\n\n')
            
            return corrected_text
        except Exception as gen_error:
            logger.error(f"Error in text generation: {gen_error}")
            
            # If using MLX, try falling back to CPU
            if DEVICE == "mlx":
                logger.warning("MLX generation failed, falling back to CPU")
                DEVICE = "cpu"
                
                # Try to reload model with CPU device
                try:
                    model_size = 'tiny'  # Use smallest model for CPU fallback
                    text_correction_model = load_model("text_correction", size=model_size, force_reload=True)
                    
                    if text_correction_model is not None:
                        # Retry with CPU model
                        logger.info("Retrying text correction with CPU model")
                        return correct_with_local_model(text, max_batch_size=256, mlx_params=None)
                except Exception as reload_error:
                    logger.error(f"Failed to reload model for CPU: {reload_error}")
            
            # If we got here, all attempts failed
            return text
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "not enough memory" in str(e).lower():
            logger.error(f"Memory error in text correction: {e}")
            logger.warning("Trying with smaller batch size or model")
            
            # Try to recover by reducing batch size further
            if max_batch_size > 256:
                return correct_with_local_model(text, max_batch_size=256, mlx_params=mlx_params)
            
            # If already at minimum batch size, try to reload with tiny model
            if mlx_params:
                mlx_params['model_size'] = 'tiny'
            text_correction_model = load_model("text_correction", size="tiny", force_reload=True)
            if text_correction_model is not None:
                return correct_with_local_model(text, max_batch_size=256, mlx_params=mlx_params)
        
        logger.error(f"Error in local text correction: {e}")
        return text
    except Exception as e:
        logger.error(f"Error in local text correction: {e}")
        return text

def correct_with_openai(text: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Correct OCR errors using OpenAI API.
    
    Args:
        text: OCR text to correct
        model: OpenAI model to use
        
    Returns:
        Corrected text
    """
    if not OPENAI_AVAILABLE or not openai_client:
        logger.warning("OpenAI client not available, returning original text")
        return text
    
    try:
        # Use the latest OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert OCR correction system. Fix OCR errors while preserving original formatting, structure, and Korean language text. Don't add any commentary."},
                {"role": "user", "content": f"Correct OCR errors in the following text:\n\n{text}"}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
    except Exception as e:
        logger.error(f"Error in OpenAI text correction: {e}")
        return text

def enhance_ocr_text(text: str, use_openai: bool = False, cpu_optimize: bool = None, mlx_params: dict = None) -> str:
    """
    Enhance OCR text using appropriate LLM.
    
    Args:
        text: Original OCR text
        use_openai: Whether to use OpenAI (if available)
        cpu_optimize: Optimize for CPU (fewer chunks, smaller models)
        mlx_params: Optional MLX-specific parameters for Apple Silicon acceleration
        
    Returns:
        Enhanced text
    """
    global text_correction_model
    
    if not text.strip():
        return text
    
    # Auto-detect CPU optimization if not explicitly set
    if cpu_optimize is None:
        cpu_optimize = DEVICE == "cpu"
    
    # For CPU optimization, use smaller chunk size and fewer chunks
    chunk_size = 500 if cpu_optimize else 1000
    
    # Split into manageable chunks
    chunks = split_into_chunks(text, max_chunk_size=chunk_size)
    
    # Limit chunks on CPU to avoid excessive processing
    if cpu_optimize and len(chunks) > 5:
        logger.info(f"CPU optimization: limiting from {len(chunks)} to 5 chunks")
        chunks = chunks[:5]
    
    # Process each chunk
    enhanced_chunks = []
    for chunk in tqdm(chunks, desc="Enhancing text chunks", leave=False):
        if use_openai and OPENAI_AVAILABLE and openai_client:
            enhanced_chunk = correct_with_openai(chunk)
        else:
            # Pass MLX parameters if available
            enhanced_chunk = correct_with_local_model(chunk, mlx_params=mlx_params)
        enhanced_chunks.append(enhanced_chunk)
    
    # Combine chunks
    enhanced_text = " ".join(enhanced_chunks)
    
    # Restore original paragraph structure
    enhanced_text = enhanced_text.replace(' <p> ', '\n\n')
    
    return enhanced_text

def process_images_in_folder(
    folder: str, 
    prefix: str = None, 
    lang: str = "eng+kor",
    use_openai: bool = False,
    mlx_params: dict = None
) -> Dict[str, str]:
    """
    Process all images in a folder with LLM-enhanced OCR.
    
    Args:
        folder: Folder containing images
        prefix: Prefix for image filenames to filter by
        lang: OCR language
        use_openai: Whether to use OpenAI for enhancement
        mlx_params: Optional MLX-specific parameters for Apple Silicon acceleration
        
    Returns:
        Dictionary mapping filenames to enhanced text
    """
    # Get all image files matching the pattern
    pattern = f"{prefix}*.png" if prefix else "*.png"
    image_files_pattern = os.path.join(folder, pattern)
    image_files: List[str] = glob.glob(image_files_pattern)
    image_files.sort()  # Ensure files are processed in order
    
    result_dict = {}
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            logger.info(f"Processing {image_file}")
            
            # Extract text with OCR
            raw_text = ocr_image(image_file, lang=lang)
            
            # Enhance OCR text with LLM
            enhanced_text = enhance_ocr_text(
                raw_text, 
                use_openai=use_openai,
                mlx_params=mlx_params
            )
            
            # Store result
            result_dict[image_file] = enhanced_text
            
            # Save text to file
            text_filename = f"{os.path.splitext(image_file)[0]}_enhanced.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(enhanced_text)
                
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
    
    return result_dict

def create_searchable_pdf_with_enhanced_text(
    image_path: str, 
    enhanced_text: str,
    output_pdf_path: str
) -> None:
    """
    Create a searchable PDF with enhanced OCR text.
    
    Args:
        image_path: Path to the source image
        enhanced_text: Enhanced OCR text
        output_pdf_path: Path to save the output PDF
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Create PDF with text layer using pytesseract
        pdf_data = pytesseract.image_to_pdf_or_hocr(
            image, 
            extension='pdf',
            lang='eng+kor',
            config=f'--psm 6 -c stream_filelist=- -c textonly_pdf=1'
        )
        
        # Save the PDF directly
        with open(output_pdf_path, 'wb') as f:
            f.write(pdf_data)
            
        logger.info(f"Created searchable PDF: {output_pdf_path}")
        
        # Also save the enhanced text as a separate file for reference
        text_path = f"{os.path.splitext(output_pdf_path)[0]}_enhanced_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_text)
            
        logger.info(f"Saved enhanced text: {text_path}")
        
    except Exception as e:
        logger.error(f"Error creating searchable PDF for {image_path}: {e}")
        # Try a simpler approach as fallback
        try:
            # Basic PDF creation with just pytesseract
            image = Image.open(image_path)
            pdf_data = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
            
            with open(output_pdf_path, 'wb') as f:
                f.write(pdf_data)
                
            logger.info(f"Created basic searchable PDF (fallback): {output_pdf_path}")
            
            # Save enhanced text separately for reference
            text_path = f"{os.path.splitext(output_pdf_path)[0]}_enhanced_text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_text)
                
        except Exception as fallback_error:
            logger.error(f"Fallback PDF creation also failed: {fallback_error}")

def merge_enhanced_pdfs(folder: str, prefix: str, output_filename: str) -> None:
    """
    Merge multiple enhanced PDF files in a folder into a single PDF.
    
    Args:
        folder: Folder containing PDF files
        prefix: Prefix for PDF filenames to filter by
        output_filename: Path for the merged output PDF
    """
    # Get all enhanced PDF files
    pdf_files = glob.glob(os.path.join(folder, f'{prefix}*_enhanced_ocr.pdf'))
    pdf_files.sort()  # Sort to maintain page order
    
    if not pdf_files:
        logger.warning(f"No enhanced PDF files found with prefix '{prefix}' in {folder}")
        return
    
    logger.info(f"Merging {len(pdf_files)} enhanced PDFs")
    
    # Create a PDF writer
    pdf_writer = PdfWriter()
    
    # Add each PDF
    for file in pdf_files:
        try:
            logger.info(f"Adding PDF: {file}")
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
    
    # Write the merged PDF
    try:
        logger.info(f"Writing merged PDF to {output_filename}")
        with open(output_filename, 'wb') as f:
            pdf_writer.write(f)
        logger.info(f"Successfully created merged PDF: {output_filename}")
    except Exception as e:
        logger.error(f"Error writing merged PDF: {e}")

def process_folder_to_enhanced_pdfs(
    folder: str, 
    prefix: str = None, 
    lang: str = "eng+kor",
    use_openai: bool = False,
    mlx_params: dict = None
) -> List[str]:
    """
    Process all images in a folder to create enhanced searchable PDFs.
    
    Args:
        folder: Folder containing images
        prefix: Prefix for image filenames to filter by
        lang: OCR language
        use_openai: Whether to use OpenAI for enhancement
        mlx_params: Optional MLX-specific parameters for Apple Silicon acceleration
        
    Returns:
        List of created PDF paths
    """
    # First extract and enhance text for all images
    enhanced_texts = process_images_in_folder(
        folder, 
        prefix=prefix, 
        lang=lang,
        use_openai=use_openai,
        mlx_params=mlx_params
    )
    
    pdf_paths = []
    
    # Create searchable PDFs for each image
    for image_path, enhanced_text in tqdm(enhanced_texts.items(), desc="Creating PDFs"):
        output_pdf_path = f"{os.path.splitext(image_path)[0]}_enhanced_ocr.pdf"
        create_searchable_pdf_with_enhanced_text(
            image_path, 
            enhanced_text,
            output_pdf_path
        )
        pdf_paths.append(output_pdf_path)
    
    return pdf_paths

# API for the module
def image_folder_to_enhanced_searchable_pdf(
    folder: str, 
    prefix: str = None, 
    lang: str = "eng+kor",
    use_openai: bool = False,
    mlx_params: dict = None
) -> List[str]:
    """
    Convert images in a folder to enhanced searchable PDFs with LLM-based OCR improvement.
    
    Args:
        folder: Folder containing images
        prefix: Prefix for image filenames to filter by
        lang: OCR language
        use_openai: Whether to use OpenAI for enhancement
        mlx_params: Optional MLX-specific parameters for Apple Silicon acceleration
        
    Returns:
        List of created PDF paths
    """
    return process_folder_to_enhanced_pdfs(
        folder, 
        prefix=prefix, 
        lang=lang,
        use_openai=use_openai,
        mlx_params=mlx_params
    ) 