"""
LLM enhancement functionality for the e-book page capture tool.
"""

import os
import logging
import platform
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from utils.config import get_settings

logger = logging.getLogger('llm')

class LLMEnhancer:
    """Class for handling LLM-based enhancements like OCR correction and text formatting."""
    
    def __init__(self, model_size: str = "small", use_mlx: Optional[bool] = None):
        """
        Initialize the LLM enhancer.
        
        Args:
            model_size: Size of the model to use ("tiny", "small", "medium")
            use_mlx: Whether to use MLX optimization on Apple Silicon (if None, auto-detect)
        """
        self.settings = get_settings()
        self.model_size = model_size
        
        # Determine if we should use MLX
        if use_mlx is None:
            # Auto-detect Apple Silicon and check if MLX is enabled in settings
            is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')
            use_mlx = is_apple_silicon and self.settings['llm']['mlx_enabled']
        
        self.use_mlx = use_mlx
        self.models_loaded = False
        self.text_model = None
        self.embedding_model = None
        
        logger.info(f"LLM Enhancer initialized with model_size={model_size}, use_mlx={use_mlx}")
    
    def load_models(self) -> bool:
        """
        Load the required models based on configuration.
        
        Returns:
            Boolean indicating success or failure
        """
        if self.models_loaded:
            logger.debug("Models already loaded")
            return True
        
        try:
            logger.debug("FLOW: Entering try block in load_models() - loading LLM models")
            # Get model names based on configuration
            if self.use_mlx:
                model_group = self.settings['llm']['mlx_models']
                logger.info("Using MLX-optimized models")
            else:
                model_group = self.settings['llm']['cpu_models']
                logger.info("Using CPU-friendly models")
            
            # Get the specific models for text correction and embedding
            text_model_name = model_group['text_correction'][self.model_size]
            embedding_model_name = model_group['embedding'][self.model_size]
            
            logger.info(f"Loading text model: {text_model_name}")
            logger.info(f"Loading embedding model: {embedding_model_name}")
            
            # Import required libraries
            if self.use_mlx:
                try:
                    logger.debug("FLOW: Entering try block for MLX model loading")
                    import mlx
                    import mlx.core as mx
                    from utils.mlx_utils import load_mlx_model, generate_with_mlx
                    
                    # Load text model
                    self.text_model = load_mlx_model(text_model_name)
                    
                    # For embedding, load appropriate model
                    self.embedding_model = load_mlx_model(embedding_model_name, model_type='embedding')
                        
                except ImportError as e:
                    logger.error(f"MLX import error: {e}")
                    logger.warning("Falling back to CPU models")
                    self.use_mlx = False
                    return self.load_models()  # Retry with CPU models
            else:
                # Using standard PyTorch/Transformers
                try:
                    logger.debug("FLOW: Entering try block for PyTorch model loading")
                    import torch
                    from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
                    
                    # Load the text correction model - handle different model types
                    if "bart" in text_model_name.lower():
                        self.text_model = {
                            'model': AutoModelForSeq2SeqLM.from_pretrained(text_model_name),
                            'tokenizer': AutoTokenizer.from_pretrained(text_model_name)
                        }
                    elif "bert" in text_model_name.lower() or "roberta" in text_model_name.lower():
                        self.text_model = {
                            'model': AutoModelForSequenceClassification.from_pretrained(text_model_name),
                            'tokenizer': AutoTokenizer.from_pretrained(text_model_name)
                        }
                    else:
                        # Default sequence-to-sequence model
                        self.text_model = {
                            'model': AutoModelForSeq2SeqLM.from_pretrained(text_model_name),
                            'tokenizer': AutoTokenizer.from_pretrained(text_model_name)
                        }
                    
                    # Load the embedding model
                    self.embedding_model = {
                        'model': AutoModel.from_pretrained(embedding_model_name),
                        'tokenizer': AutoTokenizer.from_pretrained(embedding_model_name)
                    }
                    
                except ImportError as e:
                    logger.error(f"PyTorch/Transformers import error: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Error loading models: {e}")
                    return False
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def correct_text(self, text: str, instruction: Optional[str] = None) -> str:
        """
        Correct and enhance text using the loaded text model.
        
        Args:
            text: Input text to correct
            instruction: Optional instruction for the model
            
        Returns:
            Corrected text
        """
        if not text:
            logger.warning("Empty text provided for correction")
            return ""
        
        # Load models if not already loaded
        if not self.models_loaded:
            if not self.load_models():
                logger.error("Failed to load models, returning original text")
                return text
        
        try:
            logger.debug("FLOW: Entering try block in correct_text() - processing text correction")
            # Default instruction if none provided
            if instruction is None:
                instruction = "Fix OCR errors, correct spacing and punctuation. Maintain original format."
            
            logger.info(f"Correcting text with size: {len(text)} characters")
            
            # Use MLX or PyTorch depending on configuration
            if self.use_mlx:
                try:
                    logger.debug("FLOW: Entering MLX text generation try block")
                    from utils.mlx_utils import generate_with_mlx
                    
                    # Prepare input for MLX model
                    prompt = f"{instruction}:\n\n{text}"
                    corrected = generate_with_mlx(self.text_model, prompt, max_tokens=len(text) * 2)
                    
                    logger.info(f"MLX text correction complete: {len(corrected)} characters")
                    return corrected
                except Exception as e:
                    logger.error(f"MLX text correction error: {e}")
                    return text
            else:
                try:
                    logger.debug("FLOW: Entering PyTorch text generation try block")
                    # Use PyTorch/Transformers
                    model = self.text_model['model']
                    tokenizer = self.text_model['tokenizer']
                    
                    # Determine model type and handle accordingly
                    model_name = type(model).__name__.lower()
                    
                    if "seq2seq" in model_name or "bart" in model_name:
                        # Sequence-to-sequence model (like BART)
                        prompt = f"{instruction}:\n\n{text}"
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                        
                        # Generate output
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=min(len(text) * 2, 1024),
                            num_beams=4,
                            temperature=0.7,
                            early_stopping=True
                        )
                        
                        # Decode output
                        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        # For classification models (BERT, RoBERTa), we'll use them for sentence fixing
                        # This is a simplified approach - in real-world usage, you might want a more sophisticated method
                        logger.warning("Using classification model for text correction - limited capability")
                        sentences = text.split(". ")
                        corrected_sentences = []
                        
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                                
                            # Simple corrections to fix common OCR issues
                            fixed = sentence.strip()
                            # Add more OCR fix patterns here as needed
                                
                            corrected_sentences.append(fixed)
                        
                        corrected = ". ".join(corrected_sentences)
                        if not corrected.endswith("."):
                            corrected += "."
                    
                    logger.info(f"PyTorch text correction complete: {len(corrected)} characters")
                    return corrected
                except Exception as e:
                    logger.error(f"PyTorch text correction error: {e}")
                    return text
        except Exception as e:
            logger.error(f"Text correction error: {e}")
            return text
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors or None if failed
        """
        if not texts:
            logger.warning("Empty texts provided for embedding")
            return []
        
        # Load models if not already loaded
        if not self.models_loaded:
            if not self.load_models():
                logger.error("Failed to load models, cannot generate embeddings")
                return None
        
        try:
            logger.debug("FLOW: Entering try block in get_embeddings() - generating embeddings")
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings based on model type
            if isinstance(self.embedding_model, type({})) and 'tokenizer' in self.embedding_model:
                # Standard Hugging Face model
                try:
                    logger.debug("FLOW: Entering standard model embedding generation try block")
                    import torch
                    import numpy as np
                    
                    model = self.embedding_model['model']
                    tokenizer = self.embedding_model['tokenizer']
                    
                    # Process each text and get embeddings
                    all_embeddings = []
                    for text in texts:
                        # Tokenize
                        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        
                        # Get model output
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Get the embeddings (use the CLS token embedding or mean pooling)
                        if hasattr(outputs, "pooler_output"):
                            # BERT/RoBERTa models have this
                            embedding = outputs.pooler_output.numpy()[0]
                        else:
                            # Fallback: use mean of last hidden state
                            embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                        
                        all_embeddings.append(embedding.tolist())
                    
                    return all_embeddings
                except Exception as e:
                    logger.error(f"Standard model embedding error: {e}")
                    return None
            else:
                # This assumes it's an MLX model or other type
                try:
                    logger.debug("FLOW: Entering MLX embedding generation try block")
                    from utils.mlx_utils import get_embeddings_with_mlx
                    
                    embeddings = get_embeddings_with_mlx(
                        self.embedding_model['model'],
                        self.embedding_model['tokenizer'],
                        texts
                    )
                    return embeddings
                except Exception as e:
                    logger.error(f"MLX embedding error: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to split
            max_length: Maximum chunk length
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        # Split by newlines first to maintain structure
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(paragraph) + 1 <= max_length:
                if current_chunk:
                    current_chunk += '\n' + paragraph
                else:
                    current_chunk = paragraph
            # If paragraph is too big alone, split it further
            elif len(paragraph) > max_length:
                # If we have accumulated text, add it as a chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split large paragraph by sentences or spaces
                if '.' in paragraph:
                    sentences = paragraph.split('.')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        sentence = sentence.strip() + '.'
                        
                        if len(temp_chunk) + len(sentence) + 1 <= max_length:
                            if temp_chunk:
                                temp_chunk += ' ' + sentence
                            else:
                                temp_chunk = sentence
                        else:
                            chunks.append(temp_chunk)
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    # Last resort: split by spaces
                    words = paragraph.split()
                    temp_chunk = ""
                    
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_length:
                            if temp_chunk:
                                temp_chunk += ' ' + word
                            else:
                                temp_chunk = word
                        else:
                            chunks.append(temp_chunk)
                            temp_chunk = word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                # Current paragraph would make chunk too big, finish current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap between chunks
        if overlap > 0 and len(chunks) > 1:
            overlap_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlap_chunks.append(chunks[i])
                else:
                    prev_chunk = chunks[i-1]
                    current_chunk = chunks[i]
                    
                    # Get overlap from previous chunk
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    
                    # Add overlap to current chunk
                    overlap_chunks.append(overlap_text + current_chunk)
            
            chunks = overlap_chunks
        
        return chunks
    
    def enhance_ocr_text(self, text: str, max_chunk_size: int = 512) -> str:
        """
        Enhance OCR text using LLM correction.
        
        Args:
            text: OCR text to enhance
            max_chunk_size: Maximum chunk size for processing
            
        Returns:
            Enhanced text
        """
        if not text:
            return ""
        
        # If text is shorter than max chunk size, process it directly
        if len(text) <= max_chunk_size:
            return self.correct_text(text)
        
        # Split text into chunks
        chunks = self.chunk_text(text, max_chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks for processing")
        
        # Process each chunk
        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            corrected = self.correct_text(chunk)
            corrected_chunks.append(corrected)
        
        # Rejoin chunks
        result = '\n'.join(corrected_chunks)
        
        return result
    
    def enhance_text(self, text: str) -> str:
        """
        Public method to enhance OCR text.
        
        Args:
            text: OCR text to enhance
            
        Returns:
            Enhanced text
        """
        if not text:
            logger.warning("Empty text provided for enhancement")
            return ""
        
        # Use internal enhance_ocr_text method
        return self.enhance_ocr_text(text)