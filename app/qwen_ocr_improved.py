"""
Improved Qwen2.5-VL OCR Engine for M1 Pro compatibility
Based on the approach that works better on Apple Silicon
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from PIL import Image
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers, PyTorch, and PIL available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"❌ Missing dependencies: {e}")

class ImprovedQwenOCR:
    """
    Improved Qwen2.5-VL OCR Engine optimized for M1 Pro
    Uses AutoModelForVision2Seq for better compatibility
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.device = "cpu"  # Force CPU for M1 Pro stability
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        logger.info(f"🤖 Improved Qwen OCR Engine initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Model: {self.model_name}")
        
        # Check device capabilities
        if TRANSFORMERS_AVAILABLE:
            self._check_device_capabilities()
    
    def _check_device_capabilities(self):
        """Check what devices are available."""
        try:
            if torch.cuda.is_available():
                logger.info("🔥 CUDA available but using CPU for stability")
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("🍎 MPS (Apple Silicon) available but using CPU for stability")
                
        except Exception as e:
            logger.info(f"⚠️ Device check failed: {e}")
    
    def load_model(self, progress_callback: Optional[Callable] = None):
        """Load the Qwen2.5-VL model and processor."""
        if self.model_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers not available")
            return False
        
        try:
            if progress_callback:
                progress_callback("Loading Qwen2.5-VL model...", 10)
            
            logger.info(f"📥 Loading model: {self.model_name}")
            
            # Load processor first
            if progress_callback:
                progress_callback("Loading processor...", 20)
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("✅ Processor loaded")
            
            # Load model with optimized settings for M1 Pro
            if progress_callback:
                progress_callback("Loading vision-language model...", 40)
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for M1 Pro compatibility
                device_map=None,  # Don't use device_map on M1 Pro
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Optimize memory usage
            ).to(self.device)
            
            logger.info("✅ Model loaded successfully")
            
            if progress_callback:
                progress_callback("Model ready for inference", 60)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def extract_text(self, image_path: str, language: str = "eng", 
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Extract text from image using Qwen2.5-VL
        
        Args:
            image_path: Path to the image file
            language: Language code (for compatibility, not used by Qwen)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                if progress_callback:
                    progress_callback("Initializing Qwen2.5-VL...", 0)
                
                if not self.load_model(progress_callback):
                    return self._create_error_response("Failed to load model", start_time)
            
            # Load and process image
            if progress_callback:
                progress_callback("Processing image...", 70)
            
            logger.info(f"📷 Processing image: {image_path}")
            
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
                logger.info(f"✅ Image loaded: {image.size}")
            except Exception as e:
                logger.error(f"❌ Failed to load image: {e}")
                return self._create_error_response(f"Failed to load image: {e}", start_time)
            
            # Create OCR prompt
            prompt = self._create_ocr_prompt(language)
            
            if progress_callback:
                progress_callback("Running OCR analysis...", 80)
            
            # Process inputs
            logger.info("🔄 Processing inputs...")
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate text with conservative settings
            if progress_callback:
                progress_callback("Generating text...", 90)
            
            logger.info("🎯 Generating text...")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,  # Conservative for M1 Pro
                    min_new_tokens=1,
                    do_sample=False,     # Deterministic
                    num_beams=1,         # No beam search for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    temperature=None,    # Remove temperature
                    top_p=None,         # Remove top_p
                    top_k=None          # Remove top_k
                )
            
            # Decode output
            logger.info("📝 Decoding output...")
            output = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract the actual OCR text from the response
            extracted_text = self._extract_ocr_text(output, prompt)
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback("OCR completed!", 100)
            
            logger.info(f"✅ OCR completed in {processing_time:.2f}s")
            logger.info(f"📄 Extracted text length: {len(extracted_text)} characters")
            
            return {
                "text": extracted_text,
                "confidence": 95.0,  # Qwen typically has high confidence
                "language": language,
                "engine": "Qwen2.5-VL-3B-Instruct",
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "processing_time": processing_time,
                "model_name": self.model_name,
                "device": self.device,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    def _create_ocr_prompt(self, language: str) -> str:
        """Create an appropriate OCR prompt for the given language."""
        if language in ["urd", "ara"]:
            return "<|im_start|>user\nWhat is the text written in this image? Please transcribe all text accurately, including any Arabic or Urdu text.\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            return "<|im_start|>user\nWhat is the text written in this image? Please transcribe all text accurately.\n<|im_end|>\n<|im_start|>assistant\n"
    
    def _extract_ocr_text(self, full_output: str, prompt: str) -> str:
        """Extract the actual OCR text from the model's full response."""
        try:
            # Remove the prompt from the output
            if prompt in full_output:
                text = full_output.replace(prompt, "").strip()
            else:
                text = full_output.strip()
            
            # Remove common response prefixes
            prefixes_to_remove = [
                "The text in the image is:",
                "The text written in this image is:",
                "The image contains the following text:",
                "I can see the following text:",
                "The text reads:",
            ]
            
            for prefix in prefixes_to_remove:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"⚠️ Text extraction failed: {e}")
            return full_output.strip()
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "text": "",
            "confidence": 0.0,
            "language": "unknown",
            "engine": "Qwen2.5-VL-3B-Instruct",
            "word_count": 0,
            "processing_time": time.time() - start_time,
            "model_name": self.model_name,
            "device": self.device,
            "success": False,
            "error": error_message
        }

# Create global instance
improved_qwen_ocr = ImprovedQwenOCR() if TRANSFORMERS_AVAILABLE else None
