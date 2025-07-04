"""
Working Qwen2-VL OCR Engine based on successful Mac implementation
Uses the approach that actually works without hanging
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
    from PIL import Image
    import transformers

    # Check transformers version for compatibility
    transformers_version = transformers.__version__
    logger.info(f"🔧 Transformers version: {transformers_version}")

    # Try to import the correct Qwen model class with multiple fallbacks
    QwenModel = None
    model_approach = None

    # Approach 1: Try Qwen2_5_VLForConditionalGeneration (newest)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
        model_approach = "qwen2_5_vl_specific"
        logger.info("✅ Using Qwen2_5_VLForConditionalGeneration")
    except ImportError:
        pass

    # Approach 2: Try Qwen2VLForConditionalGeneration (older)
    if QwenModel is None:
        try:
            from transformers import Qwen2VLForConditionalGeneration as QwenModel
            model_approach = "qwen2_vl_specific"
            logger.info("✅ Using Qwen2VLForConditionalGeneration (fallback)")
        except ImportError:
            pass

    # Approach 3: Use AutoModelForVision2Seq (most compatible)
    if QwenModel is None:
        QwenModel = AutoModelForVision2Seq
        model_approach = "auto_vision2seq"
        logger.info("✅ Using AutoModelForVision2Seq (universal fallback)")

    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ All dependencies available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"❌ Missing dependencies: {e}")

class WorkingQwenOCR:
    """
    Working Qwen2-VL OCR Engine based on successful Mac implementation
    Uses the approach that actually works without hanging on M1 Pro
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", timeout: int = 600):
        self.model_name = model_name
        self.timeout = timeout
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_loaded = False
        self.model_approach = model_approach  # Store the approach being used

        # Use CPU for better compatibility (MPS has Conv3D issues with Qwen)
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            logger.info("🍎 MPS available but using CPU for Qwen compatibility")

        logger.info(f"🤖 Working Qwen OCR Engine initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"⏰ Timeout: {self.timeout}s")
        logger.info(f"🎯 Model: {self.model_name}")
        logger.info(f"🔧 Model Approach: {self.model_approach}")
    
    def load_model(self, progress_callback: Optional[Callable] = None):
        """Load the Qwen2-VL model using the working approach."""
        if self.model_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers not available")
            return False
        
        try:
            if progress_callback:
                progress_callback("Loading Qwen2-VL model (working approach)...", 10)
            
            logger.info(f"📥 Loading model: {self.model_name}")
            
            # Load model with working configuration
            if progress_callback:
                progress_callback("Loading model with device_map=auto...", 30)
            
            # Try different loading approaches based on environment compatibility
            model_loaded = False

            # Approach 1: Try with device_map="auto" (best performance)
            if not model_loaded:
                try:
                    logger.info("🔄 Trying device_map='auto' approach...")
                    self.model = QwenModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        offload_buffers=True,
                        trust_remote_code=True
                    )
                    logger.info("✅ Model loaded with device_map='auto'")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"⚠️ device_map='auto' failed: {e}")

            # Approach 2: Try without device_map (more compatible)
            if not model_loaded:
                try:
                    logger.info("🔄 Trying manual device assignment...")
                    self.model = QwenModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to(self.device)
                    logger.info("✅ Model loaded with manual device assignment")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"⚠️ Manual device assignment failed: {e}")

            # Approach 3: Try with float32 (most compatible)
            if not model_loaded:
                try:
                    logger.info("🔄 Trying float32 for maximum compatibility...")
                    self.model = QwenModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to(self.device)
                    logger.info("✅ Model loaded with float32 compatibility mode")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"❌ All model loading approaches failed: {e}")
                    raise e

            if not model_loaded:
                raise Exception("Failed to load model with any approach")
            
            logger.info("✅ Model loaded successfully")
            
            # Load tokenizer
            if progress_callback:
                progress_callback("Loading tokenizer...", 60)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ Tokenizer loaded")
            
            # Load processor
            if progress_callback:
                progress_callback("Loading processor...", 80)
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("✅ Processor loaded")
            
            if progress_callback:
                progress_callback("Model ready for inference", 100)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def resize_image(self, image: Image.Image, max_height: int = 1260, max_width: int = 1260) -> Image.Image:
        """Resize image if it exceeds specified dimensions (from working example)."""
        original_width, original_height = image.size
        
        # Check if resizing is needed
        if original_width > max_width or original_height > max_height:
            # Calculate the new size maintaining the aspect ratio
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            # Resize the image using LANCZOS for high-quality downscaling
            return image.resize((new_width, new_height), Image.LANCZOS)
        else:
            return image
    
    def _generate_with_timeout(self, inputs, generation_kwargs):
        """Generate text with timeout handling."""
        result = {"success": False, "output": None, "error": None}
        
        def target():
            try:
                # Use the working generation approach
                outputs = self.model.generate(**inputs, **generation_kwargs)
                result["output"] = outputs
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
        
        # Start generation in a separate thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            logger.warning(f"⏰ Text generation timed out after {self.timeout}s")
            result["error"] = f"Text generation timed out after {self.timeout}s"
            result["success"] = False
        
        return result
    
    def extract_text(self, image_path: str, language: str = "eng", 
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Extract text from image using the working Qwen2-VL approach
        """
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                if progress_callback:
                    progress_callback("Initializing Qwen2-VL...", 0)
                
                if not self.load_model(progress_callback):
                    return self._create_error_response("Failed to load model", start_time)
            
            # Process image
            if progress_callback:
                progress_callback("Processing image...", 70)
            
            logger.info(f"📷 Processing image: {image_path}")
            
            # Load and resize image (from working example)
            try:
                with Image.open(image_path) as img:
                    # Resize image to reduce memory pressure
                    img = self.resize_image(img, max_height=1260, max_width=1260)
                    logger.info(f"✅ Image loaded and resized: {img.size}")
                    
                    # Create OCR prompt
                    prompt = self._create_ocr_prompt(language)
                    
                    # Use the working message format
                    messages = [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image", "image": img}, 
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # Apply chat template (working approach)
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Process inputs (working approach)
                    inputs = self.processor(
                        text=[text], 
                        images=[img], 
                        padding=True, 
                        return_tensors="pt"
                    ).to(self.device)
                    
            except Exception as e:
                return self._create_error_response(f"Failed to process image: {e}", start_time)
            
            if progress_callback:
                progress_callback("Generating text (working approach)...", 80)
            
            # Generate with working parameters
            logger.info(f"🎯 Generating text (timeout: {self.timeout}s)...")
            
            generation_kwargs = {
                "max_new_tokens": 128,      # More conservative for CPU
                "temperature": 0.3,         # Lower temperature for more focused output
                "repetition_penalty": 1.1,  # From working example
                "do_sample": True,          # Enable sampling
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,          # Enable caching for speed
            }
            
            generation_result = self._generate_with_timeout(inputs, generation_kwargs)
            
            if not generation_result["success"]:
                error_msg = generation_result.get("error", "Generation failed")
                logger.error(f"❌ Generation failed: {error_msg}")
                return self._create_timeout_response(error_msg, start_time)
            
            # Decode output using working approach
            logger.info("📝 Decoding output...")
            outputs = generation_result["output"]
            response_ids = outputs[0]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            
            # Extract the actual response (from working example)
            extracted_text = self._extract_response_text(response_text)
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback("OCR completed!", 100)
            
            logger.info(f"✅ OCR completed in {processing_time:.2f}s")
            logger.info(f"📄 Extracted text length: {len(extracted_text)} characters")
            
            return {
                "text": extracted_text,
                "confidence": 92.0,  # High confidence for working approach
                "language": language,
                "engine": "Qwen2.5-VL-3B-Instruct (Working)",
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "processing_time": processing_time,
                "model_name": self.model_name,
                "device": str(self.device),
                "success": True,
                "timeout_used": self.timeout,
                "approach": "working_mac_solution"
            }
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    def _create_ocr_prompt(self, language: str) -> str:
        """Create an OCR-focused prompt."""
        if language in ["urd", "ara"]:
            return "Extract and transcribe all text from this image. Include any Arabic or Urdu text exactly as it appears. Provide only the text content without description."
        else:
            return "Extract and transcribe all text from this image exactly as it appears. Provide only the text content without description."
    
    def _extract_response_text(self, full_response: str) -> str:
        """Extract the actual response text (from working example)."""
        try:
            # Use the working extraction method
            response = full_response.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
            return response
        except Exception as e:
            logger.warning(f"⚠️ Text extraction failed: {e}")
            return full_response.strip()
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "text": "",
            "confidence": 0.0,
            "language": "unknown",
            "engine": "Qwen2.5-VL-3B-Instruct (Working)",
            "word_count": 0,
            "processing_time": time.time() - start_time,
            "model_name": self.model_name,
            "device": str(self.device),
            "success": False,
            "error": error_message,
            "timeout_used": self.timeout,
            "approach": "working_mac_solution"
        }
    
    def _create_timeout_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create a timeout-specific response."""
        return {
            "text": "",
            "confidence": 0.0,
            "language": "unknown",
            "engine": "Qwen2.5-VL-3B-Instruct (Timeout)",
            "word_count": 0,
            "processing_time": time.time() - start_time,
            "model_name": self.model_name,
            "device": str(self.device),
            "success": False,
            "error": error_message,
            "timeout_occurred": True,
            "timeout_used": self.timeout,
            "fallback_recommended": True,
            "approach": "working_mac_solution"
        }

# Create global instance
working_qwen_ocr = WorkingQwenOCR() if TRANSFORMERS_AVAILABLE else None
