"""
Robust Qwen2.5-VL OCR Engine with timeout and graceful fallback
Handles M1 Pro text generation hanging issue
"""

import logging
import time
import signal
import threading
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
    import transformers

    # Check transformers version
    logger.info(f"🔧 Transformers version: {transformers.__version__}")

    # Try to import qwen_vl_utils (may not be available in all environments)
    try:
        from qwen_vl_utils import process_vision_info
        QWEN_VL_UTILS_AVAILABLE = True
        logger.info("✅ qwen_vl_utils available")
    except ImportError:
        QWEN_VL_UTILS_AVAILABLE = False
        logger.warning("⚠️ qwen_vl_utils not available - using fallback approach")

    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ All dependencies available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"❌ Missing dependencies: {e}")

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

class RobustQwenOCR:
    """
    Robust Qwen2.5-VL OCR Engine with timeout handling
    Gracefully handles M1 Pro text generation hanging
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", timeout: int = 30):
        # Model compatibility fallback list (Linux-compatible variants)
        # Prioritize Qwen2.5-VL-3B and avoid 7B models
        self.model_candidates = [
            "Qwen/Qwen2.5-VL-3B-Instruct", # Primary target (3B model)
            "Qwen/Qwen-VL-Chat",           # Fallback (older but compatible)
            # Note: Avoiding 7B models as requested - too large
        ]

        self.model_name = model_name
        self.actual_model_used = None  # Track which model actually loaded
        self.device = "cpu"  # Force CPU for M1 Pro stability
        self.timeout = timeout  # Timeout for text generation
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.memory_issues_detected = False  # Track memory problems

        # Check available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"💾 Available memory: {available_memory_gb:.1f}GB")

            if available_memory_gb < 8:  # Less than 8GB available
                logger.warning(f"⚠️ Low memory detected ({available_memory_gb:.1f}GB). Qwen may fail.")
                self.memory_issues_detected = True
        except ImportError:
            logger.info("💾 psutil not available, cannot check memory")

        logger.info(f"🛡️  Robust Qwen OCR Engine initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"⏰ Timeout: {self.timeout}s")
        logger.info(f"🎯 Primary Model: {self.model_name}")
        logger.info(f"🔄 Fallback Models: {self.model_candidates[1:]}")

        if self.memory_issues_detected:
            logger.warning(f"⚠️ Memory constraints detected - recommend using PaddleOCR instead")
    
    def load_model(self, progress_callback: Optional[Callable] = None):
        """Load the Qwen2.5-VL model and processor."""
        if self.model_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers not available")
            return False
        
        # Try loading different models for compatibility
        model_loaded = False

        for model_candidate in self.model_candidates:
            if model_loaded:
                break

            try:
                if progress_callback:
                    progress_callback(f"Trying model: {model_candidate}...", 10)

                logger.info(f"📥 Trying to load model: {model_candidate}")

                # Load processor first to test model compatibility
                if progress_callback:
                    progress_callback("Loading processor...", 20)

                test_processor = AutoProcessor.from_pretrained(
                    model_candidate,
                    trust_remote_code=True
                )
                logger.info(f"✅ Processor loaded for {model_candidate}")

                # If processor loads successfully, try the model
                self.model_name = model_candidate
                self.actual_model_used = model_candidate
                self.processor = test_processor
                model_loaded = True

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_candidate}: {e}")
                continue

        if not model_loaded:
            raise Exception(f"Failed to load any compatible model from: {self.model_candidates}")

        logger.info(f"✅ Using compatible model: {self.actual_model_used}")

        try:
            # Load model with multiple fallback approaches for compatibility
            if progress_callback:
                progress_callback("Loading vision-language model...", 40)

            model_loaded = False

            # Try different loading approaches for maximum compatibility
            loading_approaches = [
                {
                    "name": "AutoModelForVision2Seq with float32",
                    "kwargs": {
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                },
                {
                    "name": "AutoModelForVision2Seq with float16",
                    "kwargs": {
                        "torch_dtype": torch.float16,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                },
                {
                    "name": "AutoModelForVision2Seq basic",
                    "kwargs": {
                        "trust_remote_code": True
                    }
                }
            ]

            for approach in loading_approaches:
                if model_loaded:
                    break
                try:
                    logger.info(f"🔄 Trying {approach['name']}...")
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        **approach['kwargs']
                    ).to(self.device)
                    logger.info(f"✅ Model loaded with {approach['name']}")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"⚠️ {approach['name']} failed: {e}")

            if not model_loaded:
                raise Exception("Failed to load model with any approach")

            logger.info("✅ Model loaded successfully")
            
            if progress_callback:
                progress_callback("Model ready for inference", 60)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _generate_with_timeout(self, inputs, generation_kwargs):
        """Generate text with timeout handling."""
        result = {"success": False, "output": None, "error": None}
        
        def target():
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **generation_kwargs)
                    result["output"] = generated_ids
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
            # Generation is still running - timeout occurred
            logger.warning(f"⏰ Text generation timed out after {self.timeout}s")
            result["error"] = f"Text generation timed out after {self.timeout}s"
            result["success"] = False
            # Note: We can't actually kill the thread, but we can return early
        
        return result
    
    def extract_text(self, image_path: str, language: str = "eng", 
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Extract text from image using Qwen2.5-VL with timeout protection
        """
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            if not self.model_loaded:
                if progress_callback:
                    progress_callback("Initializing Qwen2.5-VL...", 0)
                
                if not self.load_model(progress_callback):
                    return self._create_error_response("Failed to load model", start_time)
            
            # Process image
            if progress_callback:
                progress_callback("Processing image...", 70)
            
            logger.info(f"📷 Processing image: {image_path}")

            # Load image with memory optimization
            try:
                image = Image.open(image_path).convert("RGB")
                original_size = image.size
                logger.info(f"✅ Image loaded: {original_size}")

                # Resize large images to reduce memory usage (critical for cloud deployment)
                max_dimension = 1024  # Reduce from default to save memory
                if max(image.size) > max_dimension:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_dimension / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.LANCZOS)
                    logger.info(f"🔄 Image resized from {original_size} to {image.size} for memory optimization")

            except Exception as e:
                return self._create_error_response(f"Failed to load image: {e}", start_time)
            
            # Create message format and process inputs with fallback approaches
            if QWEN_VL_UTILS_AVAILABLE:
                # Use qwen_vl_utils approach (preferred)
                logger.info("🔄 Using qwen_vl_utils approach")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": self._create_ocr_prompt(language)},
                        ],
                    }
                ]

                # Apply chat template
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Process vision info
                image_inputs, video_inputs = process_vision_info(messages)

                # Process inputs
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            else:
                # Fallback approach without qwen_vl_utils
                logger.info("🔄 Using fallback approach (no qwen_vl_utils)")
                prompt = self._create_ocr_prompt(language)

                # Simple approach - process image and text directly
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            
            if progress_callback:
                progress_callback("Generating text (with timeout protection)...", 80)
            
            # Memory cleanup before generation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ultra-conservative generation for cloud memory limits
            generation_kwargs = {
                "max_new_tokens": 32,   # Reduced further to minimize memory
                "min_new_tokens": 1,
                "do_sample": False,     # Deterministic generation
                "num_beams": 1,         # Single beam to save memory
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": False,     # Disable cache to save memory
                "output_attentions": False,  # Disable attention outputs
                "output_hidden_states": False,  # Disable hidden states
            }

            # Generate with timeout protection
            logger.info(f"🎯 Generating text (timeout: {self.timeout}s)...")
            logger.info(f"💾 Memory optimization: max_tokens={generation_kwargs['max_new_tokens']}, cache=False")
            
            generation_result = self._generate_with_timeout(inputs, generation_kwargs)
            
            if not generation_result["success"]:
                error_msg = generation_result.get("error", "Generation failed")
                logger.error(f"❌ Generation failed: {error_msg}")
                return self._create_timeout_response(error_msg, start_time)
            
            # Decode output
            logger.info("📝 Decoding output...")
            generated_ids = generation_result["output"]
            
            # Trim input tokens
            input_token_len = inputs["input_ids"].shape[1]
            response_token_ids = generated_ids[:, input_token_len:]
            
            # Decode
            output = self.processor.batch_decode(
                response_token_ids, skip_special_tokens=True
            )[0]
            
            extracted_text = output.strip()
            processing_time = time.time() - start_time

            # Clean up memory to prevent accumulation
            del generated_ids, response_token_ids, inputs, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            if progress_callback:
                progress_callback("OCR completed!", 100)
            
            logger.info(f"✅ OCR completed in {processing_time:.2f}s")
            logger.info(f"📄 Extracted text length: {len(extracted_text)} characters")
            
            return {
                "text": extracted_text,
                "confidence": 90.0,  # High confidence for Qwen
                "language": language,
                "engine": "Qwen2.5-VL-3B-Instruct (Robust)",
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "processing_time": processing_time,
                "model_name": self.model_name,
                "device": self.device,
                "success": True,
                "timeout_used": self.timeout
            }
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    def _create_ocr_prompt(self, language: str) -> str:
        """Create an appropriate OCR prompt."""
        if language in ["urd", "ara"]:
            return "What is the text written in this image? Please transcribe all text accurately, including any Arabic or Urdu text."
        else:
            return "What is the text written in this image? Please transcribe all text accurately."
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "text": "",
            "confidence": 0.0,
            "language": "unknown",
            "engine": "Qwen2.5-VL-3B-Instruct (Robust)",
            "word_count": 0,
            "processing_time": time.time() - start_time,
            "model_name": self.model_name,
            "device": self.device,
            "success": False,
            "error": error_message,
            "timeout_used": self.timeout
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
            "device": self.device,
            "success": False,
            "error": error_message,
            "timeout_occurred": True,
            "timeout_used": self.timeout,
            "fallback_recommended": True
        }

# Create global instance with 30-second timeout
robust_qwen_ocr = RobustQwenOCR(timeout=30) if TRANSFORMERS_AVAILABLE else None
