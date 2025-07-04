"""Qwen2.5-VL OCR Engine for text extraction from images."""

import logging
from PIL import Image
from typing import Dict, Any, Optional
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers and torch, but handle gracefully if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor
    # Use the CORRECT model class for Qwen2.5-VL
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN_MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
        logger.info("Using Qwen2_5_VLForConditionalGeneration (correct class)")
    except ImportError:
        # Fallback to old class if new one not available
        from transformers import Qwen2VLForConditionalGeneration
        QWEN_MODEL_CLASS = Qwen2VLForConditionalGeneration
        logger.warning("Using Qwen2VLForConditionalGeneration (old class - may cause issues)")

    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers, PyTorch, and qwen-vl-utils available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Required libraries not available: {e}")
    logger.info("Running in demo mode without Qwen2.5-VL model")

class QwenOCREngine:
    """OCR engine using Qwen2.5-VL-3B-Instruct model (optimized for M1 Pro)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cpu"  # Force CPU for compatibility
        self.model_loaded = False
        logger.info(f"Qwen OCR Engine initialized. Device: {self.device}")
        logger.info(f"Model will be loaded on first use: {self.model_name}")

        # Try to detect if we can use CUDA safely
        if TRANSFORMERS_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA available but using CPU for stability")
            except Exception as e:
                logger.info(f"CUDA check failed: {e}")
        
    def load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        if self.model_loaded:
            return True

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - running in demo mode")
            return False

        try:
            logger.info(f"Loading {self.model_name}...")
            logger.info("This may take several minutes for the first time...")

            # Load model using the CORRECT class for Qwen2.5-VL
            self.model = QWEN_MODEL_CLASS.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model_loaded = True
            logger.info("Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Make sure you have sufficient memory and internet connection")
            return False
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model_loaded and self.model is not None and self.processor is not None
    
    def extract_text(self, image_path: str, language: str = "eng", progress_callback=None) -> Dict[str, Any]:
        """
        Extract text from image using Qwen2.5-VL.

        Args:
            image_path: Path to the image file
            language: Language code (eng, urd, etc.)

        Returns:
            Dictionary with extracted text and metadata
        """
        start_time = time.time()

        # Progress tracking
        if progress_callback:
            try:
                if hasattr(progress_callback, '__call__'):
                    progress_callback("Initializing OCR process...", 0)
            except Exception:
                pass  # Ignore progress callback errors

        # If transformers not available, return demo response
        if not TRANSFORMERS_AVAILABLE:
            return self._create_demo_response(image_path, language, start_time)

        if progress_callback:
            progress_callback("Loading Qwen2.5-VL model...", 10)

        if not self.is_available():
            if not self.load_model():
                return self._create_demo_response(image_path, language, start_time)

        if progress_callback:
            progress_callback("Model loaded, preparing image...", 30)

        try:
            # Load and prepare image - resize to prevent tensor mask issues
            image = Image.open(image_path).convert('RGB')
            # Resize to square to prevent token mask mismatches
            max_size = 1024
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            if progress_callback:
                progress_callback("Image prepared, creating prompt...", 40)

            # Create language-specific prompt
            prompt = self._create_ocr_prompt(language)

            # Prepare conversation format for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Use the CORRECT Qwen2.5-VL approach following the official example
            if progress_callback:
                progress_callback("Processing vision inputs...", 50)

            # Step 1: Apply chat template to get text prompt
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Step 2: Process vision info to get properly formatted inputs
            image_inputs, video_inputs = process_vision_info(messages)

            if progress_callback:
                progress_callback("Tokenizing inputs...", 60)

            # Step 3: Tokenize inputs - KEY FIX: text as list!
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )

            # Move to device (following the exact example pattern)
            inputs = inputs.to(self.model.device)

            if progress_callback:
                progress_callback("Generating text (this may take a while)...", 70)

            # Step 4: Generate response with timeout and optimized settings
            with torch.no_grad():
                # Use much more conservative settings for M1 Pro
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,   # Very conservative for M1 Pro
                    min_new_tokens=5,     # Ensure some meaningful output
                    do_sample=False,      # Deterministic output
                    num_beams=1,          # No beam search for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,       # Enable KV cache
                    repetition_penalty=1.1,  # Prevent repetition
                    length_penalty=1.0,   # No length penalty
                    temperature=None,     # Remove temperature for deterministic output
                    top_p=None,          # Remove top_p for deterministic output
                    top_k=None           # Remove top_k for deterministic output
                )

            if progress_callback:
                progress_callback("Processing generated output...", 90)

            # Step 5: Trim output (exact pattern from example)
            output_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]

            # Step 6: Decode response (exact pattern from example)
            response_list = self.processor.batch_decode(output_trimmed, skip_special_tokens=True)

            response = response_list[0] if response_list else ""

            if progress_callback:
                progress_callback("Finalizing results...", 95)

            processing_time = time.time() - start_time

            # Parse and structure the response
            result = self._parse_ocr_response(response, language, processing_time)

            if progress_callback:
                progress_callback("OCR completed successfully!", 100)

            logger.info(f"OCR completed in {processing_time:.2f}s for language: {language}")
            return result

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "engine": "qwen2.5-vl",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _create_ocr_prompt(self, language: str) -> str:
        """Create language-specific OCR prompt."""
        
        language_prompts = {
            "eng": """Please extract all text from this image. Return only the text content exactly as it appears, maintaining the original formatting and line breaks. Do not add any explanations or descriptions.""",
            
            "urd": """اس تصویر سے تمام متن نکالیں۔ صرف متن کا مواد واپس کریں جیسا کہ یہ ظاہر ہوتا ہے، اصل فارمیٹنگ اور لائن بریکس کو برقرار رکھتے ہوئے۔ کوئی وضاحت یا تفصیل شامل نہ کریں۔

Please extract all text from this image in Urdu. Return only the text content exactly as it appears.""",
            
            "ara": """يرجى استخراج جميع النصوص من هذه الصورة. أعد فقط محتوى النص كما يظهر بالضبط، مع الحفاظ على التنسيق الأصلي وفواصل الأسطر. لا تضف أي تفسيرات أو أوصاف.

Please extract all text from this image in Arabic. Return only the text content exactly as it appears.""",
            
            "fra": """Veuillez extraire tout le texte de cette image. Retournez uniquement le contenu du texte exactement tel qu'il apparaît, en conservant le formatage original et les sauts de ligne. N'ajoutez aucune explication ou description.""",
            
            "deu": """Bitte extrahieren Sie den gesamten Text aus diesem Bild. Geben Sie nur den Textinhalt genau so zurück, wie er erscheint, unter Beibehaltung der ursprünglichen Formatierung und Zeilenumbrüche. Fügen Sie keine Erklärungen oder Beschreibungen hinzu.""",
            
            "spa": """Por favor, extraiga todo el texto de esta imagen. Devuelva solo el contenido del texto exactamente como aparece, manteniendo el formato original y los saltos de línea. No agregue explicaciones o descripciones."""
        }
        
        return language_prompts.get(language, language_prompts["eng"])
    
    def _parse_ocr_response(self, response: str, language: str, processing_time: float) -> Dict[str, Any]:
        """Parse the model response and structure the result."""
        
        # Clean up the response
        text = response.strip()
        
        # Remove common prefixes that the model might add
        prefixes_to_remove = [
            "The text in the image is:",
            "The text content is:",
            "Text extracted:",
            "The image contains:",
            "I can see the following text:",
            "Here is the text from the image:",
            "The text reads:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Estimate confidence based on text length and structure
        confidence = self._estimate_confidence(text)
        
        # Count words
        word_count = len(text.split()) if text else 0
        
        return {
            "text": text,
            "confidence": confidence,
            "language": language,
            "engine": "qwen2.5-vl",
            "word_count": word_count,
            "processing_time": processing_time,
            "model_name": self.model_name,
            "device": self.device
        }
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics."""
        if not text:
            return 0.0
        
        # Basic confidence estimation
        base_confidence = 85.0
        
        # Reduce confidence for very short text
        if len(text) < 10:
            base_confidence -= 20.0
        
        # Reduce confidence for text with many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            base_confidence -= 15.0
        
        # Increase confidence for well-structured text
        if any(word.istitle() for word in text.split()):
            base_confidence += 5.0
        
        return max(0.0, min(100.0, base_confidence))

    def _create_demo_response(self, image_path: str, language: str, start_time: float) -> Dict[str, Any]:
        """Create a demo response when the model is not available."""
        processing_time = time.time() - start_time

        # Create demo text based on language
        demo_texts = {
            "eng": "This is a demo response. The Qwen2.5-VL model is not loaded. Please install the required dependencies to use the actual OCR functionality.",
            "urd": "یہ ایک ڈیمو جواب ہے۔ Qwen2.5-VL ماڈل لوڈ نہیں ہے۔ اصل OCR فیچر استعمال کرنے کے لیے ضروری dependencies انسٹال کریں۔",
            "ara": "هذه استجابة تجريبية. نموذج Qwen2.5-VL غير محمل. يرجى تثبيت التبعيات المطلوبة لاستخدام وظيفة OCR الفعلية.",
            "fra": "Ceci est une réponse de démonstration. Le modèle Qwen2.5-VL n'est pas chargé. Veuillez installer les dépendances requises pour utiliser la fonctionnalité OCR réelle.",
            "deu": "Dies ist eine Demo-Antwort. Das Qwen2.5-VL-Modell ist nicht geladen. Bitte installieren Sie die erforderlichen Abhängigkeiten, um die tatsächliche OCR-Funktionalität zu nutzen.",
            "spa": "Esta es una respuesta de demostración. El modelo Qwen2.5-VL no está cargado. Instale las dependencias requeridas para usar la funcionalidad OCR real."
        }

        demo_text = demo_texts.get(language, demo_texts["eng"])

        return {
            "text": demo_text,
            "confidence": 95.0,
            "language": language,
            "engine": "qwen2.5-vl-demo",
            "word_count": len(demo_text.split()),
            "processing_time": processing_time,
            "model_name": self.model_name,
            "device": self.device,
            "demo_mode": True
        }

# Global instance
qwen_ocr = QwenOCREngine()
