"""TrOCR OCR Engine as a fallback for Qwen2.5-VL."""

import logging
from PIL import Image
from typing import Dict, Any
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers, but handle gracefully if not available
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
    logger.info("TrOCR available")
except ImportError as e:
    TROCR_AVAILABLE = False
    logger.warning(f"TrOCR not available: {e}")

class TrOCREngine:
    """OCR engine using TrOCR model."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.model_loaded = False
        logger.info(f"TrOCR Engine initialized. Device: {self.device}")
        logger.info(f"Model will be loaded on first use: {self.model_name}")
        
    def load_model(self):
        """Load the TrOCR model and processor."""
        if self.model_loaded:
            return True
            
        if not TROCR_AVAILABLE:
            logger.warning("TrOCR not available - running in demo mode")
            return False
            
        try:
            logger.info(f"Loading {self.model_name}...")
            logger.info("This may take a few minutes for the first time...")
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("TrOCR model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model_loaded and self.model is not None and self.processor is not None
    
    def extract_text(self, image_path: str, language: str = "eng") -> Dict[str, Any]:
        """
        Extract text from image using TrOCR.
        
        Args:
            image_path: Path to the image file
            language: Language code (eng, urd, etc.)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        start_time = time.time()
        
        # If TrOCR not available, return demo response
        if not TROCR_AVAILABLE:
            return self._create_demo_response(image_path, language, start_time)
        
        if not self.is_available():
            if not self.load_model():
                return self._create_demo_response(image_path, language, start_time)
        
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (simple heuristic)
            confidence = self._estimate_confidence(generated_text)
            
            result = {
                "text": generated_text.strip(),
                "confidence": confidence,
                "language": language,
                "engine": "trocr",
                "word_count": len(generated_text.split()) if generated_text else 0,
                "processing_time": processing_time,
                "model_name": self.model_name,
                "device": self.device
            }
            
            logger.info(f"TrOCR completed in {processing_time:.2f}s for language: {language}")
            return result
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": language,
                "engine": "trocr",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "word_count": 0,
                "model_name": self.model_name,
                "device": self.device
            }
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics."""
        if not text:
            return 0.0
        
        # Basic confidence estimation
        base_confidence = 80.0
        
        # Reduce confidence for very short text
        if len(text) < 5:
            base_confidence -= 20.0
        
        # Reduce confidence for text with many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            base_confidence -= 15.0
        
        # Increase confidence for well-structured text
        if any(word.istitle() for word in text.split()):
            base_confidence += 10.0
        
        return max(0.0, min(100.0, base_confidence))
    
    def _create_demo_response(self, image_path: str, language: str, start_time: float) -> Dict[str, Any]:
        """Create a demo response when the model is not available."""
        processing_time = time.time() - start_time
        
        # Create demo text based on language
        demo_texts = {
            "eng": "TrOCR demo: This is sample extracted text. Install transformers to use actual OCR.",
            "urd": "TrOCR ڈیمو: یہ نمونہ متن ہے۔ اصل OCR کے لیے transformers انسٹال کریں۔",
            "ara": "عرض TrOCR: هذا نص تجريبي. قم بتثبيت transformers لاستخدام OCR الفعلي.",
            "fra": "Démo TrOCR: Ceci est un texte d'exemple. Installez transformers pour utiliser l'OCR réel.",
            "deu": "TrOCR Demo: Dies ist ein Beispieltext. Installieren Sie transformers für echte OCR.",
            "spa": "Demo TrOCR: Este es texto de muestra. Instale transformers para usar OCR real."
        }
        
        demo_text = demo_texts.get(language, demo_texts["eng"])
        
        return {
            "text": demo_text,
            "confidence": 85.0,
            "language": language,
            "engine": "trocr-demo",
            "word_count": len(demo_text.split()),
            "processing_time": processing_time,
            "model_name": self.model_name,
            "device": self.device,
            "demo_mode": True
        }

# Global instance
trocr_ocr = TrOCREngine()
