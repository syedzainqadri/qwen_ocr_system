"""OCR engine using PaddleOCR - better alternative to TrOCR."""

import logging
from PIL import Image
from typing import Dict, Any, List, Tuple
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    logger.warning(f"PaddleOCR not available: {e}")
    logger.info("Running in demo mode")


class PaddleOCREngine:
    """OCR engine using PaddleOCR - trainable and more accurate than TrOCR."""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        """
        Initialize PaddleOCR engine.
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language code ('en', 'ch', 'fr', 'german', 'korean', 'japan', etc.)
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.ocr = None
        self.model_loaded = False
        
        logger.info(f"PaddleOCR Engine initialized. Language: {lang}")
        
        if PADDLEOCR_AVAILABLE:
            self.load_model()
    
    def load_model(self) -> bool:
        """Load the PaddleOCR model."""
        if self.model_loaded:
            return True
            
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available")
            return False
        
        try:
            logger.info("Loading PaddleOCR model...")
            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang
            )
            self.model_loaded = True
            logger.info("PaddleOCR model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR model: {e}")
            return False
    
    def extract_text(self, image_path: str, language: str = "eng", progress_callback=None) -> Dict[str, Any]:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            language: Language code (e.g., 'eng', 'urd', 'ara')
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        start_time = time.time()

        if progress_callback:
            progress_callback("Initializing PaddleOCR...", 0)

        if not PADDLEOCR_AVAILABLE:
            return self._create_demo_response(image_path, language, start_time)

        if progress_callback:
            progress_callback("Loading PaddleOCR model...", 20)

        if not self.load_model():
            return self._create_demo_response(image_path, language, start_time)

        if progress_callback:
            progress_callback("Processing image with PaddleOCR...", 50)

        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)

            # Run OCR
            logger.info(f"Running PaddleOCR on {image_path}...")
            if progress_callback:
                progress_callback("Running OCR analysis...", 70)

            # Note: Some PaddleOCR versions don't support 'cls' parameter
            try:
                result = self.ocr.ocr(image_array, cls=self.use_angle_cls)
            except TypeError:
                # Fallback without cls parameter
                result = self.ocr.ocr(image_array)

            if progress_callback:
                progress_callback("Processing OCR results...", 90)

            # Process results
            extracted_text = self._process_paddle_results(result)
            processing_time = time.time() - start_time
            
            # Calculate confidence and word count
            confidence = self._calculate_confidence(result)
            word_count = len(extracted_text.split()) if extracted_text else 0

            if progress_callback:
                progress_callback("PaddleOCR completed!", 100)

            return {
                "text": extracted_text,
                "confidence": confidence,
                "language": language,
                "engine": "PaddleOCR",
                "word_count": word_count,
                "processing_time": processing_time,
                "model_name": f"PaddleOCR-{self.lang}",
                "device": "cpu",
                "success": True,
                "demo_mode": False
            }

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return self._create_demo_response(image_path, language, start_time)
    
    def _process_paddle_results(self, results: List) -> str:
        """
        Process PaddleOCR results to extract text.
        
        Args:
            results: Raw PaddleOCR results
            
        Returns:
            Extracted text string
        """
        if not results or not results[0]:
            return ""
        
        text_lines = []
        for line in results[0]:
            if len(line) >= 2:
                # line[1] contains (text, confidence)
                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                text_lines.append(text)
        
        return "\n".join(text_lines)
    
    def _calculate_confidence(self, results: List) -> float:
        """
        Calculate average confidence from PaddleOCR results.
        
        Args:
            results: Raw PaddleOCR results
            
        Returns:
            Average confidence score (0-100)
        """
        if not results or not results[0]:
            return 0.0
        
        confidences = []
        for line in results[0]:
            if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 2:
                confidence = line[1][1]  # confidence score
                confidences.append(confidence)
        
        if not confidences:
            return 0.0
        
        # Convert to percentage and return average
        avg_confidence = sum(confidences) / len(confidences)
        return min(100.0, max(0.0, avg_confidence * 100))
    
    def _create_demo_response(self, image_path: str, language: str, start_time: float) -> Dict[str, Any]:
        """Create a demo response when PaddleOCR is not available."""
        processing_time = time.time() - start_time
        
        return {
            "text": "PaddleOCR demo mode - model not loaded",
            "confidence": 85.0,
            "language": language,
            "engine": "PaddleOCR-demo",
            "word_count": 6,
            "processing_time": processing_time,
            "model_name": "PaddleOCR-demo",
            "device": "cpu",
            "success": True,
            "demo_mode": True,
            "error": "PaddleOCR not available - running in demo mode"
        }
    
    def set_language(self, lang: str):
        """
        Change the OCR language and reload model.
        
        Args:
            lang: Language code
        """
        if lang != self.lang:
            self.lang = lang
            self.model_loaded = False
            logger.info(f"Language changed to: {lang}")
            if PADDLEOCR_AVAILABLE:
                self.load_model()
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            'en',      # English
            'ch',      # Chinese
            'ta',      # Tamil
            'te',      # Telugu
            'ka',      # Kannada
            'hi',      # Hindi
            'ur',      # Urdu
            'ar',      # Arabic
            'fa',      # Persian
            'fr',      # French
            'german',  # German
            'korean',  # Korean
            'japan',   # Japanese
            'it',      # Italian
            'es',      # Spanish
            'pt',      # Portuguese
            'ru',      # Russian
            'uk',      # Ukrainian
            'be',      # Belarusian
            'bg',      # Bulgarian
            'th',      # Thai
            'vi',      # Vietnamese
            'ms',      # Malay
            'id',      # Indonesian
        ]
