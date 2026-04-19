import logging
import numpy as np
import easyocr
from typing import List, Dict

logger = logging.getLogger(__name__)

class OCRAgent:
    """Extracts text from images using EasyOCR"""

    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize EasyOCR reader
        
        Args:
            languages: List of language codes
        """
        logger.info(f"Initializing EasyOCR with languages: {languages}")
        try:
            # gpu=True will be used if CUDA is available
            self.reader = easyocr.Reader(languages)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Dictionary with extracted text and raw results
        """
        try:
            # EasyOCR expects RGB
            import cv2
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            results = self.reader.readtext(rgb_image)
            
            full_text = "\n".join([res[1] for res in results])
            
            logger.info(f"Extracted {len(results)} text segments")
            
            return {
                "text": full_text,
                "raw_results": [
                    {
                        "box": res[0],
                        "text": res[1],
                        "confidence": float(res[2])
                    } for res in results
                ]
            }

        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return {
                "text": "",
                "raw_results": [],
                "error": str(e)
            }
