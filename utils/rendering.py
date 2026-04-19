import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List
from utils.drawing_state import TextBox

logger = logging.getLogger(__name__)

class DrawingRenderer:
    """Utility for drawing text and geometry onto CAD images"""

    @staticmethod
    def draw_labels(image: np.ndarray, labels: List[TextBox], color: tuple = (0, 0, 0)) -> np.ndarray:
        """
        Draw a list of repositioned text labels onto an image
        
        Args:
            image: Background image (BGR numpy array)
            labels: List of TextBox objects with final coordinates
            color: Text color (B, G, R)
            
        Returns:
            Image with labels drawn
        """
        if image is None:
            return None

        # Convert to PIL for better text rendering (antialiasing)
        if len(image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(image)
        
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load a clean font, fallback to default
        try:
            # Common paths on Windows
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        for label in labels:
            # Use final coordinates (x, y are centers)
            x_left = label.x - label.w / 2
            y_top = label.y - label.h / 2
            
            # Draw text
            text = label.text if label.text else "Label"
            draw.text((x_left, y_top), text, fill=color, font=font)

        # Convert back to BGR
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return res

    @staticmethod
    def draw_comparison_footer(image: np.ndarray, text: str, footer_h: int = 160) -> np.ndarray:
        """
        Add a descriptive footer to an image
        """
        h, w = image.shape[:2]
        # This is already partially handled in PDFCompiler, but we can centralize it here if needed
        return image
