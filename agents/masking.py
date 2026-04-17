import logging
import numpy as np
import cv2
from utils.drawing_state import DrawingState, TextBox

logger = logging.getLogger(__name__)

class MaskingAgent:
    """Creates masks to remove text and overlapping geometry"""

    @staticmethod
    def generate_mask_matrix(text_boxes: list, image_shape: tuple) -> np.ndarray:
        """
        Generate binary mask matrix for text regions

        Args:
            text_boxes: List of TextBox objects
            image_shape: (height, width) of image

        Returns:
            Binary mask where 1 = text region, 0 = background
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for text_box in text_boxes:
            x1 = int(text_box.x - text_box.w / 2)
            y1 = int(text_box.y - text_box.h / 2)
            x2 = int(text_box.x + text_box.w / 2)
            y2 = int(text_box.y + text_box.h / 2)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Draw filled rectangle
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

        return mask

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to image using Hadamard product (element-wise multiplication)

        Args:
            image: Input image (BGR or grayscale)
            mask: Binary mask (same height/width as image)

        Returns:
            Masked image with text regions removed (set to white/255)
        """
        if len(image.shape) == 3:
            # Color image: expand mask to 3 channels
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            # Invert mask: 1 where text, 0 where background
            mask_inv = 1 - mask_3ch
            # Apply: keep original where mask=0, set to white where mask=1
            damaged = image.astype(float) * mask_inv + 255 * (1 - mask_inv)
            return damaged.astype(np.uint8)
        else:
            # Grayscale image
            mask_inv = 1 - mask
            damaged = image.astype(float) * mask_inv + 255 * mask
            return damaged.astype(np.uint8)

    def run(self, state: DrawingState) -> DrawingState:
        """
        Execute masking agent on drawing state

        Args:
            state: Current DrawingState with detected text boxes

        Returns:
            Updated DrawingState with masked/damaged geometry
        """
        if state.original_image is None:
            logger.warning("No image in state")
            return state

        if not state.text_boxes:
            logger.warning("No text boxes detected, skipping masking")
            return state

        logger.info(f"[Iteration {state.iteration}] Running Masking Agent")

        new_state = state.copy()

        # Generate mask for text regions
        mask = self.generate_mask_matrix(state.text_boxes, state.original_image.shape)
        new_state.mask_matrix = mask

        # Apply mask to create damaged geometry
        damaged = self.apply_mask(state.original_image, mask)
        new_state.damaged_geometry = damaged

        logger.info(f"Mask created: {np.sum(mask)} pixels masked")

        return new_state
