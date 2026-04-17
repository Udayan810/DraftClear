import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw
from utils.drawing_state import DrawingState

logger = logging.getLogger(__name__)

class HealingAgent:
    """Performs basic inpainting to heal damaged geometry"""

    @staticmethod
    def inpaint_simple(image: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        Simple inpainting using OpenCV's inpainting algorithm

        Args:
            image: Input image with holes (damaged geometry)
            mask: Binary mask where 1 = region to inpaint, 0 = keep
            radius: Inpainting radius

        Returns:
            Inpainted image
        """
        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Use Telea's algorithm for inpainting
        healed = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        return healed

    @staticmethod
    def inpaint_morphological(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpainting using morphological operations (faster, simpler)

        Args:
            image: Input image
            mask: Binary mask

        Returns:
            Inpainted image
        """
        if image.dtype != np.uint8:
            image = (image).astype(np.uint8)

        # Dilate mask to expand inpainting region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)

        # Use morphological closing to fill holes
        healed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

        return healed

    def run(self, state: DrawingState) -> DrawingState:
        """
        Execute healing agent to repair damaged geometry

        Args:
            state: Current DrawingState with damaged geometry

        Returns:
            Updated DrawingState with healed geometry
        """
        logger.info(f"[Iteration {state.iteration}] Running Healing Agent")

        new_state = state.copy()

        # If no damaged geometry, use original as healed
        if state.damaged_geometry is None or state.mask_matrix is None:
            logger.warning("Missing damaged geometry or mask, using original image as healed fallback")
            if state.original_image is not None:
                new_state.healed_geometry = state.original_image.copy()
            else:
                new_state.healed_geometry = None
            return new_state

        try:
            # Simple inpainting (not ML-based for MVP)
            healed = self.inpaint_morphological(state.damaged_geometry, state.mask_matrix)
            new_state.healed_geometry = healed
            logger.info("Geometry healed using morphological inpainting")
        except Exception as e:
            logger.error(f"Healing error: {e}, using damaged geometry as fallback")
            new_state.healed_geometry = state.damaged_geometry.copy()

        return new_state
