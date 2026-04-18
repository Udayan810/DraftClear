import logging
import numpy as np
from typing import List, Tuple
from shapely.geometry import box
from config.settings import ORTHOGONAL_ANGLES, PADDING, COLLISION_THRESHOLD
from utils.drawing_state import DrawingState, TextBox
from utils.observability import observe
from utils.geometry import create_box_polygon, find_safe_zone, calculate_collision_area

logger = logging.getLogger(__name__)

class SpatialResolutionAgent:
    """Calculates safe spatial coordinates for text labels"""

    def detect_geometry_regions(self, image: np.ndarray) -> List[object]:
        """
        Detect non-white regions in image as geometry (simplified)

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of bounding boxes for detected geometry
        """
        if len(image.shape) == 3:
            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use Otsu's thresholding to handle various background shades/colors automatically
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        geometry_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter out noise
                poly = box(x, y, x + w, y + h)
                geometry_boxes.append(poly)

        return geometry_boxes

    def check_collision_with_geometry(
        self,
        text_box: TextBox,
        geometry_boxes: List[object],
        collision_threshold: float = COLLISION_THRESHOLD
    ) -> bool:
        """
        Check if text box collides with geometry

        Args:
            text_box: TextBox to check
            geometry_boxes: List of geometry polygons
            collision_threshold: Minimum collision area to flag

        Returns:
            True if collision detected
        """
        text_poly = create_box_polygon(
            text_box.x,
            text_box.y,
            text_box.w + PADDING,
            text_box.h + PADDING
        )

        for geom_box in geometry_boxes:
            collision_area = calculate_collision_area(text_poly, geom_box)
            if collision_area > collision_threshold:
                return True

        return False

    def calculate_new_coordinates(
        self,
        state: DrawingState,
        geometry_boxes: List[object]
    ) -> Tuple[List[TextBox], int]:
        """
        Calculate new safe coordinates for all text boxes

        Args:
            state: Current DrawingState
            geometry_boxes: List of geometry polygons

        Returns:
            Tuple of (new_text_boxes, collision_count)
        """
        new_coordinates = []
        collision_count = 0

        for text_box in state.text_boxes:
            # Check if current position collides
            if self.check_collision_with_geometry(text_box, geometry_boxes):
                collision_count += 1

                # Find safe zone
                new_x, new_y = find_safe_zone(
                    text_box,
                    geometry_boxes,
                    state.original_image.shape
                )

                # Create new text box at safe position
                new_box = TextBox(
                    x=new_x,
                    y=new_y,
                    w=text_box.w,
                    h=text_box.h,
                    confidence=text_box.confidence,
                    text=text_box.text,
                    rotated=text_box.rotated,
                    angle=text_box.angle
                )
                new_coordinates.append(new_box)
            else:
                # No collision, keep original position
                new_coordinates.append(text_box)

        return new_coordinates, collision_count

    @observe()
    def run(self, state: DrawingState) -> DrawingState:
        """
        Execute spatial resolution agent

        Args:
            state: Current DrawingState

        Returns:
            Updated DrawingState with new text coordinates
        """
        if state.original_image is None or not state.text_boxes:
            logger.warning("Missing image or text boxes")
            return state

        logger.info(f"[Iteration {state.iteration}] Running Spatial Resolution Agent")

        new_state = state.copy()

        # Detect geometry regions
        geometry_boxes = self.detect_geometry_regions(state.original_image)
        logger.info(f"Detected {len(geometry_boxes)} geometry regions")

        # Calculate new coordinates
        new_coords, collision_count = self.calculate_new_coordinates(state, geometry_boxes)
        new_state.new_coordinates = new_coords
        new_state.collision_count = collision_count

        if collision_count > 0:
            logger.warning(f"Found {collision_count} collisions, attempting repositioning")
        else:
            logger.info("No collisions detected!")

        return new_state


# Import cv2 for geometry detection
import cv2
