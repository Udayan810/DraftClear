import logging
from typing import List
import numpy as np
import cv2
from ultralytics import YOLO
from config.settings import YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from utils.drawing_state import DrawingState, TextBox

logger = logging.getLogger(__name__)

class PerceptionAgent:
    """Detects text labels in CAD drawings using YOLOv10"""

    def __init__(self, model_name: str = YOLO_MODEL):
        """Initialize YOLOv10 model"""
        logger.info(f"Loading YOLOv10 model: {model_name}")
        try:
            self.model = YOLO(model_name)
            logger.info("YOLOv10 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv10 model: {e}")
            raise

    def detect_text_labels(self, image: np.ndarray) -> List[TextBox]:
        """
        Detect text labels in image using YOLOv10

        Args:
            image: Input image as numpy array (BGR)

        Returns:
            List of TextBox objects with detected labels
        """
        try:
            # Run inference
            results = self.model(
                image,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )

            text_boxes = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes

                    for box in boxes:
                        # Extract coordinates (center x, y, width, height)
                        x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                        confidence = box.conf[0].cpu().item()

                        text_box = TextBox(
                            x=float(x_center),
                            y=float(y_center),
                            w=float(width),
                            h=float(height),
                            confidence=float(confidence),
                            text="",  # OCR not implemented yet
                            rotated=False,
                            angle=0
                        )
                        text_boxes.append(text_box)

            logger.info(f"Detected {len(text_boxes)} text labels")
            return text_boxes

        except Exception as e:
            logger.error(f"Error during text detection: {e}")
            return []

    def run(self, state: DrawingState) -> DrawingState:
        """
        Execute perception agent on drawing state

        Args:
            state: Current DrawingState

        Returns:
            Updated DrawingState with detected text boxes
        """
        if state.original_image is None:
            logger.warning("No image in state")
            return state

        logger.info(f"[Iteration {state.iteration}] Running Perception Agent")

        new_state = state.copy()
        detected_boxes = self.detect_text_labels(state.original_image)
        new_state.text_boxes = detected_boxes

        return new_state
