from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class TextBox:
    """Represents a detected text label"""
    x: float
    y: float
    w: float  # width
    h: float  # height
    confidence: float = 0.0
    text: str = ""
    rotated: bool = False
    angle: int = 0

    def get_corners(self) -> Tuple[float, float, float, float]:
        """Get bounding box corners (x1, y1, x2, y2)"""
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        return x1, y1, x2, y2

    def get_area(self) -> float:
        """Get area of bounding box"""
        return self.w * self.h

@dataclass
class DrawingState:
    """Immutable state maintained throughout the pipeline"""
    iteration: int = 0
    original_image: Optional[np.ndarray] = None
    text_boxes: List[TextBox] = field(default_factory=list)
    detected_geometry: Optional[np.ndarray] = None
    mask_matrix: Optional[np.ndarray] = None
    damaged_geometry: Optional[np.ndarray] = None
    new_coordinates: List[TextBox] = field(default_factory=list)
    healed_geometry: Optional[np.ndarray] = None
    collision_count: int = 0
    collision_details: List[str] = field(default_factory=list)
    supervisor_decision: str = "continue"  # "continue" or "compile"
    supervisor_reasoning: str = ""

    def copy(self):
        """Create a copy of the state"""
        return DrawingState(
            iteration=self.iteration + 1,
            original_image=self.original_image.copy() if self.original_image is not None else None,
            text_boxes=[TextBox(tb.x, tb.y, tb.w, tb.h, tb.confidence, tb.text, tb.rotated, tb.angle) for tb in self.text_boxes],
            detected_geometry=self.detected_geometry.copy() if self.detected_geometry is not None else None,
            mask_matrix=self.mask_matrix.copy() if self.mask_matrix is not None else None,
            damaged_geometry=self.damaged_geometry.copy() if self.damaged_geometry is not None else None,
            new_coordinates=[TextBox(nc.x, nc.y, nc.w, nc.h, nc.confidence, nc.text, nc.rotated, nc.angle) for nc in self.new_coordinates],
            healed_geometry=self.healed_geometry.copy() if self.healed_geometry is not None else None,
            collision_count=self.collision_count,
            collision_details=self.collision_details.copy(),
            supervisor_decision=self.supervisor_decision,
            supervisor_reasoning=self.supervisor_reasoning,
        )
