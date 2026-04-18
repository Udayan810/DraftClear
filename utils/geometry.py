import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from typing import List, Tuple
from utils.drawing_state import TextBox
import logging

logger = logging.getLogger(__name__)

def create_box_polygon(x: float, y: float, w: float, h: float) -> Polygon:
    """Create a Shapely box from center coordinates and dimensions"""
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return box(x1, y1, x2, y2)

def check_collision(box1: Polygon, box2: Polygon) -> bool:
    """Check if two polygons intersect"""
    return box1.intersects(box2)

def calculate_collision_area(box1: Polygon, box2: Polygon) -> float:
    """Calculate intersection area between two polygons"""
    if not box1.intersects(box2):
        return 0.0
    intersection = box1.intersection(box2)
    return intersection.area

def find_safe_zone(text_box: TextBox, geometry_boxes: List[Polygon], image_shape: Tuple[int, int], search_radius: int = 200) -> Tuple[float, float]:
    """
    Find a safe coordinate for text box away from geometry using circular search

    Args:
        text_box: TextBox to relocate
        geometry_boxes: List of Shapely polygons representing geometry
        image_shape: (height, width) of image
        search_radius: radius in pixels to search for safe zone

    Returns:
        (new_x, new_y) coordinates
    """
    h, w = image_shape[:2]
    text_poly = create_box_polygon(text_box.x, text_box.y, text_box.w, text_box.h)

    # Combine all geometry boxes
    if geometry_boxes:
        combined_geometry = unary_union(geometry_boxes)
    else:
        combined_geometry = None

    # Try to move in different directions (up, down, left, right, diagonals)
    directions = [
        (0, -1),    # up
        (0, 1),     # down
        (-1, 0),    # left
        (1, 0),     # right
        (-1, -1),   # up-left
        (1, -1),    # up-right
        (-1, 1),    # down-left
        (1, 1),     # down-right
    ]

    best_position = (text_box.x, text_box.y)
    best_distance = 0

    step_size = 10
    for direction in directions:
        for step in range(1, search_radius // step_size):
            new_x = text_box.x + direction[0] * step * step_size
            new_y = text_box.y + direction[1] * step * step_size

            # Check bounds
            if new_x - text_box.w/2 < 0 or new_x + text_box.w/2 > w:
                continue
            if new_y - text_box.h/2 < 0 or new_y + text_box.h/2 > h:
                continue

            # Create new position polygon
            new_poly = create_box_polygon(new_x, new_y, text_box.w, text_box.h)

            # Check collision with geometry
            if combined_geometry and new_poly.intersects(combined_geometry):
                continue

            # Calculate distance from original position
            distance = np.sqrt((new_x - text_box.x)**2 + (new_y - text_box.y)**2)
            if distance > best_distance:
                best_distance = distance
                best_position = (new_x, new_y)

            break  # Found safe zone in this direction

    return best_position

def get_bounding_box_from_poly(poly: Polygon) -> Tuple[float, float, float, float]:
    """Extract center x, y, width, height from a Shapely polygon bounding box"""
    minx, miny, maxx, maxy = poly.bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    width = maxx - minx
    height = maxy - miny
    return center_x, center_y, width, height

def quantize_to_orthogonal(angle: float, allowed_angles: List[int] = None) -> int:
    """Snap angle to nearest orthogonal angle (0, 90, 180, 270)"""
    if allowed_angles is None:
        allowed_angles = [0, 90, 180, 270]

    # Normalize angle to 0-360
    angle = angle % 360

    # Find closest allowed angle
    closest_angle = min(allowed_angles, key=lambda a: abs(angle - a))
    return closest_angle
