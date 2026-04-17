"""
Phase 0: Synthetic Data Pre-Training
Generates CAD drawings with intentional text-geometry collisions
"""
import numpy as np
import ezdxf
from pathlib import Path
import random
from config.settings import DATA_DIR
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic CAD data with collisions for training"""

    def __init__(self, output_dir: str = None):
        """Initialize data generator"""
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_mechanical_drawing(self, drawing_name: str, num_labels: int = 5, collision_ratio: float = 0.8):
        """
        Create a synthetic mechanical drawing with text labels

        Args:
            drawing_name: Name of output drawing
            num_labels: Number of text labels to add
            collision_ratio: Proportion of labels that should collide with geometry (0-1)
        """
        logger.info(f"Generating drawing: {drawing_name}")

        # Create DXF document
        doc = ezdxf.new()
        msp = doc.modelspace()

        # Draw random mechanical elements
        self._draw_geometry(msp)

        # Add text labels with controlled collisions
        self._add_text_labels(msp, num_labels, collision_ratio)

        # Save DXF
        dxf_path = self.output_dir / f"{drawing_name}.dxf"
        doc.saveas(dxf_path)
        logger.info(f"Saved: {dxf_path}")

        # Also export to PNG for testing
        png_path = self.output_dir / f"{drawing_name}.png"
        self._render_dxf_to_png(str(dxf_path), str(png_path))

        return str(dxf_path)

    def _draw_geometry(self, msp):
        """Draw random mechanical geometry"""
        # Draw rectangle
        msp.add_lwpolyline([
            (100, 100),
            (500, 100),
            (500, 400),
            (100, 400),
            (100, 100)
        ])

        # Draw circles
        msp.add_circle(center=(150, 150), radius=30)
        msp.add_circle(center=(450, 350), radius=40)

        # Draw lines (dimension lines, etc)
        for i in range(150, 450, 50):
            msp.add_line((100, i), (500, i))
            msp.add_line((i, 100), (i, 400))

        # Draw hatching pattern
        for i in range(150, 400, 20):
            msp.add_line((150, i), (200, i + 20))

        logger.info("Geometry drawn: rectangles, circles, lines, hatching")

    def _add_text_labels(self, msp, num_labels: int, collision_ratio: float):
        """Add text labels with controlled collisions"""
        label_texts = [
            "A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8",
            "R2.5", "R5.0", "DIM 100", "REF 001", "SCALE 1:1",
            "PART A", "ASSEMBLY", "TOP VIEW"
        ]

        num_collisions = int(num_labels * collision_ratio)

        for i in range(num_labels):
            text = random.choice(label_texts)
            is_collision = i < num_collisions

            if is_collision:
                # Place text on/near geometry to create collision
                x = random.uniform(150, 350)
                y = random.uniform(150, 350)
            else:
                # Place text in safe area (margins)
                if random.random() < 0.5:
                    x = random.uniform(20, 100)
                else:
                    x = random.uniform(520, 580)
                y = random.uniform(50, 350)

            # Add text entity
            msp.add_text(text, dxfattribs={
                'insert': (x, y),
                'height': 20,
                'rotation': random.choice([0, 90, 180, 270])
            })

        logger.info(f"Added {num_labels} text labels ({num_collisions} colliding, {num_labels - num_collisions} safe)")

    def _render_dxf_to_png(self, dxf_path: str, png_path: str, size: tuple = (800, 600)):
        """Render DXF to PNG using simple rasterization"""
        try:
            from PIL import Image, ImageDraw
            import ezdxf

            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            # Create blank image
            img = Image.new('RGB', size, color='white')
            draw = ImageDraw.Draw(img)

            # Get bounding box
            extents = msp.extent()
            if extents.is_empty:
                logger.warning(f"No entities in {dxf_path}")
                return

            min_x, min_y = extents.min
            max_x, max_y = extents.max

            # Scale and translate
            scale_x = size[0] / (max_x - min_x + 100)
            scale_y = size[1] / (max_y - min_y + 100)
            scale = min(scale_x, scale_y)

            def to_image_coords(x, y):
                img_x = int((x - min_x) * scale + 20)
                img_y = int(size[1] - (y - min_y) * scale - 20)
                return (img_x, img_y)

            # Draw all entities
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    p1 = to_image_coords(entity.dxf.start.x, entity.dxf.start.y)
                    p2 = to_image_coords(entity.dxf.end.x, entity.dxf.end.y)
                    draw.line([p1, p2], fill='black', width=2)

                elif entity.dxftype() == 'CIRCLE':
                    center = to_image_coords(entity.dxf.center.x, entity.dxf.center.y)
                    radius = int(entity.dxf.radius * scale)
                    draw.ellipse([
                        (center[0] - radius, center[1] - radius),
                        (center[0] + radius, center[1] + radius)
                    ], outline='black', width=2)

                elif entity.dxftype() == 'LWPOLYLINE':
                    points = []
                    for point in entity.get_points():
                        points.append(to_image_coords(point[0], point[1]))
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i + 1]], fill='black', width=2)

                elif entity.dxftype() == 'TEXT':
                    pos = to_image_coords(entity.dxf.insert.x, entity.dxf.insert.y)
                    text = entity.dxf.text
                    draw.text(pos, text, fill='blue')

            img.save(png_path)
            logger.info(f"Rendered to PNG: {png_path}")

        except Exception as e:
            logger.warning(f"Could not render to PNG: {e}")

    def generate_dataset(self, num_drawings: int = 10):
        """Generate full dataset"""
        logger.info(f"Generating {num_drawings} synthetic drawings...")

        for i in range(num_drawings):
            drawing_name = f"synthetic_CAD_{i:03d}"
            num_labels = random.randint(3, 8)
            collision_ratio = random.uniform(0.4, 0.95)

            self.create_mechanical_drawing(
                drawing_name,
                num_labels=num_labels,
                collision_ratio=collision_ratio
            )

        logger.info(f"Dataset generation complete")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    generator = SyntheticDataGenerator()
    generator.generate_dataset(num_drawings=5)
    print("Synthetic data generation done!")
