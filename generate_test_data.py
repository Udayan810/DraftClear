"""Generate simple test images for DraftClear"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_blueprint(output_path: str):
    """Create a simple mechanical drawing with overlapping text"""
    width, height = 800, 600

    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some simple geometric shapes (simulating CAD geometry)
    # Rectangle
    draw.rectangle([100, 100, 300, 250], outline='black', width=2)
    # Circle
    draw.ellipse([350, 150, 500, 300], outline='black', width=2)
    # Lines
    draw.line([(150, 350), (650, 350)], fill='black', width=2)
    draw.line([(150, 400), (650, 400)], fill='black', width=2)

    # Add hatching pattern
    for i in range(100, 300, 10):
        draw.line([(i, 100), (i+50, 150)], fill='gray', width=1)

    # Add overlapping text labels (this is what needs to be detected)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Text that overlaps with geometry
    draw.text((200, 150), "Label A", fill='black', font=font)
    draw.text((420, 200), "R2.5", fill='black', font=font)
    draw.text((300, 370), "Dimension", fill='black', font=font)
    draw.text((100, 480), "Reference 001", fill='black', font=font)

    img.save(output_path)
    print(f"Created: {output_path}")

def create_overlap_example(output_path: str):
    """Create example with heavy text-geometry overlap"""
    width, height = 800, 600

    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw dense geometry
    for i in range(150, 500, 30):
        draw.line([(150, i), (750, i)], fill='black', width=1)
        draw.line([(i, 150), (i, 500)], fill='black', width=1)

    # Add thick boundary
    draw.rectangle([150, 150, 750, 500], outline='black', width=3)

    # Overlapping text
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw.text((300, 250), "COLLISION TEXT", fill='blue', font=font)
    draw.text((400, 300), "OVERLAP", fill='red', font=font)
    draw.text((250, 350), "DENSE", fill='green', font=font)

    img.save(output_path)
    print(f"Created: {output_path}")

if __name__ == "__main__":
    os.makedirs("data/test_inputs", exist_ok=True)
    create_simple_blueprint("data/test_inputs/simple_blueprint.png")
    create_overlap_example("data/test_inputs/overlap_example.png")
    print("Test data created successfully!")
