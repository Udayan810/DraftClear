"""
CAD File Converter
Converts DWG/DXF files to PNG images for processing
"""
import logging
import numpy as np
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class CADConverter:
    """Convert CAD files (DWG, DXF) to images"""

    @staticmethod
    def is_cad_file(filename: str) -> bool:
        """Check if file is a CAD format"""
        extensions = ['.dwg', '.dxf', '.DWG', '.DXF']
        return any(filename.endswith(ext) for ext in extensions)

    @staticmethod
    def convert_dxf_to_image(dxf_bytes: bytes, output_width: int = 800, output_height: int = 600) -> np.ndarray:
        """
        Convert DXF file bytes to numpy image array

        Args:
            dxf_bytes: DXF file content as bytes
            output_width: Output image width
            output_height: Output image height

        Returns:
            numpy array representing the image
        """
        try:
            import ezdxf
            from PIL import Image, ImageDraw

            # Write temp file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
                tmp.write(dxf_bytes)
                tmp_path = tmp.name

            try:
                # Read DXF file
                doc = ezdxf.readfile(tmp_path)
                msp = doc.modelspace()

                # Create blank image
                img = Image.new('RGB', (output_width, output_height), color='white')
                draw = ImageDraw.Draw(img)

                # Get bounding box
                extents = msp.extent()

                if extents.is_empty:
                    logger.warning("Empty DXF file")
                    return np.array(img)

                min_x, min_y = extents.min
                max_x, max_y = extents.max

                # Calculate scaling
                dx = max_x - min_x
                dy = max_y - min_y

                if dx == 0 or dy == 0:
                    logger.warning("Invalid DXF dimensions")
                    return np.array(img)

                scale_x = (output_width - 40) / dx
                scale_y = (output_height - 40) / dy
                scale = min(scale_x, scale_y)

                def to_image_coords(x, y):
                    img_x = int((x - min_x) * scale + 20)
                    img_y = int(output_height - (y - min_y) * scale - 20)
                    return (img_x, img_y)

                # Draw all entities
                for entity in msp:
                    try:
                        dxf_type = entity.dxftype()

                        if dxf_type == 'LINE':
                            p1 = to_image_coords(entity.dxf.start.x, entity.dxf.start.y)
                            p2 = to_image_coords(entity.dxf.end.x, entity.dxf.end.y)
                            draw.line([p1, p2], fill='black', width=2)

                        elif dxf_type == 'CIRCLE':
                            center = to_image_coords(entity.dxf.center.x, entity.dxf.center.y)
                            radius = int(entity.dxf.radius * scale)
                            draw.ellipse(
                                [(center[0] - radius, center[1] - radius),
                                 (center[0] + radius, center[1] + radius)],
                                outline='black', width=2
                            )

                        elif dxf_type == 'ARC':
                            center = to_image_coords(entity.dxf.center.x, entity.dxf.center.y)
                            radius = int(entity.dxf.radius * scale)
                            draw.ellipse(
                                [(center[0] - radius, center[1] - radius),
                                 (center[0] + radius, center[1] + radius)],
                                outline='black', width=2
                            )

                        elif dxf_type == 'LWPOLYLINE':
                            points = [to_image_coords(p[0], p[1]) for p in entity.get_points()]
                            if len(points) > 1:
                                for i in range(len(points) - 1):
                                    draw.line([points[i], points[i + 1]], fill='black', width=2)

                        elif dxf_type == 'POLYLINE':
                            points = [to_image_coords(p[0], p[1]) for p in entity.get_points()]
                            if len(points) > 1:
                                for i in range(len(points) - 1):
                                    draw.line([points[i], points[i + 1]], fill='black', width=2)

                        elif dxf_type == 'TEXT':
                            pos = to_image_coords(entity.dxf.insert.x, entity.dxf.insert.y)
                            text = entity.dxf.text
                            draw.text(pos, str(text), fill='blue', font=None)

                        elif dxf_type == 'MTEXT':
                            pos = to_image_coords(entity.dxf.insert.x, entity.dxf.insert.y)
                            text = entity.dxf.text
                            draw.text(pos, str(text[:50]), fill='blue', font=None)

                    except Exception as e:
                        logger.debug(f"Error drawing entity {dxf_type}: {e}")
                        continue

                # Convert PIL Image to numpy array
                return np.array(img)

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"DXF conversion error: {e}")
            return None

    @staticmethod
    def convert_dwg_to_image(dwg_bytes: bytes, output_width: int = 800, output_height: int = 600) -> np.ndarray:
        """
        Convert DWG file bytes to numpy image array

        Note: DWG conversion requires additional libraries
        For now, we'll try multiple approaches

        Args:
            dwg_bytes: DWG file content as bytes
            output_width: Output image width
            output_height: Output image height

        Returns:
            numpy array representing the image
        """
        try:
            # Try using ezdxf's DWG support if available
            import ezdxf
            from PIL import Image, ImageDraw

            with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                tmp.write(dwg_bytes)
                tmp_path = tmp.name

            try:
                # ezdxf can read some DWG files
                doc = ezdxf.readfile(tmp_path)
                msp = doc.modelspace()

                # Same rendering as DXF
                img = Image.new('RGB', (output_width, output_height), color='white')
                draw = ImageDraw.Draw(img)

                extents = msp.extent()
                if extents.is_empty:
                    logger.warning("Empty DWG file")
                    return np.array(img)

                min_x, min_y = extents.min
                max_x, max_y = extents.max

                dx = max_x - min_x
                dy = max_y - min_y

                if dx == 0 or dy == 0:
                    return np.array(img)

                scale_x = (output_width - 40) / dx
                scale_y = (output_height - 40) / dy
                scale = min(scale_x, scale_y)

                def to_image_coords(x, y):
                    img_x = int((x - min_x) * scale + 20)
                    img_y = int(output_height - (y - min_y) * scale - 20)
                    return (img_x, img_y)

                for entity in msp:
                    try:
                        if entity.dxftype() == 'LINE':
                            p1 = to_image_coords(entity.dxf.start.x, entity.dxf.start.y)
                            p2 = to_image_coords(entity.dxf.end.x, entity.dxf.end.y)
                            draw.line([p1, p2], fill='black', width=2)
                        elif entity.dxftype() == 'CIRCLE':
                            center = to_image_coords(entity.dxf.center.x, entity.dxf.center.y)
                            radius = int(entity.dxf.radius * scale)
                            draw.ellipse(
                                [(center[0] - radius, center[1] - radius),
                                 (center[0] + radius, center[1] + radius)],
                                outline='black', width=2
                            )
                    except Exception as e:
                        logger.debug(f"Error drawing entity: {e}")
                        continue

                return np.array(img)

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"DWG conversion error: {e}")
            logger.warning("DWG support limited. Please convert DWG to DXF first or use image formats.")
            return None

    @staticmethod
    def convert_cad_file(file_bytes: bytes, filename: str) -> np.ndarray:
        """
        Convert any CAD file to image

        Args:
            file_bytes: File content as bytes
            filename: Original filename to determine type

        Returns:
            numpy array representing the image or None
        """
        logger.info(f"Converting CAD file: {filename}")

        filename_lower = filename.lower()

        if filename_lower.endswith('.dxf'):
            return CADConverter.convert_dxf_to_image(file_bytes)
        elif filename_lower.endswith('.dwg'):
            return CADConverter.convert_dwg_to_image(file_bytes)
        else:
            logger.error(f"Unsupported CAD format: {filename}")
            return None
