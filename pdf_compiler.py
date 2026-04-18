"""
PDF Compilation Module
Generates final PDF output from healed geometry and repositioned text
"""
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from utils.drawing_state import DrawingState
from config.settings import OUTPUTS_DIR, PDF_DPI

logger = logging.getLogger(__name__)


class PDFCompiler:
    """Compiles final PDF from DraftClear pipeline output"""

    @staticmethod
    def _normalize_image(image: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray | None:
        """Normalize any pipeline image into a BGR uint8 array."""
        candidate = image if image is not None else fallback
        if candidate is None:
            return None

        normalized = np.asarray(candidate)
        if normalized.size == 0:
            return PDFCompiler._normalize_image(fallback) if candidate is not fallback else None

        if normalized.dtype != np.uint8:
            if np.issubdtype(normalized.dtype, np.floating):
                scale = 255.0 if normalized.max(initial=0) <= 1.0 else 1.0
                normalized = np.clip(normalized * scale, 0, 255).astype(np.uint8)
            else:
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        if normalized.ndim == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        elif normalized.ndim == 3 and normalized.shape[2] == 4:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)
        elif normalized.ndim != 3 or normalized.shape[2] != 3:
            if candidate is fallback:
                return None
            return PDFCompiler._normalize_image(fallback)

        return np.ascontiguousarray(normalized)

    @staticmethod
    def compile_pdf(states: list[DrawingState], output_name: str = "output") -> str:
        """
        Compile final multi-page PDF with healed geometry and repositioned text for each page
        """
        logger.info(f"Compiling Multi-page PDF: {output_name}")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{output_name}.pdf"

        # Make states a list if it's not
        if not isinstance(states, list):
            states = [states]

        try:
            c = canvas.Canvas(str(output_path), pagesize=letter)
            width, height = letter

            for idx, state in enumerate(states):
                page_num = idx + 1
                logger.info(f"Adding page {page_num} to PDF")
                
                # Draw healed geometry image
                image_to_draw = PDFCompiler._normalize_image(state.healed_geometry, fallback=state.original_image)
                if image_to_draw is not None:
                    geometry_img_path = OUTPUTS_DIR / f"{output_name}_p{page_num}_geometry_temp.png"
                    cv2.imwrite(str(geometry_img_path), image_to_draw)

                    img_h, img_w = image_to_draw.shape[:2]
                    max_width = width - 100
                    max_height = 400
                    scale = min(max_width / img_w, max_height / img_h)
                    draw_width = img_w * scale
                    draw_height = img_h * scale
                    draw_x = (width - draw_width) / 2
                    draw_y = height - 100 - draw_height

                    c.drawString(50, height - 50, f"DraftClear - Page {page_num} ({output_name})")
                    c.drawImage(str(geometry_img_path), draw_x, draw_y, width=draw_width, height=draw_height)
                    
                    if geometry_img_path.exists():
                        geometry_img_path.unlink()
                else:
                    c.drawString(50, height - 50, f"DraftClear - Page {page_num}: No geometry available")

                # Add repositioned text labels
                y_pos = height - 550
                c.drawString(50, y_pos, f"Repositioned Text Labels (Page {page_num}):")
                y_pos -= 20

                if state.new_coordinates:
                    for i, text_box in enumerate(state.new_coordinates):
                        label = f"{i+1}. Position: ({text_box.x:.1f}, {text_box.y:.1f}) | Color: Blue"
                        c.drawString(70, y_pos, label)
                        y_pos -= 15
                        if y_pos < 50:
                            c.showPage()
                            y_pos = height - 50
                else:
                    c.drawString(70, y_pos, "No collisions detected/resolved on this page.")

                # Add metadata at bottom
                c.drawString(50, 30, f"Page {page_num} | Iterations: {state.iteration} | Collisions: {state.collision_count}")
                
                # Close the page and start a new one for next state
                c.showPage()

            c.save()
            logger.info(f"Multi-page PDF saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Multi-page PDF compilation error: {e}")
            return None

        except Exception as e:
            logger.error(f"PDF compilation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def create_comparison_report(state: DrawingState, output_name: str = "comparison") -> str:
        """
        Create side-by-side comparison report (before/after)

        Args:
            state: Final DrawingState
            output_name: Name for output

        Returns:
            Path to comparison image
        """
        logger.info("Creating comparison report")

        # Ensure output directory exists
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            # Load images (fallback to original if healed is None)
            original = PDFCompiler._normalize_image(state.original_image)
            healed = PDFCompiler._normalize_image(state.healed_geometry, fallback=original)

            if original is None:
                logger.warning("Missing original image for comparison")
                return None

            # Create side-by-side image
            h, w = original.shape[:2]
            comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
            comparison[:, :w] = original
            if healed is not None:
                if healed.shape[:2] != original.shape[:2]:
                    healed = cv2.resize(healed, (w, h), interpolation=cv2.INTER_AREA)
                comparison[:, w+20:] = healed
            else:
                # Fill right side with gray if healed is still None
                comparison[:, w+20:] = np.full((h, w, 3), 128)

            # Add labels
            cv2.putText(comparison, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            healed_label = "HEALED" if state.healed_geometry is not None else "ORIGINAL (No changes)"
            cv2.putText(comparison, healed_label, (w + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Save
            output_path = OUTPUTS_DIR / f"{output_name}_comparison.png"
            cv2.imwrite(str(output_path), comparison)
            logger.info(f"Comparison saved: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Comparison creation error: {e}")
            import traceback
            traceback.print_exc()
            return None
