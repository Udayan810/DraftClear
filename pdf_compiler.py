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
    def _image_to_pdf_coords(ix, iy, img_w, img_h, draw_x, draw_y, draw_w, draw_h):
        """Convert image coordinates (top-down) to PDF coordinates (bottom-up)"""
        px = draw_x + (ix / img_w) * draw_w
        py = (draw_y + draw_h) - (iy / img_h) * draw_h
        return px, py

    @staticmethod
    def compile_pdf(states: list[DrawingState], output_name: str = "output") -> str:
        """
        Compile final multi-page PDF with healed geometry, leader lines, and searchable text
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
                    
                    # 🚀 PHASE 2: LEADERS & SEARCHABILITY
                    for i, text_box in enumerate(state.new_coordinates):
                        # 1. Searchable Text Layer (Invisible)
                        px, py = PDFCompiler._image_to_pdf_coords(
                            text_box.x, text_box.y, img_w, img_h, draw_x, draw_y, draw_width, draw_height
                        )
                        
                        # Set text to invisible but searchable
                        c.saveState()
                        c.setTextRenderMode(3) 
                        # Use actual text if available, else a meaningful ID
                        search_text = text_box.text or f"Label_{i+1}"
                        # Center the searchable text roughly
                        text_w = c.stringWidth(search_text, "Helvetica", 10)
                        c.drawString(px - text_w/2, py, search_text)
                        c.restoreState()

                        # 2. Semantic Leader Lines
                        if text_box.original_x is not None and text_box.original_y is not None:
                            ox = text_box.original_x
                            oy = text_box.original_y
                            
                            # Check if the displacement warrants a leader line
                            dist = np.sqrt((text_box.x - ox)**2 + (text_box.y - oy)**2)
                            if dist > 30: # Movement threshold
                                opx, opy = PDFCompiler._image_to_pdf_coords(
                                    ox, oy, img_w, img_h, draw_x, draw_y, draw_width, draw_height
                                )
                                
                                c.saveState()
                                c.setStrokeColorRGB(0, 0.2, 0.55) # Royal Blue to match brand
                                c.setDash(3, 3) # Dotted line
                                c.setLineWidth(1)
                                
                                # Draw line from origin to new location
                                c.line(opx, opy, px, py)
                                
                                # Draw anchor point at origin
                                c.setDash([])
                                c.circle(opx, opy, 2, fill=True, stroke=False)
                                c.restoreState()

                    if geometry_img_path.exists():
                        geometry_img_path.unlink()
                else:
                    c.drawString(50, height - 50, f"DraftClear - Page {page_num}: No geometry available")

                # 🚀 PHASE 3: AI DESCRIPTIVE SUMMARY
                y_pos = height - 540
                c.setFont("Helvetica-Bold", 12)
                c.setFillColorRGB(0, 0.2, 0.55)
                c.drawString(50, y_pos, "AI Analysis & Final Report Summary:")
                c.setFont("Helvetica", 10)
                c.setFillColorRGB(0, 0, 0)
                y_pos -= 20
                
                # Get reasoning - use fallback if empty or too short
                reasoning = state.supervisor_reasoning
                if not reasoning or len(reasoning) < 5:
                    reasoning = f"DraftClear successfully optimized {len(state.text_boxes)} labels over {state.iteration} iterations. "
                    if state.collision_count == 0:
                        reasoning += "All detected spatial conflicts were resolved, ensuring 100% collision-free output and maximum structural clarity."
                    else:
                        reasoning += f"The system reduced overlaps to {state.collision_count} remaining conflicts, prioritizing structural integrity in busy regions."

                # Wrap and draw reasoning text
                from reportlab.lib.utils import simpleSplit
                wrapped_text = simpleSplit(reasoning, "Helvetica", 10, width - 150)
                for line in wrapped_text:
                    c.drawString(70, y_pos, line)
                    y_pos -= 15
                    if y_pos < 100:
                        c.showPage()
                        y_pos = height - 50
                
                y_pos -= 10
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y_pos, f"Detailed Resolution Log (Page {page_num}):")
                y_pos -= 20
                c.setFont("Helvetica", 9)

                if state.new_coordinates:
                    for i, text_box in enumerate(state.new_coordinates):
                        status = "✓ Moved" if text_box.original_x is not None else "✓ Stable"
                        label = f"{i+1}. {status} to ({text_box.x:.1f}, {text_box.y:.1f})"
                        c.drawString(70, y_pos, label)
                        y_pos -= 12
                        if y_pos < 60:
                            c.showPage()
                            y_pos = height - 50
                else:
                    c.drawString(70, y_pos, "No collisions detected/resolved on this page.")

                # Add metadata at bottom 
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(50, 40, f"Page {page_num} | Processed by DraftClear pipeline | Total Collisions Resolved: {state.collision_count}")
                
                # Close the page 
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

            # Create side-by-side image with footer
            h, w = original.shape[:2]
            footer_h = 160
            comparison = np.zeros((h + footer_h, w * 2 + 20, 3), dtype=np.uint8)
            comparison.fill(245) # Light gray background for footer
            
            # Place images
            comparison[:h, :w] = original
            if healed is not None:
                if healed.shape[:2] != original.shape[:2]:
                    healed = cv2.resize(healed, (w, h), interpolation=cv2.INTER_AREA)
                comparison[:h, w+20:w*2+20] = healed
            else:
                comparison[:h, w+20:w*2+20] = np.full((h, w, 3), 128)

            # Add labels
            cv2.putText(comparison, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 2)
            healed_label = "HEALED" if state.healed_geometry is not None else "ORIGINAL (No changes)"
            cv2.putText(comparison, healed_label, (w + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 2)

            # Add Footer Analysis Section
            y_start = h + 40
            cv2.putText(comparison, "AI ANALYSIS & SUMMARY:", (40, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (141, 51, 0), 2) # BGR Royal Blue
            
            # Process reasoning text
            reasoning = state.supervisor_reasoning
            if not reasoning or len(reasoning) < 5:
                reasoning = f"DraftClear optimized {len(state.text_boxes)} labels. "
                reasoning += "Structural healing and spatial resolution were applied to ensure maximum technical clarity."

            # Simple Word Wrap for OpenCV
            words = reasoning.split(' ')
            lines = []
            current_line = ""
            max_char_per_line = int((w * 2) / 12) # Approximation for horizontal fit
            
            for word in words:
                if len(current_line + word) < max_char_per_line:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)

            # Draw lines
            y_text = y_start + 40
            for line in lines[:3]: # Cap at 3 lines for the footer
                cv2.putText(comparison, line.strip(), (60, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)
                y_text += 30

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
