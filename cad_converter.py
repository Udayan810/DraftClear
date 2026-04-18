"""
CAD File Converter
Converts DWG/DXF files to PNG images for processing
"""
import logging
import numpy as np
from pathlib import Path
import tempfile
import os
import subprocess
import shutil
import cv2

logger = logging.getLogger(__name__)


class DWGNotSupportedError(Exception):
    """Raised when a DWG file cannot be converted due to missing dependencies."""
    pass

class CADConverter:
    """Convert CAD files (DWG, DXF) to images"""

    ODA_ENV_VAR = "DRAFTCLEAR_ODA_FILE_CONVERTER"
    ODA_CANDIDATES = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
    ]

    @staticmethod
    def is_cad_file(filename: str) -> bool:
        """Check if file is a CAD format"""
        extensions = ['.dwg', '.dxf', '.DWG', '.DXF']
        return any(filename.endswith(ext) for ext in extensions)

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize any rendered image into a contiguous BGR uint8 array."""
        if image is None:
            return None

        normalized = np.asarray(image)
        if normalized.size == 0:
            return None

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
            raise ValueError(f"Unsupported image shape: {normalized.shape}")

        return np.ascontiguousarray(normalized)

    @classmethod
    def _render_dxf_document_to_image(cls, doc, output_width: int, output_height: int) -> np.ndarray:
        """Render a DXF document to PNG using ezdxf's drawing backend."""
        from ezdxf.addons.drawing.config import Configuration
        from ezdxf.addons.drawing.matplotlib import qsave

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            png_path = tmp_png.name

        try:
            size_inches = (
                max(output_width / 100.0, 1.0),
                max(output_height / 100.0, 1.0),
            )
            config = Configuration(
                custom_bg_color="#FFFFFF",
                custom_fg_color="#000000",
            )
            qsave(
                doc.modelspace(),
                png_path,
                bg="#FFFFFF",
                fg="#000000",
                dpi=100,
                size_inches=size_inches,
                config=config,
            )

            rendered = cv2.imread(png_path, cv2.IMREAD_COLOR)
            return cls._normalize_image(rendered)
        finally:
            if os.path.exists(png_path):
                os.unlink(png_path)

    @classmethod
    def _convert_dxf_to_image_fallback(
        cls,
        dxf_bytes: bytes,
        output_width: int = 800,
        output_height: int = 600,
    ) -> np.ndarray:
        """Fallback DXF renderer for cases ezdxf drawing backend cannot handle."""
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
            import ezdxf.bbox as bbox_module
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

                # Get bounding box using ezdxf 1.x API
                extents = bbox_module.extents(msp)

                if not extents.has_data:
                    logger.warning("Empty DXF file or no drawable entities")
                    return np.array(img)

                min_x, min_y = extents.extmin[:2]
                max_x, max_y = extents.extmax[:2]

                # Calculate scaling
                dx = max_x - min_x
                dy = max_y - min_y

                if dx == 0 or dy == 0:
                    logger.warning("Invalid DXF dimensions (zero width or height)")
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
                            if radius > 0:
                                draw.ellipse(
                                    [(center[0] - radius, center[1] - radius),
                                     (center[0] + radius, center[1] + radius)],
                                    outline='black', width=2
                                )

                        elif dxf_type == 'ARC':
                            center = to_image_coords(entity.dxf.center.x, entity.dxf.center.y)
                            radius = int(entity.dxf.radius * scale)
                            if radius > 0:
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
                            draw.text(pos, str(text), fill='blue')

                        elif dxf_type == 'MTEXT':
                            pos = to_image_coords(entity.dxf.insert.x, entity.dxf.insert.y)
                            text = entity.plain_mtext() if hasattr(entity, 'plain_mtext') else entity.dxf.text
                            draw.text(pos, str(text[:50]), fill='blue')

                    except Exception as entity_err:
                        logger.debug(f"Error drawing entity {entity.dxftype()}: {entity_err}")
                        continue

                # Convert PIL Image to numpy array (BGR for OpenCV)
                return cls._normalize_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"DXF conversion error: {e}", exc_info=True)
            return None

    @classmethod
    def convert_dxf_to_image(cls, dxf_bytes: bytes, output_width: int = 800, output_height: int = 600) -> np.ndarray:
        """
        Convert DXF file bytes to numpy image array using ezdxf's renderer first,
        then fall back to the lightweight entity renderer.
        """
        try:
            import ezdxf.recover

            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
                tmp.write(dxf_bytes)
                tmp_path = tmp.name

            try:
                doc, auditor = ezdxf.recover.readfile(tmp_path)
                if auditor.has_errors:
                    logger.warning("DXF auditor reported errors; attempting render anyway")

                rendered = cls._render_dxf_document_to_image(doc, output_width, output_height)
                if rendered is not None:
                    return rendered

                logger.warning("Primary DXF renderer returned no image, using fallback renderer")
            finally:
                os.unlink(tmp_path)
        except Exception as render_err:
            logger.warning(f"Primary DXF render failed, using fallback renderer: {render_err}")

        return cls._convert_dxf_to_image_fallback(dxf_bytes, output_width, output_height)

    @classmethod
    def _find_oda_converter(cls) -> str | None:
        """Locate ODA File Converter on the local machine."""
        env_path = os.environ.get(cls.ODA_ENV_VAR)
        candidates = [
            env_path or "",
            *cls.ODA_CANDIDATES,
            shutil.which("ODAFileConverter") or "",
            shutil.which("ODAFileConverter.exe") or "",
        ]
        return next((path for path in candidates if path and os.path.isfile(path)), None)

    @staticmethod
    def _escape_powershell_path(path: str) -> str:
        return path.replace("'", "''")

    @classmethod
    def _convert_dwg_with_oda(cls, oda_exe: str, source_dir: str, output_dir: str) -> bytes | None:
        """Convert DWG to DXF using ODA File Converter."""
        logger.info(f"Using ODA File Converter: {oda_exe}")
        result = subprocess.run(
            [oda_exe, source_dir, output_dir, "ACAD2018", "DXF", "0", "1"],
            timeout=120,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"ODA conversion failed: {result.stderr.strip() or result.stdout.strip()}")
            return None

        dxf_files = list(Path(output_dir).glob("*.dxf"))
        if not dxf_files:
            logger.error("ODA converter completed without producing a DXF file")
            return None

        with open(dxf_files[0], 'rb') as converted_file:
            return converted_file.read()

    @classmethod
    def _convert_dwg_with_autocad(cls, dwg_path: str, dxf_path: str) -> bytes | None:
        """Convert DWG to DXF using a locally installed AutoCAD COM automation server."""
        powershell_exe = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell_exe:
            return None

        source = cls._escape_powershell_path(dwg_path)
        target = cls._escape_powershell_path(dxf_path)
        script = f"""
$ErrorActionPreference = 'Stop'
$src = '{source}'
$dst = '{target}'
$acad = $null
$doc = $null
try {{
    $acad = New-Object -ComObject AutoCAD.Application
    $acad.Visible = $false
    $doc = $acad.Documents.Open($src)
    $doc.SaveAs($dst, 64)
    $doc.Close()
}} finally {{
    if ($doc -ne $null) {{
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }}
    if ($acad -ne $null) {{
        $acad.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($acad)
    }}
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}}
"""
        result = subprocess.run(
            [powershell_exe, "-NoProfile", "-NonInteractive", "-Command", script],
            timeout=180,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"AutoCAD COM conversion failed: {result.stderr.strip() or result.stdout.strip()}")
            return None

        if not os.path.exists(dxf_path):
            logger.warning("AutoCAD COM completed without producing a DXF file")
            return None

        with open(dxf_path, 'rb') as converted_file:
            return converted_file.read()

    @classmethod
    def convert_dwg_to_dxf_bytes(cls, dwg_bytes: bytes) -> bytes:
        """Convert DWG bytes to DXF bytes using local converters when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dwg_path = os.path.join(tmpdir, "input.dwg")
            dxf_output_dir = os.path.join(tmpdir, "dxf_out")
            dxf_path = os.path.join(dxf_output_dir, "input.dxf")
            os.makedirs(dxf_output_dir, exist_ok=True)

            with open(dwg_path, 'wb') as source_file:
                source_file.write(dwg_bytes)

            oda_exe = cls._find_oda_converter()
            if oda_exe:
                converted = cls._convert_dwg_with_oda(oda_exe, tmpdir, dxf_output_dir)
                if converted:
                    return converted

            converted = cls._convert_dwg_with_autocad(dwg_path, dxf_path)
            if converted:
                return converted

        raise DWGNotSupportedError(
            "DWG conversion needs a local converter that DraftClear can call automatically.\n"
            "Install one of these once, then DWG uploads will convert internally:\n"
            f"  • ODA File Converter (recommended) and optionally set {cls.ODA_ENV_VAR}\n"
            "  • AutoCAD desktop app with COM automation enabled\n"
            "After that, upload the DWG directly and DraftClear will convert it to DXF automatically."
        )

    @classmethod
    def convert_dwg_to_image(cls, dwg_bytes: bytes, output_width: int = 800, output_height: int = 600) -> np.ndarray:
        """
        Convert DWG file bytes to numpy image array.

        DWG is a proprietary Autodesk binary format that ezdxf cannot read directly.
        This method automatically converts DWG -> DXF using any supported local
        converter, then renders the DXF for downstream processing.

        Args:
            dwg_bytes: DWG file content as bytes
            output_width: Output image width
            output_height: Output image height

        Returns:
            numpy array (BGR) representing the rendered DWG, or raises DWGNotSupportedError
        """
        dxf_bytes = cls.convert_dwg_to_dxf_bytes(dwg_bytes)
        return cls.convert_dxf_to_image(dxf_bytes, output_width, output_height)

    @classmethod
    def convert_cad_file(cls, file_bytes: bytes, filename: str) -> np.ndarray:
        """
        Convert any CAD file to image.

        Args:
            file_bytes: File content as bytes
            filename: Original filename to determine type

        Returns:
            numpy array (BGR) representing the image

        Raises:
            DWGNotSupportedError: if a DWG file cannot be converted
        """
        logger.info(f"Converting CAD file: {filename}")

        filename_lower = filename.lower()

        if filename_lower.endswith('.dxf'):
            return cls.convert_dxf_to_image(file_bytes)
        elif filename_lower.endswith('.dwg'):
            return cls.convert_dwg_to_image(file_bytes)
        else:
            logger.error(f"Unsupported CAD format: {filename}")
            return None
