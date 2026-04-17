"""
FastAPI Backend for DraftClear
REST API for frontend integration
"""
import logging
import io
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pydantic import BaseModel

from orchestrator import LangGraphOrchestrator
from utils.drawing_state import DrawingState
from pdf_compiler import PDFCompiler
from cad_converter import CADConverter
from config.settings import OUTPUTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DraftClear API",
    description="CAD-to-PDF Conflict Resolution & Extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
try:
    orchestrator = LangGraphOrchestrator()
    logger.info("Orchestrator initialized")
except Exception as e:
    logger.error(f"Failed to initialize orchestrator: {e}")

# Pydantic models
class ProcessRequest(BaseModel):
    """Request model for processing"""
    image_base64: str
    output_name: str = "processed"

class ProcessResponse(BaseModel):
    """Response model for processing"""
    success: bool
    message: str
    iterations: int
    text_labels: int
    collision_count: int
    supervisor_decision: str
    original_image: str  # base64
    healed_image: str  # base64
    pdf_url: str
    comparison_url: str


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for transmission"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64"""
    if image_array is None:
        return ""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "DraftClear",
        "version": "1.0.0"
    }


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...), output_name: str = "processed"):
    """
    Process CAD image through DraftClear pipeline

    Args:
        file: Input image file (PNG, JPG, BMP, DXF, DWG)
        output_name: Name for output artifacts

    Returns:
        ProcessResponse with results
    """
    logger.info(f"Processing file: {file.filename}")

    try:
        # Read uploaded file
        contents = await file.read()

        # Check if it's a CAD file
        if CADConverter.is_cad_file(file.filename):
            logger.info(f"Detected CAD file: {file.filename}")
            image = CADConverter.convert_cad_file(contents, file.filename)

            if image is None:
                raise HTTPException(status_code=400, detail="Failed to convert CAD file. Please ensure it's a valid DXF or DWG file.")
        else:
            # It's an image file
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

        # Create initial state
        initial_state = DrawingState(original_image=image)

        # Run orchestrator
        final_state = orchestrator.run(initial_state)

        # Compile PDF
        pdf_compiler = PDFCompiler()
        pdf_path = pdf_compiler.compile_pdf(final_state, output_name)
        comparison_path = pdf_compiler.create_comparison_report(final_state, output_name)

        # Prepare response
        return ProcessResponse(
            success=True,
            message="File processed successfully",
            iterations=final_state.iteration,
            text_labels=len(final_state.text_boxes),
            collision_count=final_state.collision_count,
            supervisor_decision=final_state.supervisor_decision,
            original_image=image_to_base64(final_state.original_image),
            healed_image=image_to_base64(final_state.healed_geometry),
            pdf_url=f"/api/download/pdf/{output_name}",
            comparison_url=f"/api/download/image/{output_name}_comparison"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/pdf/{filename}")
async def download_pdf(filename: str):
    """Download generated PDF"""
    from pathlib import Path

    pdf_path = Path(OUTPUTS_DIR) / f"{filename}.pdf"

    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}.pdf")

    logger.info(f"Downloading PDF: {pdf_path}")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{filename}.pdf"
    )


@app.get("/api/download/image/{filename}")
async def download_image(filename: str):
    """Download generated image"""
    from pathlib import Path

    img_path = Path(OUTPUTS_DIR) / f"{filename}.png"

    if not img_path.exists():
        logger.warning(f"Image not found: {img_path}")
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}.png")

    logger.info(f"Downloading image: {img_path}")
    return FileResponse(
        path=str(img_path),
        media_type="image/png",
        filename=f"{filename}.png"
    )


@app.get("/api/results/{output_name}")
async def get_results(output_name: str):
    """Get processing results"""
    report_path = OUTPUTS_DIR / f"{output_name}_report.txt"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    with open(report_path, 'r') as f:
        report = f.read()

    return {
        "output_name": output_name,
        "report": report,
        "links": {
            "pdf": f"/api/download/pdf/{output_name}",
            "comparison": f"/api/download/image/{output_name}_comparison"
        }
    }


@app.post("/api/process-base64")
async def process_base64_image(request: ProcessRequest):
    """Process image from base64 string"""
    logger.info(f"Processing base64 image: {request.output_name}")

    try:
        # Decode base64
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Create initial state
        initial_state = DrawingState(original_image=image)

        # Run orchestrator
        final_state = orchestrator.run(initial_state)

        # Compile PDF
        pdf_compiler = PDFCompiler()
        pdf_path = pdf_compiler.compile_pdf(final_state, request.output_name)

        return ProcessResponse(
            success=True,
            message="Base64 image processed successfully",
            iterations=final_state.iteration,
            text_labels=len(final_state.text_boxes),
            collision_count=final_state.collision_count,
            supervisor_decision=final_state.supervisor_decision,
            original_image=image_to_base64(final_state.original_image),
            healed_image=image_to_base64(final_state.healed_geometry),
            pdf_url=f"/api/download/pdf/{request.output_name}",
            comparison_url=f"/api/download/image/{request.output_name}_comparison"
        )

    except Exception as e:
        logger.error(f"Base64 processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve static frontend files
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
