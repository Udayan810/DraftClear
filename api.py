"""
FastAPI Backend for DraftClear
REST API for frontend integration
"""
import logging
import io
import base64
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import cv2
import numpy as np
from pydantic import BaseModel, EmailStr

from utils.auth import (
    get_password_hash, verify_password, create_access_token,
    get_user, save_users, get_users, get_current_user
)
from orchestrator import LangGraphOrchestrator
from utils.drawing_state import DrawingState
from pdf_compiler import PDFCompiler
from cad_converter import CADConverter, DWGNotSupportedError
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

class PageResult(BaseModel):
    """Result for a single page"""
    page_number: int
    iterations: int
    text_labels: int
    collision_count: int
    supervisor_decision: str
    original_image_url: str
    processed_image_url: str

class ProcessResponse(BaseModel):
    """Response model for processing"""
    success: bool
    message: str
    total_pages: int
    pages: list[PageResult]
    pdf_url: str
    comparison_url: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for transmission"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64"""
    normalized = normalize_image_for_output(image_array)
    if normalized is None:
        return ""
    success, buffer = cv2.imencode('.png', normalized)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')


def normalize_image_for_output(image_array: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray | None:
    """Normalize output images into PNG-safe BGR uint8 arrays."""
    candidate = image_array if image_array is not None else fallback
    if candidate is None:
        return None

    normalized = np.asarray(candidate)
    if normalized.size == 0:
        return normalize_image_for_output(fallback) if candidate is not fallback else None

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
        return normalize_image_for_output(fallback)

    return np.ascontiguousarray(normalized)


def sanitize_output_name(output_name: str) -> str:
    """Convert arbitrary user-provided names into filesystem-safe artifact names."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', output_name.strip())
    cleaned = cleaned.strip('._-')
    return cleaned or "processed"


def save_preview_artifact(image_array: np.ndarray, output_name: str, suffix: str) -> str:
    """Save a preview artifact and return its download URL."""
    normalized = normalize_image_for_output(image_array)
    if normalized is None:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview image: {suffix}")

    artifact_path = OUTPUTS_DIR / f"{output_name}_{suffix}.png"
    if not cv2.imwrite(str(artifact_path), normalized):
        raise HTTPException(status_code=500, detail=f"Failed to save preview image: {suffix}")

    return f"/api/download/image/{output_name}_{suffix}"


def build_process_response(final_states: list[DrawingState], output_name: str, message: str) -> ProcessResponse:
    """Build a stable API response with saved artifacts and reliable preview URLs for multiple pages."""
    page_results = []
    
    for idx, state in enumerate(final_states):
        page_num = idx + 1
        original_image = normalize_image_for_output(state.original_image)
        processed_image = normalize_image_for_output(state.healed_geometry, fallback=original_image)

        if original_image is None or processed_image is None:
            logger.warning(f"Page {page_num} artifacts are unavailable")
            continue

        original_url = save_preview_artifact(original_image, output_name, f"p{page_num}_original")
        processed_url = save_preview_artifact(processed_image, output_name, f"p{page_num}_processed")

        page_results.append(PageResult(
            page_number=page_num,
            iterations=state.iteration,
            text_labels=len(state.text_boxes),
            collision_count=state.collision_count,
            supervisor_decision=state.supervisor_decision,
            original_image_url=original_url,
            processed_image_url=processed_url
        ))

    pdf_url = f"/api/download/pdf/{output_name}"
    comparison_url = f"/api/download/image/{output_name}_comparison"

    return ProcessResponse(
        success=True,
        message=message,
        total_pages=len(page_results),
        pages=page_results,
        pdf_url=pdf_url,
        comparison_url=comparison_url
    )


# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "DraftClear",
        "version": "1.0.0"
    }


# Auth Endpoints

@app.post("/api/auth/register")
async def register(user: UserCreate):
    users = get_users()
    if user.email in users:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    from datetime import datetime
    new_user = {
        "email": user.email,
        "password": get_password_hash(user.password),
        "role": "user",
        "created_at": datetime.utcnow().isoformat()
    }
    users[user.email] = new_user
    save_users(users)
    return {"message": "User registered successfully"}

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user["email"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"email": current_user["email"], "role": current_user["role"]}

@app.post("/api/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...), output_name: str = "processed", current_user: dict = Depends(get_current_user)):
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
        safe_output_name = sanitize_output_name(output_name)

        # Read uploaded file
        contents = await file.read()

        images = []
        # Check if it's a CAD file
        if CADConverter.is_cad_file(file.filename):
            logger.info(f"Detected CAD file: {file.filename}")
            try:
                images = CADConverter.convert_cad_file(contents, file.filename)
            except DWGNotSupportedError as dwg_err:
                raise HTTPException(status_code=400, detail=str(dwg_err))

            if not images:
                raise HTTPException(status_code=400, detail="Failed to convert CAD file. Please ensure it is a valid DWG/DXF file.")
        else:
            # It's an image file
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
            else:
                raise HTTPException(status_code=400, detail="Invalid image file")

        # Limit pages to prevent timeouts if requested by system constraints
        MAX_PAGES = 10
        if len(images) > MAX_PAGES:
            logger.warning(f"Truncating file to {MAX_PAGES} pages")
            images = images[:MAX_PAGES]

        # Process each page
        final_states = []
        for idx, image in enumerate(images):
            logger.info(f"Processing page {idx+1}/{len(images)}")
            initial_state = DrawingState(original_image=image)
            final_state = orchestrator.run(initial_state)
            final_states.append(final_state)

        # Compile Multi-page PDF
        pdf_compiler = PDFCompiler()
        pdf_compiler.compile_pdf(final_states, safe_output_name)
        
        # Comparison report (just for first page for now in the overall response)
        pdf_compiler.create_comparison_report(final_states[0], safe_output_name)

        return build_process_response(final_states, safe_output_name, f"Processed {len(final_states)} pages successfully")

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
        safe_output_name = sanitize_output_name(request.output_name)

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
        pdf_compiler.compile_pdf(final_state, safe_output_name)
        pdf_compiler.create_comparison_report(final_state, safe_output_name)

        return build_process_response(final_state, safe_output_name, "Base64 image processed successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve static frontend files
frontend_path = Path(__file__).parent / "frontend"

@app.get("/")
async def read_index():
    return FileResponse(frontend_path / "index.html")

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
