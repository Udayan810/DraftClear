import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_INPUTS_DIR = DATA_DIR / "test_inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
YOLO_MODEL = "yolov10n.pt"  # nano model for CPU
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

# Geometry & Collision Detection
COLLISION_THRESHOLD = 10  # pixels - minimum distance to avoid collision
ORTHOGONAL_ANGLES = [0, 90, 180, 270]  # allowed text rotation angles
PADDING = 5  # pixel padding around text boxes

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # lightweight model for CPU
OLLAMA_TIMEOUT = 30

# Pipeline Configuration
MAX_ITERATIONS = 5  # max loops in supervisor logic
COLLISION_TOLERANCE = 0  # 0 = no collisions allowed

# Logging
LOG_LEVEL = "INFO"

# Image Processing
IMAGE_FORMAT = "PNG"
PDF_DPI = 300
