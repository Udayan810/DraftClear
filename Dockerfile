FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV, PyMuPDF, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgles2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (keeps image lean — no CUDA overhead)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p data/outputs data/test_inputs

EXPOSE 7860

CMD ["python", "run.py"]
