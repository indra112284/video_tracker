FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .

# Upgrade pip and install CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p results/sort

# Run YOLO11m + SORT tracker
CMD ["python", "tracker/sort_tracker.py"]