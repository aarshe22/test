FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Edmonton

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libglib2.0-bin \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install \
    transformers datasets accelerate bitsandbytes huggingface_hub \
    streamlit python-docx pandas PyPDF2 "unstructured[pdf]" pypdfium2 GPUtil psutil opencv-python

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

