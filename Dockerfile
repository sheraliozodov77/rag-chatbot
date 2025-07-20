# Use a PyTorch + Transformers compatible base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Hugging Face login (optional: can use mounted token instead)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

# Use .env support
RUN pip install python-dotenv

# Expose FastAPI port
EXPOSE 8000

# Default command (FastAPI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]