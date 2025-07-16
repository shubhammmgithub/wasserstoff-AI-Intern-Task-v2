# Production-ready Dockerfile (located at project root)
FROM python:3.10-slim

# Set critical environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    UPLOAD_FOLDER=/app/docs \
    CHROMA_DB_PATH=/app/chroma_db \
    TESSERACT_PATH=/usr/bin/tesseract

# Install system dependencies for OCR/PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and copy files
WORKDIR /app
COPY backend/requirements.txt .
COPY backend/app ./app
#COPY backend/docs ./docs
COPY backend/chunk_data.json ./app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt gunicorn

# Create persistent storage directories
RUN mkdir -p ${UPLOAD_FOLDER} ${CHROMA_DB_PATH}

# Expose port and health check
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/ || exit 1

# Limit Gunicorn workers (reduce from default)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "90", "app.api_server:app"]