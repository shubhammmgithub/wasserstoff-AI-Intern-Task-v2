# Production-ready Dockerfile (located at project root)
FROM python:3.10-slim as builder

# Set critical environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

# Stage 1: Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

# Runtime environment variables
ENV UPLOAD_FOLDER=/app/docs \
    CHROMA_DB_PATH=/app/chroma_db \
    TESSERACT_PATH=/usr/bin/tesseract \
    PATH=/root/.local/bin:$PATH

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY backend/app ./app
COPY backend/chunk_data.json ./app/

# Create directories (ensure correct permissions)
RUN mkdir -p ${UPLOAD_FOLDER} ${CHROMA_DB_PATH} && \
    chmod 755 ${UPLOAD_FOLDER} ${CHROMA_DB_PATH}

# Runtime configuration
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Secure Gunicorn configuration
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
    "--workers", "1", \
    "--threads", "2", \
    "--timeout", "90", \
    "--worker-class", "gevent", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--preload", \
    "app.api_server:app"]