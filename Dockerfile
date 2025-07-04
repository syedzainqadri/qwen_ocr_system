
# Lightweight OCR system for cloud deployment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* || true

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with NumPy compatibility fix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0.0,>=1.24.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads static logs training_data

# Set permissions
RUN chmod +x run_server.py

# Expose port (default to 3030, but will be overridden by cloud platform)
EXPOSE 3030

# Health check disabled temporarily - models take too long to load
# HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=5 \
#     CMD curl -f http://localhost:${PORT:-3030}/health || exit 1

# Run the application
CMD ["python", "run_server.py"]
