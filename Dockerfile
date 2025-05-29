# AI-Powered Personal Finance Advisor Main Dockerfile
# From Hasif's Workspace
# This Dockerfile builds the complete application

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY backend/requirements.txt backend_requirements.txt
COPY frontend/requirements.txt frontend_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt  
RUN pip install --no-cache-dir -r frontend_requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-c", "print('AI-Powered Personal Finance Advisor - From Hasif\\'s Workspace')"]
