# ─────────────────────────────────────────────────────────────
# Dockerfile for MSP4GROWTH API (optimized for security and performance)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim-bullseye AS builder

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install build dependencies
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      build-essential \
      gcc \
      ca-certificates \
      p7zip-full \
      zip \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
ARG GRB_VERSION=12.0.1
RUN pip install --upgrade pip \
 && pip install wheel setuptools \
 && pip install gurobipy==${GRB_VERSION} \
 && pip install -r requirements.txt \
 && pip install fastapi uvicorn pydantic

# Second stage - copy only what we need
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      ca-certificates \
      p7zip-full \
      zip \
      curl \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN useradd -m -s /bin/bash mspuser \
 && mkdir -p /app/data /app/results /app/logs /app/msp4growth \
 && chown -R mspuser:mspuser /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory and copy application code
WORKDIR /app
# Create log directory with proper permissions
RUN mkdir -p /app/logs && chmod 777 /app/logs
COPY --chown=mspuser:mspuser . /app

# Create directories for persistent data
VOLUME ["/app/data", "/app/results", "/app/logs"]

# Setup proper permissions
RUN chmod -R +x /app/scripts/*.sh || true \
  && chmod -R +x /app/scripts/*.py || true \
  && chown -R mspuser:mspuser /app

# Switch to non-root user
USER mspuser

# Environment: thread caps (will be overridden by env vars in docker-compose)
ENV OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the API server
ENTRYPOINT ["python", "-m", "uvicorn", "msp4growth.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]