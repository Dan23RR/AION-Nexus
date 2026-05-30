# AION-NEXUS production container
# Build: docker build -t aion-nexus:1.0.0 .
# Run:   docker run -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints aion-nexus:1.0.0

FROM python:3.11-slim AS base

# Install minimal system deps for numpy/torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash aion
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY aion_nexus/ ./aion_nexus/
COPY server/ ./server/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Drop to non-root user
RUN chown -R aion:aion /app
USER aion

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AION_DEVICE=cpu \
    AION_CHECKPOINT=/app/checkpoints/aion_nexus_v1.pth

EXPOSE 8080

# Default command — uvicorn with single worker (replicate via orchestration)
CMD ["uvicorn", "server.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
