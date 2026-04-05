# ─────────────────────────────────────────────
#  BMS FastAPI — Production Dockerfile
# ─────────────────────────────────────────────
FROM python:3.10-slim

# Keeps Python output unbuffered (good for logs in Docker/Render)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000

WORKDIR /app

# ── System dependencies ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────
# Copy only requirements first — lets Docker cache this layer
# so a code change doesn't re-install all packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────
COPY . .

# ── Create required runtime directories ──────
# (app.py also does this, but better to have them in the image)
RUN mkdir -p dataset models static

# ── Non-root user (security best practice) ───
RUN useradd --create-home --shell /bin/bash bmsuser \
    && chown -R bmsuser:bmsuser /app
USER bmsuser

# ── Port ─────────────────────────────────────
EXPOSE ${PORT}

# ── Health check ─────────────────────────────
# Docker will mark the container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entrypoint ───────────────────────────────
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]