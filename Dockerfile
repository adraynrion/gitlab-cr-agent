# Multi-stage build for production
FROM python:3.11-slim AS base

# Copy version file first to cache version info
COPY version.txt /app/version.txt

# Add version label
LABEL version=$(cat /app/version.txt)
LABEL maintainer="Adraynrion"
LABEL description="AI-powered code review automation for GitLab using PydanticAI"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Builder stage
FROM base AS builder

COPY pyproject.toml .
RUN pip install --user --no-cache-dir -e ".[dev,test]"

# Production stage
FROM base AS production

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
