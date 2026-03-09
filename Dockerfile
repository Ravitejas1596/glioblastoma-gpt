# ─────────────────────────────────────────────────────────────────────────────
# GlioblastomaGPT Dockerfile
# Multi-stage: builder (installs deps) → runtime (slim image)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for scispaCy (requires C++ build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8001 8501

# Default: run FastAPI
CMD ["uvicorn", "gbm_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
