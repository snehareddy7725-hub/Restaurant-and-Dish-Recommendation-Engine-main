# Hybrid Restaurant Recommender - FastAPI service
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    RECO_DATA_DIR=/app/data \
    RECO_LOG_DIR=/app/logs

# libgomp1: OpenMP runtime used by implicit's ALS
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommender/ recommender/
COPY api/ api/
COPY train_models.py .
COPY data/ data/

# Train inside the image so the pickles always match the installed library versions
RUN python train_models.py && mkdir -p /app/logs

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
